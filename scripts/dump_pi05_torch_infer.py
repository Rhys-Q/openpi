"""Dump deterministic pi0.5 PyTorch policy inference inputs and outputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import time
from typing import Any

import jax
import numpy as np
import safetensors.torch
import torch

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.models_pytorch import pi0_pytorch
from openpi.shared import download

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

PRECISIONS = ("bfloat16", "float16")


def _dtype_name(precision: str) -> str:
    if precision == "bfloat16":
        return "bf16"
    if precision == "float16":
        return "fp16"
    raise ValueError(f"Unsupported precision: {precision}")


def _generate_inputs(num_examples: int, seed: int, action_horizon: int, action_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    prompts = np.asarray([f"put the object into the container {i}" for i in range(num_examples)])
    return {
        "base_0_rgb": rng.integers(0, 256, size=(num_examples, 224, 224, 3), dtype=np.uint8),
        "left_wrist_0_rgb": rng.integers(0, 256, size=(num_examples, 224, 224, 3), dtype=np.uint8),
        "right_wrist_0_rgb": rng.integers(0, 256, size=(num_examples, 224, 224, 3), dtype=np.uint8),
        "state": rng.uniform(-1.0, 1.0, size=(num_examples, action_dim)).astype(np.float32),
        "noise": rng.standard_normal(size=(num_examples, action_horizon, action_dim)).astype(np.float32),
        "prompt": prompts,
    }


def _save_inputs(path: pathlib.Path, inputs: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **inputs)


def _dump_tokenizer_model(out_dir: pathlib.Path) -> pathlib.Path:
    tokenizer_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    output_path = out_dir / "tokenizer.model"
    shutil.copy2(tokenizer_path, output_path)
    return output_path


def _load_inputs(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _make_transform(model_config: pi0_config.Pi0Config) -> transforms.DataTransformFn:
    return transforms.compose(
        [
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=model_config.discrete_state_input,
            ),
            transforms.PadStatesAndActions(model_config.action_dim),
        ]
    )


def _make_model(ckpt_dir: pathlib.Path, precision: str, device: torch.device) -> pi0_pytorch.PI0Pytorch:
    model_config = pi0_config.Pi0Config(pi05=True, dtype=precision, pytorch_compile_mode=None)
    model = pi0_pytorch.PI0Pytorch(model_config)
    safetensors.torch.load_model(model, ckpt_dir / "model.safetensors")
    model.paligemma_with_expert.to_bfloat16_for_selected_params(precision)
    return model.to(device).eval()


def _example_from_inputs(inputs: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        "image": {
            "base_0_rgb": inputs["base_0_rgb"][index],
            "left_wrist_0_rgb": inputs["left_wrist_0_rgb"][index],
            "right_wrist_0_rgb": inputs["right_wrist_0_rgb"][index],
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "state": inputs["state"][index],
        "prompt": str(inputs["prompt"][index]),
    }


def _to_torch_batch(tree: Any, device: torch.device) -> Any:
    return jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...], tree)


@torch.inference_mode()
def run_inference(
    ckpt_dir: pathlib.Path,
    inputs: dict[str, np.ndarray],
    precision: str,
    device: torch.device,
    num_steps: int,
) -> np.ndarray:
    model_config = pi0_config.Pi0Config(pi05=True, dtype=precision, pytorch_compile_mode=None)
    input_transform = _make_transform(model_config)
    model = _make_model(ckpt_dir, precision, device)
    outputs = []

    for index in range(inputs["state"].shape[0]):
        transformed = input_transform(_example_from_inputs(inputs, index))
        batch = _to_torch_batch(transformed, device)
        observation = _model.Observation.from_dict(batch)
        noise = torch.from_numpy(inputs["noise"][index]).to(device)[None, ...]
        actions = model.sample_actions(device, observation, noise=noise, num_steps=num_steps)
        outputs.append(actions[0].detach().cpu().to(torch.float32).numpy())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.stack(outputs, axis=0)


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(reference - candidate)
    rel_diff = abs_diff / np.maximum(np.abs(reference), 1e-6)
    return {
        "abs_max": float(abs_diff.max()),
        "abs_mean": float(abs_diff.mean()),
        "abs_p50": float(np.percentile(abs_diff, 50)),
        "abs_p95": float(np.percentile(abs_diff, 95)),
        "abs_p99": float(np.percentile(abs_diff, 99)),
        "rel_max": float(rel_diff.max()),
        "rel_mean": float(rel_diff.mean()),
    }


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=pathlib.Path, default=pathlib.Path("/root/autodl-tmp/tools/pi05-pytorch-base"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/pi05_torch_infer"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    if not (args.ckpt / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing model.safetensors under {args.ckpt}")

    device = torch.device(args.device)
    model_config = pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=None)
    inputs = _generate_inputs(args.num_examples, args.seed, model_config.action_horizon, model_config.action_dim)
    args.out.mkdir(parents=True, exist_ok=True)
    _save_inputs(args.out / "inputs.npz", inputs)
    tokenizer_output_path = _dump_tokenizer_model(args.out)

    outputs: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}
    for precision in PRECISIONS:
        start = time.monotonic()
        actions = run_inference(args.ckpt, inputs, precision, device, args.num_steps)
        timing[_dtype_name(precision)] = time.monotonic() - start
        output_dir = args.out / _dtype_name(precision)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "outputs.npz", actions=actions)
        outputs[_dtype_name(precision)] = actions

    metric_payload = {
        "bf16_vs_fp16": _metrics(outputs["bf16"], outputs["fp16"]),
        "shape": list(outputs["bf16"].shape),
    }
    _write_json(args.out / "metrics.json", metric_payload)

    metadata = {
        "ckpt": str(args.ckpt),
        "device": str(device),
        "seed": args.seed,
        "num_examples": args.num_examples,
        "num_steps": args.num_steps,
        "precisions": list(PRECISIONS),
        "tokenizer_model": str(tokenizer_output_path),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "timing_sec": timing,
        "model_config": dataclasses.asdict(model_config),
    }
    _write_json(args.out / "metadata.json", metadata)
    print(json.dumps(metric_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
