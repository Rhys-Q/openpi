"""Replay dumped pi0.5 fp16 inputs and compare against dumped outputs."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts import dump_pi05_torch_infer

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=pathlib.Path, default=pathlib.Path("/root/autodl-tmp/tools/pi05-pytorch-base"))
    parser.add_argument("--dump-dir", type=pathlib.Path, default=pathlib.Path("outputs/pi05_torch_infer"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    inputs = dump_pi05_torch_infer._load_inputs(args.dump_dir / "inputs.npz")  # noqa: SLF001
    with np.load(args.dump_dir / "fp16" / "outputs.npz") as dumped:
        expected = dumped["actions"]

    actual = dump_pi05_torch_infer.run_inference(
        args.ckpt,
        inputs,
        "float16",
        torch.device(args.device),
        args.num_steps,
    )
    diff = np.abs(actual - expected)
    result = {
        "allclose": bool(np.allclose(actual, expected, rtol=args.rtol, atol=args.atol)),
        "atol": args.atol,
        "rtol": args.rtol,
        "abs_max": float(diff.max()),
        "abs_mean": float(diff.mean()),
        "shape": list(actual.shape),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["allclose"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
