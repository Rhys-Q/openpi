# pi0.5 PyTorch bf16/fp16 Inference Dump Analysis

## Run Summary

- Checkpoint: `/root/autodl-tmp/tools/pi05-pytorch-base`
- Dump directory: `outputs/pi05_torch_infer`
- Device: NVIDIA GeForce RTX 4090
- Torch: `2.7.1+cu126`
- Examples: 10
- Action output shape: `(10, 50, 32)`
- Denoising steps: 10
- Shared inputs: images, state, prompt, and noise are identical for bf16 and fp16.

## Dumped Files

- `outputs/pi05_torch_infer/inputs.npz`: 10 generated input groups and fixed noise.
- `outputs/pi05_torch_infer/tokenizer.model`: tokenizer used by the pi0.5 prompt preprocessor.
- `outputs/pi05_torch_infer/bf16/outputs.npz`: bf16 policy outputs.
- `outputs/pi05_torch_infer/fp16/outputs.npz`: fp16 policy outputs.
- `outputs/pi05_torch_infer/metrics.json`: numeric bf16/fp16 comparison.
- `outputs/pi05_torch_infer/metadata.json`: run metadata.

## Difference Metrics

The comparison below uses bf16 outputs as the reference and fp16 outputs as the candidate.

| Metric | Value |
| --- | ---: |
| max absolute diff | 0.1266273260 |
| mean absolute diff | 0.0042807190 |
| p50 absolute diff | 0.0023893937 |
| p95 absolute diff | 0.0133071091 |
| p99 absolute diff | 0.0260543734 |
| max relative diff | 323.3483581543 |
| mean relative diff | 0.1451961398 |

The large max relative diff is expected when the bf16 reference value is very close to zero; the absolute error is the more useful signal for this dump. The p99 absolute error is about `2.6e-2`, while the mean absolute error is about `4.3e-3`.

## Interpretation

The observed bf16/fp16 difference is normal for this model path. The model runs a vision-language prefix pass followed by iterative flow denoising, so small numeric differences in attention, MLPs, and AdaRMS conditioning can accumulate across denoising steps. bf16 and fp16 also make different precision tradeoffs: bf16 has a wider exponent range with fewer mantissa bits, while fp16 has more mantissa bits but narrower range. Because the same inputs and noise are used, the measured differences are attributable to dtype-specific arithmetic rather than sampling variance.

For deployment precision matching, use `outputs/pi05_torch_infer/fp16/outputs.npz` as the fp16 golden output and `outputs/pi05_torch_infer/inputs.npz` as the shared replay input set.
