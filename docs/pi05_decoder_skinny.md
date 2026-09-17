# Pi0.5 decoder on the skinny FP8 GEMM family (RTX 5090, sm_120a)

This experimental path requires `-DFLASHRT_ENABLE_PI05_SKINNY=ON` at
build time and `decoder_kernel="skinny"` at runtime. Both the frontend
and pipeline default to `cublaslt`; a default build excludes these kernels.

The action expert of Pi0.5 runs ten denoising steps over a ten-row action
chunk, and every step streams the whole Gemma-300M decoder (about 311 MB of
FP8 weights) from HBM. This small-row workload has limited weight reuse
and benefits from model-specific GEMMs and fused consumers.

This page documents the replacement: a small family of kernels that stream
each weight row exactly once, split K across CTAs, hand FP32 partial sums to
one fused consumer per GEMM, and chain through programmatic dependent launch
so every GEMM prefetches its weights while the previous kernel drains.

## What it is

Source: `csrc/kernels/pi05/pi05_decoder_skinny_fp8_sm120.cu` (bindings `pi05_dec_skinny_*`),
selected by the `Pi05Pipeline` / `Pi05TorchFrontendRtx` keyword
`decoder_kernel` (`"cublaslt"` default, `"skinny"`, `"auto"`) or the
environment variable `FLASHRT_PI05_DECODER_KERNEL`. Only the calibrated FP8
decoder uses it; calibration, the BF16 decoder, INT8, the encoder and the
vision tower are unchanged. `FLASHRT_PI05_SKINNY_PDL=0` disables the
programmatic dependent launch attribute for debugging.

Per decoder layer the family issues ten kernels instead of eleven, and
per denoising step two more replace the seven around the layer stack:

| stage | kernel | replaces |
|---|---|---|
| step start | `action_in_norm`: input projection, bias, first adaptive norm to FP8 | cuBLASLt (K = 32) + bias add + `ada_rms_norm_style_fp8` |
| QKV projection | `gemm` (K-split partials) | cuBLASLt |
| | `sum_rope`: scale, BF16 round, RoPE, K/V cache append | `qkv_split_rope` |
| attention | `attn` split-KV pair: one CTA per 64 keys, head and sample (QK^T and PV on `mma.sync` bf16, K/V/Q staged with `cp.async`, V overlapped with QK^T), then one CTA per row, head and sample combining the partials; keys past the device count are masked, both launches join the PDL chain | FlashAttention-2 split-KV pair (outside the chain) |
| output projection | `gemm_bf16_act`: BF16 attention output quantized on load | `quantize_fp8_static` + cuBLASLt |
| | `residual_ada_norm`: gated residual, adaptive RMS norm, FP8 quantize | `gate_residual_ada_norm_fp8` |
| gate/up | `gemm` | cuBLASLt |
| | `gate_gelu_fp8`: GeGLU, FP8 quantize | `gate_geglu_merged_fp8` |
| down | `gemm` | cuBLASLt |
| | `residual_ada_norm` with the next layer's style (or the final norm in BF16 on the last layer) | `gate_residual_ada_norm_fp8` (+ `ada_rms_norm_style` at the end of the step) |
| step end | `action_out_residual`: output projection (K = 1024, N = 32), bias, trace copies, Euler update | cuBLASLt (12.9 µs for a 64 KB weight) + bias add + two copies + `residual_add` |

The GEMM kernel: `mma.sync.kind::f8f6f4.m16n8k32` on e4m3 operands, weight
rows `[N, K]` loaded straight from global memory into the B fragments (a
per-lane 16-byte load covers 16 consecutive k of one row; A and W share the
same k permutation, so the reduction is exact up to FP32 summation order),
the activation tile staged once in shared memory, one CTA per (BN weight
rows, KC k, 16 activation rows). Partial sums are written as
`[K / KC][M][N]` FP32 and reduced by the consumer in a fixed order, so
replays are bit-identical. Rows beyond 16 use a grid dimension; the weight
re-reads then come from L2.

The `"kn"` FP8 layout used on sm_120 keeps the decoder weights as `[K, N]`
for the library path; the frontend adds a transposed `[N, K]` copy of the 72
decoder tensors under `<name>__nk` (same values, same per-tensor scale,
about 311 MB) only when the family is explicitly selected. The `"nk"` layout uses the
tensors as they are.

Consumers apply `alpha = a_scale * w_scale` from the device scale buffers,
round the projection to BF16 first (the unfused path consumed a BF16 GEMM
output), then perform exactly the arithmetic of the kernels they replace;
`tests/test_pi05_decoder_skinny.py` checks them bit-identical against those
kernels on identical inputs.

## Numbers

Report only synchronized observation-to-final-action E2E. Use the same
container, checkpoint, observations, seed and input contract for both arms:

```bash
PYTHONPATH=. python tools/bench_pi05_e2e.py \
  --checkpoint /path/to/pi05_checkpoint \
  --observations /path/to/observations.npz \
  --profile default --output /path/to/default.json
PYTHONPATH=. python tools/bench_pi05_e2e.py \
  --checkpoint /path/to/pi05_checkpoint \
  --observations /path/to/observations.npz \
  --profile skinny --output /path/to/skinny.json
```

The default benchmark includes per-observation normalized state in the
prompt, image preprocessing and transfer, vision/prefix, all ten denoise
steps, and action unnormalization/download. It synchronizes before and after
the timed call; loading, calibration and initial graph capture are excluded.
The fixed-text-only contract requires an explicit `--no-state-prompt`.
Do not compare timings across these input contracts or report graph/kernel
time as E2E. JSON records the contract, environment, fixture and kernel hashes;
the companion NPZ records final actions for cross-arm numerical comparisons.

The tests separately cover fused-consumer rounding, deterministic replay,
attention parity and final-action agreement with cuBLASLt. Approximate FP8
summation orders may differ; task-level qualification is still required.

## Limits and next steps

- sm_120a only (`mma.kind::f8f6f4`); other targets keep the library path
  automatically (`pi05_dec_skinny_available()` gates it at run time).
- `residual_ada_norm` is specialised for the 1024-wide decoder.
- The attention kernels assume the packed decoder output buffer and a
  chunk of at most 16 rows; other configurations keep the FlashAttention-2
  pair (`FLASHRT_PI05_SKINNY_ATTN=0` forces it).
- Next: fold the consumers into last-CTA GEMM epilogues, a weight-sharing
  variant for M > 16 (batched rollouts), and the prefix, which is now two
  thirds of the graph.
