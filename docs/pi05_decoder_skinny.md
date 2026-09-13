# Pi0.5 decoder on the skinny FP8 GEMM family (RTX 5090, sm_120a)

The action expert of Pi0.5 runs ten denoising steps over a ten-row action
chunk, and every step streams the whole Gemma-300M decoder (about 311 MB of
FP8 weights) from HBM. On the library GEMM path that decoder took 8.8 ms of
an 18.5 ms graph: 720 launches of a 64x32-tile kernel that was written for
reuse the shape does not have, plus two 3 µs norm kernels per layer.

This page documents the replacement: a small family of kernels that stream
each weight row exactly once, split K across CTAs, hand FP32 partial sums to
one fused consumer per GEMM, and chain through programmatic dependent launch
so every GEMM prefetches its weights while the previous kernel drains.

## What it is

Source: `csrc/kernels/decoder_skinny_fp8_sm120.cu` (bindings `dec_skinny_*`),
selected by the `Pi05Pipeline` / `Pi05TorchFrontendRtx` keyword
`decoder_kernel` (`"auto"` default, `"skinny"`, `"cublaslt"`) or the
environment variable `FLASHRT_PI05_DECODER_KERNEL`. Only the calibrated FP8
decoder uses it; calibration, the BF16 decoder, INT8, the encoder and the
vision tower are unchanged. `FLASHRT_PI05_SKINNY_PDL=0` disables the
programmatic dependent launch attribute for debugging.

Per decoder layer the family issues nine kernels instead of eleven:

| stage | kernel | replaces |
|---|---|---|
| QKV projection | `gemm` (K-split partials) | cuBLASLt |
| | `sum_rope`: scale, BF16 round, RoPE, K/V cache append | `qkv_split_rope` |
| attention | unchanged (FlashAttention-2 split-KV) | |
| output projection | `gemm_bf16_act`: BF16 attention output quantized on load | `quantize_fp8_static` + cuBLASLt |
| | `residual_ada_norm`: gated residual, adaptive RMS norm, FP8 quantize | `gate_residual_ada_norm_fp8` |
| gate/up | `gemm` | cuBLASLt |
| | `gate_gelu_fp8`: GeGLU, FP8 quantize | `gate_geglu_merged_fp8` |
| down | `gemm` | cuBLASLt |
| | `residual_ada_norm` with the next layer's style (or the final norm in BF16 on the last layer) | `gate_residual_ada_norm_fp8` (+ `ada_rms_norm_style` at the end of the step) |

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
about 311 MB) when the family is available. The `"nk"` layout uses the
tensors as they are.

Consumers apply `alpha = a_scale * w_scale` from the device scale buffers,
round the projection to BF16 first (the unfused path consumed a BF16 GEMM
output), then perform exactly the arithmetic of the kernels they replace;
`tests/test_pi05_decoder_skinny.py` checks them bit-identical against those
kernels on identical inputs.

## Numbers

RTX 5090, pi05_libero checkpoint, two 224x224 views, ten steps, FP8,
calibrated on real LIBERO frames, medians.

Standalone GEMM, cold weights (a ring of copies larger than L2), including the
partial-sum consumer, versus the autotuned cuBLASLt FP8 kernel, M = 10:

| shape (N x K) | cuBLASLt µs | family µs, no PDL | family µs, PDL | effective GB/s |
|---|---:|---:|---:|---:|
| qkv 2560 x 1024 | 4.41 | 3.98 | 2.65 | 990 |
| o 1024 x 2048 | 6.13 | 3.79 | 2.34 | 898 |
| gate_up 8192 x 1024 | 7.57 | 7.44 | 6.40 | 1310 |
| down 1024 x 4096 | 8.52 | 5.24 | 3.65 | 1148 |

At M = 80 (eight environments batched) the family wins the two N = 1024
shapes (o 4.27 vs 6.15 µs, down 6.64 vs 8.39) and loses gate_up (14.4 vs 11.0)
where its per-row-tile CTAs re-read the weights through L2; the sum over the
four shapes is even, the consumer fusion and the launch chain still pay.

End to end (`infer` median, host round trip included):

| | library decoder | family | 
|---|---:|---:|
| B = 1, graph replay | 18.56 ms | 14.36 ms |
| B = 1, `infer()` | 19.68 ms | 15.48 ms |
| decoder span inside the graph | 9.08 ms | 5.34 ms |
| B = 4, per environment | 10.09 ms | 9.24 ms |
| B = 8, per environment | 8.73 ms | 8.33 ms |

Inside the graph the launch gaps were already 0.1–0.2 µs, so the gain comes
from kernel efficiency and from overlapping the weight fetch of each GEMM
with its predecessor, not from fewer graph nodes. The remaining decoder time
per layer is roughly 30 µs: about 9 µs in the two FlashAttention-2 split-KV
kernels (no PDL), the rest in the four GEMMs and four consumers whose weight
traffic floor on this GPU is about 11 µs.

Agreement, same prompt, real frames, same noise:

| comparison | cosine (actions) | max abs diff |
|---|---:|---:|
| family vs library decoder, B = 1 | 0.99998 | 0.012 |
| library decoder, this build vs the previous kernel build | 0.99998 | 0.012 |
| family vs library decoder, B = 4, per slot | 0.99998–0.99999 | |
| 300 graph replays of one input | bit-identical | |

The family's difference from the library path is the same size as the
difference between two builds of the library path (cuBLASLt algorithm
selection is timing-based), i.e. within FP8 summation-order noise. The BF16
decoder is untouched and bit-identical across builds.

## Limits and next steps

- sm_120a only (`mma.kind::f8f6f4`); other targets keep the library path
  automatically (`dec_skinny_available()` gates it at run time).
- `residual_ada_norm` is specialised for the 1024-wide decoder.
- The attention pair (about 9 µs per layer) is the largest remaining item;
  a single-kernel decoder attention that joins the PDL chain is the next
  step, followed by a per-row-tile weight-sharing variant for M > 16.
