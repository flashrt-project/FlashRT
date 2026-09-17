# Pi0.5 prefix on NVFP4 (RTX 5090, sm_120a): an optional tier

Build with `-DFLASHRT_ENABLE_PI05_NVFP4=ON` to use this optional tier.
The option defaults to OFF and requires `GPU_ARCH=120`.

The Pi0.5 prefix (SigLIP-L over two views plus the Gemma-2B encoder over
about 520 tokens) is compute-bound on RTX 5090. On this GPU the per-tensor
FP8 GEMMs with FP32 accumulation that cuBLASLt runs top out around 500-600
TFLOPS, while the block-scaled NVFP4 tensor-core path runs 1000-1300
TFLOPS. With the decoder on the skinny family (`docs/pi05_decoder_skinny.md`)
the prefix was two thirds of the graph, so this page adds a prefix tier
that runs the vision and encoder GEMMs on NVFP4 weights and activations.

It is a tier, not the default: 4-bit operands cost about 2.5e-3 of cosine
against the BF16 engine on real frames where FP8 costs 5e-5. Use it where
that is acceptable (off-policy data collection, evaluation sweeps) and
qualify it per task.

## Use

```python
rt = Pi05TorchFrontendRtx(ckpt, num_views=2, prefix_precision="nvfp4")   # default "fp8"
```

or `FLASHRT_PI05_PREFIX_PRECISION=nvfp4`. Needs the FP8 frontend
(`use_fp8=True`), an sm_120a kernel build and a Blackwell GPU; the decoder
keeps its FP8 path (calibrated as before). `reload_weights` re-quantizes the
NVFP4 weights in place.

## What runs where

- Weights: every vision and encoder GEMM (SigLIP qkv/o/up/down, the
  multimodal projector, Gemma qkv/o/gate|up/down) is quantized at load to
  e2m1 with per-16 UE4M3 block scales in the swizzled layout and a
  per-tensor global scale (`bf16_weight_to_nvfp4_swizzled`), laid out
  [N, K]. SigLIP FFN down has K = 4304, which the GEMM does not accept; its
  weight is padded to K = 4352 with zero columns and the activation is
  staged into a zero-tailed buffer by a strided device copy.
- Activations: quantized per call to e2m1 with per-16 UE4M3 scales
  (`quantize_bf16_to_nvfp4_swizzled_v2`); no calibration. The encoder FFN
  tail uses one fused kernel, `pi05_geglu_merged_to_nvfp4_swizzled`, that reads
  the merged [gate | up] buffer once, applies GeGLU and quantizes (bit
  identical to the unfused GeGLU + quantizer pair).
- GEMM: `fp4_w4a16_gemm_sm120_bf16out_pingpong` (the pingpong schedule
  won every prefix shape in the standalone sweep).
- Dispatch: the pipeline's `_fp8_gemm` (single) and `_fp8_gemm_b2`
  (batched) route a site to NVFP4 when its name is in the frontend's
  NVFP4 table; the encoder's fused FP8 norm→GEMM branch and the FP8
  autotuning of prefix shapes are skipped in this tier. The FP8 copies of
  the prefix weights are still built (1.8 GB) so the calibration and
  fallback paths stay as they are; dropping them is a later cleanup.

## Numbers

Standalone GEMM sweep, cold weights, RTX 5090 (TFLOPS; cosine against an
FP32 matmul of the same BF16 operands, Gaussian data):

| shape (N x K) | M | per-tensor FP8 (cuBLASLt) | block-128 FP8 (CUTLASS) | NVFP4 pingpong | NVFP4 cos / FP8 cos |
|---|---:|---:|---:|---:|---|
| enc gate_up 32768 x 2048 | 520 | 538 | 347 | 996 | 0.990 / 0.9993 |
| enc down 2048 x 16384 | 520 | 365 | 184 | 677 | 0.990 / 0.9993 |
| enc qkv 2560 x 2048 | 520 | 344 | 203 | 546 | 0.990 / 0.9993 |
| enc gate_up | 4160 | 584 | 419 | 1284 | |
| enc down | 4160 | 617 | 359 | 1248 | |
| enc qkv | 4160 | 632 | 422 | 1222 | |
| vis up 4304 x 1152 | 4096 | 473 | n/a | 1008 | |

The block-128 FP8 CUTLASS kernel is slower than the library path at every
shape and was dropped.

End to end, pi05_libero, two views, FP8 decoder on the skinny family:

| | FP8 prefix | NVFP4 prefix |
|---|---:|---:|
| B = 1 `infer()` | 14.04 ms | 11.82 ms |
| B = 1 graph replay | 12.94 ms | 10.64 ms |
| B = 4, per environment | 8.73 ms | 6.85 ms |
| B = 8, per environment | 7.89 ms | 5.96 ms |

Agreement on real LIBERO frames, same prompt and noise (six frames):

| comparison | cosine min | median |
|---|---:|---:|
| FP8 prefix vs BF16 engine | 0.99968 | 0.99995 |
| NVFP4 prefix vs BF16 engine | 0.99568 | 0.99755 |
| NVFP4 prefix vs FP8 prefix | 0.99542 | 0.99735 |
| batched B = 4 vs single, NVFP4, per slot | 0.99798 | 0.99986 |

Replays are bit-identical. `tests/test_pi05_prefix_nvfp4.py` gates the
fused quantizer bit-for-bit against the unfused pair, the tier against FP8
on synthetic frames (0.99-0.998 there), the batched pipeline and a reload.

## Limits and next steps

- sm_120a only; the FP4 kernels reach about 60 % of the 1990 TFLOPS the
  instruction rate allows, so the prefix still has headroom in the GEMMs
  themselves.
- The activation quantizer for K = 4352 (SigLIP down) falls back to the
  row-per-CTA kernel; the strided staging copy costs 27 x 4.5 µs per
  forward. Writing the FFN up projection into a padded buffer would remove
  both.
- Per-task success-rate qualification decides whether a deployment uses
  this tier; the numbers above are the engineering gate only.
