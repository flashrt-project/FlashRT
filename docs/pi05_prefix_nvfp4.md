# Pi0.5 prefix on NVFP4 (RTX 5090, sm_120a): an optional tier

Build with `-DFLASHRT_ENABLE_PI05_NVFP4=ON` to use this optional tier.
The option defaults to OFF and requires `GPU_ARCH=120`.

This optional tier runs the Pi0.5 prefix (SigLIP-L over two views plus the
Gemma-2B encoder) on NVFP4 weights and activations. It can be combined with
the explicitly selected skinny decoder (`docs/pi05_decoder_skinny.md`).

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

Measure synchronized observation-to-final-action E2E with
`tools/bench_pi05_e2e.py --profile nvfp4`, using the same checkpoint,
observation fixture, input contract and container as `--profile skinny`.
Both profiles explicitly select skinny; only the prefix precision differs.
See the skinny documentation for the complete timing boundary. Kernel and
partial graph timings are not E2E results.

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

- sm_120a only.
- The activation quantizer for K = 4352 (SigLIP down) falls back to the
  row-per-CTA kernel. Writing the FFN up projection into a padded buffer
  could remove the strided staging copy.
- Per-task success-rate qualification decides whether a deployment uses
  this tier; the numbers above are the engineering gate only.
