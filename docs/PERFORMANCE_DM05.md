# DM05-LIBERO FlashRT — Performance Specifications

RTX 5060 Ti (SM120, 16 GB, 36 SMs, ~387 GB/s measured), CUDA 13.2, driver 595.84,
PyTorch 2.11.0+cu130, Python 3.12.

Model: DM05-LIBERO. SigLIP vision tower (27 layers, 448 px, 2 views, head_dim 72,
16 heads), Gemma3 prefix (34 layers, D=2560, FF=10240, 8 query / 4 KV heads,
head_dim 256, 5 full + 29 sliding layers) and an action expert (34 layers, D=1024,
FF=4096, 8 query / 4 KV heads, head_dim 256, action dim 32, 10 Euler steps).

The pipeline lives in `flash_rt/models/pi05/` — DM05 is the Pi0.5 architecture at
identical dimensions, so it is a variant pipeline there rather than a separate model
directory. It reaches its kernels through `_ext`, an adapter over this repository's
`flash_rt_kernels`.

## Build Configuration

```sh
cmake -DFLASHRT_ENABLE_DM05=ON -DGPU_ARCH=120 ..
make -j$(nproc) flash_rt_kernels
```

## Kernel Modules

| Module | Contents | Gate |
|--------|----------|------|
| `flash_rt_kernels` | FP8/NVFP4 GEMM, the norm/quantize family, and the three DM05 fusions below | Always built |
| `flash_rt_fa2` | FlashAttention2 BF16/FP8 forward | Always built |

## Entry Point

```python
from flash_rt.models.pi05 import serve
native = serve.enable(inference_config, calibration_state=state)
```

`serve.enable` builds the pipeline, installs this package's preprocessing, and
replaces `inference_config._predict` with a host-staged predict whose entire GPU side
is one CUDA graph replay. Use it rather than wiring the pieces up by hand: the
preprocessing install and `make_predict` have to come from the same module tree, and
mixing two copies of `preprocess.py` fails silently (see Precision Notes).

## End-to-End Inference

Equal-scope E2E: 8 real observations, warm-up, then 7 timed `_predict` +
`synchronize`, all 3675 kernels in one CUDA graph, 0 eager, 0 ATen, 0 Triton.
Measured with the same profiler on the same session as the native control — the
delivered number was taken on an earlier day on a shared card, so a cross-day
comparison would not attribute anything.

| Configuration | E2E median | Kernels/predict | GPU busy |
|---------------|-----------|-----------------|----------|
| DM05 native pipeline (delivered) | 70.62 ms | 3675 | — |
| DM05 native pipeline, same-session control | 70.93 ms | 3700 | 0.9945 |
| **This pipeline** | **70.91 ms** | **3675** | **0.9967** |

The port is level with the native control (70.91 against 70.93, inside noise), carries
five fewer kernels, and sits 0.29 ms from the delivered figure measured on an earlier
day.

### Where the time went

The port initially measured 75.7 ms. Three substitutions accounted for the whole gap;
each was found by diffing two profiler receipts by **kernel time**, then confirmed with
a no-model A/B before being wired in.

| Change | E2E | Kernels |
|--------|-----|---------|
| — | 75.7 ms | 4904 |
| fuse `qk_norm_rope_v8` | 73.9 ms | 3782 |
| vectorize `bias_gelu_quant_fp8_v8` | 71.8 ms | 3782 |
| fuse `res_ln_quant` | 70.91 ms | 3675 |

## Accuracy (444 real LIBERO observations)

| Metric | Value |
|--------|-------|
| Minimum cosine | 0.990153 |
| 1st percentile | 0.996626 |
| Median | 0.999921 |
| Observations below 0.99 | 0 |
| Action vs native, cosine | 0.99999845 (max abs 6.9e-3) |

The gate reads identically across every configuration above — the composition, the
fused kernels and the vectorized kernel all give min 0.990153 / median 0.999921. That
is what makes the latency series attributable: the output never moved, so the clock
only measured kernel work.

## The Three DM05 Fusions

Each of these replaced a composition of this repository's kernels that was already
**bit-exact** with the reference. They are optimisations, not correctness fixes, and
the gate confirms it by not moving.

| Kernel | Replaces | Gain |
|--------|----------|------|
| `gemma3_qk_norm_rope_v8_bf16` | `gemma3_rms_norm_v8_bf16` ×2 + `rope_hf_bf16` ×2 | 1 launch instead of 4; +1191 of the +1204 extra launches |
| `bias_gelu_quantize_fp8_static_bf16_rounded_v8` | the same kernel at one element per thread | 306 → 756 GB/s, 2.47× |
| `res_ln_quant_v8_bf16` | `bias_residual` + `layer_norm` + `quantize_fp8_static` | 3 launches → 1, and removes a 4.7 MB/call scratch |

`gemma3_qk_norm_rope_v8_bf16` was necessary because the existing
`csrc/kernels/qk_norm_rope_fused.cu` is fp16 with a layer norm, laid out for
head_dim 128 (the Chameleon shape); DM05 is bf16 with a `(1+w)` RMSNorm at
head_dim 256, so none of it applies. `bias_gelu_quantize_fp8_static_bf16_rounded_v8`
keeps the sibling's arithmetic verbatim and changes only the access pattern to 8
columns per thread. `res_ln_quant_v8_bf16` keeps the residual and normalised values in
`v0/v1` registers across both passes, which is why it can leave `xn` untouched — the
composition had to write it to DRAM purely because `layer_norm` and
`quantize_fp8_static` are separate kernels.

## Precision Notes

**Division is not IEEE in this repository's kernels.** `flash_rt_kernels` is compiled
with `--prec-div=false`; an externally-built reference is compiled with nvcc defaults,
where division *is* correctly rounded. The two agree exactly when the divisor is a
power of two and differ by an ulp otherwise. The DM05 path pins the division sites
with `__fdiv_rn` — in `resize_norm.cu`, in six places in the `v8_block_sum` family, and
in `layer_norm_kernel`'s mean and inv_std. The last two were found only after a 444
gate run came back at min 0.989149 against a 0.99 threshold with a median of 0.999922:
a handful of elements per 2.36M moved across a bf16 boundary, which at the real
calibration scales flipped a few quantized FP8 values. A tail effect, never a median
one — if the median moves, look for a topology error instead.

**Two copies of `preprocess.py` make two distinct `RawAwareTokenization` classes.**
`install` inserts an instance of whichever copy it was imported from; `make_predict`
tests with `isinstance` against its own. Mixing the two trees leaves the transform
stack without a tokenizer and raises a bare `StopIteration` from inside the predict
path. `serve.enable` exists to keep them on one tree and raises if `install` refuses
the stack.

**The reference's fast-math primitives are pinned in the fusions.**
`rcp.approx.ftz.f32` and `mul.rn.ftz.f32` are reproduced as inline asm in
`res_ln_quant_v8_bf16`; `1.0f / x` is not the same operation. Its reduction likewise
reproduces `frt_block_sum`'s tree rather than calling `block_reduce_sum`.
