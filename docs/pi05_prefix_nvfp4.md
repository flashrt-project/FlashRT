# Pi0.5 on NVFP4 (RTX 5090, sm_120a): an optional tier

Build with `-DGPU_ARCH=120 -DFLASHRT_ENABLE_PI05_SKINNY=ON
-DFLASHRT_ENABLE_PI05_NVFP4=ON` to use this optional tier. Both options
default to OFF.

The tier runs the Pi0.5 vision tower, the Gemma-2B FFNs and the action
decoder on NVFP4 (e2m1 values, UE4M3 scales per 16 elements, a per-tensor
global scale). Every quantization happens inside the kernel that produces the
operand — a norm, a GEMM epilogue or a decoder consumer — so the graph has no
standalone quantize or dequantize pass around the 4-bit GEMMs.

It is a tier, not the default: against the FP8 skinny profile it trades about
1e-4 of action cosine (vs the FP16 engine) for ~1.46x lower E2E latency.
Qualify it per task before deploying; the LIBERO-10 closed-loop numbers are
below.

## Use

```python
model = flash_rt.load_model(ckpt, config="pi05", framework="torch",
                            hardware="rtx_sm120", num_views=2, use_fp4=True)
```

`use_fp4=True` on `rtx_sm120` selects both halves of the tier. They can also be
chosen separately on the frontend:

```python
rt = Pi05TorchFrontendRtx(ckpt, num_views=2,
                          prefix_precision="nvfp4",    # default "fp8"
                          decoder_precision="nvfp4")   # default "fp8"
```

or with `FLASHRT_PI05_PREFIX_PRECISION` / `FLASHRT_PI05_DECODER_PRECISION`.
The tier needs the FP8 frontend (`use_fp8=True`): its calibration supplies the
static global scales below. `reload_weights` re-quantizes in place.

## What runs where

| stage | precision | kernels per layer |
|---|---|---|
| SigLIP attention (27 layers) | NVFP4 | LayerNorm → e2m1; QKV GEMM with the bias in its epilogue; attention read straight from the QKV rows (`fused_qkv_attention`, head_dim 72 padded to 80, no head-split pass); output → e2m1; O GEMM with bias + residual in its epilogue |
| SigLIP FFN | NVFP4 | LayerNorm → e2m1; up GEMM with bias + GELU → e2m1 epilogue (hidden padded 4304 → 4352); down GEMM with bias + residual |
| Gemma attention (18 layers) | FP8 | 4-bit Q/K at head_dim 256 costs ~0.2 action cosine on real frames, so Q/K/V/O stay FP8; FA2 runs unsplit on SM120 |
| Gemma FFN | NVFP4 | residual + RMSNorm → e2m1; gate\|up GEMM over row-interleaved weights with GeGLU → e2m1 in the epilogue; even-column compaction; down GEMM (stream-K) |
| decoder (10 steps x 18 layers) | NVFP4 | skinny W4A4 family: weights prefetched into registers before the programmatic-dependent-launch wait, K-split partials, fused consumers (residual + AdaRMSNorm → e2m1, gate GELU → e2m1, Q/K RoPE) |

Static global scales: the inputs of the down projections are GELU / GeGLU
outputs whose range reaches ~1e4 in late Gemma layers, beyond the 2688 that a
global scale of 1 can represent. Their global scale is the FP8 calibration's
`amax / 448`, applied as the producing epilogue's norm constant and folded into
the down GEMM's alpha. Normalized inputs run with a global scale of 1. Weights
use `amax / (6 * 448)`; the prefix weights also search the block scale for the
least squared error (on the decoder that search measured slightly worse, so its
block scales round up).

`FLASHRT_PI05_ENC_O_NVFP4=1` (opt-in) moves the Gemma attention output
projection to NVFP4 as well: ~1% faster, action cosine min 0.9992 → 0.9990 on
the frames below.

## Numbers

RTX 5090, 2 views, openpi `pi05_libero` (converted to PyTorch with openpi's
`examples/convert_jax_model_to_pytorch.py`). Synchronized observation-to-final-
action E2E with `tools/bench_pi05_e2e.py` (200 iterations after 30 warmups,
8 LIBERO frames; every row uses the same build, checkpoint and fixture):

| `--profile` | prefix | decoder | state in prompt | p50 | p95 |
|---|---|---|---|---:|---:|
| `default` | FP8 | FP8 | yes | 20.88 ms | 21.17 ms |
| `skinny` | FP8 | FP8 skinny | yes | 15.08 ms | 15.28 ms |
| `nvfp4` | NVFP4 | FP8 skinny | yes | 11.18 ms | 11.41 ms |
| `nvfp4-full` | NVFP4 | NVFP4 skinny | yes | 10.39 ms | 10.60 ms |
| `skinny --no-state-prompt` | FP8 | FP8 skinny | no | 13.89 ms | 14.12 ms |
| `nvfp4-full --no-state-prompt` | NVFP4 | NVFP4 skinny | no | 9.31 ms | 9.46 ms |

`pi05_libero` is trained without the state in the prompt
(`discrete_state_input=False`), so the `--no-state-prompt` rows are its
deployment latency; the state rows add the per-frame prompt rebuild. The
captured graph of the `nvfp4-full` tier replays in 8.24 ms (vision 1.75,
encoder 3.3, decoder 3.25 ms under CUPTI); the rest of the E2E is image
upload, action download and the Python call.

Agreement on 24 real LIBERO frames, same prompt and pinned noise, against the
FP16 frontend:

| tier | action cosine mean | min |
|---|---:|---:|
| FP8 prefix + FP8 skinny decoder | 0.99998 | 0.99997 |
| NVFP4 prefix + NVFP4 decoder | 0.99986 | 0.99915 |

Replays are bit-identical. `tests/test_pi05_prefix_nvfp4.py` gates the tier
against FP8, the NVFP4 decoder, the batched pipeline and a weight reload.

### LIBERO-10 closed loop

`use_fp4=True` on LIBERO-10, 10 seeds (7, 17, ..., 97) x 10 tasks x 50
episodes = 5000 episodes: **4666 / 5000 = 93.32%** (Wilson 95% interval
92.59-93.98%; per seed 91.8-94.6%, standard deviation 0.85 points).

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| success | 94.6% | 99.4% | 95.0% | 96.8% | 97.8% | 99.2% | 93.6% | 99.8% | 70.4% | 86.6% |

Protocol: openpi's `examples/libero/main.py` unchanged except that the policy
server call is `model.predict(images=[agentview, wrist], prompt=task)` (images
rotated 180 degrees, `openpi_client.image_tools.resize_with_pad` to 224 and
`convert_to_uint8`, 10 settle steps, replanning every 5 actions, 520 steps,
the suite's initial states, no state input). Each task prompt is calibrated
with `calibrate()` on 16 LIBERO frames before its first episode.

## Limits

- sm_120a only.
- The Gemma attention (FP8 projections, FA2 at head_dim 256, ~31 us per layer)
  is now the largest prefix cost after the GEMMs.
- The numbers above are the engineering gate; closed-loop success rates decide
  per task whether a deployment uses the tier.


## JAX / Orbax frontend compatibility

The Orbax frontend accepts the same `decoder_precision` and `prefix_precision`
options as the Torch frontend. Both default to `"fp8"`. To select the complete
NVFP4 tier directly from an Orbax checkpoint:

```python
from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx

model = Pi05JaxFrontendRtx(
    "/path/to/pi05_libero", num_views=2, hardware="rtx_sm120",
    prefix_precision="nvfp4", decoder_precision="nvfp4",
)
```

The frontend initializes the separate FFN and decoder NVFP4 weight stores and
calls their current shared quantizers. `decoder_precision="nvfp4"` selects the
skinny decoder unless the caller explicitly supplies an incompatible kernel,
which is rejected before checkpoint loading. SM120 encoder split-KV selection
also follows the shared Torch helper. Orbax checkpoint conversion is unchanged.

### Runtime library version

The verified NGC `nvcr.io/nvidia/pytorch:25.10-py3` environment supplies
cuBLASLt **13.1**. In the audit follow-up on RTX 5090, system cuBLASLt 13.0
returned `CUBLAS_STATUS_NOT_SUPPORTED` from the default `kn` FP8 GEMM heuristic
on both main and this branch. The same minimal GEMM and full inference passed
with the NGC cuBLASLt 13.1 library. Check the actual library used by the process;
the CUDA compiler version alone does not identify its cuBLASLt version.

For example, in the intended runtime environment:

```python
import ctypes

library = ctypes.CDLL("libcublasLt.so.13")
library.cublasLtGetVersion.restype = ctypes.c_size_t
print(library.cublasLtGetVersion())  # 130100 in the verified environment
```

Use a runtime with the verified library support. The frontend does not silently
change FP8 layout in response to a failed cuBLAS heuristic.


### Compatibility follow-up (2026-10-16)

Checked on RTX 5090 with the actual Orbax `pi05_libero` checkpoint and its
converted PyTorch checkpoint, 8 real LIBERO observations, 2 cameras. Runtime:
PyTorch 2.11.0+cu128, JAX 0.11.2 / Orbax 0.12.7 / Flax 0.12.10 for CPU weight
loading, and cuBLASLt 13.1 from NGC 25.10. Both revisions used the same runtime
and observations. Baseline: main `b171b5a6` (including #227 and #220).

Synchronized CPU-observation-to-final-action timing, fixed text prompt without
state tokens, 30 warmups and 200 timed observations per process. FP8 values are
the median of 3 fresh processes per revision, with alternating revision order;
NVFP4 is one 200-iteration process. Loading, calibration and graph capture are
excluded.

| frontend / tier | main p50 | branch p50 | branch p95 |
| --- | ---: | ---: | ---: |
| TORCH FP8 | 18.873 ms | 18.736 ms | 18.772 ms |
| JAX FP8 | 18.900 ms | 18.769 ms | 18.802 ms |
| JAX NVFP4 prefix + decoder | — | 8.778 ms | 8.872 ms |

FP8 showed no slowdown in this comparison. With fixed noise over the 8 frames,
JAX FP8 vs main had action cosine min **0.999935**, and JAX NVFP4 vs FP8 had
min **0.996538** (the existing tier gate is 0.99). Repeating an observation after
other observations produced identical actions with fixed noise. Consecutive
public `predict()` calls returned finite `(10, 7)` arrays.

Regression checks:

- Constructor/API checks: **71 passed**, with the existing Thor routing test
  excluded after confirming its same failure on main.
- Existing prefix/decoder NVFP4, batched and weight-reload tests: **5 passed**.
- New `tests/test_pi05_jax_precision.py`: checks both FP8 and NVFP4 against real
  Orbax weights and real observations, including graph replay and public calls.
  Run it with `PI05_LIBERO_ORBAX_CHECKPOINT` and
  `PI05_REAL_OBSERVATIONS_NPZ` set to local input paths.

This follow-up did not rerun LIBERO closed-loop task success or GR00T model
benchmarks. The follow-up code changes are confined to Pi0.5 JAX initialization.


### Latest verified modes (2026-10-16)

RTX 5090, `pi05_libero`, two views, eight real observations, fixed text prompt
without state tokens. The latest confirmation measured each mode with
30 warmups and 200 synchronized observation-to-action calls.
The table reports that run's p50/p95 and reciprocal p50 frequency;
loading, calibration and capture are excluded. Runtime is cuBLASLt 13.1.

| Mode | p50 | p95 | Frequency |
|---|---:|---:|---:|
| FP8 skinny decoder | 13.10 ms | 13.18 ms | 76 Hz |
| NVFP4 prefix + decoder | 8.75 ms | 8.78 ms | 114 Hz |

The two checkpoint loaders use the same inference pipeline and precision tiers.
These latest timing samples used the safetensors loader.
The confirmation agrees with three earlier fresh-process measurements
(FP8 median p50 13.10 ms; NVFP4 median p50 8.74 ms). Earlier Orbax samples
above used the same pipeline, model family, real observations and timing boundary.
No closed-loop task evaluation was repeated for this documentation update.
