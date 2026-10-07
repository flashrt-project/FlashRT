# Verified results — 2026-10-07

Both routes passed all four numerical checks. Docker `run-validation.sh all` returned exit code 0. Native π0.5 and GR00T checks each passed.

| Route | Model/tier | Mean cosine | Worst cosine | RMSE | Max absolute error | p50 ms | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| docker | pi05 FP8 | 0.999900 | 0.999744 | 0.006363 | 0.046260 | 40.51 | [PASS](../../repro/thor/evidence/docker/pi05-fp8-accuracy.json) |
| docker | pi05 FP4 | 0.999691 | 0.999299 | 0.013666 | 0.081038 | 20.06 | [PASS](../../repro/thor/evidence/docker/pi05-fp4-accuracy.json) |
| docker | groot FP8 | 0.999958 | 0.999958 | 0.007647 | 0.032351 | 48.94 | [PASS](../../repro/thor/evidence/docker/groot-fp8.json) |
| docker | groot FP4 | 0.999844 | 0.999844 | 0.014266 | 0.059842 | 29.60 | [PASS](../../repro/thor/evidence/docker/groot-fp4.json) |
| native | pi05 FP8 | 0.999902 | 0.999755 | 0.006318 | 0.045361 | 40.67 | [PASS](../../repro/thor/evidence/native/pi05-fp8-accuracy.json) |
| native | pi05 FP4 | 0.999692 | 0.999294 | 0.013605 | 0.079344 | 20.21 | [PASS](../../repro/thor/evidence/native/pi05-fp4-accuracy.json) |
| native | groot FP8 | 0.999950 | 0.999950 | 0.008030 | 0.032921 | 49.10 | [PASS](../../repro/thor/evidence/native/groot-fp8.json) |
| native | groot FP4 | 0.999776 | 0.999776 | 0.016821 | 0.060902 | 29.66 | [PASS](../../repro/thor/evidence/native/groot-fp4.json) |

π0.5: official OpenPI checkpoint/config, 8 LIBERO frames, 2 real cameras, horizon 10, 7 delivered action dimensions. Both calibration and comparison use the same frames. All gripper signs agree.

GR00T: fixed real DROID sample, 2 cameras × 2 historical frames, raw RGB 180×320, 1024 patches/277 tokens, 4 diffusion steps, horizon 40, 17 delivered dimensions. Reference was freshly executed from the raw observations. Each JSON includes per-modality errors and repeated-input stability.

Timing boundaries differ: π0.5 uses the Python observation-to-action API; GR00T uses post-patch features/image-text embeddings to normalized actions, excluding their generation and physical decode. warmup=20, iterations=100, batch=1. The official instrumented capture timing is not a performance baseline. No cross-boundary speedup is claimed.

These latencies are verification observations on the supplied machine, whose EXT_POWER/gpu_slow_factor was already 0. Native measurements overlapped CPU-only dependency building. Rebenchmark serially under recorded stock hardware protection before publishing performance.

Docker: pinned NGC torch 2.12.0a0+5aff3928d8.nv26.05 / CUDA 13.2. Native: CUDA torch 2.14.0+cu130 / runtime 13.0, host toolkit/driver 13.2. See runtime JSON, freezes and [reproduction guide](README.md). Numerical similarity is not robot task success rate.

## Packaged release candidate

Both the internal CLI candidate and the source-only repackaged release candidate completed `validate all` with exit code 0. See [release build and validation log](../../repro/thor/evidence/release-validation.log). Public registry push and anonymous pull have not yet been tested because the team registry account is not configured.

- [pi05-fp8-accuracy.json](../../repro/thor/evidence/release/pi05-fp8-accuracy.json): PASS.
- [pi05-fp4-accuracy.json](../../repro/thor/evidence/release/pi05-fp4-accuracy.json): PASS.
- [groot-fp8.json](../../repro/thor/evidence/release/groot-fp8.json): PASS.
- [groot-fp4.json](../../repro/thor/evidence/release/groot-fp4.json): PASS.

Published evidence replaces private host paths with generic paths and omits sudo user prompts. Numerical results and runtime versions are unchanged.
