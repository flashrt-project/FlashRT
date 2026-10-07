# Verified Thor results

## LIBERO one camera

The current GR00T default is **GR00T-N1.7-LIBERO/libero_10**, one camera,
four denoising steps, batch 1 and two CPU threads. The native public validator
directly measured these medians over 100 calls after 20 warmups:

| Tier | Model inference | Preprocessing | Physical decode | Complete call | Overall cosine |
|---|---:|---:|---:|---:|---:|
| FP4 | **24.09 ms** | **4.68 ms** | 0.23 ms | **29.45 ms** | **0.999825** |

Model inference includes input transfer, patch/vision backbone and the action
head. Complete-call latency is measured independently; component medians do
not necessarily add up to its median. No images or state are cached.

The overall numerical acceptance passed and repeated actions were bitwise
identical. Strict rotation cosine diagnostics did not pass: cosine 0.97134,
maximum absolute error 0.00556. All diagnostics remain in the
[numerical report](../../repro/thor/evidence/libero/groot-fp4-native.json).
This checks a fixed real observation, not robot task success.

In a separate balanced same-process thread comparison, 2 versus 8 CPU threads
reduced full-call median from 30.19 to 29.00 ms, with bitwise-identical output.
That controlled gain is 1.19 ms (3.9%); the table above uses the final public
validator rather than substituting the separate optimization probe.

The sections below retain earlier DROID/base-checkpoint and installation
validation records. They use different inputs and timing boundaries.

### Cold-start FP8 correction

The Thor FP8 attention specification now covers the full padded KV stride
when initializing the LLM and visual self-attention logits. The earlier
unpadded initialization left tail cells undefined for non-aligned sequence
lengths. With the targeted correction, the [fresh native FP8 check](../../repro/thor/evidence/libero/groot-fp8-native.json)
achieved overall cosine **0.999748** and bitwise-identical repeated output.
This keeps the original attention backend and quantization configuration.

Calibration-cache identity also includes actual sharded weight contents,
configuration/statistics and calibration inputs, so distinct fine-tunes or
observations do not reuse scales merely because their shard-index layout matches.

## Earlier DROID and installation checks

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

## Independent Thor native reproduction

On a second Jetson Thor, all three CUDA targets were compiled from the packaged source. Separate reference environments were installed for the pinned official OpenPI, LeRobot and Isaac GR00T revisions. `run-validation.sh all` completed with exit code 0.

| Model/tier | Mean cosine | Worst cosine | RMSE | Max absolute error | Latency ms | Gate |
|---|---:|---:|---:|---:|---:|---|
| π0.5 FP8 | 0.999900 | 0.999742 | 0.006365 | 0.047594 | 38.52 | PASS |
| π0.5 FP4 | 0.999690 | 0.999297 | 0.013618 | 0.079562 | 20.16 | PASS |
| GR00T FP8 | 0.999950 | 0.999950 | 0.008030 | 0.032921 | 48.02 | PASS |
| GR00T FP4 | 0.999776 | 0.999776 | 0.016821 | 0.060902 | 28.75 | PASS |

See [numerical evidence](../../repro/thor/evidence/cross-machine/accuracy.json) for the complete per-modality and repeated-input checks. Both π0.5 tiers have zero gripper sign disagreement. These use the same fixtures and timing boundaries described above; GR00T timing excludes image/text feature generation.

This verifies fresh native compilation and new pinned reference installations while reusing existing immutable checkpoints, PyTorch/CUDA dependencies and CUTLASS source. It does not certify a fresh Docker installation on this second machine. Host dependency conflicts required isolated package/search-order adjustments; [installation scope](../../repro/thor/evidence/cross-machine/installation-scope.json) lists each exception. The [environment](../../repro/thor/evidence/cross-machine/environment.json) uses PyTorch 2.14.0+cu130, CUDA 13.0 and L4T R38.2.1. Hardware protection settings were left unchanged; EXT_POWER state was not readable without privileged access, so these latencies are verification observations rather than a certified stock-power benchmark.

## GR00T raw-input validation

The newly shipped GR00T validator defaults to **raw RGB/state/language → physical actions**. On each call it reruns the official processor, then FlashRT patch embedding, visual merger, backbone and action head, and physical action decoding. Prompt/grid graph setup and fixed-sample calibration are outside timing; no official model embeddings are replayed per call. The processor's patches, token IDs and grid exactly matched a freshly captured official run.

| Execution | Mean cosine | EEF cosine | Joint cosine | Latency ms | Result |
|---|---:|---:|---:|---:|---|
| [FlashRT FP8](../../repro/thor/evidence/cross-machine/groot-raw-full-fp8.json) | 0.999835 | 0.999536 | 0.999927 | 76.74 | PASS |
| [FlashRT FP4](../../repro/thor/evidence/cross-machine/groot-raw-full-fp4.json) | 0.999751 | 0.999868 | 0.999718 | 56.69 | PASS |
| [Official eager PyTorch](../../repro/thor/evidence/cross-machine/groot-official-eager.json) | 1.000000 | — | — | 133.38 | Exact reference actions |

All FlashRT per-modality and repeated-input gates passed. These tests use the same base 3B DROID checkpoint, two-camera/history sample, four denoising steps and 40×17 decoded physical actions. Each benchmark uses 20 warmups, 100 measured calls and GPU synchronization before and after the complete call. The eager comparator runs without capture hooks. This verifies one calibrated sample on the recorded native environment; it is not an optimized PyTorch or TensorRT benchmark.

Older GR00T numbers above remain **feature-input** results, reproduced with explicit `--boundary feature`. Their approximately 29 ms excludes processor/embedding generation and physical decoding. The [comparison contract](comparison.md) explains both boundaries and the incompatible checkpoint/camera settings in the current JAL TensorRT tutorial. No JAL TensorRT speedup ratio is claimed.

## Complete public-source Docker build

The full `Dockerfile.thor` build completed on Thor; `validate all` completed successfully. All four checks passed:

- [pi05-fp8-accuracy.json](../../repro/thor/evidence/public-docker/pi05-fp8-accuracy.json): mean cosine 0.99990046, PASS.
- [pi05-fp4-accuracy.json](../../repro/thor/evidence/public-docker/pi05-fp4-accuracy.json): mean cosine 0.99969050, PASS.
- [groot-fp8.json](../../repro/thor/evidence/public-docker/groot-fp8.json): mean cosine 0.99995844, PASS.
- [groot-fp4.json](../../repro/thor/evidence/public-docker/groot-fp4.json): mean cosine 0.99984395, PASS.
