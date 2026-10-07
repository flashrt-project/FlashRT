# Thor verification results

## Published image validation

Image: `ghcr.io/flashrt-project/flashrt-thor:thor-v0.1.1`.

Verified digest: `sha256:031908d864d047c08ce91aaf8567c7258e8fdf99f70c05067ca41220027aadff`.

An empty Docker credential directory pulled this digest anonymously. The
pulled image ran `validate all` and the CPU-two-thread official eager check;
both exited 0. Existing model files and Docker layer storage were reused on
the validation Thor. No host source or reference environment was mounted.

| Model / tier | Mean cosine | Latency | Result |
|---|---:|---:|---|
| [π0.5 FP8](../../repro/thor/evidence/published-image/pi05-fp8-accuracy.json) | 0.999900 | 40.63 ms | PASS |
| [π0.5 FP4](../../repro/thor/evidence/published-image/pi05-fp4-accuracy.json) | 0.999691 | 20.03 ms | PASS |
| [GR00T LIBERO FP8](../../repro/thor/evidence/published-image/groot-fp8.json) | 0.999467 | 40.85 ms complete call | PASS |
| [GR00T LIBERO FP4](../../repro/thor/evidence/published-image/groot-fp4.json) | 0.999694 | 27.91 ms complete call | PASS |

π0.5 timing and options: [FP8](../../repro/thor/evidence/published-image/pi05-fp8.json),
[FP4](../../repro/thor/evidence/published-image/pi05-fp4.json).

## LIBERO one camera

GR00T `libero_10`: one camera, batch 1, four denoising steps, two CPU threads,
20 warmups and 100 measured calls.

| Tier | Model inference | Preprocessing | Physical decode | Complete call |
|---|---:|---:|---:|---:|
| FP8 | 37.26 ms | 3.23 ms | 0.22 ms | 40.85 ms |
| FP4 | 24.27 ms | 3.25 ms | 0.22 ms | 27.91 ms |

The [matched official eager run](../../repro/thor/evidence/published-image/groot-official-eager.json)
measured 93.66 ms complete-call latency. The prior run of this image measured
FP4 model 24.26 ms / preprocessing 3.25 ms / complete call 27.89 ms.

Overall numerical and repeatability gates passed. Strict action-group
diagnostics did not all pass: FP4 rotation cosine 0.96167 / maximum absolute
error 0.00760; FP8 rotation cosine 0.97314 / maximum absolute error 0.00702.
All group errors remain in the linked reports. These are fixed-input checks,
not robot task-success results.

Latency depends on power mode and clocks. The recorded device used modified
power-protection settings; these numbers are not a stock-power guarantee.
No same-device TensorRT speedup was measured.

## Environment and audit records

- [Runtime](../../repro/thor/evidence/published-image/runtime.json)
- [Anonymous validation and full image privacy audit](../../repro/thor/evidence/published-image/validation-summary.json)
- [Earlier full source Docker build](../../repro/thor/evidence/public-docker)
- [Earlier second-Thor native setup](../../repro/thor/evidence/cross-machine/installation-scope.json)

<details>
<summary>Earlier DROID feature-input measurements</summary>

These use the base DROID checkpoint with two cameras and two history frames;
they exclude image/text feature generation and physical decoding.

| Tier | Latency | Report |
|---|---:|---|
| FP8 | 48.94 ms | [Report](../../repro/thor/evidence/docker/groot-fp8.json) |
| FP4 | 29.60 ms | [Report](../../repro/thor/evidence/docker/groot-fp4.json) |

</details>
