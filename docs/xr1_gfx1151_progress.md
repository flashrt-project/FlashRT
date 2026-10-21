# XR-1 gfx1151 validation report

Initial validation recorded on 2026-10-21. The full-model HIP Graph follow-up
appears below; the initial measurements are retained as a historical snapshot.

Implemented and GPU validated:

- `flash_rt.load_model(config="xr1", framework="torch", hardware="amd_rdna35", num_views=3)`.
- Published XR-1 5B `model_states.pt`: all 1135 tensors load strictly;
  output `(30,60)`, quantile state normalization, upstream camera ordering,
  resize and prompt, action normalization, and asynchronous action prefixes.
- RoboCasa deploy checkpoint: output `(10,7)`, upstream custom processor,
  and changing state while reusing the instruction.
- BF16 reference mode without FlashRT extensions or `flash-attn`.
- Opt-in native HIP DiT GEMM/RMSNorm using existing FlashRT kernels.
  Vision Conv3d retains MIOpen. No new kernel/CMake ABI changes.
- Independent Python 3.12 / ROCm 7.2 environment, setup/build script, pinned
  downloads, normalization preparation, synthetic benchmark and real-video
  validation scripts. See [deployment guide](xr1_gfx1151.md).

Final implementation measurements on this gfx1151 device, PyTorch ROCm 2.10.0,
Transformers 4.57.1, five denoise steps, and experimental AOTriton enabled:

| Input / checkpoint | Reference median | Native median / p95 |
|---|---:|---:|
| Synthetic 3x256x256 / RoboCasa | 435.12 ms | 360.90 / 365.33 ms |
| Synthetic 3x256x256 / 5B | 453.12 ms | 377.52 / 383.86 ms |
| Real demo camera crops / 5B | 707.17 ms | 652.80 ms median |

Synthetic results use ten measured calls after first-call warmup. The real
input medians aggregate three measured calls at each of twelve frames from
`XiaomiRobotics/xr1_post_train_demo/json1` (0 through 440, step 40). Timers
include preprocessing, model execution and action download; model loading
and first-call initialization are recorded separately.

All twelve reference outputs exactly matched the original upstream runtime
message/resize/state utilities and direct generation on the same loaded ROCm
policy, including upstream action decoding. Native real-input action cosine
was at least 0.999851; maximum absolute difference was 0.031529. Fixed-seed
predictions are deterministic and preserve the caller's CPU/GPU RNG. These
checks establish numerical agreement under the tested inputs, not robot
control limits or task success.

A vision patch GEMM optimization was removed after real video exposed larger
BF16 errors than synthetic inputs. The final benchmarks above use retained
MIOpen vision projection and replace the earlier experimental latency figures.

Validation:

- 107 XR-1/API/RDNA tests passed, including both real-checkpoint E2E cases,
  async prefixes, and all twelve real-camera observations.
- The standalone demo validator passed all twelve upstream/reference/native
  comparisons using the final implementation.
- `pip check`, Python/shell syntax checks, and `git diff --check` passed.
- Wheel built and imported from outside the source tree; vendored sources and
  license are included, without compiled binaries or test files.

Final local evidence under `results/xr1/`:

- `final_tests.txt`
- `robocasa_reference_final.json`, `robocasa_native_final.json`
- `5b_reference_final.json`, `5b_native_final.json`
- `demo_validation_final.json` and matching `.npz` observations
- `real_upstream_parity.json`, `wheel_build_final.txt`

Weights, environments, videos and generated evidence remain local and are
excluded from Git.

Practical limits:

- The 5B smoke checkpoint uses upstream `load_washer` demonstration statistics.
  Deployment requires the target embodiment's statistics and post-training.
- No robot/simulation task-success benchmark was run and no robot was controlled.
- Native BF16 is experimental and opt-in (`fuse=True`); reference mode is the
  default. Attention remains ROCm SDPA. These initial measurements used
  uncaptured inference; full-model HIP Graph was added in the follow-up below.
- Hardware qualification covers gfx1151 only.

## Full-model HIP Graph follow-up — 2026-10-21

Implemented on branch `feat/xr1-gfx1151-hip-graph`, based on `75421a27`.
The opt-in `load_model(..., use_graph=True)` captures the original full vision,
VLM and DiT forward for both checkpoints, with either reference or native
DiT operations. The default stays uncaptured. See the
[deployment guide](xr1_gfx1151.md#optional-full-model-hip-graph) and
[stable API](stable_api.md) for the contract and reproducible commands.

One frontend owns one graph profile, persistent input/noise buffers, static
metadata and the graph memory pool. Token/layout/shape/step/prefix-length
changes release and rebuild that profile; image, state, prefix and seed changes
refresh inputs. Image features and K/V are computed on every observation.
Instance-local capture helpers restore model methods on success and failure;
they do not modify the checkpoint or process-wide globals. Every capture must
match the uncaptured raw output exactly or raise an error.

Validation covers 78 paired raw/decoded comparisons across reference/native
and 5B/RoboCasa, including recovery after injected capture failures. All had
maximum absolute difference zero. Cases include changed images, states and
seeds, same-length and longer prompts, image resolution, denoise steps and 5B
prefix values/lengths. Both 5B modes also passed all twelve real observations.
Caller CPU/GPU RNG state was preserved. Native `attention="eager"` graph smoke
checks also passed for both checkpoints; performance below uses SDPA.

Same-process, alternating-order public `predict` measurements; BF16 native HIP
DiT, batch one, five steps, eight CPU threads, PyTorch `2.10.0+rocm7.2.0.gitb6ee5fde`,
HIP `7.2.26015`, Transformers `4.57.1`, experimental AOTriton enabled:

| Case | Uncaptured P50 / P95 | Graph P50 / P95 | P50 reduction |
|---|---:|---:|---:|
| RoboCasa, synthetic 3×256×256 | 375.53 / 383.46 ms | 362.69 / 363.91 ms | 3.42% |
| 5B, synthetic 3×256×256 | 393.67 / 395.51 ms | 367.38 / 371.73 ms | 6.68% |
| 5B, twelve real demo observations | 653.05 / 656.65 ms | 635.15 / 638.82 ms | 2.74% |

All modes warm up three calls; synthetic runs measure 50 calls per mode, real
runs measure three cycles of twelve observations (36 per mode). Timers include
preprocessing, upload, full computation, action decoding and download. These
are fresh paired baselines, not comparisons against the earlier ten-call data.

Graph first-call latency on the warmed model was 1.85 / 1.92 / 3.09 seconds for
the three rows. Warm resident PyTorch allocations increased by about
80 / 80 / 85 MiB. Setup overhead makes stable profiles preferable. Numerical
validation and offline observation replay do not establish simulation success.

Evidence is local under `results/xr1_graph/`: paired JSON/NPZ files,
`run_benchmarks.py`, `benchmark_commands.json`, graph/regression logs and the
experiment `REPORT.md`. Earlier evidence remains under `results/xr1/` and
`results/gfx1151_models/`.

The combined graph/affected regression runs contain 213 unique passing tests.
The broader API checks also have 23 failures caused by missing NVIDIA
extensions; the unmodified `75421a27` baseline reproduces the exact same 23
failures. This is not a full-repository clean test result. `pip check`, Python
compilation and `git diff --check` passed.
