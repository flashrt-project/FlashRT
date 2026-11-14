# Apple native BF16 pipeline

The supported model route is a compact variable-length Pi0.5 Metal ICB covering patch,
vision, projector, prefix and all denoising steps. Torch MPS remains the
numerical reference. Legacy vision-only, FP32 MPSMatrix and standalone replay
probes have been removed; historical results remain in local `.cache` reports.

GPU kernel sources live in [`csrc/apple/`](../../csrc/apple/README.md), grouped
by operation (`gemm/`, `attention/`, `conv/`, `kernels/`) and named by precision.
The explicit kernel registry resolves each dispatch to its operation source.
This directory owns the host runtime, C ABI adapter, build targets and tests;
it does not own the Metal kernel implementations.
Pi0.5 plan assembly lives in
[`flash_rt/models/pi05/native_plan_mps.py`](../../flash_rt/models/pi05/native_plan_mps.py).
The hardware package owns model-independent tensors, operations and dispatch
recording. Tensor ownership, BF16 storage, layouts and supported attention
shapes are validated when recording, before allocation of attention scratch
or replay.

## Build and validate

Apple Silicon and macOS 26.2+ are required for BF16 TensorOps. Host code builds
with the installed SDK; MSL compiles at runtime.

The repository-root CMake build selects this backend explicitly:

| Option | Default | Scope and effect |
| --- | --- | --- |
| `FLASHRT_BUILD_APPLE_NATIVE` | `OFF` | Root build selector. `ON` builds only `cpp/apple` and bypasses CUDA configuration. |
| `FLASHRT_TEST_APPLE_TENSOROPS` | `OFF` | Apple test gate. With `BUILD_TESTING=ON`, enables registry, BF16 GEMM and attention GPU qualification; requires macOS 26.2+ and fails on unsupported hosts. |
| `FLASHRT_BUILD_APPLE_INT8_EXPERIMENT` | `OFF` | Adds fourteen experimental INT8/SmoothQuant plan entries; validated on M5 Pro with macOS/SDK 27.2 using MSL 4.0. Default registry and model precision stay BF16. |
| `BUILD_TESTING` | `ON` | CTest switch. Enables runtime, Metal validation and exact dylib export tests. Export checks also run after linking even when this is `OFF`. |

```sh
cmake -S . -B .cache/apple-cmake -DFLASHRT_BUILD_APPLE_NATIVE=ON \
  -DFLASHRT_TEST_APPLE_TENSOROPS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/apple-cmake --parallel 2
ctest --test-dir .cache/apple-cmake --output-on-failure --parallel 1
```

Configure output reports the Apple backend selection and both test gates.
The generic host runtime targets macOS 14+, while the complete BF16 pipeline
requires 26.2+. The native opt-in does not change the default CUDA build.
Alternatively, the SDK-only script below builds without CMake and places the
libraries where the Python adapter finds them by default. For a CMake build,
pass its `cpp/apple/libflashrt_apple_plan.dylib` path as `library=` to
`enable_native_pipeline()`.

```sh
bash scripts/apple/build_native.sh
MTL_DEBUG_LAYER=1 .cache/apple-native/flashrt_apple_kernel_registry_smoke
MTL_DEBUG_LAYER=1 .cache/apple-native/flashrt_apple_smoke
MTL_DEBUG_LAYER=1 .cache/apple-native/flashrt_apple_bf16_smoke
MTL_DEBUG_LAYER=1 .cache/apple-native/flashrt_apple_attention_smoke
MTL_DEBUG_LAYER=1 .venv/bin/python -m pytest tests/test_apple_native_contracts.py tests/test_apple_native_plan.py tests/test_apple_profiling.py -q
PYTHONPATH=. .venv/bin/python examples/apple/benchmark_native_pipeline.py \
  --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer.model \
  --output .cache/apple-native-qualification --prompt-length 10
```

The registry test compiles every entry and rejects unregistered names. The
runtime ownership/bounds test and independent GEMM/attention oracles are
kernel tests, not alternative model pipelines. Keep them when tuning kernels.
`flashrt_apple_gemm_tune` compares tile geometries using paired GPU timings.
Run timing without Metal validation or Instruments; use a separate process
for tracing. No CMake or offline shader compiler is required by the build script.

Both build paths hide C++ implementation symbols and use the same Mach-O
allowlists: [`plan.exports`](plan.exports) (16 private plan C entry points)
and [`signpost.exports`](signpost.exports) (2 tracing entry points). Every link
runs [`check_exports.sh`](../../scripts/apple/check_exports.sh), which fails
on missing or extra exports. These are Apple adapter interfaces, separate
from the frozen deployment runtime ABI.

One plan can hold several programs over the same buffers and kernels.
Operations recorded before the first `frt_apple_plan_program` call form
program 0 (the single program of the Pi0.5 plans, run by
`frt_apple_plan_run_timed`); each `frt_apple_plan_program` call starts program
1, 2, ... and later dispatches and copies append to it. `frt_apple_plan_finish`
seals every non-empty program into its own ICB (an empty additional program is
an error), residency covers all of them, and
`frt_apple_plan_run_program(plan, id, ...)` runs one as a command buffer;
program 0 is unknown when nothing was recorded before the first additional
program. `frt_apple_plan_profile_program` is the per-operation diagnostic of an
additional program. The torch-free `flash_rt.hardware.apple_native.native.NativePlan`
binding records only additional programs; `tests/test_apple_w4a16_affine.py`
covers the API and its error paths.

### Fixed vision + prefix + decoder replay

`NativePi05Plan` records the existing `Pi05TorchPipeline._vision_blocks`,
`_encoder`, and `_decoder` traversal using build-time tensor descriptors and
native operation providers. C++ owns buffers and dispatches; it does not own
another model-layer or denoising loop. The recorded plan contains projector,
18 prefix QKV layers (17 full prefix layers), and all configured decode steps.
GQA uses a shared KV head. Prefix attention uses an online FP32 softmax over
128-key tiles. Small-M decode attention (M <= 16, KV > 128) runs these tiles
in parallel and merges FP32 partial numerators, maxima and denominators with
a stable softmax reduction. The workspace is reused across layers and steps;
only the final output is rounded to BF16.
GEMMs, activations, residuals and KV storage remain BF16. Modulations are
precomputed by the reference setup and uploaded once.

```python
# frontend.set_prompt(...) first. One plan covers 1..max_prompt_len tokens.
pipeline.enable_native_pipeline(frontend._prompt_len)
# Repeated forward_with_fixed_noise / infer calls may change prompt length.
# Exact-shape diagnostics retain the compact specialization:
# pipeline.enable_native_pipeline(length, dynamic_prompt=False)
```

The profile reserves storage for the configured prompt capacity and fixes
views, horizon, steps and weights. Default replay is compact: GEMM, norm,
activation, quantization and attention extents use the actual token count. The
decoder KV suffix follows the valid prefix in storage and RoPE position.
On length changes the host updates bounded uniform expressions. Kernel
objects, bindings, buffers, the recorded ICB and residency stay shared; weights
are neither uploaded nor packed again. Launch grids reserve capacity, while
whole threadgroups beyond the actual extent return before TensorOps/normalization.
This incurs some launch overhead but does not perform capacity-sized GEMMs or
compute padded token activations. Empty and over-capacity inputs fail before replay. There is one plan,
not a cache of full plans for every length.

`pad_prompt=True` explicitly selects the separate padded-compute strategy:
prefix kernels process capacity rows, attention masks exclude padding, and
decoder KV uses a fixed suffix window. It is not the default compact path.
`dynamic_prompt=False` retains exact-length validation and compact row counts
for calibration and historical performance comparisons. SmoothQuant observers
currently require this setting for exact-length calibration.
Weight updates require rebuilding. Native K/V remains in persistent buffers
across denoising steps and is overwritten by every full inference. It does not
populate the separate Torch decode-only cache. Call `disable_native_pipeline()`
to release native buffers and return to the existing providers.

Patch convolution/position addition is recorded in the same ICB: pack the
nonoverlapping NHWC patches, perform BF16 GEMM with bias in the FP32 accumulator,
round to BF16, then add position embeddings. `infer()` stages CPU camera pixels,
prompt embeddings and noise directly into persistent shared Metal buffers.
It returns owned CPU action arrays after one completion wait; no MPS input/output
round trip is needed. `forward_with_fixed_noise()` retains its MPS tensor contract
and therefore still stages MPS inputs and copies the action back to MPS.
Prompt embeddings are written directly into the text slice of the encoder input;
initial noise is written directly into the denoising state. Exact self-copies are
elided during recording, so production plans contain no buffer-copy operations.
There are no host transfers or Python dispatches between vision, prefix or
individual decode steps. `capture_probes=True` when building creates explicit
snapshots using native Metal blit copies between compute spans; benchmark a
separate plan without those snapshots. Compact dynamic prefix probes expose only the current valid rows. Padded-mode
prefix probes include capacity rows, with padding outside the valid prefix.
Production replay remains one ICB range
in one compute encoder. There is no copy shader or copy kernel registry entry.

`examples/apple/pi05.py --native-pipeline` enables this path. The CLI builds
one capacity plan after setting the first prompt, outside model timing.
Use `scripts/apple/validate_pipeline.sh --checkpoint ... --tokenizer ...
--output ...` for kernel validation plus checkpoint probes, direct/ICB parity,
repeat determinism and alternating PyTorch / full-ICB timing.
`--prompt-length 10` on the benchmark selects the first 10 real prompt embeddings
for the requested M=522 profile; it is a synthetic shape qualification, not a
complete natural-language task evaluation. Omit it for the full prompt.
The two `benchmark_*_pipeline.py` programs explicitly use compact plans to keep
their shape and calibration comparisons unchanged. Use the `lengths` mode of
`qualify_int8.py` to compare compact and dynamic plans at the same input length.

For Metal System Trace use `--trace-only --no-parity --iters 3` in a separate
process, with the `Metal System Trace` template and `os_signpost` instrument.
Only stage markers are enabled; no per-operation MPS waits are inserted.
`native.full_pipeline` includes upload, native execution and download. The
single native encoder's duration is not shader occupancy and does not expose
all internal dispatch stalls.

The benchmark's `full-icb` mode uses prepared CPU BF16 inputs and returns CPU
actions. `full-icb-mps` measures the compatibility path; the other modes use
prepared MPS tensor inputs/outputs. `--modes full-icb --schedule serial` isolates
steady native replay from the default alternating-provider workload. Both
include input updates and completion. CPU image normalization/tokenization and
output unnormalization are excluded; use `pi05.py` for frontend latency.
`samples.jsonl` records each sample's UTC start time and native timestamps.
`commit_to_gpu_us` starts immediately before commit; `gpu_to_return_us` ends
after the completion wait. `kernel_us` is Metal's kernelStartTime/endTime
interval, not shader execution. The host and GPU absolute seconds share the
mach absolute host clock. Shared buffer mappings remain private to the plan;
its replay lock covers writes, execution and owned output conversion.

Full native plans request persistent `MTLResidencySet` residency for their
buffers and ICB at build time, and end/remove it when the plan is closed.
This prevents repeated driver memory wiring between replays. It holds the
plan's allocations resident for its lifetime; close unused plans promptly.
The diagnostic `--no-residency` flag reproduces the old behavior for A/B tests.
If exported as `driver.xml`, the analyzer correlates Metal driver memory events
with each inference and renders Wire Memory intervals in orange. Event-duration
unions avoid double counting; these events are not GPU shader work.

## Kernel diagnostics and tuning

`benchmark_native_pipeline.py --profile-kernels` profiles every dispatch in
model order after normal replay timing. Diagnostics use isolated synchronous
submissions, include per-command GPU overhead, and must not be summed as model
latency. The profiler verifies that the final output exactly matches ICB replay.
Reports group by stage, kernel and shape; this is the priority list for tuning.
The model adapter's `profile(images, prompt, noise)` restores all inputs before
each diagnostic replay and returns an owned output with dispatch timings. Use it
instead of calling `native.plan.profile()` directly: execution overwrites the
prompt/noise storage. Probe snapshots use separate buffers and native blit copies
so later layers cannot overwrite the captured values.

`--compare-gqa` performs a warmed unsplit/split/split/unsplit comparison of the
complete model, rebuilding between blocks and keeping only one native plan
resident. Each block uses `--iters` samples. `--no-split-gqa` selects the old
attention implementation for diagnosis; it is still a complete model replay.

`--compare-encoder-ffn` uses the same ABBA protocol for separate versus
concatenated encoder gate/up GEMMs. The default is already concatenated:
M=522, K=2048, N=32768. `--split-encoder-ffn` instead records two N=16384
GEMMs into column slices of the same intermediate buffer. The strided output
kernel avoids extra copies, both paths use the same GELU/multiply kernel,
and both preserve the BF16 rounding after GEMM. Weights are packed once at
plan construction; decode kernels and residency remain unchanged.

For GEMM tile sweeps run `flashrt_apple_gemm_tune prefix`, `decode`, or `vision`.
The baseline uses the production 64x64 / 16x64 tile and two SIMD groups. Each
candidate passes exact small/tail oracles and uses paired ICB GPU timings.
Qualify a chosen candidate on checkpoint probes and full replay before adoption.
`analyze_gemm_sol.py --measurements REPORT.json` reads full-pipeline kernel
diagnostics, including the merged gate/up GEMMs, to compare against an explicitly
assumed compute/bandwidth roofline.

### Experimental INT8 Pi0.5 lowering

INT8 is explicitly opt-in; the default precision remains BF16.

The existing platform code provides these starting points, not a blanket
accuracy qualification for another backend:

| Platform path | Quantized Pi0.5 operations | Important limits |
| --- | --- | --- |
| [RTX / Orin](../../flash_rt/frontends/torch/pi05_rtx.py) | INT8 prefix and decoder QKV, attention output, FFN gate/up/down; optional SigLIP QKV/output/up/down | Per-output-channel weights, dynamic row activations. `FVK_PI05_RTX_FORCE_INT8` selects prefix+decoder; encoder-only and vision have separate flags. Static per-tensor vision is disabled after precision regression; frozen encoder scales are also experimental. INT8 projector is not included in these weight packers. |
| [Ascend](../../flash_rt/npu/models/pi05/quantization.py) | INT8 prefix Q/K/V/O and gate/up/down; decoder packed QKV/O/gate-up/down | Real-observation calibration: prefix image-row/language-group activation scales and decoder per-layer/per-step scales. Vision stays on its existing floating-point path. |
| [AMD CDNA](../../flash_rt/amd/frontends/torch/pi05.py) | FP8 vision, projector, prefix and decoder large GEMMs | This is FP8, not an INT8 implementation. The separate [RDNA frontend](../../flash_rt/amd/frontends/torch/pi05_rdna.py) supports BF16 only. |
| [Thor](../../flash_rt/frontends/torch/_pi05_thor_spec.py) | FP8 SigLIP, prefix and decoder weight specifications | Separate FP4 routes exist; these do not establish an Apple INT8 precision contract. |

Apple `INT8Plan` uses symmetric signed W8A8: weights have one FP32 scale per
output channel, activations one dynamic FP32 scale per token, INT32 dot
products, FP32 descale, and BF16 output. Packing uses nearest-even rounding
and [-127,127]. Zero rows/channels use a minimum scale of 1e-12. Activations
and scales are overwritten on the GPU on every replay. Weights, scales and
scratch are allocated at recording time. Plain INT8 needs no calibration;
optional SmoothQuant collects channel maxima during setup. There is no
persistent calibration cache, per-operation host readback or replay allocation.
Weight identity includes its name, storage offset, shape and strides; ambiguous
storage identities fail. Rebuild the plan after changing weights or shapes.

`NativePi05Plan(source, prompt_len, library=..., int8_options={})` explicitly
selects the experiment, **decoder only by default**. It reuses the existing
model traversal; this is not a new public `load_model` precision route.
The selected decoder linears include 18 layers of QKV, O, merged gate/up and
down for every denoising step. Small action projections remain BF16. Patch
embedding, attention, normalization, nonlinearities, residual/state and KV
storage retain their existing floating-point semantics.

`int8_options={"stages": ("prefix", "decode")}` or all three stages also
exposes diagnostic coverage. Without SmoothQuant, these broader modes **fail action accuracy on
the checked LIBERO observations** and are not qualified deployment settings.
The prefix selection includes the image projector and the final cache-only
QKV layer; all-stage selection additionally covers vision block linears.
Contiguous linear outputs with K,N >= 256 are eligible; other operations use
the unchanged BF16 lowering. Bias preserves GEMM -> BF16 -> bias -> BF16.

```sh
bash scripts/apple/build_native.sh .cache/apple-int8 --int8-experiment
MTL_DEBUG_LAYER=1 .cache/apple-int8/flashrt_apple_kernel_registry_smoke
FLASHRT_APPLE_INT8_LIBRARY=.cache/apple-int8/libflashrt_apple_plan.dylib \
  MTL_DEBUG_LAYER=1 PYTHONPATH=. python -m pytest tests/test_apple_native_int8.py -q
PYTHONPATH=. python examples/apple/benchmark_int8_pipeline.py \
  --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer.model \
  --library .cache/apple-int8/libflashrt_apple_plan.dylib \
  --variants decode prefix-decode all --output .cache/apple-int8-qualification
```

CMake uses `-DFLASHRT_BUILD_APPLE_INT8_EXPERIMENT=ON` alongside the Apple
backend selector above. The default build registers 27 kernels; the opted-in
build registers 41. Requesting INT8 from a default library fails on its unknown
kernel entry.

The benchmark uses four fixture observations (indices 4–7), natural prompt
lengths, two views, horizon 10 and ten denoising steps. It checks actions and
stage probes against the unchanged native BF16 plan, direct/ICB agreement,
repeated inputs, and exact agreement between probe-enabled and timing plans.
The action gate is cosine >= 0.999 and NRMSE <= 0.03 for the seven robot axes;
all 32 channels and intermediate probes are also reported. This is numerical
qualification, not LIBERO task-success validation.

Timing is CPU prepared inputs -> complete native vision/prefix/decoder -> CPU
raw actions. It excludes preprocessing, tokenization, unnormalization and
setup. One resident model plan runs at a time, with forward/reverse variant
order, 10 warmups and 20 timed calls per block (40 calls per variant). The JSON
records checkpoint, tokenizer, fixture and library digests, shape, hardware,
coverage and accuracy failures alongside latency. Reports are diagnostics,
not reloadable calibration artifacts. The script defaults to decoder only;
without `--smoothquant`, broader `--variants` selections are failed-accuracy
diagnostics. The script now stops before timing if any accuracy gate fails.
The SmoothQuant qualification is described below.

Plain INT8 measured 2026-11-02 on Apple M5 Pro (20 GPU cores, 24 GiB), macOS/SDK 27.2,
Apple clang 21.0.0, Pi0.5 LIBERO checkpoint. Natural validation prompt lengths
were 150/149/148/148; timing used 150 tokens (prefix M=662).

| Precision scope | Wall p50 / p95 (ms) | GPU p50 (ms) | Minimum robot cosine | Maximum robot NRMSE | Accuracy gate |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 | 164.706 / 165.798 | 164.245 | 1.0 | 0 | Reference |
| Decoder INT8 | 155.094 / 155.589 | 154.639 | 0.999971 | 0.8024% | Pass |
| Prefix + decoder INT8 | 128.140 / 128.334 | 127.702 | 0.847926 | 54.03% | **Fail; diagnostic only** |
| All eligible INT8 | 124.396 / 124.806 | 123.951 | 0.838405 | 55.50% | **Fail; diagnostic only** |

The qualified decoder subset reduces latency by 5.84% (1.062x throughput).
It replaces 720 linear calls across ten steps, retains 20 small action
projections in BF16, and adds 720 quantization dispatches. Plan weight storage
falls from 5,209,139,744 to 4,898,634,272 bytes; these are not process-memory
measurements. Maximum robot action absolute error is 0.01171875. Prefix-only
expansion leaves vision probes exact but accumulates large errors in prefix
layers and their final K/V. Subsequent per-operation CPU checks and ablations
identified FFN gate/up activation quantization as the dominant error source:
large channel outliers cause the row INT8 scale to erase small values. The
faster unsmoothed broader timings are not accepted optimization results.

Validation: 15 INT8 tests (byte oracles, NumPy INT64 dots up to K=16384,
nonfinite/zero rows, ties, bias rounding, changed-input direct/ICB replay,
unchanged unselected BF16 stages and stage guards), 109 existing Python
regressions, seven CTests, exact export checks, and SDK builds with the flag
off/on. Both pack geometries produce the same quantization result. Wider
datasets, task success, and other Apple devices remain unqualified.

### SmoothQuant for prefix INT8

The opt-in INT8 provider now supports the channel transform from
[SmoothQuant (Xiao et al., ICML 2023)](https://arxiv.org/abs/2211.10438).
This implementation uses the paper's equation; no upstream source was copied.
For each linear input channel `j`, setup collects `a[j] = max(abs(X[:,j]))`
from the unchanged BF16 traversal and computes `w[j] = max(abs(W[j,:]))` from
the existing, already gamma-folded weight. It fits
`S[j] = a[j]^alpha / w[j]^(1-alpha)` in FP32, with 1e-5 floors for `a`, `w`
and `S`, then packs `W' = S * W` once. GPU activation packing computes
`X' = X / S` before fitting the usual dynamic row INT8 scale. Thus the
unquantized transform preserves `X @ W`; the shared checkpoint conversion,
BF16 normalization and other consumers do not change. Bias keeps its original
BF16 rounding boundary. The channel multiply is fused into packing: no new
production dispatch or BF16 intermediate is needed.

`INT8Plan(..., stages=("prefix",), observe=True)` records channel observers
alongside BF16 linears. `activation_maxima()` reads the setup result; repeated
calls/replays accumulate maxima. The host selects observations, merges maxima
across natural prompt lengths, and supplies that mapping as `smooth_amax` to a
new plan. The source checkpoint and weight layout must match the observed
plan. Missing sites, invalid channel shapes, nonfinite observations/weights
and alpha outside [0,1] fail explicitly. The mapping is copied at construction;
Changing calibration, alpha, weights or model dimensions other than the valid
prompt length requires rebuilding. No
persistent or reloadable calibration artifact is introduced.

For `NativePi05Plan`, pass
`int8_options={"stages": ("prefix", "decode"), "smooth_amax": maxima,
"smooth_alpha": 0.5}`. `smooth_stages` defaults to prefix: all 70 executed
prefix linear sites (including the projector) are observed/smoothed; decoder INT8 and
optional vision INT8 keep their dynamic row/channel scheme. Production plans
contain no observers, CPU reductions or per-layer host readbacks.

```sh
PYTHONPATH=. python examples/apple/benchmark_int8_pipeline.py \
  --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer.model \
  --library .cache/apple-int8/libflashrt_apple_plan.dylib \
  --smoothquant --smooth-alpha 0.5 --calibration-samples 4 \
  --sample-offset 4 --validation-samples 4 \
  --variants decode prefix-decode all --output .cache/apple-int8-smoothquant
FLASHRT_APPLE_INT8_LIBRARY=.cache/apple-int8/libflashrt_apple_plan.dylib \
  MTL_DEBUG_LAYER=1 PYTHONPATH=. python -m pytest \
  tests/test_apple_native_int8.py tests/test_apple_native_smoothquant.py -q
```

Calibration uses fixture observations 0–3, validation 4–7. Alpha is fixed at
0.5, with no validation-set tuning. Channel aggregation is a maximum over
BF16 tokens/calls/observations; runtime row scales remain dynamic. Reports
include calibration indices, alpha, stage, site/channel counts and hashes of
channel maxima, alongside existing checkpoint/tokenizer/input/library identity.
The default BF16 route and plain decoder INT8 remain unchanged.

On the same M5 Pro, SDK and Pi0.5 setup described above, all four validation
observations pass the action gate with prefix smoothing:

| INT8 scope (SmoothQuant on prefix only) | Minimum robot cosine | Maximum robot NRMSE | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Decoder | 0.999971 | 0.8024% | 0.01171875 |
| Prefix + decoder | 0.999806 | 2.0335% | 0.02734375 |
| All eligible linears | 0.999735 | 2.3148% | 0.03515625 |

Direct/ICB and repeat outputs agree exactly; probe-free timed plans reproduce
qualified outputs. The plain prefix+decoder baseline's minimum cosine was
0.847926. These four observations establish numerical coverage only, not
LIBERO task success or general calibration coverage.

The first sequential timing run encountered concurrent desktop GPU load:
BF16 block medians drifted from 173.884 to 191.026 ms. Pooled wall p50 values
were BF16 182.352 ms, prefix+decoder 137.918 ms and all eligible 135.256 ms.
These loaded-session measurements are diagnostic and must not be compared
directly with the earlier idle-session absolute latencies or presented as
stable speedup guarantees.

Validation adds ten SmoothQuant tests to the existing 15 INT8 tests, including
an independent channel-transform identity, unchanged observer-mode BF16 output,
per-channel maxima across changed inputs, NumPy INT64 dot oracles for both pack
geometries, bias rounding, missing scales, nonfinite observations and alpha
bounds. The 109 existing Python regressions and seven CTests also pass. Default
and opted-in SDK builds registered 24 and 33 kernels respectively before the
additional tile and fusion entries described below.

### INT8 tile selection and activation fusion

Inside the opt-in INT8 provider, `tune_tiles=True` selects these measured
geometries for 512 <= M <= 1024 (all sizes use `M,N,K` linear notation):

| K,N | Tile M,N / SIMD groups |
| --- | --- |
| 2048,2560 | 64,64 / 2 |
| 2048,32768 or 16384,2048 | 64,128 / 4 |

Other shapes retain 32,64 / 2, or 16,128 / 4 for M <= 16. Selection depends
only on geometry, not model names, weight identities or stage labels.
`fuse_gelu=True` combines an immediately adjacent merged GELU multiply with
SmoothQuant activation packing when M > 16 and K >= 4096. It retains the
materialized BF16 hidden tensor, its original rounding boundary and subsequent
consumers. The fused producer computes row maxima while writing BF16 values,
then reads them once for packing. Setup-only lookahead flushes the unfused
producer at an intervening dispatch/copy or finish, or before a consumer in a
different stage. Observer plans remain BF16. No model traversal is duplicated.
The fused kernel uses 512 threads per row. Pi0.5 uses this fusion at 17 prefix
FFN down inputs, removing 17 dispatches.

Set `tune_tiles=False, fuse_gelu=False` in `int8_options` to reproduce the
original INT8 lowering. The benchmark exposes `all-original` (both disabled),
`all-tiled` (tiles only) and `all` (both enabled). Including `all-original`
requires optimized actions to be bit-identical on every held-out observation,
in addition to the BF16 accuracy gates. `--paired BASELINE CANDIDATE` keeps two
resident plans, releases setup model weights, and alternates AB/BA for 40 timed
calls per plan after ten warmups. This avoids slow drift between sequential
blocks; its residency differs from the default one-plan timing mode.

```sh
PYTHONPATH=. python examples/apple/benchmark_int8_pipeline.py \
  --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer.model \
  --library .cache/apple-int8/libflashrt_apple_plan.dylib \
  --smoothquant --variants all-original all-tiled all \
  --paired all-original all --output .cache/apple-int8-optimized
# Use --variants all --paired bf16 all for the BF16 comparison.
FLASHRT_APPLE_INT8_LIBRARY=.cache/apple-int8/libflashrt_apple_plan.dylib \
  MTL_DEBUG_LAYER=1 PYTHONPATH=. python -m pytest \
  tests/test_apple_native_int8.py tests/test_apple_native_smoothquant.py \
  tests/test_apple_native_int8_fusion.py -q
```

Measured 2026-11-02 on the same M5 Pro (20 GPU cores, 24 GiB), macOS/SDK 27.2,
clang 21.0.0 and Pi0.5 LIBERO checkpoint. Calibration used observations 0–3;
all three configurations produced bit-identical full actions on observations
4–7, with natural prompt lengths 150/149/148/148. The BF16 accuracy metrics
remain the SmoothQuant values above. Timing used observation 4, two views,
prefix M=662, horizon 10 and ten denoising steps, ten warmups and 40 timed
calls per plan. Separate AB/BA runs gave:

| Paired comparison | Baseline wall p50 / p95 (ms) | Candidate wall p50 / p95 (ms) | Median paired ratio |
| --- | ---: | ---: | ---: |
| Original INT8 -> tiles + fusion | 132.467 / 135.530 | 127.293 / 129.629 | 1.044x |
| Tiles only -> tiles + fusion | 129.699 / 133.306 | 128.101 / 131.968 | 1.012x |
| BF16 -> tiles + fusion INT8 | 180.040 / 185.193 | 124.717 / 126.545 | 1.449x |

The first comparison reduces wall p50 by 3.9% (5.174 ms). GPU p50 moves from
131.983 to 126.735 ms. Dispatches fall from 3563 to 3546; plan weight storage
stays at 2,615,077,344 bytes. Compare numbers within each pair: desktop GPU
load changes absolute timings across runs. `--validation-samples 1` was used
for the final original/candidate and BF16 timing repetitions after four-observation
qualification. These timings retain the prepared-input/raw-action scope
described above, excluding preprocessing and setup.

Validation: 135 Apple/INT8 Python tests with Metal API validation enabled,
33 shared Pi0.5 pipeline tests, seven CTests, and both SDK builds/export checks
(24 default and 38 opted-in registry entries). The new tests compare hidden
BF16 values, INT8 bytes, scales and outputs against the unfused path, including
nonfinite inputs, padded K, direct/ICB replay, later hidden-tensor consumers,
lookahead flush boundaries, tiled tails and bias rounding. The factored BF16
GELU helper also matched the previous compiled library's raw output bits across
all 65,536 BF16 input encodings. The shared PyTorch MPS tests run without Metal
API validation: enabling it independently reproduces a PyTorch
`mul_strided_bfloat_bfloat` offset-alignment assertion in the existing RoPE
path, which does not use this native INT8 implementation.

### Closed-loop and host acceptance

`examples/apple/qualify_int8.py` exercises the candidate through the public
`predict()` call with a setup host that installs `NativePi05Plan`. The host
normalizes each state, updates its prompt, and reuses the bounded native
plan when the token count changes. Reported host latency includes all of those
steps, pixel normalization, noise creation, replay and action unnormalization.
LIBERO camera rendering and 256-to-224 image resizing precede that timer.

The initial 2026-11-04 pilot (before bounded prompt replay) used the same M5 Pro and checkpoint, LIBERO commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01`, robosuite 1.4.1, MuJoCo 3.2.7 with
macOS CGL rendering, PyTorch 2.14.1, NumPy 1.26.4 and Pillow 13.0.0. Camera
rotation, PIL bilinear resizing and the position/axis-angle/gripper state
follow the OpenPI LIBERO evaluation. The host reuses robosuite's quaternion
conversion. Simulator and model noise
seeds match within each pair, and the initial images and state must have
identical hashes. Variants alternate order across initial states.
[OpenPI evaluation source](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)

Calibration remains fixture observations 0–3 with alpha 0.5. All 70 channel
maximum hashes match the preceding numerical qualification. The paired pilot
uses `libero_spatial` tasks 6, 8 and 9, initial-state indices 17 and 18, ten
settling steps, ten denoising steps, horizon 10, and five actions per replan.
The 20 Hz simulator control rate gives a 250 ms replanning budget. Episodes
end at task success or the suite's 220-step limit.

| Task / initial state | BF16 | INT8 |
| --- | --- | --- |
| 6 / 17 | Success, 116 steps | Success, 113 steps |
| 6 / 18 | Success, 121 steps | Success, 119 steps |
| 8 / 17 | Success, 109 steps | Success, 110 steps |
| 8 / 18 | Success, 109 steps | Success, 107 steps |
| 9 / 17 | Limit reached, 220 steps | Limit reached, 220 steps |
| 9 / 18 | Success, 145 steps | Success, 144 steps |

Both variants succeed on 5/6 initial states, with the same failing pair.
Six pairs provide pilot coverage. Wider task-success qualification remains open.

| Host metric across the six episodes | BF16 | INT8 |
| --- | ---: | ---: |
| Calls | 166 | 164 |
| Ordinary-call p50 / p95 / p99 (ms) | 194.122 / 195.767 / 198.384 | 147.229 / 148.180 / 148.757 |
| Plan builds, including six episode starts | 29 | 37 |
| Median plan-build time (ms) | 692.917 | 4815.937 |
| Calls exceeding 250 ms | 29 | 37 |

Every deadline miss in this pilot coincides with a plan build. After excluding
the six episode starts, there are 23 BF16 and 31 INT8 length-change rebuilds.
This exposed the need for bounded-cost prompt-length changes within the control
period in both precisions. The native producer now handles these changes with
the bounded replay described above.

Install LIBERO and its simulator dependencies in a separate Python environment,
and configure its asset, BDDL and initial-state paths through
`LIBERO_CONFIG_PATH`. The example additionally requires `psutil` for soak RSS
measurements. Run the two modes sequentially to isolate GPU and memory load.

```sh
MUJOCO_GL=cgl PYTHONPATH=. python examples/apple/qualify_int8.py rollout \
  --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer.model \
  --library .cache/apple-int8/libflashrt_apple_plan.dylib \
  --tasks 6 8 9 --trials 17 18 --output .cache/apple-int8-rollouts
PYTHONPATH=. python examples/apple/qualify_int8.py soak \
  --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer.model \
  --library .cache/apple-int8/libflashrt_apple_plan.dylib \
  --calls 1200 --output .cache/apple-int8-soak
```

Current soak mode cycles held-out observations 4–7, including their natural
length changes, at a 250 ms cadence. It checks
periodic fixed-input/fixed-noise outputs for exact repeatability, and samples
process RSS and Torch MPS driver allocation. It then runs 24 calls across
observations 4–7 to expose length-change costs. Report files include individual
call latency, rebuild time, shape, calibration hashes and artifact identity.

The initial 2026-11-04 soak completed 1200 measured calls over 307.334 seconds after
ten warmups and two reference calls. All calls used natural prompt length 148
(prefix M=660). Wall p50/p95/p99/max were 165.090/170.703/172.606/185.287 ms,
with zero 250 ms deadline misses, zero nonfinite outputs and twelve exact
fixed-input/fixed-noise replay checks. Process RSS ranged from 2851.95 to
4354.30 MiB and ended at 2857.30 MiB. Torch's reported Metal driver allocation
remained 8166.02 MiB throughout the sampled interval.

The subsequent 24-call length-switch test used prompt lengths 150/149/148/148.
It triggered 18 plan builds and 18 deadline misses. Full host p50/p95/max were
4959.031/5055.435/5081.859 ms. Median plan-build time was 4813.350 ms. These
measurements identify the pre-fix cost of rebuilding exact-length plans.

### Padded-mode comparison

The earlier padded-mode candidate kept one capacity-compute plan for BF16 and
INT8. This strategy remains available only as `pad_prompt=True`. On the
same M5 Pro, `qualify_int8.py lengths` compares it with freshly built compact
plans at natural lengths 150/149/148/148 and synthetic lengths
1/10/127/128/129/199/200, walking the cases in both directions. Each case uses
three warmups and ten timed replays, with exact repeated-output checks. Timings
cover prepared CPU inputs through raw CPU actions; capacity is 200 text tokens
(prefix M=712), two views, horizon 10 and ten denoising steps.

| Precision | Minimum cosine vs compact | Maximum NRMSE | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| BF16 | 0.999995911 | 0.002878626 | 0.00390625 |
| INT8 + SmoothQuant | 0.999976287 | 0.006963251 | 0.0078125 |

These 44 padded-mode comparisons pass the cosine >= 0.999 / NRMSE <= 0.03 action gate.
All 70 calibration hashes match the earlier compact calibration. Parameter
updates leave the native handle, dispatch count, weight bytes and uniform-block
count unchanged. Padding changes attention reduction grouping, so comparison
with a compact plan is numerical; repeated execution of the same plan is exact.

| Prepared-input case | BF16 compact / bounded (ms) | INT8 compact / bounded (ms) |
| --- | ---: | ---: |
| Natural lengths 148–150 | 164.635 / 172.172 | 118.605 / 124.709 |
| Synthetic length 10 (compact M=522) | 141.439 / 170.825 | 103.838 / 123.651 |

These are medians of case p50 values from the same run. Fixed-capacity prefix
GEMMs process padding, adding about 7.5 ms BF16 / 6.1 ms INT8 for natural
prompts. The short-prompt case pays more padding overhead. The historical
approximately 140 ms BF16 result remains the compact M=522 measurement;
it is not the latency of a capacity-200 deployment plan.

```sh
PYTHONPATH=. python examples/apple/qualify_int8.py lengths \
  --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer.model \
  --library .cache/apple-int8/libflashrt_apple_plan.dylib \
  --output .cache/apple-int8-lengths
```

Kernel regressions cover entirely masked KV tiles, stale padding, 128-key
boundaries, separate mutable and constant uniforms, RoPE updates, invalid
lengths and direct/ICB parity. That candidate passed 141 native tests with Metal
validation enabled; the shared traversal suite passes 33 tests. All seven
INT8-enabled CTests pass, and the default SDK build compiles all 26 registry
entries. The INT8 build contains 40 entries; native exports are unchanged.

A repeat of the same six LIBERO initial states with bounded replay produced:

| Task / initial state | BF16 | INT8 |
| --- | --- | --- |
| 6 / 17 | Success, 112 steps | Success, 113 steps |
| 6 / 18 | Success, 122 steps | Success, 118 steps |
| 8 / 17 | Success, 110 steps | Success, 110 steps |
| 8 / 18 | Success, 108 steps | Success, 112 steps |
| 9 / 17 | Limit reached, 220 steps | Success, 155 steps |
| 9 / 18 | Success, 147 steps | Success, 161 steps |

| Bounded host replay metric | BF16 | INT8 |
| --- | ---: | ---: |
| Total calls / initial plan builds | 166 / 6 | 156 / 6 |
| Hot calls (excluding the first call of each episode) | 160 | 150 |
| In-episode prompt-length changes | 24 | 26 |
| Hot plan builds / calls exceeding 250 ms | 0 / 0 | 0 / 0 |
| Hot p50 / p95 / p99 / max (ms) | 200.112 / 201.469 / 201.827 / 202.626 | 152.081 / 152.835 / 153.300 / 153.319 |

These calls use natural prompt lengths 150–154. Initial plan creation remains
setup work and exceeds the control-period budget; it must finish before the
control loop starts. All 50 later length changes reuse the plan. Success on
5/6 BF16 and 6/6 INT8 initial states is pilot evidence, not a statistically
established task-success improvement or broad production qualification.

The padded-mode INT8 soak completed 1200 measured calls in 305.300 seconds at the
same requested 250 ms cadence. Full host p50/p95/p99/max were
171.133/177.240/178.462/182.702 ms, with zero deadline misses and zero rebuilds.
Twelve periodic fixed-input/noise checks passed exactly. Process RSS started at
4342.91 MiB, ended at 2794.20 MiB and ranged from 2765.23 to 4352.73 MiB;
Torch's reported Metal driver allocation stayed at 8166.92 MiB.

The subsequent 24 natural-length switches (150/149/148/148 tokens) also made
zero builds and missed no deadlines. Full host p50/p95/p99/max were
152.458/155.488/162.167/164.108 ms, compared with the earlier rebuild-dominated
p95 of 5055.435 ms. Each of the four inputs uses a fixed noise seed during this
new mixed-length phase; all 20 later outputs match their first occurrence
exactly. The earlier mixed-length run used changing seeds, so this is a host
latency/regeneration comparison, not a paired action-quality measurement.

### Compact variable-length replay

Default native replay now computes the actual prefix length. The padded mode
above is an explicit alternative (`pad_prompt=True`); it does not fix compact
shape handling. The original compact implementation rebuilt the entire plan
when a state token changed the prompt length, repeating weight uploads and INT8
packing. Bounded integer descriptors now carry the actual row count through the
shared traversal into every projection, norm, activation, quantizer and GQA
uniform. An offset-enabled RoPE entry writes decoder KV immediately after the
valid prefix. No padded token is computed or attended to in this mode.

The existing ICB and its launch bounds remain fixed. Surplus whole threadgroups
return before matrix multiplication/normalization/attention; normal partial-tile
handling is unchanged. This avoids shader compilation, weight repacking, GPU
allocation and ICB re-recording during length changes. It still has measurable
launch overhead relative to a separately recorded exact-length plan, which is
reported below. Memory capacity and computation extent are separate quantities.

On the same M5 Pro/checkpoint configuration, all 44 forward/reverse comparisons
across natural lengths 150/149/148/148 and synthetic lengths
1/10/127/128/129/199/200 are **bit-exact** against freshly constructed compact
plans, for both BF16 and INT8. All 70 calibration hashes remain unchanged.
The qualifier records resolved prefix GEMM/RMS rows: a ten-token input uses
M=522, while 150 tokens use M=662. Kernel tests additionally poison the unused
rows and verify they remain untouched, check KV placement and stale-key
exclusion, and exercise changing-length probe copies and profiling.

| Prepared-input case | BF16 exact / variable compact (ms) | INT8 exact / variable compact (ms) |
| --- | ---: | ---: |
| Ten text tokens, M=522 | 141.535 / 145.807 | 104.503 / 108.049 |
| 150 text tokens, M=662 | 164.414 / 167.736 | 119.431 / 120.257 |

Values are medians of case p50 values, three warmups and ten timed replays per
case/direction, including CPU input updates and raw CPU action output. These are
not zero-overhead shape changes: the retained launch bounds cost about 4.3 ms
BF16 / 3.5 ms INT8 in the ten-token case. Other shapes also retain launch
capacity overhead; the `lengths` report includes every case rather than only
these two benchmark points. The native library SHA-256 for this qualification
is `49a25b138e0e2701c0a3830297615f6d7a9c48fea1b87dc3c6493c95672abac5`.

The repeated six-pair LIBERO pilot exactly reproduces the original compact
pilot's success/failure **and step count in every episode**: both precisions
succeed on 5/6 initial states. Across 160 hot BF16 calls and 158 hot INT8 calls,
23 BF16 and 31 INT8 prompt-length changes cause no plan builds and no 250 ms
deadline misses. Initial setup remains outside the control loop.

| Compact host replay metric | BF16 | INT8 |
| --- | ---: | ---: |
| Hot p50 / p95 / p99 / max (ms) | 194.432 / 195.979 / 196.755 / 198.470 | 147.764 / 149.106 / 150.459 / 177.469 |
| Changed-length uniform update p50 / p95 / max (µs) | 60.792 / 68.396 / 69.625 | 61.709 / 66.708 / 68.167 |

Uniform-update timing covers the length-dependent host parameter writes only;
full host latency includes preprocessing, inference and action unnormalization.
The model still has one compiled plan and one set of packed weights.

The current native suite passes 144 tests under Metal validation and all seven
INT8 CTests pass. Both SDK/default and INT8 builds pass export checks. The
registry contains 27 default entries and 41 with INT8 enabled. No C ABI or
runtime-contract changes are needed for compact variable-length execution.

The final 1200-call INT8 soak cycles observations 4–7 throughout the run,
producing 899 transitions between adjacent measured prompt lengths. It records
zero full plan builds and zero 250 ms misses. Full host p50/p95/p99/max are
165.657/172.119/173.805/184.283 ms. All 24 periodic fixed-input/noise checks
are exact after intervening lengths. Reported Metal driver allocation is
constant. The subsequent 24-call switching phase has p50/p95/p99/max
147.913/151.312/156.878/158.450 ms, zero builds/misses and 20 exact cross-length
replay checks. The `soak` command above now runs this variable-length protocol.
