# Apple native BF16 pipeline

The supported model route is a fixed-shape Pi0.5 Metal ICB covering patch,
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
allowlists: [`plan.exports`](plan.exports) (13 private plan C entry points)
and [`signpost.exports`](signpost.exports) (2 tracing entry points). Every link
runs [`check_exports.sh`](../../scripts/apple/check_exports.sh), which fails
on missing or extra exports. These are Apple adapter interfaces, separate
from the frozen deployment runtime ABI.

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
# frontend.set_prompt(...) first: the exact token count fixes prefix M.
pipeline.enable_native_pipeline(frontend._prompt_len)
# Repeated forward_with_fixed_noise / infer calls update contents only.
# When the prompt length changes, explicitly rebuild the shape specialization:
pipeline.disable_native_pipeline()
pipeline.enable_native_pipeline(new_prompt_len)
```

The profile fixes views, prompt length, horizon, steps, and current weights.
Changing input content does not invalidate it. Mismatched input lengths fail
before replay; there is no silent padding or reuse of an incompatible plan.
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
separate plan without those snapshots. Production replay remains one ICB range
in one compute encoder. There is no copy shader or copy kernel registry entry.

`examples/apple/pi05.py --native-pipeline` enables this path. The CLI builds
or rebuilds the specialization after setting the prompt, outside model timing.
Use `scripts/apple/validate_pipeline.sh --checkpoint ... --tokenizer ...
--output ...` for kernel validation plus checkpoint probes, direct/ICB parity,
repeat determinism and alternating PyTorch / full-ICB timing.
`--prompt-length 10` on the benchmark selects the first 10 real prompt embeddings
for the requested M=522 profile; it is a synthetic shape qualification, not a
complete natural-language task evaluation. Omit it for the full prompt.

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
