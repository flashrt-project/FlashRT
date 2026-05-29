# Contributing to FlashRT

FlashRT is a realtime inference engine. Contributions are welcome, but
changes need to preserve the repository's main contract: predictable
latency, explicit hardware routing, stable public APIs, and clear failure
modes.

This guide summarizes the development rules that are spread across the
README, install docs, model-integration docs, and regression tests.

## Start Here

Before opening a PR:

1. Read the relevant docs.
   - Setup and build: [`docs/INSTALL.md`](docs/INSTALL.md)
   - Public API surface: [`docs/stable_api.md`](docs/stable_api.md)
   - New model integration: [`docs/adding_new_model.md`](docs/adding_new_model.md)
   - Kernel catalog: [`docs/kernel_catalog.md`](docs/kernel_catalog.md)
   - Calibration contract: [`docs/calibration.md`](docs/calibration.md)
2. Build the extension modules locally.
3. Run the smallest test set that covers your change.
4. Include the exact GPU, CUDA, command lines, and latency/precision numbers
   in the PR description when the change touches runtime behavior.

## Development Setup

Use an editable install. CMake writes the compiled extension modules into
the source tree under `flash_rt/`, so non-editable installs commonly import
a stale copy.

```bash
git clone https://github.com/LiangSu8899/FlashRT.git
cd FlashRT
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass

pip install -e ".[torch]"        # or ".[jax]" / ".[all]"
cmake -B build -S .
cmake --build build -j$(nproc)
```

For single-GPU developer builds, `FA2_ARCH_NATIVE_ONLY=ON` can cut build
time substantially:

```bash
cmake -B build -S . -DFA2_ARCH_NATIVE_ONLY=ON
cmake --build build -j$(nproc)
```

## Repository Rules

### Public API

Public API changes must be reflected in [`docs/stable_api.md`](docs/stable_api.md).
Do not remove or change documented signatures without a major-version plan.

The stable user entry point is:

```python
import flash_rt
model = flash_rt.load_model(...)
actions = model.predict(...)
```

Internal frontend, pipeline, and kernel helper signatures may change, but
PRs should keep call sites consistent and update tests when behavior changes.

### Model And Hardware Routing

New model code must follow the split-file routing contract from
[`docs/adding_new_model.md`](docs/adding_new_model.md):

- One compute path per `(model, hardware)` file:
  `flash_rt/models/<model>/pipeline_<hw>.py`
- One frontend per `(model, framework, hardware)` file:
  `flash_rt/frontends/<framework>/<model>_<hw>.py`
- One `_PIPELINE_MAP` entry per supported `(config, framework, arch)` tuple.
- Do not add new runtime hardware forks such as `if arch == ...` inside a
  shared frontend or pipeline.

`pi0fast` is a historical exception and should not be copied for new models.

### Hardware Helpers

Keep `flash_rt/hardware/<hw>/shared_primitives.py` model-agnostic. Shared
helpers are appropriate there only when they can be reused across models
without model-specific tensor names, dimensions, or control flow.

Model-specific decoder, DiT, or checkpoint logic belongs under
`flash_rt/models/<model>/` or `flash_rt/frontends/<framework>/`.

### Error Handling

Do not silently continue after CUDA, cuBLASLt, CUTLASS, or allocation errors.
Unsupported shapes/layouts should raise a clear exception with the operation
name and shape. Undefined outputs, all-zero fallthroughs, and warning-only
failures are not acceptable for runtime kernels.

### Kernel Bindings And CMake Ownership

Every pybind entry must have matching CMake target ownership.

- If `csrc/bindings.cpp` exposes a function unconditionally, the `.cu` file
  that implements it must be compiled into `flash_rt_kernels` for every
  supported `GPU_ARCH`, or the binding must call an unconditional stub that
  raises a clear "not built / not supported" error.
- If a kernel implementation is compiled only behind a hardware or feature
  gate, the binding must use the same compile-time guard. Do not leave an
  unconditional `m.def(...)` that references symbols from a gated object
  library.
- Model-specific object libraries, such as Motus SM120-only targets, should
  contain only model- and hardware-specific kernels. Shared quantize, layout,
  RoPE, activation, and utility kernels belong in the main target unless
  every binding and caller is gated with the same condition.
- When moving sources between object libraries and `flash_rt_kernels`, verify
  both sides: the target that was missing the symbols imports successfully,
  and the target that already had them does not fail with duplicate
  definitions.

### Kernel Alias And Shape Contracts

Legacy pybind names are part of the public runtime ABI. When a new shared
helper replaces an older model- or hardware-specific kernel, the wrapper must
preserve both the shape contract and the numerical contract of the old name.
This rule exists because #30 introduced an integration regression by wiring an
older `bias_gelu_bf16(_strict)` alias to a newer shared helper with the wrong
shape mapping; #40 is the reference fix for that class of bug.

- Do not silently reinterpret `(seq_len, dim)` as `(seq_len * dim, dim)` or
  `(M, N)` without proving the old and new kernels use the same indexing.
  If the helper expects `(M, N)`, pass `M=seq_len` and `N=dim` for a
  row-major `(seq_len, dim)` tensor.
- Names containing `strict`, `bf16`, `fp16`, `rowwise`, `static`, or
  `inplace` must keep that behavior after refactors. If the behavior changes,
  add a new binding name or update all callers and docs in the same PR.
- A fused replacement must be validated against the unfused reference path
  for the exact dtype and rounding semantics it claims to preserve. For BF16
  strict paths, this means checking whether the old path rounded to BF16
  between operations.
- Bindings that exist only for backward compatibility should say so in a
  comment next to the `m.def(...)`, including the expected argument shapes.

Minimum validation for binding / CMake changes:

```bash
cmake -B build -S . -DGPU_ARCH=<arch>
cmake --build build -j$(nproc) --target flash_rt_kernels
python - <<'PY'
from flash_rt import flash_rt_kernels
print(flash_rt_kernels.__file__)
PY
```

Run this for each affected hardware family, not only for the GPU in your
workstation. Undefined symbols often show up at Python import time even when
the CMake build itself completed.

### Execution Contract And Serving Layer

The `exec/` execution contract and the `serving/` examples built on it follow a
strict mechanism-not-policy rule. The full rule and a per-PR review checklist
live in [`docs/exec_contract.md`](docs/exec_contract.md) §9; the essentials:

- The contract is `Buffer` / `Graph` / `Plan` / `Event` / `ShapeKey`, and nothing
  more. The only allowed extensions are ShapeKey semantics and the number of
  buffers/graphs. Do not add a session, KV, cache, scheduler, batching-policy,
  protocol, agent, or robot field or verb to `exec/`.
- Scenario policy — sessions and prefix/KV reuse, eviction/scheduling,
  OpenAI/MCP protocol, tool-call parsing, agent loops, robot
  episode/cadence/interruption orchestration — lives in `serving/` or the user
  host, never in the contract.
- GPU/model state stays owned by the frontend. The serving layer holds metadata
  only (token journal, cache plan, episode bookkeeping).
- Capture, calibration, and warmup stay in the Python frontend; the contract
  only **adopts** the resulting instantiated graph and owns replay-time.
- `exec/` is a top-level sibling of `csrc/` with zero `csrc` dependency. It must
  not include or link kernel sources.
- Integration is additive and opt-in (e.g. `FLASHRT_QWEN36_USE_EXEC`): the
  default path stays byte-identical. A new code path must produce output
  identical to the path it replaces — bit-identical for deterministic decode,
  token-exact for speculative decode, cosine ≥ 0.999 for VLA diffusion.

`serving/qwen36_agent` is the reference example that satisfies every item above;
use it as the comparison point when adding a new serving host.

### Calibration And Precision

FP8/NVFP4 changes must preserve the calibration cache contract described in
[`docs/calibration.md`](docs/calibration.md). When changing quantization,
calibration, or graph capture behavior, include a precision comparison:

- cosine vs the relevant reference when a fixture exists
- action sanity check for quickstart-only paths
- latency before/after for performance-sensitive changes

### Performance Measurement

Use the right metric for the claim:

- `quickstart.py --benchmark` reports wall-clock `model.predict(...)`
  latency, including graph-external preprocessing/copy/postprocessing.
- CUDA Graph replay measurements report captured graph latency only.

Do not compare wall-clock quickstart numbers directly against replay numbers.
README performance tables state which metric is being reported.

## Testing

Use focused tests first, then broaden based on risk.

### Basic Smoke Tests

```bash
python -m pytest \
  tests/test_install_smoke.py \
  tests/test_load_model_use_fp8_kwarg.py \
  tests/test_calibration_helpers.py \
  -q
```

### Runtime Quickstart

For VLA runtime changes, at least run the affected model's quickstart:

```bash
python examples/quickstart.py \
  --checkpoint /path/to/pi05_checkpoint \
  --config pi05 \
  --framework torch \
  --hardware rtx_sm120 \
  --benchmark 20
```

Use the corresponding `--config` for `pi0`, `groot`, or `pi0fast`.

### Precision And Regression Tests

Use these when the change touches model math, calibration, graph capture, or
kernel dispatch:

```bash
python -m pytest tests/test_pi05_batched_precision.py -q -s
python tests/test_all_models_precision.py --model pi0
python tests/test_pi0fast_precision.py --backend pi0fast_jax
```

Some precision tests require local checkpoints, reference fixtures, or an
environment with `openpi` installed. If a test cannot run in your environment,
say so in the PR and include the reason.

## Pull Request Checklist

For external contributors, use the standard fork workflow:

```bash
# 1. Fork LiangSu8899/FlashRT on GitHub, then clone your fork.
git clone git@github.com:<your-user>/FlashRT.git
cd FlashRT

# 2. Keep the upstream repository available for sync.
git remote add upstream git@github.com:LiangSu8899/FlashRT.git
git fetch upstream

# 3. Start from the latest upstream main.
git checkout main
git merge --ff-only upstream/main

# 4. Create a focused branch for the change.
git checkout -b fix/short-description

# 5. Commit, push to your fork, then open a PR to upstream main.
git push -u origin fix/short-description
```

Open the pull request from:

```text
<your-user>:fix/short-description -> LiangSu8899:main
```

Before requesting review:

- Rebase or fast-forward onto the latest `main`.
- Keep the change scoped to one behavior or model path.
- Update docs when the user-facing API, build flow, supported hardware, or
  performance claims change.
- Add or update tests for new behavior.
- For CMake / kernel binding changes, confirm each affected `GPU_ARCH` builds
  `flash_rt_kernels` and imports it from Python; check for missing or duplicate
  symbols when a source moves between targets.
- For kernel alias or fused-kernel refactors, compare the old public binding
  signature, wrapper argument mapping, tensor shape comments, and numerical
  semantics against the new helper before review. Use #30 / #40 as the
  concrete checklist example: legacy alias, new shared helper, shape mapping,
  and strict BF16 semantics must all be checked together.
- For hardware additions, confirm the change is additive: unrelated hardware
  must not inherit new compile units, runtime branches, env vars, or default
  feature paths unless the PR explicitly validates those devices.
- `hasattr(fvk, "...")` is allowed only for optional fast paths with a correct
  fallback. Do not use it as hardware routing for required model behavior.
- For `exec/` contract or `serving/` example changes, run the
  [`docs/exec_contract.md`](docs/exec_contract.md) §9.2 checklist: confirm
  `git diff main -- exec/` adds no scenario field, that policy stays in
  `serving/`, that the change is additive/opt-in, and that the new path's output
  matches the path it replaces. Tests that need a serving-only dependency such as
  `fastapi` must `pytest.importorskip(...)` so kernel-only environments skip
  rather than fail.
- Include validation commands and results in the PR description.
- Mention unsupported hardware or missing local fixtures explicitly.
- Avoid committing generated build outputs, local checkpoints, logs, or
  `third_party/cutlass`.

## Reporting Hardware Results

Hardware validation reports are useful even without code changes. Include:

- GPU model and compute capability
- CUDA toolkit and driver version
- PyTorch/JAX versions
- build flags, especially `GPU_ARCH` and FA2 slim-build flags
- checkpoint/config/framework used
- command line
- P50 latency and whether it is wall-clock or graph replay
- relevant error trace if the run failed

## Commit Style

Use direct, technical commit messages:

```text
Fix FP8 descale GEMM error handling
Add Pi0.5 SM89 fallback routing
Document Qwen3.6 NVFP4 cache requirements
```

Prefer small PRs. Runtime changes are easier to review when tests and
benchmarks map directly to the touched path.
