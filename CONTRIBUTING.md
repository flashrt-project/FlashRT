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
