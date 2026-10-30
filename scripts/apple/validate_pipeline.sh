#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
python_bin="${FLASHRT_APPLE_PYTHON:-$root/.venv/bin/python}"
"$python_bin" -c 'import torch; assert torch.backends.mps.is_available(), "Host MPS GPU required; qualification did not run"'
bash scripts/apple/build_native.sh
MTL_DEBUG_LAYER=1 .cache/apple-native/flashrt_apple_kernel_registry_smoke
MTL_DEBUG_LAYER=1 "$python_bin" -m pytest tests/test_apple_native_contracts.py tests/test_apple_native_plan.py tests/test_apple_profiling.py -q -x
# Separate uninstrumented process. Requires checkpoint/tokenizer/output args.
PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" examples/apple/benchmark_native_pipeline.py "$@"
