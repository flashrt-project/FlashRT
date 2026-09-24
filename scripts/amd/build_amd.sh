#!/usr/bin/env bash
# Build flash_rt_amd_kernels in a ROCm environment (GPU visible or not).
#   bash scripts/amd/build_amd.sh [gfx942|gfx950]
#
# CMake owns the architecture-specific source set. Output:
# flash_rt/amd/flash_rt_amd_kernels*.so
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -gt 0 ]]; then
  GPU_ARCH="$1"
else
  GPU_ARCH="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]\+' || true)"
  if [[ -z "${GPU_ARCH}" ]]; then
    echo "error: pass gfx942 or gfx950 when no AMD GPU is visible" >&2
    exit 1
  fi
fi

# Compare the base target exactly; a prefix test would accept future,
# unvalidated architectures such as gfx9420 or gfx9500.
GPU_ARCH_BASE="${GPU_ARCH%%:*}"
if [[ "${GPU_ARCH_BASE}" != "gfx942" && "${GPU_ARCH_BASE}" != "gfx950" ]]; then
  echo "error: GPU_ARCH='${GPU_ARCH}' is unsupported; expected gfx942 or gfx950." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python3}"
JOBS="${SLURM_CPUS_PER_TASK:-8}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH=${ROCM_PATH}/bin:${PATH}

# pybind11 headers: use the interpreter's copy, else a one-time --target
# install into FLASHRT_AMD_PYDEPS (never into a read-only/shared venv).
if ! "${PYTHON_BIN}" -m pybind11 --includes >/dev/null 2>&1; then
  PYDEPS="${FLASHRT_AMD_PYDEPS:-${ROOT}/.pydeps}"
  if [[ ! -d "${PYDEPS}/pybind11" ]]; then
    "${PYTHON_BIN}" -m pip install --target "${PYDEPS}" pybind11
  fi
  export PYTHONPATH="${PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
fi
mkdir -p "${ROOT}/flash_rt/amd"
if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake is required for the architecture-specific AMD build" >&2
  exit 1
fi
cmake -B "${ROOT}/build-amd" -S "${ROOT}/csrc/amd" \
  -DGPU_ARCH="${GPU_ARCH}" -DPython_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${ROOT}/build-amd" -j "${JOBS}"
echo "built via cmake: $(ls "${ROOT}"/flash_rt/amd/flash_rt_amd_kernels*.so)"

"${PYTHON_BIN}" - <<PY
import sys; sys.path.insert(0, "${ROOT}")
from flash_rt.amd import flash_rt_amd_kernels as k
print("import ok:", dict(k.build_info()))
PY
