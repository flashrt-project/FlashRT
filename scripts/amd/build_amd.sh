#!/usr/bin/env bash
# Build flash_rt_amd_kernels in a ROCm environment (GPU visible or not).
#   bash scripts/amd/build_amd.sh [gfx942|gfx950|gfx1151]
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
    echo "error: pass gfx942, gfx950, or gfx1151 when no AMD GPU is visible" >&2
    exit 1
  fi
fi

# Compare the base target exactly; a prefix test would accept future,
# unvalidated architectures such as gfx9420 or gfx9500.
GPU_ARCH_BASE="${GPU_ARCH%%:*}"
if [[ "${GPU_ARCH_BASE}" != "gfx942" && "${GPU_ARCH_BASE}" != "gfx950" \
      && "${GPU_ARCH_BASE}" != "gfx1151" \
      && "${FLASHRT_AMD_ALLOW_ARCH:-0}" != "1" ]]; then
  echo "error: GPU_ARCH='${GPU_ARCH}' has no FlashRT AMD source set." >&2
  echo "       Pass gfx942/gfx950/gfx1151, or set FLASHRT_AMD_ALLOW_ARCH=1." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python3}"
PYTHON_BIN="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
JOBS="${SLURM_CPUS_PER_TASK:-8}"
if [[ -z "${ROCM_PATH:-}" ]]; then
  ROCM_SDK_BIN="$(dirname "${PYTHON_BIN}")/rocm-sdk"
  if [[ -x "${ROCM_SDK_BIN}" ]]; then
    ROCM_PATH="$("${ROCM_SDK_BIN}" path --root)"
  else
    ROCM_PATH="/opt/rocm"
  fi
fi
export ROCM_PATH
export PATH=${ROCM_PATH}/bin:${PATH}
BUILD_DIR="${FLASHRT_AMD_BUILD_DIR:-${ROOT}/build-amd-${GPU_ARCH_BASE}}"

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
cmake -B "${BUILD_DIR}" -S "${ROOT}/csrc/amd" \
  -DGPU_ARCH="${GPU_ARCH}" \
  -DROCM_PATH="${ROCM_PATH}" \
  -Dhip_DIR="${ROCM_PATH}/lib/cmake/hip" \
  -Dhipblaslt_DIR="${ROCM_PATH}/lib/cmake/hipblaslt" \
  -DFLASHRT_AMD_ALLOW_ARCH="${FLASHRT_AMD_ALLOW_ARCH:-0}" \
  -DPython_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${BUILD_DIR}" -j "${JOBS}"
echo "built via cmake: $(ls "${ROOT}"/flash_rt/amd/flash_rt_amd_kernels*.so)"

"${PYTHON_BIN}" - <<PY
import sys; sys.path.insert(0, "${ROOT}")
from flash_rt.amd import flash_rt_amd_kernels as k
print("import ok:", dict(k.build_info()))
PY
