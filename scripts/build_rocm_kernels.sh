#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
GPU_ARCH="${GPU_ARCH:-gfx942}"

EXT_SUFFIX="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
)"

PYBIND_INCLUDES="$("${PYTHON_BIN}" -m pybind11 --includes)"
OUT="${ROOT}/flash_rt/flash_rt_rocm_kernels${EXT_SUFFIX}"

mkdir -p "${ROOT}/flash_rt"

hipcc -O3 -std=c++17 -fPIC -shared \
  --offload-arch="${GPU_ARCH}" \
  ${PYBIND_INCLUDES} \
  -I"${ROOT}/rsrc" \
  "${ROOT}/rsrc/bindings.cpp" \
  "${ROOT}/rsrc/gemm/hipblaslt_matmul.cpp" \
  "${ROOT}/rsrc/gemm/hipblaslt_probe.cpp" \
  "${ROOT}/rsrc/kernels/activation.hip" \
  "${ROOT}/rsrc/kernels/norm.hip" \
  "${ROOT}/rsrc/kernels/patch_embed.hip" \
  "${ROOT}/rsrc/kernels/qkv_split.hip" \
  "${ROOT}/rsrc/kernels/quantize.hip" \
  "${ROOT}/rsrc/kernels/vector_add.hip" \
  -L/opt/rocm/lib \
  -lhipblaslt \
  -o "${OUT}"

echo "built ${OUT}"
