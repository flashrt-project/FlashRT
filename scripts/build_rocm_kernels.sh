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

CK_INCLUDE_DIR="${CK_INCLUDE_DIR:-/opt/venv/lib/python3.12/site-packages/aiter_meta/3rdparty/composable_kernel/include}"
CK_LIBRARY_INCLUDE_DIR="${CK_LIBRARY_INCLUDE_DIR:-/opt/venv/lib/python3.12/site-packages/aiter_meta/3rdparty/composable_kernel/library/include}"
CK_INCLUDES=""
if [[ -d "${CK_INCLUDE_DIR}" ]]; then
  CK_INCLUDES="${CK_INCLUDES} -I${CK_INCLUDE_DIR}"
fi
if [[ -d "${CK_LIBRARY_INCLUDE_DIR}" ]]; then
  CK_INCLUDES="${CK_INCLUDES} -I${CK_LIBRARY_INCLUDE_DIR}"
fi

mkdir -p "${ROOT}/flash_rt"

hipcc -O3 -std=c++17 -fPIC -shared \
  --offload-arch="${GPU_ARCH}" \
  ${PYBIND_INCLUDES} \
  -I"${ROOT}/rsrc" \
  ${CK_INCLUDES} \
  "${ROOT}/rsrc/bindings.cpp" \
  "${ROOT}/rsrc/gemm/hipblaslt_matmul.cpp" \
  "${ROOT}/rsrc/gemm/hipblaslt_probe.cpp" \
  "${ROOT}/rsrc/kernels/activation.hip" \
  "${ROOT}/rsrc/kernels/attention_decode.hip" \
  "${ROOT}/rsrc/kernels/attention_pi05_ck.hip" \
  "${ROOT}/rsrc/kernels/embedding.hip" \
  "${ROOT}/rsrc/kernels/norm.hip" \
  "${ROOT}/rsrc/kernels/patch_embed.hip" \
  "${ROOT}/rsrc/kernels/qkv_split.hip" \
  "${ROOT}/rsrc/kernels/qwen36_linear.hip" \
  "${ROOT}/rsrc/kernels/quantize.hip" \
  "${ROOT}/rsrc/kernels/vector_add.hip" \
  -L/opt/rocm/lib \
  -lhipblaslt \
  -o "${OUT}"

echo "built ${OUT}"
