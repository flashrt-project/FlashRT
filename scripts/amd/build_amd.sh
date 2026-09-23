#!/usr/bin/env bash
# Build flash_rt_amd_kernels in a ROCm environment (GPU visible or not).
#   bash scripts/amd/build_amd.sh [gfx950|gfx1151]
#
# Prefers CMake; falls back to a direct hipcc one-shot when no usable
# cmake is present. Output: flash_rt/amd/flash_rt_amd_kernels*.so
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ARCH="${1:-gfx950}"

# Supported native source sets are gfx950 (CDNA4) and gfx1151 (RDNA 3.5).
# Compare the base target only (for example, gfx950 from gfx950:xnack-).
GPU_ARCH_BASE="${GPU_ARCH%%:*}"
if [[ "${GPU_ARCH_BASE}" != "gfx950" && "${GPU_ARCH_BASE}" != "gfx1151" \
      && "${FLASHRT_AMD_ALLOW_ARCH:-0}" != "1" ]]; then
  echo "error: GPU_ARCH='${GPU_ARCH}' has no FlashRT AMD source set." >&2
  echo "       Pass gfx950/gfx1151, or set FLASHRT_AMD_ALLOW_ARCH=1." >&2
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
PYBIND_INCLUDES="$("${PYTHON_BIN}" -m pybind11 --includes)"
EXT_SUFFIX="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"

mkdir -p "${ROOT}/flash_rt/amd"
OUT="${ROOT}/flash_rt/amd/flash_rt_amd_kernels${EXT_SUFFIX}"

if command -v cmake >/dev/null 2>&1 \
   && cmake -B "${BUILD_DIR}" -S "${ROOT}/csrc/amd" \
        -DGPU_ARCH="${GPU_ARCH}" \
        -DROCM_PATH="${ROCM_PATH}" \
        -Dhip_DIR="${ROCM_PATH}/lib/cmake/hip" \
        -Dhipblaslt_DIR="${ROCM_PATH}/lib/cmake/hipblaslt" \
        -DFLASHRT_AMD_ALLOW_ARCH="${FLASHRT_AMD_ALLOW_ARCH:-0}" \
        -DPython_EXECUTABLE="${PYTHON_BIN}" 2>&1; then
  cmake --build "${BUILD_DIR}" -j "${JOBS}"
  echo "built via cmake: $(ls "${ROOT}"/flash_rt/amd/flash_rt_amd_kernels*.so)"
else
  echo "cmake unavailable or failed — falling back to direct hipcc"
  if [[ "${GPU_ARCH_BASE}" == "gfx1151" ]]; then
    hipcc -O3 -std=c++17 -fPIC -shared \
      --offload-arch="${GPU_ARCH}" \
      -ffp-contract=fast \
      -DFLASHRT_AMD_GPU_ARCH="\"${GPU_ARCH}\"" \
      ${PYBIND_INCLUDES} \
      -I"${ROOT}/csrc/amd" \
      -x hip "${ROOT}/csrc/amd/bindings_rdna.cpp" \
      "${ROOT}/csrc/amd/kernels/norm_rdna.hip" \
      "${ROOT}/csrc/amd/kernels/activation_rdna.hip" \
      "${ROOT}/csrc/amd/kernels/elementwise_rdna.hip" \
      "${ROOT}/csrc/amd/kernels/rope_rdna.hip" \
      "${ROOT}/csrc/amd/attention/decoder_flash_rdna.hip" \
      "${ROOT}/csrc/amd/attention/encoder_flash_rdna.hip" \
      "${ROOT}/csrc/amd/gemm/hipblaslt_runner_rdna.hip" \
      "${ROOT}/csrc/amd/gemm/smallm_wmma_rdna.hip" \
      -L"${ROCM_PATH}/lib" -Wl,-rpath,"${ROCM_PATH}/lib" -lhipblaslt \
      -o "${OUT}"
  else
    hipcc -O3 -std=c++17 -fPIC -shared \
      --offload-arch="${GPU_ARCH}" \
      -ffp-contract=fast \
      -DFLASHRT_AMD_GPU_ARCH="\"${GPU_ARCH}\"" \
      ${PYBIND_INCLUDES} \
      -I"${ROOT}/csrc/amd" \
      -x hip "${ROOT}/csrc/amd/bindings.cpp" \
      "${ROOT}/csrc/amd/kernels/norm.hip" \
      "${ROOT}/csrc/amd/kernels/activation.hip" \
      "${ROOT}/csrc/amd/kernels/elementwise.hip" \
      "${ROOT}/csrc/amd/kernels/rope.hip" \
      "${ROOT}/csrc/amd/kernels/patch_embed.hip" \
      "${ROOT}/csrc/amd/kernels/fusion.hip" \
      "${ROOT}/csrc/amd/kernels/quantize_fp8.hip" \
      "${ROOT}/csrc/amd/kernels/elementwise_fp16.hip" \
      "${ROOT}/csrc/amd/kernels/norm_fp16.hip" \
      "${ROOT}/csrc/amd/kernels/adaln_layer_norm.hip" \
      "${ROOT}/csrc/amd/kernels/stream_probe.hip" \
      "${ROOT}/csrc/amd/kernels/ew_tune.hip" \
      "${ROOT}/csrc/amd/attention/decoder_flash.hip" \
      "${ROOT}/csrc/amd/attention/attn_probe.hip" \
      "${ROOT}/csrc/amd/attention/encoder_flash.hip" \
      "${ROOT}/csrc/amd/gemm/hipblaslt_runner.hip" \
      "${ROOT}/csrc/amd/gemm/smallm_fp8.hip" \
      "${ROOT}/csrc/amd/gemm/smallm_mfma.hip" \
      "${ROOT}/csrc/amd/gemm/smallm_mfma_bf16.hip" \
      "${ROOT}/csrc/amd/gemm/decoder_ffn_fused.hip" \
      -L"${ROCM_PATH}/lib" -lhipblaslt \
      -o "${OUT}"
  fi
  echo "built via hipcc: ${OUT}"
fi

"${PYTHON_BIN}" - <<PY
import sys; sys.path.insert(0, "${ROOT}")
from flash_rt.amd import flash_rt_amd_kernels as k
print("import ok:", dict(k.build_info()))
PY
