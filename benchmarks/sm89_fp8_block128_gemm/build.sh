#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

# Shared production kernel header lives here; baseline includes it verbatim.
GEMM_DIR="$(cd "${SCRIPT_DIR}/../../csrc/gemm" && pwd)"

NVCC="${NVCC:-nvcc}"
# Host compiler for nvcc. The system /usr/bin/gcc may be too old for CUDA 12.x;
# point CXX_HOST at a recent g++ (e.g. a conda env's x86_64-conda-linux-gnu-g++)
# if the default toolchain is rejected.
CXX_HOST="${CXX_HOST:-g++}"
OUT="${BUILD_DIR}/bench_sm89_fp8_block128_gemm"

# Build with EXPERIMENT to compile the editable candidate kernel:
#   ./build.sh --experiment
# Without it the candidate aliases the production baseline, so `--mode both`
# reports ~0% delta -- a built-in check that the harness is faithful.
EXTRA=""
for arg in "$@"; do
  case "$arg" in
    --experiment) EXTRA="-DEXPERIMENT" ;;
    *) echo "unknown build arg: $arg" >&2; exit 2 ;;
  esac
done

"${NVCC}" \
  --compiler-bindir="${CXX_HOST}" \
  --std=c++17 \
  --expt-relaxed-constexpr \
  -O3 \
  --use_fast_math \
  -lineinfo \
  --ptxas-options=-v \
  -I"${GEMM_DIR}" \
  ${EXTRA} \
  -gencode=arch=compute_89,code=sm_89 \
  "${SCRIPT_DIR}/bench_sm89_fp8_block128_gemm.cu" \
  -o "${OUT}"

echo "${OUT}"
