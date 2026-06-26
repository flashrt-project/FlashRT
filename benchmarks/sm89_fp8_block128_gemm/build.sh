#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

NVCC="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
CXX_HOST="${CXX_HOST:-/data/home/tianjianyang/miniconda3/bin/x86_64-conda-linux-gnu-c++}"
OUT="${BUILD_DIR}/bench_sm89_fp8_block128_gemm"

"${NVCC}" \
  --compiler-bindir="${CXX_HOST}" \
  --std=c++17 \
  --expt-relaxed-constexpr \
  -O3 \
  --use_fast_math \
  -lineinfo \
  --ptxas-options=-v \
  -gencode=arch=compute_89,code=sm_89 \
  "${SCRIPT_DIR}/bench_sm89_fp8_block128_gemm.cu" \
  -o "${OUT}"

echo "${OUT}"
