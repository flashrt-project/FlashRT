#!/usr/bin/env bash
# Build the standalone vision-attention FA2 tile micro-bench for ONE
# head-dim bucket. Only links the forward template (no split-KV/causal),
# so it compiles in seconds, not the 9-minute full FA2 rebuild.
#
#   ./build.sh 64    # 2B vision (head_dim 64)
#   ./build.sh 96    # 8B vision (head_dim 72 -> padded to 96)
set -euo pipefail

HDIM="${1:-64}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

NVCC="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
CXX_HOST="${CXX_HOST:-/data/home/tianjianyang/miniconda3/bin/x86_64-conda-linux-gnu-c++}"
FA2_ROOT="${REPO_ROOT}/csrc/attention/flash_attn_2_src"
OUT="${BUILD_DIR}/bench_vision_attn_hdim${HDIM}"

"${NVCC}" \
  --compiler-bindir="${CXX_HOST}" \
  --std=c++17 \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -O3 --use_fast_math -lineinfo \
  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
  -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ \
  -DBENCH_HDIM=${HDIM} \
  -gencode=arch=compute_89,code=sm_89 \
  -I"${REPO_ROOT}/csrc" \
  -I"${FA2_ROOT}" \
  -I"${FA2_ROOT}/flash_attn" \
  -I"${FA2_ROOT}/cutlass/include" \
  "${SCRIPT_DIR}/bench_vision_attn.cu" \
  -o "${OUT}"

echo "${OUT}"
