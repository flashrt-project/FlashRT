#!/usr/bin/env bash
# Standalone Ascend build: no root CMake, CUDA, PyTorch extension or nvcc.
set -euo pipefail
npu_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
npu_toolkit_root="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
npu_output_dir="${FLASHRT_NPU_BUILD_DIR:-$npu_repo_root/flash_rt/npu/lib}"
mkdir -p "$npu_output_dir"
"$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
 --cce-soc-version="${ASCEND_SOC_VERSION:-Ascend910B4}" --cce-soc-core-type=VecCore \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" -I"$npu_toolkit_root/include" \
 -o "$npu_output_dir/libflashrt_npu.so" "$npu_repo_root/csrc/npu/kernels/row_quant_910b.cpp"
