#!/usr/bin/env bash
# Standalone Ascend build: no root CMake, CUDA, PyTorch extension or nvcc.
set -euo pipefail
npu_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
npu_toolkit_root="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
npu_output_dir="${FLASHRT_NPU_BUILD_DIR:-$npu_repo_root/flash_rt/npu/lib}"
# Ascend910B4 only, deliberately. The host tiling in csrc/npu/gu_tiling.cpp
# names that part, several kernels divide work by its twenty cube cores, and it
# is the only part any of this has been measured on. Widening the accepted set
# would let the build produce libraries whose tiling is wrong for the target.
npu_soc_version="${ASCEND_SOC_VERSION:-Ascend910B4}"
if [[ "$npu_soc_version" != "Ascend910B4" ]]; then
    echo "The native kernels are validated for Ascend910B4 only (got $npu_soc_version)." >&2
    exit 1
fi
# Bumped whenever an exported entry point's signature or contract changes, so a
# stale shared object is refused at load rather than called with wrong arguments.
npu_abi_version="${FLASHRT_NPU_ABI_VERSION:-2}"
npu_abi_defines=(
  "-DFLASHRT_NPU_SOC_VERSION=\"$npu_soc_version\""
  "-DFLASHRT_NPU_ABI_VERSION=$npu_abi_version"
)
mkdir -p "$npu_output_dir"
"$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
 --cce-soc-version="$npu_soc_version" --cce-soc-core-type=VecCore \
 "${npu_abi_defines[@]}" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" -I"$npu_toolkit_root/include" \
 -o "$npu_output_dir/libflashrt_npu.so" "$npu_repo_root/csrc/npu/dispatch_910b.cpp"
# Mixed Cube/Vector translation unit and host tiling stay independent of torch.
npu_arch_root="$npu_toolkit_root/$(uname -m)-linux"
if [[ ! -d "$npu_arch_root/asc/include" ]]; then npu_arch_root="$npu_toolkit_root"; fi
npu_tiling_object="$(mktemp "$npu_output_dir/gu_tiling.XXXXXX.o")"
trap 'rm -f "$npu_tiling_object"' EXIT
c++ -c -fPIC -O2 -std=c++17 \
 -I"$npu_arch_root/asc/include" -I"$npu_arch_root/asc/include/adv_api" \
 -I"$npu_arch_root/include" "$npu_repo_root/csrc/npu/gu_tiling.cpp" -o "$npu_tiling_object"
"$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
 --cce-aicore-arch=dav-c220 "${npu_abi_defines[@]}" \
 -I"$npu_arch_root/asc/include" -I"$npu_arch_root/asc/include/adv_api" \
 -I"$npu_arch_root/asc" -I"$npu_arch_root/asc/impl/basic_api" \
 -I"$npu_arch_root/asc/impl/adv_api" -I"$npu_arch_root/include" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" \
 "$npu_repo_root/csrc/npu/kernels/gu_int8_910b.cpp" -x none "$npu_tiling_object" \
 -L"$npu_arch_root/lib64" -ltiling_api -lplatform -lregister -lascendalog -lruntime -ldl \
 -o "$npu_output_dir/libflashrt_npu_cube.so"
# The decoder INT8 GEMM declares itself cube only at file scope, so it cannot
# share a translation unit with the mixed gate/up kernel, and the kernel
# headers define a per-unit tiling symbol, so it cannot share a library.
"$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
 --cce-aicore-arch=dav-c220 "${npu_abi_defines[@]}" \
 -I"$npu_arch_root/asc/include" -I"$npu_arch_root/asc/include/adv_api" \
 -I"$npu_arch_root/asc" -I"$npu_arch_root/asc/impl/basic_api" \
 -I"$npu_arch_root/asc/impl/adv_api" -I"$npu_arch_root/include" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" \
 "$npu_repo_root/csrc/npu/kernels/decoder_gemm_910b.cpp" \
 -L"$npu_arch_root/lib64" -lascendalog -lruntime -ldl \
 -o "$npu_output_dir/libflashrt_npu_decoder.so"
# Decode attention drives the cube and the vector cores inside one launch, so
# it is a mixed unit and cannot join either of the two above.
"$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
 --cce-aicore-arch=dav-c220 "${npu_abi_defines[@]}" \
 -I"$npu_arch_root/asc/include" -I"$npu_arch_root/asc/include/adv_api" \
 -I"$npu_arch_root/asc" -I"$npu_arch_root/asc/impl/basic_api" \
 -I"$npu_arch_root/asc/impl/adv_api" -I"$npu_arch_root/include" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" \
 "$npu_repo_root/csrc/npu/kernels/decode_attn_910b.cpp" \
 -L"$npu_arch_root/lib64" -lascendalog -lruntime -ldl \
 -o "$npu_output_dir/libflashrt_npu_attn.so"
# The action head's two elementwise shapes are vector only, so they need
# neither the mixed nor the cube toolchain -- only their own library, because a
# loader that can be pointed at a stale shared object has to be able to refuse
# exactly one unit.
"$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
 --cce-soc-version="$npu_soc_version" --cce-soc-core-type=VecCore \
 "${npu_abi_defines[@]}" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" -I"$npu_toolkit_root/include" \
 -o "$npu_output_dir/libflashrt_npu_dit_vector.so" \
 "$npu_repo_root/csrc/npu/kernels/dit_vector_910b.cpp"
# The evaluation transform's resize is vector only and belongs to the image
# path rather than to the action head, so it gets its own unit and its own
# shared object.
"$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
 --cce-soc-version="$npu_soc_version" --cce-soc-core-type=VecCore \
 "${npu_abi_defines[@]}" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" -I"$npu_toolkit_root/include" \
 -o "$npu_output_dir/libflashrt_npu_image.so" \
 "$npu_repo_root/csrc/npu/kernels/area_resize_910b.cpp"
# The DiT attention kernel is a second mixed unit: same reason as above, it
# cannot share a translation unit with a cube-only one nor a library with any
# other kernel unit.
"$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
 --cce-aicore-arch=dav-c220 "${npu_abi_defines[@]}" \
 -I"$npu_arch_root/asc/include" -I"$npu_arch_root/asc/include/adv_api" \
 -I"$npu_arch_root/asc" -I"$npu_arch_root/asc/impl/basic_api" \
 -I"$npu_arch_root/asc/impl/adv_api" -I"$npu_arch_root/include" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
 -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" \
 "$npu_repo_root/csrc/npu/kernels/dit_attn_910b.cpp" \
 -L"$npu_arch_root/lib64" -lascendalog -lruntime -ldl \
 -o "$npu_output_dir/libflashrt_npu_dit_attn.so"
