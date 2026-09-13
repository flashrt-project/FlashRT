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
npu_abi_version="${FLASHRT_NPU_ABI_VERSION:-1}"
npu_abi_defines=(
  "-DFLASHRT_NPU_SOC_VERSION=\"$npu_soc_version\""
  "-DFLASHRT_NPU_ABI_VERSION=$npu_abi_version"
)
# Which models' kernels enter the build. A shared object that no selected model
# loads has no business being compiled: a Pi0.5 deployment should not have to
# build the GR00T N1.7 action head's units, and the default here is what the
# Pi0.5 build did before those units existed.
#
#   FLASHRT_ENABLE_NPU_PI05=ON|OFF        (default ON)
#   FLASHRT_ENABLE_NPU_GROOT_N17=ON|OFF   (default OFF)
# Unset takes the default; set-but-empty does not, because that is a mistake
# rather than a choice and guessing which one it meant is how a deployment ends
# up with units it did not ask for.
npu_enable_pi05="${FLASHRT_ENABLE_NPU_PI05-ON}"
npu_enable_groot_n17="${FLASHRT_ENABLE_NPU_GROOT_N17-OFF}"
# The standard artifacts of each model, listed once so that the build and the
# prune below cannot disagree about what belongs to whom.
npu_units_pi05=(libflashrt_npu.so libflashrt_npu_cube.so libflashrt_npu_decoder.so
                libflashrt_npu_attn.so)
npu_units_groot_n17=(libflashrt_npu_groot_n17_dit_attn.so
                     libflashrt_npu_groot_n17_dit_norm.so
                     libflashrt_npu_groot_n17_image.so)
for npu_switch in npu_enable_pi05 npu_enable_groot_n17; do
    case "${!npu_switch}" in
        ON|OFF) ;;
        *) echo "${npu_switch^^} must be ON or OFF (got ${!npu_switch})" >&2; exit 1 ;;
    esac
done
if [[ "$npu_enable_pi05" == "OFF" && "$npu_enable_groot_n17" == "OFF" ]]; then
    echo "no model selected: set FLASHRT_ENABLE_NPU_PI05=ON or FLASHRT_ENABLE_NPU_GROOT_N17=ON" >&2
    exit 1
fi
mkdir -p "$npu_output_dir"
# A deselected model's libraries are removed, not merely skipped. Building with
# a model ON and then building again with it OFF otherwise leaves its shared
# objects behind, so the output directory disagrees with the selection that
# produced it and with what this script reports -- and a loader would bind them.
# Only the standard names above are touched; an override lives wherever the
# caller put it and is none of this script's business.
npu_prune() {
    local -n units=$1
    local unit
    for unit in "${units[@]}"; do
        if [[ -e "$npu_output_dir/$unit" ]]; then
            rm -f "$npu_output_dir/$unit"
            echo "removed $unit (its model is not selected)"
        fi
    done
}
[[ "$npu_enable_pi05" == "ON" ]] || npu_prune npu_units_pi05
[[ "$npu_enable_groot_n17" == "ON" ]] || npu_prune npu_units_groot_n17
# Resolved for every selected model: the mixed cube/vector units and the host
# tiling both need it, and it stays independent of torch.
npu_arch_root="$npu_toolkit_root/$(uname -m)-linux"
if [[ ! -d "$npu_arch_root/asc/include" ]]; then npu_arch_root="$npu_toolkit_root"; fi

if [[ "$npu_enable_pi05" == "ON" ]]; then
    "$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
     --cce-soc-version="$npu_soc_version" --cce-soc-core-type=VecCore \
     "${npu_abi_defines[@]}" \
     -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
     -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
     -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" -I"$npu_toolkit_root/include" \
     -o "$npu_output_dir/libflashrt_npu.so" "$npu_repo_root/csrc/npu/dispatch_910b.cpp"
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
fi

if [[ "$npu_enable_groot_n17" == "ON" ]]; then
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
     -o "$npu_output_dir/libflashrt_npu_groot_n17_dit_norm.so" \
     "$npu_repo_root/csrc/npu/kernels/groot_n17/dit_norm_910b.cpp"
    # The evaluation transform's resize is vector only and belongs to the image
    # path rather than to the action head, so it gets its own unit and its own
    # shared object.
    "$npu_toolkit_root/bin/bisheng" -fPIC -shared -xcce -O2 -std=c++17 \
     --cce-soc-version="$npu_soc_version" --cce-soc-core-type=VecCore \
     "${npu_abi_defines[@]}" \
     -I"$npu_toolkit_root/compiler/tikcpp/tikcfw" \
     -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/impl" \
     -I"$npu_toolkit_root/compiler/tikcpp/tikcfw/interface" -I"$npu_toolkit_root/include" \
     -o "$npu_output_dir/libflashrt_npu_groot_n17_image.so" \
     "$npu_repo_root/csrc/npu/kernels/groot_n17/area_resize_910b.cpp"
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
     "$npu_repo_root/csrc/npu/kernels/groot_n17/dit_attn_910b.cpp" \
     -L"$npu_arch_root/lib64" -lascendalog -lruntime -ldl \
     -o "$npu_output_dir/libflashrt_npu_groot_n17_dit_attn.so"
fi

echo "Ascend kernels in $npu_output_dir (pi05=$npu_enable_pi05 groot_n17=$npu_enable_groot_n17)"
