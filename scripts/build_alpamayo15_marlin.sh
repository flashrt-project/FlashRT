#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build only the vLLM v0.16.0 W4A8 Marlin op required by FlashDrive on SM110.
set -euo pipefail
SRC=$(realpath "${1:?provide the vLLM v0.16.0 source checkout}")
B=$(realpath -m "${2:?provide a fresh build directory}")
PY=$(command -v "${PYTHON:-python}")
[ ! -e "$B" ] || { echo "Build directory already exists; choose a fresh directory" >&2; exit 2; }
mkdir -p "$B"
cp -r "$SRC/csrc" "$B/csrc"
cd "$B"
$PY - <<'PY'
import importlib.util, sys
sys.argv = ["generate_kernels.py", "11.0"]
spec = importlib.util.spec_from_file_location("gk", "csrc/quantization/marlin/generate_kernels.py")
gk = importlib.util.module_from_spec(spec); spec.loader.exec_module(gk)
assert gk.SUPPORT_SM80 and not gk.SUPPORT_FP8 and not gk.SUPPORT_SM75
keep = [c for c in gk.QUANT_CONFIGS if c.get("a_type") == ["kS8"] and c["b_type"] == "kU4"]
assert len(keep) == 1, keep
gk.QUANT_CONFIGS = keep
gk.remove_old_kernels(); gk.generate_new_kernels()
PY
ls csrc/quantization/marlin/sm80_kernel_*.cu
grep -c "kernel = " csrc/quantization/marlin/kernel_selector.h
cat > marlin_ext.cpp <<'CPP'
// Registers vLLM v0.16.0's marlin_gemm schema (verbatim from csrc/torch_bindings.cpp) in the `_C` library.
#include "core/registration.h"
#include <torch/library.h>
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
CPP
export TORCH_CUDA_ARCH_LIST=11.0 MAX_JOBS=${MAX_JOBS:-12}
$PY - <<'PY'
import glob, torch
from torch.utils.cpp_extension import load
srcs = ["csrc/quantization/marlin/marlin.cu"] + sorted(glob.glob("csrc/quantization/marlin/sm80_kernel_*.cu")) + ["marlin_ext.cpp"]
m = load(name="_C", sources=srcs, build_directory=__import__("os").getcwd(), extra_include_paths=[__import__("os").path.join(__import__("os").getcwd(), "csrc")],
         extra_cflags=["-O3", "-std=c++17"],
         extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-static-global-template-stub=false"],
         is_python_module=True, verbose=True)
print("built", m.__file__, hasattr(torch.ops._C, "marlin_gemm"))
PY
sha256sum "$B/_C.so"
"$PY" - "$B/_C.so" <<'PY'
import pathlib, shutil, sys, vllm
package = pathlib.Path(vllm.__file__).parent
if list(package.glob("_C*.so")):
    raise RuntimeError("vLLM already has a CUDA extension; use a fresh empty-device vLLM installation")
shutil.copy2(sys.argv[1], package / "_C.so")
PY
