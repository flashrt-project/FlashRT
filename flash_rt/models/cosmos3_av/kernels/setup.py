#!/usr/bin/env python3
"""Build the Cosmos3-AV model-local kernel extension `cosmos3_av_kernels`.

Additive + isolated: produces a standalone .so inside this package; does NOT touch
the production flash_rt_kernels.so or its CMake (docs/adding_new_model.md §4.5). Run
ONCE on the target machine (RTX 5090 / sm_120):

    cd flash_rt/models/cosmos3_av/kernels && python3 setup.py build_ext --inplace

All include paths are derived from this file's location — no hard-coded host paths.
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_HERE = os.path.dirname(os.path.abspath(__file__))
# flash_rt/models/cosmos3_av/kernels -> FlashRT root is 4 levels up
_FR = os.environ.get("FLASHRT_ROOT", os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..")))
_CSRC = os.path.join(_HERE, "csrc")

_includes = [
    _CSRC,
    os.path.join(_FR, "third_party", "cutlass", "include"),
    os.path.join(_FR, "third_party", "cutlass", "tools", "util", "include"),
    os.path.join(_FR, "csrc", "attention", "sage2"),
    os.path.join(_FR, "csrc", "attention", "sage2", "qattn"),
]
# -DNDEBUG: CUDA's fp4/fp6/fp8 headers use device-side assert() (-> __assert_fail,
# undefined in device code); release build makes assert() a no-op and sidesteps it.
_nvcc = ["-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "--use_fast_math",
         "-gencode=arch=compute_120a,code=sm_120a"]

setup(
    name="cosmos3_av_kernels",
    ext_modules=[CUDAExtension(
        name="cosmos3_av_kernels",
        sources=[os.path.join(_CSRC, f) for f in (
            "bindings.cpp", "fp4_silu_aux.cu", "sage_gqa_d128.cu",
            "sage_gqa_f8_d128.cu", "fused_qk_norm_rope.cu")],
        include_dirs=_includes,
        extra_compile_args={"cxx": ["-O3"], "nvcc": _nvcc},
    )],
    cmdclass={"build_ext": BuildExtension},
)
