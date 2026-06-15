#!/usr/bin/env python3
"""Build the Cosmos3-video model-local kernel extension `cosmos3_video_kernels`.

Additive + isolated: produces a standalone .so inside this package; does NOT touch
the production flash_rt_kernels.so or its CMake (docs/adding_new_model.md §4.5). The
one kernel here (fused qk-norm + rope) depends only on the CUDA toolkit. Run ONCE on
the target machine (RTX 5090 / sm_120):

    cd flash_rt/models/cosmos3_video/kernels && python3 setup.py build_ext --inplace

All paths are derived from this file's location — no hard-coded host paths.
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")
# -DNDEBUG: CUDA's bf16 headers use device-side assert(); release build no-ops it.
_nvcc = ["-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "--use_fast_math",
         "-gencode=arch=compute_120a,code=sm_120a"]

setup(
    name="cosmos3_video_kernels",
    ext_modules=[CUDAExtension(
        name="cosmos3_video_kernels",
        sources=[os.path.join(_CSRC, f) for f in (
            "bindings.cpp", "fused_qk_norm_rope.cu")],
        include_dirs=[_CSRC],
        extra_compile_args={"cxx": ["-O3"], "nvcc": _nvcc},
    )],
    cmdclass={"build_ext": BuildExtension},
)
