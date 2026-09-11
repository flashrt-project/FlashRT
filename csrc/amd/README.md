# FlashRT AMD Source

`csrc/amd` is the AMD/ROCm kernel tree. The existing `csrc` tree stays
CUDA-only; the two never build together.

## Contract

Same raw-pointer ABI as the CUDA module: every binding takes `uintptr_t`
device pointers plus a `uintptr_t` stream, never tensors. Entries keep the
CUDA-side names and signatures so pipeline code stays portable text.

Kernels are written in plain HIP C++ against CDNA (wave64) — MFMA compiler
intrinsics, LDS, no third-party template libraries. `hipBLASLt` (shipped
with ROCm) backs the `GemmRunner` and is the baseline every hand-written
GEMM must beat standalone before entering the pipeline.

## Build

Standalone CMake project (the root `CMakeLists.txt` is CUDA and untouched):

```bash
cmake -B build-amd -S csrc/amd [-DGPU_ARCH=gfx950]
cmake --build build-amd -j 8
```

or via the wrapper (with hipcc fallback for cmake-less environments):

```bash
bash scripts/amd/build_amd.sh gfx950
```

Output: `flash_rt/amd/flash_rt_amd_kernels*.so`, imported as
`from flash_rt.amd import flash_rt_amd_kernels`.

## Layout

```
bindings.cpp        pybind11 module flash_rt_amd_kernels (raw-pointer ABI)
kernels/            elementwise/norm/activation/quantize kernels (.hip)
  common_hip.h      wave64 reductions, dtype templates (mirrors common.cuh)
gemm/               hipBLASLt runner + probe (added with the pipeline port)
attention/          hand-written CDNA attention (added with the pipeline port)
```

Python runtime twins live in `flash_rt/amd/core/` (`hip_buffer.py`,
`hip_graph.py`).
