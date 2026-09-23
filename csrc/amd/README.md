# FlashRT AMD Source

`csrc/amd` is the AMD/ROCm kernel tree. The existing `csrc` tree stays
CUDA-only; the two never build together.

## Contract

Same raw-pointer ABI as the CUDA module: every binding takes `uintptr_t`
device pointers plus a `uintptr_t` stream, never tensors. Entries keep the
CUDA-side names and signatures so pipeline code stays portable text.

Kernels are written in plain HIP C++. The original files target CDNA
(wave64); RDNA implementations stay in the same directories and use an
`_rdna` filename suffix. `hipBLASLt` (shipped with ROCm) backs both the CDNA
`GemmRunner` and the BF16-only RDNA runner, and remains the baseline every
hand-written GEMM must beat standalone before entering a pipeline.

## Build

Standalone CMake project (the root `CMakeLists.txt` is CUDA and untouched):

```bash
cmake -B build-amd -S csrc/amd [-DGPU_ARCH=gfx950|gfx1151]
cmake --build build-amd -j 8
```

or via the wrapper (with hipcc fallback for cmake-less environments):

```bash
bash scripts/amd/build_amd.sh gfx950
bash scripts/amd/build_amd.sh gfx1151
```

Output: `flash_rt/amd/flash_rt_amd_kernels*.so`, imported as
`from flash_rt.amd import flash_rt_amd_kernels`.

## Layout

```
bindings.cpp        pybind11 module flash_rt_amd_kernels (raw-pointer ABI)
bindings_rdna.cpp   reduced BF16 binding surface selected for gfx1151
kernels/            elementwise/norm/activation/quantize kernels (.hip)
  common_hip.h      wave64 reductions, dtype templates (mirrors common.cuh)
  *_rdna.hip        wave32 RDNA variants, selected only for gfx1151
gemm/               hipBLASLt plus CDNA MFMA and RDNA wave32 WMMA kernels
attention/          hand-written CDNA and `_rdna` encoder/decoder attention
```

Python runtime twins live in `flash_rt/amd/core/` (`hip_buffer.py`,
`hip_graph.py`).
