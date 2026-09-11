"""FlashRT AMD (ROCm/HIP) backend.

Self-contained AMD tree: HIP runtime twins of core/cuda_buffer and
core/cuda_graph, the flash_rt_amd_kernels extension (built by
csrc/amd/CMakeLists.txt, dropped into this directory), and — as the
port progresses — CDNA attention backends, pi05 pipeline, frontends.

The NVIDIA package tree never imports from here and vice versa.
"""
