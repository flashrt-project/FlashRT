"""ROCm backend marker package.

The AMD path is intentionally additive: CUDA/Thor/RTX code remains in the
existing hardware packages while ROCm-specific runtime, graph, GEMM, and
attention pieces land here incrementally.
"""

BACKEND_NAME = "rocm"
