# Standalone Ascend kernels

Build with `bash scripts/npu/build.sh` after activating CANN. This subtree
has no CUDA, HIP, PyTorch C++ extension, or root CMake dependency.

`flashrt_npu_quantize_rows(stream, input, inverse_scales, output, rows, columns)`
accepts contiguous BF16 input, one FP32 inverse scale per row, and INT8 output.
Columns must be divisible by 32. The return value checks host arguments;
device errors surface through the owning runtime's checked completion API.
Submission does not allocate memory or synchronize the CPU. The caller must
retain buffers until completion. Kernels use FP32 arithmetic and the CANN
AscendQuant rounding sequence, whose intermediate conversion can differ by
one INT8 unit from an ideal FP32-only rounding expression.

The 910B implementation uses 4096-element tiles for smaller dimensions and
8192-element tiles for large dimensions, with up to 40 vector blocks. Native
replay and memory ownership remain model-independent in `flash_rt/npu/core`.
