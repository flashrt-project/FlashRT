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

`flashrt_npu_decoder_rope` splits merged BF16 QKV, computes rotary products
in FP32, and appends the action rows to persistent KV buffers in one launch.
The existing prefix remains unchanged.

`flashrt_npu_gated_ada` combines a gated branch update with shifted AdaRMS
normalization for 1024-wide decoder rows. Residuals remain FP32. The branch
product and normalized value preserve the pipeline's BF16 rounding boundaries;
the RMS reduction uses FP32. It returns a BF16 normalized activation and,
when a branch is present, the updated FP32 residual. All inputs and outputs
are caller-owned and must remain alive through stream completion.

`flashrt_npu_gelu_mul_quant` combines tanh-approximate GELU, a BF16-rounded
gate/up product and frozen row quantization. It writes INT8 directly, avoiding
the full GELU and product intermediates. The sigmoid form of tanh GELU is
computed in FP32; both BF16 rounding boundaries remain explicit. The kernel
uses 8192-element tiles and up to 40 vector blocks.

`flashrt_npu_soc_version()` exposes the compiled target. Frontends compare it
with the active device before loading weights, matching the AMD backend's
architecture validation contract.
