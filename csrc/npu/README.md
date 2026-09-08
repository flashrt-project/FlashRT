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

`flashrt_npu_rms_row_quant` fuses an optional residual addition, RMSNorm and
frozen row quantization for 2048-wide encoder rows. The residual addition is
rounded to BF16 before the FP32 variance reduction; the normalized value is
also rounded to BF16 before scaling and direct INT8 output. With no addition,
the residual input remains unchanged. Gamma is BF16 and inverse row scales
are FP32. The caller owns all storage and the completion boundary.

`flashrt_npu_encoder_rope` rotates eight BF16 query heads and one BF16 key
head together, with 256 channels per head. Input/output matrices are
contiguous `(rows,2048)` and `(rows,256)`. FP32 cosine/sine tables have 256
columns and cover all rows; each table repeats its first 128 columns in the
second half, as required by half-split RoPE. The kernel keeps products and
addition in FP32, then rounds once to BF16. It removes separate conversion
and rotary-output intermediates while retaining the original table precision.

`flashrt_npu_euler_update` accepts contiguous FP32 actions and BF16 biased
velocities, with a count divisible by 32 and a finite step in `(0,1]`. It
promotes velocity to FP32, multiplies by the step, then subtracts separately.
The biased GEMM's BF16 rounding precedes the update; there is no BF16 rounding
of the scaled velocity. The native symbol rejects invalid pointers and sizes.
