# Standalone Ascend kernels

Build with `bash scripts/npu/build.sh` after activating CANN. This subtree
has no CUDA, HIP, PyTorch C++ extension, or root CMake dependency.

Which units enter the build is chosen per model, because a shared object that no
selected model loads has no business being compiled:
`FLASHRT_ENABLE_NPU_PI05` (default `ON`) covers everything below except the last
section, and `FLASHRT_ENABLE_NPU_GROOT_N17` (default `OFF`) covers that one.

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

`flashrt_npu_image_patches(stream, raw, indices, lut, output, views)` accepts
one to three contiguous uint8 HWC224 images and writes FP32 `(views,256,588)`
patch tokens. The FP32 LUT has 256 entries preserving the original
normalization and BF16 rounding. The 608 int32 indices encode byte offsets
from padded 14-row HWC tiles into CHW patches; the last 20 entries are zero.
Setup creates and retains both tables. The native call checks pointer and
view-count arguments, then copies only the 588 valid output values per patch.

`libflashrt_npu_cube.so` provides the standalone 910B INT8 gate/up path.
Weights are packed as adjacent 512-channel gate/up blocks during setup.
The mixed kernel uses 128×1024 tiles, overlapping Cube INT32 accumulation
with Vector dequantization, BF16 rounding, GELU product and frozen row-scale
INT8 quantization. The intermediate is a bounded 20 MiB GM double buffer;
this is not a direct Cube-to-UB transfer or a claim of measured L2 residency.
The runtime C2C control address is initialized before cross-core events.

Each captured Pi0.5 runner owns its workspace and shares it only between
its sequential encoder layers. Packed weights and calibration scales may
be shared by runners. The down projection remains a separate INT8 GEMM.
Calibration uses the existing real-observation statistics and precision
specification; fusion does not introduce another scale-selection method.
Both native libraries are built by `scripts/npu/build.sh` without CUDA,
HIP, or a PyTorch C++ extension dependency. `FLASHRT_NPU_CUBE_LIBRARY`
can select an alternate mixed-kernel library during setup.

## GR00T N1.7 action head and image path

Three units, built only with `FLASHRT_ENABLE_NPU_GROOT_N17=ON`, and named for
the operation rather than the model in keeping with the rest of this subtree.
Their Python wrappers live under `flash_rt/npu/models/groot_n17/`, because it is
only that model's geometries that have been validated; every one of them refuses
anything outside its envelope with a code rather than returning an undefined
result. See `docs/deployment_npu_groot_n17.md`.

`flashrt_npu_dit_attn(stream, q, k, vt, out, scores, probs, ctx, heads, sq, skv,
hd, cores, stride, vstride, scale)` — multi-head attention on the raw `Mmad`
path, for a short query against a short key sequence. One head per iteration and
the whole head in L0 untiled, which is what the geometry refusals protect: L0C,
L0A, L0B and the UB plane are each checked against the requested extents, as are
a 16-aligned head width and a query count the plane-wise softmax can hold. `q`
and `k` may be column slices of one wider projection (`stride` is their row
pitch). `vt` is the value either transposed to `(heads*hd, skv)` with
`vstride = 0`, or in the layout the projection wrote with `vstride` as its row
pitch, in which case the load from L1 to L0B transposes it. Every operand is
padded to the fractal with real zeros by the caller: `Nd2Nz` fills only the rows
it is given, so a tail the kernel assumed were zero would be uninitialised L1
that the `Mmad` then reads. Scores, probabilities and the FP32 context are
caller-owned scratch, written and consumed inside one launch.

`flashrt_npu_dit_add_layer_norm(stream, residual, branch, gamma, beta, norm,
total, branchBias, rows, cols, pitch, eps)` — `residual + branch`, and the
affine LayerNorm of that sum, in one launch. The sum is rounded to BF16 before it
is normalised, because the sum is what the next block carries forward, and the
normalisation then runs in FP32 from it. `branchBias` is optional and is added to
the branch and rounded there, which is where the biased matmul it replaces
rounded. `cols` must be a whole number of 64-element repeats and at most 2048;
`norm` is written at `pitch`, which must be at least `cols` and a multiple of 16
past it, so its row can end in a constant the next projection reads as a bias
channel.

`flashrt_npu_area_resize(stream, src, dst, offsets, weights, rows, rowWeights,
images, srcPlane, srcRowStride, rowOffset, srcSamples, dstPlane, dstRowStride,
dstHeight, dstSamples, tableStride, cores)` — one resize step of OpenCV's
`INTER_AREA` **enlarging** path on uint8, bit for bit. Shrinking is a different
branch inside OpenCV and is refused on the host rather than approximated here.
The tap tables are built on the host and each is padded to a 32-byte block,
because a copy out of global memory has to start on one. Row strides must be
whole 32-byte blocks. The horizontal pass is exact in FP32; the vertical pass is
the 8-bit specialisation, which truncates three separate times, and rounding its
accumulator once instead is off by one least significant bit on about eight
percent of the samples.
