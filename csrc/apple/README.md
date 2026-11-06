# Apple Metal kernels

Apple follows the CUDA tree's operation-first organization (`gemm/`,
`attention/`, `conv/`, `kernels/`). Precision and implementation belong in
filenames, not in a model-stage bundle such as "vision" or "transformer".

```text
common/source.h                  Metal / TensorOps include preambles
gemm/bf16_tensor_ops.h            GEMM core, bias and strided-output epilogues
gemm/int8_tensor_ops.h            opt-in W8A8, INT32 accumulation, BF16 output
quantize/bf16_int8_rowwise.h      dynamic per-token scale, nearest-even packing
quantize/amax_channel_bf16.h      setup-only BF16 channel maxima for SmoothQuant
quantize/gelu_mul_int8_rowwise.h  BF16 GELU multiply + SmoothQuant row packing
attention/
  attention_d72_bf16.h            short-sequence TensorOps attention
  attention_simd_bf16.h           scalar/SIMD reference attention
  gqa_online_bf16.h               online-softmax GQA
  gqa_split_bf16.h                BF16 Q/K/V -> FP32 partials and statistics
  gqa_merge_fp32_bf16.h           FP32 partials -> BF16 output
conv/
  patch_pack_bf16.h               NHWC patch extraction
  patch_probe_bf16.h              diagnostic NCHW patch view
kernels/
  layer_norm_bf16.h
  rms_norm_bf16.h
  ada_rms_norm_bf16.h
  gelu_bf16.h
  gelu_mul_bf16.h
  bias_bf16.h
  residual_bf16.h
  rope_bf16.h
  position_add_bf16.h
kernel_registry.h                dispatch name -> source builder / entry point
```

Each header owns one operation or a closely related kernel family. Parameter
structs stay with the owning operation; `common/` contains only shared source
infrastructure. GEMM tile dimensions are explicit source-builder arguments,
used by both the runtime and tuning tool, rather than string replacements.

The registry contains no shader implementations. It preserves existing entry
names (including the BF16 `tr_*` aliases), explicitly selects the source and
specialization, and rejects unknown names. Runtime compilation sees only the
selected operation family. Headers embed MSL so the installed library needs
neither source files at runtime nor an offline Metal compiler.

The GQA sources also produce the additive `tr_gqa_masked` and
`tr_gqa_split_masked` entries. Their extended uniforms describe the valid
prefix and the fixed prefix-storage capacity. Keys in the intervening padding
window receive zero softmax weight; suffix keys remain visible. Entirely masked
split tiles emit zero numerators/denominators and a negative-infinity maximum
for the existing merge operation. The original compact entries retain their
four-field parameter layout. RoPE already accepts its position as a uniform.
The Python provider can bind bounded, mutable uint32 uniforms without changing
existing kernel parameter layouts. Compact replay also uses `tr_rope_offsets`
to write KV at a uniform-controlled element offset; buffer bindings stay fixed.
The ICB remains unchanged. Whole surplus threadgroups return before GEMM,
RMSNorm and GQA work; every per-token producer uses the actual row extent.
Weight packing, buffer allocation and compilation stay in setup.

## Adding a precision or implementation

Add the implementation alongside its operation, as in
`gemm/int8_tensor_ops.h`. Quantization and scale-layout kernels belong under
`quantize/`.

Use a distinct registry entry and provider selection for a new numerical
contract. Specify the numerical format, scale granularity/layout, accumulator and
output types, and rounding boundaries in the implementation and tests. A BF16
alias must not silently select a quantized kernel. Shared arithmetic helpers belong
in `common/` only when there are actual consumers with the same semantics.

Native resource ownership, ICB recording/replay, the C ABI adapter and GPU
tests remain in [`cpp/apple/`](../../cpp/apple/README.md). The model traversal
stays in `flash_rt/models/`; the registry owns no model topology. The build
script and CMake both build `flashrt_apple_kernel_registry_smoke`, which
compiles every registered Metal entry, including aliases and diagnostics.

The INT8 experiment adds fourteen distinct entries only when
`FLASHRT_BUILD_APPLE_INT8_EXPERIMENT=ON`. Weights use one FP32 scale per output
channel; activations use a dynamic FP32 scale per token. Both use symmetric
signed bytes in [-127,127]. NT operands have zero-padded K rows aligned to
128 bytes. GEMM uses INT32 accumulation and FP32 descale, then BF16 rounding;
bias follows a second BF16 rounding boundary. The provider guards INT32
indexing and accumulation overflow. Nonfinite activation rows propagate a NaN
scale, and nonfinite weights fail before recording.
For M <= 16 the pack kernel uses one 128-thread group per row; larger M uses
four independent SIMD rows per group. This avoids assigning only three groups
to the usual ten-token decoder. Both paths use the same quantization rule.

Optional SmoothQuant multiplies each activation channel by a persistent FP32
inverse factor inside both the amax and packing passes, before row quantization.
Its matching weight factor is applied once before weight packing. BF16 producer
values and their other consumers remain unchanged. Calibration uses the
same model traversal with a setup-only per-input-channel observer; production
replay contains no observer dispatches. No additional BF16 rounding is inserted
between channel scaling and INT8 packing.

The provider selects measured 64x64/2-SIMD and 64x128/4-SIMD GEMM tiles for
large-row projection geometries, retaining the original tile elsewhere.
For an immediately adjacent GELU-multiply producer with M > 16 and K >= 4096,
SmoothQuant packing can fuse the producer and row maximum pass. The fused
kernel uses one 512-thread group per row. It writes the same BF16 hidden tensor
before channel scaling, then reads it to pack INT8, preserving both the rounding
boundary and later consumers. A shared BF16 value helper keeps the nonlinear
expression identical. Setup lookahead flushes on other dispatches, copies,
stage changes or plan finish; replay contains only fixed native dispatches.
