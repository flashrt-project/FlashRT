# Apple Metal kernels

Apple follows the CUDA tree's operation-first organization (`gemm/`,
`attention/`, `conv/`, `kernels/`). Precision and implementation belong in
filenames, not in a model-stage bundle such as "vision" or "transformer".

```text
common/source.h                  Metal / TensorOps include preambles
gemm/bf16_tensor_ops.h            GEMM core, bias and strided-output epilogues
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

## Adding a precision or implementation

Add the implementation alongside its operation, for example
`gemm/fp8_e4m3_tensor_ops.h` or `attention/gqa_online_fp8_e4m3.h`. Add
quantization and scale-layout kernels under `quantize/` when they exist.
These are naming examples; no FP8 implementation is provided today.

Use a distinct registry entry and provider selection for a new numerical
contract. Specify the FP8 format, scale granularity/layout, accumulator and
output types, and rounding boundaries in the implementation and tests. A BF16
alias must not silently select an FP8 kernel. Shared arithmetic helpers belong
in `common/` only when there are actual consumers with the same semantics.

Native resource ownership, ICB recording/replay, the C ABI adapter and GPU
tests remain in [`cpp/apple/`](../../cpp/apple/README.md). The model traversal
stays in `flash_rt/models/`; the registry owns no model topology. The build
script and CMake both build `flashrt_apple_kernel_registry_smoke`, which
compiles every registered Metal entry, including aliases and diagnostics.
