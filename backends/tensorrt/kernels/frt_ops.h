// FlashRT edge kernels: model-free operators over raw device pointers.
//
// One C ABI for FlashRT's Thor (SM110) NVFP4 and FlashAttention-4 kernels, so
// a host can call them from its own plugin, runtime or graph. The FlashRT
// TensorRT plugins in ../plugins are thin shells over exactly these calls, and
// so are the pi0.5 stages in csrc/stages.
//
// Conventions:
//   - fp16 tensors are row-major and contiguous unless a stride is given;
//   - NVFP4 is e2m1 with a UE4M3 block scale every 16 elements along K:
//     packed weights are uint8 [N, K/2] (two nibbles per byte), block scales
//     are uint8 [N, K/16] holding the UE4M3 bit pattern;
//   - every call is asynchronous on `stream` and returns 0 on success.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// How the activation is normalized before it is quantized to NVFP4.
typedef enum {
    FRT_NORM_NONE = 0,       // quantize the input as it is
    FRT_NORM_RMS = 1,        // RMS normalization, optionally scaled by awq_inv_s
    FRT_NORM_LAYERNORM = 2,  // LayerNorm(gamma, beta), optionally scaled by awq_inv_s
} frt_norm_mode;

// What the GEMM epilogue does with the accumulator.
typedef enum {
    FRT_EPI_NONE = 0,      // out = A * B
    FRT_EPI_ACCUM = 1,     // out += A * B (residual already in out)
    FRT_EPI_BIAS_RES = 2,  // out = A * B + bias + residual
} frt_epilogue_mode;

// How the first GEMM of an MLP produces the second GEMM's NVFP4 input.
typedef enum {
    FRT_GATE_GEGLU_IL = 0,  // interleaved gate/up weights [2H, K], fused GeGLU
    FRT_GATE_BIAS_GELU = 1, // weights [H, K] with bias, fused GELU
} frt_gate_mode;

// ---------------------------------------------------------------- linear ---
// out[M, N] = epilogue(quantize(norm(x[M, K])) * w[N, K]^T)
//
// The activation quantization, the GEMM and the epilogue are the kernels
// FlashRT runs in production, in the same order.
size_t frt_nvfp4_linear_workspace(int32_t M, int32_t K);

int32_t frt_nvfp4_linear(const void* x,           // fp16 [M, K]
                         const void* norm_gamma,  // fp16 [K] or null
                         const void* norm_beta,   // fp16 [K] or null
                         const void* awq_inv_s,   // fp16 [K] or null
                         const void* w_packed,    // NVFP4 [N, K/2]
                         const void* w_sfb,       // UE4M3 [N, K/16]
                         const void* bias,        // fp16 [N] or null
                         const void* residual,    // fp16 [M, N] or null
                         void* out,               // fp16 [M, N]
                         int32_t M, int32_t N, int32_t K,
                         int32_t norm_mode,       // frt_norm_mode
                         int32_t epilogue,        // frt_epilogue_mode
                         int32_t variant,         // GEMM tile variant
                         float eps,               // LayerNorm epsilon
                         void* workspace, cudaStream_t stream);

// ------------------------------------------------------------------- mlp ---
// out[M, D] = epilogue(down(gate_act(gate_up(quantize(norm(x[M, D]))))))
//
// The gate/up GEMM writes its NVFP4 output and block scales straight from the
// epilogue, so the activation is never materialized in fp16.
size_t frt_nvfp4_mlp_workspace(int32_t M, int32_t D, int32_t H, int32_t gate_mode);

int32_t frt_nvfp4_mlp(const void* x,                // fp16 [M, D]
                      const void* norm_gamma,       // fp16 [D] or null
                      const void* norm_beta,        // fp16 [D] or null
                      const void* awq_inv_s,        // fp16 [D] or null
                      const void* gate_up_packed,   // NVFP4 [2H, D/2] or [H, D/2]
                      const void* gate_up_sfb,
                      const void* gate_up_bias,     // fp16 [H] or null
                      const void* down_packed,      // NVFP4 [D, H/2]
                      const void* down_sfb,
                      const void* down_bias,        // fp16 [D] or null
                      const void* residual,         // fp16 [M, D] or null
                      void* out,                    // fp16 [M, D]
                      int32_t M, int32_t D, int32_t H,
                      int32_t norm_mode,            // frt_norm_mode
                      int32_t gate_mode,            // frt_gate_mode
                      int32_t gate_variant,         // gate/up GEMM tile variant
                      int32_t down_variant,         // down GEMM tile variant
                      int32_t epilogue,             // frt_epilogue_mode
                      float eps,
                      void* workspace, cudaStream_t stream);

// ------------------------------------------------------------- attention ---
// FlashAttention-4 on the ahead-of-time CuTe DSL modules. Load once per
// process, never during CUDA graph capture.
int32_t frt_fa4_load(void);

// 1 / sqrt(head_dim), computed the way FlashRT's FA4 entry does (in double);
// the float-domain quotient differs by one ulp for some head dims.
float frt_fa4_default_scale(int32_t head_dim);

// Grouped-query attention, head_dim 256, one KV head, non-causal:
// q/o fp16 [1, sq, nh, 256], k/v fp16 [1, sk, 1, 256].
int32_t frt_fa4_gqa_hd256(const void* q, const void* k, const void* v, void* o,
                          int32_t sq, int32_t sk, int32_t nh, float scale,
                          cudaStream_t stream);

// Multi-head attention, head_dim 72, non-causal: fp16 [b, s, nh, 72]. Row
// strides are in elements and let q/k/v be views of one interleaved buffer.
int32_t frt_fa4_mha_hd72(const void* q, int64_t q_row_stride,
                         const void* k, int64_t k_row_stride,
                         const void* v, int64_t v_row_stride, void* o,
                         int32_t b, int32_t s, int32_t nh, float scale,
                         cudaStream_t stream);

// -------------------------------------------------------------- variants ---
// A `variant`, `gate_variant` or `down_variant` index names a tile shape in the
// table of the kernel that the mode and the epilogue select, so the tables are
// enumerated per kernel. `gate_variant` is ignored for FRT_GATE_GEGLU_IL, whose
// interleaved kernel has a single tile shape.
typedef enum {
    FRT_VARIANT_GEMM = 0,            // frt_nvfp4_linear, and the MLP down GEMM
                                     // under FRT_EPI_NONE or FRT_EPI_ACCUM
    FRT_VARIANT_GATE_BIAS_GELU = 1,  // the MLP gate/up GEMM under FRT_GATE_BIAS_GELU
    FRT_VARIANT_DOWN_BIAS_RES = 2,   // the MLP down GEMM under FRT_EPI_BIAS_RES
} frt_variant_kind;

// How many tile variants the kernel has, and a short description of one. The
// best tile depends on the shape, so a host that knows its own shapes can walk
// the table once and keep the index it measured.
int32_t frt_nvfp4_num_variants(int32_t kind);
const char* frt_nvfp4_variant_name(int32_t kind, int32_t idx);

// Programmatic dependent launch inside the GEMM kernels (default on).
void frt_set_pdl(int32_t on);

#ifdef __cplusplus
}  // extern "C"
#endif
