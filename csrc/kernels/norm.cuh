// ================================================================
// FlashRT — Normalization kernel declarations
// RMSNorm, LayerNorm, AdaRMSNorm
// Dual dtype: BF16 (__nv_bfloat16) + FP16 (__half)
// ================================================================
#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── RMSNorm ──

void rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
              __nv_bfloat16* out, int seq_len, int dim, float eps,
              cudaStream_t stream = 0);

void bias_rms_norm_bf16(const __nv_bfloat16* x,
                        const __nv_bfloat16* bias,
                        const __nv_bfloat16* weight,
                        __nv_bfloat16* out,
                        int seq_len, int dim, float eps,
                        cudaStream_t stream = 0);

void rms_norm_fp16(const __half* x, const __half* weight,
                    __half* out, int seq_len, int dim, float eps,
                    cudaStream_t stream = 0);

void rms_norm_inplace(const __nv_bfloat16* weight,
                      __nv_bfloat16* x, int seq_len, int dim, float eps,
                      cudaStream_t stream = 0);

// ── LayerNorm ──

void layer_norm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                const __nv_bfloat16* bias, __nv_bfloat16* out,
                int seq_len, int dim, float eps, cudaStream_t stream = 0);

void layer_norm_fp16(const __half* x, const __half* weight,
                      const __half* bias, __half* out,
                      int seq_len, int dim, float eps, cudaStream_t stream = 0);

// ── LayerNorm → FP8 (fused, no scale — direct cast) ──

void layer_norm_fp8(const __half* x, __nv_fp8_e4m3* out,
                     const __half* gamma, const __half* beta,
                     int seq_len, int dim, float eps, cudaStream_t stream = 0);

void layer_norm_fp8_bf16(const __nv_bfloat16* x, __nv_fp8_e4m3* out,
                          const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
                          int seq_len, int dim, float eps, cudaStream_t stream = 0);

void awq_layer_norm_fp8_bf16(const __nv_bfloat16* x,
                             __nv_fp8_e4m3* out,
                             const __nv_bfloat16* gamma,
                             const __nv_bfloat16* beta,
                             const __nv_bfloat16* inv_s,
                             const float* d_scale,
                             int seq_len, int dim, float eps,
                             cudaStream_t stream = 0);

// ── AdaRMSNorm + Style ──

void ada_rms_norm_style(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                        const __nv_bfloat16* style,
                        __nv_bfloat16* out, __nv_bfloat16* gate_out,
                        int seq_len, int dim, float eps,
                        cudaStream_t stream = 0);

void ada_rms_norm_style_fp16(const __half* x, const __half* weight,
                             const __half* style,
                             __half* out, __half* gate_out,
                             int seq_len, int dim, float eps,
                             cudaStream_t stream = 0);

// ── Fused Norm → FP8 (with scale) ──

void rms_norm_fp8_fp16(const __half* x, const __half* weight,
                        __nv_fp8_e4m3* out, int seq_len, int dim, float eps,
                        const float* d_scale, cudaStream_t stream = 0);

void rms_norm_fp8(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                   __nv_fp8_e4m3* out, int seq_len, int dim, float eps,
                   const float* d_scale, cudaStream_t stream = 0);

void ada_rms_norm_style_fp8(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                             const __nv_bfloat16* style,
                             __nv_fp8_e4m3* out, __nv_bfloat16* gate_out,
                             int seq_len, int dim, float eps,
                             const float* d_scale, cudaStream_t stream = 0);

// ── Fused Residual + Norm ──

void residual_add_rms_norm_fp8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                const __nv_bfloat16* weight, __nv_fp8_e4m3* out,
                                int seq_len, int dim, float eps,
                                const float* d_scale, cudaStream_t stream = 0);

void residual_add_rms_norm_fp8_fp16(__half* residual, const __half* x,
                                     const __half* weight, __nv_fp8_e4m3* out,
                                     int seq_len, int dim, float eps,
                                     const float* d_scale, cudaStream_t stream = 0);

void residual_add_rms_norm(__nv_bfloat16* residual, const __nv_bfloat16* x,
                            const __nv_bfloat16* weight, __nv_bfloat16* out,
                            int seq_len, int dim, float eps,
                            cudaStream_t stream = 0);

// ── Scaled kernels without affine weight (norm weight baked into GEMM weights) ──

// RMSNorm → FP8 (no weight, with d_scale). For models with baked norm weight.
void rms_norm_fp8_noweight_fp16(const __half* x, __nv_fp8_e4m3* out,
                                 int seq_len, int dim,
                                 const float* d_scale, cudaStream_t stream = 0);

// Residual + RMSNorm → FP8 (no weight, with d_scale).
void residual_add_rms_norm_fp8_noweight_fp16(__half* residual, const __half* x,
                                               __nv_fp8_e4m3* out,
                                               int seq_len, int dim,
                                               const float* d_scale,
                                               cudaStream_t stream = 0);

// BF16 noweight variants — for models with activations exceeding FP16 range
void rms_norm_fp8_noweight_bf16(const __nv_bfloat16* x, __nv_fp8_e4m3* out,
                                 int seq_len, int dim,
                                 const float* d_scale, cudaStream_t stream = 0);

void residual_add_rms_norm_fp8_noweight_bf16(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                               __nv_fp8_e4m3* out,
                                               int seq_len, int dim,
                                               const float* d_scale,
                                               cudaStream_t stream = 0);

// ── Production-exact kernels (no weight, no scale) ──

// RMSNorm → FP8 (no weight, no d_scale). Matches pi05 fused_rms_fp8.
void plain_rms_fp8_fp16(const __half* x, __nv_fp8_e4m3* out,
                         int seq_len, int dim, cudaStream_t stream = 0);

// Residual add + RMSNorm → FP8 (no weight, no d_scale). Matches pi05 res_rms_fp8_k.
void plain_res_rms_fp8_fp16(__half* residual, const __half* x,
                             __nv_fp8_e4m3* out, int seq_len, int dim,
                             cudaStream_t stream = 0);

// Cast FP16 → FP8 (no scale). Matches pi05 cast_fp16_fp8_k.
void cast_fp16_fp8(const __half* input, __nv_fp8_e4m3* output,
                    int n, cudaStream_t stream = 0);

// ---- Public Orin/INT8 helper declarations ----
void ada_rms_norm_style_int8(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                             const __nv_bfloat16* style,
                             int8_t* out, __nv_bfloat16* gate_out,
                             int seq_len, int dim, float eps,
                             float* d_scales, cudaStream_t stream = 0);
void avg_pool_vision_tokens(
        const __nv_bfloat16* x, __nv_bfloat16* out,
        int nv, int H, int W, int dim, int pool_factor,
        cudaStream_t stream = 0);
void rms_norm_int8_rowwise(const __nv_bfloat16* x,
                            const __nv_bfloat16* weight,
                            int8_t* out, float* scales,
                            int seq_len, int dim, float eps,
                            cudaStream_t stream = 0);
void residual_add_rms_norm_int8_rowwise(
        __nv_bfloat16* residual, const __nv_bfloat16* x,
        const __nv_bfloat16* weight,
        int8_t* out, float* scales,
        int seq_len, int dim, float eps,
        cudaStream_t stream = 0);
void bias_residual_layer_norm_bf16(
        __nv_bfloat16* residual, const __nv_bfloat16* x,
        const __nv_bfloat16* bias_pre,
        const __nv_bfloat16* ln_weight, const __nv_bfloat16* ln_bias,
        __nv_bfloat16* out, int seq_len, int dim, float eps,
        cudaStream_t stream = 0);

void bias_residual_layer_norm_fp16(
        __half* residual, const __half* x,
        const __half* bias_pre,
        const __half* ln_weight, const __half* ln_bias,
        __half* out, int seq_len, int dim, float eps,
        cudaStream_t stream = 0);

void rms_norm_amax_fp16(const __half* x, const __half* weight,
                         __half* out, float* d_amax,
                         int seq_len, int dim, float eps,
                         cudaStream_t stream = 0);
void residual_add_rms_norm_amax_fp16(
        __half* residual, const __half* x, const __half* weight,
        __half* xn_out, float* d_amax,
        int seq_len, int dim, float eps,
        cudaStream_t stream = 0);
void residual_add_rms_norm_fp16(__half* residual, const __half* x,
                                 const __half* weight, __half* out,
                                 int seq_len, int dim, float eps,
                                 cudaStream_t stream = 0);
#ifdef FLASHRT_ENABLE_CHAMELEON
void rms_norm_int8_rowwise_fp16(const __half* x,
                                 const __half* weight,
                                 int8_t* out, float* scales,
                                 int seq_len, int dim, float eps,
                                 cudaStream_t stream = 0);
void residual_add_rms_norm_int8_rowwise_fp16(
        __half* residual, const __half* x,
        const __half* weight,
        int8_t* out, float* scales,
        int seq_len, int dim, float eps,
        cudaStream_t stream = 0);
#endif  // FLASHRT_ENABLE_CHAMELEON

// ── Gemma3 RMSNorm ──
// y = bf16( fp32(x) * rsqrt(mean(fp32(x)^2) + eps) * (1 + fp32(w)) ), with the
// (1 + w) formed in fp32 per element; pre-rounding it into the weight is not
// equivalent.
//
// Rows may be strided, which is what lets a per-head q/k norm run in place on a
// fused QKV buffer. Row r starts at
//   (r / rows_per_group) * group_stride + (r % rows_per_group) * x_stride
// For plain contiguous rows pass rows_per_group = rows with group_stride = 0.
// Bear in mind that rows_per_group = 1 does not mean "no grouping": r % 1 is
// always 0, so every row would read row 0 and the call would quietly compute
// one row's norm into all of them. Use group_stride = x_stride for one row per
// group instead.
//
// `residual`, when non-null, is updated as bf16(residual + bf16(norm(x))) to
// match HF's two roundings; `out` may be null when only the residual is wanted.
void gemma3_rms_norm_bf16(const __nv_bfloat16* x, long long x_stride,
                          const __nv_bfloat16* weight,
                          __nv_bfloat16* out, long long out_stride,
                          __nv_bfloat16* residual, long long res_stride,
                          int rows, int rows_per_group, long long group_stride,
                          int dim, float eps, cudaStream_t stream = 0);

// The same norm, written straight out as a static per-tensor FP8 (e4m3) GEMM
// input: round to bf16, then clamp(x / a_scale) to e4m3, i.e. byte-identical to
// a standalone quantize_fp8_static over the bf16 norm output. One block per
// row, dim / 8 threads; dim must be a multiple of 8.
//
// This uses a different summation order from gemma3_rms_norm_bf16 (the
// vectorized form's block reduction rather than the strided loop), so the two
// are not bit-comparable with each other.
// The same norm over contiguous rows with 8 columns per thread: same formula,
// different summation order from gemma3_rms_norm_bf16, so the two are not
// bit-comparable with each other. dim must be a multiple of 8; `residual`, when
// non-null, is updated with HF's two roundings and `out` may then be null.
// Gemma3 RMSNorm then a gated residual add, 8 columns per thread:
//   residual = bf16( residual + bf16( bf16(gemma3_norm(x)) * gate ) )
// Three roundings, one per store HF performs. Distinct from gate_mul_residual,
// which takes x directly with a single rounding and has no norm. Contiguous
// rows, one block per row; dim must be a multiple of 8. `w` and `gate` are
// per-column (dim-wide, shared by every row); `x` and `residual` are (rows, dim).
// gemma3_norm_gate_residual_v8_bf16 followed by an adaLN modulation of the
// updated residual, in one pass. `mod` is 2*dim (scale | shift); the gate is a
// separate argument. Fused because the reference computes the tail's own
// RMSNorm with the first half's reduction and block shape.
void gemma3_norm_gate_residual_ada_v8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                           const __nv_bfloat16* gate, __nv_bfloat16* residual,
                                           const __nv_bfloat16* mod, __nv_bfloat16* out,
                                           int rows, int dim, float eps, cudaStream_t stream = 0);

void gemma3_norm_gate_residual_v8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                       const __nv_bfloat16* gate, __nv_bfloat16* residual,
                                       int rows, int dim, float eps, cudaStream_t stream = 0);

// Gemma3 RMSNorm written straight into a clipped static NVFP4 GEMM input:
// e2m1 per 16-element group with UE4M3 block scales in the CUTLASS sm120
// swizzled layout, the group amax clamped to cap (null or <= 0 disables the
// clip). Same encoding as quantize_bf16_to_nvfp4_swizzled_clipped. dim must be
// a multiple of 16 with dim/8 <= 1024.
void gemma3_rms_norm_nvfp4_swizzled(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                    const float* cap, uint8_t* packed, uint8_t* sf_swz,
                                    int rows, int dim, float eps, cudaStream_t stream = 0);
long long gemma3_rms_norm_nvfp4_sf_bytes(int rows, int dim);

void gemma3_rms_norm_v8_bf16(const __nv_bfloat16* x, long long x_stride,
                             const __nv_bfloat16* weight,
                             __nv_bfloat16* out, long long out_stride,
                             __nv_bfloat16* residual, long long res_stride,
                             int rows, int rows_per_group, long long group_stride,
                             int dim, float eps, cudaStream_t stream = 0);

void gemma3_rms_norm_fp8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                              __nv_fp8_e4m3* out, const float* a_scale,
                              int rows, int dim, float eps, cudaStream_t stream = 0);

// residual add + LayerNorm + static FP8 quantize in one pass. Bit-identical to the
// reference kernel; the composition it replaces is also bit-identical but costs three
// launches and materialises an xn the reference never writes. y == 0 skips the
// residual, xn == 0 skips the normalised store; dim/2 must be <= 1024.
void res_ln_quant_v8_bf16(__nv_bfloat16* r, const __nv_bfloat16* y, const __nv_bfloat16* b,
                          const __nv_bfloat16* w, const __nv_bfloat16* beta, __nv_bfloat16* xn,
                          __nv_fp8_e4m3* q, const float* scale, int rows, int dim, float eps,
                          cudaStream_t stream = 0);

// (1+w) RMSNorm + HF rotate-half RoPE in one pass over q and the cached k rows,
// head_dim 256 (DM05). Bit-identical to the two-norm + two-rope composition but
// one launch instead of four.
void gemma3_qk_norm_rope_v8_bf16(__nv_bfloat16* q, long long q_stride, int nhq,
                                 __nv_bfloat16* k, long long k_stride, int nhk,
                                 const __nv_bfloat16* wq, const __nv_bfloat16* wk,
                                 const __nv_bfloat16* cosT, const __nv_bfloat16* sinT,
                                 int seq, float eps, cudaStream_t stream = 0);

// ── Adaptive RMSNorm with one modulation shared by every row ──
// y = bf16( bf16(x) * rsqrt(mean(fp32(x)^2) + eps) * (1 + scale) + shift ),
// where `mod` is one 2*dim bf16 vector for all rows: scale at mod[0:dim],
// shift at mod[dim:2*dim]. No weight and no gate output.
//
// Distinct from ada_rms_norm_style, which takes a per-row 3*dim style, applies
// a per-channel weight and writes a gate; the two also reduce the sum of
// squares in a different order.
void ada_rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* mod,
                  __nv_bfloat16* out, int rows, int dim, float eps,
                  cudaStream_t stream = 0);
