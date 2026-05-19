// ================================================================
// FlashRT — Normalization kernel declarations
// RMSNorm, LayerNorm, AdaRMSNorm
// Dual dtype: BF16 (__nv_bfloat16) + FP16 (__half)
// ================================================================
#pragma once

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
