// ================================================================
// FlashRT AMD — declarations for the FP16 backbone kernel port
// (kernels/elementwise_fp16.hip, kernels/norm_fp16.hip,
//  kernels/adaln_layer_norm.hip). Additive surface; the existing
// kernel files and their declarations in bindings.cpp are untouched.
// ================================================================
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// ── kernels/elementwise_fp16.hip ──
void add_bias_fp16(__half* x, const __half* b, int S, int D, hipStream_t stream);
void gelu_inplace_fp16(__half* x, int n, hipStream_t stream);
void silu_inplace_fp16(__half* x, int n, hipStream_t stream);
void relu_inplace_bf16(__hip_bfloat16* x, int n, hipStream_t stream);
void mul_fp16(const __half* a, const __half* b, __half* out, int n, hipStream_t stream);
void residual_add_fp16(__half* residual, const __half* x, int n, hipStream_t stream);
void gpu_fill_neginf_fp16(__half* dst, int n, hipStream_t stream);
void gpu_strided_copy_fp16(const __half* src, __half* dst,
                           int rows, int dst_cols, int src_stride, int col_offset,
                           hipStream_t stream);
void gpu_repeat_interleave_heads(const __half* src, __half* dst,
                                 int S, int NH_src, int HD, int repeat,
                                 hipStream_t stream);
void cast_fp16_to_bf16(const __half* in, __hip_bfloat16* out, int n, hipStream_t stream);
void cast_bf16_to_fp16(const __hip_bfloat16* in, __half* out, int n, hipStream_t stream);
void concat2_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b,
                  __hip_bfloat16* out, int rows, int cols_a, int cols_b,
                  hipStream_t stream);
void quantize_fp8_static_fp16(const __half* input, __hip_fp8_e4m3* output,
                              const float* d_scale, int n, hipStream_t stream);
void silu_mul_split_fp8_fp16(const __half* gate, const __half* up,
                             __hip_fp8_e4m3* out, int n,
                             const float* d_scale, hipStream_t stream);
int residual_add_fp16_vec(__half* residual, const __half* x, int n,
                          hipStream_t stream);
int quantize_fp8_static_fp16_vec(const __half* in, __hip_fp8_e4m3* out,
                                 const float* descale_ptr, int n,
                                 hipStream_t stream);
int gpu_repeat_interleave_heads_vec(const __half* src, __half* dst,
                                    int S, int NH_src, int HD, int repeat,
                                    hipStream_t stream);

// ── kernels/norm_fp16.hip ──
void layer_norm_fp16(const __half* x, const __half* weight,
                     const __half* bias, __half* out,
                     int seq_len, int dim, float eps, hipStream_t stream);
void rope_rotate_half_fp16(__half* x, const __half* cos_table,
                           const __half* sin_table,
                           int S, int NH, int HD, hipStream_t stream);
void rms_norm_fp8_fp16(const __half* x, const __half* weight,
                       __hip_fp8_e4m3* out, int seq_len, int dim, float eps,
                       const float* d_scale, hipStream_t stream);
void residual_add_rms_norm_fp8_fp16(__half* residual, const __half* x,
                                    const __half* weight, __hip_fp8_e4m3* out,
                                    int seq_len, int dim, float eps,
                                    const float* d_scale, hipStream_t stream);
int rms_norm_fp16_vec(const __half* x, const __half* w, __half* out,
                      int rows, int dim, float eps, hipStream_t stream);
int layer_norm_fp16_vec(const __half* x, const __half* w, const __half* b,
                        __half* out, int rows, int dim, float eps,
                        hipStream_t stream);
int layer_norm_fp8_static_fp16_vec(const __half* x, const __half* w,
                                   const __half* b, __hip_fp8_e4m3* out,
                                   const float* d_scale, int rows, int dim,
                                   float eps, hipStream_t stream);
int rope_rotate_half_fp16_vec(__half* x, const __half* cos_t,
                              const __half* sin_t, int S, int NH, int HD,
                              hipStream_t stream);

// ── kernels/adaln_layer_norm.hip ──
void layer_norm_no_affine_bf16(const __hip_bfloat16* x, __hip_bfloat16* out,
                               int seq_len, int dim, float eps,
                               hipStream_t stream);
void ada_layer_norm_bf16(const __hip_bfloat16* x,
                         const __hip_bfloat16* scale, const __hip_bfloat16* shift,
                         __hip_bfloat16* out, int seq_len, int dim, float eps,
                         hipStream_t stream);
void ada_layer_norm_fp8(const __hip_bfloat16* x,
                        const __hip_bfloat16* scale, const __hip_bfloat16* shift,
                        __hip_fp8_e4m3* out, const float* act_scale,
                        int seq_len, int dim, float eps,
                        hipStream_t stream);
void bias_gelu_quantize_fp8_static_bf16(const __hip_bfloat16* in,
                                        const __hip_bfloat16* bias,  // may be nullptr
                                        __hip_fp8_e4m3* out,
                                        const float* act_scale,
                                        long long M, int N,
                                        hipStream_t stream);
