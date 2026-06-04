// ================================================================
// FlashRT — Decoder fused kernel declarations (FP16)
// Port of pi05 ae_forward_static fused kernels.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>

// C1: AdaRMSNorm(x, style) → FP8 + gate
void fused_adarms_fp8_static_fp16(const __half* x, const __half* style,
                                    __nv_fp8_e4m3* out, __half* gate_out,
                                    int S, int D, const float* descale_ptr,
                                    cudaStream_t stream = 0);

// C4→C5: gate×residual + AdaRMSNorm → FP8 + gate
void gate_res_adarms_fp8_static_fp16(const __half* gemm_out, const __half* prev_gate,
                                       __half* residual, const __half* style,
                                       __nv_fp8_e4m3* fp8_out, __half* gate_out,
                                       int S, int D, const float* descale_ptr,
                                       cudaStream_t stream = 0);

// C6: Merged GeGLU → FP8 (GELU tanh approx, not SiLU)
void geglu_fp8_static_fp16(const __half* merged, __nv_fp8_e4m3* out,
                             int S, int H, const float* descale_ptr,
                             cudaStream_t stream = 0);

// C7 last layer: gate × residual (no norm)
void gate_res_fp16(const __half* gemm_out, const __half* gate,
                    __half* residual, int n, cudaStream_t stream = 0);

// Final step: AdaRMSNorm → FP16 output + gate
void adarms_fp16(const __half* x, const __half* style,
                  __half* out, __half* gate_out, int S, int D,
                  cudaStream_t stream = 0);

// Simple bias add: x[i] += b[i % D] (pi05 bias_k)
void add_bias_fp16(__half* x, const __half* b, int S, int D,
                    cudaStream_t stream = 0);

// cuBLAS NN GEMM: C = A @ B + beta * C (FP16, pi05 gmm)
// Stateless: receives cuBLAS handle from caller (FvkContext).
void gmm_fp16(cublasHandle_t handle,
               const __half* A, const __half* B, __half* C,
               int M, int N, int K, float beta,
               cudaStream_t stream = 0);
void gmm_fp16_alpha(cublasHandle_t handle,
                    const __half* A, const __half* B, __half* C,
                    int M, int N, int K, float alpha, float beta,
                    cudaStream_t stream = 0);
void gmm_fp16_out_fp32(cublasHandle_t handle,
                       const __half* A, const __half* B, float* C,
                       int M, int N, int K,
                       cudaStream_t stream = 0);
void action_update_from_fp32(const float* delta, const __half* bias,
                             __half* noise, int S, int D, float dt,
                             bool residual = true,
                             cudaStream_t stream = 0);

// FP8 GEMM with device descale → FP16 output (pi05 gmm_fp8_kn_descale)
// Exact match: col-major layout, A_SCALE=w_descale, B_SCALE=act_descale
void fp8_gemm_descale_fp16(const void* A_fp8, const void* B_fp8, void* C_fp16,
                             int M, int N, int K,
                             const float* act_descale, const float* w_descale,
                             cudaStream_t stream = 0);

// FP32 output variant — for models with activations exceeding FP16 range
void fp8_gemm_descale_f32out(const void* A_fp8, const void* B_fp8, void* C_fp32,
                              int M, int N, int K,
                              const float* act_descale, const float* w_descale,
                              cudaStream_t stream = 0);

// BF16 output variant — for models trained in BF16 with activations exceeding
// FP16 range (e.g., Pi0-FAST decode_step where hidden state reaches ~569K).
// FP8 inputs, BF16 accumulation in cuBLASLt, BF16 output.
void fp8_gemm_descale_bf16out(const void* A_fp8, const void* B_fp8, void* C_bf16,
                               int M, int N, int K,
                               const float* act_descale, const float* w_descale,
                               cudaStream_t stream = 0);
