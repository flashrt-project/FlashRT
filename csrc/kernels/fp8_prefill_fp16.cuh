// SPDX-License-Identifier: Apache-2.0
#pragma once
// Fused producers / consumers around per-tensor FP8 (e4m3) GEMMs on an FP16
// residual stream (prefill-shaped transformer layers: vision towers, VLM
// prefixes). Each kernel replaces a chain of bias / residual / norm /
// activation / quantize launches; quantization is e4m3(y / *scale) saturated
// to +-448 (the static-scale convention of quantize_fp8_static_fp16).
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp8_prefill {

// Residual update and row norm in one pass, one CTA per row:
//   x += delta + delta_bias + delta2             (each term nullable, FP16 in place)
//   x_copy = x                                   (nullable)
//   y = norm(x) * weight + bias                  (mode 0 LayerNorm, 1 RMSNorm; weight / bias nullable)
//   out_fp8 = e4m3(y / *scale)                   (nullable)
//   out_f16 = y                                  (nullable)
// dim % 8 == 0, dim <= 8192.
int residual_norm_quant(__half* x, const __half* delta, const __half* delta_bias, const __half* delta2,
                        __half* x_copy, const __half* weight, const __half* bias, int mode, float eps,
                        __nv_fp8_e4m3* out_fp8, const float* scale, __half* out_f16, int rows, int dim,
                        cudaStream_t stream);

// out = e4m3(gelu_tanh(x + bias) / *scale) for x [rows, n] (row stride ld).
// bias nullable; n % 8 == 0, ld % 8 == 0.
int bias_gelu_quant(const __half* x, int ld, const __half* bias, __nv_fp8_e4m3* out, const float* scale, int rows,
                    int n, cudaStream_t stream);

// Attention operand split for one projection row s: Q [S, n_q, hd], K / V
// [S, n_k, hd] from three FP16 sources (row stride ld; a fused [S, ld] QKV
// projection passes q, q + n_q*hd, q + (n_q+n_k)*hd). Per head: + bias
// (nullable), * k_mul / v_mul on K / V (weight-scale ratios when the
// projections share one GEMM scale), per-head RMSNorm (q_norm / k_norm
// nullable), rotate-half RoPE with FP16 [S, hd] tables (cos / sin nullable;
// first half of each row used). Outputs are FP16, or BF16 with out_bf16; they
// may alias the sources element for element. hd in {64, 128}.
int qkv_split(const __half* q_in, const __half* k_in, const __half* v_in, int ld, const __half* q_bias,
              const __half* k_bias, const __half* v_bias, float k_mul, float v_mul, const __half* q_norm,
              const __half* k_norm, float eps, const __half* cos, const __half* sin, void* q_out, void* k_out,
              void* v_out, bool out_bf16, int S, int n_q, int n_k, int hd, cudaStream_t stream);

}  // namespace fp8_prefill
}  // namespace flash_rt
