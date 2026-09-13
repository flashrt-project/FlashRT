// SPDX-License-Identifier: Apache-2.0
#pragma once

// Skinny FP8 GEMM family for small-row action decoders on sm_120a.
//
// The action expert of a flow-matching VLA runs its denoising steps over a
// handful of rows (the action chunk) while every step streams the full
// decoder weight set from HBM. cuBLASLt tiles that shape for reuse it does
// not have; these kernels stream each weight row once into mma fragments,
// split K across CTAs and hand the FP32 partial sums to a fused consumer
// (RoPE + cache write, gated residual + adaptive RMS norm + FP8 quantize,
// GeGLU + FP8 quantize). Every launch may use programmatic dependent launch
// so that a GEMM prefetches its weights while the previous kernel drains.
//
// Layouts: A e4m3 [M, K] row-major; W e4m3 [N, K] row-major (the "nk" FP8
// layout); partials fp32 [K / k_chunk][M][N]. Per-tensor scales are device
// pointers, applied by the consumers as alpha = (*a_scale) * (*w_scale).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace dec_skinny {

// Tile configurations (BN x KC x warps). Returns the number of configs.
int config_count();
// K chunk of a configuration (partials have K / k_chunk splits); 0 if invalid.
int config_k_chunk(int cfg);
// Whether (N, K) is divisible for the configuration.
bool config_supports(int cfg, int N, int K);

// partials[s] = A(M, Ks) @ W(N, Ks)^T. Rows beyond 16 are covered by a grid
// dimension (weights re-read through L2). Returns cudaError_t as int.
int gemm(const void* A, const void* W, float* partials, int M, int N, int K,
         int cfg, bool pdl, cudaStream_t stream);

// Same with a BF16 activation quantized on load: a = e4m3(clamp(x / *a_scale)).
int gemm_bf16_act(const __nv_bfloat16* A, const float* a_scale, const void* W,
                  float* partials, int M, int N, int K, int cfg, bool pdl,
                  cudaStream_t stream);

// QKV consumer: x = bf16(alpha * sum partials); Q gets RoPE, K gets RoPE and
// is written at cache row (pos + row), V is copied to (pos + row); pos comes
// from *devpos (may be null -> 0). Pair layout matches qkv_split_rope. Rows
// are grouped into samples of sample_rows (0 -> all rows one sample); sample
// b writes its cache rows at row b * kv_sample_stride + pos + r (row units).
int sum_rope(const float* partials, int splits, const float* a_scale,
             const float* w_scale, const __nv_bfloat16* rope,
             __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
             const int* devpos, int rows, int q_dim, int k_dim, int v_dim,
             int head_dim, int sample_rows, long long kv_sample_stride,
             bool pdl, cudaStream_t stream);

// Residual consumer: x = bf16(alpha * sum partials); residual += x * gate;
// then the adaptive RMS norm of the updated row with style (scale | shift |
// gate) and the norm weight; out is e4m3(clamp(normed / *out_scale)) when
// out_fp8 is given, else BF16 into out_bf16; gate_out receives the style gate.
int residual_ada_norm(const float* partials, int splits, const float* a_scale,
                      const float* w_scale, __nv_bfloat16* residual,
                      const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                      const __nv_bfloat16* style, __nv_fp8_e4m3* out_fp8,
                      __nv_bfloat16* out_bf16, const float* out_scale,
                      __nv_bfloat16* gate_out, int rows, int dim, float eps,
                      bool pdl, cudaStream_t stream);

// Residual-only consumer for the last layer: residual += bf16(alpha * sum) * gate.
int residual_gate_mul(const float* partials, int splits, const float* a_scale,
                      const float* w_scale, __nv_bfloat16* residual,
                      const __nv_bfloat16* gate, int rows, int dim, bool pdl,
                      cudaStream_t stream);

// GeGLU consumer over merged [gate | up] partials: out = e4m3(clamp(
// gelu_tanh(bf16(g)) * bf16(u) / *out_scale)).
int gate_gelu_fp8(const float* partials, int splits, const float* a_scale,
                  const float* w_scale, __nv_fp8_e4m3* out, int rows, int half,
                  const float* out_scale, bool pdl, cudaStream_t stream);

}  // namespace dec_skinny
}  // namespace flash_rt
