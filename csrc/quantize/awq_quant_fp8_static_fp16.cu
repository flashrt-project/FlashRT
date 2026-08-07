// SPDX-License-Identifier: Apache-2.0
//
// Fused AWQ activation per-K scale + per-tensor static FP8 e4m3 quantize
// for FP16 inputs (Chameleon-7B variant).
//
// Mirrors ``awq_quant_fp8_static_bf16`` but consumes FP16 activations,
// matching the Chameleon-7B residual-stream dtype. Pre-scales xn (the
// post-RMSNorm input to V_proj) by a per-input-channel SmoothQuant factor
// before per-tensor FP8 quantize:
//
//     out[m, k] = clip( in[m, k] * inv_s[k] / act_scale, ±448 )
//
// where ``inv_s`` is the SmoothQuant inverse-scale vector (FP16, length K)
// and ``act_scale`` is the per-tensor activation amax (1 fp32 device
// scalar). Outputs are FP8 E4M3, packed [M, K] row-major.
//
// Equivalent to the math
//     x' = x * inv_s            (per-K, broadcast over M)
//     w' = w * s                (per-K, broadcast over N) — folded offline
//     y  = x' @ w'^T == x @ w^T (mathematically)
// — but x' has a flatter per-K magnitude distribution, so the single
// per-tensor FP8 act_scale captures both small and large channels well.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr float kFp8Max = 448.0f;

__global__ void awq_quant_fp8_static_fp16_kernel(
    const __half*        __restrict__ in,            // (M, K) fp16
    const __half*        __restrict__ inv_s,         // (K,)   fp16
    __nv_fp8_e4m3*       __restrict__ out,           // (M, K) fp8
    const float*         __restrict__ act_scale_ptr, // 1 fp32 device scalar
    long long total,                                  // M * K
    int K)
{
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int k = (int)(idx % (long long)K);
  const float v       = __half2float(in[idx]);
  const float s       = __half2float(inv_s[k]);
  const float inv_a   = 1.0f / *act_scale_ptr;
  float q = v * s * inv_a;
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

}  // namespace

// Public entry — bound from csrc/bindings.cpp.
void awq_quant_fp8_static_fp16(
    const void*  in_fp16,
    const void*  inv_s_fp16,
    void*        out_fp8,
    const float* act_scale,
    long long M, int K,
    cudaStream_t stream)
{
  const long long total = M * (long long)K;
  if (total <= 0) return;
  const int block_sz = 256;
  const unsigned grid =
      (unsigned)((total + block_sz - 1) / block_sz);
  awq_quant_fp8_static_fp16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __half*>(in_fp16),
      reinterpret_cast<const __half*>(inv_s_fp16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      total, K);
}

}  // namespace quantize
}  // namespace flash_rt

// C-callable forward declaration consumed by csrc/bindings.cpp.
extern "C" void flash_rt_awq_quant_fp8_static_fp16(
    const void*  in_fp16,
    const void*  inv_s_fp16,
    void*        out_fp8,
    const float* act_scale,
    long long M, int K,
    cudaStream_t stream)
{
  flash_rt::quantize::awq_quant_fp8_static_fp16(
      in_fp16, inv_s_fp16, out_fp8, act_scale, M, K, stream);
}
