// SPDX-License-Identifier: Apache-2.0
// G7.14 — Fused AWQ activation per-K scale + per-tensor static FP8
// e4m3 quantize. See header.

#include "awq_quant_fp8_static_bf16.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr float kFp8Max = 448.0f;

__global__ void awq_quant_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,        // (M, K) bf16 row-major
    const __nv_bfloat16* __restrict__ inv_s,     // (K,) bf16, broadcast over M
    __nv_fp8_e4m3*       __restrict__ out,       // (M, K) fp8
    const float*          __restrict__ act_scale_ptr,
    long long total,                              // M * K
    int K)
{
  const long long idx =
      (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int k = (int)(idx % (long long)K);
  const float v = __bfloat162float(in[idx]);
  const float s = __bfloat162float(inv_s[k]);
  const float inv_a = 1.0f / *act_scale_ptr;
  float q = v * s * inv_a;
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

__global__ void awq_quant2_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ in0,
    const __nv_bfloat16* __restrict__ inv_s0,
    __nv_fp8_e4m3* __restrict__ out0,
    const float* __restrict__ act_scale0,
    long long total0, int K0,
    const __nv_bfloat16* __restrict__ in1,
    const __nv_bfloat16* __restrict__ inv_s1,
    __nv_fp8_e4m3* __restrict__ out1,
    const float* __restrict__ act_scale1,
    long long total1, int K1)
{
  const long long idx =
      (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = total0 + total1;
  if (idx >= total) return;
  if (idx < total0) {
    const int k = (int)(idx % (long long)K0);
    float q = __bfloat162float(in0[idx]) *
              __bfloat162float(inv_s0[k]) *
              (1.0f / *act_scale0);
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    out0[idx] = __nv_fp8_e4m3(q);
  } else {
    const long long j = idx - total0;
    const int k = (int)(j % (long long)K1);
    float q = __bfloat162float(in1[j]) *
              __bfloat162float(inv_s1[k]) *
              (1.0f / *act_scale1);
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    out1[j] = __nv_fp8_e4m3(q);
  }
}

}  // namespace

void awq_quant_fp8_static_bf16(
    const void*  in_bf16,
    const void*  inv_s_bf16,
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
  awq_quant_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in_bf16),
      reinterpret_cast<const __nv_bfloat16*>(inv_s_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      total, K);
}

void awq_quant2_fp8_static_bf16(
    const void*  in0_bf16,
    const void*  inv_s0_bf16,
    void*        out0_fp8,
    const float* act_scale0,
    long long M0, int K0,
    const void*  in1_bf16,
    const void*  inv_s1_bf16,
    void*        out1_fp8,
    const float* act_scale1,
    long long M1, int K1,
    cudaStream_t stream)
{
  const long long total0 = M0 * (long long)K0;
  const long long total1 = M1 * (long long)K1;
  const long long total = total0 + total1;
  if (total <= 0) return;
  const int block_sz = 256;
  const unsigned grid =
      (unsigned)((total + block_sz - 1) / block_sz);
  awq_quant2_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in0_bf16),
      reinterpret_cast<const __nv_bfloat16*>(inv_s0_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out0_fp8),
      act_scale0, total0, K0,
      reinterpret_cast<const __nv_bfloat16*>(in1_bf16),
      reinterpret_cast<const __nv_bfloat16*>(inv_s1_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out1_fp8),
      act_scale1, total1, K1);
}

}  // namespace quantize
}  // namespace flash_rt
