// SPDX-License-Identifier: Apache-2.0
// G7.10 — Fused bias + GELU(tanh) + FP8 e4m3 quantize. See header.

#include "bias_gelu_quantize_fp8.cuh"

#include <cstdint>
#include <stdexcept>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float gelu_tanh(float x) {
  // Matches existing csrc/kernels/activation.cu:gelu_kernel (tanh approx).
  // 0.7978845608f = sqrt(2/pi)
  return 0.5f * x * (1.0f + tanhf(
      0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_quantize_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ bias,   // may be nullptr
    __nv_fp8_e4m3*       __restrict__ out,
    const float*          __restrict__ act_scale_ptr,
    long long total,                          // M * N
    int N,
    int has_bias)
{
  const long long idx =
      (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  float v = __bfloat162float(in[idx]);
  if (has_bias) {
    int n = (int)(idx % N);
    v += __bfloat162float(bias[n]);
  }
  float g = gelu_tanh(v);
  float inv_s = 1.0f / *act_scale_ptr;
  float q = g * inv_s;
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

// Same epilogue, but the activated value is rounded to bf16 before the scale and
// the e4m3 cast.
//
// That extra rounding is the whole difference, and it is not cosmetic: it is
// what running the two kernels this fuses -- `bias_gelu_bf16`, which stores a
// bf16 activation, then `quantize_fp8_static` over it -- produces. A caller
// calibrated against that pair therefore rounds here and not in the sibling
// above, and the two disagree on the elements sitting near an e4m3 boundary
// (measured: ~2% of a 2048x4304 activation). Kept as a separate entry rather
// than a flag on the sibling so that neither caller's behaviour moves.
__global__ void bias_gelu_quantize_fp8_static_bf16_rounded_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ bias,   // may be nullptr
    __nv_fp8_e4m3*       __restrict__ out,
    const float*          __restrict__ act_scale_ptr,
    long long total,
    int N,
    int has_bias)
{
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  float v = __bfloat162float(in[idx]);
  if (has_bias) {
    int n = (int)(idx % N);
    v += __bfloat162float(bias[n]);
  }
  // the intermediate bf16 round the unfused pair performs
  float g = __bfloat162float(__float2bfloat16(gelu_tanh(v)));
  float inv_s = 1.0f / *act_scale_ptr;
  float q = g * inv_s;
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  out[idx] = __nv_fp8_e4m3(q);
}

}  // namespace

void bias_gelu_quantize_fp8_static_bf16_rounded(
    const __nv_bfloat16* in, const __nv_bfloat16* bias, __nv_fp8_e4m3* out,
    const float* act_scale_ptr, long long M, int N, cudaStream_t stream);

void bias_gelu_quantize_fp8_static_bf16(
    const void*  in_bf16,
    const void*  bias_bf16,
    void*        out_fp8,
    const float* act_scale,
    long long M, int N,
    cudaStream_t stream)
{
  const long long total = M * (long long)N;
  const int block_sz = 256;
  const unsigned grid =
      (unsigned)((total + block_sz - 1) / block_sz);
  const int has_bias = (bias_bf16 != nullptr) ? 1 : 0;
  bias_gelu_quantize_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      total, N, has_bias);
}

void bias_gelu_quantize_fp8_static_bf16_rounded(
    const __nv_bfloat16* in, const __nv_bfloat16* bias, __nv_fp8_e4m3* out,
    const float* act_scale_ptr, long long M, int N, cudaStream_t stream)
{
  if (M <= 0 || N <= 0) return;
  const long long total = M * N;
  const int block = 256;
  const int grid = (int)((total + block - 1) / block);
  bias_gelu_quantize_fp8_static_bf16_rounded_kernel<<<grid, block, 0, stream>>>(
      in, bias, out, act_scale_ptr, total, N, bias != nullptr ? 1 : 0);
}

// Same arithmetic as bias_gelu_quantize_fp8_static_bf16_rounded -- the identical
// per-element expression, so the two agree bit for bit (measured 0 of 8814592 at
// 2048x4304) -- but 8 columns per thread with a 16-byte load and an 8-byte store
// instead of one 2-byte access per thread.
//
// The sibling above is correct but not competitive at the shapes DM05 asks for:
// measured at 2048x4304 it reaches 306 GB/s against 782 GB/s for the reference
// kernel it stands in for, a factor of 2.56, and the vision tower calls it 27 times
// per predict. The arithmetic is kept verbatim rather than re-derived so a caller
// already calibrated against the rounded sibling sees no change in its output.
__global__ void bias_gelu_quantize_fp8_static_bf16_rounded_v8_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ bias,
    __nv_fp8_e4m3*       __restrict__ out,
    const float*          __restrict__ act_scale_ptr,
    long long rows,
    int N)
{
  union BiasGeluV8 { uint4 u; __nv_bfloat16 h[8]; };
  const int vpr = N / 8;
  const long long n = rows * (long long)vpr;
  const float inv_s = 1.0f / *act_scale_ptr;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) {
    const int c = (int)(i % vpr) * 8;
    BiasGeluV8 hv, bv;
    hv.u = reinterpret_cast<const uint4*>(in)[i];
    bv.u = *reinterpret_cast<const uint4*>(bias + c);
    __align__(8) __nv_fp8_e4m3 q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = __bfloat162float(hv.h[j]) + __bfloat162float(bv.h[j]);
      const float g = __bfloat162float(__float2bfloat16(gelu_tanh(v)));
      const float t = g * inv_s;
      q[j] = __nv_fp8_e4m3(fminf(fmaxf(t, -kFp8Max), kFp8Max));
    }
    reinterpret_cast<uint2*>(out)[i] = *reinterpret_cast<uint2*>(q);
  }
}

void bias_gelu_quantize_fp8_static_bf16_rounded_v8(
    const __nv_bfloat16* in, const __nv_bfloat16* bias, __nv_fp8_e4m3* out,
    const float* act_scale_ptr, long long M, int N, cudaStream_t stream)
{
  if (M <= 0 || N <= 0) return;
  if (N % 8)
    throw std::runtime_error("bias_gelu_quantize_fp8_static_bf16_rounded_v8: N must be a multiple of 8");
  if (!bias)
    throw std::runtime_error("bias_gelu_quantize_fp8_static_bf16_rounded_v8: bias is required");
  const long long vecs = M * (long long)(N / 8);
  const int block = 256;
  long long grid = (vecs + block - 1) / block;
  if (grid > 65535) grid = 65535;            // the grid-stride loop covers the remainder
  bias_gelu_quantize_fp8_static_bf16_rounded_v8_kernel<<<(int)grid, block, 0, stream>>>(
      in, bias, out, act_scale_ptr, M, N);
}

}  // namespace quantize
}  // namespace flash_rt
