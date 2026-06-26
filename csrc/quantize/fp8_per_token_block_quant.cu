// SPDX-License-Identifier: Apache-2.0
//
// Per-token x per-128 FP8 e4m3 quantization. See header for spec.

#include "fp8_per_token_block_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace flash_rt {
namespace quantize {

namespace {

constexpr int kBlock = 128;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float block_reduce_sum_128(float v, float* sh) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < 4) ? sh[lane] : 0.0f;
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    if (lane == 0) sh[0] = v;
  }
  __syncthreads();
  const float out = sh[0];
  __syncthreads();
  return out;
}

__device__ __forceinline__ float block_reduce_max_128(float v, float* sh) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  }
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < 4) ? sh[lane] : 0.0f;
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    if (lane == 0) sh[0] = v;
  }
  __syncthreads();
  const float out = sh[0];
  __syncthreads();
  return out;
}

__global__ void fp8_per_token_block_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int M, int K)
{
  // One block per (m, kb). 128 threads cover the 128-element scale block.
  const int m = blockIdx.y;
  const int kb = blockIdx.x;
  if (m >= M || kb * kBlock >= K) return;

  const int t = threadIdx.x;
  const int k = kb * kBlock + t;

  // Load.
  const float v = (k < K)
      ? static_cast<float>(input[m * K + k])
      : 0.0f;
  const float a = fabsf(v);

  // Block-reduce |max| across 128 threads (4 warps).
  float amax = a;
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  }
  __shared__ float warp_amax[4];
  const int lane = t & 31;
  const int warp = t >> 5;
  if (lane == 0) warp_amax[warp] = amax;
  __syncthreads();

  // Final reduce in warp 0.
  if (warp == 0) {
    amax = (lane < 4) ? warp_amax[lane] : 0.0f;
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    if (lane == 0) {
      // Avoid div-by-zero; use small epsilon equivalent to fp8-eps.
      const float s = fmaxf(amax / kFp8Max, 1.0e-12f);
      warp_amax[0] = s;
      scale[m * (K / kBlock) + kb] = s;
    }
  }
  __syncthreads();

  const float inv_s = 1.0f / warp_amax[0];

  // Quantize and store.
  if (k < K) {
    float q = v * inv_s;
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    output[m * K + k] = __nv_fp8_e4m3(q);
  }
}

__global__ void fp8_per_token_block_quant_linear_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int M, int K)
{
  const int k_blocks = K / kBlock;
  const int tile = blockIdx.x;
  const int m = tile / k_blocks;
  const int kb = tile - m * k_blocks;
  if (m >= M || kb * kBlock >= K) return;

  const int t = threadIdx.x;
  const int k = kb * kBlock + t;

  // Load.
  const float v = (k < K)
      ? static_cast<float>(input[m * K + k])
      : 0.0f;
  const float a = fabsf(v);

  // Block-reduce |max| across 128 threads (4 warps).
  float amax = a;
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  }
  __shared__ float warp_amax[4];
  const int lane = t & 31;
  const int warp = t >> 5;
  if (lane == 0) warp_amax[warp] = amax;
  __syncthreads();

  // Final reduce in warp 0.
  if (warp == 0) {
    amax = (lane < 4) ? warp_amax[lane] : 0.0f;
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    if (lane == 0) {
      // Avoid div-by-zero; use small epsilon equivalent to fp8-eps.
      const float s = fmaxf(amax / kFp8Max, 1.0e-12f);
      warp_amax[0] = s;
      scale[m * (K / kBlock) + kb] = s;
    }
  }
  __syncthreads();

  const float inv_s = 1.0f / warp_amax[0];

  // Quantize and store.
  if (k < K) {
    float q = v * inv_s;
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    output[m * K + k] = __nv_fp8_e4m3(q);
  }
}

__global__ void rms_norm_to_fp8_block128_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int K, float eps)
{
  extern __shared__ unsigned char smem[];
  float* red = reinterpret_cast<float*>(smem);
  __nv_bfloat16* normed = reinterpret_cast<__nv_bfloat16*>(red + 4);
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  const __nv_bfloat16* row = input + (size_t)m * K;
  __nv_fp8_e4m3* out = output + (size_t)m * K;

  float ssq = 0.0f;
  for (int i = t; i < K; i += kBlock) {
    const float v = __bfloat162float(row[i]);
    ssq += v * v;
  }
  const float inv_rms = rsqrtf(block_reduce_sum_128(ssq, red) / K + eps);

  for (int i = t; i < K; i += kBlock) {
    const float v = __bfloat162float(row[i]) * inv_rms *
                    __bfloat162float(weight[i]);
    normed[i] = __float2bfloat16(v);
  }
  __syncthreads();

  const int n_kb = K / kBlock;
  for (int kb = 0; kb < n_kb; ++kb) {
    const int i = kb * kBlock + t;
    const float v = __bfloat162float(normed[i]);
    const float amax = block_reduce_max_128(fabsf(v), red);
    const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
    if (t == 0) scale[m * n_kb + kb] = sc;
    float q = v / sc;
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    out[i] = __nv_fp8_e4m3(q);
  }
}

__global__ void residual_add_rms_norm_to_fp8_block128_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ residual_out,
    const __nv_bfloat16* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int K, float eps)
{
  extern __shared__ unsigned char smem[];
  float* red = reinterpret_cast<float*>(smem);
  __nv_bfloat16* normed = reinterpret_cast<__nv_bfloat16*>(red + 4);
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  const __nv_bfloat16* rrow = residual + (size_t)m * K;
  const __nv_bfloat16* xrow = x + (size_t)m * K;
  __nv_bfloat16* res_out = residual_out + (size_t)m * K;
  __nv_fp8_e4m3* out = output + (size_t)m * K;

  float ssq = 0.0f;
  for (int i = t; i < K; i += kBlock) {
    const float rv = __bfloat162float(rrow[i]) + __bfloat162float(xrow[i]);
    const __nv_bfloat16 rb = __float2bfloat16(rv);
    res_out[i] = rb;
    const float rbf = __bfloat162float(rb);
    ssq += rbf * rbf;
  }
  const float inv_rms = rsqrtf(block_reduce_sum_128(ssq, red) / K + eps);

  for (int i = t; i < K; i += kBlock) {
    const float v = __bfloat162float(res_out[i]) * inv_rms *
                    __bfloat162float(weight[i]);
    normed[i] = __float2bfloat16(v);
  }
  __syncthreads();

  const int n_kb = K / kBlock;
  for (int kb = 0; kb < n_kb; ++kb) {
    const int i = kb * kBlock + t;
    const float v = __bfloat162float(normed[i]);
    const float amax = block_reduce_max_128(fabsf(v), red);
    const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
    if (t == 0) scale[m * n_kb + kb] = sc;
    float q = v / sc;
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    out[i] = __nv_fp8_e4m3(q);
  }
}

__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void silu_mul_to_fp8_block128_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int K)
{
  const int m = blockIdx.y;
  const int kb = blockIdx.x;
  const int t = threadIdx.x;
  const int k = kb * kBlock + t;
  const size_t idx = (size_t)m * K + k;
  __shared__ float red[4];

  const float g = __bfloat162float(gate[idx]);
  const float u = __bfloat162float(up[idx]);
  const float silu_g = silu_f32(g);
  const float silu_bf = __bfloat162float(__float2bfloat16(silu_g));
  const float v = __bfloat162float(__float2bfloat16(silu_bf * u));
  const float amax = block_reduce_max_128(fabsf(v), red);
  const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
  if (t == 0) scale[m * (K / kBlock) + kb] = sc;
  float q = v / sc;
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  output[idx] = __nv_fp8_e4m3(q);
}

__global__ void silu_mul_merged_to_fp8_block128_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int K)
{
  const int m = blockIdx.y;
  const int kb = blockIdx.x;
  const int t = threadIdx.x;
  const int k = kb * kBlock + t;
  const size_t out_idx = (size_t)m * K + k;
  const size_t gate_idx = (size_t)m * (2 * K) + k;
  const size_t up_idx = gate_idx + K;
  __shared__ float red[4];

  const float g = __bfloat162float(gate_up[gate_idx]);
  const float u = __bfloat162float(gate_up[up_idx]);
  const float silu_g = silu_f32(g);
  const float silu_bf = __bfloat162float(__float2bfloat16(silu_g));
  const float v = __bfloat162float(__float2bfloat16(silu_bf * u));
  const float amax = block_reduce_max_128(fabsf(v), red);
  const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
  if (t == 0) scale[m * (K / kBlock) + kb] = sc;
  float q = v / sc;
  q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
  output[out_idx] = __nv_fp8_e4m3(q);
}

}  // namespace

void fp8_per_token_block128_quant_bf16(
    const void* input,
    void* output_fp8,
    float* output_scale,
    int M, int K,
    cudaStream_t stream)
{
  if ((K % kBlock) != 0)
    throw std::runtime_error(
        "fp8_per_token_block128_quant_bf16 requires K multiple of 128");
  dim3 block(kBlock);
  if (M <= 65535) {
    dim3 grid(K / kBlock, M);
    fp8_per_token_block_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_fp8_e4m3*>(output_fp8),
        output_scale,
        M, K);
  } else {
    dim3 grid((K / kBlock) * M);
    fp8_per_token_block_quant_linear_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_fp8_e4m3*>(output_fp8),
        output_scale,
        M, K);
  }
}

void rms_norm_to_fp8_block128_bf16(
    const void* input,
    const void* weight,
    void* output_fp8,
    float* output_scale,
    int M, int K, float eps,
    cudaStream_t stream)
{
  if ((K % kBlock) != 0)
    throw std::runtime_error(
        "rms_norm_to_fp8_block128_bf16 requires K multiple of 128");
  size_t smem = 4 * sizeof(float) + (size_t)K * sizeof(__nv_bfloat16);
  rms_norm_to_fp8_block128_kernel<<<M, kBlock, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<__nv_fp8_e4m3*>(output_fp8),
      output_scale, K, eps);
}

void residual_add_rms_norm_to_fp8_block128_bf16(
    const void* residual,
    const void* x,
    void* residual_out,
    const void* weight,
    void* output_fp8,
    float* output_scale,
    int M, int K, float eps,
    cudaStream_t stream)
{
  if ((K % kBlock) != 0)
    throw std::runtime_error(
        "residual_add_rms_norm_to_fp8_block128_bf16 requires K multiple of 128");
  size_t smem = 4 * sizeof(float) + (size_t)K * sizeof(__nv_bfloat16);
  residual_add_rms_norm_to_fp8_block128_kernel<<<M, kBlock, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<__nv_bfloat16*>(residual_out),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<__nv_fp8_e4m3*>(output_fp8),
      output_scale, K, eps);
}

void silu_mul_to_fp8_block128_bf16(
    const void* gate,
    const void* up,
    void* output_fp8,
    float* output_scale,
    int M, int K,
    cudaStream_t stream)
{
  if ((K % kBlock) != 0)
    throw std::runtime_error(
        "silu_mul_to_fp8_block128_bf16 requires K multiple of 128");
  dim3 block(kBlock);
  dim3 grid(K / kBlock, M);
  silu_mul_to_fp8_block128_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<const __nv_bfloat16*>(up),
      reinterpret_cast<__nv_fp8_e4m3*>(output_fp8),
      output_scale, K);
}

void silu_mul_merged_to_fp8_block128_bf16(
    const void* gate_up,
    void* output_fp8,
    float* output_scale,
    int M, int K,
    cudaStream_t stream)
{
  if ((K % kBlock) != 0)
    throw std::runtime_error(
        "silu_mul_merged_to_fp8_block128_bf16 requires K multiple of 128");
  dim3 block(kBlock);
  dim3 grid(K / kBlock, M);
  silu_mul_merged_to_fp8_block128_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate_up),
      reinterpret_cast<__nv_fp8_e4m3*>(output_fp8),
      output_scale, K);
}

}  // namespace quantize
}  // namespace flash_rt
