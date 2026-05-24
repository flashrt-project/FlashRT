// SPDX-License-Identifier: Apache-2.0
//
// Per-token x per-128 FP8 e4m3 quantization. See header for spec.

#include "fp8_per_token_block_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr int kBlock = 128;
constexpr float kFp8Max = 448.0f;

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

}  // namespace

void fp8_per_token_block128_quant_bf16(
    const void* input,
    void* output_fp8,
    float* output_scale,
    int M, int K,
    cudaStream_t stream)
{
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

}  // namespace quantize
}  // namespace flash_rt
