// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini M=1 BF16 GEMV (memory-level-parallel). See header.

#include "kernels/nexn2_bf16_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kWarps = 8;            // 8 outputs / block
constexpr int kThreads = kWarps * 32;
constexpr int kUnroll = 8;           // int4 loads in flight per warp

__device__ __forceinline__ float dot_i4(const int4& wv, const int4& xv) {
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    __nv_bfloat162 wb = *reinterpret_cast<const __nv_bfloat162*>(
        &(reinterpret_cast<const int*>(&wv)[k]));
    __nv_bfloat162 xb = *reinterpret_cast<const __nv_bfloat162*>(
        &(reinterpret_cast<const int*>(&xv)[k]));
    float2 wf = __bfloat1622float2(wb);
    float2 xf = __bfloat1622float2(xb);
    acc = fmaf(xf.x, wf.x, acc);
    acc = fmaf(xf.y, wf.y, acc);
  }
  return acc;
}

// 1 warp per output row, UNROLL int4 loads in flight. x staged in smem,
// reused across the 8 warps' outputs.
__global__ void bf16_matvec_mlp_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int N, int K) {
  extern __shared__ __nv_bfloat16 x_sh[];
  const int K_int4 = K >> 3;                 // bf16 per int4 = 8
  const int4* x_i4 = reinterpret_cast<const int4*>(x);
  int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
  for (int j = threadIdx.x; j < K_int4; j += kThreads) x_sh_i4[j] = x_i4[j];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (n >= N) return;
  const int4* w_i4 = reinterpret_cast<const int4*>(W) + (size_t)n * K_int4;

  float acc = 0.0f;
  int i4 = lane;
  const int step = 32 * kUnroll;
  for (; i4 + 32 * (kUnroll - 1) < K_int4; i4 += step) {
    int4 wv[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) wv[u] = w_i4[i4 + 32 * u];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) acc += dot_i4(wv[u], x_sh_i4[i4 + 32 * u]);
  }
  for (; i4 < K_int4; i4 += 32) acc += dot_i4(w_i4[i4], x_sh_i4[i4]);

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  if (lane == 0) out[n] = __float2bfloat16(acc);
}

}  // namespace

int nexn2_bf16_matvec_bf16(
    const void*  x,
    const void*  W,
    void*        out,
    int          N,
    int          K,
    cudaStream_t stream) {
  if (!x || !W || !out) return 1;
  if (N <= 0 || K <= 0 || (K & 7) != 0) return 2;
  dim3 grid((N + kWarps - 1) / kWarps);
  dim3 block(kThreads);
  size_t smem = (size_t)K * sizeof(__nv_bfloat16);
  bf16_matvec_mlp_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(W),
      reinterpret_cast<__nv_bfloat16*>(out),
      N, K);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
