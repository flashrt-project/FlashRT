// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini fused activation-gate kernels. See header.

#include "kernels/act_fuse_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}
__device__ __forceinline__ float sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// Vectorised bf162 (2 elements / thread); n must be even.
__global__ void silu_mul_kernel(const __nv_bfloat162* __restrict__ g,
                                const __nv_bfloat162* __restrict__ u,
                                __nv_bfloat162* __restrict__ out, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  const float2 gf = __bfloat1622float2(g[i]);
  const float2 uf = __bfloat1622float2(u[i]);
  out[i] = __floats2bfloat162_rn(silu(gf.x) * uf.x, silu(gf.y) * uf.y);
}

__global__ void sigmoid_mul_kernel(const __nv_bfloat162* __restrict__ x,
                                   const __nv_bfloat162* __restrict__ gate,
                                   __nv_bfloat162* __restrict__ out, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  const float2 xf = __bfloat1622float2(x[i]);
  const float2 gf = __bfloat1622float2(gate[i]);
  out[i] = __floats2bfloat162_rn(xf.x * sigmoid(gf.x), xf.y * sigmoid(gf.y));
}

}  // namespace

int silu_mul_sm120_bf16(const void* g, const void* u, void* out, int n,
                        cudaStream_t stream) {
  if (!g || !u || !out) return 1;
  if (n <= 0 || (n & 1)) return 2;
  const int n2 = n >> 1;
  silu_mul_kernel<<<(n2 + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat162*>(g),
      reinterpret_cast<const __nv_bfloat162*>(u),
      reinterpret_cast<__nv_bfloat162*>(out), n2);
  return 0;
}

int sigmoid_mul_sm120_bf16(const void* x, const void* gate, void* out, int n,
                           cudaStream_t stream) {
  if (!x || !gate || !out) return 1;
  if (n <= 0 || (n & 1)) return 2;
  const int n2 = n >> 1;
  sigmoid_mul_kernel<<<(n2 + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat162*>(x),
      reinterpret_cast<const __nv_bfloat162*>(gate),
      reinterpret_cast<__nv_bfloat162*>(out), n2);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
