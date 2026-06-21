// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE router top-k (M=1). See header.

#include "kernels/moe_router_topk_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// argmax (value, carrying index) across a warp via shuffles, no __syncthreads.
__device__ __forceinline__ void warp_argmax(float& v, int& iv) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffff, v, o);
    const int oi = __shfl_xor_sync(0xffffffff, iv, o);
    if (ov > v || (ov == v && oi < iv)) { v = ov; iv = oi; }
  }
}

// One block. Each thread owns its strided logits; k rounds of block-argmax via
// warp shuffles (lower-index wins ties, deterministic), masking the winner.
__global__ void router_topk_kernel(const __nv_bfloat16* __restrict__ logits,
                                   int* __restrict__ out_idx,
                                   float* __restrict__ out_val,
                                   int n, int k) {
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int wid = t >> 5;
  __shared__ float sw_val[kWarps];
  __shared__ int sw_idx[kWarps];
  __shared__ int s_win;

  extern __shared__ float s_log[];
  for (int i = t; i < n; i += kThreads) s_log[i] = static_cast<float>(logits[i]);
  __syncthreads();

  for (int r = 0; r < k; ++r) {
    float best = -FLT_MAX;
    int bidx = -1;
    for (int i = t; i < n; i += kThreads) {
      const float v = s_log[i];
      if (v > best || (v == best && i < bidx)) { best = v; bidx = i; }
    }
    warp_argmax(best, bidx);                    // lane 0 of each warp has it
    if (lane == 0) { sw_val[wid] = best; sw_idx[wid] = bidx; }
    __syncthreads();
    if (wid == 0) {
      best = (lane < kWarps) ? sw_val[lane] : -FLT_MAX;
      bidx = (lane < kWarps) ? sw_idx[lane] : -1;
      warp_argmax(best, bidx);
      if (lane == 0) {
        out_idx[r] = bidx;
        out_val[r] = best;
        s_win = bidx;
      }
    }
    __syncthreads();
    if (t == 0 && s_win >= 0) s_log[s_win] = -FLT_MAX;
    __syncthreads();
  }
}

}  // namespace

int moe_router_topk_sm120_bf16(const void* logits, void* out_idx, void* out_val,
                           int n_experts, int k, cudaStream_t stream) {
  if (!logits || !out_idx || !out_val) return 1;
  if (n_experts <= 0 || k <= 0 || k > 32) return 2;
  const size_t smem = (size_t)n_experts * sizeof(float);
  router_topk_kernel<<<1, kThreads, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits),
      reinterpret_cast<int*>(out_idx),
      reinterpret_cast<float*>(out_val),
      n_experts, k);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
