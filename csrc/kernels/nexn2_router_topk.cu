// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE router top-k (M=1). See header.

#include "kernels/nexn2_router_topk.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kThreads = 256;

// One block. Each thread owns several logits (grid-stride within the block);
// k rounds of block-wide argmax, masking the winner each round.
__global__ void router_topk_kernel(const __nv_bfloat16* __restrict__ logits,
                                   int* __restrict__ out_idx,
                                   float* __restrict__ out_val,
                                   int n, int k) {
  const int t = threadIdx.x;
  // Per-thread running best over its strided slice.
  __shared__ float s_val[kThreads];
  __shared__ int s_idx[kThreads];

  // Load this thread's slice into registers (small n: usually 1 each).
  // We re-scan the (masked) logits each round from a local copy in smem so
  // masking is cheap. Stage the logits in smem floats.
  extern __shared__ float s_log[];
  for (int i = t; i < n; i += kThreads) s_log[i] = static_cast<float>(logits[i]);
  __syncthreads();

  for (int r = 0; r < k; ++r) {
    float best = -FLT_MAX;
    int bidx = -1;
    for (int i = t; i < n; i += kThreads) {
      const float v = s_log[i];
      if (v > best) { best = v; bidx = i; }
    }
    s_val[t] = best;
    s_idx[t] = bidx;
    __syncthreads();
    // Block reduce (max with index) over the kThreads partials.
    for (int stride = kThreads >> 1; stride > 0; stride >>= 1) {
      if (t < stride) {
        if (s_val[t + stride] > s_val[t]) {
          s_val[t] = s_val[t + stride];
          s_idx[t] = s_idx[t + stride];
        }
      }
      __syncthreads();
    }
    const int win = s_idx[0];
    if (t == 0) {
      out_idx[r] = win;
      out_val[r] = s_val[0];
    }
    if (t == 0 && win >= 0) s_log[win] = -FLT_MAX;   // mask the winner
    __syncthreads();
  }
}

}  // namespace

int nexn2_router_topk_bf16(const void* logits, void* out_idx, void* out_val,
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
