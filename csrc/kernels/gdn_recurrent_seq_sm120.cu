// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini GDN recurrent sequential-scan kernel (prefill). See header.
// Math mirrors gated_deltanet_recurrent_kernel; the state column lives in
// registers across the whole S-step scan instead of HBM per token.

#include "kernels/gdn_recurrent_seq_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHD = 128;
constexpr float kEps = 1e-6f;

template <int HD>
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  // 128 threads = 4 warps. Warp-reduce, stash partials, reduce, broadcast.
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) scratch[wid] = v;
  __syncthreads();
  float s = (threadIdx.x < (HD >> 5)) ? scratch[threadIdx.x] : 0.0f;
  if (wid == 0) {
    for (int o = (HD >> 5) >> 1; o > 0; o >>= 1)
      s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) scratch[0] = s;
  }
  __syncthreads();
  return scratch[0];
}

// grid = (num_v_heads,), block = HD. One block scans one head over all S.
template <int HD>
__global__ void gdn_recurrent_seq_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads, int S, bool use_qk_l2norm) {
  const int h = blockIdx.x;
  const int t = threadIdx.x;

  __shared__ float qs[HD];
  __shared__ float ks[HD];
  __shared__ float scratch[32];

  // Thread t owns column t of state[h]: col[i] = state[h, i, t]. Load once.
  const size_t state_h_off = (size_t)h * HD * HD;
  float col[HD];
#pragma unroll 16
  for (int i = 0; i < HD; ++i)
    col[i] = static_cast<float>(state[state_h_off + (size_t)i * HD + t]);

  const float inv_sqrt_hd = rsqrtf(static_cast<float>(HD));

  for (int ts = 0; ts < S; ++ts) {
    const size_t qkv_off = ((size_t)ts * num_v_heads + h) * HD + t;
    qs[t] = static_cast<float>(q_in[qkv_off]);
    ks[t] = static_cast<float>(k_in[qkv_off]);
    __syncthreads();

    if (use_qk_l2norm) {
      float q_sq = block_reduce_sum<HD>(qs[t] * qs[t], scratch);
      __syncthreads();
      float k_sq = block_reduce_sum<HD>(ks[t] * ks[t], scratch);
      const float q_inv = rsqrtf(q_sq + kEps);
      const float k_inv = rsqrtf(k_sq + kEps);
      qs[t] *= q_inv;
      ks[t] *= k_inv;
      __syncthreads();
    }
    qs[t] *= inv_sqrt_hd;
    __syncthreads();

    const float g_t = __expf(static_cast<float>(g_in[ts * num_v_heads + h]));
    const float beta_t = static_cast<float>(beta_in[ts * num_v_heads + h]);

    // decay + kv_mem = sum_i (col[i]*g_t) * ks[i]
    float kv_mem = 0.0f;
#pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      col[i] *= g_t;
      kv_mem = fmaf(col[i], ks[i], kv_mem);
    }
    const float v_t = static_cast<float>(v_in[qkv_off]);
    const float delta = (v_t - kv_mem) * beta_t;

    float out_t = 0.0f;
#pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      col[i] = fmaf(ks[i], delta, col[i]);
      out_t = fmaf(col[i], qs[i], out_t);
    }
    out_[qkv_off] = __float2bfloat16(out_t);
    __syncthreads();          // before next ts overwrites qs/ks
  }

  // Final state writeback.
#pragma unroll 16
  for (int i = 0; i < HD; ++i)
    state[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
}

}  // namespace

int gdn_recurrent_seq_sm120_bf16(
    const void*  q,
    const void*  k,
    const void*  v,
    const void*  g,
    const void*  beta,
    void*        state,
    void*        out,
    int          S,
    int          num_v_heads,
    int          head_dim,
    bool         use_qk_l2norm,
    cudaStream_t stream) {
  if (!q || !k || !v || !g || !beta || !state || !out) return 1;
  if (head_dim != kHD || S <= 0 || num_v_heads <= 0) return 2;
  dim3 grid(num_v_heads);
  dim3 block(kHD);
  gdn_recurrent_seq_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, S, use_qk_l2norm);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
