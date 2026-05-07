// SPDX-License-Identifier: Apache-2.0
//
// Fused q_norm/k_norm + RoPE + Q_buf/KV cache write.
// See qwen3_qkv_post_proc.cuh for design notes.

#include "qwen3_qkv_post_proc.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int HEAD_DIM = 128;
constexpr int HALF = HEAD_DIM / 2;        // 64
constexpr int THREADS = HEAD_DIM;         // 1 thread per head_dim element
constexpr int N_WARPS = THREADS / 32;     // 4

// Block-wide sum reduction (4 warps × 32 lanes).
//
// First reduces within each warp via __shfl_xor_sync, then aggregates
// across warps via a 4-element smem scratch + final warp shuffle.
__device__ __forceinline__ float block_sum_4warp(float v, float* smem4) {
  // Intra-warp reduction.
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  if (lane == 0) smem4[wid] = v;
  __syncthreads();
  // Final warp reduces the 4 partial sums.
  if (wid == 0) {
    float t = (lane < N_WARPS) ? smem4[lane] : 0.f;
    #pragma unroll
    for (int off = 2; off > 0; off >>= 1) {
      t += __shfl_xor_sync(0xffffffff, t, off);
    }
    if (lane == 0) smem4[0] = t;
  }
  __syncthreads();
  return smem4[0];
}

// Q kernel: gridDim.x = n_q_heads, blockDim.x = HEAD_DIM (128).
__global__ void q_norm_rope_qstage_kernel(
    const __nv_bfloat16* __restrict__ q_pre,      // (n_q, 128)
    const __nv_bfloat16* __restrict__ q_norm_w,   // (128,)
    const __nv_bfloat16* __restrict__ cos_v,      // (64,)
    const __nv_bfloat16* __restrict__ sin_v,      // (64,)
    __nv_bfloat16* __restrict__ q_buf,            // (n_q, 128)
    int n_q,
    float eps) {
  int head = blockIdx.x;
  if (head >= n_q) return;
  int tid = threadIdx.x;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* q_row = q_pre + head * HEAD_DIM;
  float v = __bfloat162float(q_row[tid]);
  float w = __bfloat162float(q_norm_w[tid]);

  // Sum-of-squares reduction across the 128 threads.
  float sq = v * v;
  float sum_sq = block_sum_4warp(sq, s_smem4);
  float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + eps);

  // Apply RMSNorm + weight.
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  // Apply RoPE (full rotary; rotary_dim = head_dim).
  // Pair index: tid < half pairs with (tid + half), and rotate_half
  // uses negation on the lo half.
  float partner;
  float c, sn;
  if (tid < HALF) {
    partner = s_normed[tid + HALF];
    c = __bfloat162float(cos_v[tid]);
    sn = __bfloat162float(sin_v[tid]);
    // x_out = normed * cos - partner * sin
    float out = normed * c - partner * sn;
    q_buf[head * HEAD_DIM + tid] = __float2bfloat16(out);
  } else {
    partner = s_normed[tid - HALF];
    int half_idx = tid - HALF;
    c = __bfloat162float(cos_v[half_idx]);
    sn = __bfloat162float(sin_v[half_idx]);
    // x_out = normed * cos + partner * sin
    float out = normed * c + partner * sn;
    q_buf[head * HEAD_DIM + tid] = __float2bfloat16(out);
  }
}

// K kernel: gridDim.x = n_kv_heads, blockDim.x = HEAD_DIM (128).
// Same RoPE path as Q. ALSO writes V[head, tid] to V_cache (V is just
// copied — no norm, no RoPE).
__global__ void k_norm_rope_kvwrite_kernel(
    const __nv_bfloat16* __restrict__ k_pre,      // (n_kv, 128)
    const __nv_bfloat16* __restrict__ v_pre,      // (n_kv, 128)
    const __nv_bfloat16* __restrict__ k_norm_w,   // (128,)
    const __nv_bfloat16* __restrict__ cos_v,      // (64,)
    const __nv_bfloat16* __restrict__ sin_v,      // (64,)
    __nv_bfloat16* __restrict__ k_cache_dst,      // base of (n_kv, 128)
    __nv_bfloat16* __restrict__ v_cache_dst,      // base of (n_kv, 128)
    int n_kv,
    float eps) {
  int head = blockIdx.x;
  if (head >= n_kv) return;
  int tid = threadIdx.x;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* k_row = k_pre + head * HEAD_DIM;
  float v = __bfloat162float(k_row[tid]);
  float w = __bfloat162float(k_norm_w[tid]);

  float sq = v * v;
  float sum_sq = block_sum_4warp(sq, s_smem4);
  float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + eps);

  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  // Apply RoPE → write to K_cache slot.
  float partner, c, sn;
  if (tid < HALF) {
    partner = s_normed[tid + HALF];
    c = __bfloat162float(cos_v[tid]);
    sn = __bfloat162float(sin_v[tid]);
    float out = normed * c - partner * sn;
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(out);
  } else {
    partner = s_normed[tid - HALF];
    int half_idx = tid - HALF;
    c = __bfloat162float(cos_v[half_idx]);
    sn = __bfloat162float(sin_v[half_idx]);
    float out = normed * c + partner * sn;
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(out);
  }

  // V is just copied (no norm, no RoPE).
  v_cache_dst[head * HEAD_DIM + tid] = v_pre[head * HEAD_DIM + tid];
}

}  // namespace

int qwen3_q_norm_rope_qstage_bf16(
    const void* q_pre,
    const void* q_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    int         n_q_heads,
    float       eps,
    cudaStream_t stream) {
  if (!q_pre || !q_norm_w || !cos || !sin || !q_buf_dst) return 1;
  if (n_q_heads <= 0) return 2;
  dim3 grid(n_q_heads);
  dim3 block(THREADS);
  q_norm_rope_qstage_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      n_q_heads, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_bf16(
    const void* k_pre,
    const void* v_pre,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin
      || !k_cache_dst || !v_cache_dst) return 1;
  if (n_kv_heads <= 0) return 2;
  dim3 grid(n_kv_heads);
  dim3 block(THREADS);
  k_norm_rope_kvwrite_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      n_kv_heads, eps);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
