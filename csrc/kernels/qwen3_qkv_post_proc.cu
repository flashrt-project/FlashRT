// SPDX-License-Identifier: Apache-2.0
//
// Fused q_norm/k_norm + RoPE + Q_buf/KV cache write.
// See qwen3_qkv_post_proc.cuh for design notes.

#include "pdl_launch.cuh"
#include "qwen3_qkv_post_proc.cuh"

#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
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

__global__ void qk_norm_rope_kvwrite_kernel(
    const __nv_bfloat16* __restrict__ q_pre,
    const __nv_bfloat16* __restrict__ k_pre,
    const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_v,
    const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ q_buf,
    __nv_bfloat16* __restrict__ k_cache_dst,
    __nv_bfloat16* __restrict__ v_cache_dst,
    int n_q,
    int n_kv,
    float eps) {
  const int block = blockIdx.x;
  const int tid = threadIdx.x;
  const bool is_q = block < n_q;
  const int head = is_q ? block : (block - n_q);
  if ((!is_q) && head >= n_kv) return;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* x_row =
      is_q ? (q_pre + head * HEAD_DIM) : (k_pre + head * HEAD_DIM);
  const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;
  float v = __bfloat162float(x_row[tid]);
  float w = __bfloat162float(norm_w[tid]);

  float sum_sq = block_sum_4warp(v * v, s_smem4);
  float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + eps);
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  float partner, c, sn, out;
  if (tid < HALF) {
    partner = s_normed[tid + HALF];
    c = __bfloat162float(cos_v[tid]);
    sn = __bfloat162float(sin_v[tid]);
    out = normed * c - partner * sn;
  } else {
    partner = s_normed[tid - HALF];
    int half_idx = tid - HALF;
    c = __bfloat162float(cos_v[half_idx]);
    sn = __bfloat162float(sin_v[half_idx]);
    out = normed * c + partner * sn;
  }

  if (is_q) {
    q_buf[head * HEAD_DIM + tid] = __float2bfloat16(out);
  } else {
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(out);
    v_cache_dst[head * HEAD_DIM + tid] = v_pre[head * HEAD_DIM + tid];
  }
}

__global__ void qk_norm_rope_kvwrite_batched_kernel(
    const __nv_bfloat16* __restrict__ q_pre,
    const __nv_bfloat16* __restrict__ k_pre,
    const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_buf,
    __nv_bfloat16* __restrict__ k_cache_dst,
    __nv_bfloat16* __restrict__ v_cache_dst,
    int seq_len,
    int q_pre_row_elems,
    int k_pre_row_elems,
    int v_pre_row_elems,
    int q_dst_row_elems,
    int kv_dst_row_elems,
    int n_q,
    int n_kv,
    float eps) {
  const int block = blockIdx.x;
  const int tid = threadIdx.x;
  const int heads_total = n_q + n_kv;
  const int token = block / heads_total;
  if (token >= seq_len) return;
  const int inner = block - token * heads_total;
  const bool is_q = inner < n_q;
  const int head = is_q ? inner : (inner - n_q);
  if ((!is_q) && head >= n_kv) return;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* x_row =
      is_q ? (q_pre + token * q_pre_row_elems + head * HEAD_DIM)
           : (k_pre + token * k_pre_row_elems + head * HEAD_DIM);
  const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;
  const __nv_bfloat16* cos_row = cos + token * HALF;
  const __nv_bfloat16* sin_row = sin + token * HALF;
  float v = __bfloat162float(x_row[tid]);
  float w = __bfloat162float(norm_w[tid]);

  float sum_sq = block_sum_4warp(v * v, s_smem4);
  float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + eps);
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  float partner, c, sn, out;
  if (tid < HALF) {
    partner = s_normed[tid + HALF];
    c = __bfloat162float(cos_row[tid]);
    sn = __bfloat162float(sin_row[tid]);
    out = normed * c - partner * sn;
  } else {
    partner = s_normed[tid - HALF];
    int half_idx = tid - HALF;
    c = __bfloat162float(cos_row[half_idx]);
    sn = __bfloat162float(sin_row[half_idx]);
    out = normed * c + partner * sn;
  }

  if (is_q) {
    q_buf[token * q_dst_row_elems + head * HEAD_DIM + tid] =
        __float2bfloat16(out);
  } else {
    k_cache_dst[token * kv_dst_row_elems + head * HEAD_DIM + tid] =
        __float2bfloat16(out);
    v_cache_dst[token * kv_dst_row_elems + head * HEAD_DIM + tid] =
        v_pre[token * v_pre_row_elems + head * HEAD_DIM + tid];
  }
}

// Device-position variant: writes K/V into K_cache[*cur_pos] / V_cache[*cur_pos]
// where cur_pos is read from device memory, so a single captured graph serves
// every decode position (the host bumps *cur_pos before each replay). row_elems
// = elements between consecutive position slots (n_kv * HEAD_DIM). Same rope/norm
// math as k_norm_rope_kvwrite_kernel.
__global__ void k_norm_rope_kvwrite_devpos_kernel(
    const __nv_bfloat16* __restrict__ k_pre,
    const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_v,
    const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ k_cache_base,
    __nv_bfloat16* __restrict__ v_cache_base,
    const int* __restrict__ cur_pos,
    int row_elems,
    int n_kv,
    float eps) {
  int head = blockIdx.x;
  if (head >= n_kv) return;
  int tid = threadIdx.x;
  const size_t slot = (size_t)(*cur_pos) * row_elems;
  __nv_bfloat16* k_cache_dst = k_cache_base + slot;
  __nv_bfloat16* v_cache_dst = v_cache_base + slot;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* k_row = k_pre + head * HEAD_DIM;
  float v = __bfloat162float(k_row[tid]);
  float w = __bfloat162float(k_norm_w[tid]);
  float sum_sq = block_sum_4warp(v * v, s_smem4);
  float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + eps);
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  float partner, c, sn;
  if (tid < HALF) {
    partner = s_normed[tid + HALF];
    c = __bfloat162float(cos_v[tid]);
    sn = __bfloat162float(sin_v[tid]);
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(normed * c - partner * sn);
  } else {
    partner = s_normed[tid - HALF];
    int half_idx = tid - HALF;
    c = __bfloat162float(cos_v[half_idx]);
    sn = __bfloat162float(sin_v[half_idx]);
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(normed * c + partner * sn);
  }
  v_cache_dst[head * HEAD_DIM + tid] = v_pre[head * HEAD_DIM + tid];
}

// ── Prefill (S>1) batched variants ──
// One block per (row, head); grid = (n_heads, S). Read strided q/k/v from
// the fused QKV output (in_row_stride = qkv_N), per-row cos/sin (stride
// HALF), write to contiguous Q_buf / K_cache / V_cache (out/cache row
// stride passed in). Same norm+RoPE math as the decode kernels; folds the
// per-layer rms_norm + multi-op RoPE + Q/K/V copies into 2 launches.
__global__ void q_norm_rope_qstage_prefill_kernel(
    const __nv_bfloat16* __restrict__ q_pre,      // (S, *) strided
    const __nv_bfloat16* __restrict__ q_norm_w,   // (128,)
    const __nv_bfloat16* __restrict__ cos_v,      // (S, 64)
    const __nv_bfloat16* __restrict__ sin_v,      // (S, 64)
    __nv_bfloat16* __restrict__ q_buf,            // (S, n_q*128)
    int n_q, int S,
    int in_row_stride, int out_row_stride,
    float eps) {
  int head = blockIdx.x;
  int row = blockIdx.y;
  if (head >= n_q || row >= S) return;
  int tid = threadIdx.x;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* q_row = q_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  float v = __bfloat162float(q_row[tid]);
  float w = __bfloat162float(q_norm_w[tid]);
  float rstd = rsqrtf(block_sum_4warp(v * v, s_smem4) / float(HEAD_DIM) + eps);
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  __nv_bfloat16* dst = q_buf + (size_t)row * out_row_stride + head * HEAD_DIM;
  if (tid < HALF) {
    float partner = s_normed[tid + HALF];
    float c = __bfloat162float(cos_r[tid]);
    float sn = __bfloat162float(sin_r[tid]);
    dst[tid] = __float2bfloat16(normed * c - partner * sn);
  } else {
    float partner = s_normed[tid - HALF];
    int hi = tid - HALF;
    float c = __bfloat162float(cos_r[hi]);
    float sn = __bfloat162float(sin_r[hi]);
    dst[tid] = __float2bfloat16(normed * c + partner * sn);
  }
}

__global__ void k_norm_rope_kvwrite_prefill_kernel(
    const __nv_bfloat16* __restrict__ k_pre,      // (S, *) strided
    const __nv_bfloat16* __restrict__ v_pre,      // (S, *) strided
    const __nv_bfloat16* __restrict__ k_norm_w,   // (128,)
    const __nv_bfloat16* __restrict__ cos_v,      // (S, 64)
    const __nv_bfloat16* __restrict__ sin_v,      // (S, 64)
    __nv_bfloat16* __restrict__ k_cache,          // (S, n_kv*128)
    __nv_bfloat16* __restrict__ v_cache,          // (S, n_kv*128)
    int n_kv, int S,
    int in_row_stride, int cache_row_stride,
    float eps) {
  int head = blockIdx.x;
  int row = blockIdx.y;
  if (head >= n_kv || row >= S) return;
  int tid = threadIdx.x;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* k_row = k_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
  const __nv_bfloat16* v_row = v_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  float v = __bfloat162float(k_row[tid]);
  float w = __bfloat162float(k_norm_w[tid]);
  float rstd = rsqrtf(block_sum_4warp(v * v, s_smem4) / float(HEAD_DIM) + eps);
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  __nv_bfloat16* kdst = k_cache + (size_t)row * cache_row_stride + head * HEAD_DIM;
  if (tid < HALF) {
    float partner = s_normed[tid + HALF];
    float c = __bfloat162float(cos_r[tid]);
    float sn = __bfloat162float(sin_r[tid]);
    kdst[tid] = __float2bfloat16(normed * c - partner * sn);
  } else {
    float partner = s_normed[tid - HALF];
    int hi = tid - HALF;
    float c = __bfloat162float(cos_r[hi]);
    float sn = __bfloat162float(sin_r[hi]);
    kdst[tid] = __float2bfloat16(normed * c + partner * sn);
  }
  // V is copied (no norm, no RoPE).
  v_cache[(size_t)row * cache_row_stride + head * HEAD_DIM + tid] = v_row[tid];
}

// ── v3 (HEAD_DIM=128 only): WARP-per-row with VECTORIZED contiguous-4 access.
// Lane t owns the CONTIGUOUS 4 elements {4t,4t+1,4t+2,4t+3} via one 8-byte (uint2)
// load — so the warp issues ONE 256-byte coalesced transaction instead of v1/v2's
// 64-256B fragments (this is the lever: q_norm_rope is memory-throughput-bound at
// ~44% BW, NOT sync-bound). The RMS sum is a warp-shuffle; the RoPE partner of an
// element i is i±64, owned by lane t^16, fetched with one __shfl_xor(.,16) per
// element — no smem, no __syncthreads. cos/sin index = 4*(t&15)+j; the rotate sign
// is - for the low half (t<16) and + for the high half. NOT bit-identical (fp32
// ssq grouping ~1 ULP). blockDim.x = 32 (1 warp = 1 row). ──
__device__ __forceinline__ void load4_bf16(const __nv_bfloat16* p,
                                            float& a, float& b, float& c, float& d) {
  uint2 raw = *reinterpret_cast<const uint2*>(p);
  // Convert 2-at-once (one cvt per pair) instead of 4 scalar cvts.
  float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
  float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
  a = lo.x; b = lo.y; c = hi.x; d = hi.y;
}
__device__ __forceinline__ void store4_bf16(__nv_bfloat16* p,
                                             float a, float b, float c, float d) {
  __nv_bfloat162 lo = __float22bfloat162_rn(make_float2(a, b));
  __nv_bfloat162 hi = __float22bfloat162_rn(make_float2(c, d));
  uint2 raw;
  raw.x = *reinterpret_cast<const uint32_t*>(&lo);
  raw.y = *reinterpret_cast<const uint32_t*>(&hi);
  *reinterpret_cast<uint2*>(p) = raw;
}

__device__ __forceinline__ void norm_rope_warp4(
    const __nv_bfloat16* row_in, const __nv_bfloat16* norm_w,
    const __nv_bfloat16* cos_r, const __nv_bfloat16* sin_r,
    __nv_bfloat16* dst, int lane, float eps) {
  float v0, v1, v2, v3;
  load4_bf16(row_in + 4 * lane, v0, v1, v2, v3);
  float sq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
  float rstd = rsqrtf(sq / float(HEAD_DIM) + eps);
  float w0, w1, w2, w3;
  load4_bf16(norm_w + 4 * lane, w0, w1, w2, w3);
  float n0 = v0 * rstd * w0, n1 = v1 * rstd * w1, n2 = v2 * rstd * w2, n3 = v3 * rstd * w3;
  float pn0 = __shfl_xor_sync(0xffffffff, n0, 16);
  float pn1 = __shfl_xor_sync(0xffffffff, n1, 16);
  float pn2 = __shfl_xor_sync(0xffffffff, n2, 16);
  float pn3 = __shfl_xor_sync(0xffffffff, n3, 16);
  int ci = 4 * (lane & 15);
  float c0, c1, c2, c3, s0, s1, s2, s3;
  load4_bf16(cos_r + ci, c0, c1, c2, c3);
  load4_bf16(sin_r + ci, s0, s1, s2, s3);
  float sgn = (lane < 16) ? -1.f : 1.f;
  store4_bf16(dst + 4 * lane,
              n0 * c0 + sgn * pn0 * s0, n1 * c1 + sgn * pn1 * s1,
              n2 * c2 + sgn * pn2 * s2, n3 * c3 + sgn * pn3 * s3);
}

// ── v3 = ONE BLOCK PER ROW (grid=(S,), block=256=8 warps); each warp loops over
// its heads (head = warp, += n_warps) via the warp-per-row vectorized helper.
// This cuts the block count from n_heads*S (16384) to S (512) — matching the fast
// row-grid kernels — killing the per-tiny-block overhead that left the old
// one-block-per-(head,row) kernel at 2.5× its 4.1µs memory-copy floor. (Earlier
// single-head warp variants — sync-elimination, vectorized transactions — gave 0;
// the bottleneck was block COUNT, not sync/transaction/compute.) ──
__global__ void q_norm_rope_qstage_prefill_v3_kernel(
    const __nv_bfloat16* __restrict__ q_pre, const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ cos_v, const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ q_buf,
    int n_q, int S, int in_row_stride, int out_row_stride, float eps) {
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  for (int head = warp; head < n_q; head += n_warps) {
    norm_rope_warp4(q_pre + (size_t)row * in_row_stride + head * HEAD_DIM, q_norm_w,
                    cos_r, sin_r,
                    q_buf + (size_t)row * out_row_stride + head * HEAD_DIM, lane, eps);
  }
}

__global__ void k_norm_rope_kvwrite_prefill_v3_kernel(
    const __nv_bfloat16* __restrict__ k_pre, const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_v, const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    int n_kv, int S, int in_row_stride, int cache_row_stride, float eps) {
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  for (int head = warp; head < n_kv; head += n_warps) {
    norm_rope_warp4(k_pre + (size_t)row * in_row_stride + head * HEAD_DIM, k_norm_w,
                    cos_r, sin_r,
                    k_cache + (size_t)row * cache_row_stride + head * HEAD_DIM, lane, eps);
    const __nv_bfloat16* v_row = v_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
    __nv_bfloat16* vdst = v_cache + (size_t)row * cache_row_stride + head * HEAD_DIM;
    *reinterpret_cast<uint2*>(vdst + 4 * lane) =
        *reinterpret_cast<const uint2*>(v_row + 4 * lane);
  }
}

// ── v3 + FOLDED per-token fp8 emit (STEP D) ──
// Same RMS-norm + RoPE as norm_rope_warp4, but ALSO emits e4m3 Q/K with a
// per-(token,head) scale, with NO extra HBM round-trip (the row is already in
// registers). amax is a 32-lane warp reduce over the head's 128 channels; each
// lane packs its 4 RoPE'd values to e4m3; lane 0 writes scale[head*S + row].
// The bf16 dst is still written (decode / bf16-FA2 fallback path stays valid).
__device__ __forceinline__ void norm_rope_warp4_fp8(
    const __nv_bfloat16* row_in, const __nv_bfloat16* norm_w,
    const __nv_bfloat16* cos_r, const __nv_bfloat16* sin_r,
    __nv_bfloat16* dst, __nv_fp8_e4m3* dst8, float* scale_slot,
    int lane, float eps) {
  float v0, v1, v2, v3;
  load4_bf16(row_in + 4 * lane, v0, v1, v2, v3);
  float sq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
  float rstd = rsqrtf(sq / float(HEAD_DIM) + eps);
  float w0, w1, w2, w3;
  load4_bf16(norm_w + 4 * lane, w0, w1, w2, w3);
  float n0 = v0 * rstd * w0, n1 = v1 * rstd * w1, n2 = v2 * rstd * w2, n3 = v3 * rstd * w3;
  float pn0 = __shfl_xor_sync(0xffffffff, n0, 16);
  float pn1 = __shfl_xor_sync(0xffffffff, n1, 16);
  float pn2 = __shfl_xor_sync(0xffffffff, n2, 16);
  float pn3 = __shfl_xor_sync(0xffffffff, n3, 16);
  int ci = 4 * (lane & 15);
  float c0, c1, c2, c3, s0, s1, s2, s3;
  load4_bf16(cos_r + ci, c0, c1, c2, c3);
  load4_bf16(sin_r + ci, s0, s1, s2, s3);
  float sgn = (lane < 16) ? -1.f : 1.f;
  float o0 = n0 * c0 + sgn * pn0 * s0, o1 = n1 * c1 + sgn * pn1 * s1;
  float o2 = n2 * c2 + sgn * pn2 * s2, o3 = n3 * c3 + sgn * pn3 * s3;
  // bf16 dst is OPTIONAL: the fp8 prefill attention reads q8, so the Q fold
  // skips the redundant 4MB/layer bf16 Q_buf write (dst==nullptr). K keeps it
  // (decode / bf16-FA2 read K_cache/V_cache bf16).
  if (dst != nullptr) store4_bf16(dst + 4 * lane, o0, o1, o2, o3);
  // per-token e4m3: warp amax over the 128 channels, scale = amax/448.
  float a = fmaxf(fmaxf(fabsf(o0), fabsf(o1)), fmaxf(fabsf(o2), fabsf(o3)));
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
  float inv = 448.0f / fmaxf(a, 1e-7f);
  __nv_fp8x4_e4m3 packed(make_float4(o0 * inv, o1 * inv, o2 * inv, o3 * inv));
  *reinterpret_cast<__nv_fp8x4_e4m3*>(dst8 + 4 * lane) = packed;
  if (lane == 0) *scale_slot = a * (1.0f / 448.0f);
}

__global__ void q_norm_rope_qstage_prefill_v3_fp8_kernel(
    const __nv_bfloat16* __restrict__ q_pre, const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ cos_v, const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ q_buf, __nv_fp8_e4m3* __restrict__ q8,
    float* __restrict__ q_scale,
    int n_q, int S, int in_row_stride, int out_row_stride, float eps) {
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  for (int head = warp; head < n_q; head += n_warps) {
    // q8 is contiguous [S, n_q, 128]; q_scale is [n_q, S] (head-major).
    norm_rope_warp4_fp8(
        q_pre + (size_t)row * in_row_stride + head * HEAD_DIM, q_norm_w, cos_r, sin_r,
        q_buf + (size_t)row * out_row_stride + head * HEAD_DIM,
        q8 + ((size_t)row * n_q + head) * HEAD_DIM,
        q_scale + (size_t)head * S + row, lane, eps);
  }
}

__global__ void k_norm_rope_kvwrite_prefill_v3_fp8_kernel(
    const __nv_bfloat16* __restrict__ k_pre, const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_v, const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    __nv_fp8_e4m3* __restrict__ k8, float* __restrict__ k_scale,
    __half* __restrict__ v_fp16,
    int n_kv, int S, int in_row_stride, int cache_row_stride, float eps) {
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  for (int head = warp; head < n_kv; head += n_warps) {
    norm_rope_warp4_fp8(
        k_pre + (size_t)row * in_row_stride + head * HEAD_DIM, k_norm_w, cos_r, sin_r,
        k_cache + (size_t)row * cache_row_stride + head * HEAD_DIM,
        k8 + ((size_t)row * n_kv + head) * HEAD_DIM,
        k_scale + (size_t)head * S + row, lane, eps);
    // V: copy bf16 -> v_cache AND cast -> v_fp16 (folds the prefill V cast).
    const __nv_bfloat16* v_row = v_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
    __nv_bfloat16* vdst = v_cache + (size_t)row * cache_row_stride + head * HEAD_DIM;
    float a0, a1, a2, a3;
    load4_bf16(v_row + 4 * lane, a0, a1, a2, a3);
    store4_bf16(vdst + 4 * lane, a0, a1, a2, a3);
    __half* v16 = v_fp16 + ((size_t)row * n_kv + head) * HEAD_DIM + 4 * lane;
    v16[0] = __float2half(a0); v16[1] = __float2half(a1);
    v16[2] = __float2half(a2); v16[3] = __float2half(a3);
  }
}

// ── v3 + DIRECT e4m3 emit (no per-token scale) ──
// Same RMS-norm + RoPE fold as norm_rope_warp4_fp8, but the e4m3 cast is
// direct: post-norm Q/K magnitudes sit well inside e4m3's dynamic range
// (max observed ~228 vs the 448 format max), so a scale adds no relative
// precision — e4m3 keeps 3 mantissa bits at any exponent. Dropping the
// amax warp-reduce and scale write makes the fold cheaper and feeds the
// all-fp8 attention kernel (fmha_fp8_causal_gqa_nhd_d128) directly.
__device__ __forceinline__ void norm_rope_warp4_fp8_direct(
    const __nv_bfloat16* row_in, const __nv_bfloat16* norm_w,
    const __nv_bfloat16* cos_r, const __nv_bfloat16* sin_r,
    __nv_bfloat16* dst, __nv_fp8_e4m3* dst8, int lane, float eps) {
  float v0, v1, v2, v3;
  load4_bf16(row_in + 4 * lane, v0, v1, v2, v3);
  float sq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
  float rstd = rsqrtf(sq / float(HEAD_DIM) + eps);
  float w0, w1, w2, w3;
  load4_bf16(norm_w + 4 * lane, w0, w1, w2, w3);
  float n0 = v0 * rstd * w0, n1 = v1 * rstd * w1, n2 = v2 * rstd * w2, n3 = v3 * rstd * w3;
  float pn0 = __shfl_xor_sync(0xffffffff, n0, 16);
  float pn1 = __shfl_xor_sync(0xffffffff, n1, 16);
  float pn2 = __shfl_xor_sync(0xffffffff, n2, 16);
  float pn3 = __shfl_xor_sync(0xffffffff, n3, 16);
  int ci = 4 * (lane & 15);
  float c0, c1, c2, c3, s0, s1, s2, s3;
  load4_bf16(cos_r + ci, c0, c1, c2, c3);
  load4_bf16(sin_r + ci, s0, s1, s2, s3);
  float sgn = (lane < 16) ? -1.f : 1.f;
  float o0 = n0 * c0 + sgn * pn0 * s0, o1 = n1 * c1 + sgn * pn1 * s1;
  float o2 = n2 * c2 + sgn * pn2 * s2, o3 = n3 * c3 + sgn * pn3 * s3;
  if (dst != nullptr) store4_bf16(dst + 4 * lane, o0, o1, o2, o3);
  __nv_fp8x4_e4m3 packed(make_float4(o0, o1, o2, o3));
  *reinterpret_cast<__nv_fp8x4_e4m3*>(dst8 + 4 * lane) = packed;
}

// norm_rope_warp4_fp8_direct with the head's values, the norm weights and
// the RoPE row already in registers (callers hoist those loads so a warp's
// heads have their global latencies in flight together). Same arithmetic.
__device__ __forceinline__ void norm_rope_regs_fp8(
    float v0, float v1, float v2, float v3, float w0, float w1, float w2, float w3,
    float c0, float c1, float c2, float c3, float s0, float s1, float s2, float s3,
    __nv_fp8_e4m3* dst8, int lane, float eps) {
  float sq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
  float rstd = rsqrtf(sq / float(HEAD_DIM) + eps);
  float n0 = v0 * rstd * w0, n1 = v1 * rstd * w1, n2 = v2 * rstd * w2, n3 = v3 * rstd * w3;
  float pn0 = __shfl_xor_sync(0xffffffff, n0, 16);
  float pn1 = __shfl_xor_sync(0xffffffff, n1, 16);
  float pn2 = __shfl_xor_sync(0xffffffff, n2, 16);
  float pn3 = __shfl_xor_sync(0xffffffff, n3, 16);
  float sgn = (lane < 16) ? -1.f : 1.f;
  float o0 = n0 * c0 + sgn * pn0 * s0, o1 = n1 * c1 + sgn * pn1 * s1;
  float o2 = n2 * c2 + sgn * pn2 * s2, o3 = n3 * c3 + sgn * pn3 * s3;
  __nv_fp8x4_e4m3 packed(make_float4(o0, o1, o2, o3));
  *reinterpret_cast<__nv_fp8x4_e4m3*>(dst8 + 4 * lane) = packed;
}

__global__ void q_norm_rope_qstage_prefill_v3_fp8_direct_kernel(
    const __nv_bfloat16* __restrict__ q_pre, const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ cos_v, const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ q_buf, __nv_fp8_e4m3* __restrict__ q8,
    int n_q, int S, int in_row_stride, int out_row_stride, float eps) {
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  for (int head = warp; head < n_q; head += n_warps) {
    norm_rope_warp4_fp8_direct(
        q_pre + (size_t)row * in_row_stride + head * HEAD_DIM, q_norm_w, cos_r, sin_r,
        q_buf == nullptr ? nullptr
                         : q_buf + (size_t)row * out_row_stride + head * HEAD_DIM,
        q8 + ((size_t)row * n_q + head) * HEAD_DIM, lane, eps);
  }
}

__global__ void k_norm_rope_kvwrite_prefill_v3_fp8_direct_kernel(
    const __nv_bfloat16* __restrict__ k_pre, const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_v, const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    __nv_fp8_e4m3* __restrict__ k8, __nv_fp8_e4m3* __restrict__ v8,
    int n_kv, int S, int in_row_stride, int cache_row_stride, float eps) {
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  for (int head = warp; head < n_kv; head += n_warps) {
    norm_rope_warp4_fp8_direct(
        k_pre + (size_t)row * in_row_stride + head * HEAD_DIM, k_norm_w, cos_r, sin_r,
        k_cache == nullptr
            ? nullptr
            : k_cache + (size_t)row * cache_row_stride + head * HEAD_DIM,
        k8 + ((size_t)row * n_kv + head) * HEAD_DIM, lane, eps);
    // V: cast -> e4m3 for the all-fp8 prefill attention, plus the optional
    // bf16 cache write the bf16-attention route needs (null in the fp8 route).
    const __nv_bfloat16* v_row = v_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
    float a0, a1, a2, a3;
    load4_bf16(v_row + 4 * lane, a0, a1, a2, a3);
    if (v_cache != nullptr) {
      __nv_bfloat16* vdst = v_cache + (size_t)row * cache_row_stride + head * HEAD_DIM;
      store4_bf16(vdst + 4 * lane, a0, a1, a2, a3);
    }
    __nv_fp8x4_e4m3 packed(make_float4(a0, a1, a2, a3));
    *reinterpret_cast<__nv_fp8x4_e4m3*>(
        v8 + ((size_t)row * n_kv + head) * HEAD_DIM + 4 * lane) = packed;
  }
}

// Single-launch Q + K/V producer for the all-fp8 prefill route. grid.y = 0
// executes exactly the Q-stage blocks and grid.y = 1 exactly the K/V-write
// blocks of the two standalone kernels (same device function, same thread
// mapping), so the outputs are identical; the two independent launches merge
// into one and neither writes cache-shaped bf16 scratch the fp8 route never
// reads.
__global__ void qk_norm_rope_prefill_v3_fp8_direct_kernel(
    const __nv_bfloat16* __restrict__ q_pre,
    const __nv_bfloat16* __restrict__ k_pre,
    const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_v,
    const __nv_bfloat16* __restrict__ sin_v,
    __nv_fp8_e4m3* __restrict__ q8,
    __nv_fp8_e4m3* __restrict__ k8,
    __nv_fp8_e4m3* __restrict__ v8,
    int n_q, int n_kv, int S, int in_row_stride, float eps) {
  flash_rt::pdl::pdl_trigger();
  flash_rt::pdl::pdl_wait();
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  if (blockIdx.y == 0 && n_q == 2 * n_warps) {
    // Q stage, two heads per warp: every global load of both heads (values,
    // norm weights, RoPE row) is issued before any reduction.
    const int h0 = warp, h1 = warp + n_warps;
    const __nv_bfloat16* r = q_pre + (size_t)row * in_row_stride + 4 * lane;
    float a0, a1, a2, a3, b0, b1, b2, b3, w0, w1, w2, w3, c0, c1, c2, c3, s0, s1, s2, s3;
    load4_bf16(r + h0 * HEAD_DIM, a0, a1, a2, a3);
    load4_bf16(r + h1 * HEAD_DIM, b0, b1, b2, b3);
    load4_bf16(q_norm_w + 4 * lane, w0, w1, w2, w3);
    const int ci = 4 * (lane & 15);
    load4_bf16(cos_r + ci, c0, c1, c2, c3);
    load4_bf16(sin_r + ci, s0, s1, s2, s3);
    norm_rope_regs_fp8(a0, a1, a2, a3, w0, w1, w2, w3, c0, c1, c2, c3, s0, s1, s2, s3,
                       q8 + ((size_t)row * n_q + h0) * HEAD_DIM, lane, eps);
    norm_rope_regs_fp8(b0, b1, b2, b3, w0, w1, w2, w3, c0, c1, c2, c3, s0, s1, s2, s3,
                       q8 + ((size_t)row * n_q + h1) * HEAD_DIM, lane, eps);
  } else if (blockIdx.y == 0) {
    for (int head = warp; head < n_q; head += n_warps) {
      norm_rope_warp4_fp8_direct(
          q_pre + (size_t)row * in_row_stride + head * HEAD_DIM, q_norm_w,
          cos_r, sin_r, nullptr,
          q8 + ((size_t)row * n_q + head) * HEAD_DIM, lane, eps);
    }
  } else if (n_kv == n_warps) {
    // K/V stage, one head per warp: the K values, V values, norm weights and
    // RoPE row are all requested before the K reduction.
    const int head = warp;
    float a0, a1, a2, a3, v0, v1, v2, v3, w0, w1, w2, w3, c0, c1, c2, c3, s0, s1, s2, s3;
    load4_bf16(k_pre + (size_t)row * in_row_stride + head * HEAD_DIM + 4 * lane, a0, a1, a2, a3);
    load4_bf16(v_pre + (size_t)row * in_row_stride + head * HEAD_DIM + 4 * lane, v0, v1, v2, v3);
    load4_bf16(k_norm_w + 4 * lane, w0, w1, w2, w3);
    const int ci = 4 * (lane & 15);
    load4_bf16(cos_r + ci, c0, c1, c2, c3);
    load4_bf16(sin_r + ci, s0, s1, s2, s3);
    norm_rope_regs_fp8(a0, a1, a2, a3, w0, w1, w2, w3, c0, c1, c2, c3, s0, s1, s2, s3,
                       k8 + ((size_t)row * n_kv + head) * HEAD_DIM, lane, eps);
    __nv_fp8x4_e4m3 packed(make_float4(v0, v1, v2, v3));
    *reinterpret_cast<__nv_fp8x4_e4m3*>(
        v8 + ((size_t)row * n_kv + head) * HEAD_DIM + 4 * lane) = packed;
  } else {
    for (int head = warp; head < n_kv; head += n_warps) {
      norm_rope_warp4_fp8_direct(
          k_pre + (size_t)row * in_row_stride + head * HEAD_DIM, k_norm_w,
          cos_r, sin_r, nullptr,
          k8 + ((size_t)row * n_kv + head) * HEAD_DIM, lane, eps);
      // V: cast -> e4m3 for the all-fp8 prefill attention.
      const __nv_bfloat16* v_row =
          v_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
      float a0, a1, a2, a3;
      load4_bf16(v_row + 4 * lane, a0, a1, a2, a3);
      __nv_fp8x4_e4m3 packed(make_float4(a0, a1, a2, a3));
      *reinterpret_cast<__nv_fp8x4_e4m3*>(
          v8 + ((size_t)row * n_kv + head) * HEAD_DIM + 4 * lane) = packed;
    }
  }
}

}  // namespace

int qwen3_qk_norm_rope_prefill_v3_fp8_direct(
    const void* q_pre, const void* k_pre, const void* v_pre,
    const void* q_norm_w, const void* k_norm_w, const void* cos, const void* sin,
    void* q8_dst, void* k8_dst, void* v8_dst,
    int n_q_heads, int n_kv_heads, int S, int in_row_stride, float eps,
    cudaStream_t stream) {
  if (!q_pre || !k_pre || !v_pre || !q_norm_w || !k_norm_w || !cos || !sin
      || !q8_dst || !k8_dst || !v8_dst) return 1;
  if (n_q_heads <= 0 || n_kv_heads <= 0 || S <= 0) return 2;
  flash_rt::pdl::launch(qk_norm_rope_prefill_v3_fp8_direct_kernel, dim3(S, 2), dim3(256), 0,
      stream, flash_rt::pdl::enabled(),
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_fp8_e4m3*>(q8_dst),
      reinterpret_cast<__nv_fp8_e4m3*>(k8_dst),
      reinterpret_cast<__nv_fp8_e4m3*>(v8_dst),
      n_q_heads, n_kv_heads, S, in_row_stride, eps);
  return 0;
}

int qwen3_q_norm_rope_qstage_prefill_v3_fp8(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, void* q8_dst, void* q_scale_dst,
    int n_q_heads, int S, int in_row_stride, int out_row_stride, float eps,
    cudaStream_t stream) {
  // q_buf_dst may be null (fp8 path skips the redundant bf16 Q_buf write).
  if (!q_pre || !q_norm_w || !cos || !sin || !q8_dst || !q_scale_dst)
    return 1;
  if (n_q_heads <= 0 || S <= 0) return 2;
  q_norm_rope_qstage_prefill_v3_fp8_kernel<<<dim3(S), dim3(256), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      reinterpret_cast<__nv_fp8_e4m3*>(q8_dst),
      reinterpret_cast<float*>(q_scale_dst),
      n_q_heads, S, in_row_stride, out_row_stride, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_prefill_v3_fp8(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    void* k8_dst, void* k_scale_dst, void* v_fp16_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride, float eps,
    cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin || !k_cache_dst
      || !v_cache_dst || !k8_dst || !k_scale_dst || !v_fp16_dst) return 1;
  if (n_kv_heads <= 0 || S <= 0) return 2;
  k_norm_rope_kvwrite_prefill_v3_fp8_kernel<<<dim3(S), dim3(256), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      reinterpret_cast<__nv_fp8_e4m3*>(k8_dst),
      reinterpret_cast<float*>(k_scale_dst),
      reinterpret_cast<__half*>(v_fp16_dst),
      n_kv_heads, S, in_row_stride, cache_row_stride, eps);
  return 0;
}

int qwen3_q_norm_rope_qstage_prefill_v3_fp8_direct(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, void* q8_dst,
    int n_q_heads, int S, int in_row_stride, int out_row_stride, float eps,
    cudaStream_t stream) {
  // q_buf_dst may be null (fp8 path skips the redundant bf16 Q_buf write).
  if (!q_pre || !q_norm_w || !cos || !sin || !q8_dst) return 1;
  if (n_q_heads <= 0 || S <= 0) return 2;
  q_norm_rope_qstage_prefill_v3_fp8_direct_kernel<<<dim3(S), dim3(256), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      reinterpret_cast<__nv_fp8_e4m3*>(q8_dst),
      n_q_heads, S, in_row_stride, out_row_stride, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_prefill_v3_fp8_direct(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    void* k8_dst, void* v8_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride, float eps,
    cudaStream_t stream) {
  // k_cache_dst / v_cache_dst are the bf16-attention route's cache-shaped
  // scratch and may be null: the all-fp8 route consumes only k8/v8, and the
  // kernel skips the dead bf16 stores when they are null.
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin || !k8_dst || !v8_dst) return 1;
  if (n_kv_heads <= 0 || S <= 0) return 2;
  k_norm_rope_kvwrite_prefill_v3_fp8_direct_kernel<<<dim3(S), dim3(256), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      reinterpret_cast<__nv_fp8_e4m3*>(k8_dst),
      reinterpret_cast<__nv_fp8_e4m3*>(v8_dst),
      n_kv_heads, S, in_row_stride, cache_row_stride, eps);
  return 0;
}

int qwen3_q_norm_rope_qstage_prefill_bf16(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, int n_q_heads, int S, int in_row_stride,
    int out_row_stride, float eps, cudaStream_t stream) {
  if (!q_pre || !q_norm_w || !cos || !sin || !q_buf_dst) return 1;
  if (n_q_heads <= 0 || S <= 0) return 2;
  q_norm_rope_qstage_prefill_kernel<<<dim3(n_q_heads, S), dim3(THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      n_q_heads, S, in_row_stride, out_row_stride, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_prefill_bf16(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride,
    float eps, cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin
      || !k_cache_dst || !v_cache_dst) return 1;
  if (n_kv_heads <= 0 || S <= 0) return 2;
  k_norm_rope_kvwrite_prefill_kernel<<<dim3(n_kv_heads, S), dim3(THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      n_kv_heads, S, in_row_stride, cache_row_stride, eps);
  return 0;
}

// v3 launches: 32 threads (1 warp) per (head, row), vectorized contiguous-4.
int qwen3_q_norm_rope_qstage_prefill_v3_bf16(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, int n_q_heads, int S, int in_row_stride,
    int out_row_stride, float eps, cudaStream_t stream) {
  if (!q_pre || !q_norm_w || !cos || !sin || !q_buf_dst) return 1;
  if (n_q_heads <= 0 || S <= 0) return 2;
  q_norm_rope_qstage_prefill_v3_kernel<<<dim3(S), dim3(256), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      n_q_heads, S, in_row_stride, out_row_stride, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_prefill_v3_bf16(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride,
    float eps, cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin
      || !k_cache_dst || !v_cache_dst) return 1;
  if (n_kv_heads <= 0 || S <= 0) return 2;
  // n_kv (8) < warps(8); use n_kv warps (256 threads still fine, extra warps idle).
  k_norm_rope_kvwrite_prefill_v3_kernel<<<dim3(S), dim3(256), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      n_kv_heads, S, in_row_stride, cache_row_stride, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_devpos_bf16(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_base, void* v_cache_base,
    const void* cur_pos, int row_elems, int n_kv_heads, float eps,
    cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin
      || !k_cache_base || !v_cache_base || !cur_pos) return 1;
  if (n_kv_heads <= 0) return 2;
  k_norm_rope_kvwrite_devpos_kernel<<<dim3(n_kv_heads), dim3(THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_base),
      reinterpret_cast<__nv_bfloat16*>(v_cache_base),
      reinterpret_cast<const int*>(cur_pos),
      row_elems, n_kv_heads, eps);
  return 0;
}

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

int qwen3_qk_norm_rope_kvwrite_bf16(
    const void* q_pre,
    const void* k_pre,
    const void* v_pre,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         n_q_heads,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream) {
  if (!q_pre || !k_pre || !v_pre || !q_norm_w || !k_norm_w || !cos || !sin
      || !q_buf_dst || !k_cache_dst || !v_cache_dst) return 1;
  if (n_q_heads <= 0 || n_kv_heads <= 0) return 2;
  dim3 grid(n_q_heads + n_kv_heads);
  dim3 block(THREADS);
  qk_norm_rope_kvwrite_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      n_q_heads, n_kv_heads, eps);
  return 0;
}

int qwen3_qk_norm_rope_kvwrite_batched_bf16(
    const void* q_pre,
    const void* k_pre,
    const void* v_pre,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         seq_len,
    int         q_pre_row_elems,
    int         k_pre_row_elems,
    int         v_pre_row_elems,
    int         q_dst_row_elems,
    int         kv_dst_row_elems,
    int         n_q_heads,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream) {
  if (!q_pre || !k_pre || !v_pre || !q_norm_w || !k_norm_w || !cos || !sin
      || !q_buf_dst || !k_cache_dst || !v_cache_dst) return 1;
  if (seq_len <= 0 || n_q_heads <= 0 || n_kv_heads <= 0) return 2;
  dim3 grid(seq_len * (n_q_heads + n_kv_heads));
  dim3 block(THREADS);
  qk_norm_rope_kvwrite_batched_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      seq_len,
      q_pre_row_elems, k_pre_row_elems, v_pre_row_elems,
      q_dst_row_elems, kv_dst_row_elems,
      n_q_heads, n_kv_heads, eps);
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
