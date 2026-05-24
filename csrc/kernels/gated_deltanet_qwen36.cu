// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet recurrent (single-token decode) kernel.
//
// Block layout: one block per (b, h) where h indexes ``num_v_heads``.
// Within a block, threadIdx.x = t in [0, head_v_dim) owns column t of
// the state matrix state[b, h, :, t] (head_k_dim elements).
//
// Per-thread state column lives in registers (head_k_dim fp32 = 128
// regs/thread on Qwen3.6). Q/K/V are loaded into shared memory once
// per block, then broadcast across threads.

#include "gated_deltanet_qwen36.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHD = 128;   // Qwen3.6 head_k_dim == head_v_dim
constexpr float kEps = 1e-6f;
constexpr int kSplitThreads = 256;

template <int HD>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  // Warp reduce.
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, off);
  }
  // Cross-warp reduce via smem (4 warps for HD=128).
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = val;
  __syncthreads();

  if (warp == 0) {
    val = (lane < (HD / 32)) ? smem[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, off);
    }
    if (lane == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

template <int HD>
__global__ void gated_deltanet_recurrent_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6 (single instantiation)");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  // Smem layout: qs[HD], ks[HD], scratch[8] (warp-reduce buffer).
  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  // Load Q and K to smem (each thread loads its element).
  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  // L2 norm Q and K (in-place in smem).
  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  // Scale Q by 1 / sqrt(HD).
  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  // exp(g_t) and beta_t (broadcast scalars).
  const float g_t =
      __expf(static_cast<float>(g_in[b * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  // Each thread holds column t of state[b, h, :, :] in registers.
  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] =
        static_cast<float>(state[state_h_off + (size_t)i * HD + t]) * g_t;
  }

  // kv_mem[t] = sum_i col[i] * ks[i]
  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  // delta[t] = (V[t] - kv_mem) * beta
  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  // state[i, t] += k[i] * delta
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  // Write back state column.
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }

  // out[t] = sum_i col[i] * qs[i]
  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    // Could template more dims; for Qwen3.6 only HD=128 is needed.
    return;  // silently no-op; caller checks output is unchanged
  }

  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

// In/out-state variant: reads col from state_in, writes updated col
// to state_out (separate buffer). Eliminates the standalone
// .copy_(state_out, state) launch in the K-iter verify loop by
// chaining state_in[k+1] := state_out[k] across iterations. Bit-
// identical to (existing kernel + .copy_) under same inputs because
// the math is unchanged; only the writeback target differs.
namespace {

template <int HD>
__global__ void gated_deltanet_recurrent_inout_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    const __nv_bfloat16* __restrict__ state_in,
    __nv_bfloat16* __restrict__ state_out,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  const float g_t =
      __expf(static_cast<float>(g_in[b * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] =
        static_cast<float>(state_in[state_h_off + (size_t)i * HD + t]) * g_t;
  }

  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_out[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }

  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_inout_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    const void* state_in,
    void*       state_out,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    return;
  }
  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_inout_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(state_in),
      reinterpret_cast<__nv_bfloat16*>(state_out),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

namespace {

template <int HD>
__global__ void gated_deltanet_chunk_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int S,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  float col[HD];
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = static_cast<float>(
        state[state_h_off + (size_t)i * HD + t]);
  }

  for (int s = 0; s < S; ++s) {
    const size_t qkv_off = ((size_t)s * num_v_heads + h) * HD + t;
    qs[t] = static_cast<float>(q_in[qkv_off]);
    ks[t] = static_cast<float>(k_in[qkv_off]);
    __syncthreads();

    if (use_qk_l2norm) {
      float q_sq = qs[t] * qs[t];
      float k_sq = ks[t] * ks[t];
      q_sq = block_reduce_sum<HD>(q_sq, scratch);
      k_sq = block_reduce_sum<HD>(k_sq, scratch);
      const float q_inv = rsqrtf(q_sq + kEps);
      const float k_inv = rsqrtf(k_sq + kEps);
      qs[t] *= q_inv;
      ks[t] *= k_inv;
      __syncthreads();
    }

    qs[t] *= rsqrtf(static_cast<float>(HD));
    __syncthreads();

    const float g_t =
        __expf(static_cast<float>(g_in[s * num_v_heads + h]));
    const float beta_t =
        static_cast<float>(beta_in[s * num_v_heads + h]);

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      col[i] *= g_t;
    }

    float kv_mem = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      kv_mem = fmaf(col[i], ks[i], kv_mem);
    }

    const float v_t = static_cast<float>(v_in[qkv_off]);
    const float delta = (v_t - kv_mem) * beta_t;

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      col[i] = fmaf(ks[i], delta, col[i]);
    }

    float out_t = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      out_t = fmaf(col[i], qs[i], out_t);
    }
    out_[qkv_off] = __float2bfloat16(out_t);

    // Serial decode stores bf16 state after each token, so the next
    // token reads bf16-quantized state. Mirror that quantization point
    // without paying global memory traffic for intermediate states.
    if (s + 1 < S) {
      #pragma unroll 16
      for (int i = 0; i < HD; ++i) {
        col[i] = static_cast<float>(__float2bfloat16(col[i]));
      }
    }
    __syncthreads();
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }
}

template <int HD>
__global__ void gated_deltanet_chunk_smem_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int S,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  extern __shared__ float smem[];
  float* state_s = smem;
  float* qs = state_s + HD * HD;
  float* ks = qs + HD;
  float* scratch = ks + HD;

  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_s[i * HD + t] = static_cast<float>(
        state[state_h_off + (size_t)i * HD + t]);
  }
  __syncthreads();

  for (int s = 0; s < S; ++s) {
    const size_t qkv_off = ((size_t)s * num_v_heads + h) * HD + t;
    qs[t] = static_cast<float>(q_in[qkv_off]);
    ks[t] = static_cast<float>(k_in[qkv_off]);
    __syncthreads();

    if (use_qk_l2norm) {
      float q_sq = qs[t] * qs[t];
      float k_sq = ks[t] * ks[t];
      q_sq = block_reduce_sum<HD>(q_sq, scratch);
      k_sq = block_reduce_sum<HD>(k_sq, scratch);
      const float q_inv = rsqrtf(q_sq + kEps);
      const float k_inv = rsqrtf(k_sq + kEps);
      qs[t] *= q_inv;
      ks[t] *= k_inv;
      __syncthreads();
    }

    qs[t] *= rsqrtf(static_cast<float>(HD));
    __syncthreads();

    const float g_t =
        __expf(static_cast<float>(g_in[s * num_v_heads + h]));
    const float beta_t =
        static_cast<float>(beta_in[s * num_v_heads + h]);

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] *= g_t;
    }

    float kv_mem = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      kv_mem = fmaf(state_s[i * HD + t], ks[i], kv_mem);
    }

    const float v_t = static_cast<float>(v_in[qkv_off]);
    const float delta = (v_t - kv_mem) * beta_t;

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] =
          fmaf(ks[i], delta, state_s[i * HD + t]);
    }

    float out_t = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      out_t = fmaf(state_s[i * HD + t], qs[i], out_t);
    }
    out_[qkv_off] = __float2bfloat16(out_t);

    if (s + 1 < S) {
      #pragma unroll 16
      for (int i = 0; i < HD; ++i) {
        state_s[i * HD + t] =
            static_cast<float>(__float2bfloat16(state_s[i * HD + t]));
      }
    }
    __syncthreads();
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] =
        __float2bfloat16(state_s[i * HD + t]);
  }
}

__global__ void qwen36_lin_split_qkv_broadcast_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    __nv_bfloat16* __restrict__ q48,
    __nv_bfloat16* __restrict__ k48,
    __nv_bfloat16* __restrict__ v48,
    int S)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * 48 * kHD;
  if (idx >= total) return;

  const int t = idx % kHD;
  const int h = (idx / kHD) % 48;
  const int s = idx / (48 * kHD);
  const int src_h = h / 3;
  const size_t row = static_cast<size_t>(s) * 10240;
  q48[idx] = conv_out[row + src_h * kHD + t];
  k48[idx] = conv_out[row + 2048 + src_h * kHD + t];
  v48[idx] = conv_out[row + 4096 + h * kHD + t];
}

__global__ void qwen36_lin_split_qkv_gqa_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    __nv_bfloat16* __restrict__ q16,
    __nv_bfloat16* __restrict__ k16,
    __nv_bfloat16* __restrict__ v48,
    int S)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * 10240;
  if (idx >= total) return;

  const int col = idx % 10240;
  const int row = idx / 10240;
  const __nv_bfloat16 x = conv_out[idx];
  if (col < 2048) {
    q16[static_cast<size_t>(row) * 2048 + col] = x;
  } else if (col < 4096) {
    k16[static_cast<size_t>(row) * 2048 + (col - 2048)] = x;
  } else {
    v48[static_cast<size_t>(row) * 6144 + (col - 4096)] = x;
  }
}

__global__ void qwen36_split_q_gate_kernel(
    const __nv_bfloat16* __restrict__ q_proj,
    __nv_bfloat16* __restrict__ q_pre,
    __nv_bfloat16* __restrict__ gate,
    int S)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * 24 * 256;
  if (idx >= total) return;

  const int t = idx % 256;
  const int h = (idx / 256) % 24;
  const int s = idx / (24 * 256);
  const size_t src = (static_cast<size_t>(s) * 24 + h) * 512 + t;
  q_pre[idx] = q_proj[src];
  gate[idx] = q_proj[src + 256];
}

__global__ void qwen36_gdn_gating_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ neg_exp_A_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ beta_out,
    int S,
    int num_heads)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * num_heads;
  if (idx >= total) return;
  const int h = idx % num_heads;

  const float av = static_cast<float>(a[idx]) + dt_bias[h];
  const float sp = log1pf(__expf(av));
  const float gv = neg_exp_A_log[h] * sp;
  const float bv = static_cast<float>(b[idx]);
  const float beta = 1.0f / (1.0f + __expf(-bv));
  g_out[idx] = __float2bfloat16(gv);
  beta_out[idx] = __float2bfloat16(beta);
}

__global__ void qwen36_gdn_gating_strided_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ neg_exp_A_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ beta_out,
    int S,
    int num_heads,
    int a_stride,
    int b_stride)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * num_heads;
  if (idx >= total) return;
  const int row = idx / num_heads;
  const int h = idx - row * num_heads;

  const float av = static_cast<float>(a[row * a_stride + h]) + dt_bias[h];
  const float sp = log1pf(__expf(av));
  const float gv = neg_exp_A_log[h] * sp;
  const float bv = static_cast<float>(b[row * b_stride + h]);
  const float beta = 1.0f / (1.0f + __expf(-bv));
  g_out[idx] = __float2bfloat16(gv);
  beta_out[idx] = __float2bfloat16(beta);
}

template <int HD>
__global__ void qwen36_gdn_chunk_from_conv_smem_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ a_in,
    const __nv_bfloat16* __restrict__ b_in,
    const float* __restrict__ neg_exp_A_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int S,
    int num_v_heads,
    int a_stride,
    int b_stride,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  extern __shared__ float smem[];
  float* state_s = smem;
  float* qs = state_s + HD * HD;
  float* ks = qs + HD;
  float* scratch = ks + HD;

  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_s[i * HD + t] = static_cast<float>(
        state[state_h_off + (size_t)i * HD + t]);
  }
  __syncthreads();

  const int src_h = h / 3;
  for (int s = 0; s < S; ++s) {
    const size_t row = static_cast<size_t>(s) * 10240;
    const size_t out_off = ((size_t)s * num_v_heads + h) * HD + t;
    qs[t] = static_cast<float>(conv_out[row + src_h * HD + t]);
    ks[t] = static_cast<float>(conv_out[row + 2048 + src_h * HD + t]);
    __syncthreads();

    if (use_qk_l2norm) {
      float q_sq = qs[t] * qs[t];
      float k_sq = ks[t] * ks[t];
      q_sq = block_reduce_sum<HD>(q_sq, scratch);
      k_sq = block_reduce_sum<HD>(k_sq, scratch);
      const float q_inv = rsqrtf(q_sq + kEps);
      const float k_inv = rsqrtf(k_sq + kEps);
      qs[t] *= q_inv;
      ks[t] *= k_inv;
      __syncthreads();
    }

    qs[t] *= rsqrtf(static_cast<float>(HD));
    __syncthreads();

    const float av =
        static_cast<float>(a_in[s * a_stride + h]) + dt_bias[h];
    const float sp = log1pf(__expf(av));
    const float g_log = static_cast<float>(
        __float2bfloat16(neg_exp_A_log[h] * sp));
    const float g_t = __expf(g_log);
    const float bv = static_cast<float>(b_in[s * b_stride + h]);
    const float beta_t = static_cast<float>(
        __float2bfloat16(1.0f / (1.0f + __expf(-bv))));

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] *= g_t;
    }

    float kv_mem = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      kv_mem = fmaf(state_s[i * HD + t], ks[i], kv_mem);
    }

    const float v_t =
        static_cast<float>(conv_out[row + 4096 + h * HD + t]);
    const float delta = (v_t - kv_mem) * beta_t;

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] =
          fmaf(ks[i], delta, state_s[i * HD + t]);
    }

    float out_t = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      out_t = fmaf(state_s[i * HD + t], qs[i], out_t);
    }
    out_[out_off] = __float2bfloat16(out_t);

    if (s + 1 < S) {
      #pragma unroll 16
      for (int i = 0; i < HD; ++i) {
        state_s[i * HD + t] =
            static_cast<float>(__float2bfloat16(state_s[i * HD + t]));
      }
    }
    __syncthreads();
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] =
        __float2bfloat16(state_s[i * HD + t]);
  }
}

}  // namespace

void gated_deltanet_chunk_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int S, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD || S <= 0) {
    return;
  }
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  gated_deltanet_chunk_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, use_qk_l2norm);
}

void qwen36_lin_split_qkv_broadcast_bf16(
    const void* conv_out,
    void*       q48,
    void*       k48,
    void*       v48,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * 48 * kHD;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_lin_split_qkv_broadcast_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<__nv_bfloat16*>(q48),
      reinterpret_cast<__nv_bfloat16*>(k48),
      reinterpret_cast<__nv_bfloat16*>(v48),
      S);
}

void qwen36_lin_split_qkv_gqa_bf16(
    const void* conv_out,
    void*       q16,
    void*       k16,
    void*       v48,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * 10240;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_lin_split_qkv_gqa_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<__nv_bfloat16*>(q16),
      reinterpret_cast<__nv_bfloat16*>(k16),
      reinterpret_cast<__nv_bfloat16*>(v48),
      S);
}

void qwen36_split_q_gate_bf16(
    const void* q_proj,
    void*       q_pre,
    void*       gate,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * 24 * 256;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_split_q_gate_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_proj),
      reinterpret_cast<__nv_bfloat16*>(q_pre),
      reinterpret_cast<__nv_bfloat16*>(gate),
      S);
}

void qwen36_gdn_gating_bf16(
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       g_out,
    void*       beta_out,
    int S,
    int num_heads,
    cudaStream_t stream)
{
  if (S <= 0 || num_heads <= 0) return;
  const int total = S * num_heads;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_gdn_gating_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log,
      dt_bias,
      reinterpret_cast<__nv_bfloat16*>(g_out),
      reinterpret_cast<__nv_bfloat16*>(beta_out),
      S, num_heads);
}

void qwen36_gdn_gating_strided_bf16(
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       g_out,
    void*       beta_out,
    int S,
    int num_heads,
    int a_stride,
    int b_stride,
    cudaStream_t stream)
{
  if (S <= 0 || num_heads <= 0) return;
  const int total = S * num_heads;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  qwen36_gdn_gating_strided_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log,
      dt_bias,
      reinterpret_cast<__nv_bfloat16*>(g_out),
      reinterpret_cast<__nv_bfloat16*>(beta_out),
      S, num_heads, a_stride, b_stride);
}

void qwen36_gdn_chunk_from_conv_smem_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       state,
    void*       out,
    int S,
    int num_v_heads,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  constexpr size_t kSmemBytes =
      (kHD * kHD + 2 * kHD + 32) * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        qwen36_gdn_chunk_from_conv_smem_kernel<kHD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  qwen36_gdn_chunk_from_conv_smem_kernel<kHD><<<
      grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log,
      dt_bias,
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, num_v_heads, num_v_heads, use_qk_l2norm);
}

void qwen36_gdn_chunk_from_conv_smem_strided_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       state,
    void*       out,
    int S,
    int num_v_heads,
    int a_stride,
    int b_stride,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  constexpr size_t kSmemBytes =
      (kHD * kHD + 2 * kHD + 32) * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        qwen36_gdn_chunk_from_conv_smem_kernel<kHD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  qwen36_gdn_chunk_from_conv_smem_kernel<kHD><<<
      grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log,
      dt_bias,
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, a_stride, b_stride, use_qk_l2norm);
}

void gated_deltanet_chunk_smem_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int S, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD || S <= 0) {
    return;
  }
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  constexpr size_t kSmemBytes =
      (kHD * kHD + 2 * kHD + 32) * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        gated_deltanet_chunk_smem_kernel<kHD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  gated_deltanet_chunk_smem_kernel<kHD><<<
      grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, use_qk_l2norm);
}

}  // namespace kernels
}  // namespace flash_rt
