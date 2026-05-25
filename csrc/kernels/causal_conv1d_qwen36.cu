// SPDX-License-Identifier: Apache-2.0
//
// Implementation of causal_conv1d_qwen36 — see header for spec.
//
// Layout strategy (multi-token forward):
//   * One thread per (b, s, c). 256 threads/block, swept along
//     the channel axis (innermost → coalesced). Grid (ceil(C/256), S, B).
//   * Each thread reads k bf16 elements from input (x), k bf16 from
//     weights (w), 1 bf16 from bias, 1 fp32 add (silu), 1 bf16 store.
//   * For Qwen3.6 (C=10240, k=4): 4 input loads = 8 bytes per thread,
//     1 store = 2 bytes — purely memory-bound. Coalesced reads at
//     channel granularity hit 100% L1 hit on second token onward.
//
// Layout strategy (single-token decode update):
//   * One thread per (b, c). Each thread reads (k-1) state values +
//     1 new x value, computes dot product, shifts state, stores
//     back. No shared memory; straight register staging.

#include "causal_conv1d_qwen36.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kMaxK = 8;
constexpr int kThreadsX = 256;

__device__ __forceinline__ float silu(float x) {
  // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  return x / (1.0f + __expf(-x));
}

__global__ void causal_conv1d_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int B, int S, int conv_dim, int k,
    bool apply_silu)
{
  const int c = blockIdx.x * kThreadsX + threadIdx.x;
  const int s = blockIdx.y;
  const int b = blockIdx.z;
  if (c >= conv_dim) return;

  // Pre-load weight[c, 0..k-1] into registers (small).
  float wv[kMaxK];
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    wv[i] = (i < k) ? static_cast<float>(w[c * k + i]) : 0.0f;
  }

  float acc = (bias != nullptr) ? static_cast<float>(bias[c]) : 0.0f;

  // y[s, c] = sum_{i=0..k-1} x[s + i - (k-1), c] * w[c, i]
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    if (i < k) {
      const int t = s + i - (k - 1);
      if (t >= 0) {
        const float xv = static_cast<float>(
            x[b * S * conv_dim + t * conv_dim + c]);
        acc = fmaf(xv, wv[i], acc);
      }
    }
  }

  if (apply_silu) acc = silu(acc);
  out[b * S * conv_dim + s * conv_dim + c] = __float2bfloat16(acc);
}

__global__ void causal_conv1d_update_kernel(
    const __nv_bfloat16* __restrict__ x_new,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ state,
    int B, int conv_dim, int k,
    bool apply_silu)
{
  const int c = blockIdx.x * kThreadsX + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= conv_dim) return;

  const int sk = k - 1;  // state holds the last k-1 tokens
  const int state_base = (b * conv_dim + c) * sk;

  // Load weight[c, 0..k-1].
  float wv[kMaxK];
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    wv[i] = (i < k) ? static_cast<float>(w[c * k + i]) : 0.0f;
  }

  // Load state[b, c, 0..sk-1] = previous k-1 tokens (oldest first).
  float sv[kMaxK];
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    sv[i] = (i < sk)
        ? static_cast<float>(state[state_base + i])
        : 0.0f;
  }

  // New token.
  const float x_v = static_cast<float>(x_new[b * conv_dim + c]);

  // y = bias + sum_{i=0..sk-1} state[i] * w[i] + x_new * w[sk]
  float acc = (bias != nullptr) ? static_cast<float>(bias[c]) : 0.0f;
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    if (i < sk) acc = fmaf(sv[i], wv[i], acc);
  }
  acc = fmaf(x_v, wv[sk], acc);

  if (apply_silu) acc = silu(acc);
  out[b * conv_dim + c] = __float2bfloat16(acc);

  // Shift state: state[0..sk-2] = state[1..sk-1]; state[sk-1] = x_new
  #pragma unroll
  for (int i = 0; i < kMaxK - 1; ++i) {
    if (i < sk - 1) {
      state[state_base + i] = __float2bfloat16(sv[i + 1]);
    }
  }
  if (sk >= 1) {
    state[state_base + sk - 1] = __float2bfloat16(x_v);
  }
}

__global__ void causal_conv1d_update_chunk_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ state,
    int B, int S, int conv_dim, int k,
    bool apply_silu)
{
  const int c = blockIdx.x * kThreadsX + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= conv_dim) return;

  const int sk = k - 1;
  const int state_base = (b * conv_dim + c) * sk;

  float wv[kMaxK];
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    wv[i] = (i < k) ? static_cast<float>(w[c * k + i]) : 0.0f;
  }

  float sv[kMaxK];
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    sv[i] = (i < sk)
        ? static_cast<float>(state[state_base + i])
        : 0.0f;
  }

  for (int s = 0; s < S; ++s) {
    const float x_v = static_cast<float>(
        x[(size_t)b * S * conv_dim + (size_t)s * conv_dim + c]);

    float acc = (bias != nullptr) ? static_cast<float>(bias[c]) : 0.0f;
    #pragma unroll
    for (int i = 0; i < kMaxK; ++i) {
      if (i < sk) acc = fmaf(sv[i], wv[i], acc);
    }
    acc = fmaf(x_v, wv[sk], acc);

    if (apply_silu) acc = silu(acc);
    out[(size_t)b * S * conv_dim + (size_t)s * conv_dim + c] =
        __float2bfloat16(acc);

    #pragma unroll
    for (int i = 0; i < kMaxK - 1; ++i) {
      if (i < sk - 1) sv[i] = sv[i + 1];
    }
    if (sk >= 1) {
      sv[sk - 1] = x_v;
    }
  }

  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    if (i < sk) {
      state[state_base + i] = __float2bfloat16(sv[i]);
    }
  }
}

__global__ void causal_conv1d_update_chunk_parallel_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out,
    int B, int S, int conv_dim, int k,
    bool apply_silu)
{
  const int c = blockIdx.x * kThreadsX + threadIdx.x;
  const int s = blockIdx.y;
  const int b = blockIdx.z;
  if (c >= conv_dim) return;

  const int sk = k - 1;
  const int state_base = (b * conv_dim + c) * sk;
  float acc = (bias != nullptr) ? static_cast<float>(bias[c]) : 0.0f;

  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    if (i < k) {
      const int t = s + i - sk;
      float xv = 0.0f;
      if (t >= 0) {
        xv = static_cast<float>(
            x[(size_t)b * S * conv_dim + (size_t)t * conv_dim + c]);
      } else if (t >= -sk) {
        xv = static_cast<float>(state[state_base + (t + sk)]);
      }
      const float wv = static_cast<float>(w[c * k + i]);
      acc = fmaf(xv, wv, acc);
    }
  }

  if (apply_silu) acc = silu(acc);
  out[(size_t)b * S * conv_dim + (size_t)s * conv_dim + c] =
      __float2bfloat16(acc);
}

__global__ void causal_conv1d_update_chunk_parallel_gqa_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ q16,
    __nv_bfloat16* __restrict__ k16,
    __nv_bfloat16* __restrict__ v48,
    int B, int S, int conv_dim, int k,
    bool apply_silu)
{
  const int c = blockIdx.x * kThreadsX + threadIdx.x;
  const int s = blockIdx.y;
  const int b = blockIdx.z;
  if (c >= conv_dim) return;

  const int sk = k - 1;
  const int state_base = (b * conv_dim + c) * sk;
  float acc = (bias != nullptr) ? static_cast<float>(bias[c]) : 0.0f;

  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    if (i < k) {
      const int t = s + i - sk;
      float xv = 0.0f;
      if (t >= 0) {
        xv = static_cast<float>(
            x[(size_t)b * S * conv_dim + (size_t)t * conv_dim + c]);
      } else if (t >= -sk) {
        xv = static_cast<float>(state[state_base + (t + sk)]);
      }
      const float wv = static_cast<float>(w[c * k + i]);
      acc = fmaf(xv, wv, acc);
    }
  }

  if (apply_silu) acc = silu(acc);
  const __nv_bfloat16 y = __float2bfloat16(acc);
  const size_t bs = (size_t)b * S + s;
  if (c < 2048) {
    q16[bs * 2048 + c] = y;
  } else if (c < 4096) {
    k16[bs * 2048 + (c - 2048)] = y;
  } else {
    v48[bs * 6144 + (c - 4096)] = y;
  }
}

__global__ void causal_conv1d_update_chunk_state_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ state,
    int B, int S, int conv_dim, int k)
{
  const int c = blockIdx.x * kThreadsX + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= conv_dim) return;

  const int sk = k - 1;
  const int state_base = (b * conv_dim + c) * sk;
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    if (i < sk) {
      const int t = S - sk + i;
      if (t >= 0) {
        state[state_base + i] = x[
            (size_t)b * S * conv_dim + (size_t)t * conv_dim + c];
      } else {
        state[state_base + i] = state[state_base + S + i];
      }
    }
  }
}

}  // namespace

void causal_conv1d_qwen36_bf16(
    const void* x, const void* w, const void* bias,
    void* out,
    int B, int S, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream)
{
  dim3 grid((conv_dim + kThreadsX - 1) / kThreadsX, S, B);
  dim3 block(kThreadsX);
  causal_conv1d_fwd_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<__nv_bfloat16*>(out),
      B, S, conv_dim, k, apply_silu);
}

void causal_conv1d_qwen36_update_bf16(
    const void* x_new, const void* w, const void* bias,
    void* out, void* state,
    int B, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream)
{
  dim3 grid((conv_dim + kThreadsX - 1) / kThreadsX, B);
  dim3 block(kThreadsX);
  causal_conv1d_update_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_new),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<__nv_bfloat16*>(state),
      B, conv_dim, k, apply_silu);
}

void causal_conv1d_qwen36_update_chunk_bf16(
    const void* x, const void* w, const void* bias,
    void* out, void* state,
    int B, int S, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream)
{
  dim3 grid((conv_dim + kThreadsX - 1) / kThreadsX, B);
  dim3 block(kThreadsX);
  causal_conv1d_update_chunk_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<__nv_bfloat16*>(state),
      B, S, conv_dim, k, apply_silu);
}

void causal_conv1d_qwen36_update_chunk_parallel_bf16(
    const void* x, const void* w, const void* bias,
    void* out, void* state,
    int B, int S, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream)
{
  dim3 conv_grid((conv_dim + kThreadsX - 1) / kThreadsX, S, B);
  dim3 block(kThreadsX);
  causal_conv1d_update_chunk_parallel_kernel<<<
      conv_grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<const __nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      B, S, conv_dim, k, apply_silu);

  dim3 state_grid((conv_dim + kThreadsX - 1) / kThreadsX, B);
  causal_conv1d_update_chunk_state_kernel<<<
      state_grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<__nv_bfloat16*>(state),
      B, S, conv_dim, k);
}

void causal_conv1d_qwen36_update_chunk_parallel_gqa_bf16(
    const void* x, const void* w, const void* bias,
    void* q16, void* k16, void* v48, void* state,
    int B, int S, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream)
{
  dim3 conv_grid((conv_dim + kThreadsX - 1) / kThreadsX, S, B);
  dim3 block(kThreadsX);
  causal_conv1d_update_chunk_parallel_gqa_kernel<<<
      conv_grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<const __nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(q16),
      reinterpret_cast<__nv_bfloat16*>(k16),
      reinterpret_cast<__nv_bfloat16*>(v48),
      B, S, conv_dim, k, apply_silu);

  dim3 state_grid((conv_dim + kThreadsX - 1) / kThreadsX, B);
  causal_conv1d_update_chunk_state_kernel<<<
      state_grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<__nv_bfloat16*>(state),
      B, S, conv_dim, k);
}

// In/out-state variant: reads state from state_in, writes shifted
// state (with x_new appended) to state_out. Same math as the in-place
// kernel; supports K-iter chained per-step state save without an
// extra .copy_(state_save, state) launch per step.
namespace {

__global__ void causal_conv1d_update_inout_kernel(
    const __nv_bfloat16* __restrict__ x_new,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ state_in,
    __nv_bfloat16* __restrict__ state_out,
    int B, int conv_dim, int k,
    bool apply_silu)
{
  const int c = blockIdx.x * kThreadsX + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= conv_dim) return;

  const int sk = k - 1;
  const int state_base = (b * conv_dim + c) * sk;

  float wv[kMaxK];
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    wv[i] = (i < k) ? static_cast<float>(w[c * k + i]) : 0.0f;
  }

  float sv[kMaxK];
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    sv[i] = (i < sk)
        ? static_cast<float>(state_in[state_base + i])
        : 0.0f;
  }

  const float x_v = static_cast<float>(x_new[b * conv_dim + c]);

  float acc = (bias != nullptr) ? static_cast<float>(bias[c]) : 0.0f;
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    if (i < sk) acc = fmaf(sv[i], wv[i], acc);
  }
  acc = fmaf(x_v, wv[sk], acc);

  if (apply_silu) acc = silu(acc);
  out[b * conv_dim + c] = __float2bfloat16(acc);

  #pragma unroll
  for (int i = 0; i < kMaxK - 1; ++i) {
    if (i < sk - 1) {
      state_out[state_base + i] = __float2bfloat16(sv[i + 1]);
    }
  }
  if (sk >= 1) {
    state_out[state_base + sk - 1] = __float2bfloat16(x_v);
  }
}

}  // namespace

void causal_conv1d_qwen36_update_inout_bf16(
    const void* x_new, const void* w, const void* bias,
    void* out,
    const void* state_in, void* state_out,
    int B, int conv_dim, int k,
    bool apply_silu,
    cudaStream_t stream)
{
  dim3 grid((conv_dim + kThreadsX - 1) / kThreadsX, B);
  dim3 block(kThreadsX);
  causal_conv1d_update_inout_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_new),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<const __nv_bfloat16*>(state_in),
      reinterpret_cast<__nv_bfloat16*>(state_out),
      B, conv_dim, k, apply_silu);
}

}  // namespace kernels
}  // namespace flash_rt
