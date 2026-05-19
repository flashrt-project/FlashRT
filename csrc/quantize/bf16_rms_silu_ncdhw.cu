// SPDX-License-Identifier: Apache-2.0
// Fused BF16 NCDHW RMSNorm + SiLU for Motus VAE T=1 BF16 fallback sites.

#include "bf16_rms_silu_ncdhw.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {
namespace {

constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;
constexpr int kThreads = kThreadsX * kThreadsY;
constexpr int kMaxBf162 = 64;  // C <= 1024 with 8 y-lanes.

__device__ __forceinline__ float silu_f32(float x) {
  return x * (1.0f / (1.0f + __expf(-x)));
}

__global__ void rms_silu_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ prev_cache,
    __nv_bfloat16* __restrict__ next_cache,
    int B, int C, int T, int H, int W,
    int W_blocks_per_row,
    float eps)
{
  __shared__ float sm_red[kThreads];

  const int wb = blockIdx.x % W_blocks_per_row;
  const int rest = blockIdx.x / W_blocks_per_row;
  const int hwt = T * H;
  const int b = rest / hwt;
  const int rh = rest - b * hwt;
  const int t = rh / H;
  const int h = rh - t * H;
  if (b >= B) return;

  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;
  const int w = wb * kThreadsX + tx;
  const bool active = (w < W);

  const int c_per_y = (C + kThreadsY - 1) / kThreadsY;
  const int c_start = ty * c_per_y;
  const int c_end = min(c_start + c_per_y, C);
  const int n_c = c_end - c_start;
  const int n_pair = (n_c + 1) >> 1;

  const long long stride_C = (long long)T * H * W;
  const long long row_off = (long long)t * H * W + (long long)h * W + w;
  const long long b_off = (long long)b * C * stride_C;
  const int hw = h * W + w;

  __nv_bfloat162 xcache[kMaxBf162];
  float sum_sq = 0.0f;

  if (active) {
    #pragma unroll 1
    for (int p = 0; p < n_pair; ++p) {
      int c0 = c_start + (p << 1);
      int c1 = c0 + 1;
      __nv_bfloat16 v0 = x[b_off + (long long)c0 * stride_C + row_off];
      __nv_bfloat16 v1 = (c1 < c_end)
          ? x[b_off + (long long)c1 * stride_C + row_off]
          : __float2bfloat16(0.0f);
      xcache[p] = __nv_bfloat162{v0, v1};
      float f0 = __bfloat162float(v0);
      float f1 = __bfloat162float(v1);
      sum_sq = fmaf(f0, f0, sum_sq);
      if (c1 < c_end) sum_sq = fmaf(f1, f1, sum_sq);
    }
  }

  sm_red[ty * kThreadsX + tx] = active ? sum_sq : 0.0f;
  __syncthreads();

  float total_sum_sq = 0.0f;
  #pragma unroll
  for (int yi = 0; yi < kThreadsY; ++yi) {
    total_sum_sq += sm_red[yi * kThreadsX + tx];
  }

  const float inv_rms = active
      ? rsqrtf(total_sum_sq * (1.0f / static_cast<float>(C)) + eps)
      : 0.0f;

  if (active) {
    #pragma unroll 1
    for (int p = 0; p < n_pair; ++p) {
      int c0 = c_start + (p << 1);
      int c1 = c0 + 1;
      __nv_bfloat162 vp = xcache[p];

      float n0_f = __bfloat162float(vp.x) * inv_rms
          * __bfloat162float(gamma[c0]);
      float n0 = __bfloat162float(__float2bfloat16(n0_f));
      float s0 = __bfloat162float(__float2bfloat16(silu_f32(n0)));
      y[b_off + (long long)c0 * stride_C + row_off] =
          __float2bfloat16(s0);
      if (next_cache != nullptr) {
        const long long HW = (long long)H * W;
        int cache_t = -1;
        if (T >= 2 && t >= T - 2) {
          cache_t = t - (T - 2);
        } else if (T == 1) {
          cache_t = 1;
          long long prev_dst = (((b * C + c0) * 2LL) * HW) + hw;
          next_cache[prev_dst] = (prev_cache != nullptr)
              ? prev_cache[prev_dst + HW]
              : __float2bfloat16(0.0f);
        }
        if (cache_t >= 0) {
          long long dst = (((b * C + c0) * 2LL + cache_t) * HW) + hw;
          next_cache[dst] = __float2bfloat16(s0);
        }
      }

      if (c1 < c_end) {
        float n1_f = __bfloat162float(vp.y) * inv_rms
            * __bfloat162float(gamma[c1]);
        float n1 = __bfloat162float(__float2bfloat16(n1_f));
        float s1 = __bfloat162float(__float2bfloat16(silu_f32(n1)));
        y[b_off + (long long)c1 * stride_C + row_off] =
            __float2bfloat16(s1);
        if (next_cache != nullptr) {
          const long long HW = (long long)H * W;
          int cache_t = -1;
          if (T >= 2 && t >= T - 2) {
            cache_t = t - (T - 2);
          } else if (T == 1) {
            cache_t = 1;
            long long prev_dst = (((b * C + c1) * 2LL) * HW) + hw;
            next_cache[prev_dst] = (prev_cache != nullptr)
                ? prev_cache[prev_dst + HW]
                : __float2bfloat16(0.0f);
          }
          if (cache_t >= 0) {
            long long dst = (((b * C + c1) * 2LL + cache_t) * HW) + hw;
            next_cache[dst] = __float2bfloat16(s1);
          }
        }
      }
    }
  }
}

__global__ void rms_norm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    int B, int C, int T, int H, int W,
    int W_blocks_per_row,
    float eps)
{
  __shared__ float sm_red[kThreads];

  const int wb = blockIdx.x % W_blocks_per_row;
  const int rest = blockIdx.x / W_blocks_per_row;
  const int hwt = T * H;
  const int b = rest / hwt;
  const int rh = rest - b * hwt;
  const int t = rh / H;
  const int h = rh - t * H;
  if (b >= B) return;

  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;
  const int w = wb * kThreadsX + tx;
  const bool active = (w < W);

  const int c_per_y = (C + kThreadsY - 1) / kThreadsY;
  const int c_start = ty * c_per_y;
  const int c_end = min(c_start + c_per_y, C);
  const int n_c = c_end - c_start;
  const int n_pair = (n_c + 1) >> 1;

  const long long stride_C = (long long)T * H * W;
  const long long row_off = (long long)t * H * W + (long long)h * W + w;
  const long long b_off = (long long)b * C * stride_C;

  __nv_bfloat162 xcache[kMaxBf162];
  float sum_sq = 0.0f;

  if (active) {
    #pragma unroll 1
    for (int p = 0; p < n_pair; ++p) {
      int c0 = c_start + (p << 1);
      int c1 = c0 + 1;
      __nv_bfloat16 v0 = x[b_off + (long long)c0 * stride_C + row_off];
      __nv_bfloat16 v1 = (c1 < c_end)
          ? x[b_off + (long long)c1 * stride_C + row_off]
          : __float2bfloat16(0.0f);
      xcache[p] = __nv_bfloat162{v0, v1};
      float f0 = __bfloat162float(v0);
      float f1 = __bfloat162float(v1);
      sum_sq = fmaf(f0, f0, sum_sq);
      if (c1 < c_end) sum_sq = fmaf(f1, f1, sum_sq);
    }
  }

  sm_red[ty * kThreadsX + tx] = active ? sum_sq : 0.0f;
  __syncthreads();

  float total_sum_sq = 0.0f;
  #pragma unroll
  for (int yi = 0; yi < kThreadsY; ++yi) {
    total_sum_sq += sm_red[yi * kThreadsX + tx];
  }

  const float inv_rms = active
      ? rsqrtf(total_sum_sq * (1.0f / static_cast<float>(C)) + eps)
      : 0.0f;

  if (active) {
    #pragma unroll 1
    for (int p = 0; p < n_pair; ++p) {
      int c0 = c_start + (p << 1);
      int c1 = c0 + 1;
      __nv_bfloat162 vp = xcache[p];

      float v0 = __bfloat162float(vp.x) * inv_rms
          * __bfloat162float(gamma[c0]);
      if (bias != nullptr) {
        v0 += __bfloat162float(bias[c0]);
      }
      y[b_off + (long long)c0 * stride_C + row_off] =
          __float2bfloat16(v0);

      if (c1 < c_end) {
        float v1 = __bfloat162float(vp.y) * inv_rms
            * __bfloat162float(gamma[c1]);
        if (bias != nullptr) {
          v1 += __bfloat162float(bias[c1]);
        }
        y[b_off + (long long)c1 * stride_C + row_off] =
            __float2bfloat16(v1);
      }
    }
  }
}

__global__ void pack_t1_cache3_nchw_cl_kernel(
    const __nv_bfloat16* __restrict__ prev,
    const __nv_bfloat16* __restrict__ cur,
    __nv_bfloat16* __restrict__ out,
    int C, int H, int W,
    long long n)
{
  for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       idx < n; idx += (long long)blockDim.x * gridDim.x) {
    const int c3 = (int)(idx % (3LL * C));
    const long long wh = idx / (3LL * C);
    const int w = (int)(wh % W);
    const int h = (int)((wh / W) % H);
    const int plane = c3 / C;
    const int c = c3 - plane * C;
    const long long hw = (long long)h * W + w;
    if (plane < 2) {
      out[idx] = prev[((long long)c * 2 + plane) * H * W + hw];
    } else {
      out[idx] = cur[(long long)c * H * W + hw];
    }
  }
}

}  // namespace

int bf16_rms_silu_ncdhw(
    const void* x_bf16,
    const void* gamma_bf16,
    void* y_bf16,
    const void* prev_cache_bf16,
    void* next_cache_bf16,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
  if ((C & 1) != 0) return -2;
  if (C > 1024) return -3;

  const int W_blocks_per_row = (W + kThreadsX - 1) / kThreadsX;
  const long long n_ctas =
      (long long)B * T * H * (long long)W_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -4;

  rms_silu_kernel<<<static_cast<unsigned>(n_ctas), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(gamma_bf16),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(prev_cache_bf16),
      reinterpret_cast<__nv_bfloat16*>(next_cache_bf16),
      B, C, T, H, W, W_blocks_per_row, eps);
  return static_cast<int>(cudaGetLastError());
}

int bf16_rms_norm_ncdhw(
    const void* x_bf16,
    const void* gamma_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
  if ((C & 1) != 0) return -2;
  if (C > 1024) return -3;

  const int W_blocks_per_row = (W + kThreadsX - 1) / kThreadsX;
  const long long n_ctas =
      (long long)B * T * H * (long long)W_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -4;

  rms_norm_kernel<<<static_cast<unsigned>(n_ctas), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(gamma_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      B, C, T, H, W, W_blocks_per_row, eps);
  return static_cast<int>(cudaGetLastError());
}

int bf16_pack_t1_cache3_nchw_channels_last(
    const void* prev_cache_bf16,
    const void* cur_bf16,
    void* out_bf16,
    int C, int H, int W,
    cudaStream_t stream)
{
  if (!prev_cache_bf16 || !cur_bf16 || !out_bf16) return -1;
  if (C <= 0 || H <= 0 || W <= 0) return -2;
  const long long n = 3LL * C * H * W;
  if (n <= 0) return -3;
  const int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 4096) grid = 4096;
  pack_t1_cache3_nchw_cl_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(prev_cache_bf16),
      reinterpret_cast<const __nv_bfloat16*>(cur_bf16),
      reinterpret_cast<__nv_bfloat16*>(out_bf16),
      C, H, W, n);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace quantize
}  // namespace flash_rt
