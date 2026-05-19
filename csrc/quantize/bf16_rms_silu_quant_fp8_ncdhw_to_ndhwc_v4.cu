// SPDX-License-Identifier: Apache-2.0
// G7.23 v4 — v3 + x cached in registers (bf162 packed). See header.

#include "bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4.cuh"

#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr int   kThreadsX  = 32;
constexpr int   kThreadsY  = 8;
constexpr int   kThreads   = kThreadsX * kThreadsY;
constexpr int   kWBlock    = kThreadsX;
constexpr int   kPadFp8    = 4;
constexpr int   kMaxBf162  = 64;     // covers c_per_y up to 128 (C=1024)
constexpr float kFp8Max    = 448.0f;

__device__ __forceinline__ float silu_f32(float x) {
  return x * (1.0f / (1.0f + __expf(-x)));
}

__global__ void v4_kernel(
    const __nv_bfloat16* __restrict__ x,        // [B,C,T,H,W]
    const __nv_bfloat16* __restrict__ gamma,    // [C]
    __nv_fp8_e4m3*       __restrict__ y,        // [B,T,H,W,C]
    int B, int C, int T, int H, int W,
    int W_blocks_per_row,
    float inv_act_scale, float eps)
{
  extern __shared__ __align__(16) char sm_buf[];
  const int sm_out_stride = C + kPadFp8;
  __nv_fp8_e4m3* sm_out = reinterpret_cast<__nv_fp8_e4m3*>(sm_buf);
  float* sm_red = reinterpret_cast<float*>(
      reinterpret_cast<char*>(sm_out)
      + (size_t)kWBlock * sm_out_stride * sizeof(__nv_fp8_e4m3));

  const int wb   = blockIdx.x % W_blocks_per_row;
  const int rest = blockIdx.x / W_blocks_per_row;
  const int hwt  = T * H;
  const int b    = rest / hwt;
  const int rh   = rest - b * hwt;
  const int t    = rh / H;
  const int h    = rh - t * H;
  if (b >= B) return;

  const int w_start = wb * kWBlock;

  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;

  const int  my_w   = w_start + tx;
  const bool active = (my_w < W);

  const int c_per_y    = (C + kThreadsY - 1) / kThreadsY;
  const int my_c_start = ty * c_per_y;
  const int my_c_end   = min(my_c_start + c_per_y, C);
  const int my_n_c     = my_c_end - my_c_start;             // ≤ c_per_y
  const int my_n_pair  = (my_n_c + 1) >> 1;                 // bf162 pairs

  const long long stride_C = (long long)T * H * W;
  const long long row_off  = (long long)t * H * W + (long long)h * W;
  const long long b_off    = (long long)b * (long long)C * stride_C;

  // ── Pass 1: read x (HBM) → cache to regs as bf162 pairs, sum_sq accum ──
  // Each thread reads its own c-stripe; values stay in register file
  // (no smem cache). Compiler unrolls the bounded loop and keeps
  // xcache[] in registers.
  __nv_bfloat162 xcache[kMaxBf162];
  float sum_sq = 0.f;

  if (active) {
    #pragma unroll 1
    for (int p = 0; p < my_n_pair; ++p) {
      int c0 = my_c_start + (p << 1);
      int c1 = c0 + 1;
      __nv_bfloat16 v0 = x[b_off + (long long)c0 * stride_C + row_off + my_w];
      __nv_bfloat16 v1 = (c1 < my_c_end)
          ? x[b_off + (long long)c1 * stride_C + row_off + my_w]
          : __float2bfloat16(0.f);
      xcache[p] = __nv_bfloat162{v0, v1};
      float f0 = __bfloat162float(v0);
      float f1 = __bfloat162float(v1);
      sum_sq = fmaf(f0, f0, sum_sq);
      if (c1 < my_c_end) sum_sq = fmaf(f1, f1, sum_sq);
    }
  }

  sm_red[ty * kThreadsX + tx] = active ? sum_sq : 0.f;
  __syncthreads();

  float total_sum_sq = 0.f;
  #pragma unroll
  for (int yi = 0; yi < kThreadsY; ++yi) {
    total_sum_sq += sm_red[yi * kThreadsX + tx];
  }

  const float invC    = 1.0f / static_cast<float>(C);
  const float inv_rms = active ? rsqrtf(total_sum_sq * invC + eps) : 0.f;

  // ── Pass 2: cached x (reg) → RMS·γ·SiLU·quant → sm_out ──
  if (active) {
    #pragma unroll 1
    for (int p = 0; p < my_n_pair; ++p) {
      int c0 = my_c_start + (p << 1);
      int c1 = c0 + 1;
      __nv_bfloat162 vp = xcache[p];
      float xv0 = __bfloat162float(vp.x);
      float xv1 = __bfloat162float(vp.y);
      float gv0 = __bfloat162float(gamma[c0]);

      float n0_f = xv0 * inv_rms * gv0;
      float n0   = __bfloat162float(__float2bfloat16(n0_f));
      float s0_f = silu_f32(n0);
      float s0   = __bfloat162float(__float2bfloat16(s0_f));
      float q0   = fminf(fmaxf(s0 * inv_act_scale, -kFp8Max), kFp8Max);
      sm_out[tx * sm_out_stride + c0] = __nv_fp8_e4m3(q0);

      if (c1 < my_c_end) {
        float gv1 = __bfloat162float(gamma[c1]);
        float n1_f = xv1 * inv_rms * gv1;
        float n1   = __bfloat162float(__float2bfloat16(n1_f));
        float s1_f = silu_f32(n1);
        float s1   = __bfloat162float(__float2bfloat16(s1_f));
        float q1   = fminf(fmaxf(s1 * inv_act_scale, -kFp8Max), kFp8Max);
        sm_out[tx * sm_out_stride + c1] = __nv_fp8_e4m3(q1);
      }
    }
  }
  __syncthreads();

  // ── Coalesced uint32-vec global write (same as v3) ──
  const long long y_base = ((long long)b * T * H * W
                          + (long long)t * H * W
                          + (long long)h * W
                          + w_start) * (long long)C;
  const int total_words = kWBlock * (C >> 2);
  const int tid         = threadIdx.x;

  #pragma unroll 1
  for (int idx = tid; idx < total_words; idx += kThreads) {
    int word_per_row = C >> 2;
    int w_off = idx / word_per_row;
    int wd    = idx - w_off * word_per_row;
    if (w_start + w_off < W) {
      uint32_t pack = *reinterpret_cast<const uint32_t*>(
          &sm_out[w_off * sm_out_stride + (wd << 2)]);
      *reinterpret_cast<uint32_t*>(
          &y[y_base + (long long)w_off * (long long)C
                    + (long long)(wd << 2)]) = pack;
    }
  }
}

}  // namespace

int bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4(
    const void*  x_bf16,
    const void*  gamma_bf16,
    void*        y_fp8,
    int B, int C, int T, int H, int W,
    float act_scale,
    float eps,
    cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
  if (act_scale <= 0.f) return -2;
  if ((C & 3) != 0) return -6;
  // c_per_y = ceil(C/8); kMaxBf162=64 means c_per_y must be ≤ 128 ⇒ C ≤ 1024.
  if (C > 1024) return -7;

  const int W_blocks_per_row = (W + kWBlock - 1) / kWBlock;
  const long long n_ctas =
      (long long)B * T * H * (long long)W_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -3;

  const size_t sm_out_bytes = (size_t)kWBlock * (C + kPadFp8) * 1;
  const size_t sm_red_bytes = (size_t)kThreadsX * kThreadsY * 4;
  const size_t smem_bytes   = sm_out_bytes + sm_red_bytes;

  // 32*1028 + 1024 ≈ 33.9 KB max for C=1024 → fits default 48 KB ceiling.
  // No carveout needed for v4; default smem config is sufficient.

  dim3 grid(static_cast<unsigned>(n_ctas));
  dim3 block(kThreads);
  const float inv_act = 1.0f / act_scale;
  v4_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(gamma_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(y_fp8),
      B, C, T, H, W, W_blocks_per_row, inv_act, eps);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fused_quant_v4] launch err: %s\n",
                 cudaGetErrorString(e));
    return -10;
  }
  return 0;
}

}  // namespace quantize
}  // namespace flash_rt
