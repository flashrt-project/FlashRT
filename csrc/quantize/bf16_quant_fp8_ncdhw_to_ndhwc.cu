// SPDX-License-Identifier: Apache-2.0
// G7.23 v19 — bare quant + NCDHW→NDHWC. See header.

#include "bf16_quant_fp8_ncdhw_to_ndhwc.cuh"

#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr int   kThreadsX = 32;
constexpr int   kThreadsY = 8;
constexpr int   kThreads  = kThreadsX * kThreadsY;
constexpr int   kWBlock   = kThreadsX;     // 32 W per CTA
constexpr int   kPadFp8   = 4;
constexpr float kFp8Max   = 448.0f;

// One CTA per (b, t, h, w_block_of_32). 256 threads = 32 (lane=w-in-block)
// × 8 (c-stripes). Each thread reads its (w, c-stripe) NCDHW bf16
// elements (warp-coalesced over W since W is innermost in NCDHW),
// quantizes, writes FP8 to a smem tile, then the CTA does a coalesced
// uint32_t-vec write to global y in NDHWC layout (C innermost).
__global__ void bf16_quant_fp8_ncdhw_to_ndhwc_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3*       __restrict__ y,
    int B, int C, int T, int H, int W,
    int W_blocks_per_row,
    float inv_act_scale)
{
  extern __shared__ __align__(16) char sm_buf[];
  const int sm_out_stride = C + kPadFp8;
  __nv_fp8_e4m3* sm_out = reinterpret_cast<__nv_fp8_e4m3*>(sm_buf);

  // Decode CTA → (b, t, h, w_block).
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

  const long long stride_C = (long long)T * H * W;
  const long long row_off  = (long long)t * H * W + (long long)h * W;
  const long long b_off    = (long long)b * (long long)C * stride_C;

  // ── Quantize each (w, c) pair → write to smem in NDHWC layout. ──
  if (active) {
    #pragma unroll 1
    for (int c = my_c_start; c < my_c_end; ++c) {
      float xv = __bfloat162float(
          x[b_off + (long long)c * stride_C + row_off + my_w]);
      float q  = fminf(fmaxf(xv * inv_act_scale, -kFp8Max), kFp8Max);
      sm_out[tx * sm_out_stride + c] = __nv_fp8_e4m3(q);
    }
  }
  __syncthreads();

  // ── Coalesced 4-byte vec global write (matches v4's write stage). ──
  const long long y_base = ((long long)b * T * H * W
                          + (long long)t * H * W
                          + (long long)h * W
                          + w_start) * (long long)C;
  const int total_words = kWBlock * (C >> 2);   // C must be %4
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

int bf16_quant_fp8_ncdhw_to_ndhwc(
    const void*  x_bf16,
    void*        y_fp8,
    int B, int C, int T, int H, int W,
    float act_scale,
    cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
  if (act_scale <= 0.f) return -2;
  if ((C & 3) != 0) return -3;          // C must be multiple of 4

  const int W_blocks_per_row = (W + kWBlock - 1) / kWBlock;
  const long long n_ctas =
      (long long)B * T * H * (long long)W_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -4;

  const size_t smem_bytes = (size_t)kWBlock * (C + kPadFp8) * 1;
  // Default smem is 48KB; with C up to 2048 we'd need 65KB → opt in.
  static int s_attr_set = 0;
  if (!s_attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)bf16_quant_fp8_ncdhw_to_ndhwc_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    if (e != cudaSuccess) return -5;
    s_attr_set = 1;
  }
  if (smem_bytes > 99 * 1024) return -6;

  dim3 grid(static_cast<unsigned>(n_ctas));
  dim3 block(kThreads);
  const float inv_act = 1.0f / act_scale;
  bf16_quant_fp8_ncdhw_to_ndhwc_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(y_fp8),
      B, C, T, H, W, W_blocks_per_row, inv_act);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[bare_quant] launch err: %s\n",
                 cudaGetErrorString(e));
    return -10;
  }
  return 0;
}

__global__ void bf16_upsample2x_quant_fp8_nchw_to_nhwc_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3*       __restrict__ y,
    int C, int H, int W, int out_W_blocks_per_row,
    float inv_act_scale)
{
  extern __shared__ __align__(16) char sm_buf[];
  const int sm_out_stride = C + kPadFp8;
  __nv_fp8_e4m3* sm_out = reinterpret_cast<__nv_fp8_e4m3*>(sm_buf);

  const int out_H = H << 1;
  const int out_W = W << 1;
  const int wb = blockIdx.x % out_W_blocks_per_row;
  const int rest = blockIdx.x / out_W_blocks_per_row;
  const int n = rest / out_H;
  const int oh = rest - n * out_H;
  const int ow_start = wb * kWBlock;

  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;
  const int ow = ow_start + tx;
  const bool active = (ow < out_W);
  const int ih = oh >> 1;
  const int iw = ow >> 1;

  const int c_per_y = (C + kThreadsY - 1) / kThreadsY;
  const int c_start = ty * c_per_y;
  const int c_end = min(c_start + c_per_y, C);
  const long long in_base = ((long long)n * C * H * W)
                            + (long long)ih * W + iw;
  const long long stride_C = (long long)H * W;

  if (active) {
    #pragma unroll 1
    for (int c = c_start; c < c_end; ++c) {
      float xv = __bfloat162float(x[in_base + (long long)c * stride_C]);
      float q = fminf(fmaxf(xv * inv_act_scale, -kFp8Max), kFp8Max);
      sm_out[tx * sm_out_stride + c] = __nv_fp8_e4m3(q);
    }
  }
  __syncthreads();

  const long long y_base = ((long long)n * out_H * out_W
                           + (long long)oh * out_W + ow_start)
                           * (long long)C;
  const int total_words = kWBlock * (C >> 2);
  const int tid = threadIdx.x;
  #pragma unroll 1
  for (int idx = tid; idx < total_words; idx += kThreads) {
    int word_per_row = C >> 2;
    int w_off = idx / word_per_row;
    int wd = idx - w_off * word_per_row;
    if (ow_start + w_off < out_W) {
      uint32_t pack = *reinterpret_cast<const uint32_t*>(
          &sm_out[w_off * sm_out_stride + (wd << 2)]);
      *reinterpret_cast<uint32_t*>(
          &y[y_base + (long long)w_off * C + (long long)(wd << 2)]) = pack;
    }
  }
}

int bf16_upsample2x_quant_fp8_nchw_to_nhwc(
    const void*  x_bf16,
    void*        y_fp8,
    int N, int C, int H, int W,
    float act_scale,
    cudaStream_t stream)
{
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return -1;
  if (act_scale <= 0.f) return -2;
  if ((C & 3) != 0) return -3;
  const int out_W = W << 1;
  const int out_H = H << 1;
  const int W_blocks_per_row = (out_W + kWBlock - 1) / kWBlock;
  const long long n_ctas = (long long)N * out_H * W_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -4;
  const size_t smem_bytes = (size_t)kWBlock * (C + kPadFp8) * 1;
  static int s_attr_set = 0;
  if (!s_attr_set) {
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)bf16_upsample2x_quant_fp8_nchw_to_nhwc_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    if (e != cudaSuccess) return -5;
    s_attr_set = 1;
  }
  if (smem_bytes > 99 * 1024) return -6;
  bf16_upsample2x_quant_fp8_nchw_to_nhwc_kernel
      <<<static_cast<unsigned>(n_ctas), kThreads, smem_bytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x_bf16),
          reinterpret_cast<__nv_fp8_e4m3*>(y_fp8),
          C, H, W, W_blocks_per_row, 1.0f / act_scale);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[up2x_quant] launch err: %s\n",
                 cudaGetErrorString(e));
    return -10;
  }
  return 0;
}

}  // namespace quantize
}  // namespace flash_rt
