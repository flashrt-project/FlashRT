// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE grouped W4A16 GEMV (NVFP4 weight x BF16 act). See header.
// Body mirrors nexn2_w4a16_gemv.cu with per-slot/per-expert base pointers.

#include "kernels/nexn2_moe_grouped_w4a16.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

// UE4M3 (unsigned; bit7 is not a sign bit) -> fp32, host-init.
__device__ __constant__ float c_ue4m3_g[256];

constexpr int gWarps = 8;                  // 8 output rows / block
constexpr int gThreads = gWarps * 32;      // 256
constexpr int gUnroll = 4;

__device__ __forceinline__ int sf_off_g(int rb_ncs, int row_inner,
                                         int k_block) {
  return (rb_ncs + (k_block >> 2)) * 512 + row_inner + (k_block & 3);
}

__device__ __forceinline__ float blockdot_g(uint64_t b_pack,
                                             const __nv_bfloat162* xb2) {
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const __nv_fp4x2_storage_t bb =
        static_cast<__nv_fp4x2_storage_t>(b_pack >> (j * 8));
    const __half2_raw wr = __nv_cvt_fp4x2_to_halfraw2(bb, __NV_E2M1);
    const float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&wr));
    const float2 xf = __bfloat1622float2(xb2[j]);
    acc = fmaf(wf.x, xf.x, acc);
    acc = fmaf(wf.y, xf.y, acc);
  }
  return acc;
}

// grid = (ceil(N/8), slots). Block computes 8 output rows of one slot.
__global__ void moe_grouped_w4a16_kernel(
    const __nv_bfloat16* __restrict__ A_stack,
    const uint8_t* __restrict__ W_stack,
    const uint8_t* __restrict__ SFB_stack,
    const float* __restrict__ alpha_stack,
    const int* __restrict__ expert_idx,
    __nv_bfloat16* __restrict__ D,
    int N, int K, int n_col_super,
    long a_stride, long w_stride, long sfb_stride) {
  const int slot = blockIdx.y;
  const int e = expert_idx[slot];
  const __nv_bfloat16* x = A_stack + (long)slot * a_stride;
  const uint8_t* W = W_stack + (long)e * w_stride;
  const uint8_t* SFB = SFB_stack + (long)e * sfb_stride;
  const float alpha = alpha_stack[e];
  __nv_bfloat16* out = D + (long)slot * N;

  extern __shared__ __nv_bfloat16 x_sh[];
  const int K_int4 = K >> 3;
  const int4* x_i4 = reinterpret_cast<const int4*>(x);
  int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
  for (int j = threadIdx.x; j < K_int4; j += gThreads) x_sh_i4[j] = x_i4[j];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * gWarps + (threadIdx.x >> 5);
  if (row >= N) return;

  const int K_BLOCKS = K >> 4;
  const uint64_t* w_blk =
      reinterpret_cast<const uint64_t*>(W + (size_t)row * (K >> 1));
  const __nv_bfloat162* x_blk = reinterpret_cast<const __nv_bfloat162*>(x_sh);

  const int rb = row >> 7;
  const int ri = row & 127;
  const int rb_ncs = rb * n_col_super;
  const int row_inner = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;

  float acc = 0.0f;
  int kb = lane;
  const int step = 32 * gUnroll;
  for (; kb + 32 * (gUnroll - 1) < K_BLOCKS; kb += step) {
    uint64_t wv[gUnroll];
    float sf[gUnroll];
#pragma unroll
    for (int u = 0; u < gUnroll; ++u) wv[u] = w_blk[kb + 32 * u];
#pragma unroll
    for (int u = 0; u < gUnroll; ++u)
      sf[u] = c_ue4m3_g[__ldg(SFB + sf_off_g(rb_ncs, row_inner, kb + 32 * u))];
#pragma unroll
    for (int u = 0; u < gUnroll; ++u)
      acc += blockdot_g(wv[u], x_blk + (size_t)(kb + 32 * u) * 8) * sf[u];
  }
  for (; kb < K_BLOCKS; kb += 32) {
    const float s = c_ue4m3_g[__ldg(SFB + sf_off_g(rb_ncs, row_inner, kb))];
    acc += blockdot_g(w_blk[kb], x_blk + (size_t)kb * 8) * s;
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  if (lane == 0) out[row] = __float2bfloat16(acc * alpha);
}

void init_ue4m3_g() {
  static bool inited = false;
  if (inited) return;
  inited = true;
  float lut[256];
  for (int i = 0; i < 256; ++i) {
    const int ee = (i >> 3) & 0xF;
    const int m = i & 0x7;
    lut[i] = (ee == 0)
        ? static_cast<float>(m) * std::ldexp(1.0f, -9)
        : (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, ee - 7);
  }
  cudaMemcpyToSymbol(c_ue4m3_g, lut, sizeof(lut));
}

}  // namespace

int nexn2_moe_grouped_w4a16_bf16(
    const void*  A_stack,
    const void*  W_stack,
    const void*  SFB_stack,
    const void*  alpha_stack,
    const void*  expert_idx,
    void*        D,
    int          slots,
    int          N,
    int          K,
    long         a_stride,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream) {
  if (!A_stack || !W_stack || !SFB_stack || !alpha_stack || !expert_idx || !D)
    return 1;
  if (slots <= 0 || N <= 0 || K <= 0 || (K & 15) != 0) return 2;
  init_ue4m3_g();
  const int n_col_super = ((K >> 4) + 3) / 4;
  dim3 grid((N + gWarps - 1) / gWarps, slots);
  dim3 block(gThreads);
  size_t smem = (size_t)K * sizeof(__nv_bfloat16);
  moe_grouped_w4a16_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A_stack),
      reinterpret_cast<const uint8_t*>(W_stack),
      reinterpret_cast<const uint8_t*>(SFB_stack),
      reinterpret_cast<const float*>(alpha_stack),
      reinterpret_cast<const int*>(expert_idx),
      reinterpret_cast<__nv_bfloat16*>(D),
      N, K, n_col_super, a_stride, w_stride, sfb_stride);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
