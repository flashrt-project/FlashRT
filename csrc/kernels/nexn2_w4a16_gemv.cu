// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini M=1 W4A16 GEMV (NVFP4 weight x BF16 activation). See header.

#include "kernels/nexn2_w4a16_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

// UE4M3 (FP8 e4m3 magnitude) block-scale -> fp32. 256-entry LUT, host-init.
__device__ __constant__ float c_ue4m3[256];

constexpr int kWarps = 8;                  // 8 output rows / block
constexpr int kThreads = kWarps * 32;      // 256
constexpr int kUnroll = 4;                 // packed-weight loads in flight
                                           // (K=2048 -> 4 blocks/lane; fires once)

// SF swizzle byte offset, identical packing to bf16_weight_to_nvfp4_swizzled
// (rb = row/128, ri = row%128, cb = k_block/4, ci = k_block%4).
__device__ __forceinline__ int sf_off(int rb_ncs, int row_inner,
                                       int k_block) {
  return (rb_ncs + (k_block >> 2)) * 512 + row_inner + (k_block & 3);
}

// One NVFP4 block (16 elements / 8 packed bytes) dotted with 16 bf16 acts.
// FP4 e2m1 -> half2 via the hardware cvt.rn.f16x2.e2m1x2 (no LUT / divergence).
__device__ __forceinline__ float blockdot(uint64_t b_pack,
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

// 1 warp / output row. x staged in smem (bf16), reused across 8 warps.
// UNROLL packed-weight + SFB loads in flight to stay bandwidth-bound.
__global__ void w4a16_matvec_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W,       // (N, K/2)
    const uint8_t* __restrict__ SFB,     // swizzled
    __nv_bfloat16* __restrict__ out,
    float alpha,
    int N, int K, int n_col_super) {
  extern __shared__ __nv_bfloat16 x_sh[];
  const int K_int4 = K >> 3;                 // 8 bf16 per int4
  const int4* x_i4 = reinterpret_cast<const int4*>(x);
  int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
  for (int j = threadIdx.x; j < K_int4; j += kThreads) x_sh_i4[j] = x_i4[j];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (row >= N) return;

  const int K_BLOCKS = K >> 4;               // 16 elems / block
  const uint64_t* w_blk =
      reinterpret_cast<const uint64_t*>(W + (size_t)row * (K >> 1));
  const __nv_bfloat162* x_blk =
      reinterpret_cast<const __nv_bfloat162*>(x_sh);

  // Per-row constant parts of the SF swizzle offset.
  const int rb = row >> 7;
  const int ri = row & 127;
  const int rb_ncs = rb * n_col_super;
  const int row_inner = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;

  float acc = 0.0f;
  int kb = lane;
  const int step = 32 * kUnroll;
  for (; kb + 32 * (kUnroll - 1) < K_BLOCKS; kb += step) {
    uint64_t wv[kUnroll];
    float sf[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) wv[u] = w_blk[kb + 32 * u];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
      sf[u] = c_ue4m3[__ldg(SFB + sf_off(rb_ncs, row_inner, kb + 32 * u))];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
      acc += blockdot(wv[u], x_blk + (size_t)(kb + 32 * u) * 8) * sf[u];
  }
  for (; kb < K_BLOCKS; kb += 32) {
    const float s = c_ue4m3[__ldg(SFB + sf_off(rb_ncs, row_inner, kb))];
    acc += blockdot(w_blk[kb], x_blk + (size_t)kb * 8) * s;
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  if (lane == 0) out[row] = __float2bfloat16(acc * alpha);
}

void init_ue4m3_lut() {
  static bool inited = false;
  if (inited) return;
  inited = true;
  // UE4M3 is UNSIGNED (4-bit exp in bits 3..6, 3-bit mantissa in bits 0..2).
  // Bit 7 is not a sign bit: the quantizer's saturation byte 0xFE decodes
  // (via (v>>3)&0xF, v&7) to e=15,m=6 = 448 and must stay positive.
  float lut[256];
  for (int i = 0; i < 256; ++i) {
    const int e = (i >> 3) & 0xF;
    const int m = i & 0x7;
    if (e == 0) {
      lut[i] = static_cast<float>(m) * std::ldexp(1.0f, -9);   // m/8 * 2^-6
    } else {
      lut[i] = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, e - 7);
    }
  }
  cudaMemcpyToSymbol(c_ue4m3, lut, sizeof(lut));
}

}  // namespace

int nexn2_w4a16_matvec_bf16(
    const void*  x_bf16,
    const void*  W_packed,
    const void*  SFB,
    void*        out,
    int          N,
    int          K,
    float        alpha,
    cudaStream_t stream) {
  if (!x_bf16 || !W_packed || !SFB || !out) return 1;
  if (N <= 0 || K <= 0 || (K & 15) != 0) return 2;
  init_ue4m3_lut();
  const int n_col_super = ((K >> 4) + 3) / 4;
  dim3 grid((N + kWarps - 1) / kWarps);
  dim3 block(kThreads);
  size_t smem = (size_t)K * sizeof(__nv_bfloat16);
  w4a16_matvec_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const uint8_t*>(W_packed),
      reinterpret_cast<const uint8_t*>(SFB),
      reinterpret_cast<__nv_bfloat16*>(out),
      alpha, N, K, n_col_super);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
