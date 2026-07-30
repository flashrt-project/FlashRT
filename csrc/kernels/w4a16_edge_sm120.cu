// SPDX-License-Identifier: Apache-2.0
//
// W4A16 GEMV variants for a bandwidth-poor part. See header for what differs
// and why.

#include "kernels/w4a16_edge_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "kernels/fp4_e2m1_compat.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kWarps = 8;                  // 8 output rows / block
constexpr int kThreads = kWarps * 32;      // 256
constexpr int kUnroll = 4;                 // packed-weight loads in flight

// A 16-element NVFP4 block is 16 bf16 of activation, 32 bytes. Held at that
// stride, the eight lanes of a 128-bit shared-load phase land on banks
// 0,8,16,24,0,8,16,24 -- four banks, two-way conflicted. At 48 bytes they land
// on 0,12,24,4,16,28,8,20: eight distinct banks. The 16 spare bytes per block
// cost K/2 bytes of shared memory (12 KB at K=4096) and buy back the 2.41x
// wavefront overhead the conflict was costing.
constexpr int kBlockSlots = 24;            // bf16 slots per 16-element block
constexpr int kBlockInt4 = kBlockSlots / 8;  // 3 int4 per block, 2 used

// UE4M3 -> fp32 without a table.
//
// The value is (1 + m/8) * 2^(e-7) for e > 0, which is exactly an fp32 with
// exponent field e+120 and mantissa m<<20, and m * 2^-9 for e == 0. Four
// integer ops and a select, against a __constant__ load whose index differs
// per lane -- and constant memory serves one address per cycle, so a divergent
// index serialises the warp.
//
// Bit 7 is not a sign bit: UE4M3 is unsigned, and the quantizer's saturation
// byte 0xFE must decode to +448.
__device__ __forceinline__ float ue4m3_to_float(uint32_t v) {
  const uint32_t e = (v >> 3) & 0xFu;
  const uint32_t m = v & 0x7u;
  const float normal = __uint_as_float(((e + 120u) << 23) | (m << 20));
  const float subnormal = static_cast<float>(m) * (1.0f / 512.0f);
  return e == 0u ? subnormal : normal;
}

// SF swizzle byte offset, identical packing to bf16_weight_to_nvfp4_swizzled.
__device__ __forceinline__ int sf_off(int rb_ncs, int row_inner, int k_block) {
  return (rb_ncs + (k_block >> 2)) * 512 + row_inner + (k_block & 3);
}

// One NVFP4 block (16 elements / 8 packed bytes) dotted with 16 bf16 acts.
__device__ __forceinline__ float blockdot(uint64_t b_pack,
                                          const __nv_bfloat162* xb2) {
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const __half2_raw wr = flash_rt::fp4::cvt_e2m1x2_to_halfraw2(
        static_cast<uint8_t>(b_pack >> (j * 8)));
    const float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&wr));
    const float2 xf = __bfloat1622float2(xb2[j]);
    acc = fmaf(wf.x, xf.x, acc);
    acc = fmaf(wf.y, xf.y, acc);
  }
  return acc;
}

// Stage x into the padded shared layout: block b occupies int4 slots
// 3b and 3b+1, leaving 3b+2 as the padding that separates the banks.
__device__ __forceinline__ void stage_padded(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* x_sh, int K) {
  const int4* x_i4 = reinterpret_cast<const int4*>(x);
  int4* sh_i4 = reinterpret_cast<int4*>(x_sh);
  const int n_i4 = K >> 3;                 // 8 bf16 per int4, 2 per block
  for (int j = threadIdx.x; j < n_i4; j += kThreads)
    sh_i4[(j >> 1) * kBlockInt4 + (j & 1)] = x_i4[j];
}

// The K loop, shared by both entry points: 1 warp per output row, kUnroll
// packed-weight loads in flight.
__device__ __forceinline__ float row_dot(
    const uint64_t* __restrict__ w_blk, const uint8_t* __restrict__ SFB,
    const __nv_bfloat16* x_sh, int K_BLOCKS, int rb_ncs, int row_inner,
    int lane) {
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
      sf[u] = ue4m3_to_float(
          __ldg(SFB + sf_off(rb_ncs, row_inner, kb + 32 * u)));
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
      acc += blockdot(
          wv[u], reinterpret_cast<const __nv_bfloat162*>(
                     x_sh + (size_t)(kb + 32 * u) * kBlockSlots)) * sf[u];
  }
  for (; kb < K_BLOCKS; kb += 32) {
    const float s = ue4m3_to_float(
        __ldg(SFB + sf_off(rb_ncs, row_inner, kb)));
    acc += blockdot(
        w_blk[kb], reinterpret_cast<const __nv_bfloat162*>(
                       x_sh + (size_t)kb * kBlockSlots)) * s;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  return acc;
}

__global__ void w4a16_matvec_edge_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ SFB,
    __nv_bfloat16* __restrict__ out,
    float alpha, int N, int K, int n_col_super) {
  extern __shared__ __nv_bfloat16 x_sh[];
  stage_padded(x, x_sh, K);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (row >= N) return;

  const int rb = row >> 7;
  const int ri = row & 127;
  const float acc = row_dot(
      reinterpret_cast<const uint64_t*>(W + (size_t)row * (K >> 1)), SFB,
      x_sh, K >> 4, rb * n_col_super,
      (ri & 31) * 16 + ((ri >> 5) & 3) * 4, lane);
  if (lane == 0) out[row] = __float2bfloat16(acc * alpha);
}

// grid = (ceil(N/8), slots). Block computes 8 output rows of one slot.
__global__ void moe_grouped_w4a16_edge_kernel(
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

  extern __shared__ __nv_bfloat16 x_sh[];
  stage_padded(A_stack + (long)slot * a_stride, x_sh, K);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (row >= N) return;

  const int rb = row >> 7;
  const int ri = row & 127;
  const float acc = row_dot(
      reinterpret_cast<const uint64_t*>(
          W_stack + (long)e * w_stride + (size_t)row * (K >> 1)),
      SFB_stack + (long)e * sfb_stride, x_sh, K >> 4, rb * n_col_super,
      (ri & 31) * 16 + ((ri >> 5) & 3) * 4, lane);
  if (lane == 0)
    D[(long)slot * N + row] = __float2bfloat16(acc * alpha_stack[e]);
}

// Shared memory for the padded stage: kBlockSlots bf16 per 16 elements.
inline size_t smem_bytes(int K) {
  return (size_t)(K >> 4) * kBlockSlots * sizeof(__nv_bfloat16);
}

}  // namespace

int w4a16_matvec_edge_sm120_bf16(
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
  const int n_col_super = ((K >> 4) + 3) / 4;
  w4a16_matvec_edge_kernel<<<dim3((N + kWarps - 1) / kWarps),
                             dim3(kThreads), smem_bytes(K), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const uint8_t*>(W_packed),
      reinterpret_cast<const uint8_t*>(SFB),
      reinterpret_cast<__nv_bfloat16*>(out),
      alpha, N, K, n_col_super);
  return 0;
}

int moe_grouped_w4a16_edge_sm120_bf16(
    const void*  A_stack,
    const void*  W_stack,
    const void*  SFB_stack,
    const void*  alpha_stack,
    const void*  eidx,
    void*        D,
    int          slots,
    int          N,
    int          K,
    long         a_stride,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream) {
  if (!A_stack || !W_stack || !SFB_stack || !alpha_stack || !eidx || !D)
    return 1;
  if (slots <= 0 || N <= 0 || K <= 0 || (K & 15) != 0) return 2;
  const int n_col_super = ((K >> 4) + 3) / 4;
  moe_grouped_w4a16_edge_kernel<<<dim3((N + kWarps - 1) / kWarps, slots),
                                  dim3(kThreads), smem_bytes(K), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A_stack),
      reinterpret_cast<const uint8_t*>(W_stack),
      reinterpret_cast<const uint8_t*>(SFB_stack),
      reinterpret_cast<const float*>(alpha_stack),
      reinterpret_cast<const int*>(eidx),
      reinterpret_cast<__nv_bfloat16*>(D),
      N, K, n_col_super, a_stride, w_stride, sfb_stride);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
