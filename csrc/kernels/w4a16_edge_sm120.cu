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

constexpr int kWarps = 2;                  // output-row groups per block
constexpr int kThreads = kWarps * 32;      // 256
// Packed-weight loads in flight per row.
//
// The default is four, which is the value the SM120 path was validated with.
// Thor (sm_110) measures faster at two: ncu puts this kernel at 121 registers a
// thread there, which caps it at 8 blocks per SM when shared memory, warps and
// the SM limit all allow 24 -- Block Limit Registers is the only binding one,
// and achieved occupancy is 31%. wv[R][kUnroll] alone is R * kUnroll eight-byte
// values, 64 registers at R=8, so halving it buys back the warps. Swept in the
// captured decode step on that part, one build each:
//
//     kUnroll   1        2        3        4
//     step      10.025   9.743    10.360   10.384 ms
//     tok/s     99.7     102.6    96.5     96.3
//
// That trade is a property of a 20-SM part with 244 GB/s, so it is set per
// architecture in CMake rather than globally -- a device with far more SMs and
// bandwidth may well prefer the deeper per-thread parallelism, and this branch
// has no measurement for one.
//
// The accumulation order does not move either way: the main loop advances by
// 32*kUnroll and the tail takes the remainder, so a lane visits the same
// k-blocks in the same sequence for any kUnroll and the result is bit-identical.
#ifndef FLASHRT_W4A16_EDGE_UNROLL
#define FLASHRT_W4A16_EDGE_UNROLL 4
#endif
constexpr int kUnroll = FLASHRT_W4A16_EDGE_UNROLL;

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

// The K loop, shared by both entry points: R output rows per warp, kUnroll
// packed-weight loads per row in flight.
//
// R exists because a warp with one row does not have enough memory-level
// parallelism on this part. In situ the dominant stall is the global-load
// dependency (long scoreboard, 5.8-13 cycles per issued instruction) while the
// ALU pipe sits at 27-50%: the loop is waiting on memory it has not asked for
// yet. Each lane keeps R*kUnroll eight-byte loads outstanding instead of
// kUnroll, and the K=512 shapes -- where K_BLOCKS is exactly 32, so the
// unrolled body never runs and the tail leaves ONE load in flight -- get the
// whole factor from R.
//
// The rows a warp takes are consecutive and 32-aligned by construction, so
// their scale offsets differ by a constant and cost no extra registers. The
// per-row arithmetic is untouched: same lane-to-block mapping, same order, same
// reduction, so the result is bit-identical to R = 1.
template <int R>
__device__ __forceinline__ void row_dot(
    const uint64_t* __restrict__ w_row0, size_t row_stride_u64,
    const uint8_t* __restrict__ SFB, const __nv_bfloat16* x_sh,
    int K_BLOCKS, int rb_ncs, int row_inner, int lane, float (&acc)[R]) {
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;

  int kb = lane;
  const int step = 32 * kUnroll;
  for (; kb + 32 * (kUnroll - 1) < K_BLOCKS; kb += step) {
    uint64_t wv[R][kUnroll];
    float sf[R][kUnroll];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int u = 0; u < kUnroll; ++u)
        wv[r][u] = w_row0[r * row_stride_u64 + kb + 32 * u];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int u = 0; u < kUnroll; ++u)
        sf[r][u] = ue4m3_to_float(__ldg(
            SFB + sf_off(rb_ncs, row_inner + 16 * r, kb + 32 * u)));
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int u = 0; u < kUnroll; ++u)
        acc[r] += blockdot(
            wv[r][u], reinterpret_cast<const __nv_bfloat162*>(
                          x_sh + (size_t)(kb + 32 * u) * kBlockSlots))
            * sf[r][u];
  }
  for (; kb < K_BLOCKS; kb += 32) {
    uint64_t wv[R];
    float sf[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wv[r] = w_row0[r * row_stride_u64 + kb];
#pragma unroll
    for (int r = 0; r < R; ++r)
      sf[r] = ue4m3_to_float(
          __ldg(SFB + sf_off(rb_ncs, row_inner + 16 * r, kb)));
#pragma unroll
    for (int r = 0; r < R; ++r)
      acc[r] += blockdot(
          wv[r], reinterpret_cast<const __nv_bfloat162*>(
                     x_sh + (size_t)kb * kBlockSlots)) * sf[r];
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[r] += __shfl_xor_sync(0xffffffff, acc[r], off);
}

// Rows per warp. More rows means more outstanding loads and more registers, so
// the useful value is where the added parallelism stops paying for the
// occupancy it costs -- and that turned out to differ between the two entry
// points, which is why they do not share a constant. Measured at the shapes
// the decode issues, cold: the dense GEMV peaks at 2 (q_proj 47.9 us against
// 53.9 at 4, lm_head 1205 against 1366) while the grouped one peaks at 4
// (gate_up 42.9 against 47.8 at 2). The grouped launch carries a slot per grid
// row, so it has fewer blocks per row tile and leans harder on what each
// thread keeps in flight.
constexpr int kRowsDense = 2;    // K >= 2048: kUnroll fires, 2 * 4 = 8
constexpr int kRowsGrouped = 4;  // K >= 2048: 4 * 4 = 16
constexpr int kRowsSmall = 8;    // K <  2048: tail only, 8 * 1 = 8

// The row block a warp owns must not straddle a 32-row scale group, or
// row_inner + 16 * r stops describing the swizzle. Warps take R consecutive
// rows starting at a multiple of R, so this holds for any R dividing 32.
static_assert(32 % kRowsDense == 0 && 32 % kRowsGrouped == 0
                  && 32 % kRowsSmall == 0,
              "rows per warp must divide the 32-row scale group");

template <int R>
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
  const int row0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;

  const int rb = row0 >> 7;
  const int ri = row0 & 127;
  float acc[R];
  row_dot<R>(
      reinterpret_cast<const uint64_t*>(W + (size_t)row0 * (K >> 1)),
      (size_t)(K >> 1) / 8, SFB, x_sh, K >> 4, rb * n_col_super,
      (ri & 31) * 16 + ((ri >> 5) & 3) * 4, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (row0 + r < N) out[row0 + r] = __float2bfloat16(acc[r] * alpha);
  }
}

// grid = (ceil(N/(8*R)), slots). Block computes 8*R output rows of one slot.
template <int R>
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
  const int row0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;

  const int rb = row0 >> 7;
  const int ri = row0 & 127;
  float acc[R];
  row_dot<R>(
      reinterpret_cast<const uint64_t*>(
          W_stack + (long)e * w_stride + (size_t)row0 * (K >> 1)),
      (size_t)(K >> 1) / 8, SFB_stack + (long)e * sfb_stride, x_sh, K >> 4,
      rb * n_col_super, (ri & 31) * 16 + ((ri >> 5) & 3) * 4, lane, acc);
  if (lane == 0) {
    const float a = alpha_stack[e];
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (row0 + r < N)
        D[(long)slot * N + row0 + r] = __float2bfloat16(acc[r] * a);
  }
}

// Shared memory for the padded stage: kBlockSlots bf16 per 16 elements.
inline size_t smem_bytes(int K) {
  return (size_t)(K >> 4) * kBlockSlots * sizeof(__nv_bfloat16);
}

// Rows per warp for this K, dropping to 1 when N cannot fill even one warp's
// worth. Above 32 rows a warp would straddle a scale group; the constants
// enforce that, this only picks between them.
inline int rows_per_warp(int N, int K, int rows_big) {
  const int r = (K >= 2048) ? rows_big : kRowsSmall;
  return (N >= kWarps * r) ? r : 1;
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
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x_bf16);
  const auto* wp = reinterpret_cast<const uint8_t*>(W_packed);
  const auto* sp = reinterpret_cast<const uint8_t*>(SFB);
  auto* op = reinterpret_cast<__nv_bfloat16*>(out);
  const size_t smem = smem_bytes(K);
#define FLASHRT_LAUNCH_MATVEC(R)                                              \
  w4a16_matvec_edge_kernel<R><<<dim3((N + kWarps * (R) - 1) / (kWarps * (R))),\
                                dim3(kThreads), smem, stream>>>(              \
      xp, wp, sp, op, alpha, N, K, n_col_super)
  switch (rows_per_warp(N, K, kRowsDense)) {
    case kRowsDense: FLASHRT_LAUNCH_MATVEC(kRowsDense); break;
    case kRowsSmall: FLASHRT_LAUNCH_MATVEC(kRowsSmall); break;
    default:         FLASHRT_LAUNCH_MATVEC(1);          break;
  }
#undef FLASHRT_LAUNCH_MATVEC
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
  const auto* ap = reinterpret_cast<const __nv_bfloat16*>(A_stack);
  const auto* wp = reinterpret_cast<const uint8_t*>(W_stack);
  const auto* sp = reinterpret_cast<const uint8_t*>(SFB_stack);
  const auto* alp = reinterpret_cast<const float*>(alpha_stack);
  const auto* ep = reinterpret_cast<const int*>(eidx);
  auto* dp = reinterpret_cast<__nv_bfloat16*>(D);
  const size_t smem = smem_bytes(K);
#define FLASHRT_LAUNCH_GROUPED(R)                                             \
  moe_grouped_w4a16_edge_kernel<R>                                            \
      <<<dim3((N + kWarps * (R) - 1) / (kWarps * (R)), slots),                \
         dim3(kThreads), smem, stream>>>(                                     \
          ap, wp, sp, alp, ep, dp, N, K, n_col_super,                         \
          a_stride, w_stride, sfb_stride)
  switch (rows_per_warp(N, K, kRowsGrouped)) {
    case kRowsGrouped: FLASHRT_LAUNCH_GROUPED(kRowsGrouped); break;
    case kRowsSmall:   FLASHRT_LAUNCH_GROUPED(kRowsSmall);   break;
    default:           FLASHRT_LAUNCH_GROUPED(1);            break;
  }
#undef FLASHRT_LAUNCH_GROUPED
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
