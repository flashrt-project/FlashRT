#include "w4a16_mrows_edge_sm120.cuh"

#include "fp4_e2m1_compat.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

namespace {

// Same shape constants as the M=1 entry this extends; see its file for why
// each is what it is. The padded shared stride is what keeps the 16-element
// blocks off each other's banks.
constexpr int kWarps = 2;
constexpr int kThreads = kWarps * 32;
// Loads in flight per row: the same build-time constant the single-row entry
// uses, so the verify and the step it stands in for stay on the same tuning.
// See w4a16_edge_sm120.cu for why it is set per architecture.
#ifndef FLASHRT_W4A16_EDGE_UNROLL
#define FLASHRT_W4A16_EDGE_UNROLL 4
#endif
constexpr int kUnroll = FLASHRT_W4A16_EDGE_UNROLL;
constexpr int kBlockSlots = 24;
constexpr int kBlockInt4 = kBlockSlots / 8;
constexpr int kRowsDense = 2;
constexpr int kRowsSmall = 8;
constexpr int kMaxM = 8;

static_assert(32 % kRowsDense == 0 && 32 % kRowsSmall == 0,
              "rows per warp must divide the 32-row scale group");

__device__ __forceinline__ float ue4m3_to_float(uint8_t v) {
  const int e = (v >> 3) & 0xF;
  const int m = v & 0x7;
  if (e == 0) return ldexpf(static_cast<float>(m) / 8.0f, -6);
  return ldexpf(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

__device__ __forceinline__ int sf_off(int rb_ncs, int row_inner, int k_block) {
  return (rb_ncs + (k_block >> 2)) * 512 + row_inner + (k_block & 3);
}

// One packed block against M activation rows. The weight byte pair is decoded
// once and used M times, which is the whole point: the decode is the same work
// the M=1 kernel does, and the extra rows cost only shared reads and fmas.
template <int M>
__device__ __forceinline__ void blockdot_m(
    uint64_t b_pack, const __nv_bfloat162* x0, size_t x_row_slots,
    float (&acc)[M]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const __half2_raw wr = flash_rt::fp4::cvt_e2m1x2_to_halfraw2(
        static_cast<uint8_t>(b_pack >> (j * 8)));
    const float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&wr));
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float2 xf = __bfloat1622float2(
          x0[m * (x_row_slots >> 1) + j]);
      acc[m] = fmaf(wf.x, xf.x, acc[m]);
      acc[m] = fmaf(wf.y, xf.y, acc[m]);
    }
  }
}

__device__ __forceinline__ void stage_padded_row(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* x_sh, int K) {
  const int4* x_i4 = reinterpret_cast<const int4*>(x);
  int4* sh_i4 = reinterpret_cast<int4*>(x_sh);
  const int n_i4 = K >> 3;
  for (int j = threadIdx.x; j < n_i4; j += kThreads)
    sh_i4[(j >> 1) * kBlockInt4 + (j & 1)] = x_i4[j];
}

// The K loop, R output rows by M activation rows. Identical in structure to
// the M=1 version: same lane-to-block mapping, same unroll, same order of
// accumulation per (output row, activation row), same final shuffle.
template <int R, int M>
__device__ __forceinline__ void row_dot_m(
    const uint64_t* __restrict__ w_row0, size_t row_stride_u64,
    const uint8_t* __restrict__ SFB, const __nv_bfloat16* x_sh,
    size_t x_row_slots, int K_BLOCKS, int rb_ncs, int row_inner, int lane,
    float (&acc)[R][M]) {
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

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
      for (int u = 0; u < kUnroll; ++u) {
        float part[M];
#pragma unroll
        for (int m = 0; m < M; ++m) part[m] = 0.0f;
        blockdot_m<M>(
            wv[r][u],
            reinterpret_cast<const __nv_bfloat162*>(
                x_sh + (size_t)(kb + 32 * u) * kBlockSlots),
            x_row_slots, part);
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] += part[m] * sf[r][u];
      }
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
    for (int r = 0; r < R; ++r) {
      float part[M];
#pragma unroll
      for (int m = 0; m < M; ++m) part[m] = 0.0f;
      blockdot_m<M>(
          wv[r],
          reinterpret_cast<const __nv_bfloat162*>(
              x_sh + (size_t)kb * kBlockSlots),
          x_row_slots, part);
#pragma unroll
      for (int m = 0; m < M; ++m) acc[r][m] += part[m] * sf[r];
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        acc[r][m] += __shfl_xor_sync(0xffffffff, acc[r][m], off);
}

template <int R, int M>
__global__ void w4a16_mrows_edge_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ SFB,
    __nv_bfloat16* __restrict__ out,
    float alpha, int N, int K, int n_col_super) {
  extern __shared__ __nv_bfloat16 x_sh[];
  const size_t row_slots = (size_t)(K >> 4) * kBlockSlots;
#pragma unroll
  for (int m = 0; m < M; ++m)
    stage_padded_row(x + (size_t)m * K, x_sh + m * row_slots, K);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;

  const int rb = row0 >> 7;
  const int ri = row0 & 127;
  float acc[R][M];
  row_dot_m<R, M>(
      reinterpret_cast<const uint64_t*>(W + (size_t)row0 * (K >> 1)),
      (size_t)(K >> 1) / 8, SFB, x_sh, row_slots, K >> 4, rb * n_col_super,
      (ri & 31) * 16 + ((ri >> 5) & 3) * 4, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (row0 + r >= N) continue;
#pragma unroll
      for (int m = 0; m < M; ++m)
        out[(size_t)m * N + row0 + r] = __float2bfloat16(acc[r][m] * alpha);
    }
  }
}

inline size_t smem_bytes(int K, int M) {
  return (size_t)(K >> 4) * kBlockSlots * sizeof(__nv_bfloat16) * M;
}

inline int rows_per_warp(int N, int K) {
  const int r = (K >= 2048) ? kRowsDense : kRowsSmall;
  return (N >= r * kWarps) ? r : 1;
}

#define FLASHRT_MROWS_LAUNCH(R, M)                                            \
  do {                                                                        \
    const size_t sb = smem_bytes(K, M);                                       \
    if (sb > 48 * 1024) {                                                     \
      cudaFuncSetAttribute(w4a16_mrows_edge_kernel<R, M>,                     \
                           cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                           static_cast<int>(sb));                             \
    }                                                                         \
    w4a16_mrows_edge_kernel<R, M>                                             \
        <<<dim3((N + kWarps * (R) - 1) / (kWarps * (R))), kThreads, sb,       \
           stream>>>(                                                         \
            reinterpret_cast<const __nv_bfloat16*>(x_bf16),                   \
            reinterpret_cast<const uint8_t*>(W_packed),                       \
            reinterpret_cast<const uint8_t*>(SFB),                            \
            reinterpret_cast<__nv_bfloat16*>(out), alpha, N, K, n_col_super); \
  } while (0)

#define FLASHRT_MROWS_BY_M(R)                                                 \
  do {                                                                        \
    switch (M) {                                                              \
      case 1: FLASHRT_MROWS_LAUNCH(R, 1); break;                              \
      case 2: FLASHRT_MROWS_LAUNCH(R, 2); break;                              \
      case 3: FLASHRT_MROWS_LAUNCH(R, 3); break;                              \
      case 4: FLASHRT_MROWS_LAUNCH(R, 4); break;                              \
      case 5: FLASHRT_MROWS_LAUNCH(R, 5); break;                              \
      case 6: FLASHRT_MROWS_LAUNCH(R, 6); break;                              \
      case 7: FLASHRT_MROWS_LAUNCH(R, 7); break;                              \
      default: FLASHRT_MROWS_LAUNCH(R, 8); break;                             \
    }                                                                         \
  } while (0)

}  // namespace

int w4a16_mrows_edge_sm120_bf16(
    const void*  x_bf16,
    const void*  W_packed,
    const void*  SFB,
    void*        out,
    int          M,
    int          N,
    int          K,
    float        alpha,
    cudaStream_t stream) {
  if (!x_bf16 || !W_packed || !SFB || !out) return 1;
  if (N <= 0 || K <= 0 || (K & 15) != 0) return 2;
  if (M <= 0 || M > kMaxM) return 3;

  const int n_col_super = ((K >> 4) + 3) / 4;
  const int R = rows_per_warp(N, K);
  if (R == kRowsDense) {
    FLASHRT_MROWS_BY_M(kRowsDense);
  } else if (R == kRowsSmall) {
    FLASHRT_MROWS_BY_M(kRowsSmall);
  } else {
    FLASHRT_MROWS_BY_M(1);
  }
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
