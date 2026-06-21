// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE grouped W4A4 block-scaled mma, M=64 x N=16 tiled (prefill).
//
// The M=16/N=8 sibling (moe_m16_mma_sm120) re-reads the activation rows once
// per N-block (grid.x = N/8) and re-reads each expert weight once per 16-row
// tile, so its total HBM traffic ~ (1/M_tile + 1/N_tile) = 1/16 + 1/8 dominates
// and it sits ~8x off the roofline. This kernel widens BOTH the M tile to 64
// rows (4 row-blocks) and the N tile to 16 cols (2 col-blocks) -> traffic
// 1/64 + 1/16, ~2.4x less, with one warp still (32 fp32 accumulators). The mma
// body / swizzle decode / SF layout are copied verbatim from the validated M16
// kernel; only the tiling (4 row-blocks x 2 col-blocks sharing each loaded K
// chunk) is new. Tokens pre-sorted into 64-row expert tiles (pad with zeros).
// All add-only.

#include "kernels/moe_m64_mma_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace flash_rt {
namespace gemm {

namespace {

using AtomTypeM64 = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;

constexpr int M64_NCOLS = 16;         // N tile: 2 col-blocks of 8
constexpr int M64_CB = 2;
constexpr int M64_ROWS = 64;          // M tile: 4 row-blocks of 16
constexpr int M64_RB = 4;

__device__ __forceinline__ uint32_t m64_load_a(
    const uint8_t* sA, int t0, int t1, int reg_idx) {
  int row_off = ((reg_idx & 1) ? (t1 + 8) : t1) * 32;
  int col_off = t0 * 4 + ((reg_idx >> 1) & 1) * 16;
  return *reinterpret_cast<const uint32_t*>(sA + row_off + col_off);
}
__device__ __forceinline__ uint32_t m64_load_b(
    const uint8_t* sB, int t0, int t1, int reg_idx) {
  int col_off = t0 * 4 + reg_idx * 16;
  return *reinterpret_cast<const uint32_t*>(sB + t1 * 32 + col_off);
}
__device__ __forceinline__ uint32_t m64_load_sf(const uint8_t* s, int u) {
  return *reinterpret_cast<const uint32_t*>(s + u * 4);
}
__device__ __forceinline__ void cp4(uint8_t* d, const uint8_t* s) {
  uint32_t a = __cvta_generic_to_shared(d);
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n"
               :: "r"(a), "l"(s));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void waitg(int N) {
  if (N == 0) asm volatile("cp.async.wait_group 0;\n" ::);
  else if (N == 1) asm volatile("cp.async.wait_group 1;\n" ::);
  else asm volatile("cp.async.wait_all;\n" ::);
}

// grid = (ceil(N/16), num_tiles). 1 warp / block; computes 16 N-cols x 64 rows.
__global__ void moe_m64_mma_kernel(
    const uint8_t* __restrict__ A_tiled,      // (num_tiles*64, K/2)
    const uint8_t* __restrict__ B_stack,      // (E, N, K/2)
    const uint8_t* __restrict__ SFA_tiled,    // batched-quant super
    const uint8_t* __restrict__ SFB_stack,    // (E, swizzled)
    __nv_bfloat16* __restrict__ D,            // (num_tiles*64, N)
    const float* __restrict__ alpha_stack,
    const int* __restrict__ tile_expert,
    int N, int K, long sfa_stride, long w_stride, long sfb_stride) {
  const int tile = blockIdx.y;
  const int e = tile_expert[tile];
  const uint8_t* A_packed = A_tiled + (long)tile * M64_ROWS * (K / 2);
  const uint8_t* SFA = SFA_tiled;
  const uint8_t* B_packed = B_stack + (long)e * w_stride;
  const uint8_t* SFB = SFB_stack + (long)e * sfb_stride;
  __nv_bfloat16* D_t = D + (long)tile * M64_ROWS * N;
  const float alpha = alpha_stack[e];

  __shared__ alignas(16) uint8_t s_A[2][M64_ROWS * 32];
  __shared__ alignas(16) uint8_t s_SFA[2][M64_ROWS * 4];
  __shared__ alignas(16) uint8_t s_B[2][M64_NCOLS * 32];
  __shared__ alignas(16) uint8_t s_SFB[2][M64_NCOLS * 4];

  const int lane = threadIdx.x & 31;
  const int my_n_off = blockIdx.x * M64_NCOLS;
  if (my_n_off >= N) return;

  const int t0 = lane & 3;
  const int t1 = lane >> 2;
  const int sfa_unique_row = (lane & 1) * 8 + (lane >> 2);
  const int sfb_unique_col = lane >> 2;

  float c[M64_RB][M64_CB][4];
#pragma unroll
  for (int rb = 0; rb < M64_RB; ++rb)
#pragma unroll
    for (int cb = 0; cb < M64_CB; ++cb)
      c[rb][cb][0] = c[rb][cb][1] = c[rb][cb][2] = c[rb][cb][3] = 0.f;

  const int K_iters = K / 64;
  const int K_half = K / 2;
  const int K_blocks = K / 16;
  const int n_col_super = (K_blocks + 3) / 4;

  auto load = [&](int buf, int kt) {
    const int byte_off = kt * 32;
    // 64 activation rows x 32 bytes: lane -> 4 rows (rg*16 + lane/2), half.
    const int half = (lane & 1) * 16;
#pragma unroll
    for (int rg = 0; rg < M64_RB; ++rg) {
      const int row = rg * 16 + (lane >> 1);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        cp4(s_A[buf] + row * 32 + half + i * 4,
            A_packed + (size_t)row * K_half + byte_off + half + i * 4);
    }
    // 64 rows of SFA (global-row swizzle, 2 rows / lane).
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      const int r = lane + rg * 32;
      const int grow = tile * M64_ROWS + r;
      const int rb_s = grow >> 7;
      const int ri = grow & 127;
      const int off = (rb_s * n_col_super + kt) * 512
                      + (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp4(s_SFA[buf] + r * 4, SFA + (size_t)off);
    }
    // 16 B cols x 32 bytes (4 chunks of 32 lanes).
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      int chunk = lane + cc * 32;
      int col = chunk >> 3;
      int off = chunk & 7;
      cp4(s_B[buf] + chunk * 4,
          B_packed + (size_t)(my_n_off + col) * K_half + byte_off + off * 4);
    }
    // 16 cols of SFB (2 cols / lane).
#pragma unroll
    for (int cg = 0; cg < 2; ++cg) {
      int lc = lane + cg * 32;          // 0..63 -> but only first 16 cols valid
      if (lc < M64_NCOLS) {
        int col = my_n_off + lc;
        int rb = col >> 7;
        int ri = col & 127;
        int super_idx = rb * n_col_super + kt;
        int inner = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
        cp4(s_SFB[buf] + lc * 4, SFB + (size_t)super_idx * 512 + inner);
      }
    }
  };

  load(0, 0); commit();
  if (K_iters > 1) { load(1, 1); commit(); }

  for (int kt = 0; kt < K_iters; ++kt) {
    int cur = kt & 1;
    waitg(kt + 1 < K_iters ? 1 : 0);
    __syncwarp();

    uint32_t b0[M64_CB], b1[M64_CB], sfb[M64_CB];
#pragma unroll
    for (int cb = 0; cb < M64_CB; ++cb) {
      const uint8_t* sB = s_B[cur] + cb * 8 * 32;
      b0[cb] = m64_load_b(sB, t0, t1, 0);
      b1[cb] = m64_load_b(sB, t0, t1, 1);
      sfb[cb] = m64_load_sf(s_SFB[cur] + cb * 8 * 4, sfb_unique_col);
    }

#pragma unroll
    for (int rb = 0; rb < M64_RB; ++rb) {
      const uint8_t* sA = s_A[cur] + rb * 16 * 32;
      uint32_t a0 = m64_load_a(sA, t0, t1, 0);
      uint32_t a1 = m64_load_a(sA, t0, t1, 1);
      uint32_t a2 = m64_load_a(sA, t0, t1, 2);
      uint32_t a3 = m64_load_a(sA, t0, t1, 3);
      uint32_t sfa = m64_load_sf(s_SFA[cur] + rb * 16 * 4, sfa_unique_row);
#pragma unroll
      for (int cb = 0; cb < M64_CB; ++cb) {
        float d0, d1, d2, d3;
        AtomTypeM64::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0[cb], b1[cb],
                         c[rb][cb][0], c[rb][cb][1], c[rb][cb][2], c[rb][cb][3],
                         sfa, sfb[cb]);
        c[rb][cb][0] = d0; c[rb][cb][1] = d1;
        c[rb][cb][2] = d2; c[rb][cb][3] = d3;
      }
    }

    if (kt + 2 < K_iters) { load(cur, kt + 2); commit(); }
  }

  // Epilogue: 64 rows x 16 cols. c[rb][cb][0,1]->row rb*16+q, [2,3]->+8.
  const int q = lane >> 2;
  const int r = lane & 3;
#pragma unroll
  for (int rb = 0; rb < M64_RB; ++rb) {
    const int ra = rb * 16 + q;
    const int rbw = rb * 16 + q + 8;
#pragma unroll
    for (int cb = 0; cb < M64_CB; ++cb) {
      const int col0 = my_n_off + cb * 8 + r * 2;
      const int col1 = col0 + 1;
      if (col0 < N) {
        D_t[(size_t)ra * N + col0] = __float2bfloat16(c[rb][cb][0] * alpha);
        D_t[(size_t)rbw * N + col0] = __float2bfloat16(c[rb][cb][2] * alpha);
      }
      if (col1 < N) {
        D_t[(size_t)ra * N + col1] = __float2bfloat16(c[rb][cb][1] * alpha);
        D_t[(size_t)rbw * N + col1] = __float2bfloat16(c[rb][cb][3] * alpha);
      }
    }
  }
}

}  // namespace

int moe_m64_mma_sm120_bf16(
    const void* A_tiled, const void* B_stack, const void* SFA_tiled,
    const void* SFB_stack, void* D, const void* alpha_stack,
    const void* tile_expert, int num_tiles, int N, int K,
    long sfa_stride, long w_stride, long sfb_stride, cudaStream_t stream) {
  if (!A_tiled || !B_stack || !SFA_tiled || !SFB_stack || !D ||
      !alpha_stack || !tile_expert) return 1;
  if (num_tiles <= 0 || K <= 0 || (K % 64) != 0) return 2;
  if (N <= 0 || (N % M64_NCOLS) != 0) return 3;
  dim3 grid((N + M64_NCOLS - 1) / M64_NCOLS, num_tiles);
  dim3 block(32);
  moe_m64_mma_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(A_tiled),
      reinterpret_cast<const uint8_t*>(B_stack),
      reinterpret_cast<const uint8_t*>(SFA_tiled),
      reinterpret_cast<const uint8_t*>(SFB_stack),
      reinterpret_cast<__nv_bfloat16*>(D),
      reinterpret_cast<const float*>(alpha_stack),
      reinterpret_cast<const int*>(tile_expert),
      N, K, sfa_stride, w_stride, sfb_stride);
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt
