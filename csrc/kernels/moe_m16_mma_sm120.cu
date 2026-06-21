// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE grouped W4A4 block-scaled mma, M=16 tiled (prefill).
//
// The grouped W4A16 GEMV (moe_grouped_w4a16_sm120) is SIMT and runs one M=1
// GEMV per (token, expert) slot -> at large S the per-token compute is the
// wall (51% of the S=2048 prefill). This kernel feeds the SM120 block-scaled
// mma 16 REAL tokens of one expert per tile, so the tensor cores are fully
// utilised and each expert weight is read once. It is the M=16 sibling of the
// decode moe_grouped_gemv_sm120 (which pads M=1 to 16 and writes only row 0); the
// mma body / swizzle decode are copied verbatim from that validated kernel
// (file-local anon namespace, cannot be shared) with the activation load,
// SFA load and epilogue widened to all 16 rows. W4A4 (FP4 activation).
//
// Inputs are pre-sorted into 16-row expert tiles (pad short tiles with zero
// rows): A_tiled (num_tiles*16, K) FP4 + SFA_tiled (per-tile super, sfa_stride),
// tile_expert (num_tiles,). All add-only.

#include "kernels/moe_m16_mma_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace flash_rt {
namespace gemm {

namespace {

using AtomTypeM16 = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;

constexpr int M16_COLS_PER_WARP = 8;

__device__ __forceinline__ uint32_t m16_load_a(
    const uint8_t* sA, int t0, int t1, int reg_idx) {
  int row_off = ((reg_idx & 1) ? (t1 + 8) : t1) * 32;
  int col_off = t0 * 4 + ((reg_idx >> 1) & 1) * 16;
  return *reinterpret_cast<const uint32_t*>(sA + row_off + col_off);
}
__device__ __forceinline__ uint32_t m16_load_b(
    const uint8_t* sB, int t0, int t1, int reg_idx) {
  int col_off = t0 * 4 + reg_idx * 16;
  return *reinterpret_cast<const uint32_t*>(sB + t1 * 32 + col_off);
}
__device__ __forceinline__ uint32_t m16_load_sf(const uint8_t* s, int u) {
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

// grid = (ceil(N/8), num_tiles). 1 warp / block; computes 8 N-cols x 16 rows.
__global__ void moe_m16_mma_kernel(
    const uint8_t* __restrict__ A_tiled,      // (num_tiles*16, K/2)
    const uint8_t* __restrict__ B_stack,      // (E, N, K/2)
    const uint8_t* __restrict__ SFA_tiled,    // per-tile super (sfa_stride)
    const uint8_t* __restrict__ SFB_stack,    // (E, swizzled)
    __nv_bfloat16* __restrict__ D,            // (num_tiles*16, N)
    const float* __restrict__ alpha_stack,
    const int* __restrict__ tile_expert,
    int N, int K, long sfa_stride, long w_stride, long sfb_stride) {
  const int tile = blockIdx.y;
  const int e = tile_expert[tile];
  const uint8_t* A_packed = A_tiled + (long)tile * 16 * (K / 2);
  // SFA is the single batched-quant swizzle of all (num_tiles*16) rows; each
  // row reads its global-row swizzle offset (sfa_stride unused).
  const uint8_t* SFA = SFA_tiled;
  const uint8_t* B_packed = B_stack + (long)e * w_stride;
  const uint8_t* SFB = SFB_stack + (long)e * sfb_stride;
  __nv_bfloat16* D_t = D + (long)tile * 16 * N;
  const float alpha = alpha_stack[e];

  __shared__ alignas(16) uint8_t s_A[2][16 * 32];
  __shared__ alignas(16) uint8_t s_SFA[2][16 * 4];
  __shared__ alignas(16) uint8_t s_B[2][8 * 32];
  __shared__ alignas(16) uint8_t s_SFB[2][8 * 4];

  const int lane = threadIdx.x & 31;
  const int my_n_off = blockIdx.x * M16_COLS_PER_WARP;
  if (my_n_off >= N) return;

  const int t0 = lane & 3;
  const int t1 = lane >> 2;
  const int sfa_unique_row = (lane & 1) * 8 + (lane >> 2);
  const int sfb_unique_col = lane >> 2;

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  const int K_iters = K / 64;
  const int K_half = K / 2;
  const int K_blocks = K / 16;
  const int n_col_super = (K_blocks + 3) / 4;

  auto load = [&](int buf, int kt) {
    const int byte_off = kt * 32;
    // 16 activation rows x 32 bytes: lane -> row lane/2, half (lane%2)*16.
    const int row = lane >> 1;
    const int half = (lane & 1) * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      cp4(s_A[buf] + row * 32 + half + i * 4,
          A_packed + (size_t)row * K_half + byte_off + half + i * 4);
    // 16 rows of SFA from the shared batched-quant swizzle: row `lane` is
    // global row tile*16+lane -> standard NVFP4 SF swizzle offset.
    if (lane < 16) {
      int grow = tile * 16 + lane;
      int rb = grow >> 7;
      int ri = grow & 127;
      int off = (rb * n_col_super + kt) * 512
                + (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp4(s_SFA[buf] + lane * 4, SFA + (size_t)off);
    }
    // 8 B cols x 32 bytes.
    for (int c = 0; c < 2; ++c) {
      int chunk = lane + c * 32;
      int col = chunk >> 3;
      int off = chunk & 7;
      cp4(s_B[buf] + chunk * 4,
          B_packed + (size_t)(my_n_off + col) * K_half + byte_off + off * 4);
    }
    if (lane < 8) {
      int col = my_n_off + lane;
      int rb = col >> 7;
      int ri = col & 127;
      int super_idx = rb * n_col_super + kt;
      int inner = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp4(s_SFB[buf] + lane * 4, SFB + (size_t)super_idx * 512 + inner);
    }
  };

  load(0, 0); commit();
  if (K_iters > 1) { load(1, 1); commit(); }

  for (int kt = 0; kt < K_iters; ++kt) {
    int cur = kt & 1;
    waitg(kt + 1 < K_iters ? 1 : 0);
    __syncwarp();

    uint32_t a0 = m16_load_a(s_A[cur], t0, t1, 0);
    uint32_t a1 = m16_load_a(s_A[cur], t0, t1, 1);
    uint32_t a2 = m16_load_a(s_A[cur], t0, t1, 2);
    uint32_t a3 = m16_load_a(s_A[cur], t0, t1, 3);
    uint32_t b0 = m16_load_b(s_B[cur], t0, t1, 0);
    uint32_t b1 = m16_load_b(s_B[cur], t0, t1, 1);
    uint32_t sfa = m16_load_sf(s_SFA[cur], sfa_unique_row);
    uint32_t sfb = m16_load_sf(s_SFB[cur], sfb_unique_col);

    float d0, d1, d2, d3;
    AtomTypeM16::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1,
                     c0, c1, c2, c3, sfa, sfb);
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;

    if (kt + 2 < K_iters) { load(cur, kt + 2); commit(); }
  }

  // Epilogue: write all 16 rows. c0/c1 -> row q, c2/c3 -> row q+8.
  const int q = lane >> 2;
  const int r = lane & 3;
  const int col0 = my_n_off + r * 2;
  const int col1 = col0 + 1;
  if (col0 < N) {
    D_t[(size_t)q * N + col0] = __float2bfloat16(c0 * alpha);
    D_t[(size_t)(q + 8) * N + col0] = __float2bfloat16(c2 * alpha);
  }
  if (col1 < N) {
    D_t[(size_t)q * N + col1] = __float2bfloat16(c1 * alpha);
    D_t[(size_t)(q + 8) * N + col1] = __float2bfloat16(c3 * alpha);
  }
}

}  // namespace

int moe_m16_mma_sm120_bf16(
    const void* A_tiled, const void* B_stack, const void* SFA_tiled,
    const void* SFB_stack, void* D, const void* alpha_stack,
    const void* tile_expert, int num_tiles, int N, int K,
    long sfa_stride, long w_stride, long sfb_stride, cudaStream_t stream) {
  if (!A_tiled || !B_stack || !SFA_tiled || !SFB_stack || !D ||
      !alpha_stack || !tile_expert) return 1;
  if (num_tiles <= 0 || K <= 0 || (K % 64) != 0) return 2;
  if (N <= 0 || (N % M16_COLS_PER_WARP) != 0) return 3;
  dim3 grid((N + M16_COLS_PER_WARP - 1) / M16_COLS_PER_WARP, num_tiles);
  dim3 block(32);
  moe_m16_mma_kernel<<<grid, block, 0, stream>>>(
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
