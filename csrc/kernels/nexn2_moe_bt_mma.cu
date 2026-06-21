// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE grouped W4A4 block-scaled GEMM, multi-warp CTA tile (prefill).
//
// sm120 (consumer Blackwell) has no working vendor NVFP4 grouped GEMM (CUTLASS
// grouped block-scaled produces garbage, flashinfer needs patching), so this is
// a hand-tuned kernel that follows the DeepGEMM / SGLang *structure* -- a CTA
// computes a BM x BN output tile, loading the BM activation rows and BN weight
// cols ONCE into shared memory and sharing them across all warps -- specialised
// to the SM120 16x8x64 block-scaled mma + cp.async (no TMA on sm120).
//
// Total HBM traffic ~ (1/BM + 1/BN); the single-warp M16 (16x8) and M64 (64x16)
// tiles were limited by re-reading activations once per N-block. With BM=BN=64
// (4 warps, 2x2 warp grid, each warp a 32x32 = 2x4 mma sub-tile, 32 fp32 accum)
// the traffic factor is 1/64 + 1/64 -- ~2.5x less than M64x16. The mma body /
// swizzle decode / SF layout are copied from the validated M16 kernel; the CTA
// tiling + cooperative smem staging are new. Tokens pre-sorted into BM-row
// expert tiles (pad with zeros). All add-only.

#include "kernels/nexn2_moe_bt_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace flash_rt {
namespace gemm {

namespace {

using AtomTypeBT = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;

constexpr int BT_BM = 64;          // CTA M tile (4 row-blocks of 16)
constexpr int BT_BN = 64;          // CTA N tile (8 col-blocks of 8)
constexpr int BT_WARPS = 4;        // 2x2 warp grid
constexpr int BT_THREADS = BT_WARPS * 32;
constexpr int BT_WN = 2;           // warps along N
constexpr int BT_WRB = 2;          // row-blocks (of 16) per warp
constexpr int BT_WCB = 4;          // col-blocks (of 8) per warp

__device__ __forceinline__ uint32_t bt_load_a(
    const uint8_t* sA, int t0, int t1, int reg_idx) {
  int row_off = ((reg_idx & 1) ? (t1 + 8) : t1) * 32;
  int col_off = t0 * 4 + ((reg_idx >> 1) & 1) * 16;
  return *reinterpret_cast<const uint32_t*>(sA + row_off + col_off);
}
__device__ __forceinline__ uint32_t bt_load_b(
    const uint8_t* sB, int t0, int t1, int reg_idx) {
  int col_off = t0 * 4 + reg_idx * 16;
  return *reinterpret_cast<const uint32_t*>(sB + t1 * 32 + col_off);
}
__device__ __forceinline__ uint32_t bt_load_sf(const uint8_t* s, int u) {
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

// grid = (ceil(N/BN), num_tiles). BT_THREADS threads/block. BM x BN output.
__global__ __launch_bounds__(BT_THREADS) void moe_bt_mma_kernel(
    const uint8_t* __restrict__ A_tiled,      // (num_tiles*BM, K/2)
    const uint8_t* __restrict__ B_stack,      // (E, N, K/2)
    const uint8_t* __restrict__ SFA_tiled,    // batched-quant super
    const uint8_t* __restrict__ SFB_stack,    // (E, swizzled)
    __nv_bfloat16* __restrict__ D,            // (num_tiles*BM, N)
    const float* __restrict__ alpha_stack,
    const int* __restrict__ tile_expert,
    int N, int K, long sfa_stride, long w_stride, long sfb_stride) {
  const int tile = blockIdx.y;
  const int e = tile_expert[tile];
  const uint8_t* A_packed = A_tiled + (long)tile * BT_BM * (K / 2);
  const uint8_t* SFA = SFA_tiled;
  const uint8_t* B_packed = B_stack + (long)e * w_stride;
  const uint8_t* SFB = SFB_stack + (long)e * sfb_stride;
  __nv_bfloat16* D_t = D + (long)tile * BT_BM * N;
  const float alpha = alpha_stack[e];

  __shared__ alignas(16) uint8_t s_A[2][BT_BM * 32];
  __shared__ alignas(16) uint8_t s_SFA[2][BT_BM * 4];
  __shared__ alignas(16) uint8_t s_B[2][BT_BN * 32];
  __shared__ alignas(16) uint8_t s_SFB[2][BT_BN * 4];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp / BT_WN;          // 0..1
  const int warp_n = warp % BT_WN;          // 0..1
  const int my_n_off = blockIdx.x * BT_BN;
  if (my_n_off >= N) return;

  const int t0 = lane & 3;
  const int t1 = lane >> 2;
  const int sfa_unique_row = (lane & 1) * 8 + (lane >> 2);
  const int sfb_unique_col = lane >> 2;

  float c[BT_WRB][BT_WCB][4];
#pragma unroll
  for (int rb = 0; rb < BT_WRB; ++rb)
#pragma unroll
    for (int cb = 0; cb < BT_WCB; ++cb)
      c[rb][cb][0] = c[rb][cb][1] = c[rb][cb][2] = c[rb][cb][3] = 0.f;

  const int K_iters = K / 64;
  const int K_half = K / 2;
  const int K_blocks = K / 16;
  const int n_col_super = (K_blocks + 3) / 4;

  auto load = [&](int buf, int kt) {
    const int byte_off = kt * 32;
    // A: BM rows x 32 bytes = BM*8 cp4, cooperatively over BT_THREADS.
#pragma unroll
    for (int j = 0; j < (BT_BM * 8) / BT_THREADS; ++j) {
      int idx = tid + j * BT_THREADS;
      int row = idx >> 3;
      int off = idx & 7;
      cp4(s_A[buf] + row * 32 + off * 4,
          A_packed + (size_t)row * K_half + byte_off + off * 4);
    }
    // B: BN cols x 32 bytes.
#pragma unroll
    for (int j = 0; j < (BT_BN * 8) / BT_THREADS; ++j) {
      int idx = tid + j * BT_THREADS;
      int col = idx >> 3;
      int off = idx & 7;
      cp4(s_B[buf] + col * 32 + off * 4,
          B_packed + (size_t)(my_n_off + col) * K_half + byte_off + off * 4);
    }
    // SFA: BM rows from the batched-quant global-row swizzle.
    if (tid < BT_BM) {
      int grow = tile * BT_BM + tid;
      int rb_s = grow >> 7;
      int ri = grow & 127;
      int off = (rb_s * n_col_super + kt) * 512
                + (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp4(s_SFA[buf] + tid * 4, SFA + (size_t)off);
    }
    // SFB: BN cols.
    if (tid < BT_BN) {
      int col = my_n_off + tid;
      int rb = col >> 7;
      int ri = col & 127;
      int super_idx = rb * n_col_super + kt;
      int inner = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp4(s_SFB[buf] + tid * 4, SFB + (size_t)super_idx * 512 + inner);
    }
  };

  load(0, 0); commit();
  if (K_iters > 1) { load(1, 1); commit(); }

  for (int kt = 0; kt < K_iters; ++kt) {
    int cur = kt & 1;
    waitg(kt + 1 < K_iters ? 1 : 0);
    __syncthreads();

    uint32_t b0[BT_WCB], b1[BT_WCB], sfb[BT_WCB];
#pragma unroll
    for (int cb = 0; cb < BT_WCB; ++cb) {
      int gcol = warp_n * (BT_BN / BT_WN) + cb * 8;
      const uint8_t* sB = s_B[cur] + gcol * 32;
      b0[cb] = bt_load_b(sB, t0, t1, 0);
      b1[cb] = bt_load_b(sB, t0, t1, 1);
      sfb[cb] = bt_load_sf(s_SFB[cur] + gcol * 4, sfb_unique_col);
    }
#pragma unroll
    for (int rb = 0; rb < BT_WRB; ++rb) {
      int grow = warp_m * (BT_BM / 2) + rb * 16;
      const uint8_t* sA = s_A[cur] + grow * 32;
      uint32_t a0 = bt_load_a(sA, t0, t1, 0);
      uint32_t a1 = bt_load_a(sA, t0, t1, 1);
      uint32_t a2 = bt_load_a(sA, t0, t1, 2);
      uint32_t a3 = bt_load_a(sA, t0, t1, 3);
      uint32_t sfa = bt_load_sf(s_SFA[cur] + grow * 4, sfa_unique_row);
#pragma unroll
      for (int cb = 0; cb < BT_WCB; ++cb) {
        float d0, d1, d2, d3;
        AtomTypeBT::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0[cb], b1[cb],
                        c[rb][cb][0], c[rb][cb][1], c[rb][cb][2], c[rb][cb][3],
                        sfa, sfb[cb]);
        c[rb][cb][0] = d0; c[rb][cb][1] = d1;
        c[rb][cb][2] = d2; c[rb][cb][3] = d3;
      }
    }

    __syncthreads();
    if (kt + 2 < K_iters) { load(cur, kt + 2); commit(); }
  }

  // Epilogue: each warp writes its 32x32 sub-tile.
  const int q = lane >> 2;
  const int r = lane & 3;
#pragma unroll
  for (int rb = 0; rb < BT_WRB; ++rb) {
    const int base_row = warp_m * (BT_BM / 2) + rb * 16;
    const int ra = base_row + q;
    const int rbw = base_row + q + 8;
#pragma unroll
    for (int cb = 0; cb < BT_WCB; ++cb) {
      const int gcol = my_n_off + warp_n * (BT_BN / BT_WN) + cb * 8 + r * 2;
      const int col1 = gcol + 1;
      if (gcol < N) {
        D_t[(size_t)ra * N + gcol] = __float2bfloat16(c[rb][cb][0] * alpha);
        D_t[(size_t)rbw * N + gcol] = __float2bfloat16(c[rb][cb][2] * alpha);
      }
      if (col1 < N) {
        D_t[(size_t)ra * N + col1] = __float2bfloat16(c[rb][cb][1] * alpha);
        D_t[(size_t)rbw * N + col1] = __float2bfloat16(c[rb][cb][3] * alpha);
      }
    }
  }
}

}  // namespace

int nexn2_moe_bt_mma_bf16(
    const void* A_tiled, const void* B_stack, const void* SFA_tiled,
    const void* SFB_stack, void* D, const void* alpha_stack,
    const void* tile_expert, int num_tiles, int N, int K,
    long sfa_stride, long w_stride, long sfb_stride, cudaStream_t stream) {
  if (!A_tiled || !B_stack || !SFA_tiled || !SFB_stack || !D ||
      !alpha_stack || !tile_expert) return 1;
  if (num_tiles <= 0 || K <= 0 || (K % 64) != 0) return 2;
  if (N <= 0 || (N % BT_BN) != 0) return 3;
  dim3 grid((N + BT_BN - 1) / BT_BN, num_tiles);
  dim3 block(BT_THREADS);
  moe_bt_mma_kernel<<<grid, block, 0, stream>>>(
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
