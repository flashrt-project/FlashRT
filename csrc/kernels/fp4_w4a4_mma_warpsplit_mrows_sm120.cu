// SPDX-License-Identifier: Apache-2.0
//
// Small-M warp-split-K NVFP4 W4A4 GEMM for sm_120 — the verify-path
// generalization of fp4_w4a4_mma_warpsplit_sm120.cu. The M=1 warp-split
// kernel splits K across WARPS within one block (8 N-cols/block, each warp
// streams K/WARPS, warp partials summed in SHARED MEMORY) to fill the SMs
// on long-K shapes where the single-warp full_n underfills. The
// SM120_16x8x64 block-scaled MMA atom already produces a full 16-row
// output tile, so extending to M<=16 rows is "free": load A/SFA rows
// 0..M-1 (rows M..15 zero), keep the identical MMA mainloop, and reduce +
// write all M rows. No cross-block/cross-kernel intermediate -> graph-
// replay safe (same property as the M=1 warp-split kernel).
//
// Additive: new file + new entry point. The shipped M=1 warp-split kernel
// and its binding are untouched.
//
// Header: fp4_w4a4_mma_warpsplit_mrows_sm120.cuh.
#include "fp4_w4a4_mma_warpsplit_mrows_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace flash_rt {
namespace gemm {
namespace {

using AtomType = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;

__device__ __forceinline__ uint32_t fa(const uint8_t* s, int t0, int t1, int r) {
  int ro = ((r & 1) ? (t1 + 8) : t1) * 32;
  return *reinterpret_cast<const uint32_t*>(s + ro + t0 * 4 + ((r >> 1) & 1) * 16);
}
__device__ __forceinline__ uint32_t fb(const uint8_t* s, int t0, int t1, int r) {
  return *reinterpret_cast<const uint32_t*>(s + t1 * 32 + t0 * 4 + r * 16);
}
__device__ __forceinline__ uint32_t fsa(const uint8_t* p, int u) {
  return *reinterpret_cast<const uint32_t*>(p + u * 4);
}
__device__ __forceinline__ void cpa(uint8_t* d, const uint8_t* s) {
  uint32_t i = __cvta_generic_to_shared(d);
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n" :: "r"(i), "l"(s));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void waitg() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

template <int STAGES, int WARPS>
__global__ void warpsplit_mrows_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ SFA, const uint8_t* __restrict__ SFB,
    __nv_bfloat16* __restrict__ D, float alpha, int M, int N, int K) {
  // per-warp pipeline buffers
  __shared__ uint8_t sA[WARPS][STAGES][16 * 32];
  __shared__ uint8_t sSFA[WARPS][STAGES][16 * 4];
  __shared__ uint8_t sB[WARPS][STAGES][8 * 32];
  __shared__ uint8_t sSFB[WARPS][STAGES][8 * 4];
  __shared__ float s_red[WARPS][16][8];   // each warp's 16-row x 8-col partials

  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int my_n = blockIdx.x * 8;
  const int KI = K / 64, KIw = KI / WARPS;   // K-tiles per warp
  const int kt0 = warp * KIw;
  const int KH = K / 2, ncs = (K / 16 + 3) / 4;
  int t0 = lane & 3, t1 = lane >> 2, sau = (lane & 1) * 8 + (lane >> 2), sbu = lane >> 2;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;

  uint8_t (*mA)[16 * 32] = sA[warp];
  uint8_t (*mSFA)[16 * 4] = sSFA[warp];
  uint8_t (*mB)[8 * 32] = sB[warp];
  uint8_t (*mSFB)[8 * 4] = sSFB[warp];

  // zero rows M..15 of A (32B) and SFA (4B) across all stages
  if (lane >= M && lane < 16) {
    #pragma unroll
    for (int st = 0; st < STAGES; ++st) {
      int4* av = reinterpret_cast<int4*>(mA[st]); int4 z{0, 0, 0, 0};
      av[lane * 2] = z; av[lane * 2 + 1] = z;
      *reinterpret_cast<uint32_t*>(mSFA[st] + lane * 4) = 0;
    }
  }
  auto ld = [&](int bf, int kt) {
    int bo = kt * 32;
    // A rows 0..M-1: M*8 uint32 (each row = 32 bytes for this K-tile).
    for (int i = lane; i < M * 8; i += 32) { int row = i >> 3, u = i & 7;
      cpa(mA[bf] + row * 32 + u * 4, A + row * KH + bo + u * 4); }
    if (lane < M) cpa(mSFA[bf] + lane * 4, SFA + kt * 512 + lane * 16);
    for (int c = 0; c < 2; ++c) { int ch = lane + c * 32, col = ch >> 3, off = ch & 7;
      cpa(mB[bf] + ch * 4, B + (my_n + col) * KH + bo + off * 4); }
    if (lane < 8) { int col = my_n + lane, rb = col >> 7, ri = col & 127;
      int si = rb * ncs + kt, ib = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cpa(mSFB[bf] + lane * 4, SFB + si * 512 + ib); }
  };
  #pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) { if (st < KIw) ld(st, kt0 + st); commit(); }
  for (int j = 0; j < KIw; ++j) {
    int cb = j % STAGES, jp = j + STAGES - 1;
    if (jp < KIw) ld(jp % STAGES, kt0 + jp);
    commit(); waitg<STAGES - 1>(); __syncwarp();
    uint32_t a0 = fa(mA[cb], t0, t1, 0), a1 = fa(mA[cb], t0, t1, 1);
    uint32_t a2 = fa(mA[cb], t0, t1, 2), a3 = fa(mA[cb], t0, t1, 3);
    uint32_t b0 = fb(mB[cb], t0, t1, 0), b1 = fb(mB[cb], t0, t1, 1);
    uint32_t sfa = fsa(mSFA[cb], sau), sfb = fsa(mSFB[cb], sbu);
    float d0, d1, d2, d3;
    AtomType::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb);
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;
  }
  // c0,c1 -> row (lane>>2), cols 2r,2r+1 ; c2,c3 -> row (lane>>2)+8.
  int q = lane >> 2, r = lane & 3;
  s_red[warp][q][r * 2] = c0;
  s_red[warp][q][r * 2 + 1] = c1;
  s_red[warp][q + 8][r * 2] = c2;
  s_red[warp][q + 8][r * 2 + 1] = c3;
  __syncthreads();
  // warp 0 sums the WARPS partials per (row,col) and writes M rows of bf16.
  if (warp == 0) {
    for (int idx = lane; idx < M * 8; idx += 32) {
      int row = idx >> 3, col = idx & 7;
      float acc = 0.f;
      #pragma unroll
      for (int w = 0; w < WARPS; ++w) acc += s_red[w][row][col];
      int gcol = my_n + col;
      if (gcol < N) D[row * N + gcol] = __float2bfloat16(acc * alpha);
    }
  }
}

}  // namespace

int fp4_w4a4_mma_sm120_warpsplit_mrows_bf16out(
    const void* A_packed, const void* B_packed, void* D_bf16, int M, int N,
    int K, const void* SFA, const void* SFB, float alpha, int warps,
    int stages, cudaStream_t stream) {
  if (!A_packed || !B_packed || !D_bf16 || !SFA || !SFB) return 1;
  if (K <= 0 || (K % 64) != 0 || ((K / 64) % warps) != 0) return 2;
  if (N <= 0 || (N % 8) != 0) return 3;
  if (M <= 0 || M > 16) return 4;
  dim3 grid(N / 8);
  auto a = reinterpret_cast<const uint8_t*>(A_packed);
  auto b = reinterpret_cast<const uint8_t*>(B_packed);
  auto sa = reinterpret_cast<const uint8_t*>(SFA);
  auto sb = reinterpret_cast<const uint8_t*>(SFB);
  auto d = reinterpret_cast<__nv_bfloat16*>(D_bf16);
  #define WM_L(ST, WP) warpsplit_mrows_kernel<ST, WP><<<grid, WP * 32, 0, stream>>>(a, b, sa, sb, d, alpha, M, N, K)
  if (warps == 2) { if (stages == 3) WM_L(3, 2); else if (stages == 4) WM_L(4, 2); else if (stages == 6) WM_L(6, 2); else return 5; }
  else if (warps == 4) { if (stages == 3) WM_L(3, 4); else if (stages == 4) WM_L(4, 4); else if (stages == 6) WM_L(6, 4); else return 5; }
  else if (warps == 8) { if (stages == 3) WM_L(3, 8); else if (stages == 4) WM_L(4, 8); else return 5; }
  else return 6;
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt
