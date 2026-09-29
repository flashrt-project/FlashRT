// SPDX-License-Identifier: Apache-2.0
//
// See fp4_w4a4_mma_cksplit_sm120.cuh.

#include "fp4_w4a4_mma_cksplit_sm120.cuh"

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

constexpr int MPAD = 48;   // padded M rows covered by the 3 m16 atoms

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
__device__ __forceinline__ void cpa16(uint8_t* d, const uint8_t* s) {
  uint32_t i = __cvta_generic_to_shared(d);
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(i), "l"(s));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void waitg() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

template <int COLS, int KG, int STAGES>
__global__ void cksplit_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ SFA, const uint8_t* __restrict__ SFB,
    __nv_bfloat16* __restrict__ D, float alpha, int M, int N, int K) {
  constexpr int CT = COLS / 8;          // col-tiles per block
  constexpr int WARPS = CT * KG;
  __shared__ uint8_t sA[KG][STAGES][MPAD * 32];
  __shared__ uint8_t sSFA[KG][STAGES][MPAD * 4];
  __shared__ uint8_t sB[WARPS][STAGES][8 * 32];
  __shared__ uint8_t sSFB[WARPS][STAGES][8 * 4];
  __shared__ float s_red[KG][MPAD][COLS];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int ct = warp % CT;
  const int kg = warp / CT;
  const int my_n = blockIdx.x * COLS + ct * 8;

  const int t0 = lane & 3, t1 = lane >> 2;
  const int sau = (lane & 1) * 8 + (lane >> 2), sbu = lane >> 2;

  const int KI = K / 64;
  const int KIw = KI / KG;
  const int kt0 = kg * KIw;
  const int KH = K / 2;
  const int ncs = (K / 16 + 3) / 4;
  const int MT = (M + 15) >> 4;

  // Zero the padded rows M..MPAD-1 of every (kg, stage) A/SFA tile once.
  for (int i = tid; i < KG * STAGES * (MPAD - M); i += blockDim.x) {
    int rem = i;
    int row = M + rem % (MPAD - M);
    rem /= (MPAD - M);
    int st = rem % STAGES;
    int g = rem / STAGES;
    uint4 z; z.x = 0; z.y = 0; z.z = 0; z.w = 0;
    reinterpret_cast<uint4*>(sA[g][st] + row * 32)[0] = z;
    reinterpret_cast<uint4*>(sA[g][st] + row * 32)[1] = z;
    *reinterpret_cast<uint32_t*>(sSFA[g][st] + row * 4) = 0;
  }

  // Loads for this thread's K group and warp's col-tile.
  const int base = kg * (CT * 32);
  const int local = tid - base;
  const int nthr = CT * 32;

  auto ld = [&](int st, int kt) {
    const int bo = kt * 32;
    // A: M rows x 32 bytes = M*2 16-byte chunks (K/2 is 16-byte aligned).
    for (int i = local; i < M * 2; i += nthr) {
      int row = i >> 1, half = i & 1;
      cpa16(sA[kg][st] + row * 32 + half * 16,
            A + (size_t)row * KH + bo + half * 16);
    }
    for (int r = local; r < M; r += nthr) {
      int off = kt * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4;
      cpa(sSFA[kg][st] + r * 4, SFA + off);
    }
    // B: 8 cols x 32 bytes = 16 16-byte chunks (lanes 0..15).
    if (lane < 16) {
      int col = lane >> 1, half = lane & 1;
      cpa16(sB[warp][st] + col * 32 + half * 16,
            B + (size_t)(my_n + col) * KH + bo + half * 16);
    }
    if (lane < 8) {
      int col = my_n + lane, rb = col >> 7, ri = col & 127;
      int si = rb * ncs + kt, ib = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cpa(sSFB[warp][st] + lane * 4, SFB + (size_t)si * 512 + ib);
    }
  };

  float c[3][4] = {};

  // Pipeline: A/SFA are shared by the CT warps of a K group, so the loop uses
  // a block barrier (not __syncwarp) and issues the far-future load AFTER the
  // barrier that guarantees every warp finished reading the buffer it recycles.
  #pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    if (st < KIw) ld(st, kt0 + st);
    commit();
  }
  for (int j = 0; j < KIw; ++j) {
    int cb = j % STAGES;
    waitg<STAGES - 2>();
    __syncthreads();
    int nxt = j + STAGES - 1;
    if (nxt < KIw) {
      ld(nxt % STAGES, kt0 + nxt);
      commit();
    }
    uint32_t b0 = fb(sB[warp][cb], t0, t1, 0);
    uint32_t b1 = fb(sB[warp][cb], t0, t1, 1);
    uint32_t sfb = fsa(sSFB[warp][cb], sbu);
    #pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      if (mt >= MT) break;
      const uint8_t* a = sA[kg][cb] + mt * 16 * 32;
      uint32_t a0 = fa(a, t0, t1, 0), a1 = fa(a, t0, t1, 1);
      uint32_t a2 = fa(a, t0, t1, 2), a3 = fa(a, t0, t1, 3);
      uint32_t sfa = fsa(sSFA[kg][cb] + mt * 16 * 4, sau);
      float d0, d1, d2, d3;
      AtomType::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1,
                    c[mt][0], c[mt][1], c[mt][2], c[mt][3], sfa, sfb);
      c[mt][0] = d0; c[mt][1] = d1; c[mt][2] = d2; c[mt][3] = d3;
    }
  }

  // Deposit per-K-group partials into smem.
  const int q = lane >> 2, r = lane & 3;
  const int c0 = ct * 8 + r * 2;
  #pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    if (mt >= MT) break;
    int row0 = mt * 16 + q, row1 = row0 + 8;
    s_red[kg][row0][c0] = c[mt][0];
    s_red[kg][row0][c0 + 1] = c[mt][1];
    s_red[kg][row1][c0] = c[mt][2];
    s_red[kg][row1][c0 + 1] = c[mt][3];
  }
  __syncthreads();

  // Warp 0 reduces over K groups and writes M x COLS bf16.
  if (warp == 0) {
    for (int idx = lane; idx < M * COLS; idx += 32) {
      int row = idx / COLS, col = idx % COLS;
      float acc = 0.f;
      #pragma unroll
      for (int g = 0; g < KG; ++g) acc += s_red[g][row][col];
      int gcol = blockIdx.x * COLS + col;
      if (gcol < N) D[(size_t)row * N + gcol] = __float2bfloat16(acc * alpha);
    }
  }
}

}  // namespace

int fp4_w4a4_mma_cksplit_bf16out(
    const void* A_packed, const void* B_packed, void* D_bf16, int M, int N,
    int K, const void* SFA, const void* SFB, float alpha, int cols, int kg,
    int stages, cudaStream_t stream) {
  if (!A_packed || !B_packed || !D_bf16 || !SFA || !SFB) return 1;
  if (M <= 0 || M > MPAD) return 2;
  if (K <= 0 || (K % 64) != 0) return 3;
  if (N <= 0) return 4;

  auto a = reinterpret_cast<const uint8_t*>(A_packed);
  auto b = reinterpret_cast<const uint8_t*>(B_packed);
  auto sa = reinterpret_cast<const uint8_t*>(SFA);
  auto sb = reinterpret_cast<const uint8_t*>(SFB);
  auto d = reinterpret_cast<__nv_bfloat16*>(D_bf16);

  #define CK_L(CO, KG_, ST)                                                   \
    cksplit_kernel<CO, KG_, ST><<<N / CO, (CO / 8) * KG_ * 32, 0, stream>>>(  \
        a, b, sa, sb, d, alpha, M, N, K)
  if (cols == 32 && kg == 2 && stages == 3) { if ((K / 64) % 2 || N % 32) return 7; CK_L(32, 2, 3); }
  else if (cols == 32 && kg == 2 && stages == 4) { if ((K / 64) % 2 || N % 32) return 7; CK_L(32, 2, 4); }
  else if (cols == 32 && kg == 4 && stages == 2) { if ((K / 64) % 4 || N % 32) return 7; CK_L(32, 4, 2); }
  else if (cols == 64 && kg == 2 && stages == 2) { if ((K / 64) % 2 || N % 64) return 7; CK_L(64, 2, 2); }
  else if (cols == 64 && kg == 2 && stages == 3) { if ((K / 64) % 2 || N % 64) return 7; CK_L(64, 2, 3); }
  else if (cols == 16 && kg == 2 && stages == 3) { if ((K / 64) % 2 || N % 16) return 7; CK_L(16, 2, 3); }
  else return 5;
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt
