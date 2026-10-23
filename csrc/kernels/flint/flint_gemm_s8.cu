// SPDX-License-Identifier: Apache-2.0
//
// Flint W4A8 tensor-core GEMM: NVFP4 weights, int8 activations, int8 mma.sync.
//   Y[m, n] = epilogue( gs * sum_k A[m, k] * W[n, k] )
// Activations are quantized per 32-wide K block to int8 with an fp32 scale
// (quant_rows_s8). Weights stay in their checkpoint form: within one 16-wide
// NVFP4 block the E2M1 codes are integers once doubled ({0,1,2,3,4,6,8,12}),
// so the block dot product is accumulated exactly in int32 by
// mma.sync.m16n8k16.s8 and scaled once per block in fp32 by
// (E4M3 block scale) * (activation block scale); the factor 1/2 is folded
// into the global scale. The only approximation versus W4A16 is the int8
// rounding of the activations.
//
// Within every 16-wide block the K order is permuted identically on both
// operands (0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15): the B fragment of lane
// (g, t) is then the low (t even) or high (t odd) nibbles of one 32-bit word
// of weight row g, expanded through a byte-permute table.
//
// Tiles: BM x 128 x (64*KB), 8 warps, ST-stage cp.async pipeline, optional
// split-K over a (1, 1, split) cluster reduced through DSMEM. Dense tiles over
// M and expert-grouped tiles (device tile list, gathered A rows) share the body.

#include "flint_common.cuh"
#include "flint_kernels.cuh"

#include <cooperative_groups.h>
#include <cstdlib>
namespace cg = cooperative_groups;

namespace flash_rt {
namespace flint {

namespace {

constexpr int BN = 128, THREADS = 256;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, bool valid) {
  const int sz = valid ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(dst), "l"(src), "r"(sz));
}
template <int BYTES>
__device__ __forceinline__ void cp_async_bytes(uint32_t dst, const void* src) {
  if constexpr (BYTES == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(dst), "l"(src), "n"(BYTES));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
// Runtime pending-group count (the instruction takes an immediate): wait until at most n groups are pending.
__device__ __forceinline__ void cp_wait_dyn(int n) {
  switch (n) {
#define FLINT_CPW(i) case i: cp_wait<i>(); break;
    FLINT_CPW(0) FLINT_CPW(1) FLINT_CPW(2) FLINT_CPW(3) FLINT_CPW(4) FLINT_CPW(5) FLINT_CPW(6) FLINT_CPW(7)
    FLINT_CPW(8) FLINT_CPW(9) FLINT_CPW(10) FLINT_CPW(11) FLINT_CPW(12) FLINT_CPW(13) FLINT_CPW(14) FLINT_CPW(15)
    FLINT_CPW(16) FLINT_CPW(17) FLINT_CPW(18) FLINT_CPW(19) FLINT_CPW(20) FLINT_CPW(21) FLINT_CPW(22) FLINT_CPW(23)
#undef FLINT_CPW
    default: cp_wait<0>(); break;
  }
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__device__ __forceinline__ void mma16816_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}

// Four E2M1 codes (nibbles 0..3 of `sel`, bits 0..15) -> four int8 of twice their value.
__device__ __forceinline__ uint32_t e2m1x4_to_s8x4(uint32_t sel) {
  const uint32_t idx = sel & 0x7777u;
  const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, idx);
  const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, idx);
  return __byte_perm(mag, neg, 0x3210u | ((sel >> 1) & 0x4444u));
}

// The low (parity 0) or high (parity 1) nibbles of a weight word (bytes 0..3 ->
// output bytes 0..3) as int8x4.
__device__ __forceinline__ uint32_t weight_word_s8x4(uint32_t q, int parity) {
  const uint32_t x = (parity ? (q >> 4) : q) & 0x0F0F0F0Fu;
  const uint32_t y = x | (x >> 4);
  return e2m1x4_to_s8x4(__byte_perm(y, 0u, 0x4420u));
}

template <int BM, int ST, int KB>
struct GemmS8Smem {
  int8_t a[ST][BM * 64 * KB];     // BM rows x 64*KB int8; 16B chunks XOR-swizzled by row within groups of 8
  float as[ST][BM * 2 * KB];      // BM rows x 2*KB activation block scales
  uint8_t w[ST][BN * 32 * KB];    // 128 rows x 32*KB code bytes; 16B chunks XOR-swizzled by row
  uint8_t s[ST][BN * 4 * KB];     // 128 rows x 4*KB scale bytes (row-major)
};

template <int BM, int ST, int KB>
constexpr size_t gemm_s8_smem_bytes(int split) {
  return split > 1 && static_cast<size_t>(BM) * BN * 4 > sizeof(GemmS8Smem<BM, ST, KB>)
             ? static_cast<size_t>(BM) * BN * 4
             : sizeof(GemmS8Smem<BM, ST, KB>);
}

// A rows have 4*KB 16B chunks; XOR-swizzle within groups of min(8, 4*KB) chunks.
template <int KB>
__device__ __forceinline__ int a_chunk(int r, int ch) {
  constexpr int mask = 4 * KB >= 8 ? 7 : 4 * KB - 1;
  return (ch & ~mask) | ((ch ^ r) & mask);
}
template <int KB>
__device__ __forceinline__ int w_chunk(int r, int ch) { return ch ^ (r & (2 * KB - 1)); }

// Two E4M3 scale bytes (low = column 2p, high = column 2p+1) -> f16x2 (exact).
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t pair) {
  uint32_t out;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(static_cast<uint16_t>(pair)));
  return out;
}

// A chunk assignment: chunk c = tid + i * THREADS covers row c / (4 KB), 16B chunk c % (4 KB).
template <int BM, int KB>
__device__ __forceinline__ int a_row_of(int i) { return (threadIdx.x + i * THREADS) / (4 * KB); }
// Activation-scale copies: 8*KB bytes per row in pieces of min(16, 8*KB) bytes.
template <int KB>
constexpr int as_pieces() { return KB >= 2 ? KB / 2 : 1; }
template <int BM, int KB>
__device__ __forceinline__ int as_row_of(int i) { return (threadIdx.x + i * THREADS) / as_pieces<KB>(); }

// Main loop over stages [kt0, kt0 + nkt). arow[i] / asrow[i]: global row base of this thread's i-th A /
// scale copy (nullptr = zero row). gs already includes the 1/2 of the doubled E2M1 table.
template <int BM, int ST, int KB, int EPI>
__device__ __forceinline__ void gemm_s8_body(const int8_t* const* arow, const float* const* asrow, int m_rows,
                                             const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int K,
                                             int kt0, int nkt, int split, int n0, float gs, void* __restrict__ C,
                                             int c_row0, int ldc) {
  constexpr int WM = BM >= 128 ? 2 : 1;
  constexpr int WN = 8 / WM;
  constexpr int MT = BM / (16 * WM);
  constexpr int NT = BN / (WN * 8);
  constexpr int ACH = 4 * KB;                 // 16B chunks per A row per stage
  constexpr int AI = (BM * ACH + THREADS - 1) / THREADS;
  constexpr int ASP = as_pieces<KB>();
  constexpr int ASI = (BM * ASP + THREADS - 1) / THREADS;
  constexpr int SK = 64 * KB;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  GemmS8Smem<BM, ST, KB>& sm = *reinterpret_cast<GemmS8Smem<BM, ST, KB>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int wm0 = (warp / WN) * (MT * 16), wn0 = (warp % WN) * (BN / WN);

  auto load_a_stage = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < AI; ++i) {
      const int c = tid + i * THREADS;
      if (BM * ACH % THREADS == 0 || c < BM * ACH) {
        const int r = c / ACH, ch = c % ACH;
        const bool ok = arow[i] != nullptr;
        cp_async16(smem_u32(sm.a[stage] + r * SK + (a_chunk<KB>(r, ch) << 4)),
                   ok ? static_cast<const void*>(arow[i] + kt * SK + ch * 16) : static_cast<const void*>(W), ok);
      }
    }
#pragma unroll
    for (int i = 0; i < ASI; ++i) {
      const int c = tid + i * THREADS;
      if (BM * ASP % THREADS == 0 || c < BM * ASP) {
        const int r = c / ASP, p = c % ASP;
        const bool ok = asrow[i] != nullptr;
        if constexpr (KB >= 2)
          cp_async16(smem_u32(sm.as[stage] + r * 2 * KB + p * 4),
                     ok ? static_cast<const void*>(asrow[i] + kt * 2 * KB + p * 4) : static_cast<const void*>(W), ok);
        else
          cp_async8(smem_u32(sm.as[stage] + r * 2 * KB),
                    ok ? static_cast<const void*>(asrow[i] + kt * 2 * KB) : static_cast<const void*>(W), ok);
      }
    }
  };
  auto load_w_stage = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < KB; ++i) {  // 128 rows x 2*KB chunks
      const int c = tid + i * THREADS, r = c / (2 * KB), ch = c % (2 * KB);
      cp_async16(smem_u32(sm.w[stage] + r * 32 * KB + (w_chunk<KB>(r, ch) << 4)),
                 W + static_cast<size_t>(n0 + r) * (K / 2) + kt * 32 * KB + ch * 16, true);
    }
    if (tid < BN)
      cp_async_bytes<4 * KB>(smem_u32(sm.s[stage] + tid * 4 * KB),
                             S + static_cast<size_t>(n0 + tid) * (K / 16) + kt * 4 * KB);
  };
  auto load_stage = [&](int stage, int kt) { load_w_stage(stage, kt); load_a_stage(stage, kt); };

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  // Prologue: weights (static) before the grid dependency, activations after it; group 0 = all prologue
  // weights + A stage 0, groups 1.. = A stages 1.. (over-waits on later weight stages only once).
#pragma unroll
  for (int s = 0; s < ST - 1; ++s)
    if (s < nkt) load_w_stage(s, kt0 + s);
  pdl_wait();
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nkt) load_a_stage(s, kt0 + s);
    cp_commit();
  }
  pdl_launch();
  for (int j = 0; j < nkt; ++j) {
    cp_wait<ST - 2>();
    __syncthreads();
    const bool more = j + ST - 1 < nkt;
    if (more) load_stage((j + ST - 1) % ST, kt0 + j + ST - 1);
    cp_commit();
    const int stg = j % ST;
#pragma unroll
    for (int sub = 0; sub < KB; ++sub) {
      // weight words of this lane's column g for the 64-k sub-block, and the column-pair scales
      uint32_t wq[NT][8];
      uint32_t sw[NT][2];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int r = wn0 + nt * 8 + g;
        const uint8_t* wr = sm.w[stg] + r * 32 * KB;
        const uint4 w0 = *reinterpret_cast<const uint4*>(wr + (w_chunk<KB>(r, 2 * sub) << 4));
        const uint4 w1 = *reinterpret_cast<const uint4*>(wr + (w_chunk<KB>(r, 2 * sub + 1) << 4));
        wq[nt][0] = w0.x; wq[nt][1] = w0.y; wq[nt][2] = w0.z; wq[nt][3] = w0.w;
        wq[nt][4] = w1.x; wq[nt][5] = w1.y; wq[nt][6] = w1.z; wq[nt][7] = w1.w;
        const uint8_t* sp = sm.s[stg] + (wn0 + nt * 8 + 2 * t) * 4 * KB + sub * 4;  // columns 2t, 2t+1
        sw[nt][0] = *reinterpret_cast<const uint32_t*>(sp);
        sw[nt][1] = *reinterpret_cast<const uint32_t*>(sp + 4 * KB);
      }
#pragma unroll
      for (int ks2 = 0; ks2 < 2; ++ks2) {
        uint32_t af[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int r = wm0 + mt * 16 + (lane & 15);
          const int ch = sub * 4 + ks2 * 2 + (lane >> 4);
          ldmatrix_x4(af[mt], smem_u32(sm.a[stg] + r * SK + (a_chunk<KB>(r, ch) << 4)));
        }
        float as0[MT], as1[MT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const float* ap = sm.as[stg] + (wm0 + mt * 16) * 2 * KB + sub * 2 + ks2;
          as0[mt] = ap[g * 2 * KB];
          as1[mt] = ap[(g + 8) * 2 * KB];
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int ks = ks2 * 2 + h;
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            const uint32_t q = (t & 2) ? wq[nt][2 * ks + 1] : wq[nt][2 * ks];
            const uint32_t b = weight_word_s8x4(q, t & 1);
            const uint32_t sbyte2 = __byte_perm(sw[nt][0], sw[nt][1], ks | ((4 + ks) << 4));
            const float2 ws = h2_to_f2(e4m3x2_to_f16x2(sbyte2));
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              int c[4] = {0, 0, 0, 0};
              mma16816_s8(c, af[mt][2 * h], af[mt][2 * h + 1], b);
              acc[mt][nt][0] = fmaf(__int2float_rn(c[0]), as0[mt] * ws.x, acc[mt][nt][0]);
              acc[mt][nt][1] = fmaf(__int2float_rn(c[1]), as0[mt] * ws.y, acc[mt][nt][1]);
              acc[mt][nt][2] = fmaf(__int2float_rn(c[2]), as1[mt] * ws.x, acc[mt][nt][2]);
              acc[mt][nt][3] = fmaf(__int2float_rn(c[3]), as1[mt] * ws.y, acc[mt][nt][3]);
            }
          }
        }
      }
    }
  }
  cp_wait<0>();

  if (split > 1) {
    cg::cluster_group cluster = cg::this_cluster();
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem_raw);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) red[((mt * NT + nt) * 4 + e) * THREADS + tid] = acc[mt][nt][e];
    cluster.sync();
    const bool leader = cluster.block_rank() == 0;
    if (leader) {
      for (int r = 1; r < split; ++r) {
        const float* rem = cluster.map_shared_rank(red, r);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[mt][nt][e] += rem[((mt * NT + nt) * 4 + e) * THREADS + tid];
      }
    }
    cluster.sync();
    if (!leader) return;
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = wm0 + mt * 16 + g + 8 * h;
      if (r >= m_rows) continue;
      const size_t crow = static_cast<size_t>(c_row0 + r) * ldc;
      if (EPI == kEpiBf16) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int col = n0 + wn0 + nt * 8 + 2 * t;
          *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(C) + crow + col) =
              pack_bf16x2(acc[mt][nt][2 * h] * gs, acc[mt][nt][2 * h + 1] * gs);
        }
      } else if (EPI == kEpiF32) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int col = n0 + wn0 + nt * 8 + 2 * t;
          *reinterpret_cast<float2*>(static_cast<float*>(C) + crow + col) =
              make_float2(acc[mt][nt][2 * h] * gs, acc[mt][nt][2 * h + 1] * gs);
        }
      } else {
#pragma unroll
        for (int q = 0; q < NT / 2; ++q) {
          const int col = (n0 + wn0) / 2 + q * 8 + 2 * t;  // act column
          float a0, a1;
          const float g0 = acc[mt][2 * q][2 * h] * gs, g1 = acc[mt][2 * q][2 * h + 1] * gs;
          const float u0 = acc[mt][2 * q + 1][2 * h] * gs, u1 = acc[mt][2 * q + 1][2 * h + 1] * gs;
          if (EPI == kEpiSwigluDense) {
            a0 = silu_mul_precise(round_bf16(g0), round_bf16(u0));
            a1 = silu_mul_precise(round_bf16(g1), round_bf16(u1));
          } else {
            a0 = silu_mul_routed(g0, u0);
            a1 = silu_mul_routed(g1, u1);
          }
          *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(C) + crow + col) = pack_bf16x2(a0, a1);
        }
      }
    }
  }
}

// grid (ceil(M / BM), N / BN, split); split > 1 as (1, 1, split) clusters.
template <int BM, int ST, int KB, int EPI>
__global__ void __launch_bounds__(THREADS)
gemm_s8_dense_kernel(const int8_t* __restrict__ A, const float* __restrict__ AS, int M, int K,
                     const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, float gs, void* __restrict__ C,
                     int ldc) {
  constexpr int AI = (BM * 4 * KB + THREADS - 1) / THREADS;
  constexpr int ASI = (BM * as_pieces<KB>() + THREADS - 1) / THREADS;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int rows = min(BM, M - m0);
  const int8_t* arow[AI];
  const float* asrow[ASI];
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int r = a_row_of<BM, KB>(i);
    arow[i] = r < rows ? A + static_cast<size_t>(m0 + r) * K : nullptr;
  }
#pragma unroll
  for (int i = 0; i < ASI; ++i) {
    const int r = as_row_of<BM, KB>(i);
    asrow[i] = r < rows ? AS + static_cast<size_t>(m0 + r) * (K / 32) : nullptr;
  }
  const int split = gridDim.z, nkt = K / (64 * KB) / split;
  gemm_s8_body<BM, ST, KB, EPI>(arow, asrow, rows, W, S, K, blockIdx.z * nkt, nkt, split, n0, gs, C, m0, ldc);
}

// Grouped: tile i covers sorted rows [row0, row0 + rows) of expert e; A rows gathered through `gather`
// (nullptr: the row itself). grid (max_tiles, N / BN, split).
template <int BM, int ST, int KB, int EPI>
__global__ void __launch_bounds__(THREADS)
gemm_s8_grouped_kernel(const int8_t* __restrict__ A, const float* __restrict__ AS, const int* __restrict__ gather,
                       const int* __restrict__ tiles, const int* __restrict__ n_tiles, const uint8_t* __restrict__ W,
                       const uint8_t* __restrict__ S, int N, int K, const float* __restrict__ gs_e,
                       void* __restrict__ C, int ldc) {
  constexpr int AI = (BM * 4 * KB + THREADS - 1) / THREADS;
  constexpr int ASI = (BM * as_pieces<KB>() + THREADS - 1) / THREADS;
  pdl_wait();
  // grid (N / BN, tiles, split): the column tiles of one row tile are adjacent in launch order, so the gathered
  // A rows and the expert's weight slabs are re-read from L2 rather than DRAM (prefill-sized expert slices)
  const int t = blockIdx.y;
  if (t >= *n_tiles) return;
  const int e = tiles[3 * t], row0 = tiles[3 * t + 1], rows = tiles[3 * t + 2];
  const int n0 = blockIdx.x * BN;
  const int8_t* arow[AI];
  const float* asrow[ASI];
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int r = a_row_of<BM, KB>(i);
    const int src = gather ? (r < rows ? gather[row0 + r] : 0) : row0 + r;
    arow[i] = r < rows ? A + static_cast<size_t>(src) * K : nullptr;
  }
#pragma unroll
  for (int i = 0; i < ASI; ++i) {
    const int r = as_row_of<BM, KB>(i);
    const int src = gather ? (r < rows ? gather[row0 + r] : 0) : row0 + r;
    asrow[i] = r < rows ? AS + static_cast<size_t>(src) * (K / 32) : nullptr;
  }
  const uint8_t* We = W + static_cast<size_t>(e) * N * (K / 2);
  const uint8_t* Se = S + static_cast<size_t>(e) * N * (K / 16);
  const int split = gridDim.z, nkt = K / (64 * KB) / split;
  gemm_s8_body<BM, ST, KB, EPI>(arow, asrow, rows, We, Se, K, blockIdx.z * nkt, nkt, split, n0, 0.5f * gs_e[e], C,
                                row0, ldc);
}

// ---- activation quantization: bf16 rows -> int8 (per-32 fp32 scale), block-permuted K order ----
// One thread per 32-wide block (independent: one 64-byte load, one 32-byte store).
__global__ void __launch_bounds__(256) quant_rows_s8_kernel(const uint16_t* __restrict__ X, int M, int K, int ldx,
                                                            int8_t* __restrict__ Q, float* __restrict__ S) {
  pdl_wait();
  const int nb = K / 32;
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= M * nb) return;
  const int row = idx / nb, b = idx - row * nb;
  const uint16_t* x = X + static_cast<size_t>(row) * ldx + b * 32;
  float v[32];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 u = *reinterpret_cast<const uint4*>(x + i * 8);
    v[i * 8 + 0] = bf16lo(u.x); v[i * 8 + 1] = bf16hi(u.x);
    v[i * 8 + 2] = bf16lo(u.y); v[i * 8 + 3] = bf16hi(u.y);
    v[i * 8 + 4] = bf16lo(u.z); v[i * 8 + 5] = bf16hi(u.z);
    v[i * 8 + 6] = bf16lo(u.w); v[i * 8 + 7] = bf16hi(u.w);
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const float scale = amax / 127.f;
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  // permuted order within each 16-block: 0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15
  uint32_t o[8];
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    uint32_t packed = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int p = w * 4 + j, blk = p >> 4, pp = p & 15;
      const int src = blk * 16 + ((pp & 8) | ((pp & 3) << 1) | ((pp >> 2) & 1));
      const int r = max(-127, min(127, __float2int_rn(v[src] * inv)));
      packed |= (static_cast<uint32_t>(r) & 0xffu) << (8 * j);
    }
    o[w] = packed;
  }
  int8_t* q = Q + static_cast<size_t>(row) * K + b * 32;
  *reinterpret_cast<uint4*>(q) = make_uint4(o[0], o[1], o[2], o[3]);
  *reinterpret_cast<uint4*>(q + 16) = make_uint4(o[4], o[5], o[6], o[7]);
  S[static_cast<size_t>(row) * nb + b] = scale;
}

// Per-row variant (quality experiment): one warp per row, one scale for the whole row, replicated per block.
__global__ void __launch_bounds__(256) quant_rows_s8_perrow_kernel(const uint16_t* __restrict__ X, int M, int K,
                                                                   int ldx, int8_t* __restrict__ Q,
                                                                   float* __restrict__ S) {
  pdl_wait();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= M) return;
  const uint16_t* x = X + static_cast<size_t>(row) * ldx;
  float amax = 0.f;
  for (int i = lane * 8; i < K; i += 256) {
    const uint4 u = *reinterpret_cast<const uint4*>(x + i);
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(bf16lo(u.x)), fabsf(bf16hi(u.x))), fmaxf(fabsf(bf16lo(u.y)), fabsf(bf16hi(u.y)))));
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(bf16lo(u.z)), fabsf(bf16hi(u.z))), fmaxf(fabsf(bf16lo(u.w)), fabsf(bf16hi(u.w)))));
  }
  amax = warp_max(amax);
  const float scale = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
  const int nb = K / 32;
  for (int b = lane; b < nb; b += 32) {
    float v[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 u = *reinterpret_cast<const uint4*>(x + b * 32 + i * 8);
      v[i * 8 + 0] = bf16lo(u.x); v[i * 8 + 1] = bf16hi(u.x); v[i * 8 + 2] = bf16lo(u.y); v[i * 8 + 3] = bf16hi(u.y);
      v[i * 8 + 4] = bf16lo(u.z); v[i * 8 + 5] = bf16hi(u.z); v[i * 8 + 6] = bf16lo(u.w); v[i * 8 + 7] = bf16hi(u.w);
    }
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      uint32_t packed = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int p = w * 4 + j, blk = p >> 4, pp = p & 15;
        const int src = blk * 16 + ((pp & 8) | ((pp & 3) << 1) | ((pp >> 2) & 1));
        const int r = max(-127, min(127, __float2int_rn(v[src] * inv)));
        packed |= (static_cast<uint32_t>(r) & 0xffu) << (8 * j);
      }
      o[w] = packed;
    }
    int8_t* q = Q + static_cast<size_t>(row) * K + b * 32;
    *reinterpret_cast<uint4*>(q) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(q + 16) = make_uint4(o[4], o[5], o[6], o[7]);
    S[static_cast<size_t>(row) * nb + b] = scale;
  }
}

template <typename Kernel>
void set_smem_attr(Kernel fn, size_t bytes) {
  cudaFuncSetAttribute(reinterpret_cast<const void*>(fn), cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(bytes));
}

template <int BM, int ST, int KB, int EPI>
void launch_s8_dense(const void* A, const void* AS, int M, const void* codes, const void* scales, int N, int K,
                     float gs, void* C, int ldc, int split, cudaStream_t s) {
  auto kern = gemm_s8_dense_kernel<BM, ST, KB, EPI>;
  static bool smem_set = false;
  if (!smem_set) { set_smem_attr(kern, gemm_s8_smem_bytes<BM, ST, KB>(8)); smem_set = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((M + BM - 1) / BM, N / BN, split);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = gemm_s8_smem_bytes<BM, ST, KB>(split);
  cfg.stream = s;
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (split > 1) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = 1; attr[na].val.clusterDim.y = 1; attr[na].val.clusterDim.z = split;
    ++na;
  }
  if (pdl_enabled()) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attr; cfg.numAttrs = na;
  cudaLaunchKernelEx(&cfg, kern, static_cast<const int8_t*>(A), static_cast<const float*>(AS), M, K,
                     static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), 0.5f * gs, C, ldc);
}

template <int BM, int ST, int KB, int EPI>
void launch_s8_grouped(const void* A, const void* AS, const int* gather, const int* tiles, const int* n_tiles,
                       int max_tiles, const void* codes, const void* scales, int N, int K, const float* gs_e, void* C,
                       int ldc, int split, cudaStream_t s) {
  auto kern = gemm_s8_grouped_kernel<BM, ST, KB, EPI>;
  static bool smem_set = false;
  if (!smem_set) { set_smem_attr(kern, gemm_s8_smem_bytes<BM, ST, KB>(8)); smem_set = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / BN, max_tiles, split);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = gemm_s8_smem_bytes<BM, ST, KB>(split);
  cfg.stream = s;
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (split > 1) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = 1; attr[na].val.clusterDim.y = 1; attr[na].val.clusterDim.z = split;
    ++na;
  }
  if (pdl_enabled()) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attr; cfg.numAttrs = na;
  cudaLaunchKernelEx(&cfg, kern, static_cast<const int8_t*>(A), static_cast<const float*>(AS), gather, tiles,
                     n_tiles, static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), N, K, gs_e,
                     C, ldc);
}

// ---------------------------------------------------------------------------
// Small-M dense GEMM (M <= 128): one CTA per BN-column strip over the whole K.
// The strip's weights (BN x K codes + scales, <= ~54 KB) are prefetched into
// shared memory up front (three cp.async groups, so the first third of K is
// computed while the rest lands); the int8 activations stream through a
// three-stage ring from L2. Warps: (BN / 16) column pairs x KS k-slices; the
// k-slice partials are reduced through shared memory before the epilogue.
// ---------------------------------------------------------------------------
template <int BM, int BN, int KB, int KMAX>
struct NarrowSmemLayout {
  static constexpr int kStages = 3;
  static constexpr int kMaxK = KMAX;
  static constexpr size_t w_bytes = static_cast<size_t>(BN) * kMaxK / 2;
  static constexpr size_t s_bytes = static_cast<size_t>(BN) * kMaxK / 16;
  static constexpr size_t a_bytes = static_cast<size_t>(kStages) * BM * 64 * KB;
  static constexpr size_t as_bytes = static_cast<size_t>(kStages) * BM * 2 * KB * 4;
  static constexpr size_t total = w_bytes + s_bytes + a_bytes + as_bytes;
};

template <int BM, int BN, int KB, int KMAX, int EPI>
__global__ void __launch_bounds__(THREADS)
gemm_s8_narrow_kernel(const int8_t* __restrict__ A, const float* __restrict__ AS, int M, int K,
                      const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, float gs, void* __restrict__ C,
                      int ldc) {
  using L = NarrowSmemLayout<BM, BN, KB, KMAX>;
  constexpr int ST = L::kStages;
  constexpr int NPAIR = BN / 16;            // warps along N (each owns two n-tiles: gate|up pairs)
  constexpr int MT = BM / 16;
  constexpr int NT = 2;
  constexpr int SK = 64 * KB;               // k per A stage
  constexpr int K16 = 4 * KB;               // k16 blocks per stage
  constexpr int KS = (8 / NPAIR) < K16 ? (8 / NPAIR) : K16;  // k-slices per stage (idle warps beyond)
  constexpr int KPW = K16 / KS;             // k16 blocks per warp per stage
  static_assert(KPW >= 1 && K16 % KS == 0, "stage too small for the k-slices");
  constexpr int ACH = 4 * KB;
  constexpr int AI = (BM * ACH + THREADS - 1) / THREADS;
  constexpr int ASP = as_pieces<KB>();
  constexpr int ASI = (BM * ASP + THREADS - 1) / THREADS;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  uint8_t* sw = smem_raw;                                         // [BN][K/2], chunk-swizzled by row
  uint8_t* ss = smem_raw + L::w_bytes;                            // [BN][K/16]
  int8_t* sa = reinterpret_cast<int8_t*>(smem_raw + L::w_bytes + L::s_bytes);
  float* sas = reinterpret_cast<float*>(smem_raw + L::w_bytes + L::s_bytes + L::a_bytes);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int pair = warp % NPAIR, ks_id = warp / NPAIR;
  const int n0 = blockIdx.x * BN, wn0 = pair * 16;
  const int rows = min(BM, M);
  const int nkt = K / SK;
  const int wrow = K / 2, wchunks = K / 32;  // bytes / 16B chunks per weight row

  // ---- weight strip: codes + scales, in three groups ----
  auto load_w_range = [&](int kt_begin, int kt_end) {
    const int c0 = kt_begin * (2 * KB), c1 = kt_end * (2 * KB);  // chunk range per row
    for (int c = tid; c < BN * (c1 - c0); c += THREADS) {
      const int r = c / (c1 - c0), ch = c0 + c % (c1 - c0);
      cp_async16(smem_u32(sw + r * wrow + ((ch ^ (r & 7)) << 4)), W + static_cast<size_t>(n0 + r) * wrow + ch * 16,
                 true);
    }
  };
  auto load_s_range = [&](int kt_begin, int kt_end) {
    // scales: row n has K/16 bytes; 4*KB bytes per k-stage, row-major in smem
    const int n = kt_end - kt_begin;
    for (int c = tid; c < BN * n; c += THREADS) {
      const int r = c / n, kt = kt_begin + c % n;
      cp_async_bytes<4 * KB>(smem_u32(ss + r * (K / 16) + kt * 4 * KB),
                             S + static_cast<size_t>(n0 + r) * (K / 16) + kt * 4 * KB);
    }
  };
  const int8_t* arow[AI];
  const float* asrow[ASI];
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int r = a_row_of<BM, KB>(i);
    arow[i] = r < rows ? A + static_cast<size_t>(r) * K : nullptr;
  }
#pragma unroll
  for (int i = 0; i < ASI; ++i) {
    const int r = as_row_of<BM, KB>(i);
    asrow[i] = r < rows ? AS + static_cast<size_t>(r) * (K / 32) : nullptr;
  }
  auto load_a_stage = [&](int stage, int kt) {
    int8_t* a = sa + static_cast<size_t>(stage) * BM * SK;
    float* as = sas + static_cast<size_t>(stage) * BM * 2 * KB;
#pragma unroll
    for (int i = 0; i < AI; ++i) {
      const int c = tid + i * THREADS;
      if (BM * ACH % THREADS == 0 || c < BM * ACH) {
        const int r = c / ACH, ch = c % ACH;
        const bool ok = arow[i] != nullptr;
        cp_async16(smem_u32(a + r * SK + (a_chunk<KB>(r, ch) << 4)),
                   ok ? static_cast<const void*>(arow[i] + kt * SK + ch * 16) : static_cast<const void*>(W), ok);
      }
    }
#pragma unroll
    for (int i = 0; i < ASI; ++i) {
      const int c = tid + i * THREADS;
      if (BM * ASP % THREADS == 0 || c < BM * ASP) {
        const int r = c / ASP, p = c % ASP;
        const bool ok = asrow[i] != nullptr;
        if constexpr (KB >= 2)
          cp_async16(smem_u32(as + r * 2 * KB + p * 4),
                     ok ? static_cast<const void*>(asrow[i] + kt * 2 * KB + p * 4) : static_cast<const void*>(W), ok);
        else
          cp_async8(smem_u32(as + r * 2 * KB),
                    ok ? static_cast<const void*>(asrow[i] + kt * 2 * KB) : static_cast<const void*>(W), ok);
      }
    }
  };
  // prologue: group s = weight stage s (+ activation stage s for the first ST), so stage j's weights are
  // complete once at most (nkt - 1) groups are pending at iteration j (each iteration commits one more group).
  for (int s = 0; s < nkt; ++s) {  // weights: one group per k-stage, issued before the grid dependency
    load_w_range(s, s + 1);
    load_s_range(s, s + 1);
    cp_commit();
  }
  pdl_wait();
#pragma unroll
  for (int s = 0; s < ST; ++s) {  // activations: three more groups
    if (s < nkt) load_a_stage(s, s);
    cp_commit();
  }
  pdl_launch();

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  for (int j = 0; j < nkt; ++j) {
    // A stage j is group nkt + j; each iteration commits one refill, so "at most ST - 1 pending" is exact.
    cp_wait<ST - 1>();
    __syncthreads();
    const int stg = j % ST;
    const int8_t* a = sa + static_cast<size_t>(stg) * BM * SK;
    const float* as = sas + static_cast<size_t>(stg) * BM * 2 * KB;
#pragma unroll
    for (int kk = 0; kk < KPW; ++kk) {
      if (ks_id >= KS) break;                   // idle k-slice warp (loads and barriers only)
      const int k16 = ks_id * KPW + kk;        // k16 block within the stage
      const int kglob = j * K16 + k16;         // global k16 block
      const int sub = k16 >> 2, ks = k16 & 3;  // 64-k sub-block and k16 within it
      // A fragments (one ldmatrix.x4 covers two k16 blocks; use the half we need)
      uint32_t af[MT][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r = mt * 16 + (lane & 15);  // lanes 16..31 repeat the addresses (ignored by .x2)
        ldmatrix_x2(af[mt], smem_u32(a + r * SK + (a_chunk<KB>(r, k16) << 4)));
      }
      float as0[MT], as1[MT];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float* ap = as + (mt * 16) * 2 * KB + (k16 >> 1);
        as0[mt] = ap[g * 2 * KB];
        as1[mt] = ap[(g + 8) * 2 * KB];
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int r = wn0 + nt * 8 + g;
        const int wch = kglob >> 1, wbyte = (kglob & 1) * 8 + (t >> 1) * 4;
        const uint32_t q = *reinterpret_cast<const uint32_t*>(sw + r * wrow + ((wch ^ (r & 7)) << 4) + wbyte);
        const uint32_t b = weight_word_s8x4(q, t & 1);
        const uint8_t* sp = ss + (wn0 + nt * 8 + 2 * t) * (K / 16) + kglob;  // columns 2t, 2t+1
        const uint32_t sbyte2 = sp[0] | (static_cast<uint32_t>(sp[K / 16]) << 8);
        const float2 ws = h2_to_f2(e4m3x2_to_f16x2(sbyte2));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          int c[4] = {0, 0, 0, 0};
          mma16816_s8(c, af[mt][0], af[mt][1], b);
          acc[mt][nt][0] = fmaf(__int2float_rn(c[0]), as0[mt] * ws.x, acc[mt][nt][0]);
          acc[mt][nt][1] = fmaf(__int2float_rn(c[1]), as0[mt] * ws.y, acc[mt][nt][1]);
          acc[mt][nt][2] = fmaf(__int2float_rn(c[2]), as1[mt] * ws.x, acc[mt][nt][2]);
          acc[mt][nt][3] = fmaf(__int2float_rn(c[3]), as1[mt] * ws.y, acc[mt][nt][3]);
        }
      }
    }
    __syncthreads();  // everyone is done with stage j before it is refilled
    if (j + ST <= nkt) { if (j + ST < nkt + 1) {} }
    if (j + ST < nkt) load_a_stage(stg, j + ST);
    cp_commit();
  }
  cp_wait<0>();
  __syncthreads();

  // ---- reduce the k-slices (fp32 [KS][BM][BN] over the weight region) and run the epilogue ----
  float* red = reinterpret_cast<float*>(smem_raw);
  static_assert(static_cast<size_t>(KS) * BM * BN * 4 <= L::total, "reduction tile too large");
  if (ks_id < KS)
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = mt * 16 + g + 8 * h, col = wn0 + nt * 8 + 2 * t;
        *reinterpret_cast<float2*>(red + (static_cast<size_t>(ks_id) * BM + r) * BN + col) =
            make_float2(acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]);
      }
  __syncthreads();
  if (ks_id != 0) return;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = mt * 16 + g + 8 * h, col = wn0 + nt * 8 + 2 * t;
        float2 v = make_float2(0.f, 0.f);
#pragma unroll
        for (int s = 0; s < KS; ++s) {
          const float2 u = *reinterpret_cast<const float2*>(red + (static_cast<size_t>(s) * BM + r) * BN + col);
          v.x += u.x; v.y += u.y;
        }
        acc[mt][nt][2 * h] = v.x; acc[mt][nt][2 * h + 1] = v.y;
      }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = mt * 16 + g + 8 * h;
      if (r >= rows) continue;
      const size_t crow = static_cast<size_t>(r) * ldc;
      if (EPI == kEpiBf16) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int col = n0 + wn0 + nt * 8 + 2 * t;
          *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(C) + crow + col) =
              pack_bf16x2(acc[mt][nt][2 * h] * gs, acc[mt][nt][2 * h + 1] * gs);
        }
      } else if (EPI == kEpiF32) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int col = n0 + wn0 + nt * 8 + 2 * t;
          *reinterpret_cast<float2*>(static_cast<float*>(C) + crow + col) =
              make_float2(acc[mt][nt][2 * h] * gs, acc[mt][nt][2 * h + 1] * gs);
        }
      } else {
        const int col = (n0 + wn0) / 2 + 2 * t;
        const float g0 = acc[mt][0][2 * h] * gs, g1 = acc[mt][0][2 * h + 1] * gs;
        const float u0 = acc[mt][1][2 * h] * gs, u1 = acc[mt][1][2 * h + 1] * gs;
        float a0, a1;
        if (EPI == kEpiSwigluDense) {
          a0 = silu_mul_precise(round_bf16(g0), round_bf16(u0));
          a1 = silu_mul_precise(round_bf16(g1), round_bf16(u1));
        } else {
          a0 = silu_mul_routed(g0, u0);
          a1 = silu_mul_routed(g1, u1);
        }
        *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(C) + crow + col) = pack_bf16x2(a0, a1);
      }
    }
  }
}

template <int BM, int BN, int KB, int KMAX, int EPI>
void launch_s8_narrow(const void* A, const void* AS, int M, const void* codes, const void* scales, int N, int K,
                      float gs, void* C, int ldc, cudaStream_t s) {
  auto kern = gemm_s8_narrow_kernel<BM, BN, KB, KMAX, EPI>;
  static bool smem_set = false;
  if (!smem_set) { set_smem_attr(kern, NarrowSmemLayout<BM, BN, KB, KMAX>::total); smem_set = true; }
  launch_pdl(kern, dim3(N / BN), dim3(THREADS), NarrowSmemLayout<BM, BN, KB, KMAX>::total, s,
             static_cast<const int8_t*>(A), static_cast<const float*>(AS), M, K, static_cast<const uint8_t*>(codes),
             static_cast<const uint8_t*>(scales), 0.5f * gs, C, ldc);
}

// Default configurations (RTX 5090; first cut, refined by the standalone sweep).
GemmConfig default_s8_dense_config(int M, int N, int K) {
  // Graph-captured sweep over the Flint shapes (tools/s8_graph_bench.py, tools/s8_test.py --bench).
  auto fit = [&](GemmConfig c) {  // keep K % (64 * kb * split) == 0
    while (c.split > 1 && K % (64 * c.kb * c.split)) c.split >>= 1;
    return c;
  };
  const bool wide = N >= 65536;  // vocabulary projection
  if (M <= 16) return fit(wide ? GemmConfig{16, 3, 1, 4} : GemmConfig{16, 3, N >= 8192 ? 1 : (N >= 4096 ? 2 : 4), 4});
  if (M <= 64) {
    if (wide || N >= 8192) return fit({64, 2, 1, 4});
    if (N >= 4096) return fit({32, 3, 2, 4});
    if (K >= 4096) return fit({16, 3, 2, 4});
    return fit({16, 4, 2, 2});
  }
  if (M <= 128) {
    if (wide || N >= 8192) return fit({128, 3, 1, 2});
    return fit({32, 3, N >= 4096 ? 1 : 2, 4});
  }
  if (M <= 256) return fit({64, 2, N >= 4096 ? 1 : 2, 4});
  if (M <= 1024 || N <= 2048) return fit({64, 2, 1, 4});
  return fit({128, 3, 1, 2});
}
GemmConfig default_s8_grouped_config(int tile_rows, int mean_rows, int K) {
  if (tile_rows <= 16) return K <= 512 ? GemmConfig{16, 4, 1, 2} : GemmConfig{16, 3, 1, 4};
  if (tile_rows <= 32) return {32, 3, 1, 4};
  if (tile_rows <= 64) return {64, 2, 1, 4};
  // prefill-sized expert slices (>= 128 rows per tile): 128-row tiles halve the per-expert weight re-reads
  return K <= 512 ? GemmConfig{128, 3, 1, 2} : GemmConfig{128, 3, 1, 2};
}
GemmConfig g_s8_dense_override = {0, 0, 0, 0};
GemmConfig g_s8_grouped_override = {0, 0, 0, 0};

}  // namespace

void set_gemm_s8_override(const GemmConfig& dense, const GemmConfig& grouped) {
  g_s8_dense_override = dense;
  g_s8_grouped_override = grouped;
}

void quant_rows_s8_perrow(const void* x, int M, int K, int ldx, void* q, float* s, cudaStream_t st) {
  launch_pdl(quant_rows_s8_perrow_kernel, dim3((M + 7) / 8), dim3(256), 0, st, static_cast<const uint16_t*>(x), M, K,
             ldx, static_cast<int8_t*>(q), s);
}

void quant_rows_s8(const void* x, int M, int K, int ldx, void* q, float* s, cudaStream_t st) {
  const long long n = static_cast<long long>(M) * (K / 32);
  launch_pdl(quant_rows_s8_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0, st,
             static_cast<const uint16_t*>(x), M, K, ldx, static_cast<int8_t*>(q), s);
}

int gemm_w4a8_cfg(const void* A, const void* AS, int M, const void* codes, const void* scales, int N, int K,
                  float gs, void* C, int ldc, int epilogue, GemmConfig c, cudaStream_t s) {
  if (c.kb != 1 && c.kb != 2 && c.kb != 4) return 1;
  if (N % BN || K % 64 || c.split < 1 || c.split > 8 || K % (64 * c.kb * c.split)) return 1;
#define FLINT_S8_CASE(BM_, ST_, KB_, EPI_)                                                              \
  if (c.bm == BM_ && c.stages == ST_ && c.kb == KB_ && epilogue == EPI_) {                              \
    launch_s8_dense<BM_, ST_, KB_, EPI_>(A, AS, M, codes, scales, N, K, gs, C, ldc, c.split, s);      \
    return 0;                                                                                           \
  }
#define FLINT_S8_EPIS(BM_, ST_, KB_)                                                        \
  FLINT_S8_CASE(BM_, ST_, KB_, kEpiBf16) FLINT_S8_CASE(BM_, ST_, KB_, kEpiF32)             \
  FLINT_S8_CASE(BM_, ST_, KB_, kEpiSwigluDense) FLINT_S8_CASE(BM_, ST_, KB_, kEpiSwigluRouted)
  FLINT_S8_EPIS(16, 4, 2)
  FLINT_S8_EPIS(16, 3, 4)
  FLINT_S8_EPIS(32, 4, 2)
  FLINT_S8_EPIS(32, 3, 4)
  FLINT_S8_EPIS(64, 3, 2)
  FLINT_S8_EPIS(64, 2, 4)
  FLINT_S8_EPIS(128, 3, 1)
  FLINT_S8_EPIS(128, 3, 2)
#undef FLINT_S8_EPIS
#undef FLINT_S8_CASE
  return 1;
}

namespace {
inline bool s8_narrow_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("FLINT_S8_NARROW");
    return !(v && v[0] == '0');
  }();
  return on;
}
inline int s8_narrow_max_m() {  // graph-captured sweep: the strip kernel wins up to 16 rows (FLINT_S8_NARROW_MAX_M)
  static const int m = [] {
    const char* v = std::getenv("FLINT_S8_NARROW_MAX_M");
    return v ? std::atoi(v) : 16;
  }();
  return m;
}
}  // namespace

// Small-M strip kernel: M <= 128, K <= 6144 (K % 256 == 0 for the KB = 4 variants). Returns non-zero if not applicable.
inline bool s8_narrow_compact() {  // FLINT_S8_NARROW_COMPACT=1: <= 50 KB strips so two CTAs share an SM
  static const bool on = [] {
    const char* v = std::getenv("FLINT_S8_NARROW_COMPACT");
    return v && v[0] == '1';
  }();
  return on;
}

int gemm_w4a8_narrow(const void* A, const void* AS, int M, const void* codes, const void* scales, int N, int K,
                     float gs, void* C, int ldc, int epilogue, cudaStream_t s) {
  if (M < 1 || M > 128 || K > 6144 || K % 64 || N % 16 || N > 8192 || epilogue == kEpiSwigluRouted) return 1;
  const int bm = M <= 16 ? 16 : (M <= 64 ? 64 : 128);
  int bn = K <= 2048 ? 32 : 16;
  int kb = bm <= 64 ? (K <= 4096 ? 4 : 2) : 2;
  int kmax = bn >= 32 ? 2048 : (kb >= 4 || bm >= 128 ? 4096 : 6144);
  if (bm == 64 && s8_narrow_compact() && K <= 4096) { bn = 16; kb = K <= 2048 ? 2 : 1; kmax = K <= 2048 ? 2048 : 4096; }
  if (bm == 128 && K > 4096) return 1;  // the 128-row ring does not fit beside a 6144-deep strip
  if (N % bn || K % (64 * kb)) return 1;
#define FLINT_S8N_CASE(BM_, BN_, KB_, KMAX_, EPI_)                                                          \
  if (bm == BM_ && bn == BN_ && kb == KB_ && kmax == KMAX_ && epilogue == EPI_) {                           \
    launch_s8_narrow<BM_, BN_, KB_, KMAX_, EPI_>(A, AS, M, codes, scales, N, K, gs, C, ldc, s);            \
    return 0;                                                                                               \
  }
#define FLINT_S8N_EPIS(BM_, BN_, KB_, KMAX_)                                                            \
  FLINT_S8N_CASE(BM_, BN_, KB_, KMAX_, kEpiBf16) FLINT_S8N_CASE(BM_, BN_, KB_, KMAX_, kEpiF32)         \
  FLINT_S8N_CASE(BM_, BN_, KB_, KMAX_, kEpiSwigluDense)
  FLINT_S8N_EPIS(16, 32, 4, 2048) FLINT_S8N_EPIS(16, 16, 4, 4096) FLINT_S8N_EPIS(16, 16, 2, 6144)
  FLINT_S8N_EPIS(64, 32, 4, 2048) FLINT_S8N_EPIS(64, 16, 4, 4096) FLINT_S8N_EPIS(64, 16, 2, 6144)
  FLINT_S8N_EPIS(64, 16, 2, 2048) FLINT_S8N_EPIS(64, 16, 1, 4096)
  FLINT_S8N_EPIS(128, 32, 2, 2048) FLINT_S8N_EPIS(128, 16, 2, 4096)
#undef FLINT_S8N_EPIS
#undef FLINT_S8N_CASE
  return 1;
}

int gemm_w4a8(const void* A, const void* AS, int M, const void* codes, const void* scales, int N, int K, float gs,
              void* C, int ldc, int epilogue, cudaStream_t s) {
  if (M <= s8_narrow_max_m() && s8_narrow_enabled() && !g_s8_dense_override.bm &&
      gemm_w4a8_narrow(A, AS, M, codes, scales, N, K, gs, C, ldc, epilogue, s) == 0)
    return 0;
  const GemmConfig c = g_s8_dense_override.bm ? g_s8_dense_override : default_s8_dense_config(M, N, K);
  return gemm_w4a8_cfg(A, AS, M, codes, scales, N, K, gs, C, ldc, epilogue, c, s);
}

int gemm_w4a8_grouped(const void* A, const void* AS, const int* gather, const int* tiles, const int* n_tiles,
                      int max_tiles, const void* codes, const void* scales, int N, int K, const float* gs_e, void* C,
                      int ldc, int epilogue, int tile_rows, int mean_rows, cudaStream_t s) {
  const GemmConfig c =
      g_s8_grouped_override.bm ? g_s8_grouped_override : default_s8_grouped_config(tile_rows, mean_rows, K);
  if (c.bm < tile_rows || N % BN || c.split < 1 || c.split > 8 || K % (64 * c.kb * c.split)) return 1;
#define FLINT_S8G_CASE(BM_, ST_, KB_, EPI_)                                                                  \
  if (c.bm == BM_ && c.stages == ST_ && c.kb == KB_ && epilogue == EPI_) {                                   \
    launch_s8_grouped<BM_, ST_, KB_, EPI_>(A, AS, gather, tiles, n_tiles, max_tiles, codes, scales, N, K, gs_e, \
                                           C, ldc, c.split, s);                                              \
    return 0;                                                                                                \
  }
#define FLINT_S8G_EPIS(BM_, ST_, KB_) FLINT_S8G_CASE(BM_, ST_, KB_, kEpiBf16) FLINT_S8G_CASE(BM_, ST_, KB_, kEpiSwigluRouted)
  FLINT_S8G_EPIS(16, 4, 2)
  FLINT_S8G_EPIS(16, 3, 4)
  FLINT_S8G_EPIS(32, 4, 2)
  FLINT_S8G_EPIS(32, 3, 4)
  FLINT_S8G_EPIS(64, 3, 2)
  FLINT_S8G_EPIS(64, 2, 4)
  FLINT_S8G_EPIS(128, 3, 1)
  FLINT_S8G_EPIS(128, 3, 2)
#undef FLINT_S8G_EPIS
#undef FLINT_S8G_CASE
  return 1;
}

}  // namespace flint
}  // namespace flash_rt
