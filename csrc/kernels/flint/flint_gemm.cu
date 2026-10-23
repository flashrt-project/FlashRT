// SPDX-License-Identifier: Apache-2.0
//
// Flint W4A16 tensor-core GEMM for prefill (many tokens):
//   Y[m, n] = epilogue( gs * sum_k A[m, k] * W[n, k] ),  A bf16 [M, K], W NVFP4 [N, K].
// W is dequantized exactly (e2m1 * e4m3 in fp16, converted to bf16 losslessly)
// straight into mma.sync m16n8k16 B fragments; accumulation is fp32.
//
// Tiles: BM x 128 x 64, 8 warps, 3-stage cp.async pipeline. One kernel serves
// dense matrices (tiles over M) and expert-grouped MoE matrices (a device tile
// list maps each CTA to (expert, row range); A rows may be gathered).
//
// Gate|up matrices use the interleaved-by-8 row layout: physical row
// 16*(a/8) + (a%8) is gate row a, +8 is up row a, so a warp's even/odd n-tiles
// hold matching gate/up columns and SwiGLU is applied in registers.

#include "flint_common.cuh"
#include "flint_kernels.cuh"

#include <cfloat>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace flash_rt {
namespace flint {

namespace {

constexpr int BN = 128, BK = 64, THREADS = 256;  // BK: router k-tile

// Epilogue inputs of kEpiQkRope (unused by the other epilogues).
struct QkRopeArgs {
  const float* q_norm;
  const float* k_norm;
  const float* cos_t;
  const float* sin_t;
  const int* pos;
  uint16_t* q_out;
  uint16_t* cache;
  int ring, rope;
  float eps;
  float* q8_scales;  // kEpiSwigluRoutedQ8: per-32 activation scales
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One code byte (two E2M1 values, low nibble first) times the f16x2 scale -> exact bf16x2.
__device__ __forceinline__ uint32_t dq_byte_bf16x2(uint32_t byte, uint32_t s2) {
  uint32_t h;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %1;\n cvt.rn.f16x2.e2m1x2 %0, b0;\n}"
      : "=r"(h) : "r"(byte));
  const float2 f = h2_to_f2(hmul2_u32(h, s2));
  uint32_t out;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(f.y), "f"(f.x));
  return out;
}

// One pipeline stage covers KB sub-tiles of 64 k (32 * KB code bytes per weight row: whole DRAM
// sectors per row for KB >= 2, which is what makes streaming-bound shapes reach full bandwidth).
template <int BM, int ST, int KB>
struct GemmSmem {
  uint16_t a[ST][BM * 64 * KB];    // BM rows x 64*KB bf16; 16B chunks XOR-swizzled by row within groups of 8
  uint8_t w[ST][BN * 32 * KB];     // 128 rows x 32*KB code bytes; 16B chunks XOR-swizzled by row
  uint8_t s[ST][BN * 4 * KB];      // 128 rows x 4*KB scale bytes
};

// Dynamic shared memory of a configuration: the stage ring, or the fp32 split-K exchange tile.
template <int BM, int ST, int KB>
constexpr size_t gemm_smem_bytes(int split) {
  return split > 1 && static_cast<size_t>(BM) * BN * 4 > sizeof(GemmSmem<BM, ST, KB>)
             ? static_cast<size_t>(BM) * BN * 4
             : sizeof(GemmSmem<BM, ST, KB>);
}

__device__ __forceinline__ int a_chunk(int r, int ch) { return (ch & ~7) | ((ch ^ r) & 7); }
template <int KB>
__device__ __forceinline__ int w_chunk(int r, int ch) { return ch ^ (r & (2 * KB - 1)); }

template <int BYTES>
__device__ __forceinline__ void cp_async_small(uint32_t dst, const void* src) {
  if constexpr (BYTES == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(dst), "l"(src), "n"(BYTES));
}

// Shared main loop over stages [kt0, kt0 + nkt) (stage = 64*KB k). arow[i] is the global address of the
// A row loaded by this thread's i-th chunk group (nullptr = zero row). split > 1: the CTAs of a (1, 1, split)
// cluster each reduce a K slice; rank 0 sums the fp32 partials in rank order through DSMEM and stores.
// Warp layout: BM = 128 -> 2 x 4 warps (64 x 32 each); BM <= 64 -> 1 x 8 warps (BM x 16 each).
template <int BM, int ST, int KB, int EPI>
__device__ __forceinline__ void gemm_body(const uint16_t* const* arow, int m_rows, const uint8_t* __restrict__ W,
                                          const uint8_t* __restrict__ S, int K, int kt0, int nkt, int split, int n0,
                                          float gs, void* __restrict__ C, int c_row0, int ldc,
                                          const QkRopeArgs& qa = QkRopeArgs{}) {
  constexpr int WM = BM >= 128 ? 2 : 1;  // warps along M
  constexpr int WN = 8 / WM;             // warps along N
  constexpr int MT = BM / (16 * WM);     // m-tiles per warp
  constexpr int NT = BN / (WN * 8);      // n-tiles per warp
  constexpr int ACH = 8 * KB;            // 16B chunks per A row per stage
  constexpr int AI = (BM * ACH + THREADS - 1) / THREADS;
  constexpr int SK = 64 * KB;            // k per stage
  extern __shared__ __align__(16) uint8_t smem_raw[];
  GemmSmem<BM, ST, KB>& sm = *reinterpret_cast<GemmSmem<BM, ST, KB>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int wm0 = (warp / WN) * (MT * 16), wn0 = (warp % WN) * (BN / WN);

  auto load_stage = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < AI; ++i) {
      const int c = tid + i * THREADS;
      if (BM * ACH % THREADS == 0 || c < BM * ACH) {
        const int r = c / ACH, ch = c % ACH;
        const bool ok = arow[i] != nullptr;
        cp_async16(smem_u32(sm.a[stage] + r * SK + (a_chunk(r, ch) << 3)),
                   ok ? static_cast<const void*>(arow[i] + kt * SK + ch * 8) : static_cast<const void*>(W), ok);
      }
    }
#pragma unroll
    for (int i = 0; i < KB; ++i) {  // 128 rows x 2*KB chunks
      const int c = tid + i * THREADS, r = c / (2 * KB), ch = c % (2 * KB);
      cp_async16(smem_u32(sm.w[stage] + r * 32 * KB + (w_chunk<KB>(r, ch) << 4)),
                 W + static_cast<size_t>(n0 + r) * (K / 2) + kt * 32 * KB + ch * 16, true);
    }
    if (tid < BN)
      cp_async_small<4 * KB>(smem_u32(sm.s[stage] + tid * 4 * KB),
                             S + static_cast<size_t>(n0 + tid) * (K / 16) + kt * 4 * KB);
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < nkt) load_stage(s, kt0 + s);
    cp_commit();
  }
  for (int j = 0; j < nkt; ++j) {
    cp_wait<ST - 2>();
    __syncthreads();
    if (j + ST - 1 < nkt) load_stage((j + ST - 1) % ST, kt0 + j + ST - 1);
    cp_commit();
    const int stg = j % ST;
#pragma unroll
    for (int sub = 0; sub < KB; ++sub) {
      uint32_t bw[NT][8];
      uint32_t bs[NT];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int r = wn0 + nt * 8 + g;
        const uint8_t* wr = sm.w[stg] + r * 32 * KB;
        const uint4 w0 = *reinterpret_cast<const uint4*>(wr + (w_chunk<KB>(r, 2 * sub) << 4));
        const uint4 w1 = *reinterpret_cast<const uint4*>(wr + (w_chunk<KB>(r, 2 * sub + 1) << 4));
        bw[nt][0] = w0.x; bw[nt][1] = w0.y; bw[nt][2] = w0.z; bw[nt][3] = w0.w;
        bw[nt][4] = w1.x; bw[nt][5] = w1.y; bw[nt][6] = w1.z; bw[nt][7] = w1.w;
        bs[nt] = *reinterpret_cast<const uint32_t*>(sm.s[stg] + r * 4 * KB + sub * 4);
      }
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t af[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const int r = wm0 + mt * 16 + (lane & 15);
          const int ch = sub * 8 + ks * 2 + (lane >> 4);
          ldmatrix_x4(af[mt], smem_u32(sm.a[stg] + r * SK + (a_chunk(r, ch) << 3)));
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const uint32_t s2 = e4m3_to_f16x2((bs[nt] >> (8 * ks)) & 0xff);
          const uint32_t b0 = dq_byte_bf16x2((bw[nt][2 * ks] >> (8 * t)) & 0xff, s2);
          const uint32_t b1 = dq_byte_bf16x2((bw[nt][2 * ks + 1] >> (8 * t)) & 0xff, s2);
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) mma16816(acc[mt][nt], af[mt], b0, b1);
        }
      }
    }
  }
  cp_wait<0>();

  if (EPI == kEpiF32Part) {
    const size_t slab = static_cast<size_t>(gridDim.x) * BM * ldc;
    float* P = static_cast<float*>(C) + (nkt > 0 ? kt0 / nkt : 0) * slab;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = wm0 + mt * 16 + g + 8 * h;
        if (r >= m_rows) continue;
        const size_t crow = static_cast<size_t>(c_row0 + r) * ldc;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int col = n0 + wn0 + nt * 8 + 2 * t;
          *reinterpret_cast<float2*>(P + crow + col) = make_float2(acc[mt][nt][2 * h] * gs, acc[mt][nt][2 * h + 1] * gs);
        }
      }
    return;
  }

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

  // ---- epilogue ----
  if (EPI == kEpiQkRope) {
    // Stage the bf16 head tile (BM x 128) in shared memory, then one warp per row does exactly what
    // qk_norm_rope_kv_rows_kernel does for this head (lane holds dims lane + 32j).
    constexpr int kQ = 32, kKV = 4, kD = BN;
    __syncthreads();
    uint16_t* tile = reinterpret_cast<uint16_t*>(smem_raw);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = wm0 + mt * 16 + g + 8 * h;
        if (r >= m_rows) continue;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
          *reinterpret_cast<uint32_t*>(tile + r * kD + wn0 + nt * 8 + 2 * t) =
              pack_bf16x2(acc[mt][nt][2 * h] * gs, acc[mt][nt][2 * h + 1] * gs);
      }
    __syncthreads();
    const int head = n0 / kD;
    for (int r = warp; r < m_rows; r += THREADS / 32) {
      const int row = c_row0 + r, p = *qa.pos + row;
      const int slot = qa.ring > 0 ? (p & (qa.ring - 1)) : p;
      uint16_t* dst;
      const float* w = nullptr;
      if (head < kQ) {
        dst = qa.q_out + (static_cast<size_t>(row) * kQ + head) * kD; w = qa.q_norm;
      } else if (head < kQ + kKV) {
        dst = qa.cache + static_cast<size_t>(slot) * 1024 + (head - kQ) * kD; w = qa.k_norm;
      } else {
        dst = qa.cache + static_cast<size_t>(slot) * 1024 + 512 + (head - kQ - kKV) * kD;
      }
      const uint16_t* src = tile + r * kD;
      float v[4] = {bf16_to_f32(src[lane]), bf16_to_f32(src[lane + 32]), bf16_to_f32(src[lane + 64]),
                    bf16_to_f32(src[lane + 96])};
      if (w != nullptr) {
        const float ss = warp_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
        const float rs = rsqrtf(ss / kD + qa.eps);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = round_bf16(v[j] * rs * w[lane + 32 * j]);
        if (qa.rope) {
          const float* ct = qa.cos_t + static_cast<size_t>(p) * 64;
          const float* st = qa.sin_t + static_cast<size_t>(p) * 64;
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int d = lane + 32 * j;
            const float c = ct[d], sn = st[d], n1 = v[j], n2 = v[j + 2];
            v[j] = fmaf(n1, c, -(n2 * sn));
            v[j + 2] = fmaf(n2, c, n1 * sn);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = f32_to_bf16_bits(v[j]);
    }
    return;
  }
  if (EPI == kEpiSwigluRoutedQ8) {
    // Routed SwiGLU, then int8 per 32 act columns: stage the BM x (BN/2) fp32 act tile, one thread per
    // (row, 32-block). Output K order inside each 16-block is the W4A8 kernels' permutation.
    __syncthreads();
    float* tile = reinterpret_cast<float*>(smem_raw);  // [BM][BN/2]
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = wm0 + mt * 16 + g + 8 * h;
#pragma unroll
        for (int q = 0; q < NT / 2; ++q) {
          const int col = wn0 / 2 + q * 8 + 2 * t;
          const float g0 = acc[mt][2 * q][2 * h] * gs, g1 = acc[mt][2 * q][2 * h + 1] * gs;
          const float u0 = acc[mt][2 * q + 1][2 * h] * gs, u1 = acc[mt][2 * q + 1][2 * h + 1] * gs;
          tile[r * (BN / 2) + col] = silu_mul_routed(g0, u0);
          tile[r * (BN / 2) + col + 1] = silu_mul_routed(g1, u1);
        }
      }
    __syncthreads();
    constexpr int NB = BN / 2 / 32;  // 32-blocks per row of this tile
    if (tid < m_rows * NB) {
      const int r = tid / NB, blk = tid % NB;
      const float* v = tile + r * (BN / 2) + blk * 32;
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
      const float inv = amax > 0.f ? 127.f / amax : 0.f;
      uint32_t o[8];
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int p = w * 4 + j, b16 = p >> 4, pp = p & 15;
          const int src = b16 * 16 + ((pp & 8) | ((pp & 3) << 1) | ((pp >> 2) & 1));
          const int rq = max(-127, min(127, __float2int_rn(v[src] * inv)));
          packed |= (static_cast<uint32_t>(rq) & 0xffu) << (8 * j);
        }
        o[w] = packed;
      }
      int8_t* dst = static_cast<int8_t*>(C) + static_cast<size_t>(c_row0 + r) * ldc + n0 / 2 + blk * 32;
      *reinterpret_cast<uint4*>(dst) = make_uint4(o[0], o[1], o[2], o[3]);
      *reinterpret_cast<uint4*>(dst + 16) = make_uint4(o[4], o[5], o[6], o[7]);
      qa.q8_scales[static_cast<size_t>(c_row0 + r) * (ldc / 32) + (n0 / 2) / 32 + blk] = amax / 127.f;
    }
    return;
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

template <int BM, int KB>
__device__ __forceinline__ int a_row_of(int i) {
  return (threadIdx.x + i * THREADS) / (8 * KB);
}

// grid (ceil(M / BM), N / BN, split); split > 1 is launched as (1, 1, split) clusters.
template <int BM, int ST, int KB, int EPI>
__global__ void __launch_bounds__(THREADS)
gemm_dense_kernel(const uint16_t* __restrict__ A, int M, int lda, const uint8_t* __restrict__ W,
                  const uint8_t* __restrict__ S, int K, float gs, void* __restrict__ C, int ldc, QkRopeArgs qa) {
  constexpr int AI = (BM * 8 * KB + THREADS - 1) / THREADS;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int rows = min(BM, M - m0);
  const uint16_t* arow[AI];
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int r = a_row_of<BM, KB>(i);
    arow[i] = r < rows ? A + static_cast<size_t>(m0 + r) * lda : nullptr;
  }
  const int split = gridDim.z, nkt = K / (64 * KB) / split;
  gemm_body<BM, ST, KB, EPI>(arow, rows, W, S, K, blockIdx.z * nkt, nkt, split, n0, gs, C, m0, ldc, qa);
}

// Grouped: tile i covers sorted rows [row0, row0 + rows) of expert e (rows <= BM).
// A rows are gathered: A + gather[row] * lda (gather == nullptr: row itself). grid (max_tiles, N / BN, split).
template <int BM, int ST, int KB, int EPI>
__global__ void __launch_bounds__(THREADS)
gemm_grouped_kernel(const uint16_t* __restrict__ A, const int* __restrict__ gather, int lda,
                    const int* __restrict__ tiles, const int* __restrict__ n_tiles, const uint8_t* __restrict__ W,
                    const uint8_t* __restrict__ S, int N, int K, const float* __restrict__ gs_e,
                    void* __restrict__ C, int ldc, float* __restrict__ sq) {
  constexpr int AI = (BM * 8 * KB + THREADS - 1) / THREADS;
  if (static_cast<int>(blockIdx.x) >= *n_tiles) return;  // whole clusters exit together (cluster spans z)
  const int e = tiles[3 * blockIdx.x], row0 = tiles[3 * blockIdx.x + 1], rows = tiles[3 * blockIdx.x + 2];
  const int n0 = blockIdx.y * BN;
  const uint16_t* arow[AI];
#pragma unroll
  for (int i = 0; i < AI; ++i) {
    const int r = a_row_of<BM, KB>(i);
    arow[i] = r < rows ? A + static_cast<size_t>(gather ? gather[row0 + r] : row0 + r) * lda : nullptr;
  }
  const uint8_t* We = W + static_cast<size_t>(e) * N * (K / 2);
  const uint8_t* Se = S + static_cast<size_t>(e) * N * (K / 16);
  const int split = gridDim.z, nkt = K / (64 * KB) / split;
  QkRopeArgs qa = QkRopeArgs{};
  qa.q8_scales = sq;
  gemm_body<BM, ST, KB, EPI>(arow, rows, We, Se, K, blockIdx.z * nkt, nkt, split, n0, gs_e[e], C, row0, ldc, qa);
}

// ---- many-token router: bf16 tensor-core logits + top-6 ----
// 32 tokens per CTA, all 256 experts (warp w owns experts [32w, 32w + 32)); logits are rounded to
// bf16 and selected with the decode rule (top-6, ties to the lower id, sigmoid, renormalize, bf16
// weights). Selected experts are counted into counts[256] for moe_scan_place, except for rows at positions
// >= *n_end (padding rows of a graph-shaped chunk; pos/n_end may be null: every row counts).
constexpr int RT_M = 32, RT_STAGES = 2, RT_EXPERTS = 256, RT_K = 2048, RT_TOPK = 6;

struct RouterStage {
  uint16_t a[RT_M * BK];
  uint16_t b[RT_EXPERTS * BK];
};
constexpr size_t kRouterSmem = sizeof(RouterStage) * RT_STAGES;
static_assert(kRouterSmem >= RT_M * RT_EXPERTS * sizeof(float), "logit tile must fit in the stage buffers");

// Top-6 of one token's 256 bf16-rounded logits held 8 per lane (lane l: experts 8l..8l+7), the decode rule
// (ties to the lower id, sigmoid, renormalize, bf16 weights); lane 0 stores the selections and counts them
// unless counts is null.
__device__ __forceinline__ void router_select_row(float (&v)[8], int lane, int row, int* __restrict__ ids,
                                                  float* __restrict__ tw, int* __restrict__ counts) {
  float sv[RT_TOPK];
  int si[RT_TOPK];
#pragma unroll
  for (int k = 0; k < RT_TOPK; ++k) {
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (v[j] > bv) { bv = v[j]; bi = lane * 8 + j; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    sv[k] = bv; si[k] = bi;
    if ((bi >> 3) == lane) v[bi & 7] = -FLT_MAX;
  }
  if (lane == 0) {
    float sg[RT_TOPK], sum = 0.f;
#pragma unroll
    for (int k = 0; k < RT_TOPK; ++k) { sg[k] = __fdiv_rn(1.0f, 1.0f + expf(-sv[k])); sum += sg[k]; }
    const float den = sum + 1e-20f;
#pragma unroll
    for (int k = 0; k < RT_TOPK; ++k) {
      ids[row * RT_TOPK + k] = si[k];
      tw[row * RT_TOPK + k] = round_bf16(__fdiv_rn(sg[k], den));
      if (counts != nullptr) atomicAdd(counts + si[k], 1);
    }
  }
}

__global__ void __launch_bounds__(THREADS)
router_mma_kernel(const uint16_t* __restrict__ xn, int n, const uint16_t* __restrict__ wr, int* __restrict__ ids,
                  float* __restrict__ tw, int* __restrict__ counts, const int* __restrict__ pos,
                  const int* __restrict__ n_end) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  RouterStage* st = reinterpret_cast<RouterStage*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * RT_M, nr = min(RT_M, n - r0);
  constexpr int KT = RT_K / BK;

  auto load_stage = [&](int stage, int kt) {
    RouterStage& sm = st[stage];
    {
      const int r = tid >> 3, ch = tid & 7;
      const bool ok = r < nr;
      cp_async16(smem_u32(sm.a + r * BK + ((ch ^ (r & 7)) << 3)),
                 ok ? static_cast<const void*>(xn + static_cast<size_t>(r0 + r) * RT_K + kt * BK + ch * 8)
                    : static_cast<const void*>(xn), ok);
    }
#pragma unroll
    for (int i = 0; i < (RT_EXPERTS * 8) / THREADS; ++i) {
      const int c = tid + i * THREADS, r = c >> 3, ch = c & 7;
      cp_async16(smem_u32(sm.b + r * BK + ((ch ^ (r & 7)) << 3)), wr + static_cast<size_t>(r) * RT_K + kt * BK + ch * 8,
                 true);
    }
  };

  float acc[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
#pragma unroll
  for (int s = 0; s < RT_STAGES - 1; ++s) {
    load_stage(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<RT_STAGES - 2>();
    __syncthreads();
    if (kt + RT_STAGES - 1 < KT) load_stage((kt + RT_STAGES - 1) % RT_STAGES, kt + RT_STAGES - 1);
    cp_commit();
    const RouterStage& sm = st[kt % RT_STAGES];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t af[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int r = mt * 16 + (lane & 15), ch = ks * 2 + (lane >> 4);
        ldmatrix_x4(af[mt], smem_u32(sm.a + r * BK + ((ch ^ (r & 7)) << 3)));
      }
#pragma unroll
      for (int np = 0; np < 2; ++np) {  // pairs of 8-expert n-tiles
        uint32_t b[4];
        const int r = warp * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3), ch = ks * 2 + ((lane >> 3) & 1);
        ldmatrix_x4(b, smem_u32(sm.b + r * BK + ((ch ^ (r & 7)) << 3)));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(acc[mt][2 * np], af[mt], b[0], b[1]);
          mma16816(acc[mt][2 * np + 1], af[mt], b[2], b[3]);
        }
      }
    }
  }
  cp_wait<0>();
  __syncthreads();
  float (*lg)[RT_EXPERTS] = reinterpret_cast<float (*)[RT_EXPERTS]>(smem_raw);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        lg[mt * 16 + g + 8 * (e >> 1)][warp * 32 + nt * 8 + 2 * t + (e & 1)] = round_bf16(acc[mt][nt][e]);
  __syncthreads();
  const int valid = pos != nullptr ? min(n, *n_end - *pos) : n;
  for (int rr = warp; rr < nr; rr += THREADS / 32) {
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = lg[rr][lane * 8 + j];
    router_select_row(v, lane, r0 + rr, ids, tw, r0 + rr < valid ? counts : nullptr);
  }
}

// ---- small-batch router (n <= 32): tensor-core bf16 logits + the same top-6 rule ----
// Kernel A: grid 32 CTAs (8 experts each), 8 warps = 8 K slices of 256; lane (g, t) owns expert
// blockIdx.x*8 + g and the B fragments of its slice's 16 k16 blocks (k 2t, 2t+1 and 2t+8, 2t+9 of each
// block, 4 bytes each, preloaded). The 8 slices are reduced in shared memory; logits are rounded to bf16
// (the decode rule) and stored as fp32 [n][256]. Kernel B: one warp per token row runs router_select_row.
constexpr int RS_KSPLIT = 4;                       // k quarters per expert group (grid.y)
constexpr int RS_KC = RT_K / RS_KSPLIT / 8 / 32;    // 32-k chunks per warp (2)
template <int MT>
__global__ void __launch_bounds__(256)
router_logits_small_kernel(const uint16_t* __restrict__ xn, int n, const uint16_t* __restrict__ wr,
                           float* __restrict__ part, int early) {
  // CTA (x, y): experts 8x..8x+7, k quarter y; warp w covers 64 k of that quarter. Lane (g, t) owns 8
  // contiguous k per 32-k chunk (k = 32c + 8t .. +7) in both the A and the B fragments: the two m16n8k16
  // blocks of a chunk map slots {2t, 2t+1, 2t+8, 2t+9} to k 8t+{0,1,2,3} and 8t+{4,5,6,7}, so every
  // operand load is one 16-byte vector. Partial logits (fp32) go to part[y][tok][expert].
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int e = blockIdx.x * 8 + g;
  const int k0 = blockIdx.y * (RT_K / RS_KSPLIT) + warp * (32 * RS_KC);
  const uint16_t* w = wr + static_cast<size_t>(e) * RT_K + k0 + 8 * t;
  uint4 bw[RS_KC];
#pragma unroll
  for (int c = 0; c < RS_KC; ++c) bw[c] = *reinterpret_cast<const uint4*>(w + 32 * c);
  pdl_wait();
  if (early) pdl_launch();
  float acc[MT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
  const uint16_t* xr[2 * MT];
#pragma unroll
  for (int i = 0; i < 2 * MT; ++i) {
    const int tok = g + 8 * (i & 1) + 16 * (i >> 1);
    xr[i] = tok < n ? xn + static_cast<size_t>(tok) * RT_K + k0 + 8 * t : nullptr;
  }
  uint4 av[RS_KC][2 * MT];
#pragma unroll
  for (int c = 0; c < RS_KC; ++c)
#pragma unroll
    for (int i = 0; i < 2 * MT; ++i)
      av[c][i] = xr[i] != nullptr ? *reinterpret_cast<const uint4*>(xr[i] + 32 * c) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int c = 0; c < RS_KC; ++c) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const uint32_t a0[4] = {av[c][2 * mt].x, av[c][2 * mt + 1].x, av[c][2 * mt].y, av[c][2 * mt + 1].y};
      mma16816(acc[mt], a0, bw[c].x, bw[c].y);
      const uint32_t a1[4] = {av[c][2 * mt].z, av[c][2 * mt + 1].z, av[c][2 * mt].w, av[c][2 * mt + 1].w};
      mma16816(acc[mt], a1, bw[c].z, bw[c].w);
    }
  }
  __shared__ float red[8][MT][4][32];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int c = 0; c < 4; ++c) red[warp][mt][c][lane] = acc[mt][c];
  __syncthreads();
  if (warp != 0) return;
  float* out = part + static_cast<size_t>(blockIdx.y) * n * RT_EXPERTS;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float v = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s) v += red[s][mt][c][lane];
      const int tok = g + 8 * (c >> 1) + 16 * mt;
      if (tok < n) out[static_cast<size_t>(tok) * RT_EXPERTS + blockIdx.x * 8 + 2 * t + (c & 1)] = v;
    }
}

__global__ void __launch_bounds__(256)
router_topk_small_kernel(const float* __restrict__ part, int n, int* __restrict__ ids, float* __restrict__ tw,
                         int* __restrict__ counts, int early) {
  pdl_wait();
  if (early) pdl_launch();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= n) return;
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = 0.f;
#pragma unroll
  for (int z = 0; z < RS_KSPLIT; ++z) {
    const float4* p = reinterpret_cast<const float4*>(part + (static_cast<size_t>(z) * n + row) * RT_EXPERTS + lane * 8);
    const float4 p0 = p[0], p1 = p[1];
    v[0] += p0.x; v[1] += p0.y; v[2] += p0.z; v[3] += p0.w; v[4] += p1.x; v[5] += p1.y; v[6] += p1.z; v[7] += p1.w;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = round_bf16(v[j]);
  router_select_row(v, lane, row, ids, tw, counts);
}

}  // namespace

int router_topk_small_bf16(const void* xn, int n, const void* wr, float* logits, int* ids, float* tw, int* counts,
                           cudaStream_t s) {
  // logits: fp32 scratch of RS_KSPLIT * n * 256 (k-quarter partial logits; summed in fixed order by the top-k kernel)
  if (n < 1 || n > 64) return 1;
  const int early = early_trigger_enabled() ? 1 : 0;
  const dim3 grid(RT_EXPERTS / 8, RS_KSPLIT);
  if (n <= 16)
    launch_pdl(router_logits_small_kernel<1>, grid, dim3(256), 0, s,
               static_cast<const uint16_t*>(xn), n, static_cast<const uint16_t*>(wr), logits, early);
  else if (n <= 32)
    launch_pdl(router_logits_small_kernel<2>, grid, dim3(256), 0, s,
               static_cast<const uint16_t*>(xn), n, static_cast<const uint16_t*>(wr), logits, early);
  else
    launch_pdl(router_logits_small_kernel<4>, grid, dim3(256), 0, s,
               static_cast<const uint16_t*>(xn), n, static_cast<const uint16_t*>(wr), logits, early);
  launch_pdl(router_topk_small_kernel, dim3((n + 7) / 8), dim3(256), 0, s, logits, n, ids, tw, counts, early);
  return 0;
}

namespace {

// Small batches: a (4, 1, 1) cluster per 32 tokens; rank r computes experts [64r, 64r + 64) with the same
// k order as router_mma_kernel (bit-identical logits) from a quarter of the router weights, the logit tiles
// are exchanged through DSMEM and each rank selects for 8 of the 32 tokens.
constexpr int RC = 4, RC_E = RT_EXPERTS / RC, RC_STAGES = 4;
struct RouterStageC {
  uint16_t a[RT_M * BK];
  uint16_t b[RC_E * BK];
};
constexpr size_t kRouterSmemC = sizeof(RouterStageC) * RC_STAGES;

__global__ void __cluster_dims__(RC, 1, 1) __launch_bounds__(THREADS)
router_mma_cluster_kernel(const uint16_t* __restrict__ xn, int n, const uint16_t* __restrict__ wr,
                          int* __restrict__ ids, float* __restrict__ tw, int* __restrict__ counts,
                          const int* __restrict__ pos, const int* __restrict__ n_end) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  __shared__ __align__(16) float lg[RT_M][RC_E];
  cg::cluster_group cluster = cg::this_cluster();
  RouterStageC* st = reinterpret_cast<RouterStageC*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int rank = static_cast<int>(cluster.block_rank());
  const int r0 = (blockIdx.x / RC) * RT_M, nr = min(RT_M, n - r0), e0 = rank * RC_E;
  const int wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps: 16 tokens x 16 experts each
  constexpr int KT = RT_K / BK;

  auto load_stage = [&](int stage, int kt) {
    RouterStageC& sm = st[stage];
    {
      const int r = tid >> 3, ch = tid & 7;
      const bool ok = r < nr;
      cp_async16(smem_u32(sm.a + r * BK + ((ch ^ (r & 7)) << 3)),
                 ok ? static_cast<const void*>(xn + static_cast<size_t>(r0 + r) * RT_K + kt * BK + ch * 8)
                    : static_cast<const void*>(xn), ok);
    }
#pragma unroll
    for (int i = 0; i < (RC_E * 8) / THREADS; ++i) {
      const int c = tid + i * THREADS, r = c >> 3, ch = c & 7;
      cp_async16(smem_u32(sm.b + r * BK + ((ch ^ (r & 7)) << 3)),
                 wr + static_cast<size_t>(e0 + r) * RT_K + kt * BK + ch * 8, true);
    }
  };

  float acc[2][4];
#pragma unroll
  for (int j = 0; j < 2; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
  for (int s = 0; s < RC_STAGES - 1; ++s) {
    load_stage(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<RC_STAGES - 2>();
    __syncthreads();
    if (kt + RC_STAGES - 1 < KT) load_stage((kt + RC_STAGES - 1) % RC_STAGES, kt + RC_STAGES - 1);
    cp_commit();
    const RouterStageC& sm = st[kt % RC_STAGES];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t af[4], b[4];
      {
        const int r = wm * 16 + (lane & 15), ch = ks * 2 + (lane >> 4);
        ldmatrix_x4(af, smem_u32(sm.a + r * BK + ((ch ^ (r & 7)) << 3)));
      }
      const int r = wn * 16 + (lane & 7) + ((lane >> 4) << 3), ch = ks * 2 + ((lane >> 3) & 1);
      ldmatrix_x4(b, smem_u32(sm.b + r * BK + ((ch ^ (r & 7)) << 3)));
      mma16816(acc[0], af, b[0], b[1]);
      mma16816(acc[1], af, b[2], b[3]);
    }
  }
  cp_wait<0>();
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e)
      lg[wm * 16 + g + 8 * (e >> 1)][wn * 16 + j * 8 + 2 * t + (e & 1)] = round_bf16(acc[j][e]);
  cluster.sync();
  const int rr = rank * (RT_M / RC) + warp;  // 8 tokens per rank, one warp each
  if (rr < nr) {
    const float* src = cluster.map_shared_rank(&lg[0][0], lane >> 3) + rr * RC_E + (lane & 7) * 8;
    const float4 x0 = reinterpret_cast<const float4*>(src)[0], x1 = reinterpret_cast<const float4*>(src)[1];
    float v[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const int valid = pos != nullptr ? min(n, *n_end - *pos) : n;
    router_select_row(v, lane, r0 + rr, ids, tw, r0 + rr < valid ? counts : nullptr);
  }
  cluster.sync();
}

template <typename Kernel>
void set_smem_attr(Kernel fn, size_t bytes) {
  check_cuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(fn), cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(bytes)), "cudaFuncSetAttribute");
}

template <int BM, int ST, int KB, int EPI>
void launch_dense(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs,
                  void* C, int ldc, int split, cudaStream_t s, QkRopeArgs qa = QkRopeArgs{}) {
  auto kern = gemm_dense_kernel<BM, ST, KB, EPI>;
  static bool smem_set = false;  // once per instantiation, before graph capture
  if (!smem_set) { set_smem_attr(kern, gemm_smem_bytes<BM, ST, KB>(8)); smem_set = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((M + BM - 1) / BM, N / BN, split);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = gemm_smem_bytes<BM, ST, KB>(split);
  cfg.stream = s;
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (split > 1 && EPI != kEpiF32Part) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = 1; attr[na].val.clusterDim.y = 1; attr[na].val.clusterDim.z = split;
    ++na;
  }
  // (no programmatic serialization here: gemm_body has no grid-dependency wait before its activation loads)
  cfg.attrs = attr; cfg.numAttrs = na;
  check_cuda(cudaLaunchKernelEx(&cfg, kern, static_cast<const uint16_t*>(A), M, lda, static_cast<const uint8_t*>(codes),
                     static_cast<const uint8_t*>(scales), K, gs, C, ldc, qa), "cudaLaunchKernelEx");
}

template <int BM, int ST, int KB, int EPI>
void launch_grouped(const void* A, const int* gather, int lda, const int* tiles, const int* n_tiles, int max_tiles,
                    const void* codes, const void* scales, int N, int K, const float* gs_e, void* C, int ldc,
                    int split, cudaStream_t s, float* sq = nullptr) {
  auto kern = gemm_grouped_kernel<BM, ST, KB, EPI>;
  static bool smem_set = false;
  if (!smem_set) { set_smem_attr(kern, gemm_smem_bytes<BM, ST, KB>(8)); smem_set = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(max_tiles, N / BN, split);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = gemm_smem_bytes<BM, ST, KB>(split);
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  if (split > 1) {
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = split;
    cfg.attrs = attr; cfg.numAttrs = 1;
  }
  check_cuda(cudaLaunchKernelEx(&cfg, kern, static_cast<const uint16_t*>(A), gather, lda, tiles, n_tiles,
                     static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), N, K, gs_e, C, ldc, sq), "cudaLaunchKernelEx");
}

// Default dense configuration for a shape (RTX 5090 search over the Flint matrices, M 16..8192):
// small M is weight-streaming bound -> narrow tiles, 128/256-k stages (whole sectors per weight row) and
// split-K for enough CTAs; large M -> BM 64 / 3 stages (two CTAs per SM); deep K keeps a 2-way split.
GemmConfig default_dense_config(int M, int N, int K) {
  const bool deep = K >= 4096;
  if (M <= 16) return {16, 4, deep ? 4 : (N >= 8192 ? 1 : 2), 2};
  if (M <= 64) return {16, 3, (N >= 4096 && !deep) ? 1 : 2, 4};
  if (M <= 128) return {32, 2, deep ? 2 : 1, 4};
  if (M <= 256) return {32, 2, 1, 4};
  if (M <= 1024) return {64, 3, 1, 1};
  return {64, 3, deep && M <= 4096 ? 2 : 1, 1};
}

// Grouped (MoE) configuration from the tile height and the mean routed rows per expert.
GemmConfig default_grouped_config(int tile_rows, int mean_rows) {
  if (tile_rows <= 16) return {16, 3, 1, 4};
  if (tile_rows <= 32) return {32, 2, 1, 4};
  return mean_rows <= 64 ? GemmConfig{64, 3, 1, 2} : GemmConfig{64, 2, 1, 1};
}

GemmConfig g_dense_override = {0, 0, 0, 0};
GemmConfig g_grouped_override = {0, 0, 0, 0};

}  // namespace

int gemm_w4a16_cfg(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs,
                   void* C, int ldc, int epilogue, GemmConfig c, cudaStream_t s) {
  if (c.kb != 1 && c.kb != 2 && c.kb != 4) return 1;
  if (N % BN || c.split < 1 || c.split > 8 || K % (64 * c.kb * c.split)) return 1;
#define FLINT_GEMM_CASE(BM_, ST_, KB_, EPI_)                                                                \
  if (c.bm == BM_ && c.stages == ST_ && c.kb == KB_ && epilogue == EPI_) {                                  \
    launch_dense<BM_, ST_, KB_, EPI_>(A, M, lda, codes, scales, N, K, gs, C, ldc, c.split, s);              \
    return 0;                                                                                               \
  }
#define FLINT_GEMM_EPIS(BM_, ST_, KB_)                                                    \
  FLINT_GEMM_CASE(BM_, ST_, KB_, kEpiBf16) FLINT_GEMM_CASE(BM_, ST_, KB_, kEpiF32) \
  FLINT_GEMM_CASE(BM_, ST_, KB_, kEpiSwigluDense)
  FLINT_GEMM_EPIS(16, 4, 1)
  FLINT_GEMM_EPIS(16, 4, 2)
  FLINT_GEMM_EPIS(16, 3, 4)
  FLINT_GEMM_EPIS(32, 4, 1)
  FLINT_GEMM_EPIS(32, 3, 2)
  FLINT_GEMM_EPIS(32, 2, 4)
  FLINT_GEMM_EPIS(64, 3, 1)
  FLINT_GEMM_EPIS(64, 3, 2)
  FLINT_GEMM_EPIS(128, 2, 1)
  FLINT_GEMM_EPIS(128, 3, 1)
#undef FLINT_GEMM_EPIS
#undef FLINT_GEMM_CASE
  return 1;
}

namespace {
// Split-K drain: C[row][.] = epilogue(sum_z part[z][row][.]) with the slices summed in order (deterministic).
// kEpiBf16: one thread per 8 columns; kEpiSwigluDense: one thread per 16-column gate|up block -> 8 act columns.
template <int EPI>
__global__ void __launch_bounds__(256) splitk_drain_kernel(const float* __restrict__ part, int split, size_t slab,
                                                           int M, int N, void* __restrict__ C, int ldc) {
  pdl_wait();
  constexpr int CW = EPI == kEpiSwigluDense ? 16 : 8;
  const int per_row = N / CW;
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= M * per_row) return;
  const int row = idx / per_row, c0 = (idx % per_row) * CW;
  float v[CW];
#pragma unroll
  for (int j = 0; j < CW; ++j) v[j] = 0.f;
  for (int z = 0; z < split; ++z) {
    const float4* p = reinterpret_cast<const float4*>(part + z * slab + static_cast<size_t>(row) * N + c0);
#pragma unroll
    for (int q = 0; q < CW / 4; ++q) {
      const float4 f = p[q];
      v[4 * q] += f.x; v[4 * q + 1] += f.y; v[4 * q + 2] += f.z; v[4 * q + 3] += f.w;
    }
  }
  if (EPI == kEpiSwigluDense) {
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o[j] = pack_bf16x2(silu_mul_precise(round_bf16(v[2 * j]), round_bf16(v[8 + 2 * j])),
                         silu_mul_precise(round_bf16(v[2 * j + 1]), round_bf16(v[8 + 2 * j + 1])));
    *reinterpret_cast<uint4*>(static_cast<uint16_t*>(C) + static_cast<size_t>(row) * ldc + c0 / 2) =
        make_uint4(o[0], o[1], o[2], o[3]);
  } else {
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = pack_bf16x2(v[2 * j], v[2 * j + 1]);
    *reinterpret_cast<uint4*>(static_cast<uint16_t*>(C) + static_cast<size_t>(row) * ldc + c0) =
        make_uint4(o[0], o[1], o[2], o[3]);
  }
}
}  // namespace

int gemm_w4a16_splitk(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs,
                      float* part, void* C, int ldc, int epilogue, GemmConfig c, cudaStream_t s) {
  if (c.kb != 1 && c.kb != 2 && c.kb != 4) return 1;
  if (N % BN || c.split < 1 || c.split > 8 || K % (64 * c.kb * c.split)) return 1;
  if (epilogue != kEpiBf16 && epilogue != kEpiSwigluDense) return 1;
  bool launched = false;
#define FLINT_GEMM_PART(BM_, ST_, KB_)                                                                       \
  if (c.bm == BM_ && c.stages == ST_ && c.kb == KB_) {                                                       \
    launch_dense<BM_, ST_, KB_, kEpiF32Part>(A, M, lda, codes, scales, N, K, gs, part, N, c.split, s);       \
    launched = true;                                                                                         \
  }
  FLINT_GEMM_PART(16, 4, 2)
  FLINT_GEMM_PART(16, 3, 4)
  FLINT_GEMM_PART(32, 3, 2)
  FLINT_GEMM_PART(32, 2, 4)
  FLINT_GEMM_PART(64, 3, 1)
  FLINT_GEMM_PART(64, 3, 2)
#undef FLINT_GEMM_PART
  if (!launched) return 1;
  const size_t slab = static_cast<size_t>((M + c.bm - 1) / c.bm) * c.bm * N;
  const int per_row = N / (epilogue == kEpiSwigluDense ? 16 : 8);
  const unsigned blocks = static_cast<unsigned>((static_cast<long long>(M) * per_row + 255) / 256);
  if (epilogue == kEpiSwigluDense)
    launch_pdl(splitk_drain_kernel<kEpiSwigluDense>, dim3(blocks), dim3(256), 0, s, static_cast<const float*>(part),
               c.split, slab, M, N, C, ldc);
  else
    launch_pdl(splitk_drain_kernel<kEpiBf16>, dim3(blocks), dim3(256), 0, s, static_cast<const float*>(part), c.split,
               slab, M, N, C, ldc);
  return 0;
}

int gemm_w4a16(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs,
               void* C, int ldc, int epilogue, cudaStream_t s) {
  GemmConfig c = g_dense_override.bm ? g_dense_override : default_dense_config(M, N, K);
  return gemm_w4a16_cfg(A, M, lda, codes, scales, N, K, gs, C, ldc, epilogue, c, s);
}

int gemm_w4a16_qkrope(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs,
                      const float* q_norm, const float* k_norm, const float* cos_t, const float* sin_t, const int* pos,
                      void* q_out, void* cache, int ring, int rope, float eps, cudaStream_t s) {
  const GemmConfig c = g_dense_override.bm ? g_dense_override : default_dense_config(M, N, K);
  if (N != 40 * BN || c.split < 1 || c.split > 8 || K % (64 * c.kb * c.split)) return 1;
  const QkRopeArgs qa{q_norm, k_norm, cos_t, sin_t, pos, static_cast<uint16_t*>(q_out), static_cast<uint16_t*>(cache),
                      ring, rope, eps};
#define FLINT_QK_CASE(BM_, ST_, KB_)                                                                         \
  if (c.bm == BM_ && c.stages == ST_ && c.kb == KB_) {                                                       \
    launch_dense<BM_, ST_, KB_, kEpiQkRope>(A, M, lda, codes, scales, N, K, gs, nullptr, 0, c.split, s, qa); \
    return 0;                                                                                                \
  }
  FLINT_QK_CASE(16, 4, 1) FLINT_QK_CASE(16, 4, 2) FLINT_QK_CASE(16, 3, 4) FLINT_QK_CASE(32, 4, 1)
  FLINT_QK_CASE(32, 3, 2) FLINT_QK_CASE(32, 2, 4) FLINT_QK_CASE(64, 3, 1) FLINT_QK_CASE(64, 3, 2)
  FLINT_QK_CASE(128, 2, 1) FLINT_QK_CASE(128, 3, 1)
#undef FLINT_QK_CASE
  return 1;
}

void set_gemm_override(const GemmConfig& dense, const GemmConfig& grouped) {
  g_dense_override = dense;
  g_grouped_override = grouped;
}

void router_topk_rows_bf16(const void* xn, int n, const void* wr, int* ids, float* tw, int* counts, const int* pos,
                           const int* n_end, cudaStream_t s) {
  static int sms = 0;
  if (sms == 0) {
    check_cuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(router_mma_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(kRouterSmem)), "cudaFuncSetAttribute");
    check_cuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(router_mma_cluster_kernel),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kRouterSmemC)), "cudaFuncSetAttribute");
    int dev = 0;
    check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
    check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
  }
  const int blocks = (n + RT_M - 1) / RT_M;
  if (blocks * RC <= 2 * sms) {  // few token blocks: spread the experts over a cluster
    router_mma_cluster_kernel<<<blocks * RC, THREADS, kRouterSmemC, s>>>(
        static_cast<const uint16_t*>(xn), n, static_cast<const uint16_t*>(wr), ids, tw, counts, pos, n_end);
  } else {
    router_mma_kernel<<<blocks, THREADS, kRouterSmem, s>>>(
        static_cast<const uint16_t*>(xn), n, static_cast<const uint16_t*>(wr), ids, tw, counts, pos, n_end);
  }
}

namespace {
int gemm_w4a16_grouped_impl(const void* A, const int* gather, int lda, const int* tiles, const int* n_tiles,
                            int max_tiles, const void* codes, const void* scales, int N, int K, const float* gs_e,
                            void* C, int ldc, int epilogue, int tile_rows, int mean_rows, float* sq, cudaStream_t s) {
  const GemmConfig c = g_grouped_override.bm ? g_grouped_override : default_grouped_config(tile_rows, mean_rows);
  if (c.bm < tile_rows || N % BN || c.split < 1 || c.split > 8 || K % (64 * c.kb * c.split)) return 1;
#define FLINT_GG_CASE(BM_, ST_, KB_, EPI_)                                                                  \
  if (c.bm == BM_ && c.stages == ST_ && c.kb == KB_ && epilogue == EPI_) {                                  \
    launch_grouped<BM_, ST_, KB_, EPI_>(A, gather, lda, tiles, n_tiles, max_tiles, codes, scales, N, K, gs_e, \
                                        C, ldc, c.split, s, sq);                                            \
    return 0;                                                                                               \
  }
#define FLINT_GG_EPIS(BM_, ST_, KB_) \
  FLINT_GG_CASE(BM_, ST_, KB_, kEpiBf16) FLINT_GG_CASE(BM_, ST_, KB_, kEpiSwigluRouted) \
  FLINT_GG_CASE(BM_, ST_, KB_, kEpiSwigluRoutedQ8)
  FLINT_GG_EPIS(16, 4, 1)
  FLINT_GG_EPIS(16, 4, 2)
  FLINT_GG_EPIS(16, 3, 4)
  FLINT_GG_EPIS(32, 4, 1)
  FLINT_GG_EPIS(32, 3, 2)
  FLINT_GG_EPIS(32, 2, 4)
  FLINT_GG_EPIS(64, 2, 1)
  FLINT_GG_EPIS(64, 3, 1)
  FLINT_GG_EPIS(64, 3, 2)
#undef FLINT_GG_EPIS
#undef FLINT_GG_CASE
  return 1;
}
}  // namespace

int gemm_w4a16_grouped(const void* A, const int* gather, int lda, const int* tiles, const int* n_tiles,
                       int max_tiles, const void* codes, const void* scales, int N, int K, const float* gs_e,
                       void* C, int ldc, int epilogue, int tile_rows, int mean_rows, cudaStream_t s) {
  if (epilogue == kEpiSwigluRoutedQ8) return 1;
  return gemm_w4a16_grouped_impl(A, gather, lda, tiles, n_tiles, max_tiles, codes, scales, N, K, gs_e, C, ldc,
                                 epilogue, tile_rows, mean_rows, nullptr, s);
}

int gemm_w4a16_grouped_q8(const void* A, const int* gather, int lda, const int* tiles, const int* n_tiles,
                          int max_tiles, const void* codes, const void* scales, int N, int K, const float* gs_e,
                          void* C, int ldc, float* sq, int tile_rows, int mean_rows, cudaStream_t s) {
  if (ldc % 32 || sq == nullptr) return 1;
  return gemm_w4a16_grouped_impl(A, gather, lda, tiles, n_tiles, max_tiles, codes, scales, N, K, gs_e, C, ldc,
                                 kEpiSwigluRoutedQ8, tile_rows, mean_rows, sq, s);
}

}  // namespace flint
}  // namespace flash_rt
