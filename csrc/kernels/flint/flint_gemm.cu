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

namespace flash_rt {
namespace flint {

namespace {

constexpr int BN = 128, BK = 64, STAGES = 3, THREADS = 256;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, bool valid) {
  const int sz = valid ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(sz));
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

struct SmemStage {
  uint16_t a[128 * BK];        // BM <= 128 rows x 64 bf16, 16B chunks XOR-swizzled by row
  uint8_t w[BN * BK / 2];      // 128 rows x 32 code bytes
  uint8_t s[BN * BK / 16];     // 128 rows x 4 scale bytes
};

// Shared main loop. a_src(r) returns the global address of A row r (nullptr = zero row).
template <int BM, int EPI, typename ARow>
__device__ __forceinline__ void gemm_body(ARow a_row, int m_rows, const uint8_t* __restrict__ W,
                                          const uint8_t* __restrict__ S, int K, int n0, float gs,
                                          void* __restrict__ C, int c_row0, int ldc) {
  constexpr int WM = BM / 64;           // warps along M
  constexpr int WN = 8 / WM;            // warps along N
  constexpr int NT = (BN / WN) / 8;     // n-tiles per warp
  constexpr int MT = 4;                 // m-tiles per warp (64 rows)
  extern __shared__ __align__(16) uint8_t smem_raw[];
  SmemStage* st = reinterpret_cast<SmemStage*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int wm0 = (warp / WN) * 64, wn0 = (warp % WN) * (BN / WN);
  const int KT = K / BK;

  auto load_stage = [&](int stage, int kt) {
    SmemStage& sm = st[stage];
    // A: BM rows x 8 chunks
#pragma unroll
    for (int i = 0; i < (BM * 8) / THREADS; ++i) {
      const int c = tid + i * THREADS;
      const int r = c >> 3, ch = c & 7;
      const uint16_t* src = (r < m_rows) ? a_row(r) : nullptr;
      const bool ok = src != nullptr;
      cp_async16(smem_u32(sm.a + r * BK + ((ch ^ (r & 7)) << 3)),
                 ok ? static_cast<const void*>(src + kt * BK + ch * 8) : static_cast<const void*>(W), ok);
    }
    // W codes: 128 rows x 2 chunks
    {
      const int r = tid >> 1, ch = tid & 1;
      cp_async16(smem_u32(sm.w + r * 32 + ch * 16), W + static_cast<size_t>(n0 + r) * (K / 2) + kt * 32 + ch * 16, true);
    }
    if (tid < BN) cp_async4(smem_u32(sm.s + tid * 4), S + static_cast<size_t>(n0 + tid) * (K / 16) + kt * 4, true);
  };

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (kt + STAGES - 1 < KT) load_stage((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
    cp_commit();
    const SmemStage& sm = st[kt % STAGES];
    // B operands for all n-tiles of this k-tile (4 k-steps).
    uint32_t bw[NT][8];
    uint32_t bs[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int r = wn0 + nt * 8 + g;
      const uint4 w0 = *reinterpret_cast<const uint4*>(sm.w + r * 32);
      const uint4 w1 = *reinterpret_cast<const uint4*>(sm.w + r * 32 + 16);
      bw[nt][0] = w0.x; bw[nt][1] = w0.y; bw[nt][2] = w0.z; bw[nt][3] = w0.w;
      bw[nt][4] = w1.x; bw[nt][5] = w1.y; bw[nt][6] = w1.z; bw[nt][7] = w1.w;
      bs[nt] = *reinterpret_cast<const uint32_t*>(sm.s + r * 4);
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t af[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r = wm0 + mt * 16 + (lane & 15);
        const int ch = ks * 2 + (lane >> 4);
        ldmatrix_x4(af[mt], smem_u32(sm.a + r * BK + ((ch ^ (r & 7)) << 3)));
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
  cp_wait<0>();

  // ---- epilogue ----
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

template <int BM, int EPI>
__global__ void __launch_bounds__(THREADS)
gemm_dense_kernel(const uint16_t* __restrict__ A, int M, int lda, const uint8_t* __restrict__ W,
                  const uint8_t* __restrict__ S, int K, float gs, void* __restrict__ C, int ldc) {
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  auto a_row = [&](int r) -> const uint16_t* { return A + static_cast<size_t>(m0 + r) * lda; };
  gemm_body<BM, EPI>(a_row, min(BM, M - m0), W, S, K, n0, gs, C, m0, ldc);
}

// Grouped: tile i covers sorted rows [row0, row0 + rows) of expert e.
// A rows are gathered: A + gather[row] * lda (gather == nullptr: row itself).
template <int BM, int EPI>
__global__ void __launch_bounds__(THREADS)
gemm_grouped_kernel(const uint16_t* __restrict__ A, const int* __restrict__ gather, int lda,
                    const int* __restrict__ tiles, const int* __restrict__ n_tiles, const uint8_t* __restrict__ W,
                    const uint8_t* __restrict__ S, int N, int K, const float* __restrict__ gs_e,
                    void* __restrict__ C, int ldc) {
  if (static_cast<int>(blockIdx.x) >= *n_tiles) return;
  const int e = tiles[3 * blockIdx.x], row0 = tiles[3 * blockIdx.x + 1], rows = tiles[3 * blockIdx.x + 2];
  const int n0 = blockIdx.y * BN;
  const uint8_t* We = W + static_cast<size_t>(e) * N * (K / 2);
  const uint8_t* Se = S + static_cast<size_t>(e) * N * (K / 16);
  auto a_row = [&](int r) -> const uint16_t* {
    const int rr = row0 + r;
    return A + static_cast<size_t>(gather ? gather[rr] : rr) * lda;
  };
  gemm_body<BM, EPI>(a_row, rows, We, Se, K, n0, gs_e[e], C, row0, ldc);
}

constexpr size_t kSmem = sizeof(SmemStage) * STAGES;

template <int BM, int EPI>
void set_smem_attr(const void* fn) {
  static bool done = false;  // set once per instantiation, before graph capture
  if (!done) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem));
    done = true;
  }
}

}  // namespace

int gemm_w4a16(const void* A, int M, int lda, const void* codes, const void* scales, int N, int K, float gs,
               void* C, int ldc, int epilogue, cudaStream_t s) {
  if (K % BK || N % BN) return 1;
  const dim3 grid((M + 127) / 128, N / BN);
#define FLINT_GEMM_CASE(EPI)                                                                              \
  if (epilogue == EPI) {                                                                                  \
    set_smem_attr<128, EPI>(reinterpret_cast<const void*>(gemm_dense_kernel<128, EPI>));                  \
    gemm_dense_kernel<128, EPI><<<grid, THREADS, kSmem, s>>>(                                              \
        static_cast<const uint16_t*>(A), M, lda, static_cast<const uint8_t*>(codes),                     \
        static_cast<const uint8_t*>(scales), K, gs, C, ldc);                                              \
    return 0;                                                                                             \
  }
  FLINT_GEMM_CASE(kEpiBf16)
  FLINT_GEMM_CASE(kEpiF32)
  FLINT_GEMM_CASE(kEpiSwigluDense)
#undef FLINT_GEMM_CASE
  return 1;
}

int gemm_w4a16_grouped(const void* A, const int* gather, int lda, const int* tiles, const int* n_tiles,
                       int max_tiles, const void* codes, const void* scales, int N, int K, const float* gs_e,
                       void* C, int ldc, int epilogue, cudaStream_t s) {
  if (K % BK || N % BN) return 1;
  const dim3 grid(max_tiles, N / BN);
#define FLINT_GG_CASE(EPI)                                                                                \
  if (epilogue == EPI) {                                                                                  \
    set_smem_attr<64, EPI>(reinterpret_cast<const void*>(gemm_grouped_kernel<64, EPI>));                  \
    gemm_grouped_kernel<64, EPI><<<grid, THREADS, kSmem, s>>>(                                             \
        static_cast<const uint16_t*>(A), gather, lda, tiles, n_tiles, static_cast<const uint8_t*>(codes), \
        static_cast<const uint8_t*>(scales), N, K, gs_e, C, ldc);                                         \
    return 0;                                                                                             \
  }
  FLINT_GG_CASE(kEpiBf16)
  FLINT_GG_CASE(kEpiSwigluRouted)
#undef FLINT_GG_CASE
  return 1;
}

}  // namespace flint
}  // namespace flash_rt
