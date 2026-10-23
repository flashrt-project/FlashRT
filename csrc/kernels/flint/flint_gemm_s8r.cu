// SPDX-License-Identifier: Apache-2.0
//
// W4A8 GEMM for decode batches of 32..128 rows with the int8 activation slice
// resident in shared memory. CTA (strip group, K slice): the M x KS int8
// activations (quant_rows_s8 order, per-32 fp32 scales) are loaded once; 64-column
// NVFP4 weight strips stream through a triple-buffered cp.async ring; 8 warps = 2
// row groups x 4 n-tile pairs run int8 m16n8k16 with the exact per-16 scaling of
// gemm_w4a8. K-slice partials are bf16; a drain kernel adds them in slice order
// and applies the epilogue. The strip loop makes the grid independent of N.

#include <cstdlib>

#ifndef FLINT_S8R_UNROLL
#define FLINT_S8R_UNROLL 2
#endif

#include "flint_common.cuh"
#include "flint_kernels.cuh"

namespace flash_rt {
namespace flint {

namespace {

constexpr int kThreads = 256, kWBuf = 3, kBN = 64;

__device__ __forceinline__ uint32_t sm_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ uint32_t e2m1x4_s8x4(uint32_t sel) {  // four E2M1 codes -> int8 of twice their value
  const uint32_t idx = sel & 0x7777u;
  const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, idx);
  const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, idx);
  return __byte_perm(mag, neg, 0x3210u | ((sel >> 1) & 0x4444u));
}
__device__ __forceinline__ uint32_t wword_s8x4(uint32_t q, int parity) {
  const uint32_t x = (parity ? (q >> 4) : q) & 0x0F0F0F0Fu;
  const uint32_t y = x | (x >> 4);
  return e2m1x4_s8x4(__byte_perm(y, 0u, 0x4420u));
}
__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t pair) {
  uint32_t out;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(static_cast<uint16_t>(pair)));
  return out;
}

__device__ __forceinline__ unsigned long long gtimer_() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define FLINT_S8R_TS(i) if (dbg != nullptr && threadIdx.x == 0) dbg[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + (i)] = gtimer_();

// MTW m-tiles per warp; rows per CTA = 32 * MTW.
template <int MTW, int EPI>
__global__ void __launch_bounds__(kThreads, 1)
gemm_s8r_kernel(const int8_t* __restrict__ A, const float* __restrict__ AS, int M, int K,
                const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int N, int KS, float gs,
                uint16_t* __restrict__ part, int nstrips, unsigned long long* __restrict__ dbg) {
  constexpr int ROWS = 32 * MTW;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  FLINT_S8R_TS(0)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int wg = warp >> 2, wp = warp & 3;
  const int q = blockIdx.y, k0 = q * KS;
  const int nkb = KS / 32;             // k32 blocks per slice
  const int a_chunks = KS / 16;        // 16-byte chunks per A row
  const int w_chunks = KS / 32;        // 16-byte chunks per weight column
  uint8_t* a_s = smem_raw;                                              // ROWS x KS int8
  float* as_s = reinterpret_cast<float*>(a_s + static_cast<size_t>(ROWS) * KS);   // ROWS x nkb
  uint8_t* w_s = reinterpret_cast<uint8_t*>(as_s + ROWS * nkb);        // kWBuf x [kBN x KS/2]
  uint8_t* s_s = w_s + static_cast<size_t>(kWBuf) * kBN * (KS / 2);   // kWBuf x [kBN x KS/16]
  const int wbytes = kBN * (KS / 2), sbytes = kBN * (KS / 16);

  auto issue_strip = [&](int strip, int slot) {
    const int col0 = strip * kBN;
#pragma unroll 1
    for (int c = tid; c < kBN * w_chunks; c += kThreads) {
      const int col = c / w_chunks, ch = c % w_chunks;
      cpa16(sm_u32(w_s + slot * wbytes + col * (KS / 2) + ((ch ^ (col & 7)) << 4)),
            codes + static_cast<size_t>(col0 + col) * (K / 2) + (k0 >> 1) + ch * 16);
    }
    const int schunk = KS / 256;  // 16-byte chunks of scales per column
    if (tid < kBN * schunk) {
      const int col = tid / schunk, ch = tid % schunk;
      cpa16(sm_u32(s_s + slot * sbytes + col * (KS / 16) + ch * 16),
            scales + static_cast<size_t>(col0 + col) * (K / 16) + (k0 >> 4) + ch * 16);
    }
  };
  const int s0 = blockIdx.x, sstride = gridDim.x;
#pragma unroll 1
  for (int i = 0; i < kWBuf - 1; ++i) {
    const int strip = s0 + i * sstride;
    if (strip < nstrips) issue_strip(strip, i);
    cpa_commit();
  }
  pdl_wait();
  FLINT_S8R_TS(1)
  // activation slice (rows >= M zero-filled) and its scales, all asynchronous (one round trip)
#pragma unroll 1
  for (int c = tid; c < ROWS * a_chunks; c += kThreads) {
    const int row = c / a_chunks, ch = c % a_chunks;
    const uint32_t dst = sm_u32(a_s + static_cast<size_t>(row) * KS + ((ch ^ (row & 7)) << 4));
    if (row < M) cpa16(dst, A + static_cast<size_t>(row) * K + k0 + ch * 16);
    else *reinterpret_cast<uint4*>(a_s + static_cast<size_t>(row) * KS + ((ch ^ (row & 7)) << 4)) = make_uint4(0u, 0u, 0u, 0u);
  }
  {
    const int sch = nkb / 4;  // 16-byte chunks of scales per row
#pragma unroll 1
    for (int c = tid; c < ROWS * sch; c += kThreads) {
      const int row = c / sch, ch = c % sch;
      if (row < M) cpa16(sm_u32(as_s + row * nkb + ch * 4), AS + static_cast<size_t>(row) * (K / 32) + (k0 >> 5) + ch * 4);
      else *reinterpret_cast<float4*>(as_s + row * nkb + ch * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  cpa_commit();
  cpa_wait<0>();
  pdl_launch();
  __syncthreads();
  FLINT_S8R_TS(2)

  const int row0 = wg * MTW * 16;
  const float gsh = 0.5f * gs;  // the LUT doubles the E2M1 values
#pragma unroll 1
  for (int i = 0; ; ++i) {
    const int strip = s0 + i * sstride;
    if (strip >= nstrips) break;
    cpa_wait<kWBuf - 2>();
    __syncthreads();
    if (i == 0) { FLINT_S8R_TS(3) }
    if (i == 1) { FLINT_S8R_TS(5) }
    {  // slot (i+2)%3 == (i-1)%3 was freed by strip i-1: issue strip i+2 into it
      const int nx = s0 + (i + kWBuf - 1) * sstride;
      if (nx < nstrips) issue_strip(nx, (i + kWBuf - 1) % kWBuf);
      cpa_commit();
    }
    const uint8_t* ws = w_s + (i % kWBuf) * wbytes;
    const uint8_t* ss = s_s + (i % kWBuf) * sbytes;
    float acc[MTW][2][4];
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
#pragma unroll 2
    for (int kb = 0; kb < nkb; ++kb) {
      // weights of this lane's B-fragment columns (g of each n-tile) for the k32 block, and the k16 scales of
      // the columns this lane's accumulators hold (2t, 2t+1 of each n-tile): wsc[nt][ks] = (col 2t, col 2t+1)
      uint32_t w[2][4];
      float2 wsc[2][2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = wp * 16 + nt * 8 + g;
        const uint4 v = *reinterpret_cast<const uint4*>(ws + col * (KS / 2) + ((kb ^ (col & 7)) << 4));
        w[nt][0] = v.x; w[nt][1] = v.y; w[nt][2] = v.z; w[nt][3] = v.w;
        const int ca = wp * 16 + nt * 8 + 2 * t;
        const uint32_t sa = *reinterpret_cast<const uint16_t*>(ss + ca * (KS / 16) + kb * 2);        // col 2t: k16 0,1
        const uint32_t sb = *reinterpret_cast<const uint16_t*>(ss + (ca + 1) * (KS / 16) + kb * 2);  // col 2t+1
        wsc[nt][0] = h2_to_f2(e4m3x2_f16x2((sa & 0xffu) | ((sb & 0xffu) << 8)));   // k16 half 0: (2t, 2t+1)
        wsc[nt][1] = h2_to_f2(e4m3x2_f16x2(((sa >> 8) & 0xffu) | (sb & 0xff00u)));  // k16 half 1
      }
      // A fragments (both k16 halves) and activation scales of this warp's m-tiles
      uint32_t af[MTW][4];
      float as0[MTW], as1[MTW];
#pragma unroll
      for (int mt = 0; mt < MTW; ++mt) {
        const int row = row0 + mt * 16 + (lane & 15);
        const int ch = kb * 2 + (lane >> 4);
        ldsm4(af[mt], sm_u32(a_s + static_cast<size_t>(row) * KS + ((ch ^ (row & 7)) << 4)));
        as0[mt] = as_s[(row0 + mt * 16 + g) * nkb + kb];
        as1[mt] = as_s[(row0 + mt * 16 + g + 8) * nkb + kb];
      }
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const uint32_t qw = (t & 2) ? w[nt][2 * ks + 1] : w[nt][2 * ks];  // select, not a lane-indexed array
          const uint32_t b = wword_s8x4(qw, t & 1);
          const float wa = wsc[nt][ks].x, wb = wsc[nt][ks].y;
#pragma unroll
          for (int mt = 0; mt < MTW; ++mt) {
            int c[4] = {0, 0, 0, 0};
            mma_s8(c, af[mt][2 * ks], af[mt][2 * ks + 1], b);
            acc[mt][nt][0] = fmaf(__int2float_rn(c[0]), as0[mt] * wa, acc[mt][nt][0]);
            acc[mt][nt][1] = fmaf(__int2float_rn(c[1]), as0[mt] * wb, acc[mt][nt][1]);
            acc[mt][nt][2] = fmaf(__int2float_rn(c[2]), as1[mt] * wa, acc[mt][nt][2]);
            acc[mt][nt][3] = fmaf(__int2float_rn(c[3]), as1[mt] * wb, acc[mt][nt][3]);
          }
        }
      }
    }
    uint16_t* P = part + static_cast<size_t>(q) * M * N;
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = row0 + mt * 16 + g + 8 * hh;
        if (row < M) {
#pragma unroll
          for (int nt = 0; nt < 2; ++nt)
            *reinterpret_cast<uint32_t*>(P + static_cast<size_t>(row) * N + strip * kBN + wp * 16 + nt * 8 + 2 * t) =
                pack_bf16x2(acc[mt][nt][2 * hh] * gsh, acc[mt][nt][2 * hh + 1] * gsh);
        }
      }
    if (i == 0) { FLINT_S8R_TS(4) }
    if (i == 1) { FLINT_S8R_TS(6) }
  }
  cpa_wait<0>();
  FLINT_S8R_TS(7)
}

template <int EPI>
__global__ void __launch_bounds__(256) s8r_drain_kernel(const uint16_t* __restrict__ part, int split, int M, int N,
                                                        void* __restrict__ C, int ldc) {
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
    const uint4* p = reinterpret_cast<const uint4*>(part + (static_cast<size_t>(z) * M + row) * N + c0);
#pragma unroll
    for (int qq = 0; qq < CW / 8; ++qq) {
      const uint4 u = p[qq];
      const uint32_t ww[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[8 * qq + 2 * j] += __uint_as_float(ww[j] << 16);
        v[8 * qq + 2 * j + 1] += __uint_as_float(ww[j] & 0xffff0000u);
      }
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
  } else if (EPI == kEpiF32) {
    float* dst = static_cast<float*>(C) + static_cast<size_t>(row) * ldc + c0;
    *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(dst + 4) = make_float4(v[4], v[5], v[6], v[7]);
  } else {
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = pack_bf16x2(v[2 * j], v[2 * j + 1]);
    *reinterpret_cast<uint4*>(static_cast<uint16_t*>(C) + static_cast<size_t>(row) * ldc + c0) =
        make_uint4(o[0], o[1], o[2], o[3]);
  }
}

template <int MTW, int EPI>
void launch_s8r(const void* A, const void* AS, int M, int K, const void* codes, const void* scales, int N, int KS,
                float gs, void* part, void* C, int ldc, cudaStream_t s, unsigned long long* dbg) {
  constexpr int ROWS = 32 * MTW;
  auto kern = gemm_s8r_kernel<MTW, EPI>;
  const size_t smem = static_cast<size_t>(ROWS) * KS + static_cast<size_t>(ROWS) * (KS / 32) * 4 +
                      static_cast<size_t>(kWBuf) * kBN * (KS / 2 + KS / 16);
  static size_t smem_set = 0;
  if (smem_set < smem) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); smem_set = smem; }
  const int split = K / KS, nstrips = N / kBN;
  static const int target = [] { const char* v = std::getenv("FLINT_RESIDENT_CTAS"); return v ? std::atoi(v) : 176; }();
  int gx = (target + split - 1) / split;
  if (gx > nstrips) gx = nstrips;
  launch_pdl(kern, dim3(gx, split), dim3(kThreads), smem, s, static_cast<const int8_t*>(A),
             static_cast<const float*>(AS), M, K, static_cast<const uint8_t*>(codes),
             static_cast<const uint8_t*>(scales), N, KS, gs, static_cast<uint16_t*>(part), nstrips, dbg);
  const int per_row = N / (EPI == kEpiSwigluDense ? 16 : 8);
  const unsigned blocks = static_cast<unsigned>((static_cast<long long>(M) * per_row + 255) / 256);
  launch_pdl(s8r_drain_kernel<EPI>, dim3(blocks), dim3(256), 0, s, static_cast<const uint16_t*>(part), split, M, N, C,
             ldc);
}

}  // namespace

// (A, AS) from quant_rows_s8 (row stride K). part: bf16 scratch of (K / KS) * M * N. KS: 256..1024, divides K.
int gemm_s8_resident(const void* A, const void* AS, int M, int K, const void* codes, const void* scales, int N, int KS,
                     float gs, void* part, void* C, int ldc, int epilogue, cudaStream_t s, unsigned long long* dbg) {
  if (M < 1 || M > 128 || K % KS || KS % 256 || KS > 1024 || N % kBN || (epilogue == kEpiSwigluDense && N % 128))
    return 1;
  if (epilogue != kEpiBf16 && epilogue != kEpiF32 && epilogue != kEpiSwigluDense) return 1;
#define FLINT_S8R_CASE(MTW_, EPI_)                                                                      \
  if (epilogue == EPI_) {                                                                               \
    launch_s8r<MTW_, EPI_>(A, AS, M, K, codes, scales, N, KS, gs, part, C, ldc, s, dbg);                \
    return 0;                                                                                           \
  }
#define FLINT_S8R_EPIS(MTW_) \
  FLINT_S8R_CASE(MTW_, kEpiBf16) FLINT_S8R_CASE(MTW_, kEpiF32) FLINT_S8R_CASE(MTW_, kEpiSwigluDense)
  if (M <= 32) { FLINT_S8R_EPIS(1) }
  else if (M <= 64) { FLINT_S8R_EPIS(2) }
  else { FLINT_S8R_EPIS(4) }
#undef FLINT_S8R_EPIS
#undef FLINT_S8R_CASE
  return 1;
}

}  // namespace flint
}  // namespace flash_rt
