// SPDX-License-Identifier: Apache-2.0
//
// W4A16 GEMM for decode batches of 16..128 rows with the activation slice resident
// in shared memory: CTA (strip group, K slice) converts its M x KS activation slice
// to f16 once (k-permuted for the lane-contiguous weight fragments), then streams
// 32/64-column weight strips through a triple-buffered cp.async ring and runs f16
// tensor cores (E2M1 x E4M3 products are exact in f16). Partial sums of the K slices
// go to global fp32 and a drain kernel adds them in slice order and applies the
// epilogue. Grid size is independent of N (strip loop), so every SM is busy.

#include <cstdlib>

#include "flint_common.cuh"
#include "flint_kernels.cuh"

namespace flash_rt {
namespace flint {

namespace {

constexpr int kThreads = 256;  // 8 warps
constexpr int kWBuf = 3;       // weight strip ring depth

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit_() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait_() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  const float lo = __uint_as_float(v << 16), hi = __uint_as_float(v & 0xffff0000u);
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// A slice in smem: rows of KS f16, each 128-k segment k-permuted so that ldmatrix block b yields the
// fragments matching lane t's contiguous 32-k weight chunk (see gemv_mma): physical k = 32t + 4b + 2h + j
// sits at position b*16 + 2t + 8h + j. 16-byte chunk c of a row is stored at chunk c ^ (row & 7).
__device__ __forceinline__ int a_pos(int k) {  // k within a 128-k segment -> permuted position
  const int t = k >> 5, b = (k & 31) >> 2, h = (k & 3) >> 1, j = k & 1;
  return b * 16 + 2 * t + 8 * h + j;
}

// MTW m-tiles per warp (rows = 16 * MTW * WGM), WGM warp groups along M (1 or 2), 8 / WGM n-tiles per strip.
template <int MTW, int WGM, int EPI>
__global__ void __launch_bounds__(kThreads, 1)
gemm_resident_kernel(const uint16_t* __restrict__ x, int M, int K, const uint8_t* __restrict__ codes,
                     const uint8_t* __restrict__ scales, int N, int KS, float gs, float* __restrict__ part,
                     int nstrips) {
  constexpr int NTS = 8 / WGM;            // n-tiles per strip
  constexpr int BN = NTS * 8;             // strip width
  constexpr int ROWS = 16 * MTW * WGM;    // activation rows held
  extern __shared__ __align__(16) uint8_t smem_raw[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int wg = warp / NTS, wnt = warp % NTS;   // warp group along M, n-tile within the strip
  const int q = blockIdx.y;                       // K slice
  const int k0 = q * KS;
  const int nseg = KS / 128;
  const size_t a_row_bytes = static_cast<size_t>(KS) * 2;
  uint8_t* a_s = smem_raw;                                     // ROWS x KS f16
  uint8_t* w_s = smem_raw + static_cast<size_t>(ROWS) * a_row_bytes;  // kWBuf x [BN cols x KS/2 bytes]
  uint8_t* s_s = w_s + static_cast<size_t>(kWBuf) * BN * (KS / 2);   // kWBuf x [BN cols x KS/16 bytes]
  const int wstrip_bytes = BN * (KS / 2), sstrip_bytes = BN * (KS / 16);

  auto issue_strip = [&](int strip, int slot) {
    const int col0 = strip * BN;
    // codes: BN cols x KS/32 chunks of 16 B (chunk c of a column stored at c ^ (col & 7))
    const int nchunk = KS / 32;
#pragma unroll 1
    for (int c = tid; c < BN * nchunk; c += kThreads) {
      const int col = c / nchunk, ch = c % nchunk;
      cp16(smem_addr(w_s + slot * wstrip_bytes + col * (KS / 2) + ((ch ^ (col & 7)) << 4)),
           codes + static_cast<size_t>(col0 + col) * (K / 2) + (k0 >> 1) + ch * 16);
    }
    // scales: BN cols x KS/16 bytes (KS/256 chunks of 16 B per column)
    const int schunk = KS / 256;
    if (tid < BN * schunk) {
      const int col = tid / schunk, ch = tid % schunk;
      cp16(smem_addr(s_s + slot * sstrip_bytes + col * (KS / 16) + ch * 16),
           scales + static_cast<size_t>(col0 + col) * (K / 16) + (k0 >> 4) + ch * 16);
    }
  };

  // first strips of this CTA (static weights) before the grid dependency
  const int s0 = blockIdx.x, sstride = gridDim.x;
  int issued = 0;
#pragma unroll 1
  for (int i = 0; i < kWBuf - 1; ++i) {
    const int strip = s0 + i * sstride;
    if (strip < nstrips) { issue_strip(strip, i); ++issued; }
    cp_commit_();
  }
  pdl_wait();

  // activation slice -> f16, k-permuted, swizzled; pieces of 8 bf16 are loaded 8 at a time per thread so the
  // L2 round trips overlap
  {
    const int pieces = ROWS * (KS / 8);
#pragma unroll 1
    for (int p0 = tid; p0 < pieces; p0 += kThreads * 8) {
      uint4 u[8];
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const int p = p0 + b * kThreads;
        u[b] = make_uint4(0u, 0u, 0u, 0u);
        if (p < pieces) {
          const int row = p / (KS / 8), kk = (p % (KS / 8)) * 8;
          if (row < M) u[b] = *reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * K + k0 + kk);
        }
      }
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const int p = p0 + b * kThreads;
        if (p >= pieces) break;
        const int row = p / (KS / 8), kk = (p % (KS / 8)) * 8;
        const uint32_t v[4] = {bf16x2_to_f16x2(u[b].x), bf16x2_to_f16x2(u[b].y), bf16x2_to_f16x2(u[b].z), bf16x2_to_f16x2(u[b].w)};
        const int seg = kk / 128, ks = kk % 128;
        uint8_t* arow = a_s + static_cast<size_t>(row) * a_row_bytes;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int pos = seg * 128 + a_pos(ks + 2 * i);
          const int chunk = pos >> 3, within = (pos & 7) * 2;
          *reinterpret_cast<uint32_t*>(arow + ((chunk ^ (row & 7)) << 4) + within) = v[i];
        }
      }
    }
  }
  pdl_launch();
  __syncthreads();

  const int my_rows0 = wg * MTW * 16;  // first row of this warp group
#pragma unroll 1
  for (int i = 0; ; ++i) {
    const int strip = s0 + i * sstride;
    if (strip >= nstrips) break;
    cp_wait_<kWBuf - 2>();
    __syncthreads();
    {  // slot (i+2)%3 == (i-1)%3 was freed by strip i-1: issue strip i+2 into it
      const int nx = s0 + (i + kWBuf - 1) * sstride;
      if (nx < nstrips) issue_strip(nx, (i + kWBuf - 1) % kWBuf);
      cp_commit_();
    }
    const uint8_t* ws = w_s + (i % kWBuf) * wstrip_bytes;
    const uint8_t* ss = s_s + (i % kWBuf) * sstrip_bytes;
    float acc[MTW][4];
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
    const int col = wnt * 8 + g;  // this lane's weight column within the strip
#pragma unroll 1
    for (int seg = 0; seg < nseg; ++seg) {
      // this lane's 32-k weight chunk of the segment and its two scale bytes
      const int ch = seg * 4 + t;
      const uint4 w4 = *reinterpret_cast<const uint4*>(ws + col * (KS / 2) + ((ch ^ (col & 7)) << 4));
      const uint16_t sc = *reinterpret_cast<const uint16_t*>(ss + col * (KS / 16) + seg * 8 + t * 2);
      const uint32_t s2lo = e4m3_to_f16x2(sc & 0xff), s2hi = e4m3_to_f16x2(sc >> 8);
      const uint32_t w[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
        uint32_t h[4];
        e2m1x8_to_f16x2(w[wi], h);
        const uint32_t s2 = wi < 2 ? s2lo : s2hi;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int b = 2 * wi + half;
          const uint32_t b0 = hmul2_u32(h[2 * half], s2), b1 = hmul2_u32(h[2 * half + 1], s2);
#pragma unroll
          for (int mt = 0; mt < MTW; ++mt) {
            const int row = my_rows0 + mt * 16 + (lane & 15);
            const int chunk = seg * 16 + b * 2 + (lane >> 4);
            uint32_t a[4];
            ldsm_x4(a, smem_addr(a_s + static_cast<size_t>(row) * a_row_bytes + ((chunk ^ (row & 7)) << 4)));
            mma_f16(acc[mt], a, b0, b1);
          }
        }
      }
    }
    // partial store: part[q][row][col]
    float* P = part + static_cast<size_t>(q) * M * N;
#pragma unroll
    for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = my_rows0 + mt * 16 + g + 8 * hh;
        if (row < M)
          *reinterpret_cast<float2*>(P + static_cast<size_t>(row) * N + strip * BN + wnt * 8 + 2 * t) =
              make_float2(acc[mt][2 * hh] * gs, acc[mt][2 * hh + 1] * gs);
      }
  }
  cp_wait_<0>();
}

// drain: C = epilogue(sum over slices), slices added in order
template <int EPI>
__global__ void __launch_bounds__(256) resident_drain_kernel(const float* __restrict__ part, int split, int M, int N,
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
    const float4* p = reinterpret_cast<const float4*>(part + (static_cast<size_t>(z) * M + row) * N + c0);
#pragma unroll
    for (int qq = 0; qq < CW / 4; ++qq) {
      const float4 f = p[qq];
      v[4 * qq] += f.x; v[4 * qq + 1] += f.y; v[4 * qq + 2] += f.z; v[4 * qq + 3] += f.w;
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

template <int MTW, int WGM, int EPI>
void launch_resident(const void* x, int M, int K, const void* codes, const void* scales, int N, int KS, float gs,
                     float* part, void* C, int ldc, cudaStream_t s) {
  constexpr int NTS = 8 / WGM, BN = NTS * 8, ROWS = 16 * MTW * WGM;
  auto kern = gemm_resident_kernel<MTW, WGM, EPI>;
  const size_t smem = static_cast<size_t>(ROWS) * KS * 2 + static_cast<size_t>(kWBuf) * BN * (KS / 2 + KS / 16);
  static size_t smem_set = 0;
  if (smem_set < smem) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); smem_set = smem; }
  const int split = K / KS, nstrips = N / BN;
  static const int target = [] { const char* v = std::getenv("FLINT_RESIDENT_CTAS"); return v ? std::atoi(v) : 176; }();
  int gx = (target + split - 1) / split;
  if (gx > nstrips) gx = nstrips;
  launch_pdl(kern, dim3(gx, split), dim3(kThreads), smem, s, static_cast<const uint16_t*>(x), M, K,
             static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), N, KS, gs, part, nstrips);
  const int per_row = N / (EPI == kEpiSwigluDense ? 16 : 8);
  const unsigned blocks = static_cast<unsigned>((static_cast<long long>(M) * per_row + 255) / 256);
  launch_pdl(resident_drain_kernel<EPI>, dim3(blocks), dim3(256), 0, s, static_cast<const float*>(part), split, M, N,
             C, ldc);
}

}  // namespace

// part: fp32 scratch of (K / KS) * M * N. KS: K-slice per CTA (256/512/1024/2048, divides K, multiple of 128).
int gemm_resident_w4a16(const void* x, int M, int K, const void* codes, const void* scales, int N, int KS, float gs,
                        float* part, void* C, int ldc, int epilogue, cudaStream_t s) {
  if (M < 1 || M > 128 || K % KS || KS % 256 || KS > 2048 || N % 64 || (epilogue == kEpiSwigluDense && N % 128)) return 1;
  if (epilogue != kEpiBf16 && epilogue != kEpiF32 && epilogue != kEpiSwigluDense) return 1;
#define FLINT_RES_CASE(MTW_, WGM_, EPI_)                                                                 \
  if (epilogue == EPI_) {                                                                                \
    launch_resident<MTW_, WGM_, EPI_>(x, M, K, codes, scales, N, KS, gs, part, C, ldc, s);              \
    return 0;                                                                                            \
  }
#define FLINT_RES_EPIS(MTW_, WGM_) \
  FLINT_RES_CASE(MTW_, WGM_, kEpiBf16) FLINT_RES_CASE(MTW_, WGM_, kEpiF32) FLINT_RES_CASE(MTW_, WGM_, kEpiSwigluDense)
  if (M <= 16) { FLINT_RES_EPIS(1, 1) }
  else if (M <= 32) { FLINT_RES_EPIS(1, 2) }
  else if (M <= 64) { FLINT_RES_EPIS(2, 2) }
  else { FLINT_RES_EPIS(4, 2) }
#undef FLINT_RES_EPIS
#undef FLINT_RES_CASE
  return 1;
}

}  // namespace flint
}  // namespace flash_rt
