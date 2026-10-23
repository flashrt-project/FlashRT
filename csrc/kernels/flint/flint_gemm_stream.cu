// SPDX-License-Identifier: Apache-2.0
//
// Weight-streaming W4A16 GEMM for small decode batches (1 <= M <= 16): the few-row
// projection as a bandwidth structure. One CTA per 16-column strip streams the
// strip's NVFP4 weights once through a deep cp.async ring (4 KB stages of 512 k,
// up to 7 in flight), so every SM keeps tens of KB of weight traffic outstanding
// regardless of how small the matrix is; the activations (M x K bf16, L2-resident)
// are read per segment. Fragment mapping and numerics are those of
// gemv_mma_w4a16 (exact bf16 dequantization, fp32 accumulation, same k
// permutation), so results match it to accumulation order.

#include <cstdlib>

#include "flint_common.cuh"
#include "flint_kernels.cuh"

namespace flash_rt {
namespace flint {

namespace {

constexpr int kStrip = 16;                 // columns per CTA (kNT n-tiles of 8)
constexpr int kSegK = 128;                 // k per segment (one 16-byte code chunk per lane per column)
constexpr int kWarps = 4;                  // one warp per segment of a stage
constexpr int kNT = kStrip / 8;
constexpr int kSegsPerStage = kWarps;      // 1024 k per stage
constexpr int kStageK = kSegK * kSegsPerStage;
constexpr int kThreads = kWarps * 32;
constexpr int kCodeBytes = kStrip * kStageK / 2;   // 16 KB
constexpr int kScaleBytes = kStrip * kStageK / 16; // 2 KB
constexpr int kMaxStages = 12;             // ring cap (54 KB): strips with K <= 6144 are entirely in flight

struct StreamStage {
  uint8_t codes[kCodeBytes];    // [seg][row][t] 16-byte chunks (global order)
  uint8_t scales[kScaleBytes];  // [seg][row] 8 bytes
};

__device__ __forceinline__ void cp16(void* smem, const void* g) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp8(void* smem, const void* g) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp_commit_() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait_() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

// f16 tensor-core path: E2M1 x E4M3 products are exact in f16 (<= 4 mantissa bits), so the B fragments are the
// f16x2 products themselves; the bf16 activations are converted to f16 (exact for |x| in [6e-5, 65504)).
__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  const float lo = __uint_as_float(v << 16), hi = __uint_as_float(v & 0xffff0000u);
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// Issue stage `stage` (k range [stage * kStageK, +kStageK)) of this strip into ring slot `slot`.
__device__ __forceinline__ void issue_stage(StreamStage* ring, int slot, const uint8_t* codes, const uint8_t* scales,
                                            int K, int row0, int stage) {
  StreamStage& st = ring[slot];
  const int tid = threadIdx.x;
  // codes: kCodeBytes/16 chunks of 16 B; chunk c = (seg, row, t)
#pragma unroll
  for (int i = 0; i < kCodeBytes / 16 / kThreads; ++i) {
    const int c = tid + i * kThreads;
    const int seg = c / (kStrip * 4), row = (c >> 2) % kStrip, t = c & 3;
    cp16(st.codes + c * 16,
         codes + static_cast<size_t>(row0 + row) * (K / 2) + (stage * kSegsPerStage + seg) * 64 + t * 16);
  }
  // scales: kSegsPerStage*kStrip chunks of 8 B; chunk c = (seg, row)
  if (tid < kSegsPerStage * kStrip) {
    const int seg = tid / kStrip, row = tid % kStrip;
    cp8(st.scales + tid * 8, scales + static_cast<size_t>(row0 + row) * (K / 16) + (stage * kSegsPerStage + seg) * 8);
  }
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define FLINT_STREAM_TS(i) if (dbg != nullptr && threadIdx.x == 0) dbg[blockIdx.x * 5 + (i)] = gtimer();

// ST = ring depth in stages (K / 1024, capped at kMaxStages); NR = live activation row groups of 8 rows
// (1 for M <= 8, 2*MT otherwise).
template <int EPI, int MT, int ST, int NR>
__global__ void __launch_bounds__(kThreads, 2)
gemm_stream_w4a16_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                         const uint16_t* __restrict__ x, int M, void* __restrict__ out, int N, int K, float gs,
                         unsigned long long* __restrict__ dbg, int late_pdl) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  FLINT_STREAM_TS(0)
  StreamStage* ring = reinterpret_cast<StreamStage*>(smem_raw);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int strip = blockIdx.x, row0 = strip * kStrip;
  const int nstages = K / kStageK;

  // static weights first (before the grid dependency): exactly ST groups are committed (empty ones past the
  // strip's last stage) so the wait_group count in the loop holds for strips shorter than the ring
#pragma unroll 1
  for (int i = 0; i < ST; ++i) {
    if (i < nstages) issue_stage(ring, i, codes, scales, K, row0, i);
    cp_commit_();
  }
  pdl_wait();
  FLINT_STREAM_TS(1)
  if (!late_pdl) pdl_launch();

  const uint16_t* xr[NR];
  bool xok[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const int tok = g + 8 * (i & 1) + 16 * (i >> 1);
    xok[i] = tok < M;
    xr[i] = x + static_cast<size_t>(xok[i] ? tok : 0) * K + t * 32;
  }
  float acc[kNT][MT][4];
#pragma unroll
  for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[nt][mt][0] = acc[nt][mt][1] = acc[nt][mt][2] = acc[nt][mt][3] = 0.f;

  // Activation fragments are software-pipelined up to four stages ahead in registers (converted to f16 at
  // load), so the L2 round trips of consecutive stages overlap instead of serializing behind the MMAs.
  auto load_a = [&](int s, uint32_t (&av)[NR][16]) {
    const int kseg = s * kSegsPerStage + warp;
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      if (xok[i]) {
        const uint4* xp = reinterpret_cast<const uint4*>(xr[i] + kseg * kSegK);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 v = xp[q];
          av[i][4 * q] = v.x; av[i][4 * q + 1] = v.y; av[i][4 * q + 2] = v.z; av[i][4 * q + 3] = v.w;
        }
      } else {
#pragma unroll
        for (int q = 0; q < 16; ++q) av[i][q] = 0u;
      }
    }
  };
  auto compute = [&](int s, uint32_t (&av)[NR][16]) {
    cp_wait_<ST - 1>();
    __syncthreads();
    if (s == 0) { FLINT_STREAM_TS(2) }
    if (s == nstages - 1) { FLINT_STREAM_TS(3) }
    // refill the slot freed by stage s-1 with stage s-1+ST
    if (s >= 1 && s - 1 + ST < nstages) issue_stage(ring, (s - 1) % ST, codes, scales, K, row0, s - 1 + ST);
    cp_commit_();
    const StreamStage& st = ring[s % ST];
    const int seg = warp;
#pragma unroll
    for (int i = 0; i < NR; ++i)
#pragma unroll
      for (int q = 0; q < 16; ++q) av[i][q] = bf16x2_to_f16x2(av[i][q]);
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
      const int row = nt * 8 + g;
      const uint4 w4 = *reinterpret_cast<const uint4*>(st.codes + ((seg * kStrip + row) * 4 + t) * 16);
      const uint16_t sc = *reinterpret_cast<const uint16_t*>(st.scales + (seg * kStrip + row) * 8 + t * 2);
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
          for (int mt = 0; mt < MT; ++mt) {
            const uint32_t a0 = av[2 * mt][2 * b], a2 = av[2 * mt][2 * b + 1];
            const uint32_t a1 = NR > 2 * mt + 1 ? av[(2 * mt + 1) % NR][2 * b] : 0u;
            const uint32_t a3 = NR > 2 * mt + 1 ? av[(2 * mt + 1) % NR][2 * b + 1] : 0u;
            mma_f16(acc[nt][mt], a0, a1, a2, a3, b0, b1);
          }
        }
      }
    }
  };
  uint32_t A0[NR][16], A1[NR][16], A2[NR][16], A3[NR][16];
  load_a(0, A0);
  if (nstages > 1) load_a(1, A1);
  if (nstages > 2) load_a(2, A2);
  if (nstages > 3) load_a(3, A3);
#pragma unroll 1
  for (int s = 0; s < nstages; s += 4) {
    compute(s, A0);
    if (s + 4 < nstages) load_a(s + 4, A0);
    if (s + 1 < nstages) { compute(s + 1, A1); if (s + 5 < nstages) load_a(s + 5, A1); }
    if (s + 2 < nstages) { compute(s + 2, A2); if (s + 6 < nstages) load_a(s + 6, A2); }
    if (s + 3 < nstages) { compute(s + 3, A3); if (s + 7 < nstages) load_a(s + 7, A3); }
  }
  cp_wait_<0>();
  if (late_pdl) pdl_launch();
  FLINT_STREAM_TS(4)
  // reduce the eight segment-warps
  __shared__ float red[kWarps - 1][kNT][MT][4][32];
  if (warp != 0) {
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int c = 0; c < 4; ++c) red[warp - 1][nt][mt][c][lane] = acc[nt][mt][c];
  }
  __syncthreads();
  if (warp != 0) return;
#pragma unroll
  for (int w2 = 0; w2 < kWarps - 1; ++w2)
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[nt][mt][c] += red[w2][nt][mt][c][lane];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tok = g + 8 * h + 16 * mt;
      if (tok >= M) continue;
      if (EPI == 2) {
        // n-tiles (2q, 2q+1) = gate|up block q -> act cols strip*(kStrip/2) + q*8 + [0,8)
#pragma unroll
        for (int q = 0; q < kNT / 2; ++q) {
          const int col = strip * (kStrip / 2) + q * 8 + 2 * t;
          const float g0 = round_bf16(acc[2 * q][mt][2 * h] * gs), u0 = round_bf16(acc[2 * q + 1][mt][2 * h] * gs);
          const float g1 = round_bf16(acc[2 * q][mt][2 * h + 1] * gs), u1 = round_bf16(acc[2 * q + 1][mt][2 * h + 1] * gs);
          *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(out) + static_cast<size_t>(tok) * (N / 2) + col) =
              pack_bf16x2(silu_mul_precise(g0, u0), silu_mul_precise(g1, u1));
        }
      } else {
#pragma unroll
        for (int nt = 0; nt < kNT; ++nt) {
          const int col = row0 + nt * 8 + 2 * t;
          if (EPI == 0)
            *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(out) + static_cast<size_t>(tok) * N + col) =
                pack_bf16x2(acc[nt][mt][2 * h] * gs, acc[nt][mt][2 * h + 1] * gs);
          else
            *reinterpret_cast<float2*>(static_cast<float*>(out) + static_cast<size_t>(tok) * N + col) =
                make_float2(acc[nt][mt][2 * h] * gs, acc[nt][mt][2 * h + 1] * gs);
        }
      }
    }
}

template <int EPI, int MT, int ST, int NR>
void launch_stream_st(const void* codes, const void* scales, const void* x, int M, void* out, int N, int K, float gs,
                      cudaStream_t s, unsigned long long* dbg) {
  auto kern = gemm_stream_w4a16_kernel<EPI, MT, ST, NR>;
  static bool set = false;
  constexpr size_t smem = sizeof(StreamStage) * ST;
  if (!set) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = true; }
  static const int late_pdl = [] { const char* v = std::getenv("FLINT_STREAM_LATE_PDL"); return v && v[0] == '1'; }();
  launch_pdl(kern, dim3(N / kStrip), dim3(kThreads), smem, s, static_cast<const uint8_t*>(codes),
             static_cast<const uint8_t*>(scales), static_cast<const uint16_t*>(x), M, out, N, K, gs, dbg, late_pdl);
}
template <int EPI, int MT, int NR>
void launch_stream(const void* codes, const void* scales, const void* x, int M, void* out, int N, int K, float gs,
                   cudaStream_t s, unsigned long long* dbg = nullptr) {
  const int nst = K / kStageK;
  if (nst <= 2) launch_stream_st<EPI, MT, 2, NR>(codes, scales, x, M, out, N, K, gs, s, dbg);
  else if (nst <= 4) launch_stream_st<EPI, MT, 4, NR>(codes, scales, x, M, out, N, K, gs, s, dbg);
  else if (nst <= 8) launch_stream_st<EPI, MT, 8, NR>(codes, scales, x, M, out, N, K, gs, s, dbg);
  else launch_stream_st<EPI, MT, 12, NR>(codes, scales, x, M, out, N, K, gs, s, dbg);
}

}  // namespace

int gemm_stream_w4a16(const void* codes, const void* scales, const void* x, int M, void* out, int N, int K, float gs,
                      int epilogue, cudaStream_t s, unsigned long long* dbg) {
  if (M < 1 || M > 16 || N % kStrip || K % kStageK || epilogue < 0 || epilogue > 2) return 1;
#define FLINT_STREAM_CASE(EPI_)                                                             \
  if (epilogue == EPI_) {                                                                   \
    if (M <= 8) launch_stream<EPI_, 1, 1>(codes, scales, x, M, out, N, K, gs, s, dbg);      \
    else launch_stream<EPI_, 1, 2>(codes, scales, x, M, out, N, K, gs, s, dbg);             \
    return 0;                                                                               \
  }
  FLINT_STREAM_CASE(0) FLINT_STREAM_CASE(1) FLINT_STREAM_CASE(2)
#undef FLINT_STREAM_CASE
  return 1;
}

}  // namespace flint
}  // namespace flash_rt
