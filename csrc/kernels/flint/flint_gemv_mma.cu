// SPDX-License-Identifier: Apache-2.0
//
// W4A16 GEMV on tensor cores for small decode batches (2 <= M <= 32).
//   out[m, n] = epilogue( gs * sum_k x[m, k] * W[n, k] ),  x bf16 [M, K], W NVFP4 [N, K].
// Weights stream once (bandwidth-bound like the single-row GEMV); every weight
// is dequantized exactly once into a bf16 B fragment and multiplied against all
// M rows by mma.sync m16n8k16 (fp32 accumulation).
//
// Ownership: a warp owns 8 weight rows per n-tile (NT = 2 for the interleaved
// gate|up epilogue). Lane (g, t) streams row g: per 128-wide K segment it holds
// the 16-byte chunk of physical k [seg*128 + 32t, +32). Those 32 weights are
// the B fragments of the segment's eight k16 blocks under a fixed permutation:
// block b, fragment k' in {2t, 2t+1, 2t+8, 2t+9} <-> physical k = seg*128 +
// 32t + 4b + {0, 1, 2, 3}. The A fragments read the same physical positions,
// so x[m, seg*128 + 32t + 4b .. +4] (8 contiguous bytes) is one lane's A pair.

#include <cstdlib>

#include "flint_common.cuh"
#include "flint_kernels.cuh"

namespace flash_rt {
namespace flint {

namespace {

constexpr int kGroups = 2;  // n-tile groups per CTA

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// f16x2 pair of exact E2M1 values times an E4M3 scale (f16x2 broadcast) -> exact bf16x2.
__device__ __forceinline__ uint32_t scale_to_bf16x2(uint32_t h, uint32_t s2) {
  const float2 f = h2_to_f2(hmul2_u32(h, s2));
  return pack_bf16x2(f.x, f.y);
}

__device__ __forceinline__ uint2 ldg_ca_v2(const void* p) {
  uint2 r;
  asm volatile("ld.global.ca.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

template <int K, int MT>
constexpr int gemv_ks() { return (K >= 4096 || MT >= 2) ? 4 : 2; }  // k-slices per n-tile group (one warp each)

// NORM: the activation rows are bf16(s * rstd * w) with s = x + res_in (fp32), computed per CTA into shared
// memory (warp per row); block 0 stores res_out = bf16(s). Requires K == 2048 (the hidden size).
struct NormArgs {
  const uint16_t* res_in;  // nullptr: no residual add
  uint16_t* res_out;
  const float* w;
  float eps;
};
template <int K, int NT, int EPI, int MT, bool NORM>
__global__ void __launch_bounds__(kGroups * gemv_ks<K, MT>() * 32)
gemv_mma_w4a16_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                      const uint16_t* __restrict__ x, int M, void* __restrict__ out, int N, float gs, int early,
                      NormArgs na) {
  static_assert(!NORM || K == 2048, "the fused norm prologue is for the hidden size");
  extern __shared__ __align__(16) uint16_t xs_smem[];  // NORM: [16 * MT][K] normed rows
  constexpr int NSEG = K / 128;
  constexpr int KS = gemv_ks<K, MT>();
  constexpr int SEGW = NSEG / KS;       // segments per warp, all preloaded into registers
  static_assert(K % 128 == 0 && NSEG % KS == 0, "K must be a multiple of 128 * KS");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int grp = blockIdx.x * kGroups + (warp % kGroups);  // n-tile group index
  const int ks = warp / kGroups;                            // k-slice of this warp
  int rbase[NT];
  if (EPI == 2) {
    const int a0 = grp * 8;             // act rows [a0, a0 + 8): gate rows 2a0.., up rows 2a0+8..
    rbase[0] = 2 * a0; rbase[1] = 2 * a0 + 8;
    if (a0 >= N / 2) return;
  } else {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) rbase[nt] = (grp * NT + nt) * 8;
    if (rbase[0] >= N) return;
  }
  // weights of this warp's k-half: static, loaded before the grid dependency
  uint4 wq[SEGW][NT];
  uint32_t sq[SEGW][NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int row = min(rbase[nt] + g, N - 1);
    const uint8_t* wr = codes + static_cast<size_t>(row) * (K / 2) + (ks * SEGW) * 64 + t * 16;
    const uint8_t* sr = scales + static_cast<size_t>(row) * (K / 16) + (ks * SEGW) * 8 + t * 2;
#pragma unroll
    for (int p = 0; p < SEGW; ++p) {
      wq[p][nt] = ldg_nc_v4(wr + p * 64);
      sq[p][nt] = ldg_nc_u16(sr + p * 8);
    }
  }
  pdl_wait();
  if (early) pdl_launch();
  if constexpr (NORM) {
    // warp per row, two rows per batch: s = x (+ res_in) in fp32, rstd, bf16(s * rstd * w) -> xs_smem
    constexpr int WARPS = kGroups * gemv_ks<K, MT>();
    constexpr int PER = K / 32;   // 64 values per lane
    constexpr int NV = PER / 8;   // 8 uint4 per lane
    for (int r0 = warp; r0 < M; r0 += 2 * WARPS) {
      const int r1 = r0 + WARPS;
      const bool two = r1 < M;
      uint4 xv[2][NV], rv[2][NV];
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        const int c = lane + v * 32;
        xv[0][v] = reinterpret_cast<const uint4*>(x + static_cast<size_t>(r0) * K)[c];
        if (na.res_in != nullptr) rv[0][v] = reinterpret_cast<const uint4*>(na.res_in + static_cast<size_t>(r0) * K)[c];
        if (two) {
          xv[1][v] = reinterpret_cast<const uint4*>(x + static_cast<size_t>(r1) * K)[c];
          if (na.res_in != nullptr) rv[1][v] = reinterpret_cast<const uint4*>(na.res_in + static_cast<size_t>(r1) * K)[c];
        }
      }
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        if (b == 1 && !two) break;
        const int row = b == 0 ? r0 : r1;
        float sv[PER];
#pragma unroll
        for (int v = 0; v < NV; ++v) {
          const uint32_t xw[4] = {xv[b][v].x, xv[b][v].y, xv[b][v].z, xv[b][v].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) { sv[8 * v + 2 * j] = bf16lo(xw[j]); sv[8 * v + 2 * j + 1] = bf16hi(xw[j]); }
          if (na.res_in != nullptr) {
            const uint32_t rw[4] = {rv[b][v].x, rv[b][v].y, rv[b][v].z, rv[b][v].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) { sv[8 * v + 2 * j] += bf16lo(rw[j]); sv[8 * v + 2 * j + 1] += bf16hi(rw[j]); }
          }
        }
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < PER; ++j) ss = fmaf(sv[j], sv[j], ss);
        ss = warp_sum(ss);
        const float rs = rsqrtf(ss / K + na.eps);
        uint16_t* xrow = xs_smem + static_cast<size_t>(row) * K;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
          const int c = lane + v * 32;
          const float4 w0 = reinterpret_cast<const float4*>(na.w + 8 * c)[0];
          const float4 w1 = reinterpret_cast<const float4*>(na.w + 8 * c)[1];
          const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
          uint4 o, r;
          uint32_t* op = reinterpret_cast<uint32_t*>(&o);
          uint32_t* rp = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            op[j] = pack_bf16x2(sv[8 * v + 2 * j] * rs * wv[2 * j], sv[8 * v + 2 * j + 1] * rs * wv[2 * j + 1]);
            rp[j] = pack_bf16x2(sv[8 * v + 2 * j], sv[8 * v + 2 * j + 1]);
          }
          reinterpret_cast<uint4*>(xrow)[c] = o;
          if (blockIdx.x == 0 && na.res_out != nullptr) reinterpret_cast<uint4*>(na.res_out + static_cast<size_t>(row) * K)[c] = r;
        }
      }
    }
    __syncthreads();
  }
  const uint16_t* xr[2 * MT];
  bool xok[2 * MT];
#pragma unroll
  for (int i = 0; i < 2 * MT; ++i) {
    const int tok = g + 8 * (i & 1) + 16 * (i >> 1);
    xok[i] = tok < M;
    const uint16_t* base = NORM ? xs_smem : x;
    xr[i] = base + static_cast<size_t>(xok[i] ? tok : 0) * K + (ks * SEGW) * 128 + t * 32;
  }
  float acc[NT][MT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[nt][mt][0] = acc[nt][mt][1] = acc[nt][mt][2] = acc[nt][mt][3] = 0.f;

#pragma unroll
  for (int p = 0; p < SEGW; ++p) {
    // A fragments: this lane's 32 contiguous activations per token row (64 bytes = 4 x 16 B)
    uint32_t av[2 * MT][16];
#pragma unroll
    for (int i = 0; i < 2 * MT; ++i) {
      if (xok[i]) {
        const uint4* xp = reinterpret_cast<const uint4*>(xr[i] + p * 128);
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
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const uint32_t s2lo = e4m3_to_f16x2(sq[p][nt] & 0xff), s2hi = e4m3_to_f16x2(sq[p][nt] >> 8);
      const uint32_t w[4] = {wq[p][nt].x, wq[p][nt].y, wq[p][nt].z, wq[p][nt].w};
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {  // word wi holds blocks 2wi (bytes 0,1) and 2wi+1 (bytes 2,3)
        uint32_t h[4];
        e2m1x8_to_f16x2(w[wi], h);
        const uint32_t s2 = wi < 2 ? s2lo : s2hi;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int b = 2 * wi + half;  // block b: lane's activations [4b, 4b+4) = words 2b, 2b+1
          const uint32_t b0 = scale_to_bf16x2(h[2 * half], s2);      // k' 2t, 2t+1
          const uint32_t b1 = scale_to_bf16x2(h[2 * half + 1], s2);  // k' 2t+8, 2t+9
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            mma_bf16_16816(acc[nt][mt], av[2 * mt][2 * b], av[2 * mt + 1][2 * b], av[2 * mt][2 * b + 1],
                           av[2 * mt + 1][2 * b + 1], b0, b1);
        }
      }
    }
  }
  // combine the k-slices: slices 1.. hand their accumulators to slice 0 through shared memory
  __shared__ float red[KS - 1][kGroups][NT][MT][4][32];
  if (ks != 0) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int c = 0; c < 4; ++c) red[ks - 1][warp % kGroups][nt][mt][c][lane] = acc[nt][mt][c];
  }
  __syncthreads();
  if (ks != 0) return;
#pragma unroll
  for (int s = 0; s < KS - 1; ++s)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[nt][mt][c] += red[s][warp % kGroups][nt][mt][c][lane];
  // epilogue: acc[nt][mt][c]: c0 = (token g + 16mt, col 2t), c1 = (.., 2t+1), c2 = (token g+8+16mt, 2t), c3
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tok = g + 8 * h + 16 * mt;
      if (tok >= M) continue;
      if (EPI == 2) {
        const int col = grp * 8 + 2 * t;
        if (col < N / 2) {
          const float g0 = round_bf16(acc[0][mt][2 * h] * gs), u0 = round_bf16(acc[1][mt][2 * h] * gs);
          const float g1 = round_bf16(acc[0][mt][2 * h + 1] * gs), u1 = round_bf16(acc[1][mt][2 * h + 1] * gs);
          *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(out) + static_cast<size_t>(tok) * (N / 2) + col) =
              pack_bf16x2(silu_mul_precise(g0, u0), silu_mul_precise(g1, u1));
        }
      } else {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int col = rbase[nt] + 2 * t;
          if (col >= N) continue;
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

template <int K, int NT, int EPI, int MT, bool NORM>
void launch_gemv_mma(const void* codes, const void* scales, const void* x, int M, void* out, int N, float gs,
                     cudaStream_t s, NormArgs na = NormArgs{nullptr, nullptr, nullptr, 0.f}) {
  const int n_groups = EPI == 2 ? (N / 2 + 7) / 8 : (N + 8 * NT - 1) / (8 * NT);
  const int blocks = (n_groups + kGroups - 1) / kGroups;
  auto kern = gemv_mma_w4a16_kernel<K, NT, EPI, MT, NORM>;
  constexpr size_t smem = NORM ? static_cast<size_t>(16 * MT) * K * 2 : 0;
  static const int carveout = [] { const char* v = std::getenv("FLINT_CARVEOUT"); return v ? std::atoi(v) : -1; }();
  static bool attr_set = false;
  if (!attr_set) {
    if (NORM) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (carveout >= 0) cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    attr_set = true;
  }
  launch_pdl(kern, dim3(blocks), dim3(kGroups * gemv_ks<K, MT>() * 32), smem, s,
             static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales),
             static_cast<const uint16_t*>(x), M, out, N, gs, early_trigger_enabled() ? 1 : 0, na);
}

}  // namespace

int gemv_mma_w4a16(const void* codes, const void* scales, const void* x, int M, void* out, int N, int K, float gs,
                   int epilogue, cudaStream_t s) {
  if (M < 1 || M > 32 || N % 8 || (epilogue == 2 && N % 16)) return 1;
#define FLINT_GMMA_MT(KK, EPI, NTT)                                                              \
  if (K == KK && epilogue == EPI) {                                                              \
    if (M <= 16) launch_gemv_mma<KK, NTT, EPI, 1, false>(codes, scales, x, M, out, N, gs, s);    \
    else launch_gemv_mma<KK, NTT, EPI, 2, false>(codes, scales, x, M, out, N, gs, s);            \
    return 0;                                                                                    \
  }
#define FLINT_GMMA_K(KK) FLINT_GMMA_MT(KK, 0, 1) FLINT_GMMA_MT(KK, 1, 1) FLINT_GMMA_MT(KK, 2, 2)
  FLINT_GMMA_K(1024) FLINT_GMMA_K(2048) FLINT_GMMA_K(4096) FLINT_GMMA_K(6144)
#undef FLINT_GMMA_K
#undef FLINT_GMMA_MT
  return 1;
}

// Few-row GEMV with the residual add + RMSNorm fused into the activation load (K == 2048; epilogue 0 or 2):
// rows = bf16((x + res_in) * rstd * norm_w); block 0 stores res_out = bf16(x + res_in). res_in may be null.
int gemv_mma_norm_w4a16(const void* codes, const void* scales, const void* x, const void* res_in, void* res_out,
                        const float* norm_w, float eps, int M, void* out, int N, int K, float gs, int epilogue,
                        cudaStream_t s) {
  if (M < 1 || M > 32 || K != 2048 || N % 8 || (epilogue == 2 && N % 16) || (epilogue != 0 && epilogue != 2)) return 1;
  const NormArgs na{static_cast<const uint16_t*>(res_in), static_cast<uint16_t*>(res_out), norm_w, eps};
  if (epilogue == 0) {
    if (M <= 16) launch_gemv_mma<2048, 1, 0, 1, true>(codes, scales, x, M, out, N, gs, s, na);
    else launch_gemv_mma<2048, 1, 0, 2, true>(codes, scales, x, M, out, N, gs, s, na);
  } else {
    if (M <= 16) launch_gemv_mma<2048, 2, 2, 1, true>(codes, scales, x, M, out, N, gs, s, na);
    else launch_gemv_mma<2048, 2, 2, 2, true>(codes, scales, x, M, out, N, gs, s, na);
  }
  return 0;
}

}  // namespace flint
}  // namespace flash_rt
