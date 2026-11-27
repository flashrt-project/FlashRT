// SPDX-License-Identifier: Apache-2.0
//
// GEMVs over several rows on tensor cores (the drafter's block rows; with fp8 the target's decode and verification
// rows). BF16 weights: y[r][n] = sum_k W[n][k] x[r][k] for up to kMaxMmaRows input rows, FP32 accumulation, BF16 or
// FP32 out. One warp per 16 output features (the A operand of mma.m16n8k16), the input rows as B (8 per n-tile, two
// n-tiles above 8 rows). Within every 64-wide slice of k the operands are permuted so that each lane's fragments
// are 16 consecutive weights / inputs (two 16-byte loads per row and slice, no shared memory): lane (g, t) holds
// elements 16 t .. 16 t + 15 of rows g and g + 8, and MMA step s takes elements 4 s .. 4 s + 3 as the logical
// columns (2 t, 2 t + 1, 2 t + 8, 2 t + 9). The order of the products differs from the per-row GEMV (gemv_multi),
// so the results are not bit-equal to it; a row's result does not depend on the other rows.

#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kWarps = 8;
constexpr int kTile = 16;

struct GemvMmaParams {
  GemvTask task[kMaxGemvTasks];
  int tile_start[kMaxGemvTasks + 1];
  int count;
  Prefetch prefetch;
};

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// the four MMA steps of one 64-wide slice: weights wa (row g) / wb (row g + 8), inputs xa (two uint4 each)
__device__ __forceinline__ void slice(float (&c)[4], const uint4 (&wa)[2], const uint4 (&wb)[2], const uint4 (&xa)[2]) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    mma_bf16(c, wa[h].x, wb[h].x, wa[h].y, wb[h].y, xa[h].x, xa[h].y);
    mma_bf16(c, wa[h].z, wb[h].z, wa[h].w, wb[h].w, xa[h].z, xa[h].w);
  }
}

template <int NT>
__global__ void __launch_bounds__(kWarps * 32) gemv_mma_kernel(const __grid_constant__ GemvMmaParams p) {
  // one block per 16 output features; warp w takes the 64-wide slices w, w + 8, ... (two in flight), the warps'
  // partial sums are added in shared memory in warp order
  __shared__ float red[kWarps][NT * 4][32];
  prefetch_share(p.prefetch);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = blockIdx.x;
  int t = 0;
  while (t + 1 < p.count && tile >= p.tile_start[t + 1]) ++t;
  const GemvTask& T = p.task[t];
  const int n0 = (tile - p.tile_start[t]) * kTile;
  const int g = lane >> 2, q = lane & 3;
  const int ra = n0 + g, rb = n0 + g + 8;
  const int K = T.k, S = K / 64;
  const uint16_t* wa = static_cast<const uint16_t*>(T.w) + size_t(ra < T.n ? ra : T.n - 1) * K + 16 * q;
  const uint16_t* wb = static_cast<const uint16_t*>(T.w) + size_t(rb < T.n ? rb : T.n - 1) * K + 16 * q;
  const int R = T.rows < 1 ? 1 : T.rows;
  const int xs = T.x_stride ? T.x_stride : K;
  // this warp's first two slices' weights before the dependency wait (static data)
  uint4 w0a[2], w0b[2], w1a[2], w1b[2];
  int s = warp;
  if (s < S) {
    w0a[0] = ldg_stream(wa + 64 * s); w0a[1] = ldg_stream(wa + 64 * s + 8);
    w0b[0] = ldg_stream(wb + 64 * s); w0b[1] = ldg_stream(wb + 64 * s + 8);
  }
  if (s + kWarps < S) {
    const int s1 = s + kWarps;
    w1a[0] = ldg_stream(wa + 64 * s1); w1a[1] = ldg_stream(wa + 64 * s1 + 8);
    w1b[0] = ldg_stream(wb + 64 * s1); w1b[1] = ldg_stream(wb + 64 * s1 + 8);
  }
  pdl_wait();
  pdl_launch();
  const uint16_t* xbase = after_wait(static_cast<const uint16_t*>(T.x));
  const uint16_t* xp[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) {
    const int r = 8 * u + g;
    xp[u] = xbase + size_t(r < R ? r : R - 1) * xs + 16 * q;
  }
  float c[NT][4];
#pragma unroll
  for (int u = 0; u < NT; ++u) c[u][0] = c[u][1] = c[u][2] = c[u][3] = 0.0f;
  for (; s < S; s += 2 * kWarps) {
    const int s1 = s + kWarps;
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      uint4 xa[2];
      xa[0] = *reinterpret_cast<const uint4*>(xp[u] + 64 * s);
      xa[1] = *reinterpret_cast<const uint4*>(xp[u] + 64 * s + 8);
      slice(c[u], w0a, w0b, xa);
    }
    if (s1 < S) {
      const int s2 = s1 + kWarps;
      if (s2 < S) {
        w0a[0] = ldg_stream(wa + 64 * s2); w0a[1] = ldg_stream(wa + 64 * s2 + 8);
        w0b[0] = ldg_stream(wb + 64 * s2); w0b[1] = ldg_stream(wb + 64 * s2 + 8);
      }
#pragma unroll
      for (int u = 0; u < NT; ++u) {
        uint4 xa[2];
        xa[0] = *reinterpret_cast<const uint4*>(xp[u] + 64 * s1);
        xa[1] = *reinterpret_cast<const uint4*>(xp[u] + 64 * s1 + 8);
        slice(c[u], w1a, w1b, xa);
      }
      const int s3 = s2 + kWarps;
      if (s3 < S) {
        w1a[0] = ldg_stream(wa + 64 * s3); w1a[1] = ldg_stream(wa + 64 * s3 + 8);
        w1b[0] = ldg_stream(wb + 64 * s3); w1b[1] = ldg_stream(wb + 64 * s3 + 8);
      }
    }
  }
#pragma unroll
  for (int u = 0; u < NT; ++u)
#pragma unroll
    for (int e = 0; e < 4; ++e) red[warp][4 * u + e][lane] = c[u][e];
  __syncthreads();
  // thread i < NT * 4 * 32: (u, e, lane) of the fragment
  for (int i = threadIdx.x; i < NT * 4 * 32; i += kWarps * 32) {
    const int ue = i >> 5, l = i & 31;
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += red[w][ue][l];
    const int u = ue >> 2, e = ue & 3;
    const int r = 8 * u + 2 * (l & 3) + (e & 1);
    const int f = n0 + (l >> 2) + (e < 2 ? 0 : 8);
    if (r >= R || f >= T.n) continue;
    const size_t o = size_t(r) * T.n + f;
    if (T.y_f32)
      static_cast<float*>(T.y)[o] = v;
    else
      static_cast<uint16_t*>(T.y)[o] = f32_to_bf16_bits(v);
  }
}

// ------------------------------------------------------------------------------------------------- E4M3 weights
// The same products from E4M3 copies of the weights with an FP32 scale per weight row (aux): codes decoded to FP16
// (exact), the BF16 inputs converted to FP16, FP16 MMA with FP32 accumulation, the row scale applied to the sums. A
// lane holds 16 consecutive codes (one 16-byte load) per row and slice. The inputs must lie within FP16's range
// (|x| < 65504; BF16 digits below 2^-14 lose precision): normalized rows, attention outputs, SwiGLU products and the
// drafter's target features of this model stay within a few hundred.
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void e4m3x16_to_h2(uint4 q, uint32_t (&h)[8]) {
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h[2 * i]) : "h"(static_cast<uint16_t>(w[i] & 0xffff)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h[2 * i + 1]) : "h"(static_cast<uint16_t>(w[i] >> 16)));
  }
}

__device__ __forceinline__ uint32_t bf16x2_to_h2(uint32_t v) {
  const __half2 h = __floats2half2_rn(bf16lo(v), bf16hi(v));
  return *reinterpret_cast<const uint32_t*>(&h);
}

template <int NT>
__global__ void __launch_bounds__(kWarps * 32) gemv_fp8_kernel(const __grid_constant__ GemvMmaParams p) {
  __shared__ float red[kWarps][NT * 4][32];
  __shared__ float xs_f[kHidden];                 // one row with the norm prologue: the input,
  __shared__ __align__(16) uint16_t xs_n[kHidden];  // and its normalized BF16 values
  __shared__ float part[kWarps];
  prefetch_share(p.prefetch);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = blockIdx.x;
  int t = 0;
  while (t + 1 < p.count && tile >= p.tile_start[t + 1]) ++t;
  const GemvTask& T = p.task[t];
  const int n0 = (tile - p.tile_start[t]) * kTile;
  const int g = lane >> 2, q = lane & 3;
  const int ra = n0 + g, rb = n0 + g + 8;
  const int K = T.k, S = K / 64;
  const uint8_t* wa = static_cast<const uint8_t*>(T.w) + size_t(ra < T.n ? ra : T.n - 1) * K + 16 * q;
  const uint8_t* wb = static_cast<const uint8_t*>(T.w) + size_t(rb < T.n ? rb : T.n - 1) * K + 16 * q;
  const int R = T.rows < 1 ? 1 : T.rows;
  const int xs = T.x_stride ? T.x_stride : K;
  uint4 w0a, w0b, w1a, w1b;
  int s = warp;
  if (s < S) { w0a = ldg_stream(wa + 64 * s); w0b = ldg_stream(wb + 64 * s); }
  if (s + kWarps < S) { w1a = ldg_stream(wa + 64 * (s + kWarps)); w1b = ldg_stream(wb + 64 * (s + kWarps)); }
  // the norm weight of the prologue (static) before the wait: elements t + 256 m
  constexpr int kNW = kHidden / (kWarps * 32);
  float nwv[kNW];
  const bool nwpre = T.prologue == kGemvNorm && T.k == kHidden;
  if (nwpre) {
#pragma unroll
    for (int m = 0; m < kNW; ++m) nwv[m] = bf16_to_f32(static_cast<const uint16_t*>(T.aux2)[threadIdx.x + m * kWarps * 32]);
  }
  pdl_wait();
  pdl_launch();
  const uint16_t* xbase = after_wait(static_cast<const uint16_t*>(T.x));
  if (T.prologue == kGemvNorm) {
    // norm_rows' arithmetic (256 threads striding k, warp sums, the 8 warp partials in order), so the products are
    // the plain products on norm_rows' output
    const void* xin = after_wait(T.x);
    for (int i = threadIdx.x; i < K; i += kWarps * 32)
      xs_f[i] = T.x_f32 ? static_cast<const float*>(xin)[i] : bf16_to_f32(static_cast<const uint16_t*>(xin)[i]);
    __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < K; i += kWarps * 32) ss = fmaf(xs_f[i], xs_f[i], ss);
    ss = warp_sum(ss);
    if (lane == 0) part[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < kWarps; ++i) tot += part[i];
    const float inv = 1.0f / sqrtf(tot / float(K) + 1e-6f);
    const uint16_t* nw = static_cast<const uint16_t*>(T.aux2);
    if (nwpre) {
#pragma unroll
      for (int m = 0; m < kNW; ++m) {
        const int i = threadIdx.x + m * kWarps * 32;
        xs_n[i] = f32_to_bf16_bits(xs_f[i] * inv * nwv[m]);
      }
    } else {
      for (int i = threadIdx.x; i < K; i += kWarps * 32) xs_n[i] = f32_to_bf16_bits(xs_f[i] * inv * bf16_to_f32(nw[i]));
    }
    __syncthreads();
    xbase = xs_n;
  }
  const uint16_t* xp[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) {
    const int r = 8 * u + g;
    xp[u] = T.prologue == kGemvNorm ? xs_n + 16 * q : xbase + size_t(r < R ? r : R - 1) * xs + 16 * q;
  }
  float c[NT][4];
#pragma unroll
  for (int u = 0; u < NT; ++u) c[u][0] = c[u][1] = c[u][2] = c[u][3] = 0.0f;
  auto slice8 = [&](int sl, uint4 qa, uint4 qb) {
    uint32_t ha[8], hb[8];
    e4m3x16_to_h2(qa, ha);
    e4m3x16_to_h2(qb, hb);
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      const uint4 x0 = *reinterpret_cast<const uint4*>(xp[u] + 64 * sl);
      const uint4 x1 = *reinterpret_cast<const uint4*>(xp[u] + 64 * sl + 8);
      const uint32_t xh[8] = {bf16x2_to_h2(x0.x), bf16x2_to_h2(x0.y), bf16x2_to_h2(x0.z), bf16x2_to_h2(x0.w),
                              bf16x2_to_h2(x1.x), bf16x2_to_h2(x1.y), bf16x2_to_h2(x1.z), bf16x2_to_h2(x1.w)};
#pragma unroll
      for (int st = 0; st < 4; ++st)
        mma_f16(c[u], ha[2 * st], hb[2 * st], ha[2 * st + 1], hb[2 * st + 1], xh[2 * st], xh[2 * st + 1]);
    }
  };
  for (; s < S; s += 2 * kWarps) {
    const int s1 = s + kWarps;
    slice8(s, w0a, w0b);
    if (s1 < S) {
      const int s2 = s1 + kWarps;
      if (s2 < S) { w0a = ldg_stream(wa + 64 * s2); w0b = ldg_stream(wb + 64 * s2); }
      slice8(s1, w1a, w1b);
      const int s3 = s2 + kWarps;
      if (s3 < S) { w1a = ldg_stream(wa + 64 * s3); w1b = ldg_stream(wb + 64 * s3); }
    }
  }
#pragma unroll
  for (int u = 0; u < NT; ++u)
#pragma unroll
    for (int e = 0; e < 4; ++e) red[warp][4 * u + e][lane] = c[u][e];
  __syncthreads();
  const float* scale = static_cast<const float*>(T.aux);
  for (int i = threadIdx.x; i < NT * 4 * 32; i += kWarps * 32) {
    const int ue = i >> 5, l = i & 31;
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += red[w][ue][l];
    const int u = ue >> 2, e = ue & 3;
    const int r = 8 * u + 2 * (l & 3) + (e & 1);
    const int f = n0 + (l >> 2) + (e < 2 ? 0 : 8);
    if (r >= R || f >= T.n) continue;
    v *= scale[f];
    const size_t o = size_t(r) * T.n + f;
    if (T.y_f32)
      static_cast<float*>(T.y)[o] = v;
    else
      static_cast<uint16_t*>(T.y)[o] = f32_to_bf16_bits(v);
  }
}

// ------------------------------------------------------------------------------------------- E2M1 weights
// Weight-only NVFP4: E2M1 codes [n][k / 2] (element 2 i in the low nibble of byte i), an E4M3 scale per 16
// consecutive weights (aux2, [n][k / 16]) and an FP32 scale per row (aux). A lane's 16 weights per row and slice
// are one group: codes decoded to FP16 (exact) and multiplied by the group scale (exact in FP16), FP16 MMA as
// gemv_fp8, the row scale on the sums.
__device__ __forceinline__ void e2m1x16_to_h2(uint2 q, uint32_t s2, uint32_t (&h)[8]) {
  asm("{\n\t.reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %8;\n\tmov.b32 {b4, b5, b6, b7}, %9;\n\t"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n\tcvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n\tcvt.rn.f16x2.e2m1x2 %3, b3;\n\t"
      "cvt.rn.f16x2.e2m1x2 %4, b4;\n\tcvt.rn.f16x2.e2m1x2 %5, b5;\n\t"
      "cvt.rn.f16x2.e2m1x2 %6, b6;\n\tcvt.rn.f16x2.e2m1x2 %7, b7;\n\t}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3]), "=r"(h[4]), "=r"(h[5]), "=r"(h[6]), "=r"(h[7])
      : "r"(q.x), "r"(q.y));
#pragma unroll
  for (int i = 0; i < 8; ++i) asm("mul.rn.f16x2 %0, %0, %1;" : "+r"(h[i]) : "r"(s2));
}
__device__ __forceinline__ uint32_t e4m3_to_h2x(uint8_t v) {
  uint32_t h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(static_cast<uint16_t>(v | (uint16_t(v) << 8))));
  return h;
}

template <int NT>
__global__ void __launch_bounds__(kWarps * 32) gemv_fp4w_kernel(const __grid_constant__ GemvMmaParams p) {
  __shared__ float red[kWarps][NT * 4][32];
  prefetch_share(p.prefetch);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = blockIdx.x;
  int t = 0;
  while (t + 1 < p.count && tile >= p.tile_start[t + 1]) ++t;
  const GemvTask& T = p.task[t];
  const int n0 = (tile - p.tile_start[t]) * kTile;
  const int g = lane >> 2, q = lane & 3;
  const int ra = n0 + g, rb = n0 + g + 8;
  const int K = T.k, S = K / 64;
  const size_t rowa = size_t(ra < T.n ? ra : T.n - 1), rowb = size_t(rb < T.n ? rb : T.n - 1);
  const uint8_t* wa = static_cast<const uint8_t*>(T.w) + rowa * (K / 2) + 8 * q;
  const uint8_t* wb = static_cast<const uint8_t*>(T.w) + rowb * (K / 2) + 8 * q;
  const uint8_t* ga = static_cast<const uint8_t*>(T.aux2) + rowa * (K / 16) + q;
  const uint8_t* gb = static_cast<const uint8_t*>(T.aux2) + rowb * (K / 16) + q;
  const int R = T.rows < 1 ? 1 : T.rows;
  const int xs = T.x_stride ? T.x_stride : K;
  uint2 w0a, w0b, w1a, w1b;
  uint8_t s0a = 0, s0b = 0, s1a = 0, s1b = 0;
  int s = warp;
  if (s < S) { w0a = ldg_stream2(wa + 32 * s); w0b = ldg_stream2(wb + 32 * s); s0a = ga[4 * s]; s0b = gb[4 * s]; }
  if (s + kWarps < S) {
    w1a = ldg_stream2(wa + 32 * (s + kWarps)); w1b = ldg_stream2(wb + 32 * (s + kWarps));
    s1a = ga[4 * (s + kWarps)]; s1b = gb[4 * (s + kWarps)];
  }
  pdl_wait();
  pdl_launch();
  const uint16_t* xbase = after_wait(static_cast<const uint16_t*>(T.x));
  const uint16_t* xp[NT];
#pragma unroll
  for (int u = 0; u < NT; ++u) {
    const int r = 8 * u + g;
    xp[u] = xbase + size_t(r < R ? r : R - 1) * xs + 16 * q;
  }
  float c[NT][4];
#pragma unroll
  for (int u = 0; u < NT; ++u) c[u][0] = c[u][1] = c[u][2] = c[u][3] = 0.0f;
  auto slice8 = [&](int sl, uint2 qa, uint2 qb, uint8_t sa, uint8_t sb) {
    uint32_t ha[8], hb[8];
    e2m1x16_to_h2(qa, e4m3_to_h2x(sa), ha);
    e2m1x16_to_h2(qb, e4m3_to_h2x(sb), hb);
#pragma unroll
    for (int u = 0; u < NT; ++u) {
      const uint4 x0 = *reinterpret_cast<const uint4*>(xp[u] + 64 * sl);
      const uint4 x1 = *reinterpret_cast<const uint4*>(xp[u] + 64 * sl + 8);
      const uint32_t xh[8] = {bf16x2_to_h2(x0.x), bf16x2_to_h2(x0.y), bf16x2_to_h2(x0.z), bf16x2_to_h2(x0.w),
                              bf16x2_to_h2(x1.x), bf16x2_to_h2(x1.y), bf16x2_to_h2(x1.z), bf16x2_to_h2(x1.w)};
#pragma unroll
      for (int st = 0; st < 4; ++st)
        mma_f16(c[u], ha[2 * st], hb[2 * st], ha[2 * st + 1], hb[2 * st + 1], xh[2 * st], xh[2 * st + 1]);
    }
  };
  for (; s < S; s += 2 * kWarps) {
    const int s1 = s + kWarps;
    slice8(s, w0a, w0b, s0a, s0b);
    if (s1 < S) {
      const int s2 = s1 + kWarps;
      if (s2 < S) { w0a = ldg_stream2(wa + 32 * s2); w0b = ldg_stream2(wb + 32 * s2); s0a = ga[4 * s2]; s0b = gb[4 * s2]; }
      slice8(s1, w1a, w1b, s1a, s1b);
      const int s3 = s2 + kWarps;
      if (s3 < S) { w1a = ldg_stream2(wa + 32 * s3); w1b = ldg_stream2(wb + 32 * s3); s1a = ga[4 * s3]; s1b = gb[4 * s3]; }
    }
  }
#pragma unroll
  for (int u = 0; u < NT; ++u)
#pragma unroll
    for (int e = 0; e < 4; ++e) red[warp][4 * u + e][lane] = c[u][e];
  __syncthreads();
  const float* scale = static_cast<const float*>(T.aux);
  for (int i = threadIdx.x; i < NT * 4 * 32; i += kWarps * 32) {
    const int ue = i >> 5, l = i & 31;
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += red[w][ue][l];
    const int u = ue >> 2, e = ue & 3;
    const int r = 8 * u + 2 * (l & 3) + (e & 1);
    const int f = n0 + (l >> 2) + (e < 2 ? 0 : 8);
    if (r >= R || f >= T.n) continue;
    v *= scale[f];
    const size_t o = size_t(r) * T.n + f;
    if (T.y_f32)
      static_cast<float*>(T.y)[o] = v;
    else
      static_cast<uint16_t*>(T.y)[o] = f32_to_bf16_bits(v);
  }
}

// the B operand's 16-byte loads: input rows 16-byte aligned (row stride a multiple of 8), weights too
void check_mma_task(const GemvTask& task, const char* name) {
  const size_t xs = task.x_stride ? size_t(task.x_stride) : size_t(task.k);
  if (xs % 8 || reinterpret_cast<uintptr_t>(task.x) % 16 || reinterpret_cast<uintptr_t>(task.w) % 16)
    throw std::invalid_argument(std::string(name) + ": input rows and weights 16-byte aligned (row stride % 8)");
  if ((task.rows < 1 ? 1 : task.rows) > kMaxMmaRows) throw std::invalid_argument(std::string(name) + ": at most 16 rows");
}

}  // namespace

void gemv_fp4w(const GemvTask* tasks, int count, const Prefetch& prefetch, cudaStream_t stream) {
  if (count < 1 || count > kMaxGemvTasks) throw std::invalid_argument("gemv_fp4w: 1 to MAX_GEMV_TASKS tasks");
  GemvMmaParams p = {};
  p.count = count;
  int tiles = 0, rows = 1;
  for (int t = 0; t < count; ++t) {
    const GemvTask& task = tasks[t];
    if (!task.w || !task.x || !task.y || !task.aux || !task.aux2 || task.n < 1 || task.k < 64 || task.k % 64)
      throw std::invalid_argument("gemv_fp4w: task needs codes, row scales (aux), group scales (aux2), k % 64 == 0");
    if (task.x_f32 || task.prologue != kGemvPlain) throw std::invalid_argument("gemv_fp4w: BF16 inputs, no prologue");
    check_mma_task(task, "gemv_fp4w");
    const int r = task.rows < 1 ? 1 : task.rows;
    rows = r > rows ? r : rows;
    p.task[t] = task;
    p.tile_start[t] = tiles;
    tiles += (task.n + kTile - 1) / kTile;
  }
  p.tile_start[count] = tiles;
  p.prefetch = prefetch;
  if (rows <= 8)
    launch_pdl(gemv_fp4w_kernel<1>, dim3(tiles), dim3(kWarps * 32), 0, stream, p);
  else
    launch_pdl(gemv_fp4w_kernel<2>, dim3(tiles), dim3(kWarps * 32), 0, stream, p);
}

void gemv_fp8(const GemvTask* tasks, int count, const Prefetch& prefetch, cudaStream_t stream) {
  if (count < 1 || count > kMaxGemvTasks) throw std::invalid_argument("gemv_fp8: 1 to MAX_GEMV_TASKS tasks");
  GemvMmaParams p = {};
  p.count = count;
  int tiles = 0, rows = 1;
  for (int t = 0; t < count; ++t) {
    const GemvTask& task = tasks[t];
    if (!task.w || !task.x || !task.y || !task.aux || task.n < 1 || task.k < 64 || task.k % 64)
      throw std::invalid_argument("gemv_fp8: task needs codes, scales (aux), n >= 1 and k a multiple of 64");
    const int r = task.rows < 1 ? 1 : task.rows;
    if (task.prologue == kGemvNorm) {
      if (r != 1 || !task.aux2 || task.k > kHidden)
        throw std::invalid_argument("gemv_fp8: the norm prologue takes one row of at most 2304 and a norm weight");
    } else if (task.x_f32 || task.prologue != kGemvPlain) {
      throw std::invalid_argument("gemv_fp8: BF16 inputs without a prologue, or one normalized row");
    }
    if (task.prologue == kGemvPlain) check_mma_task(task, "gemv_fp8");
    rows = r > rows ? r : rows;
    p.task[t] = task;
    p.tile_start[t] = tiles;
    tiles += (task.n + kTile - 1) / kTile;
  }
  p.tile_start[count] = tiles;
  p.prefetch = prefetch;
  if (rows <= 8)
    launch_pdl(gemv_fp8_kernel<1>, dim3(tiles), dim3(kWarps * 32), 0, stream, p);
  else
    launch_pdl(gemv_fp8_kernel<2>, dim3(tiles), dim3(kWarps * 32), 0, stream, p);
}

void gemv_mma(const GemvTask* tasks, int count, const Prefetch& prefetch, cudaStream_t stream) {
  if (count < 1 || count > kMaxGemvTasks) throw std::invalid_argument("gemv_mma: 1 to MAX_GEMV_TASKS tasks");
  GemvMmaParams p = {};
  p.count = count;
  int tiles = 0, rows = 1;
  for (int t = 0; t < count; ++t) {
    const GemvTask& task = tasks[t];
    if (!task.w || !task.x || !task.y || task.n < 1 || task.k < 64 || task.k % 64)
      throw std::invalid_argument("gemv_mma: task needs pointers, n >= 1 and k a multiple of 64");
    const int r = task.rows < 1 ? 1 : task.rows;
    if (task.x_f32 || task.prologue != kGemvPlain) throw std::invalid_argument("gemv_mma: BF16 inputs, no prologue");
    check_mma_task(task, "gemv_mma");
    rows = r > rows ? r : rows;
    p.task[t] = task;
    p.tile_start[t] = tiles;
    tiles += (task.n + kTile - 1) / kTile;
  }
  p.tile_start[count] = tiles;
  p.prefetch = prefetch;
  const dim3 grid(tiles);
  if (rows <= 8)
    launch_pdl(gemv_mma_kernel<1>, grid, dim3(kWarps * 32), 0, stream, p);
  else
    launch_pdl(gemv_mma_kernel<2>, grid, dim3(kWarps * 32), 0, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
