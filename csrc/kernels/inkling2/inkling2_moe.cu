// SPDX-License-Identifier: Apache-2.0
//
// Batch-one routed experts as the checkpoint declares them (hf_quant_config.json): the expert inputs are quantized
// per token, NVFP4 (E2M1, E4M3 scale per 16, scale chosen per block between max/6 and max/4 by the smaller summed
// absolute error, "fouroversix_mae", global amax / 1536) in layers 0-19 and MXFP8 (E4M3, UE8M0 scale per 32) in
// layers 20-23, and multiply the exact quantized weights. Block products are exact (FP16 pair sums of E2M1 x E2M1
// products stay below 2^9, FP32 for E4M3); scales and the block sums accumulate in FP32.
//   routing: logits = W_r x from gemv_multi (x = bf16(rms_norm(in) * w)), sigmoid + bias top-8, weights renormalized
//            x route_scale.
//   w13:     gate/up rows interleaved: act = bf16(softcapped_swiglu(bf16(g), bf16(u))) per routed slot.
//   w2:      the act quantized per slot, y = bf16(W2 act).
// One row (decode): expert_w13 / expert_w2, every block routing, normalizing and quantizing behind its first weight
// loads. Several rows (verification): expert_router_route (the router GEMV, the w13 inputs staged) and
// expert_flat_fused (w13 and w2 as flat passes over the groups of distinct experts in one kernel, the routing ranked
// in the pass), the same per-row arithmetic; prompt chunks without the MMA experts: expert_route, expert_w13_flat,
// expert_w2_flat.

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr float kFltMax = 3.4028234663852886e38f;
// a byte range into L2: 16-byte aligned outward, pieces of at most kPrefetchChunk
__device__ __forceinline__ void prefetch_range(const void* p, size_t bytes) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~uintptr_t(15);
  const uintptr_t e = (reinterpret_cast<uintptr_t>(p) + bytes + 15) & ~uintptr_t(15);
  for (uintptr_t o = a; o < e; o += kPrefetchChunk)
    prefetch_l2(reinterpret_cast<const void*>(o), uint32_t(e - o < kPrefetchChunk ? e - o : kPrefetchChunk));
}

__device__ __forceinline__ float e4m3_value(uint32_t byte) {
  const uint16_t pair = static_cast<uint16_t>(byte & 0xff);
  uint32_t out;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(pair));
  return __half2float(__ushort_as_half(static_cast<unsigned short>(out & 0xffff)));
}

__device__ __forceinline__ float e4m3_round(float v, uint32_t& bits) {
  uint16_t pair;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(v));
  bits = pair & 0xff;
  return e4m3_value(bits);
}

// Two values -> one byte of E2M1 codes (low nibble = lo) and their rounded values.
__device__ __forceinline__ uint32_t e2m1_pair(float lo, float hi, float& rlo, float& rhi) {
  uint32_t packed, halves;
  asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %3, %2; cvt.u32.u8 %0, b; cvt.rn.f16x2.e2m1x2 %1, b; }"
      : "=r"(packed), "=r"(halves) : "f"(lo), "f"(hi));
  rlo = __half2float(__ushort_as_half(static_cast<unsigned short>(halves & 0xffff)));
  rhi = __half2float(__ushort_as_half(static_cast<unsigned short>(halves >> 16)));
  return packed;
}

// fouroversix_mae for one 16-element block (one thread): codes (8 bytes), scale byte. encoder = 1536 / amax.
__device__ __forceinline__ void quant_block_nvfp4(const float* x, float encoder, float decoder, float outer,
                                                  uint8_t* codes, uint8_t& scale) {
  float bmax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(x[i]));
  uint32_t b6, b4;
  const float s6 = e4m3_round(fminf(__fmul_rn(bmax, __fmul_rn(encoder, float(1.0 / 6.0))), 448.0f), b6);
  const float s4 = e4m3_round(fminf(__fmul_rn(bmax, __fmul_rn(encoder, 0.25f)), 448.0f), b4);
  const float i6 = fminf(__fdiv_rn(1.0f, __fmul_rn(s6, decoder)), kFltMax);
  const float i4 = fminf(__fdiv_rn(1.0f, __fmul_rn(s4, decoder)), kFltMax);
  uint8_t c6[8], c4[8];
  float e6 = 0.0f, e4 = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float r0, r1;
    c6[i] = uint8_t(e2m1_pair(__fmul_rn(x[2 * i], i6), __fmul_rn(x[2 * i + 1], i6), r0, r1));
    e6 = __fadd_rn(e6, fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, s6), outer), x[2 * i])));
    e6 = __fadd_rn(e6, fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, s6), outer), x[2 * i + 1])));
    c4[i] = uint8_t(e2m1_pair(__fmul_rn(x[2 * i], i4), __fmul_rn(x[2 * i + 1], i4), r0, r1));
    e4 = __fadd_rn(e4, fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, s4), outer), x[2 * i])));
    e4 = __fadd_rn(e4, fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, s4), outer), x[2 * i + 1])));
  }
  const bool four = e4 < e6;
#pragma unroll
  for (int i = 0; i < 8; ++i) codes[i] = four ? c4[i] : c6[i];
  scale = uint8_t(four ? b4 : b6);
}

// MXFP8 32-block (one thread): UE8M0 scale = amax / 448 rounded up to a power of two, E4M3 codes.
__device__ __forceinline__ void quant_block_mxfp8(const float* x, uint8_t* codes, uint8_t& scale) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
  const float min_scale = __int_as_float(0x00400000);
  const float raw = __fmul_rn(amax, float(1.0 / 448.0));
  int ex = ((__float_as_int(raw) + 0x007fffff) & 0x7f800000) >> 23;
  ex = raw <= min_scale ? 0 : min(ex, 254);
  const float sc = fmaxf(__int_as_float(ex << 23), min_scale);
  const float inv = ex == 0 ? __int_as_float(0x7f000000) : __frcp_rn(sc);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float lo = fminf(fmaxf(__fmul_rn(x[2 * i], inv), -448.0f), 448.0f);
    const float hi = fminf(fmaxf(__fmul_rn(x[2 * i + 1], inv), -448.0f), 448.0f);
    uint16_t pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(hi), "f"(lo));
    codes[2 * i] = uint8_t(pair & 0xff);
    codes[2 * i + 1] = uint8_t(pair >> 8);
  }
  scale = uint8_t(ex);
}

__device__ __forceinline__ void e2m1x8_to_h2(uint2 codes, uint32_t (&out)[8]) {
  asm("{ .reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
      "mov.b32 {b0, b1, b2, b3}, %8;\n mov.b32 {b4, b5, b6, b7}, %9;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n cvt.rn.f16x2.e2m1x2 %4, b4;\n cvt.rn.f16x2.e2m1x2 %5, b5;\n"
      "cvt.rn.f16x2.e2m1x2 %6, b6;\n cvt.rn.f16x2.e2m1x2 %7, b7;\n }"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
      : "r"(codes.x), "r"(codes.y));
}

// Exact dot of one 16-element NVFP4 block (weights decoded to f16x2, wh[8]) with the quantized input codes (xh[8]).
__device__ __forceinline__ float block_dot_h2(const uint32_t (&wh)[8], const uint32_t* xh) {
  __half2 acc = __float2half2_rn(0.0f);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    acc = __hfma2(*reinterpret_cast<const __half2*>(&wh[i]), *reinterpret_cast<const __half2*>(&xh[i]), acc);
  const float2 f = __half22float2(acc);
  return f.x + f.y;
}

// 32 FP8 products of one MXFP8 block: weight codes (32 bytes, 2 uint4) x input values (FP32 in shared memory).
__device__ __forceinline__ float block_dot_mxfp8(uint4 w0, uint4 w1, const __half* xh16) {
  float x[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) x[i] = __half2float(xh16[i]);
  const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t lo, hi;
    const uint16_t a = static_cast<uint16_t>(ww[i] & 0xffff), b = static_cast<uint16_t>(ww[i] >> 16);
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo) : "h"(a));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi) : "h"(b));
    const float2 l = __half22float2(*reinterpret_cast<__half2*>(&lo));
    const float2 h = __half22float2(*reinterpret_cast<__half2*>(&hi));
    acc = fmaf(l.x, x[4 * i], acc);
    acc = fmaf(l.y, x[4 * i + 1], acc);
    acc = fmaf(h.x, x[4 * i + 2], acc);
    acc = fmaf(h.y, x[4 * i + 3], acc);
  }
  return acc;
}


// Stage one quantized input row: NVFP4 codes as f16x2 pairs + FP32 block scales; MXFP8 values as half (exact).
template <int Threads>
__device__ void stage_row(int k, int mx, const uint8_t* codes, const uint8_t* scales, uint32_t* xh, float* xsc,
                          __half* xv) {
  if (mx) {
    for (int i = threadIdx.x; i < k; i += Threads) xv[i] = __float2half(e4m3_value(codes[i]));
    for (int b = threadIdx.x; b < k / 32; b += Threads) xsc[b] = exp2f(float(int(scales[b]) - 127));
  } else {
    for (int i = threadIdx.x; i < k / 2; i += Threads) {
      uint32_t h;
      const uint16_t byte = codes[i];
      asm("{ .reg .b8 b; cvt.u8.u16 b, %1; cvt.rn.f16x2.e2m1x2 %0, b; }" : "=r"(h) : "h"(byte));
      xh[i] = h;
    }
    for (int b = threadIdx.x; b < k / 16; b += Threads) xsc[b] = e4m3_value(scales[b]);
  }
}

// One weight row's share for lane l of an L-lane group: blocks l, l + L, ... (NB blocks per row). NVFP4 blocks are
// 16 inputs (8 code bytes, one E4M3 scale), MXFP8 blocks 32 inputs (32 code bytes, one UE8M0 scale). load issues
// the whole share at once; dot multiplies it with n staged input rows (per-row accumulation in the one-row order).
template <int NB, int L, bool MX>
struct RowShare {
  static constexpr int kPer = (NB + L - 1) / L;
  uint4 c[MX ? 2 * kPer : (kPer + 1) / 2];     // NVFP4: two blocks per uint4 (low / high half)
  uint32_t s[kPer];
  __device__ __forceinline__ void load(const uint8_t* codes, const uint8_t* scales, int l) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int b = l + i * L;
      const bool ok = (NB % L == 0) || b < NB;
      if (MX) {
        c[2 * i] = ok ? ldg_stream(codes + 32 * b) : uint4{};
        c[2 * i + 1] = ok ? ldg_stream(codes + 32 * b + 16) : uint4{};
      } else {
        uint2 v = uint2{};
        if (ok) v = *reinterpret_cast<const uint2*>(codes + 8 * b);
        if (i & 1) { c[i / 2].z = v.x; c[i / 2].w = v.y; }
        else { c[i / 2].x = v.x; c[i / 2].y = v.y; }
      }
      s[i] = ok ? uint32_t(scales[b]) : 0u;
    }
  }
  // rows j < n: inputs xh + row(j) * xh_stride (NVFP4) / xv + row(j) * xv_stride (MXFP8), scales
  // xsc + row(j) * sc_stride (row(j) = j unless a map of the staged rows is given)
  __device__ __forceinline__ void dot(int l, int n, const uint32_t* xh, int xh_stride, const float* xsc, int sc_stride,
                                      const __half* xv, int xv_stride, float (&acc)[kMaxRows]) const {
    const int ident[kMaxRows] = {0, 1, 2, 3, 4, 5, 6, 7};
    dot(l, n, ident, xh, xh_stride, xsc, sc_stride, xv, xv_stride, acc);
  }
  __device__ __forceinline__ void dot(int l, int n, const int (&row)[kMaxRows], const uint32_t* xh, int xh_stride,
                                      const float* xsc, int sc_stride, const __half* xv, int xv_stride,
                                      float (&acc)[kMaxRows]) const {
#pragma unroll
    for (int j = 0; j < kMaxRows; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int b = l + i * L;
      if ((NB % L) && b >= NB) break;
      if (MX) {
        const float sw = exp2f(float(int(s[i]) - 127));
#pragma unroll
        for (int j = 0; j < kMaxRows; ++j) {
          if (j >= n) break;
          acc[j] = fmaf(block_dot_mxfp8(c[2 * i], c[2 * i + 1], xv + row[j] * xv_stride + 32 * b),
                        __fmul_rn(sw, xsc[row[j] * sc_stride + b]), acc[j]);
        }
      } else {
        const uint2 wc = (i & 1) ? make_uint2(c[i / 2].z, c[i / 2].w) : make_uint2(c[i / 2].x, c[i / 2].y);
        uint32_t wh[8];
        e2m1x8_to_h2(wc, wh);
        const float sw = e4m3_value(s[i]);
#pragma unroll
        for (int j = 0; j < kMaxRows; ++j) {
          if (j >= n) break;
          acc[j] = fmaf(block_dot_h2(wh, xh + row[j] * xh_stride + 8 * b), __fmul_rn(sw, xsc[row[j] * sc_stride + b]),
                        acc[j]);
        }
      }
    }
  }
};

// One block per SM: grid (kSplit, groups), 512 threads; a group = one routed expert with its rows (slots).
constexpr int kSplit = 21;
constexpr int kExpertThreads = 512;
constexpr int kExpertWarps = kExpertThreads / 32;
constexpr int kMaxSlots = kMaxRows * kTopK;

// The expert groups of a step: slots s = row * 8 + k in order; a group per distinct expert, in order of first
// appearance (one row: the 8 slots in rank order). Every block derives the same groups.
struct Groups {
  int count;
  int expert;                  // this block's group
  int n;                       // its slots
  int slot[kMaxRows];
};

__device__ __forceinline__ void make_groups(const int* ids, int slots, int g, Groups& G) {
  __shared__ int leader[kMaxSlots];
  const int t = threadIdx.x;
  if (t < slots) {
    int f = t;
    for (int s = 0; s < t; ++s)
      if (ids[s] == ids[t]) { f = s; break; }
    leader[t] = f == t;
  }
  __syncthreads();
  if (t == 0) {
    int count = 0, n = 0, expert = -1;
    for (int s = 0; s < slots; ++s) {
      if (leader[s]) {
        if (count == g) expert = ids[s];
        ++count;
      }
      if (expert >= 0 && ids[s] == expert && n < kMaxRows) G.slot[n++] = s;
    }
    G.count = count;
    G.expert = expert;
    G.n = n;
  }
  __syncthreads();
}

// Routing of `rows` rows (logits [rows][128]): top-8 by rank of sigmoid(logit) + bias (first index on ties), weights
// renormalized x route_scale in rank order (block (0, 0) writes them).
__device__ void route_rows(const ExpertParams& p, int rows, int* ids, float* sc) {
  __shared__ float bs[kMaxRows][kExperts];
  const float* lg = after_wait(p.logits);
  const int t = threadIdx.x;
  for (int i = t; i < rows * kExperts; i += kExpertThreads) {
    const int r = i / kExperts, e = i % kExperts;
    bs[r][e] = 1.0f / (1.0f + expf(-lg[i])) + p.bias[e];
  }
  __syncthreads();
  for (int i = t; i < rows * kExperts; i += kExpertThreads) {
    const int r = i / kExperts, e = i % kExperts;
    const float b = bs[r][e];
    int rank = 0;
#pragma unroll 8
    for (int j = 0; j < kExperts; ++j) {
      const float bj = bs[r][j];
      rank += (bj > b) || (bj == b && j < e);
    }
    if (rank < kTopK) { ids[r * kTopK + rank] = e; sc[r * kTopK + rank] = 1.0f / (1.0f + expf(-lg[i])); }
  }
  __syncthreads();
  if (p.ids_out && blockIdx.x == 0 && blockIdx.y == 0 && t < rows) {
    float wsum = 0.0f;
    for (int k = 0; k < kTopK; ++k) wsum = __fadd_rn(wsum, sc[t * kTopK + k]);
    int* oid = after_wait(p.ids_out) + t * kTopK;
    float* ow = after_wait(p.weights_out) + t * kTopK;
    for (int k = 0; k < kTopK; ++k) {
      oid[k] = ids[t * kTopK + k];
      ow[k] = __fmul_rn(__fdiv_rn(sc[t * kTopK + k], __fadd_rn(wsum, 1e-20f)), p.route_scale);
    }
  }
}

// n rows of values (BF16 in shared memory, [n][k]) quantized as the checkpoint requires and staged per row:
// amax per row, blocks of all rows over the block's threads.
template <bool MX, int Threads = kExpertThreads>
__device__ void quantize_stage_rows(const uint16_t* xb, int n, int k, uint32_t* xh, float* xsc, __half* xv,
                                    float* amax, uint8_t* qc, uint8_t* qs) {
  constexpr int kWarps = Threads / 32;
  __shared__ float wmax[kWarps][kMaxRows];
  const int t = threadIdx.x;
  if (!MX) {
    for (int j = 0; j < n; ++j) {
      float mloc = 0.0f;
      for (int i = t; i < k; i += Threads) mloc = fmaxf(mloc, fabsf(bf16_to_f32(xb[j * k + i])));
      mloc = warp_max(mloc);
      if ((t & 31) == 0) wmax[t >> 5][j] = mloc;
    }
    __syncthreads();
    if (t < n) {
      float a = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) a = fmaxf(a, wmax[w][t]);
      amax[t] = a;
    }
    __syncthreads();
    for (int i = t; i < n * (k / 16); i += Threads) {
      const int j = i / (k / 16), b = i % (k / 16);
      const float a = amax[j];
      const float encoder = a > 0.0f ? fminf(__fdiv_rn(1536.0f, a), kFltMax) : kFltMax;
      const float decoder = __fdiv_rn(1.0f, encoder);
      const float outer = __fmul_rn(a, float(1.0 / 1536.0));
      float x[16];
#pragma unroll
      for (int e = 0; e < 16; ++e) x[e] = bf16_to_f32(xb[j * k + 16 * b + e]);
      uint8_t sc;
      quant_block_nvfp4(x, encoder, decoder, outer, qc + j * (k / 2) + 8 * b, sc);
      qs[j * (k / 16) + b] = sc;
    }
  } else {
    for (int i = t; i < n * (k / 32); i += Threads) {
      const int j = i / (k / 32), b = i % (k / 32);
      float x[32];
#pragma unroll
      for (int e = 0; e < 32; ++e) x[e] = bf16_to_f32(xb[j * k + 32 * b + e]);
      uint8_t sc;
      quant_block_mxfp8(x, qc + j * k + 32 * b, sc);
      qs[j * (k / 32) + b] = sc;
    }
  }
  __syncthreads();
  for (int j = 0; j < n; ++j)
    stage_row<Threads>(k, MX, qc + j * (MX ? k : k / 2), qs + j * (MX ? k / 32 : k / 16), xh + j * (k / 2),
                              xsc + j * (k / 16), xv + j * k);
  __syncthreads();
}

// The one-row input staging of expert_w13_body (row r of x_in): rms norm with the weight in the body's order, then
// quantize_stage_rows for one row.
template <bool MX>
__device__ __forceinline__ void w13_stage_row(const ExpertParams& p, int r, uint16_t* xb, uint32_t* xh, float* xsc,
                                              __half* xv, float* amax, uint8_t* qc, uint8_t* qs) {
  __shared__ float wss1[kExpertWarps];
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const float4* x4 = reinterpret_cast<const float4*>(p.x_in);
  float4 xr[2];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int i = t + q * kExpertThreads;
    xr[q] = i < kHidden / 4 ? x4[r * (kHidden / 4) + i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  float ss = 0.0f;
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const float4 v = xr[q];
    ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
  }
  ss = warp_sum(ss);
  if (lane == 0) wss1[warp] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int w = 0; w < kExpertWarps; ++w) tot += wss1[w];
  const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int i = t + q * kExpertThreads;
    if (i < kHidden / 4) {
      const float4 v = xr[q];
      const uint2 nw = reinterpret_cast<const uint2*>(p.norm_w)[i];
      uint16_t* o = xb + 4 * i;
      o[0] = f32_to_bf16_bits(v.x * inv * bf16lo(nw.x));
      o[1] = f32_to_bf16_bits(v.y * inv * bf16hi(nw.x));
      o[2] = f32_to_bf16_bits(v.z * inv * bf16lo(nw.y));
      o[3] = f32_to_bf16_bits(v.w * inv * bf16hi(nw.y));
    }
  }
  __syncthreads();
  quantize_stage_rows<MX>(xb, 1, kHidden, xh, xsc, xv, amax, qc, qs);
}

// w13: a warp runs gate/up pairs (half-warps own the two rows of a pair) for every row of the block's group, the next
// pair's loads in flight. Dynamic shared memory: normalized inputs [8][kHidden] BF16, staged rows, codes.
template <bool MX>
__device__ void expert_w13_body(const ExpertParams& p) {
  // the normalized rows (BF16 [8][2304]) are dead once quantized: the staged rows (NVFP4 f16x2 [8][1152] or MXFP8
  // half [8][2304], the same 4608 bytes per row) reuse their memory
  extern __shared__ __align__(16) uint8_t dyn[];
  uint16_t* xb = reinterpret_cast<uint16_t*>(dyn);
  uint32_t* xh = reinterpret_cast<uint32_t*>(dyn);
  __half* xv = reinterpret_cast<__half*>(dyn);
  float* xsc = reinterpret_cast<float*>(dyn + size_t(kMaxRows) * kHidden * 2);           // [8][144]
  uint8_t* qc = reinterpret_cast<uint8_t*>(xsc + kMaxRows * (kHidden / 16));              // [8][2304]
  uint8_t* qs = qc + kMaxRows * kHidden;                                                   // [8][144]
  __shared__ int ids[kMaxSlots];
  __shared__ float sc[kMaxSlots];
  __shared__ float amax[kMaxRows];
  __shared__ float wss[kExpertWarps][kMaxRows];
  __shared__ Groups G;
  constexpr int NB = MX ? kHidden / 32 : kHidden / 16;
  const int rows = p.rows < 1 ? 1 : p.rows;
  const int t = threadIdx.x;
  // one row: the MLP input (two launches back: final once this pass runs) normalized and quantized before the wait
  const bool early = rows == 1;
  if (early) w13_stage_row<MX>(p, 0, xb, xh, xsc, xv, amax, qc, qs);
  pdl_wait();
  pdl_launch();
  route_rows(p, rows, ids, sc);
  make_groups(ids, rows * kTopK, blockIdx.y, G);
  if (int(blockIdx.y) >= G.count) return;
  const int e = G.expert, n = G.n;
  // the w2 rows of this expert that w2 block (x, y) reads, into L2 once warp 0's pairs are done (DRAM has room in
  // the pass's tail)
  auto w2_to_l2 = [&]() {
    constexpr int NB2 = MX ? kInter / 32 : kInter / 16, RB2 = MX ? kInter : kInter / 2;
    const int lo2 = blockIdx.x * kHidden / kSplit, hi2 = (blockIdx.x + 1) * kHidden / kSplit;
    const uint8_t* c2 = static_cast<const uint8_t*>(p.w2_codes) + size_t(e) * kHidden * RB2;
    const uint8_t* s2 = static_cast<const uint8_t*>(p.w2_scales) + size_t(e) * kHidden * NB2;
    prefetch_range(c2 + size_t(lo2) * RB2, size_t(hi2 - lo2) * RB2);
    prefetch_range(s2 + size_t(lo2) * NB2, size_t(hi2 - lo2) * NB2);
  };
  const int warp = t >> 5, lane = t & 31, l = lane & 15;
  const int rowbytes = MX ? kHidden : kHidden / 2, sbytes = NB;
  const uint8_t* codes = static_cast<const uint8_t*>(p.codes) + size_t(e) * 2 * kInter * rowbytes;
  const uint8_t* scales = static_cast<const uint8_t*>(p.scales) + size_t(e) * 2 * kInter * sbytes;
  const int lo = blockIdx.x * kInter / kSplit, hi = (blockIdx.x + 1) * kInter / kSplit;
  RowShare<NB, 16, MX> cur, nxt;
  int pair = lo + warp;
  if (pair < hi) {
    const int row = 2 * pair + (lane >> 4);
    cur.load(codes + size_t(row) * rowbytes, scales + size_t(row) * sbytes, l);
  }

  if (!early) {
  // the group's rows: rms norm with the weight (each thread: float4 t and t + 512 of the row, as one row's step)
  const float4* x4 = reinterpret_cast<const float4*>(after_wait(p.x_in));
  for (int j = 0; j < n; ++j) {
    const int r = G.slot[j] / kTopK;
    float4 xr[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int i = t + q * kExpertThreads;
      xr[q] = i < kHidden / 4 ? x4[r * (kHidden / 4) + i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float ss = 0.0f;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const float4 v = xr[q];
      ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
    }
    ss = warp_sum(ss);
    if (lane == 0) wss[warp][j] = ss;
  }
  __syncthreads();
  for (int j = 0; j < n; ++j) {
    const int r = G.slot[j] / kTopK;
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < kExpertWarps; ++w) tot += wss[w][j];
    const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int i = t + q * kExpertThreads;
      if (i < kHidden / 4) {
        const float4 v = x4[r * (kHidden / 4) + i];
        const uint2 nw = reinterpret_cast<const uint2*>(p.norm_w)[i];
        uint16_t* o = xb + j * kHidden + 4 * i;
        o[0] = f32_to_bf16_bits(v.x * inv * bf16lo(nw.x));
        o[1] = f32_to_bf16_bits(v.y * inv * bf16hi(nw.x));
        o[2] = f32_to_bf16_bits(v.z * inv * bf16lo(nw.y));
        o[3] = f32_to_bf16_bits(v.w * inv * bf16hi(nw.y));
      }
    }
  }
  __syncthreads();
  quantize_stage_rows<MX>(xb, n, kHidden, xh, xsc, xv, amax, qc, qs);
  }
  float outer[kMaxRows];
#pragma unroll
  for (int j = 0; j < kMaxRows; ++j)
    outer[j] = MX ? 1.0f : __fmul_rn(__fdiv_rn(p.amax[e], p.denominator), __fmul_rn(j < n ? amax[j] : 0.0f, float(1.0 / 1536.0)));
  uint16_t* act = after_wait(p.act);
  for (; pair < hi; pair += kExpertWarps) {
    const int next = pair + kExpertWarps;
    if (next < hi) {
      const int row = 2 * next + (lane >> 4);
      nxt.load(codes + size_t(row) * rowbytes, scales + size_t(row) * sbytes, l);
    }
    float acc[kMaxRows];
    cur.dot(l, n, xh, kHidden / 2, xsc, kHidden / 16, xv, kHidden, acc);
#pragma unroll
    for (int j = 0; j < kMaxRows; ++j) {
      if (j >= n) break;
      const float v = round_bf16(__fmul_rn(warp_sum<16>(acc[j]), outer[j]));
      const float up = __shfl_sync(0xffffffffu, v, 16);
      if (lane == 0) act[size_t(G.slot[j]) * kInter + pair] = f32_to_bf16_bits(softcapped_swiglu(v, up, p.gate_cap, p.up_cap));
    }
    cur = nxt;
  }
  if (p.w2_l2 && t == 0) w2_to_l2();
}

// one kernel per format: the NVFP4 path keeps its own register allocation (one kernel for both took the MXFP8 one's)
template <bool MX>
__global__ void __launch_bounds__(kExpertThreads) expert_w13_kernel(const __grid_constant__ ExpertParams p) {
  expert_w13_body<MX>(p);         // (the dependency wait inside, after the one-row input staging)
}

// w2: groups of the routed slots (ids written by w13); a warp runs groups of 4 rows (8 lanes per row) for every slot
// of the group; each slot's activation is quantized per block while the first weights load.
template <bool MX>
__device__ void expert_w2_body(const ExpertParams& p) {
  // activations (BF16 [8][1152]) dead once quantized: the staged rows reuse their memory (as in w13)
  extern __shared__ __align__(16) uint8_t dyn[];
  uint16_t* ab = reinterpret_cast<uint16_t*>(dyn);
  uint32_t* xh = reinterpret_cast<uint32_t*>(dyn);
  __half* xv = reinterpret_cast<__half*>(dyn);
  float* xsc = reinterpret_cast<float*>(dyn + size_t(kMaxRows) * kInter * 2);             // [8][72]
  uint8_t* qc = reinterpret_cast<uint8_t*>(xsc + kMaxRows * (kInter / 16));               // [8][1152]
  uint8_t* qs = qc + kMaxRows * kInter;                                                    // [8][72]
  __shared__ float amax[kMaxRows];
  __shared__ Groups G;
  constexpr int NB = MX ? kInter / 32 : kInter / 16;
  const int rows = p.rows < 1 ? 1 : p.rows;
  const int* ids;
  __shared__ int rids[kMaxSlots];
  __shared__ float rsc[kMaxSlots];
  if (p.logits) {
    // the routing again from the router logits (two launches back, final once the w13 pass started): the groups and
    // the first weight loads before the wait, under the w13 pass (w13 writes the routing for the combination)
    route_rows(p, rows, rids, rsc);
    ids = rids;
  } else {
    pdl_wait();
    pdl_launch();
    ids = after_wait(p.ids);
  }
  make_groups(ids, rows * kTopK, blockIdx.y, G);
  if (int(blockIdx.y) >= G.count) {
    if (p.logits) { pdl_wait(); pdl_launch(); }
    return;
  }
  const int e = G.expert, n = G.n;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, l = lane & 7;
  const int rowbytes = MX ? kInter : kInter / 2, sbytes = NB;
  const uint8_t* codes = static_cast<const uint8_t*>(p.codes) + size_t(e) * kHidden * rowbytes;
  const uint8_t* scales = static_cast<const uint8_t*>(p.scales) + size_t(e) * kHidden * sbytes;
  const int lo = blockIdx.x * kHidden / kSplit, hi = (blockIdx.x + 1) * kHidden / kSplit;
  RowShare<NB, 8, MX> cur, nxt;
  int row = lo + warp * 4 + (lane >> 3);
  if (row < hi) cur.load(codes + size_t(row) * rowbytes, scales + size_t(row) * sbytes, l);
  if (p.logits) {
    pdl_wait();
    pdl_launch();
  }
  const uint16_t* a = after_wait(p.act);
  for (int j = 0; j < n; ++j)
    for (int i = t; i < kInter / 8; i += kExpertThreads)
      reinterpret_cast<uint4*>(ab + j * kInter)[i] = reinterpret_cast<const uint4*>(a + size_t(G.slot[j]) * kInter)[i];
  __syncthreads();
  quantize_stage_rows<MX>(ab, n, kInter, xh, xsc, xv, amax, qc, qs);
  float outer[kMaxRows];
#pragma unroll
  for (int j = 0; j < kMaxRows; ++j)
    outer[j] = MX ? 1.0f : __fmul_rn(__fdiv_rn(p.amax[e], p.denominator), __fmul_rn(j < n ? amax[j] : 0.0f, float(1.0 / 1536.0)));
  uint16_t* out = after_wait(p.out);
  for (int base = lo + warp * 4; base < hi; base += 4 * kExpertWarps) {
    const int nbase = base + 4 * kExpertWarps;
    const int nrow = nbase + (lane >> 3);
    if (nbase < hi && nrow < hi) nxt.load(codes + size_t(nrow) * rowbytes, scales + size_t(nrow) * sbytes, l);
    float acc[kMaxRows];
    cur.dot(l, n, xh, kInter / 2, xsc, kInter / 16, xv, kInter, acc);
#pragma unroll
    for (int j = 0; j < kMaxRows; ++j) {
      if (j >= n) break;
      const float s = warp_sum<8>(acc[j]);
      if (l == 0 && row < hi) out[size_t(G.slot[j]) * kHidden + row] = f32_to_bf16_bits(__fmul_rn(s, outer[j]));
    }
    cur = nxt;
    row = nrow;
  }
}

template <bool MX>
__global__ void __launch_bounds__(kExpertThreads) expert_w2_kernel(const __grid_constant__ ExpertParams p) {
  prefetch_share(p.prefetch);
  expert_w2_body<MX>(p);
}

// ------------------------------------------------------------------------- flat passes over the expert groups
// Several rows (a verification step): the routing, the input norm and the input quantization run once per row
// (staged in global memory), w2 quantizes its slots' activations in the pass, and the weight passes split the flat
// list of (group, output row) items over one block per SM, so every weight byte is read once and every SM streams.
// The per-row arithmetic is the one of expert_w13_body / expert_w2_body (same lane shares, accumulation order,
// scales), so each row equals the fused kernels' result.

// The groups of make_groups for all of them at once, built in parallel: count, expert, slots per group.
struct GroupTable {
  int count;
  int expert[kMaxSlots];
  int n[kMaxSlots];
  int slot[kMaxSlots][kMaxRows];
};

// The groups of make_groups for all of them at once, built by warp 0 (at most 64 slots, two per lane): each slot's
// mask of the slots routed to its expert from the ids in shared memory, the leaders by ballot; one block barrier.
__device__ __forceinline__ void make_table_warp(const int* ids, int slots, GroupTable& T) {
  __shared__ __align__(16) int sid[kMaxSlots];
  const int t = threadIdx.x;
  if (t < 32) {
    const int e0 = t < slots ? ids[t] : -1 - t, e1 = t + 32 < slots ? ids[t + 32] : -33 - t;
    sid[t] = e0;
    sid[t + 32] = e1;
    __syncwarp();
    uint32_t a0 = 0u, b0 = 0u, a1 = 0u, b1 = 0u;        // slot t's and t + 32's masks (low, high 32 slots)
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      const int4 lo = *reinterpret_cast<const int4*>(sid + j), hi = *reinterpret_cast<const int4*>(sid + 32 + j);
      const int l4[4] = {lo.x, lo.y, lo.z, lo.w}, h4[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a0 |= uint32_t(l4[i] == e0) << (j + i);
        b0 |= uint32_t(h4[i] == e0) << (j + i);
        a1 |= uint32_t(l4[i] == e1) << (j + i);
        b1 |= uint32_t(h4[i] == e1) << (j + i);
      }
    }
    const unsigned long long m0 = (static_cast<unsigned long long>(b0) << 32) | a0;
    const unsigned long long m1 = (static_cast<unsigned long long>(b1) << 32) | a1;
    const int f0 = __ffsll(static_cast<long long>(m0)) - 1, f1 = __ffsll(static_cast<long long>(m1)) - 1;
    const bool v0 = t < slots, v1 = t + 32 < slots;
    const unsigned long long L = static_cast<unsigned long long>(__ballot_sync(0xffffffffu, v0 && f0 == t)) |
                                 (static_cast<unsigned long long>(__ballot_sync(0xffffffffu, v1 && f1 == t + 32)) << 32);
    if (v0) {
      const int g = __popcll(L & ((1ull << f0) - 1ull));
      if (f0 == t) { T.expert[g] = e0; T.n[g] = __popcll(m0); }
      T.slot[g][__popcll(m0 & ((1ull << t) - 1ull))] = t;
    }
    if (v1) {
      const int g = __popcll(L & ((1ull << f1) - 1ull));
      if (f1 == t + 32) { T.expert[g] = e1; T.n[g] = __popcll(m1); }
      T.slot[g][__popcll(m1 & ((1ull << (t + 32)) - 1ull))] = t + 32;
    }
    if (t == 0) T.count = __popcll(L);
  }
  __syncthreads();
}

// The raw quantized row n (codes, scale bytes) for the block-scaled MMA passes: NVFP4 k / 2 code bytes and k / 16
// E4M3 scales per row, MXFP8 k code bytes and k / 32 UE8M0 scales per row (rows at the format's own stride).
template <bool MX>
__device__ __forceinline__ void copy_raw(const uint8_t* qc, const uint8_t* qs, int k, uint8_t* codes, uint8_t* scales, int n) {
  const int cb = MX ? k : k / 2, sb = MX ? k / 32 : k / 16;
  for (int i = threadIdx.x; i < cb / 16; i += blockDim.x)
    reinterpret_cast<uint4*>(codes + size_t(n) * cb)[i] = reinterpret_cast<const uint4*>(qc)[i];
  for (int i = threadIdx.x; i < sb; i += blockDim.x) scales[size_t(n) * sb + i] = qs[i];
}

// expert_route: one block per row: the routing of route_rows, the norm and the quantization of expert_w13_body, the
// staged row (NVFP4 f16x2 pairs / MXFP8 half values, FP32 block scales, the amax) written to xq / xq_scale / xq_amax.
template <bool MX>
__device__ void expert_route_body(const ExpertParams& p) {
  __shared__ float bs[kExperts];
  __shared__ int ids[kTopK];
  __shared__ float sc[kTopK];
  __shared__ float wss[kExpertWarps];
  __shared__ __align__(16) uint16_t xb[kHidden];
  __shared__ __align__(16) uint8_t qc[kHidden];
  __shared__ uint8_t qs[kHidden / 16];
  // the staged row in shared memory first (written out after the wait)
  __shared__ __align__(16) uint32_t sxh[kHidden / 2];
  __shared__ float sxsc[kHidden / 16];
  __shared__ float samax;
  const int r = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  // the MLP input is two launches back (final once this kernel runs): normalized and quantized before the wait
  const float4* x4 = reinterpret_cast<const float4*>(p.x_in) + r * (kHidden / 4);
  float4 xr[2];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int i = t + q * kExpertThreads;
    xr[q] = i < kHidden / 4 ? x4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  float ss = 0.0f;
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const float4 v = xr[q];
    ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
  }
  ss = warp_sum(ss);
  if (lane == 0) wss[warp] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int w = 0; w < kExpertWarps; ++w) tot += wss[w];
  const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int i = t + q * kExpertThreads;
    if (i < kHidden / 4) {
      const float4 v = xr[q];
      const uint2 nw = reinterpret_cast<const uint2*>(p.norm_w)[i];
      uint16_t* o = xb + 4 * i;
      o[0] = f32_to_bf16_bits(v.x * inv * bf16lo(nw.x));
      o[1] = f32_to_bf16_bits(v.y * inv * bf16hi(nw.x));
      o[2] = f32_to_bf16_bits(v.z * inv * bf16lo(nw.y));
      o[3] = f32_to_bf16_bits(v.w * inv * bf16hi(nw.y));
    }
  }
  __syncthreads();
  quantize_stage_rows<MX>(xb, 1, kHidden, sxh, sxsc, reinterpret_cast<__half*>(sxh), &samax, qc, qs);
  const float bias_t = t < kExperts ? p.bias[t] : 0.0f;     // static: before the wait
  pdl_wait();
  pdl_launch();
  const float* lg = after_wait(p.logits) + r * kExperts;
  if (t < kExperts) bs[t] = 1.0f / (1.0f + expf(-lg[t])) + bias_t;
  __syncthreads();
  if (t < kExperts) {
    const float b = bs[t];
    int rank = 0;
#pragma unroll 8
    for (int j = 0; j < kExperts; ++j) {
      const float bj = bs[j];
      rank += (bj > b) || (bj == b && j < t);
    }
    if (rank < kTopK) { ids[rank] = t; sc[rank] = 1.0f / (1.0f + expf(-lg[t])); }
  }
  // the staged row out
  uint4* xq = reinterpret_cast<uint4*>(static_cast<uint8_t*>(after_wait(p.xq)) + size_t(r) * kHidden * 2);
  for (int i = t; i < kHidden * 2 / 16; i += kExpertThreads) xq[i] = reinterpret_cast<const uint4*>(sxh)[i];
  float* xsc = after_wait(p.xq_scale) + r * (kHidden / 16);
  for (int i = t; i < (MX ? kHidden / 32 : kHidden / 16); i += kExpertThreads) xsc[i] = sxsc[i];
  if (!MX && t == 0) after_wait(p.xq_amax)[r] = samax;
  __syncthreads();
  if (t == 0) {
    float wsum = 0.0f;
    for (int k = 0; k < kTopK; ++k) wsum = __fadd_rn(wsum, sc[k]);
    int* oid = after_wait(p.ids_out) + r * kTopK;
    float* ow = after_wait(p.weights_out) + r * kTopK;
    for (int k = 0; k < kTopK; ++k) {
      oid[k] = ids[k];
      ow[k] = __fmul_rn(__fdiv_rn(sc[k], __fadd_rn(wsum, 1e-20f)), p.route_scale);
    }
  }
  if (p.xq_codes) copy_raw<MX>(qc, qs, kHidden, after_wait(p.xq_codes), after_wait(p.xq_sc8), r);
}

__global__ void __launch_bounds__(kExpertThreads) expert_route_kernel(const __grid_constant__ ExpertParams p) {
  if (p.mxfp8) expert_route_body<true>(p);      // (the dependency wait inside, after the input staging)
  else expert_route_body<false>(p);
  if (p.work_reset && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 1 + 2 * kMaxSlots)
    after_wait(p.work_reset)[threadIdx.x] = 0;       // the flat passes' counters (w13, w2 per group, fused done)
}

// The router GEMV of the flat passes: grid (kRouterSplits + 1, rows), 512 threads. Block (s < kRouterSplits, r): warp
// w computes router output e = 16 s + w of row r with the rows GEMV's arithmetic (norm prologue: threads 0..255 over
// elements t + 256 m, the 8 warp partials in order; the dot: lane chunks lane + 32 u, u < 9, warp sum), its weights
// loaded before the wait, and writes the logit; block (kRouterSplits, r) stages the row's w13 input (the route body's
// norm and quantization) and, for row 0, zeroes the flat passes' counters.
constexpr int kRouterSplits = kExperts / kExpertWarps;
static_assert(kRouterSplits * kExpertWarps == kExperts, "router outputs per block");
// The top 8 of sigmoid(logit) + bias of one row (128 logits) in one warp, in expert_route_body's order (descending,
// ties to the lower expert). Each lane sorts its four values (order-preserving keys, -0 as +0), then eight rounds of
// two warp reductions (the largest key, the lowest expert holding it); lane k < 8 gets the k-th expert and (want_sg)
// its sigmoid.
__device__ __forceinline__ uint32_t order_key(float v) {
  uint32_t u = __float_as_uint(v);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
template <bool WantSg>
__device__ __forceinline__ void warp_top8(const float* lg, const float* bias, int& id, float& sg) {
  const int lane = threadIdx.x & 31;
  const float4 l4 = __ldcg(reinterpret_cast<const float4*>(lg) + lane);
  const float4 b4 = reinterpret_cast<const float4*>(bias)[lane];
  const float lx[4] = {l4.x, l4.y, l4.z, l4.w}, bx[4] = {b4.x, b4.y, b4.z, b4.w};
  float g[4];
  uint32_t k[4];
  int ix[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    g[i] = 1.0f / (1.0f + expf(-lx[i]));
    k[i] = order_key(g[i] + bx[i]);
    ix[i] = 4 * lane + i;
  }
  // sort the lane's four by (descending key, ascending expert): a total order, any network
  auto cs = [&](int a, int b) {
    if (k[b] > k[a] || (k[b] == k[a] && ix[b] < ix[a])) {
      const uint32_t tk = k[a]; k[a] = k[b]; k[b] = tk;
      const int ti = ix[a]; ix[a] = ix[b]; ix[b] = ti;
    }
  };
  cs(0, 1); cs(2, 3); cs(0, 2); cs(1, 3); cs(1, 2);
  id = 0;
  sg = 0.0f;
#pragma unroll
  for (int r = 0; r < kTopK; ++r) {
    const uint32_t m = __reduce_max_sync(0xffffffffu, k[0]);
    const int w = int(__reduce_min_sync(0xffffffffu, k[0] == m ? uint32_t(ix[0]) : 0xffffu));
    if (lane == r) id = w;
    if (WantSg) {
      const int e = w & 3;
      const float gs = e == 0 ? g[0] : e == 1 ? g[1] : e == 2 ? g[2] : g[3];
      const float v = __shfl_sync(0xffffffffu, gs, w >> 2);
      if (lane == r) sg = v;
    }
    if (ix[0] == w) {                 // pop the head
      k[0] = k[1]; ix[0] = ix[1];
      k[1] = k[2]; ix[1] = ix[2];
      k[2] = k[3]; ix[2] = ix[3];
      k[3] = 0u; ix[3] = 0xffff;
    }
  }
}

template <bool MX>
__device__ void expert_router_route_body(const ExpertParams& p) {
  __shared__ __align__(16) uint16_t xg[kHidden];       // the GEMV-order normalized row (BF16)
  __shared__ float part[8];
  __shared__ float wss[kExpertWarps];
  __shared__ __align__(16) uint16_t xb[kHidden];
  __shared__ __align__(16) uint8_t qc[kHidden];
  __shared__ uint8_t qs[kHidden / 16];
  __shared__ __align__(16) uint32_t sxh[kHidden / 2];
  __shared__ float sxsc[kHidden / 16];
  __shared__ float samax;
  const int s = blockIdx.x, r = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  // block s = kRouterSplits of row r stages the w13 input (from the wait on); the others write the router logits,
  // which the flat passes rank themselves (warp_top8)
  if (s == kRouterSplits) {
    pdl_wait();
    pdl_launch();
    const float* x = after_wait(p.x_in) + size_t(r) * kHidden;
    // the w13 input of row r: the route body's norm (512 threads, float4 t and t + 512) and quantization
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4 xr[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int i = t + q * kExpertThreads;
      xr[q] = i < kHidden / 4 ? x4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float ss = 0.0f;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const float4 v = xr[q];
      ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
    }
    ss = warp_sum(ss);
    if (lane == 0) wss[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < kExpertWarps; ++w) tot += wss[w];
    const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int i = t + q * kExpertThreads;
      if (i < kHidden / 4) {
        const float4 v = xr[q];
        const uint2 nw = reinterpret_cast<const uint2*>(p.norm_w)[i];
        uint16_t* o = xb + 4 * i;
        o[0] = f32_to_bf16_bits(v.x * inv * bf16lo(nw.x));
        o[1] = f32_to_bf16_bits(v.y * inv * bf16hi(nw.x));
        o[2] = f32_to_bf16_bits(v.z * inv * bf16lo(nw.y));
        o[3] = f32_to_bf16_bits(v.w * inv * bf16hi(nw.y));
      }
    }
    __syncthreads();
    quantize_stage_rows<MX>(xb, 1, kHidden, sxh, sxsc, reinterpret_cast<__half*>(sxh), &samax, qc, qs);
    uint4* xq = reinterpret_cast<uint4*>(static_cast<uint8_t*>(after_wait(p.xq)) + size_t(r) * kHidden * 2);
    for (int i = t; i < kHidden * 2 / 16; i += kExpertThreads) xq[i] = reinterpret_cast<const uint4*>(sxh)[i];
    float* xsc = after_wait(p.xq_scale) + r * (kHidden / 16);
    for (int i = t; i < (MX ? kHidden / 32 : kHidden / 16); i += kExpertThreads) xsc[i] = sxsc[i];
    if (!MX && t == 0) after_wait(p.xq_amax)[r] = samax;
    if (p.xq_codes) copy_raw<MX>(qc, qs, kHidden, after_wait(p.xq_codes), after_wait(p.xq_sc8), r);
    if (p.work_reset && r == 0 && t < 1 + 2 * kMaxSlots) after_wait(p.work_reset)[t] = 0;
    return;
  }
  const int e = s * kExpertWarps + warp;
  constexpr int kU = 9, kM = 9;                        // k / 8 = 288 = 9 x 32 chunks; k = 9 x 256 elements
  uint4 wv[kU];
  const uint4* wr = reinterpret_cast<const uint4*>(p.router_w) + size_t(e) * (kHidden / 8);
#pragma unroll
  for (int u = 0; u < kU; ++u) wv[u] = ldg_stream(wr + lane + 32 * u);
  float nwv[kM];
#pragma unroll
  for (int m = 0; m < kM; ++m) nwv[m] = t < 256 ? bf16_to_f32(p.norm_w[t + 256 * m]) : 0.0f;
  pdl_wait();
  pdl_launch();
  const float* x = after_wait(p.x_in) + size_t(r) * kHidden;
  float xv[kM];
  if (t < 256) {
    float ss = 0.0f;
#pragma unroll
    for (int m = 0; m < kM; ++m) { xv[m] = x[t + 256 * m]; ss = fmaf(xv[m], xv[m], ss); }
    ss = warp_sum(ss);
    if (lane == 0) part[warp] = ss;
  }
  __syncthreads();
  if (t < 256) {
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
#pragma unroll
    for (int m = 0; m < kM; ++m) xg[t + 256 * m] = f32_to_bf16_bits(xv[m] * inv * nwv[m]);
  }
  __syncthreads();
  {
    float acc = 0.0f;
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const uint4 xq4 = reinterpret_cast<const uint4*>(xg)[lane + 32 * u];
      const float xf[8] = {bf16lo(xq4.x), bf16hi(xq4.x), bf16lo(xq4.y), bf16hi(xq4.y),
                           bf16lo(xq4.z), bf16hi(xq4.z), bf16lo(xq4.w), bf16hi(xq4.w)};
      float d = 0.0f;
      const uint32_t ww[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        d = fmaf(bf16lo(ww[i]), xf[2 * i], d);
        d = fmaf(bf16hi(ww[i]), xf[2 * i + 1], d);
      }
      acc += d;
    }
    acc = warp_sum(acc);
    if (lane == 0) after_wait(p.logits_out)[size_t(r) * kExperts + e] = acc;
  }
}
__global__ void __launch_bounds__(kExpertThreads) expert_router_route_kernel(const __grid_constant__ ExpertParams p) {
  if (p.mxfp8) expert_router_route_body<true>(p);
  else expert_router_route_body<false>(p);
}

constexpr int kFlatThreads = 256;
// dynamic chunks of the flat passes: w13 items (group, gate / up pair) per claim, w2 quads per claim; w2's warps per
// group in proportion to kW2Cost + the group's slots
constexpr int kW13Chunk = 4;
constexpr int kW2Chunk = 1;
constexpr int kW2Cost = 16;
static_assert(kInter % kW13Chunk == 0, "a w13 chunk stays in one group (its count is per group)");       // one block per SM; the decoded share needs the registers of 256 threads
// (measured slower: 2-4 raw shares in flight with per-row decoding, 2 blocks per SM, L2 prefetch of the warp range,
// of sliding windows or of single lines ahead)
constexpr int kFlatWarps = kFlatThreads / 32;

// A warp's items q0 .. q1 - 1 with one weight share in flight: use(share, q, refill) decodes item q's share and
// calls refill(), which issues item q + 1's loads; they land while item q's rows are multiplied. (More shares in
// flight spill registers and run slower.)
template <class Share, class Load, class Use>
__device__ __forceinline__ void flat_run(int q0, int q1, Share& buf, Load load, Use use) {
  for (int q = q0; q < q1; ++q)
    use(buf, q, [&] {
      if (q + 1 < q1) load(buf, q + 1);
    });
}

// One item's weight share decoded once into registers and multiplied with each of its rows in a loop (the fused
// kernels unroll all eight rows): per row the arithmetic of RowShare::dot. The rows are staged in 16-byte chunks,
// chunk c of block b at c * NB + b, so the lanes of a phase read consecutive chunks (no bank conflicts).
template <int NB, int L, bool MX>
struct Decoded;

template <int NB, int L>
struct Decoded<NB, L, false> {           // NVFP4: blocks of 16 inputs, two chunks of f16x2 pairs
  static constexpr int kPer = (NB + L - 1) / L;
  static constexpr int kChunks = 2;
  uint32_t wh[kPer][8];
  float sw[kPer];
  __device__ __forceinline__ void set(const RowShare<NB, L, false>& rs) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const uint2 wc = (i & 1) ? make_uint2(rs.c[i / 2].z, rs.c[i / 2].w) : make_uint2(rs.c[i / 2].x, rs.c[i / 2].y);
      e2m1x8_to_h2(wc, wh[i]);
      sw[i] = e4m3_value(rs.s[i]);
    }
  }
  __device__ __forceinline__ float dot(int l, const uint4* x, const float* xsc) const {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int b = l + i * L;
      if ((NB % L) && b >= NB) break;
      const uint4 a = x[b], c = x[NB + b];
      const uint32_t xh[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
      acc = fmaf(block_dot_h2(wh[i], xh), __fmul_rn(sw[i], xsc[b]), acc);
    }
    return acc;
  }
};

template <int NB, int L>
struct Decoded<NB, L, true> {            // MXFP8: blocks of 32 inputs, four chunks of half values
  static constexpr int kPer = (NB + L - 1) / L;
  static constexpr int kChunks = 4;
  uint32_t wv[kPer][16];                 // the E4M3 pairs as f16x2
  float sw[kPer];
  __device__ __forceinline__ void set(const RowShare<NB, L, true>& rs) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const uint32_t ww[8] = {rs.c[2 * i].x, rs.c[2 * i].y, rs.c[2 * i].z, rs.c[2 * i].w,
                              rs.c[2 * i + 1].x, rs.c[2 * i + 1].y, rs.c[2 * i + 1].z, rs.c[2 * i + 1].w};
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const uint16_t a = static_cast<uint16_t>(ww[k] & 0xffff), b = static_cast<uint16_t>(ww[k] >> 16);
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(wv[i][2 * k]) : "h"(a));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(wv[i][2 * k + 1]) : "h"(b));
      }
      sw[i] = exp2f(float(int(rs.s[i]) - 127));
    }
  }
  __device__ __forceinline__ float dot(int l, const uint4* x, const float* xsc) const {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int b = l + i * L;
      if ((NB % L) && b >= NB) break;
      float xf[32];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 v = x[c * NB + b];
        const uint32_t vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&vv[e]));
          xf[8 * c + 2 * e] = f.x;
          xf[8 * c + 2 * e + 1] = f.y;
        }
      }
      float a = 0.0f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&wv[i][2 * k]));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&wv[i][2 * k + 1]));
        a = fmaf(lo.x, xf[4 * k], a);
        a = fmaf(lo.y, xf[4 * k + 1], a);
        a = fmaf(hi.x, xf[4 * k + 2], a);
        a = fmaf(hi.y, xf[4 * k + 3], a);
      }
      acc = fmaf(a, __fmul_rn(sw[i], xsc[b]), acc);
    }
    return acc;
  }
};

// Copy n staged rows (global, linear: block b's chunks at b * C .. b * C + C - 1, rows at row(j) * NB * C) into the
// chunked layout (rows at j * NB * C), every load of a thread issued before its stores.
template <int NB, int C, int Threads, int MaxRows, class Row>
__device__ __forceinline__ void stage_chunks(const uint4* src, uint4* dst, int n, Row row) {
  constexpr int kPer = (MaxRows * NB * C + Threads - 1) / Threads;
  uint4 v[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int i = threadIdx.x + k * Threads, j = i / (NB * C), c = i % (NB * C);
    if (j < n) v[k] = src[size_t(row(j)) * NB * C + c];
  }
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int i = threadIdx.x + k * Threads, j = i / (NB * C), c = i % (NB * C);
    if (j < n) dst[j * NB * C + (c % C) * NB + c / C] = v[k];
  }
}

// The block's and the warp's contiguous share of `total` items.
__device__ __forceinline__ void flat_share(int total, int warp, int& b0, int& b1, int& q0, int& q1) {
  b0 = int(int64_t(total) * blockIdx.x / gridDim.x);
  b1 = int(int64_t(total) * (blockIdx.x + 1) / gridDim.x);
  q0 = b0 + (b1 - b0) * warp / kFlatWarps;
  q1 = b0 + (b1 - b0) * (warp + 1) / kFlatWarps;
}

// w13 over the flat (group, gate/up pair) items; the staged rows of all rows in shared memory.
// Fused (expert_flat_fused): the table in Tx (kept for the w2 phase), block 0 writes the routing (ids_out /
// weights_out), and every finished chunk's activations are stored and counted into done[group] (fence first).
struct W2Homes;
__device__ void w2_homes_fwd(const ExpertParams& q, const GroupTable& T, W2Homes& H);
template <bool MX, bool Fused = false>
__device__ void expert_w13_flat_body(const ExpertParams& p, GroupTable* Tx = nullptr, int* done = nullptr,
                                     const ExpertParams* q2 = nullptr, W2Homes* Hx = nullptr) {
  extern __shared__ __align__(16) uint8_t dyn[];
  constexpr int NB = MX ? kHidden / 32 : kHidden / 16;
  const int rows = p.rows < 1 ? 1 : p.rows;
  float* xsc = reinterpret_cast<float*>(dyn + size_t(rows) * kHidden * 2);
  __shared__ GroupTable Town;
  GroupTable& T = Fused ? *Tx : Town;
  __shared__ float xamax[kMaxRows];
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, l = lane & 15;
  // the routing: ranked here from the router pass's logits (fused: warp_top8, one row per warp; block 0 writes ids_out
  // / weights_out for w2 and the combination) or the route pass's ids
  static_assert(kMaxRows <= kFlatWarps, "one warp per row ranks");
  if (Fused) {
    __shared__ int rid[kMaxSlots];
    if (warp < rows) {
      int id;
      float sg;
      if (blockIdx.x == 0) {
        warp_top8<true>(after_wait(p.logits) + warp * kExperts, p.bias, id, sg);
        float wsum = 0.0f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) wsum = __fadd_rn(wsum, __shfl_sync(0xffffffffu, sg, k));
        if (lane < kTopK) {
          after_wait(p.ids_out)[warp * kTopK + lane] = id;
          after_wait(p.weights_out)[warp * kTopK + lane] =
              __fmul_rn(__fdiv_rn(sg, __fadd_rn(wsum, 1e-20f)), p.route_scale);
        }
      } else {
        warp_top8<false>(after_wait(p.logits) + warp * kExperts, p.bias, id, sg);
      }
      if (lane < kTopK) rid[warp * kTopK + lane] = id;
    }
    __syncthreads();
    make_table_warp(rid, rows * kTopK, T);
  } else {
    make_table_warp(after_wait(p.ids), rows * kTopK, T);
  }
  if (Fused) w2_homes_fwd(*q2, T, *Hx);       // the w2 phase's homes (off the phase change)
  const int rowbytes = MX ? kHidden : kHidden / 2;
  const uint8_t* codes = static_cast<const uint8_t*>(p.codes);
  const uint8_t* scales = static_cast<const uint8_t*>(p.scales);
  auto load = [&](RowShare<NB, 16, MX>& rs, int q) {
    const size_t row = size_t(T.expert[q / kInter]) * 2 * kInter + 2 * (q % kInter) + (lane >> 4);
    rs.load(codes + row * rowbytes, scales + row * NB, l);
  };
  RowShare<NB, 16, MX> buf;
  // the staged inputs (the previous grid's output) into registers before the first shares' loads, stored to shared
  // memory once those are issued: the L2 reads do not queue behind the DRAM reads
  constexpr int kXc = kHidden * 2 / 16;                     // 16-byte chunks per staged row
  constexpr int kXPer = (kMaxRows * kXc + kFlatThreads - 1) / kFlatThreads;
  constexpr int kSPer = (kMaxRows * (kHidden / 16) + kFlatThreads - 1) / kFlatThreads;
  uint4 xv[kXPer];
  float sv[kSPer], av = 0.0f;
  {
    const uint4* src = static_cast<const uint4*>(after_wait(p.xq));
    const float* ssrc = after_wait(p.xq_scale);
#pragma unroll
    for (int k = 0; k < kXPer; ++k) {
      const int i = t + k * kFlatThreads;
      if (i < rows * kXc) xv[k] = src[i];
    }
#pragma unroll
    for (int k = 0; k < kSPer; ++k) {
      const int i = t + k * kFlatThreads;
      if (i < rows * (kHidden / 16)) sv[k] = ssrc[i];
    }
    if (t < rows) av = after_wait(p.xq_amax)[t];
  }
  {
    const int qf = (blockIdx.x * kFlatWarps + warp) * kW13Chunk;   // the warp's first (static) chunk
    if (qf < T.count * kInter) load(buf, qf);
  }
  using D = Decoded<NB, 16, MX>;
  constexpr int kRowChunks = kHidden * 2 / 16;
  {                         // the staged rows in the chunked layout (stage_chunks' order), the scales, the amax
    uint4* dst = reinterpret_cast<uint4*>(dyn);
#pragma unroll
    for (int k = 0; k < kXPer; ++k) {
      const int i = t + k * kFlatThreads, j = i / kXc, c = i % kXc;
      if (i < rows * kXc) dst[j * kXc + (c % D::kChunks) * NB + c / D::kChunks] = xv[k];
    }
#pragma unroll
    for (int k = 0; k < kSPer; ++k) {
      const int i = t + k * kFlatThreads;
      if (i < rows * (kHidden / 16)) xsc[i] = sv[k];
    }
    if (t < rows) xamax[t] = av;
  }
  __syncthreads();
  uint16_t* act = after_wait(p.act);
  int gc = -1;
  float we = 1.0f;
  // the activations of up to 32 (slot, pair) results, one per lane, computed and stored together
  float pg = 0.0f, pu = 0.0f;
  int pa = -1, pk = 0;
  auto flush = [&]() {
    if (pa >= 0) act[pa] = f32_to_bf16_bits(softcapped_swiglu(pg, pu, p.gate_cap, p.up_cap));
    pa = -1;
    pk = 0;
  };
  auto use = [&](const RowShare<NB, 16, MX>& cur, int q, auto refill) {
    const int g = q / kInter, pair = q % kInter;
    if (g != gc) {
      gc = g;
      we = MX ? 1.0f : __fdiv_rn(p.amax[T.expert[g]], p.denominator);
    }
    D d;
    d.set(cur);
    refill();
    const int n = T.n[g];
    // gate (lanes 0-15) / up (16-31) of slot s: lane k of the warp keeps the k-th pending pair
    auto finish = [&](float a, int s, int r) {
      const float outer = MX ? 1.0f : __fmul_rn(we, __fmul_rn(xamax[r], float(1.0 / 1536.0)));
      const float v = round_bf16(__fmul_rn(warp_sum<16>(a), outer));
      const float gv = __shfl_sync(0xffffffffu, v, 0), uv = __shfl_sync(0xffffffffu, v, 16);
      if (lane == pk) { pg = gv; pu = uv; pa = s * kInter + pair; }
      if (++pk == 32) flush();
    };
    const uint4* x = reinterpret_cast<const uint4*>(dyn);
#pragma unroll 1
    for (int j = 0; j < n; ++j) {
      const int s = T.slot[g][j], r = s / kTopK;
      finish(d.dot(l, x + r * kRowChunks, xsc + r * (kHidden / 16)), s, r);
    }
  };
  {
    // dynamic chunks: the warp's first chunk static (its first load already issued above), then chunks from the
    // counter (offset past the static ones), each next one claimed a chunk ahead so the atomic's latency hides
    constexpr int C = kW13Chunk;
    const int NW = gridDim.x * kFlatWarps, gw = blockIdx.x * kFlatWarps + warp;
    const int total = T.count * kInter;
    int q = gw * C;
    if (q < total) {
      int qend = min(total, q + C);
      int pend = lane == 0 ? atomicAdd(p.work, C) : 0;
      for (;;) {
        int qn = q + 1;
        const bool chunk_end = qn >= qend;
        if (qn >= qend) {
          qn = NW * C + __shfl_sync(0xffffffffu, pend, 0);
          if (qn < total) {
            qend = min(total, qn + C);
            pend = lane == 0 ? atomicAdd(p.work, C) : 0;
          }
        }
        use(buf, q, [&] {
          if (qn < total) load(buf, qn);
        });
        if (Fused && chunk_end) {     // the chunk's (<= 4 x 8 = 32) results stored, then counted for its group
          flush();
          __threadfence();
          __syncwarp();
          if (lane == 0) atomicAdd(done + q / kInter, C);
        }
        if (qn >= total) break;
        q = qn;
      }
    }
  }
  flush();
}

// one block per SM
template <bool MX>
__global__ void __launch_bounds__(kFlatThreads, 1) expert_w13_flat_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  expert_w13_flat_body<MX>(p);
}

// expert_act_quant: one block per slot: its activation quantized and staged (the w2 inputs of expert_w2_body).
constexpr int kActThreads = 128;
template <bool MX>
__device__ void expert_act_quant_body(const ExpertParams& p) {
  __shared__ __align__(16) uint16_t ab[kInter];
  __shared__ __align__(16) uint8_t qc[kInter];
  __shared__ uint8_t qs[kInter / 16];
  const int s = blockIdx.x;
  const uint4* a = reinterpret_cast<const uint4*>(after_wait(p.act) + size_t(s) * kInter);
  for (int i = threadIdx.x; i < kInter / 8; i += kActThreads) reinterpret_cast<uint4*>(ab)[i] = a[i];
  __syncthreads();
  uint8_t* xq = static_cast<uint8_t*>(after_wait(p.xq)) + size_t(s) * kInter * 2;
  quantize_stage_rows<MX, kActThreads>(ab, 1, kInter, reinterpret_cast<uint32_t*>(xq),
                                       after_wait(p.xq_scale) + s * (kInter / 16), reinterpret_cast<__half*>(xq),
                                       after_wait(p.xq_amax) + s, qc, qs);
  if (p.xq_codes) copy_raw<MX>(qc, qs, kInter, after_wait(p.xq_codes), after_wait(p.xq_sc8), s);
}

__global__ void __launch_bounds__(kActThreads) expert_act_quant_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_act_quant_body<true>(p);
  else expert_act_quant_body<false>(p);
}

// w2 over the flat (group, quad of output rows) items; the slots of the block's groups (at most two: a block's
// share is below one group's 576 quads), quantized by the block from the w13 activations, in shared memory. The
// table and the first loads run before the wait (the routing is two launches back).
constexpr int kQuads = kHidden / 4;
constexpr int kW2Staged = 2 * kMaxRows;
constexpr size_t kW2StagedBytes = size_t(kW2Staged) * (kInter * 2 + (kInter / 16) * 4);
// w2's dynamic-mode homes: wl[g] the first warp whose home is group g, the groups' warps in proportion to kW2Cost +
// their slots (the weighted position of warp w is Wt * w / NW); wmin the fewest warps of a group
struct W2Homes {
  int wl[kMaxSlots + 1];
  int wmin;
};
__device__ __forceinline__ void w2_homes(const ExpertParams& p, const GroupTable& T, W2Homes& H) {
  const int t = threadIdx.x, NW = gridDim.x * kFlatWarps;
  if (p.work && t <= T.count) {
    int pre = 0, wt = 0;
    for (int h = 0; h < T.count; ++h) {
      const int wh = kW2Cost + T.n[h];
      pre += h < t ? wh : 0;
      wt += wh;
    }
    H.wl[t] = int((int64_t(pre) * NW + wt - 1) / wt);
  }
  __syncthreads();
  if (t == 0) {
    int m = 1 << 30;
    if (p.work)
      for (int h = 0; h < T.count; ++h) m = min(m, H.wl[h + 1] - H.wl[h]);
    H.wmin = p.work ? m : 0;
  }
  __syncthreads();
}
__device__ void w2_homes_fwd(const ExpertParams& q, const GroupTable& T, W2Homes& H) { w2_homes(q, T, H); }
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// Fused (expert_flat_fused, after the w13 phase): the w13 phase's table (Tx), and in place of the wait each of the
// block's groups' w13 chunk count (p.done) complete (acquire)
template <bool MX, bool Fused = false>
__device__ void expert_w2_flat_body(const ExpertParams& p, const GroupTable* Tx = nullptr, const W2Homes* Hx = nullptr) {
  extern __shared__ __align__(16) uint8_t dyn[];
  constexpr int NB = MX ? kInter / 32 : kInter / 16;
  using D = Decoded<NB, 8, MX>;
  constexpr int kRowChunks = kInter * 2 / 16;
  static_assert(D::kChunks == (MX ? 4 : 2), "the in-pass quantization below writes this many chunks per block");
  float* xsc = reinterpret_cast<float*>(dyn + size_t(kW2Staged) * kInter * 2);
  __shared__ GroupTable Town;
  const GroupTable& T = Fused ? *Tx : Town;
  __shared__ float samax[kW2Staged];
  const int rows = p.rows < 1 ? 1 : p.rows;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, l = lane & 7;
  // the routing: the w13 phase's table (fused) or the route pass's ids
  if (!Fused) make_table_warp(p.ids, rows * kTopK, Town);
  int b0, b1, q0, q1;
  const int total = T.count * kQuads;
  flat_share(total, warp, b0, b1, q0, q1);
  // dynamic mode: a warp's home group is the group of its share's start (total * gw / NW); a block stages the home
  // groups of its warps (at most two when every group is the home of >= 9 warps). Each warp takes a static first
  // chunk of its home group (by its rank among the group's warps), then chunks from the group's counter (offset past
  // the static ones; zeroed by the route pass), each claimed a chunk ahead, then those of the block's other group.
  constexpr int C = kW2Chunk;
  const int gw = blockIdx.x * kFlatWarps + warp;
  // the groups' warps in proportion to their cost (kW2Cost + slots per item): wl[g] the first warp whose home is g
  __shared__ W2Homes Hown;
  if (!Fused) w2_homes(p, T, Hown);
  const W2Homes& H = Fused ? *Hx : Hown;
  const int* wl = H.wl;
  const bool dmode = p.work && H.wmin >= 9;          // every group the home of >= 9 warps (else the static shares)
  auto home = [&](int w) {
    int g = 0;
    while (g + 1 < T.count && wl[g + 1] <= w) ++g;
    return g;
  };
  auto wlo = [&](int g) { return wl[g]; };
  int g0 = b0 / kQuads, g1 = (b1 - 1) / kQuads;
  if (dmode) {
    g0 = home(blockIdx.x * kFlatWarps);
    g1 = home(blockIdx.x * kFlatWarps + kFlatWarps - 1);
  }
  const bool any = dmode || b0 < b1;
  // fused: the groups' first readiness reads before this block's first weight loads (not queued behind them)
  int rd0 = 0, rd1 = 0;
  if (Fused && t == 0 && any) {
    rd0 = ld_acquire(p.done + g0);
    rd1 = g1 != g0 ? ld_acquire(p.done + g1) : kInter;
  }
  const int rowbytes = MX ? kInter : kInter / 2;
  const uint8_t* codes = static_cast<const uint8_t*>(p.codes);
  const uint8_t* scales = static_cast<const uint8_t*>(p.scales);
  auto load = [&](RowShare<NB, 8, MX>& rs, int q) {
    const size_t row = size_t(T.expert[q / kQuads]) * kHidden + (q % kQuads) * 4 + (lane >> 3);
    rs.load(codes + row * rowbytes, scales + row * NB, l);
  };
  RowShare<NB, 8, MX> buf;
  int cg = 0, go = 0, cs = 0, os = 0, dq = -1, dqe = -1, pend = 0;
  auto next = [&](int& qq, int& qe) -> bool {      // the pending claim of the current group, then the other group
    for (;;) {
      const int base = cs + __shfl_sync(0xffffffffu, pend, 0);
      if (base < kQuads) {
        qq = cg * kQuads + base;
        qe = cg * kQuads + min(kQuads, base + C);
        pend = lane == 0 ? atomicAdd(p.work + cg, C) : 0;
        return true;
      }
      if (cg == go) return false;
      cg = go;
      cs = os;
      pend = lane == 0 ? atomicAdd(p.work + cg, C) : 0;
    }
  };
  if (dmode) {
    cg = home(gw);
    go = cg == g0 ? g1 : g0;
    const int lo = wlo(cg), k = gw - lo;
    cs = min((wlo(cg + 1) - lo) * C, kQuads);
    if (go != cg) os = go > cg ? min((wlo(go + 1) - wlo(go)) * C, kQuads) : min((lo - wlo(go)) * C, kQuads);
    pend = lane == 0 ? atomicAdd(p.work + cg, C) : 0;
    if (k * C < kQuads) {
      dq = cg * kQuads + k * C;
      dqe = cg * kQuads + min(kQuads, k * C + C);
    } else if (!next(dq, dqe)) {
      dq = -1;
    }
    if (dq >= 0) load(buf, dq);
  } else if (q0 < q1) {
    load(buf, q0);
  }
  if (Fused) {
    if (t == 0 && any) {
      while (rd0 < kInter) rd0 = ld_acquire(p.done + g0);
      while (rd1 < kInter) rd1 = ld_acquire(p.done + g1);
    }
    __syncthreads();
  } else {
    pdl_wait();
    pdl_launch();
  }
  if (!MX && any) {
    // NVFP4, one pass over the rows of all the block's groups: each thread's 16-element blocks (row, b) held packed,
    // the rows' amax as an order-free shared max, then each block quantized (quantize_stage_rows' arithmetic) and
    // written straight to the staged chunked layout (f16x2 codes, FP32 scales, amax)
    const int n0 = T.n[g0], rtot = n0 + (g1 > g0 ? T.n[g1] : 0);
    __shared__ unsigned int ramax[kW2Staged];
    if (t < kW2Staged) ramax[t] = 0u;
    __syncthreads();
    constexpr int kBlk = kInter / 16, kM = (kW2Staged * kBlk + kFlatThreads - 1) / kFlatThreads;
    const uint4* a4 = reinterpret_cast<const uint4*>(after_wait(p.act));
    uint4 xr[kM][2];
#pragma unroll
    for (int m = 0; m < kM; ++m) {
      const int i = t + m * kFlatThreads, row = i / kBlk, b = i % kBlk;
      if (row < rtot) {
        const int g = row < n0 ? g0 : g1, j = row < n0 ? row : row - n0;
        const uint4* src = a4 + size_t(T.slot[g][j]) * (kInter / 8) + 2 * b;
        xr[m][0] = __ldcg(src);
        xr[m][1] = __ldcg(src + 1);
      }
    }
#pragma unroll
    for (int m = 0; m < kM; ++m) {
      const int i = t + m * kFlatThreads, row = i / kBlk;
      if (row >= rtot) continue;
      const uint32_t w[8] = {xr[m][0].x, xr[m][0].y, xr[m][0].z, xr[m][0].w, xr[m][1].x, xr[m][1].y, xr[m][1].z, xr[m][1].w};
      float bm = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) bm = fmaxf(bm, fmaxf(fabsf(bf16lo(w[e])), fabsf(bf16hi(w[e]))));
      atomicMax(&ramax[row], __float_as_uint(bm));
    }
    __syncthreads();
    uint4* stg = reinterpret_cast<uint4*>(dyn);
#pragma unroll
    for (int m = 0; m < kM; ++m) {
      const int i = t + m * kFlatThreads, row = i / kBlk, b = i % kBlk;
      if (row >= rtot) continue;
      const int g = row < n0 ? g0 : g1, j = row < n0 ? row : row - n0;
      const int rs = (g - g0) * kMaxRows + j;                     // the staged row
      const float a = __uint_as_float(ramax[row]);
      const float encoder = a > 0.0f ? fminf(__fdiv_rn(1536.0f, a), kFltMax) : kFltMax;
      const float decoder = __fdiv_rn(1.0f, encoder);
      const float outer = __fmul_rn(a, float(1.0 / 1536.0));
      const uint32_t w[8] = {xr[m][0].x, xr[m][0].y, xr[m][0].z, xr[m][0].w, xr[m][1].x, xr[m][1].y, xr[m][1].z, xr[m][1].w};
      float x[16];
#pragma unroll
      for (int e = 0; e < 8; ++e) { x[2 * e] = bf16lo(w[e]); x[2 * e + 1] = bf16hi(w[e]); }
      uint8_t qc8[8], sc;
      quant_block_nvfp4(x, encoder, decoder, outer, qc8, sc);
      uint32_t h[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const uint16_t byte = qc8[e];
        asm("{ .reg .b8 bb; cvt.u8.u16 bb, %1; cvt.rn.f16x2.e2m1x2 %0, bb; }" : "=r"(h[e]) : "h"(byte));
      }
      stg[size_t(rs) * kRowChunks + b] = make_uint4(h[0], h[1], h[2], h[3]);
      stg[size_t(rs) * kRowChunks + NB + b] = make_uint4(h[4], h[5], h[6], h[7]);
      xsc[rs * (kInter / 16) + b] = e4m3_value(sc);
      if (b == 0) samax[rs] = a;
    }
  } else if (MX && any) {
    // MXFP8, one pass: each 32-element block (row, b) quantized from registers (quant_block_mxfp8, its own scale) and
    // written straight to the staged chunked layout (half values, FP32 scales)
    const int n0 = T.n[g0], rtot = n0 + (g1 > g0 ? T.n[g1] : 0);
    constexpr int kBlk = kInter / 32;
    const uint4* a4 = reinterpret_cast<const uint4*>(after_wait(p.act));
    uint4* stg = reinterpret_cast<uint4*>(dyn);
    for (int i = t; i < rtot * kBlk; i += kFlatThreads) {
      const int row = i / kBlk, b = i % kBlk;
      const int g = row < n0 ? g0 : g1, j = row < n0 ? row : row - n0;
      const int rs = (g - g0) * kMaxRows + j;
      const uint4* src = a4 + size_t(T.slot[g][j]) * (kInter / 8) + 4 * b;
      float x[32];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint4 v = __ldcg(src + u);
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { x[8 * u + 2 * e] = bf16lo(w[e]); x[8 * u + 2 * e + 1] = bf16hi(w[e]); }
      }
      uint8_t qc32[32], sc;
      quant_block_mxfp8(x, qc32, sc);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint32_t h[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint16_t pair = uint16_t(qc32[8 * u + 2 * e]) | (uint16_t(qc32[8 * u + 2 * e + 1]) << 8);
          asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h[e]) : "h"(pair));
        }
        stg[size_t(rs) * kRowChunks + u * NB + b] = make_uint4(h[0], h[1], h[2], h[3]);
      }
      xsc[rs * (kInter / 16) + b] = exp2f(float(int(sc) - 127));
    }
  }
  __syncthreads();
  uint16_t* out = after_wait(p.out);
  int gc = -1;
  float we = 1.0f;
  auto use = [&](const RowShare<NB, 8, MX>& cur, int q, auto refill) {
    const int g = q / kQuads, orow = (q % kQuads) * 4 + (lane >> 3);
    if (g != gc) {
      gc = g;
      we = MX ? 1.0f : __fdiv_rn(p.amax[T.expert[g]], p.denominator);
    }
    D d;
    d.set(cur);
    refill();
    const int n = T.n[g];
    const uint4* x = reinterpret_cast<const uint4*>(dyn);
    auto finish = [&](float a, int j, int r) {
      const float outer = MX ? 1.0f : __fmul_rn(we, __fmul_rn(samax[r], float(1.0 / 1536.0)));
      const float s = warp_sum<8>(a);
      if (l == 0) out[size_t(T.slot[g][j]) * kHidden + orow] = f32_to_bf16_bits(__fmul_rn(s, outer));
    };
    const int r0 = (g - g0) * kMaxRows;
#pragma unroll 1
    for (int j = 0; j < n; ++j) finish(d.dot(l, x + (r0 + j) * kRowChunks, xsc + (r0 + j) * (kInter / 16)), j, r0 + j);
  };
  if (dmode) {
    while (dq >= 0) {
      int qn = dq + 1, qe = dqe;
      if (qn >= dqe && !next(qn, qe)) qn = -1;
      use(buf, dq, [&] {
        if (qn >= 0) load(buf, qn);
      });
      dq = qn;
      dqe = qe;
    }
  } else {
    flat_run(q0, q1, buf, load, use);
  }
}

template <bool MX>
__global__ void __launch_bounds__(kFlatThreads, 1) expert_w2_flat_kernel(const __grid_constant__ ExpertParams p) {
  prefetch_share(p.prefetch);
  expert_w2_flat_body<MX>(p);
}

// w13 and w2 in one pass: the w13 phase (dynamic chunks, each chunk's activations counted per group), then in the
// same blocks the w2 phase on the same table, each block's groups awaited by their counts (no kernel boundary).
// Dependents launch once every block is past the w13 phase (all blocks resident: the waits cannot block).
template <bool MX>
__global__ void __launch_bounds__(kFlatThreads, 1)
    expert_flat_fused_kernel(const __grid_constant__ ExpertParams p, const __grid_constant__ ExpertParams q) {
  __shared__ GroupTable T;
  __shared__ W2Homes H;
  pdl_wait();
  expert_w13_flat_body<MX, true>(p, &T, q.done, &q, &H);
  __syncthreads();
  pdl_launch();
  expert_w2_flat_body<MX, true>(q, &T, &H);
}

constexpr size_t kW2Smem = size_t(kMaxRows) * kInter * 2 + size_t(kMaxRows) * (kInter / 16) * 4 +
                           size_t(kMaxRows) * kInter + size_t(kMaxRows) * (kInter / 16);
constexpr size_t kW13Smem = size_t(kMaxRows) * kHidden * 2 + size_t(kMaxRows) * (kHidden / 16) * 4 +
                            size_t(kMaxRows) * kHidden + size_t(kMaxRows) * (kHidden / 16);

}  // namespace

void expert_w13(const ExpertParams& p, cudaStream_t stream) {
  const int rows = p.rows < 1 ? 1 : p.rows;
  if (rows > kMaxRows || !p.logits || !p.x_in || !p.norm_w || !p.ids_out || !p.weights_out)
    throw std::invalid_argument("expert_w13: 1..8 rows with the folded routing (logits, bias, x_in, norm_w, ids_out, "
                                "weights_out)");
  static std::atomic<unsigned long long> smem_set{0};
  if (first_use_on_device(smem_set)) {
    check_cuda(cudaFuncSetAttribute(expert_w13_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(kW13Smem)), "expert_w13 smem");
    check_cuda(cudaFuncSetAttribute(expert_w13_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(kW13Smem)), "expert_w13 smem");
  }
  launch_pdl(p.mxfp8 ? expert_w13_kernel<true> : expert_w13_kernel<false>, dim3(kSplit, rows * kTopK),
             dim3(kExpertThreads), kW13Smem, stream, p);
}

namespace {
int sm_count() {                // of the current device
  static std::atomic<int> counts[64];
  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "device");
  int v = counts[dev & 63].load();
  if (!v) {
    check_cuda(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev), "SM count");
    counts[dev & 63].store(v);
  }
  return v;
}
// max_rows: kMaxRows for the flat weight passes, kMaxChunk for the row / slot parallel route and act quantization;
// staged: the pass reads (or writes) the staged quantized inputs xq, xq_scale, xq_amax
void check_flat(const ExpertParams& p, const char* what, int max_rows = kMaxRows, bool staged = true) {
  const int rows = p.rows < 1 ? 1 : p.rows;
  if (rows > max_rows || (staged && (!p.xq || !p.xq_scale || !p.xq_amax)))
    throw std::invalid_argument(std::string(what) + ": too many rows or missing staged inputs xq, xq_scale, xq_amax");
}
// the weights of a flat weight pass (codes, scales; NVFP4 the per-expert amax)
void check_weights(const ExpertParams& p, const char* what) {
  if (!p.codes || !p.scales || (!p.mxfp8 && !p.amax))
    throw std::invalid_argument(std::string(what) + ": codes, scales (and amax for NVFP4)");
}
}  // namespace

void expert_route(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_route", kMaxChunk);
  if (!p.logits || !p.bias || !p.x_in || !p.norm_w || !p.ids_out || !p.weights_out)
    throw std::invalid_argument("expert_route: logits, bias, x_in, norm_w, ids_out, weights_out");
  launch_pdl(expert_route_kernel, dim3(p.rows < 1 ? 1 : p.rows), dim3(kExpertThreads), 0, stream, p);
}

void expert_router_route(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_router_route", kMaxRows);
  if (!p.router_w || !p.logits_out || !p.x_in || !p.norm_w || p.logits)
    throw std::invalid_argument("expert_router_route: router_w, logits_out, x_in, norm_w (the logits are its output)");
  launch_pdl(expert_router_route_kernel, dim3(kRouterSplits + 1, p.rows < 1 ? 1 : p.rows),
             dim3(kExpertThreads), 0, stream, p);
}

void expert_w13_flat(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_w13_flat");
  check_weights(p, "expert_w13_flat");
  if (!p.ids || !p.act || !p.work || p.logits)
    throw std::invalid_argument("expert_w13_flat: the route pass's ids, act and the work counter (ranking in the "
                                "pass: expert_flat_fused)");
  const int rows = p.rows < 1 ? 1 : p.rows;
  const size_t smem = size_t(rows) * (kHidden * 2 + (kHidden / 16) * 4);
  const dim3 grid(sm_count());
  auto run = [&](auto kernel, size_t bytes) {
    if (bytes > 48 * 1024)
      check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)), "w13 flat smem");
    launch_pdl(kernel, grid, dim3(kFlatThreads), bytes, stream, p);
  };
  run(p.mxfp8 ? expert_w13_flat_kernel<true> : expert_w13_flat_kernel<false>, smem);
}

void expert_act_quant(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_act_quant", kMaxChunk);
  if (!p.act) throw std::invalid_argument("expert_act_quant: act");
  launch_pdl(expert_act_quant_kernel, dim3((p.rows < 1 ? 1 : p.rows) * kTopK), dim3(kActThreads), 0, stream, p);
}

void expert_w2_flat(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_w2_flat", kMaxRows, false);
  check_weights(p, "expert_w2_flat");
  if (sm_count() < kMaxSlots) throw std::invalid_argument("expert_w2_flat: needs at least 64 SMs");
  if (!p.ids || !p.act || !p.out || p.logits)
    throw std::invalid_argument("expert_w2_flat: the route pass's ids, act (quantized here) and out");
  static_assert(kW2StagedBytes <= 48 * 1024, "w2 staging fits the default shared memory");
  launch_pdl(p.mxfp8 ? expert_w2_flat_kernel<true> : expert_w2_flat_kernel<false>,
             dim3(sm_count()), dim3(kFlatThreads), kW2StagedBytes, stream, p);
}

void expert_flat_fused(const ExpertParams& p, const ExpertParams& q, cudaStream_t stream) {
  check_flat(p, "expert_flat_fused");
  check_flat(q, "expert_flat_fused (w2)", kMaxRows, false);
  check_weights(p, "expert_flat_fused (w13)");
  check_weights(q, "expert_flat_fused (w2)");
  if (!p.logits || !p.bias || !p.ids_out || !p.weights_out || !p.act || !p.work)
    throw std::invalid_argument("expert_flat_fused: w13 with the router logits, bias, ids_out, weights_out, act, work");
  if (!q.done || !q.act || !q.out || !q.work || q.rows != p.rows || q.mxfp8 != p.mxfp8)
    throw std::invalid_argument("expert_flat_fused: w2 with done counters, act, out, work and w13's rows / format");
  if (sm_count() < kMaxSlots) throw std::invalid_argument("expert_flat_fused: needs at least 64 SMs");
  const int rows = p.rows < 1 ? 1 : p.rows;
  const size_t s13 = size_t(rows) * (kHidden * 2 + (kHidden / 16) * 4);
  const size_t smem = s13 > kW2StagedBytes ? s13 : kW2StagedBytes;
  const auto kernel = p.mxfp8 ? expert_flat_fused_kernel<true> : expert_flat_fused_kernel<false>;
  // the w2 phase waits on counts that other blocks' w13 phases complete: one block per SM, all resident at once
  // (the whole GPU: not under an SM partition such as MPS limits or green contexts)
  int per_sm = 0;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kFlatThreads, smem), "fused occupancy");
  if (per_sm < 1) throw std::runtime_error("expert_flat_fused: one block per SM does not fit");
  launch_pdl(kernel, dim3(sm_count()), dim3(kFlatThreads), smem, stream, p, q);
}

void expert_w2(const ExpertParams& p, cudaStream_t stream) {
  const int rows = p.rows < 1 ? 1 : p.rows;
  if (rows > kMaxRows) throw std::invalid_argument("expert_w2: at most 8 rows");
  launch_pdl(p.mxfp8 ? expert_w2_kernel<true> : expert_w2_kernel<false>, dim3(kSplit, rows * kTopK),
             dim3(kExpertThreads), kW2Smem, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
