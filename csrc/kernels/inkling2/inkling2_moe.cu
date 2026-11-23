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
// loads. Several rows (verification): expert_route, expert_w13_flat, expert_act_quant, expert_w2_flat (flat passes
// over the groups of distinct experts, the same per-row arithmetic).

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr float kFltMax = 3.4028234663852886e38f;

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

__device__ __forceinline__ float softcapped_swiglu(float g, float u, float gcap, float ucap) {
  const float gc = tanhf(g / gcap) * gcap;
  const float uc = tanhf(u / ucap) * ucap;
  return gc * (1.0f / (1.0f + expf(-g))) * uc;
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
        uint2 v = ok ? *reinterpret_cast<const uint2*>(codes + 8 * b) : uint2{};
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
  if (blockIdx.x == 0 && blockIdx.y == 0 && t < rows) {
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
  route_rows(p, rows, ids, sc);
  make_groups(ids, rows * kTopK, blockIdx.y, G);
  if (int(blockIdx.y) >= G.count) return;
  const int e = G.expert, n = G.n;
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
}

__global__ void __launch_bounds__(kExpertThreads) expert_w13_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_w13_body<true>(p);
  else expert_w13_body<false>(p);
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
  const int* ids = after_wait(p.ids);
  make_groups(ids, rows * kTopK, blockIdx.y, G);
  if (int(blockIdx.y) >= G.count) return;
  const int e = G.expert, n = G.n;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, l = lane & 7;
  const int rowbytes = MX ? kInter : kInter / 2, sbytes = NB;
  const uint8_t* codes = static_cast<const uint8_t*>(p.codes) + size_t(e) * kHidden * rowbytes;
  const uint8_t* scales = static_cast<const uint8_t*>(p.scales) + size_t(e) * kHidden * sbytes;
  const int lo = blockIdx.x * kHidden / kSplit, hi = (blockIdx.x + 1) * kHidden / kSplit;
  RowShare<NB, 8, MX> cur, nxt;
  int row = lo + warp * 4 + (lane >> 3);
  if (row < hi) cur.load(codes + size_t(row) * rowbytes, scales + size_t(row) * sbytes, l);
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

__global__ void __launch_bounds__(kExpertThreads) expert_w2_kernel(const __grid_constant__ ExpertParams p) {
  prefetch_share(p.prefetch);
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_w2_body<true>(p);
  else expert_w2_body<false>(p);
}

// ------------------------------------------------------------------------- flat passes over the expert groups
// Several rows (a verification step): the routing, the input norm and every quantization run once per row / slot
// (expert_route, expert_act_quant, staged in global memory), and the weight passes split the flat list of
// (group, output row) items evenly over one block per SM, so every weight byte is read once and every SM streams.
// The per-row arithmetic is the one of expert_w13_body / expert_w2_body (same lane shares, accumulation order,
// scales), so each row equals the fused kernels' result.

// The groups of make_groups for all of them at once, built in parallel: count, expert, slots per group.
struct GroupTable {
  int count;
  int expert[kMaxSlots];
  int n[kMaxSlots];
  int slot[kMaxSlots][kMaxRows];
};

__device__ __forceinline__ void make_table(const int* ids, int slots, GroupTable& T) {
  __shared__ unsigned long long emask[kExperts];      // the slots routed to each expert
  __shared__ unsigned long long lead;                 // the slots that first route to their expert
  const int t = threadIdx.x;
  if (t < kExperts) emask[t] = 0ull;
  if (t == 0) lead = 0ull;
  __syncthreads();
  int e = 0;
  if (t < slots) {
    e = ids[t];
    atomicOr(&emask[e], 1ull << t);
  }
  __syncthreads();
  unsigned long long m = 0ull, below = (1ull << (t & 63)) - 1ull;
  int f = 0;
  if (t < slots) {
    m = emask[e];
    f = __ffsll(static_cast<long long>(m)) - 1;
    if (f == t) atomicOr(&lead, 1ull << t);
  }
  __syncthreads();
  if (t < slots) {
    const unsigned long long L = lead;
    const int g = __popcll(L & ((1ull << f) - 1ull));
    if (f == t) {
      T.expert[g] = e;
      T.n[g] = __popcll(m);
    }
    T.slot[g][__popcll(m & below)] = t;
    if (t == 0) T.count = __popcll(L);
  }
  __syncthreads();
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
  const int r = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const float* lg = after_wait(p.logits) + r * kExperts;
  if (t < kExperts) bs[t] = 1.0f / (1.0f + expf(-lg[t])) + p.bias[t];
  const float4* x4 = reinterpret_cast<const float4*>(after_wait(p.x_in)) + r * (kHidden / 4);
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
  uint8_t* xq = static_cast<uint8_t*>(after_wait(p.xq)) + size_t(r) * kHidden * 2;
  quantize_stage_rows<MX>(xb, 1, kHidden, reinterpret_cast<uint32_t*>(xq), after_wait(p.xq_scale) + r * (kHidden / 16),
                          reinterpret_cast<__half*>(xq), after_wait(p.xq_amax) + r, qc, qs);
}

__global__ void __launch_bounds__(kExpertThreads) expert_route_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_route_body<true>(p);
  else expert_route_body<false>(p);
}

constexpr int kFlatThreads = 256;       // one block per SM; the decoded share needs the registers of 256 threads
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
template <bool MX>
__device__ void expert_w13_flat_body(const ExpertParams& p) {
  extern __shared__ __align__(16) uint8_t dyn[];
  constexpr int NB = MX ? kHidden / 32 : kHidden / 16;
  const int rows = p.rows < 1 ? 1 : p.rows;
  float* xsc = reinterpret_cast<float*>(dyn + size_t(rows) * kHidden * 2);
  __shared__ GroupTable T;
  __shared__ float xamax[kMaxRows];
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, l = lane & 15;
  make_table(after_wait(p.ids), rows * kTopK, T);
  int b0, b1, q0, q1;
  flat_share(T.count * kInter, warp, b0, b1, q0, q1);
  const int rowbytes = MX ? kHidden : kHidden / 2;
  const uint8_t* codes = static_cast<const uint8_t*>(p.codes);
  const uint8_t* scales = static_cast<const uint8_t*>(p.scales);
  auto load = [&](RowShare<NB, 16, MX>& rs, int q) {
    const size_t row = size_t(T.expert[q / kInter]) * 2 * kInter + 2 * (q % kInter) + (lane >> 4);
    rs.load(codes + row * rowbytes, scales + row * NB, l);
  };
  RowShare<NB, 16, MX> buf;
  if (q0 < q1) load(buf, q0);
  using D = Decoded<NB, 16, MX>;
  constexpr int kRowChunks = kHidden * 2 / 16;
  stage_chunks<NB, D::kChunks, kFlatThreads, kMaxRows>(static_cast<const uint4*>(after_wait(p.xq)),
                                                       reinterpret_cast<uint4*>(dyn), rows, [](int j) { return j; });
  const float* ssrc = after_wait(p.xq_scale);
  for (int i = t; i < rows * (kHidden / 16); i += kFlatThreads) xsc[i] = ssrc[i];
  if (t < rows) xamax[t] = after_wait(p.xq_amax)[t];
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
  flat_run(q0, q1, buf, load, [&](const RowShare<NB, 16, MX>& cur, int q, auto refill) {
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
  });
  flush();
}

__global__ void __launch_bounds__(kFlatThreads, 1) expert_w13_flat_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_w13_flat_body<true>(p);
  else expert_w13_flat_body<false>(p);
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
}

__global__ void __launch_bounds__(kActThreads) expert_act_quant_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_act_quant_body<true>(p);
  else expert_act_quant_body<false>(p);
}

// w2 over the flat (group, quad of output rows) items; the staged slots of the block's groups (at most two: a
// block's share is below one group's 576 quads) in shared memory. The table and the first loads run before the
// wait (the routing is two launches back), under expert_act_quant.
constexpr int kQuads = kHidden / 4;
constexpr int kW2Staged = 2 * kMaxRows;
template <bool MX>
__device__ void expert_w2_flat_body(const ExpertParams& p) {
  extern __shared__ __align__(16) uint8_t dyn[];
  constexpr int NB = MX ? kInter / 32 : kInter / 16;
  using D = Decoded<NB, 8, MX>;
  constexpr int kRowChunks = kInter * 2 / 16;
  float* xsc = reinterpret_cast<float*>(dyn + size_t(kW2Staged) * kInter * 2);
  __shared__ GroupTable T;
  __shared__ float samax[kW2Staged];
  const int rows = p.rows < 1 ? 1 : p.rows;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, l = lane & 7;
  make_table(p.ids, rows * kTopK, T);
  int b0, b1, q0, q1;
  flat_share(T.count * kQuads, warp, b0, b1, q0, q1);
  const int g0 = b0 / kQuads;
  const int rowbytes = MX ? kInter : kInter / 2;
  const uint8_t* codes = static_cast<const uint8_t*>(p.codes);
  const uint8_t* scales = static_cast<const uint8_t*>(p.scales);
  auto load = [&](RowShare<NB, 8, MX>& rs, int q) {
    const size_t row = size_t(T.expert[q / kQuads]) * kHidden + (q % kQuads) * 4 + (lane >> 3);
    rs.load(codes + row * rowbytes, scales + row * NB, l);
  };
  RowShare<NB, 8, MX> buf;
  if (q0 < q1) load(buf, q0);
  pdl_wait();
  pdl_launch();
  if (b0 < b1) {
    const int g1 = (b1 - 1) / kQuads;
    const uint4* src = static_cast<const uint4*>(after_wait(p.xq));
    const float* ssrc = after_wait(p.xq_scale);
    for (int g = g0; g <= g1; ++g) {
      const int base = (g - g0) * kMaxRows;
      stage_chunks<NB, D::kChunks, kFlatThreads, kMaxRows>(src, reinterpret_cast<uint4*>(dyn) + base * kRowChunks, T.n[g],
                                                           [&](int j) { return T.slot[g][j]; });
      for (int i = t; i < T.n[g] * (kInter / 16); i += kFlatThreads)
        xsc[base * (kInter / 16) + i] = ssrc[T.slot[g][i / (kInter / 16)] * (kInter / 16) + i % (kInter / 16)];
      if (t < T.n[g]) samax[base + t] = after_wait(p.xq_amax)[T.slot[g][t]];
    }
  }
  __syncthreads();
  uint16_t* out = after_wait(p.out);
  int gc = -1;
  float we = 1.0f;
  flat_run(q0, q1, buf, load, [&](const RowShare<NB, 8, MX>& cur, int q, auto refill) {
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
  });
}

__global__ void __launch_bounds__(kFlatThreads, 1) expert_w2_flat_kernel(const __grid_constant__ ExpertParams p) {
  prefetch_share(p.prefetch);
  if (p.mxfp8) expert_w2_flat_body<true>(p);
  else expert_w2_flat_body<false>(p);
}

constexpr size_t kW2Smem = size_t(kMaxRows) * kInter * 2 + size_t(kMaxRows) * (kInter / 16) * 4 +
                           size_t(kMaxRows) * kInter + size_t(kMaxRows) * (kInter / 16);
constexpr size_t kW13Smem = size_t(kMaxRows) * kHidden * 2 + size_t(kMaxRows) * (kHidden / 16) * 4 +
                            size_t(kMaxRows) * kHidden + size_t(kMaxRows) * (kHidden / 16);

}  // namespace

void expert_w13(const ExpertParams& p, cudaStream_t stream) {
  const int rows = p.rows < 1 ? 1 : p.rows;
  if (rows > kMaxRows || !p.logits) throw std::invalid_argument("expert_w13: 1..8 rows with the folded routing");
  static bool once = [] {
    check_cuda(cudaFuncSetAttribute(expert_w13_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kW13Smem)),
               "expert_w13 smem");
    return true;
  }();
  (void)once;
  launch_pdl(expert_w13_kernel, dim3(kSplit, rows * kTopK), dim3(kExpertThreads), kW13Smem, stream, p);
}

namespace {
int sm_count() {
  static const int n = [] {
    int dev = 0, v = 0;
    check_cuda(cudaGetDevice(&dev), "device");
    check_cuda(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev), "SM count");
    return v;
  }();
  return n;
}
void check_flat(const ExpertParams& p, const char* what) {
  const int rows = p.rows < 1 ? 1 : p.rows;
  if (rows > kMaxRows || !p.xq || !p.xq_scale || !p.xq_amax)
    throw std::invalid_argument(std::string(what) + ": 1..8 rows and the staged inputs xq, xq_scale, xq_amax");
}
}  // namespace

void expert_route(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_route");
  if (!p.logits || !p.bias || !p.x_in || !p.norm_w || !p.ids_out || !p.weights_out)
    throw std::invalid_argument("expert_route: logits, bias, x_in, norm_w, ids_out, weights_out");
  launch_pdl(expert_route_kernel, dim3(p.rows < 1 ? 1 : p.rows), dim3(kExpertThreads), 0, stream, p);
}

void expert_w13_flat(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_w13_flat");
  const int rows = p.rows < 1 ? 1 : p.rows;
  const size_t smem = size_t(rows) * (kHidden * 2 + (kHidden / 16) * 4);
  launch_pdl(expert_w13_flat_kernel, dim3(sm_count()), dim3(kFlatThreads), smem, stream, p);
}

void expert_act_quant(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_act_quant");
  if (!p.act) throw std::invalid_argument("expert_act_quant: act");
  launch_pdl(expert_act_quant_kernel, dim3((p.rows < 1 ? 1 : p.rows) * kTopK), dim3(kActThreads), 0, stream, p);
}

void expert_w2_flat(const ExpertParams& p, cudaStream_t stream) {
  check_flat(p, "expert_w2_flat");
  if (sm_count() < kMaxSlots) throw std::invalid_argument("expert_w2_flat: needs at least 64 SMs");
  const size_t smem = size_t(kW2Staged) * (kInter * 2 + (kInter / 16) * 4);
  launch_pdl(expert_w2_flat_kernel, dim3(sm_count()), dim3(kFlatThreads), smem, stream, p);
}

void expert_w2(const ExpertParams& p, cudaStream_t stream) {
  const int rows = p.rows < 1 ? 1 : p.rows;
  if (rows > kMaxRows) throw std::invalid_argument("expert_w2: at most 8 rows");
  launch_pdl(expert_w2_kernel, dim3(kSplit, rows * kTopK), dim3(kExpertThreads), kW2Smem, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
