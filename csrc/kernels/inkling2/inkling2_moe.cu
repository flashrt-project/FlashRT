// SPDX-License-Identifier: Apache-2.0
//
// Batch-one routed experts as the checkpoint declares them (hf_quant_config.json): the expert inputs are quantized
// per token, NVFP4 (E2M1, E4M3 scale per 16, scale chosen per block between max/6 and max/4 by the smaller summed
// absolute error, "fouroversix_mae", global amax / 1536) in layers 0-19 and MXFP8 (E4M3, UE8M0 scale per 32) in
// layers 20-23, and multiply the exact quantized weights. Block products are exact (FP16 pair sums of E2M1 x E2M1
// products stay below 2^9, FP32 for E4M3); scales and the block sums accumulate in FP32.
//   router:  logits = W_r x (x = bf16(rms_norm(in) * w)), sigmoid + bias top-8, weights renormalized x route_scale;
//            the last block quantizes x for the experts.
//   w13:     per routed slot, gate/up rows interleaved: act = bf16(softcapped_swiglu(bf16(g), bf16(u))).
//   w2:      per slot, the act quantized per slot, y = bf16(W2 act).

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

// Quantize a row of n FP32 values held in shared memory with the block's threads (n / 16 or n / 32 blocks).
// NVFP4: codes [n / 2], scales [n / 16], *amax; MXFP8: codes [n], scales [n / 32].
template <int Threads>
__device__ void quantize_row(const float* x, int n, int mx, uint8_t* codes, uint8_t* scales, float* amax_out,
                             float* scratch) {
  if (mx) {
    for (int b = threadIdx.x; b < n / 32; b += Threads) quant_block_mxfp8(x + 32 * b, codes + 32 * b, scales[b]);
    return;
  }
  float m = 0.0f;
  for (int i = threadIdx.x; i < n; i += Threads) m = fmaxf(m, fabsf(x[i]));
  m = warp_max(m);
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = m;
  __syncthreads();
  float amax = 0.0f;
#pragma unroll
  for (int w = 0; w < Threads / 32; ++w) amax = fmaxf(amax, scratch[w]);
  __syncthreads();
  const float encoder = amax > 0.0f ? fminf(__fdiv_rn(1536.0f, amax), kFltMax) : kFltMax;
  const float decoder = __fdiv_rn(1.0f, encoder);
  const float outer = __fmul_rn(amax, float(1.0 / 1536.0));
  for (int b = threadIdx.x; b < n / 16; b += Threads) quant_block_nvfp4(x + 16 * b, encoder, decoder, outer, codes + 8 * b, scales[b]);
  if (threadIdx.x == 0) *amax_out = amax;
}

// ---------------------------------------------------------------------------------------------------- router
constexpr int kRouterBlocks = kExperts / 8;

__global__ void __launch_bounds__(256) router_kernel(const __grid_constant__ RouterParams p) {
  __shared__ __align__(16) float xs[kHidden];
  __shared__ float scratch[8];
  __shared__ int last;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  const uint4* w = reinterpret_cast<const uint4*>(p.weight) + size_t(row) * (kHidden / 8);
  pdl_wait();
  pdl_launch();
  const float* in = after_wait(p.in);
  float ss = 0.0f;
  for (int i = threadIdx.x; i < kHidden; i += 256) {
    xs[i] = in[i];
    ss = fmaf(xs[i], xs[i], ss);
  }
  ss = warp_sum(ss);
  if (lane == 0) scratch[warp] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += scratch[i];
  const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
  __syncthreads();
  for (int i = threadIdx.x; i < kHidden; i += 256) xs[i] = round_bf16(xs[i] * inv * bf16_to_f32(p.norm_w[i]));
  __syncthreads();
  float acc = 0.0f;
  for (int i = lane; i < kHidden / 8; i += 32) {
    const uint4 v = ldg_stream(w + i);
    const uint32_t ww[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc = fmaf(bf16lo(ww[j]), xs[8 * i + 2 * j], acc);
      acc = fmaf(bf16hi(ww[j]), xs[8 * i + 2 * j + 1], acc);
    }
  }
  acc = warp_sum(acc);
  float* logits = after_wait(p.logits);
  if (lane == 0) logits[row] = acc;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(after_wait(p.counter), 1) == kRouterBlocks - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (warp == 0) {
    // top-8 of sigmoid(logit) + bias (first index on ties), weights = scores / sum * route_scale
    float sc[4], bi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int e = lane + 32 * i;
      sc[i] = 1.0f / (1.0f + expf(-logits[e]));
      bi[i] = sc[i] + p.bias[e];
    }
    int ids[kTopK];
    float ws[kTopK];
    float wsum = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      float best = -INFINITY;
      int bidx = 1 << 30;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (bi[i] > best) { best = bi[i]; bidx = lane + 32 * i; }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
        if (ob > best || (ob == best && oi < bidx)) { best = ob; bidx = oi; }
      }
      ids[k] = bidx;
      const int owner = bidx & 31, slot = bidx >> 5;
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) if (i == slot) s = sc[i];
      s = __shfl_sync(0xffffffffu, s, owner);
      ws[k] = s;
      wsum = __fadd_rn(wsum, s);
      if (lane == owner) {
#pragma unroll
        for (int i = 0; i < 4; ++i) if (i == slot) bi[i] = -INFINITY;
      }
    }
    if (lane == 0) {
      int* oid = after_wait(p.ids);
      float* ow = after_wait(p.weights);
      for (int k = 0; k < kTopK; ++k) {
        oid[k] = ids[k];
        ow[k] = __fmul_rn(__fdiv_rn(ws[k], __fadd_rn(wsum, 1e-20f)), p.route_scale);
      }
    }
  }
  __syncthreads();
  quantize_row<256>(xs, kHidden, p.mxfp8, after_wait(p.x_codes), after_wait(p.x_scales), after_wait(p.x_amax), scratch);
  if (threadIdx.x == 0) *after_wait(p.counter) = 0;
}

// ------------------------------------------------------------------------------------------------ experts
__device__ __forceinline__ void e2m1x8_to_h2(uint2 codes, uint32_t (&out)[8]) {
  asm("{ .reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
      "mov.b32 {b0, b1, b2, b3}, %8;\n mov.b32 {b4, b5, b6, b7}, %9;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n cvt.rn.f16x2.e2m1x2 %4, b4;\n cvt.rn.f16x2.e2m1x2 %5, b5;\n"
      "cvt.rn.f16x2.e2m1x2 %6, b6;\n cvt.rn.f16x2.e2m1x2 %7, b7;\n }"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]), "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
      : "r"(codes.x), "r"(codes.y));
}

// Exact dot of one 16-element NVFP4 block (weights) with the quantized input codes as f16x2 (xh[8]).
__device__ __forceinline__ float block_dot_nvfp4(uint2 wcodes, const uint32_t* xh) {
  uint32_t wh[8];
  e2m1x8_to_h2(wcodes, wh);
  __half2 acc = __float2half2_rn(0.0f);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    acc = __hfma2(*reinterpret_cast<const __half2*>(&wh[i]), *reinterpret_cast<const __half2*>(&xh[i]), acc);
  const float2 f = __half22float2(acc);
  return f.x + f.y;
}

// 32 FP8 products of one MXFP8 block: weight codes (32 bytes, 2 uint4) x input values (FP32 in shared memory).
__device__ __forceinline__ float block_dot_mxfp8(uint4 w0, uint4 w1, const float* x) {
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

// Stage the quantized input in shared memory: NVFP4 codes as f16x2 pairs + FP32 block scales; MXFP8 values FP32.
template <int Threads>
__device__ void stage_input(int k, int mx, const uint8_t* codes, const uint8_t* scales, uint32_t* xh, float* xsc,
                            float* xv) {
  if (mx) {
    for (int i = threadIdx.x; i < k; i += Threads) xv[i] = e4m3_value(codes[i]);
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
  __syncthreads();
}

// One weight row's share for lane l of an L-lane group: blocks l, l + L, ... (NB blocks per row). NVFP4 blocks are
// 16 inputs (8 code bytes, one E4M3 scale), MXFP8 blocks 32 inputs (32 code bytes, one UE8M0 scale). The loads are
// issued together (load), the products follow (dot), so a row's whole share is in flight at once.
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
  __device__ __forceinline__ float dot(int l, const uint32_t* xh, const float* xsc, const float* xv) const {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int b = l + i * L;
      if ((NB % L) && b >= NB) break;
      if (MX) {
        const float sw = exp2f(float(int(s[i]) - 127));
        acc = fmaf(block_dot_mxfp8(c[2 * i], c[2 * i + 1], xv + 32 * b), __fmul_rn(sw, xsc[b]), acc);
      } else {
        const uint2 wc = (i & 1) ? make_uint2(c[i / 2].z, c[i / 2].w) : make_uint2(c[i / 2].x, c[i / 2].y);
        acc = fmaf(block_dot_nvfp4(wc, xh + 8 * b), __fmul_rn(e4m3_value(s[i]), xsc[b]), acc);
      }
    }
    return acc;
  }
};

// One block per SM: grid (kSplit, kTopK), 512 threads; slot rows split evenly over the kSplit blocks.
constexpr int kSplit = 21;
constexpr int kExpertThreads = 512;
constexpr int kExpertWarps = kExpertThreads / 32;

// w13: a warp runs gate/up pairs (half-warps own the two rows of a pair), the next pair's loads in flight.
// Folded routing (p.logits): top-8 by rank of sigmoid(logit) + bias (first index on ties), weights renormalized x
// route_scale in rank order (block (0, 0) writes them). Returns the slot's expert; the block's x_in share is loaded
// into xr first so its latency overlaps the selection.
__device__ __forceinline__ int route_select(const ExpertParams& p, int slot, float4 (&xr)[2]) {
  __shared__ int ids[kTopK];
  __shared__ float sc[kTopK];
  __shared__ float bs[kExperts];
  const int t = threadIdx.x;
  const float4* x4 = reinterpret_cast<const float4*>(after_wait(p.x_in));
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = t + r * kExpertThreads;
    xr[r] = i < kHidden / 4 ? x4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const float* lg = after_wait(p.logits);
  float s = 0.0f, b = 0.0f;
  if (t < kExperts) {
    s = 1.0f / (1.0f + expf(-lg[t]));
    b = s + p.bias[t];
    bs[t] = b;
  }
  __syncthreads();
  if (t < kExperts) {
    int rank = 0;
#pragma unroll 8
    for (int j = 0; j < kExperts; ++j) {
      const float bj = bs[j];
      rank += (bj > b) || (bj == b && j < t);
    }
    if (rank < kTopK) { ids[rank] = t; sc[rank] = s; }
  }
  __syncthreads();
  if (slot == 0 && blockIdx.x == 0 && t == 0) {
    float wsum = 0.0f;
    for (int k = 0; k < kTopK; ++k) wsum = __fadd_rn(wsum, sc[k]);
    int* oid = after_wait(p.ids_out);
    float* ow = after_wait(p.weights_out);
    for (int k = 0; k < kTopK; ++k) {
      oid[k] = ids[k];
      ow[k] = __fmul_rn(__fdiv_rn(sc[k], __fadd_rn(wsum, 1e-20f)), p.route_scale);
    }
  }
  return ids[slot];
}

// The MLP input normalized (bf16(rms_norm(x) * w)) and quantized into shared memory (codes qc, scales qs, *amax).
template <bool MX>
__device__ void norm_and_quantize(const ExpertParams& p, const float4 (&xr)[2], uint8_t* qc, uint8_t* qs, float* xs,
                                  float* amax, float* scratch) {
  const int t = threadIdx.x;
  float ss = 0.0f;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float4 v = xr[r];
    ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
  }
  ss = warp_sum(ss);
  if ((t & 31) == 0) scratch[t >> 5] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int w = 0; w < kExpertWarps; ++w) tot += scratch[w];
  const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
  __syncthreads();
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = t + r * kExpertThreads;
    if (i < kHidden / 4) {
      const uint2 nw = reinterpret_cast<const uint2*>(p.norm_w)[i];
      const float4 v = xr[r];
      xs[4 * i] = round_bf16(v.x * inv * bf16lo(nw.x));
      xs[4 * i + 1] = round_bf16(v.y * inv * bf16hi(nw.x));
      xs[4 * i + 2] = round_bf16(v.z * inv * bf16lo(nw.y));
      xs[4 * i + 3] = round_bf16(v.w * inv * bf16hi(nw.y));
    }
  }
  __syncthreads();
  quantize_row<kExpertThreads>(xs, kHidden, MX, qc, qs, amax, scratch);
  __syncthreads();
}

template <bool MX>
__device__ void expert_w13_body(const ExpertParams& p) {
  __shared__ uint32_t xh[kHidden / 2];
  __shared__ float xsc[kHidden / 16];
  __shared__ float xv[MX ? kHidden : 1];
  __shared__ __align__(16) float xs[kHidden];
  __shared__ uint8_t qc[MX ? kHidden : kHidden / 2];
  __shared__ uint8_t qs[MX ? kHidden / 32 : kHidden / 16];
  __shared__ float scratch[kExpertWarps];
  __shared__ float amax_s;
  constexpr int NB = MX ? kHidden / 32 : kHidden / 16;
  const int slot = blockIdx.y;
  const bool fold = p.logits != nullptr;
  float4 xr[2];
  const int e = fold ? route_select(p, slot, xr) : after_wait(p.ids)[slot];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, l = lane & 15;
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
  if (fold) {
    norm_and_quantize<MX>(p, xr, qc, qs, xs, &amax_s, scratch);   // while the first rows' weights load
    stage_input<kExpertThreads>(kHidden, MX, qc, qs, xh, xsc, xv);
  }
  else stage_input<kExpertThreads>(kHidden, MX, after_wait(p.x_codes), after_wait(p.x_scales), xh, xsc, xv);
  const float xa = fold ? amax_s : *after_wait(p.x_amax);
  const float outer = MX ? 1.0f : __fmul_rn(__fdiv_rn(p.amax[e], p.denominator), __fmul_rn(xa, float(1.0 / 1536.0)));
  uint16_t* act = after_wait(p.act) + slot * kInter;
  for (; pair < hi; pair += kExpertWarps) {
    const int next = pair + kExpertWarps;
    if (next < hi) {
      const int row = 2 * next + (lane >> 4);
      nxt.load(codes + size_t(row) * rowbytes, scales + size_t(row) * sbytes, l);
    }
    const float acc = warp_sum<16>(cur.dot(l, xh, xsc, xv));
    const float v = round_bf16(__fmul_rn(acc, outer));
    const float up = __shfl_sync(0xffffffffu, v, 16);
    if (lane == 0) act[pair] = f32_to_bf16_bits(softcapped_swiglu(v, up, p.gate_cap, p.up_cap));
    cur = nxt;
  }
}

__global__ void __launch_bounds__(kExpertThreads) expert_w13_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_w13_body<true>(p);
  else expert_w13_body<false>(p);
}

// w2: a warp runs groups of 4 rows (8 lanes per row), the next group's loads in flight; the slot's activation is
// quantized per block while the first rows' weights load.
template <bool MX>
__device__ void expert_w2_body(const ExpertParams& p) {
  __shared__ float act[kInter];
  __shared__ uint8_t qc[kInter];
  __shared__ uint8_t qs[kInter / 16];
  __shared__ uint32_t xh[kInter / 2];
  __shared__ float xsc[kInter / 16];
  __shared__ float xv[MX ? kInter : 1];
  __shared__ float scratch[kExpertWarps];
  __shared__ float amax;
  constexpr int NB = MX ? kInter / 32 : kInter / 16;
  const int slot = blockIdx.y;
  const int e = after_wait(p.ids)[slot];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, l = lane & 7;
  const int rowbytes = MX ? kInter : kInter / 2, sbytes = NB;
  const uint8_t* codes = static_cast<const uint8_t*>(p.codes) + size_t(e) * kHidden * rowbytes;
  const uint8_t* scales = static_cast<const uint8_t*>(p.scales) + size_t(e) * kHidden * sbytes;
  const int lo = blockIdx.x * kHidden / kSplit, hi = (blockIdx.x + 1) * kHidden / kSplit;
  RowShare<NB, 8, MX> cur, nxt;
  int row = lo + warp * 4 + (lane >> 3);
  if (row < hi) cur.load(codes + size_t(row) * rowbytes, scales + size_t(row) * sbytes, l);
  const uint16_t* a = after_wait(p.act) + slot * kInter;
  for (int i = threadIdx.x; i < kInter / 8; i += kExpertThreads) {
    const uint4 v = reinterpret_cast<const uint4*>(a)[i];
    float* o = act + 8 * i;
    o[0] = bf16lo(v.x); o[1] = bf16hi(v.x); o[2] = bf16lo(v.y); o[3] = bf16hi(v.y);
    o[4] = bf16lo(v.z); o[5] = bf16hi(v.z); o[6] = bf16lo(v.w); o[7] = bf16hi(v.w);
  }
  __syncthreads();
  quantize_row<kExpertThreads>(act, kInter, MX, qc, qs, &amax, scratch);
  __syncthreads();
  stage_input<kExpertThreads>(kInter, MX, qc, qs, xh, xsc, xv);
  const float outer = MX ? 1.0f : __fmul_rn(__fdiv_rn(p.amax[e], p.denominator), __fmul_rn(amax, float(1.0 / 1536.0)));
  uint16_t* out = after_wait(p.out) + slot * kHidden;
  // rows of this warp: group base lo + warp * 4, stepping by 4 * warps; lane group (lane >> 3) is the row in it
  for (int base = lo + warp * 4; base < hi; base += 4 * kExpertWarps) {
    const int nbase = base + 4 * kExpertWarps;
    const int nrow = nbase + (lane >> 3);
    if (nbase < hi && nrow < hi) nxt.load(codes + size_t(nrow) * rowbytes, scales + size_t(nrow) * sbytes, l);
    const float acc = warp_sum<8>(cur.dot(l, xh, xsc, xv));
    if (l == 0 && row < hi) out[row] = f32_to_bf16_bits(__fmul_rn(acc, outer));
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

}  // namespace

void router(const RouterParams& p, cudaStream_t stream) {
  launch_pdl(router_kernel, dim3(kRouterBlocks), dim3(256), 0, stream, p);
}

void expert_w13(const ExpertParams& p, cudaStream_t stream) {
  launch_pdl(expert_w13_kernel, dim3(kSplit, kTopK), dim3(kExpertThreads), 0, stream, p);
}

void expert_w2(const ExpertParams& p, cudaStream_t stream) {
  launch_pdl(expert_w2_kernel, dim3(kSplit, kTopK), dim3(kExpertThreads), 0, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
