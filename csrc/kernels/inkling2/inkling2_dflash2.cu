// SPDX-License-Identifier: Apache-2.0
//
// The DFlash 2 drafter of Inkling 2 Mini (incoai/Inkling-2-Mini-DFlash2; SGLang DFlash2DraftModel) for one sequence:
// a block of 8 rows (the anchor's normalized target embedding and 7 mask rows) through 5 bidirectional layers over
// the drafter's own context KV, the target's LM head on the mask rows and the candidate selector's greedy path.
// Drafts only propose (the target's verification decides the output): the drafter sums in FP32 and rounds the
// activations to BF16 where SGLang stores them, without matching its reduction order. The layer products run on
// gemv_fp8 (E4M3 weight copies), the mask rows' logits on gemv_fp4w (NVFP4 copy of the LM head).
//
//   context: K / V of committed positions from fc(4 target features) -> hidden_norm -> per-layer k / v projections
//            (written by df_ctx_write: k head RMSNorm + NeoX RoPE, theta 1e4).
//   layer:   residual += conv_out(prev) ; x = RMSNorm ; coefficients x W_c ; u = conv_in(x) ; qkv ; q / k head
//            RMSNorm + RoPE (the block's K / V into the cache at its positions) ; attention over the window and the
//            block ; o ; conv_out ; residual += ; RMSNorm ; coefficients ; conv_in ; gate / up ; SiLU ; down.
//   select:  per mask row the 16 largest logits (df_topk) and the largest probability; the selector's correction
//            (predecessor codebook row x projected hidden . successor codebook rows), softcap 30 after the
//            correction, the greedy path from the anchor.

#include <stdexcept>

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kThreads = 256;
constexpr int kHd = 128;
constexpr int kQ = kDfQHeads;               // 32 query heads
constexpr int kKv = kDfKvHeads;             // 8 KV heads
constexpr int kGs = 16;                     // conv group size
constexpr int kGroups = kHidden / kGs;      // 144
constexpr int kTaps = 2;

__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.0f;
#pragma unroll
  for (int i = 0; i < kThreads / 32; ++i) t += red[i];
  __syncthreads();
  return t;
}

__device__ __forceinline__ float rb(float v) { return bf16_to_f32(f32_to_bf16_bits(v)); }
constexpr int kPerT = kHidden / kThreads;      // 9 channels per thread
static_assert(kHidden % kThreads == 0, "channels per thread");

// head_norm_rope with the weight values and the rotation (cos, sin of this lane's 4 elements) given.
__device__ __forceinline__ void head_norm_rope_pre(float (&v)[4], const float (&w)[4], const float (&cs)[4],
                                                   const float (&sn)[4], bool rope, float eps) {
  const int lane = threadIdx.x & 31;
  float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  ss = warp_sum(ss);
  const float inv = rsqrtf(ss / float(kHd) + eps);
#pragma unroll
  for (int t = 0; t < 4; ++t) v[t] = rb(v[t] * inv * w[t]);
  if (!rope) return;
  float other[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) other[t] = __shfl_xor_sync(0xffffffffu, v[t], 16);
#pragma unroll
  for (int t = 0; t < 4; ++t) v[t] = lane < 16 ? v[t] * cs[t] - other[t] * sn[t] : v[t] * cs[t] + other[t] * sn[t];
}
__device__ __forceinline__ void rope_pre(int pos, float theta, float (&cs)[4], float (&sn)[4]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int d = (4 * lane + t) & 63;
    const float freq = powf(theta, -2.0f * float(d) / float(kHd));
    sincosf(float(pos) * freq, &sn[t], &cs[t]);
  }
}

// RMSNorm of one 128-value head held 4 per lane, the weight, then NeoX RoPE at `pos` (lanes l and l ^ 16 pair up).
__device__ __forceinline__ void head_norm_rope(float (&v)[4], const uint16_t* w, int pos, float theta, bool rope,
                                               float eps) {
  const int lane = threadIdx.x & 31;
  float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  ss = warp_sum(ss);
  const float inv = rsqrtf(ss / float(kHd) + eps);
#pragma unroll
  for (int t = 0; t < 4; ++t) v[t] = rb(v[t] * inv * bf16_to_f32(w[4 * lane + t]));
  if (!rope) return;
  float other[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) other[t] = __shfl_xor_sync(0xffffffffu, v[t], 16);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int d = (4 * lane + t) & 63;
    const float freq = powf(theta, -2.0f * float(d) / float(kHd));
    float s, c;
    sincosf(float(pos) * freq, &s, &c);
    v[t] = lane < 16 ? v[t] * c - other[t] * s : v[t] * c + other[t] * s;
  }
}

// a conv input value: BF16 rows src, or bf16(src_f32) (a product written in FP32)
__device__ __forceinline__ float src_at(const DfParams& p, const uint16_t* src, int j, int c) {
  const size_t i = size_t(j) * kHidden + c;
  return src ? bf16_to_f32(src[i]) : rb(p.src_f32[i]);
}

// grouped two-tap convolution along the block of row j at channel c: (base0 + d0) x[j] + (base1 + d1) x[j - 1]
__device__ __forceinline__ float conv_at(const DfParams& p, const uint16_t* src, int j, int c) {
  const uint16_t* co = p.coef + size_t(j) * (2 * kTaps * kGroups) + p.coef_part * kTaps * kGroups;
  const int g = c / kGs;
  float v = (bf16_to_f32(p.base[c]) + bf16_to_f32(co[g])) * src_at(p, src, j, c);
  if (j > 0)
    v = fmaf(bf16_to_f32(p.base[kHidden + c]) + bf16_to_f32(co[kGroups + g]), src_at(p, src, j - 1, c), v);
  return rb(v);
}

// ------------------------------------------------------------------------------------------------- block rows
// grid (8 rows): residual dh = the anchor's normalized embedding (row 0) / the mask row; x = RMSNorm(dh) * norm_w.
// The anchor (written several launches back), its embedding row / the mask row and the norm weights before the wait.
__global__ void __launch_bounds__(kThreads) df_embed_kernel(const __grid_constant__ DfParams p) {
  __shared__ float red[kThreads / 32];
  const int j = blockIdx.x, t = threadIdx.x;
  float row[kPerT], en[kPerT], nw[kPerT];
  const uint16_t* e = j == 0 ? p.embed + size_t(*p.anchor) * kHidden : p.mask_row;
#pragma unroll
  for (int m = 0; m < kPerT; ++m) {
    const int c = t + m * kThreads;
    row[m] = bf16_to_f32(e[c]);
    en[m] = j == 0 ? bf16_to_f32(p.embed_norm[c]) : 0.0f;
    nw[m] = bf16_to_f32(p.norm_w[c]);
  }
  pdl_wait();
  pdl_launch();
  if (j == 0) {
    float ss = 0.0f;
#pragma unroll
    for (int m = 0; m < kPerT; ++m) ss = fmaf(row[m], row[m], ss);
    const float inv = rsqrtf(block_sum(ss, red) / float(kHidden) + p.eps2);
#pragma unroll
    for (int m = 0; m < kPerT; ++m) row[m] = rb(row[m] * inv * en[m]);
  }
  float ss = 0.0f;
  float* dh = after_wait(p.dh) + size_t(j) * kHidden;
#pragma unroll
  for (int m = 0; m < kPerT; ++m) {
    dh[t + m * kThreads] = row[m];
    ss = fmaf(row[m], row[m], ss);
  }
  const float inv = rsqrtf(block_sum(ss, red) / float(kHidden) + p.eps);
  uint16_t* x = after_wait(p.x) + size_t(j) * kHidden;
#pragma unroll
  for (int m = 0; m < kPerT; ++m) x[t + m * kThreads] = f32_to_bf16_bits(row[m] * inv * nw[m]);
}

// grid (8 rows): dst = conv(src) with the coefficient part coef_part and base [taps][kHidden]. The base and the
// rows j, j - 1 of src (BF16, written two launches back: the coefficients' GEMV between) before the wait.
__global__ void __launch_bounds__(kThreads) df_conv_kernel(const __grid_constant__ DfParams p) {
  const int j = blockIdx.x, t = threadIdx.x;
  float b0[kPerT], b1[kPerT], x0[kPerT], x1[kPerT];
  const bool early = p.src != nullptr;
  if (early) {
#pragma unroll
    for (int m = 0; m < kPerT; ++m) {
      const int c = t + m * kThreads;
      b0[m] = bf16_to_f32(p.base[c]);
      b1[m] = bf16_to_f32(p.base[kHidden + c]);
      x0[m] = bf16_to_f32(p.src[size_t(j) * kHidden + c]);
      x1[m] = j > 0 ? bf16_to_f32(p.src[size_t(j - 1) * kHidden + c]) : 0.0f;
    }
  }
  pdl_wait();
  pdl_launch();
  uint16_t* dst = after_wait(p.dst) + size_t(j) * kHidden;
  if (early) {
    const uint16_t* co = after_wait(p.coef) + size_t(j) * (2 * kTaps * kGroups) + p.coef_part * kTaps * kGroups;
#pragma unroll
    for (int m = 0; m < kPerT; ++m) {
      const int c = t + m * kThreads, g = c / kGs;
      float v = (b0[m] + bf16_to_f32(co[g])) * x0[m];
      if (j > 0) v = fmaf(b1[m] + bf16_to_f32(co[kGroups + g]), x1[m], v);
      dst[c] = f32_to_bf16_bits(rb(v));
    }
    return;
  }
  const uint16_t* src = after_wait(p.src);
  for (int c = threadIdx.x; c < kHidden; c += kThreads) dst[c] = f32_to_bf16_bits(conv_at(p, src, j, c));
}

// grid (8 rows): dh = bf16(conv(src) + dh); x = RMSNorm(dh) * norm_w (norm_w null: the residual only).
// The base, the residual (written two or more launches back) and the norm weight before the wait.
__global__ void __launch_bounds__(kThreads) df_conv_add_norm_kernel(const __grid_constant__ DfParams p) {
  __shared__ float red[kThreads / 32];
  const int j = blockIdx.x, t = threadIdx.x;
  float b0[kPerT], b1[kPerT], dr[kPerT], nw[kPerT];
  const float* dhp = p.dh + size_t(j) * kHidden;
#pragma unroll
  for (int m = 0; m < kPerT; ++m) {
    const int c = t + m * kThreads;
    b0[m] = bf16_to_f32(p.base[c]);
    b1[m] = bf16_to_f32(p.base[kHidden + c]);
    dr[m] = dhp[c];
    nw[m] = p.norm_w ? bf16_to_f32(p.norm_w[c]) : 0.0f;
  }
  pdl_wait();
  pdl_launch();
  const uint16_t* src = after_wait(p.src);
  float* dh = after_wait(p.dh) + size_t(j) * kHidden;
  const uint16_t* co = after_wait(p.coef) + size_t(j) * (2 * kTaps * kGroups) + p.coef_part * kTaps * kGroups;
  float ss = 0.0f, row[kPerT];
#pragma unroll
  for (int m = 0; m < kPerT; ++m) {
    const int c = t + m * kThreads, g = c / kGs;
    float v = (b0[m] + bf16_to_f32(co[g])) * src_at(p, src, j, c);
    if (j > 0) v = fmaf(b1[m] + bf16_to_f32(co[kGroups + g]), src_at(p, src, j - 1, c), v);
    const float o = rb(rb(v) + dr[m]);
    dh[c] = o;
    row[m] = o;
    ss = fmaf(o, o, ss);
  }
  if (!p.norm_w) return;
  const float inv = rsqrtf(block_sum(ss, red) / float(kHidden) + p.eps);
  uint16_t* x = after_wait(p.x) + size_t(j) * kHidden;
#pragma unroll
  for (int m = 0; m < kPerT; ++m) x[t + m * kThreads] = f32_to_bf16_bits(row[m] * inv * nw[m]);
}

// grid (8 rows), 8 warps over the 32 q + 8 k + 8 v heads: q / k head RMSNorm + RoPE at pos + j; the block's K / V into
// the cache of layer `layer` at their positions.
__global__ void __launch_bounds__(kThreads) df_qk_kernel(const __grid_constant__ DfParams p) {
  // the position (written many launches back), the rotation and the norm weights before the wait
  const int j = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = *p.pos + j;
  float cs[4], sn[4], qn[4], kn[4];
  rope_pre(pos, p.theta, cs, sn);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    qn[t] = bf16_to_f32(p.q_norm[4 * lane + t]);
    kn[t] = bf16_to_f32(p.k_norm[4 * lane + t]);
  }
  pdl_wait();
  pdl_launch();
  const uint16_t* row = after_wait(p.qkv) + size_t(j) * (kQ + 2 * kKv) * kHd;
  uint16_t* layer = after_wait(p.kv) + size_t(p.layer) * 2 * kKv * p.cap * kHd;
  for (int h = warp; h < kQ + 2 * kKv; h += 8) {
    const uint2 raw = *reinterpret_cast<const uint2*>(row + h * kHd + 4 * lane);
    float v[4] = {bf16lo(raw.x), bf16hi(raw.x), bf16lo(raw.y), bf16hi(raw.y)};
    uint16_t* out;
    if (h < kQ) {
      head_norm_rope_pre(v, qn, cs, sn, true, p.eps);
      out = after_wait(p.q_out) + size_t(j) * kQ * kHd + h * kHd;
    } else if (h < kQ + kKv) {
      head_norm_rope_pre(v, kn, cs, sn, true, p.eps);
      out = layer + (size_t(h - kQ) * p.cap + pos) * kHd;
    } else {
      out = layer + (size_t(kKv + h - kQ - kKv) * p.cap + pos) * kHd;
    }
    uint2 o;
    o.x = f32_to_bf16_bits(v[0]) | (uint32_t(f32_to_bf16_bits(v[1])) << 16);
    o.y = f32_to_bf16_bits(v[2]) | (uint32_t(f32_to_bf16_bits(v[3])) << 16);
    *reinterpret_cast<uint2*>(out + 4 * lane) = o;
  }
}

// Keys read once per KV head on the tensor cores: a cluster of kAttCs blocks per head (grid (8, kAttCs)); the 32
// query rows of the head (block row j, query head a: row 4 j + a) against chunks of 64 keys of [lo, start + 8),
// lo = max(0, start - window). Block b takes chunks b, b + kAttCs, ..; its 4 warps: rows 16 (w & 1) .. + 16, keys
// 32 (w >> 1) .. + 32 of the chunk, S = Q K^T and O += P V on m16n8k16 BF16 MMA with an online softmax (row j's
// keys from start + j - window). The cached keys (< start, written before the previous kernel) go to shared memory
// before pdl_wait. Each warp's (max, sum, O) to p.dh, the cluster barrier, block b merges rows 32 b / kAttCs .. .
constexpr int kAttCs = 4;
constexpr int kAttKeys = 64;
constexpr int kAttLd = kHd + 8;                  // shared row stride (bf16): conflict-free ldmatrix
constexpr int kAttRows = kDfBlock * kQ / kKv;    // 32 query rows per KV head
constexpr int kAttPart = kHd + 4;                // floats per row partial: O[128], max, sum
constexpr size_t kAttSmem = size_t(kAttRows + 4 * kAttKeys) * kAttLd * 2;

__device__ __forceinline__ void att_ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void att_ldsm4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void att_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t att_pack(float lo, float hi) {
  return uint32_t(f32_to_bf16_bits(lo)) | (uint32_t(f32_to_bf16_bits(hi)) << 16);
}

// keys [k0, k0 + 64) of the chunk to stage (K, V), only those with from <= key < to (cp.async); keys >= hi zero
__device__ __forceinline__ void att_load(uint16_t* kst, uint16_t* vst, const uint16_t* K, const uint16_t* V, int k0,
                                         int from, int to, int hi, bool zero) {
  for (int i = threadIdx.x; i < kAttKeys * 16; i += 128) {
    const int r = i >> 4, part = i & 15, k = k0 + r;
    if (k >= from && k < to) {
      cp_async16(kst + r * kAttLd + part * 8, K + size_t(k) * kHd + part * 8);
      cp_async16(vst + r * kAttLd + part * 8, V + size_t(k) * kHd + part * 8);
    } else if (zero && k >= hi) {
      *reinterpret_cast<uint4*>(kst + r * kAttLd + part * 8) = uint4{};
      *reinterpret_cast<uint4*>(vst + r * kAttLd + part * 8) = uint4{};
    }
  }
}

__global__ void __cluster_dims__(1, kAttCs, 1) __launch_bounds__(128) df_attention_kernel(const __grid_constant__ DfParams p) {
  extern __shared__ __align__(16) uint8_t att_dyn[];
  uint16_t* qs = reinterpret_cast<uint16_t*>(att_dyn);
  uint16_t* kst[2] = {qs + kAttRows * kAttLd, qs + (kAttRows + kAttKeys) * kAttLd};
  uint16_t* vst[2] = {qs + (kAttRows + 2 * kAttKeys) * kAttLd, qs + (kAttRows + 3 * kAttKeys) * kAttLd};
  const int h = blockIdx.x, b = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int mg = warp & 1, kh = warp >> 1;
  const int start = *p.pos, hi = start + kDfBlock;          // pos: written many launches back
  const int lo = max(0, start - p.window);
  const int nc = (hi - lo + kAttKeys - 1) / kAttKeys;
  const uint16_t* layer = p.kv + size_t(p.layer) * 2 * kKv * p.cap * kHd;
  const uint16_t* K = layer + size_t(h) * p.cap * kHd;
  const uint16_t* V = layer + size_t(kKv + h) * p.cap * kHd;
  // the cached keys of the first two chunks to shared memory, the later chunks' to L2
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const int c = b + st * kAttCs;
    if (c < nc) att_load(kst[st], vst[st], K, V, lo + c * kAttKeys, lo, start, hi, true);
    cp_async_commit();
  }
  for (int c = b + 2 * kAttCs; c < nc; c += kAttCs) {
    const int k0 = lo + c * kAttKeys, k1 = min(start, k0 + kAttKeys);
    if (threadIdx.x < 2 && k1 > k0)
      prefetch_l2((threadIdx.x ? V : K) + size_t(k0) * kHd, uint32_t(k1 - k0) * kHd * 2);
  }
  pdl_wait();
  pdl_launch();
  K = after_wait(K);
  V = after_wait(V);
#pragma unroll
  for (int st = 0; st < 2; ++st) {                           // the block's new keys of those chunks
    const int c = b + st * kAttCs;
    if (c < nc) att_load(kst[st], vst[st], K, V, lo + c * kAttKeys, start, hi, hi, false);
  }
  cp_async_commit();
  const uint16_t* qg = after_wait(p.q_out);
  for (int i = threadIdx.x; i < kAttRows * 16; i += 128) {
    const int r = i >> 4, part = i & 15;
    *reinterpret_cast<uint4*>(qs + r * kAttLd + part * 8) =
        *reinterpret_cast<const uint4*>(qg + size_t(r >> 2) * kQ * kHd + ((kQ / kKv) * h + (r & 3)) * kHd + part * 8);
  }
  __syncthreads();
  uint32_t qa[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) att_ldsm4(qa[kk], qs + (mg * 16 + (lane & 15)) * kAttLd + kk * 16 + (lane >> 4) * 8);
  const float sl2 = 0.08838834764831845f * 1.4426950408889634f;   // 1/sqrt(128) * log2(e)
  const int g = lane >> 2, q4 = lane & 3;
  // rows mg*16 + g and + 8: block rows (mg*16 + g) / 4 and (mg*16 + g + 8) / 4
  const int rlo0 = max(0, start + ((mg * 16 + g) >> 2) - p.window);
  const int rlo1 = max(0, start + ((mg * 16 + g + 8) >> 2) - p.window);
  float o[16][4], m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
#pragma unroll
  for (int d = 0; d < 16; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.0f;
  int it = 0;
  for (int c = b; c < nc; c += kAttCs, ++it) {
    if (it == 0) cp_async_wait<0>(); else cp_async_wait<1>();
    __syncthreads();
    const int st = it & 1;
    const uint16_t* ks = kst[st] + kh * 32 * kAttLd;
    const uint16_t* vs = vst[st] + kh * 32 * kAttLd;
    const int kb = lo + c * kAttKeys + kh * 32;
    float sc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.0f;
#pragma unroll
      for (int kk = 0; kk < 8; kk += 2) {
        uint32_t bk[4];
        att_ldsm4(bk, ks + (nt * 8 + (lane & 7)) * kAttLd + kk * 16 + (lane >> 3) * 8);
        att_mma(sc[nt], qa[kk], bk[0], bk[1]);
        att_mma(sc[nt], qa[kk + 1], bk[2], bk[3]);
      }
    }
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int k = kb + nt * 8 + 2 * q4 + e;
        sc[nt][e] = (k >= rlo0 && k < hi) ? sc[nt][e] * sl2 : -INFINITY;
        sc[nt][2 + e] = (k >= rlo1 && k < hi) ? sc[nt][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[nt][e]);
        mx1 = fmaxf(mx1, sc[nt][2 + e]);
      }
#pragma unroll
    for (int x = 1; x <= 2; x <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x));
    }
    const float u0 = mx0 == -INFINITY ? 0.0f : mx0, u1 = mx1 == -INFINITY ? 0.0f : mx1;
    const float f0 = exp2f(m0 - u0), f1 = exp2f(m1 - u1);
    m0 = mx0; m1 = mx1;
    l0 *= f0; l1 *= f1;
#pragma unroll
    for (int d = 0; d < 16; ++d) { o[d][0] *= f0; o[d][1] *= f0; o[d][2] *= f1; o[d][3] *= f1; }
    // P as BF16 hi + lo (P V on two products: the probabilities to ~16 bits, as the FP32 attention)
    uint32_t pa[2][4], pl[2][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const float p0 = exp2f(sc[nt][0] - u0), p1 = exp2f(sc[nt][1] - u0);
      const float p2 = exp2f(sc[nt][2] - u1), p3 = exp2f(sc[nt][3] - u1);
      l0 += p0 + p1; l1 += p2 + p3;
      const uint32_t h01 = att_pack(p0, p1), h23 = att_pack(p2, p3);
      pa[nt >> 1][(nt & 1) * 2] = h01;
      pa[nt >> 1][(nt & 1) * 2 + 1] = h23;
      pl[nt >> 1][(nt & 1) * 2] = att_pack(p0 - bf16lo(h01), p1 - bf16hi(h01));
      pl[nt >> 1][(nt & 1) * 2 + 1] = att_pack(p2 - bf16lo(h23), p3 - bf16hi(h23));
    }
#pragma unroll
    for (int ks2 = 0; ks2 < 2; ++ks2)
#pragma unroll
      for (int dt = 0; dt < 16; dt += 2) {
        uint32_t bv[4];
        att_ldsm4t(bv, vs + (ks2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kAttLd + (dt + (lane >> 4)) * 8);
        att_mma(o[dt], pa[ks2], bv[0], bv[1]);
        att_mma(o[dt + 1], pa[ks2], bv[2], bv[3]);
        att_mma(o[dt], pl[ks2], bv[0], bv[1]);
        att_mma(o[dt + 1], pl[ks2], bv[2], bv[3]);
      }
    __syncthreads();
    const int cn = c + 2 * kAttCs;                           // the chunk after next into this stage
    if (cn < nc) att_load(kst[st], vst[st], K, V, lo + cn * kAttKeys, lo, hi, hi, true);
    cp_async_commit();
  }
  // the warp's partial: rows mg*16 + g (+ 8), split 2 b + kh
#pragma unroll
  for (int x = 1; x <= 2; x <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, x);
    l1 += __shfl_xor_sync(0xffffffffu, l1, x);
  }
  float* part = after_wait(p.dh) + (size_t(h) * 2 * kAttCs + 2 * b + kh) * kAttRows * kAttPart;
  float* r0 = part + (mg * 16 + g) * kAttPart;
  float* r1 = r0 + 8 * kAttPart;
#pragma unroll
  for (int d = 0; d < 16; ++d) {
    *reinterpret_cast<float2*>(r0 + d * 8 + 2 * q4) = make_float2(o[d][0], o[d][1]);
    *reinterpret_cast<float2*>(r1 + d * 8 + 2 * q4) = make_float2(o[d][2], o[d][3]);
  }
  if (q4 == 0) {
    r0[kHd] = m0; r0[kHd + 1] = l0;
    r1[kHd] = m1; r1[kHd + 1] = l1;
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  // merge: block b's rows, 16 threads per row (8 values each)
  constexpr int kRowsPer = kAttRows / kAttCs;
  const int row = b * kRowsPer + (threadIdx.x >> 4), d0 = (threadIdx.x & 15) * 8;
  const float* base = after_wait(p.dh) + size_t(h) * 2 * kAttCs * kAttRows * kAttPart + row * kAttPart;
  float ms[2 * kAttCs], M = -INFINITY;
#pragma unroll
  for (int s2 = 0; s2 < 2 * kAttCs; ++s2) {
    ms[s2] = __ldcg(base + size_t(s2) * kAttRows * kAttPart + kHd);
    M = fmaxf(M, ms[s2]);
  }
  float L = 0.0f, acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int s2 = 0; s2 < 2 * kAttCs; ++s2) {
    const float* e = base + size_t(s2) * kAttRows * kAttPart;
    const float w = ms[s2] == -INFINITY ? 0.0f : exp2f(ms[s2] - M);
    L = fmaf(__ldcg(e + kHd + 1), w, L);
    const float4 x0 = __ldcg(reinterpret_cast<const float4*>(e + d0));
    const float4 x1 = __ldcg(reinterpret_cast<const float4*>(e + d0 + 4));
    acc[0] = fmaf(x0.x, w, acc[0]); acc[1] = fmaf(x0.y, w, acc[1]);
    acc[2] = fmaf(x0.z, w, acc[2]); acc[3] = fmaf(x0.w, w, acc[3]);
    acc[4] = fmaf(x1.x, w, acc[4]); acc[5] = fmaf(x1.y, w, acc[5]);
    acc[6] = fmaf(x1.z, w, acc[6]); acc[7] = fmaf(x1.w, w, acc[7]);
  }
  const float il = 1.0f / L;
  uint4 ov;
  ov.x = att_pack(acc[0] * il, acc[1] * il); ov.y = att_pack(acc[2] * il, acc[3] * il);
  ov.z = att_pack(acc[4] * il, acc[5] * il); ov.w = att_pack(acc[6] * il, acc[7] * il);
  uint16_t* out = after_wait(p.attn_out) + size_t(row >> 2) * kQ * kHd + ((kQ / kKv) * h + (row & 3)) * kHd + d0;
  *reinterpret_cast<uint4*>(out) = ov;
}

// act[j][i] = bf16(silu(gu[j][i]) * gu[j][inter + i])
__global__ void __launch_bounds__(kThreads) df_silu_kernel(const __grid_constant__ DfParams p) {
  pdl_wait();
  pdl_launch();
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i >= kDfBlock * p.inter) return;
  const int j = i / p.inter, c = i % p.inter;
  const uint16_t* gu = after_wait(p.gu) + size_t(j) * 2 * p.inter;
  const float g = bf16_to_f32(gu[c]), u = bf16_to_f32(gu[p.inter + c]);
  after_wait(p.act)[i] = f32_to_bf16_bits(rb(g / (1.0f + __expf(-g))) * u);
}

// ------------------------------------------------------------------------------------------------- context
// grid (rows, layers), 8 warps = 8 KV heads: row j of the context projections kvc [layers][rows][2][1024] (FP32):
// k head RMSNorm + RoPE, v, into the cache at pos - count - count_add + j.
__global__ void __launch_bounds__(kThreads) df_ctx_write_kernel(const __grid_constant__ DfParams p) {
  pdl_wait();
  pdl_launch();
  const int j = blockIdx.x, l = blockIdx.y, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = *after_wait(p.pos) - (p.count ? *after_wait(p.count) : 0) - p.count_add + j;
  const float* kv = after_wait(p.kvc) + (size_t(l) * p.rows + j) * 2 * kKv * kHd;
  uint16_t* layer = after_wait(p.kv) + size_t(l) * 2 * kKv * p.cap * kHd;
  float k[4], v[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    k[t] = rb(kv[h * kHd + 4 * lane + t]);
    v[t] = rb(kv[(kKv + h) * kHd + 4 * lane + t]);
  }
  head_norm_rope(k, p.k_norm_l[l], pos, p.theta, true, p.eps);
  uint2 ko, vo;
  ko.x = f32_to_bf16_bits(k[0]) | (uint32_t(f32_to_bf16_bits(k[1])) << 16);
  ko.y = f32_to_bf16_bits(k[2]) | (uint32_t(f32_to_bf16_bits(k[3])) << 16);
  vo.x = f32_to_bf16_bits(v[0]) | (uint32_t(f32_to_bf16_bits(v[1])) << 16);
  vo.y = f32_to_bf16_bits(v[2]) | (uint32_t(f32_to_bf16_bits(v[3])) << 16);
  *reinterpret_cast<uint2*>(layer + (size_t(h) * p.cap + pos) * kHd + 4 * lane) = ko;
  *reinterpret_cast<uint2*>(layer + (size_t(kKv + h) * p.cap + pos) * kHd + 4 * lane) = vo;
}

// ------------------------------------------------------------------------------------------------- selector
// one block of 16 warps (one per candidate): the greedy path over the 7 mask rows. Step e: predecessor = the anchor
// (e = 0) or the previous step's candidate; score_c = softcap(logit_c + sum_r bf16(pred[r] hp[e][r]) succ_c[r]).
__global__ void __launch_bounds__(512) df_select_kernel(const __grid_constant__ DfParams p) {
  __shared__ float hp[kDfRank];
  __shared__ float sc[kDfTopK];
  __shared__ int chosen;
  pdl_wait();
  pdl_launch();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long* cand = after_wait(p.cand);
  const float* vals = after_wait(p.vals);
  int pred = *after_wait(p.anchor);
  for (int e = 0; e < kDfBlock - 1; ++e) {
    const uint16_t* hrow = after_wait(p.hp) + size_t(e) * kDfRank;
    const uint16_t* prow = p.pred_cb + size_t(pred) * kDfRank;
    for (int r = threadIdx.x; r < kDfRank; r += 512) hp[r] = rb(bf16_to_f32(hrow[r]) * bf16_to_f32(prow[r]));
    __syncthreads();
    const int c = int(cand[e * kDfTopK + warp]);
    const uint16_t* srow = p.succ_cb + size_t(c) * kDfRank;
    float s = 0.0f;
    for (int r = lane; r < kDfRank; r += 32) s = fmaf(hp[r], bf16_to_f32(srow[r]), s);
    s = warp_sum(s);
    if (lane == 0) sc[warp] = p.softcap * tanhf((vals[e * kDfTopK + warp] + rb(s)) / p.softcap);
    __syncthreads();
    if (threadIdx.x == 0) {
      int best = 0;
      for (int i = 1; i < kDfTopK; ++i)
        if (sc[i] > sc[best]) best = i;
      chosen = int(cand[e * kDfTopK + best]);
      after_wait(p.drafts)[e] = chosen;
    }
    __syncthreads();
    pred = chosen;
  }
}

// ------------------------------------------------------------------------------------------------- top 16
// The 16 largest logits of each mask row (descending, lower id first on ties) and the row's largest softmax
// probability, from the logits [rows][vocab]. Phase 1: grid (rows, kTopBlocks), each warp keeps its chunk's top 16
// on lanes 0..15 (inserting only candidates above its 16th) and the chunk's (max, sum of exp); the block merges its
// warps. Phase 2: one block per row merges the blocks.
constexpr int kTopBlocks = 64;
struct TopEntry { float v; int i; };
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

// insert (v, i) into the warp's sorted list (lanes 0..15 hold entries 0..15, descending)
__device__ __forceinline__ void warp_insert(float& lv, int& li, float v, int i) {
  const int lane = threadIdx.x & 31;
  const bool after = lane < 16 && better(lv, li, v, i);            // entries that stay above
  const unsigned m = __ballot_sync(0xffffffffu, after);
  const int at = __popc(m & 0xffffu);                               // insert position
  const float pv = __shfl_up_sync(0xffffffffu, lv, 1);
  const int pi = __shfl_up_sync(0xffffffffu, li, 1);
  if (lane < 16 && lane > at) { lv = pv; li = pi; }
  if (lane == at && at < 16) { lv = v; li = i; }
}

// order-preserving keys of better(): the larger value first (-0 as +0), the lower index on equal values
__device__ __forceinline__ uint32_t top_key(float v) {
  uint32_t u = __float_as_uint(v);
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// one round of a warp merge: every lane offers (v, i); all lanes get the winner (the largest key, then the lowest
// index holding it); returns whether this lane holds it (two 32-bit REDUX reductions: the key, then the index)
__device__ __forceinline__ bool warp_best(float v, int i, float& gv, int& gi) {
  const uint32_t key = top_key(v);
  const uint32_t mk = __reduce_max_sync(0xffffffffu, key);
  gi = int(__reduce_min_sync(0xffffffffu, key == mk ? uint32_t(i) : 0xffffffffu));
  const bool mine = key == mk && i == gi;
  gv = __shfl_sync(0xffffffffu, v, __ffs(__ballot_sync(0xffffffffu, mine)) - 1);
  return mine;
}

__global__ void __launch_bounds__(256) df_top1_kernel(const __grid_constant__ DfParams p) {
  __shared__ float wv[8][16], wm[8], ws[8];
  __shared__ int wi[8][16];
  pdl_wait();
  pdl_launch();
  const int row = blockIdx.x, b = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int V = p.vocab;
  const int per = (V + kTopBlocks - 1) / kTopBlocks;
  const int lo = b * per, hi = min(V, lo + per);
  const float* x = after_wait(p.logits) + size_t(row) * V;
  float lv = -INFINITY, m = -INFINITY, sum = 0.0f;
  int li = 1 << 30;
  for (int base = lo + warp * 32; base < hi; base += 8 * 32) {
    const int i = base + lane;
    const float v = i < hi ? x[i] : -INFINITY;
    if (v > m) { sum = sum * __expf(m - v) + 1.0f; m = v; } else if (v != -INFINITY) { sum += __expf(v - m); }
    const float th = __shfl_sync(0xffffffffu, lv, 15);
    const int ti = __shfl_sync(0xffffffffu, li, 15);
    unsigned c = __ballot_sync(0xffffffffu, i < hi && better(v, i, th, ti));
    while (c) {
      const int src = __ffs(c) - 1;
      c &= c - 1;
      const float cv = __shfl_sync(0xffffffffu, v, src);
      const int ci = __shfl_sync(0xffffffffu, i, src);
      warp_insert(lv, li, cv, ci);
    }
  }
  // the warp's (max, sum)
  float wmx = warp_max(m);
  float wsum = warp_sum(m == -INFINITY ? 0.0f : sum * __expf(m - wmx));
  if (lane < 16) { wv[warp][lane] = lv; wi[warp][lane] = li; }
  if (lane == 0) { wm[warp] = wmx; ws[warp] = wsum; }
  __syncthreads();
  if (warp != 0) return;
  // warp 0 merges the 8 lists: lanes hold 4 candidates each (128), 16 rounds of the warp's best
  float cv[4]; int ci[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) { const int e = lane * 4 + k; cv[k] = wv[e / 16][e % 16]; ci[k] = wi[e / 16][e % 16]; }
  float* outv = after_wait(p.top_part) + (size_t(row) * kTopBlocks + b) * (2 * kDfTopK + 2);
  for (int r = 0; r < kDfTopK; ++r) {
    float bv = -INFINITY; int bi = 1 << 30, bk = -1;
#pragma unroll
    for (int k = 0; k < 4; ++k) if (better(cv[k], ci[k], bv, bi)) { bv = cv[k]; bi = ci[k]; bk = k; }
    float gv = bv; int gi = bi;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, gv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, gi, o);
      if (better(ov, oi, gv, gi)) { gv = ov; gi = oi; }
    }
    if (bk >= 0 && bi == gi && bv == gv) cv[bk] = -INFINITY, ci[bk] = 1 << 30;
    if (lane == 0) { outv[2 * r] = gv; reinterpret_cast<int*>(outv)[2 * r + 1] = gi; }
  }
  if (lane == 0) {
    float M = wm[0];
    for (int w = 1; w < 8; ++w) M = fmaxf(M, wm[w]);
    float S = 0.0f;
    for (int w = 0; w < 8; ++w) S += wm[w] == -INFINITY ? 0.0f : ws[w] * __expf(wm[w] - M);
    outv[2 * kDfTopK] = M;
    outv[2 * kDfTopK + 1] = S;
  }
}

// phase 2: one block (one warp) per row: kTopBlocks x 16 candidates, the top 16, the largest probability
__global__ void __launch_bounds__(32) df_top2_kernel(const __grid_constant__ DfParams p) {
  pdl_wait();
  pdl_launch();
  const int row = blockIdx.x, lane = threadIdx.x;
  const float* part = after_wait(p.top_part) + size_t(row) * kTopBlocks * (2 * kDfTopK + 2);
  constexpr int kPer = kTopBlocks * kDfTopK / 32;                   // candidates per lane
  float cv[kPer]; int ci[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int e = lane * kPer + k, blk = e / kDfTopK, j = e % kDfTopK;
    cv[k] = part[blk * (2 * kDfTopK + 2) + 2 * j];
    ci[k] = reinterpret_cast<const int*>(part)[blk * (2 * kDfTopK + 2) + 2 * j + 1];
  }
  float M = -INFINITY;
  for (int blk = lane; blk < kTopBlocks; blk += 32) M = fmaxf(M, part[blk * (2 * kDfTopK + 2) + 2 * kDfTopK]);
  M = warp_max(M);
  float S = 0.0f;
  for (int blk = lane; blk < kTopBlocks; blk += 32) {
    const float bm = part[blk * (2 * kDfTopK + 2) + 2 * kDfTopK];
    if (bm != -INFINITY) S += part[blk * (2 * kDfTopK + 2) + 2 * kDfTopK + 1] * __expf(bm - M);
  }
  S = warp_sum(S);
  float* vals = after_wait(p.top_vals) + size_t(row) * kDfTopK;
  long long* cand = after_wait(p.top_ids) + size_t(row) * kDfTopK;
  float top = -INFINITY;
  // the blocks' sorted lists merged: lane l holds blocks 2 l, 2 l + 1 (shared memory), the winning list pops its head
  static_assert(kPer == 2 * kDfTopK, "two lists per lane");
  __shared__ float sv[32 * kPer];
  __shared__ int si[32 * kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) { sv[lane * kPer + k] = cv[k]; si[lane * kPer + k] = ci[k]; }
  __syncwarp();
  int h0 = 0, h1 = 0;
#pragma unroll 1
  for (int r = 0; r < kDfTopK; ++r) {
    const float v0 = h0 < kDfTopK ? sv[lane * kPer + h0] : -INFINITY;
    const float v1 = h1 < kDfTopK ? sv[lane * kPer + kDfTopK + h1] : -INFINITY;
    const int i0 = h0 < kDfTopK ? si[lane * kPer + h0] : (1 << 30);
    const int i1 = h1 < kDfTopK ? si[lane * kPer + kDfTopK + h1] : (1 << 30);
    const bool first = better(v0, i0, v1, i1);
    float gv;
    int gi;
    if (warp_best(first ? v0 : v1, first ? i0 : i1, gv, gi)) { if (first) ++h0; else ++h1; }
    if (lane == 0) { vals[r] = gv; cand[r] = gi; }
    if (r == 0) top = gv;
  }
  if (lane == 0) after_wait(p.top_prob)[row] = __expf(top - (M + logf(S)));
}

}  // namespace

namespace {
void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}
// the block kernels' common inputs: the position and the KV cache of `layer`
void check_block(const DfParams& p, const char* what) {
  require(p.pos && p.kv && p.cap > 0 && p.layer >= 0 && p.layer < kDfMaxLayers, what);
}
}  // namespace

void df_topk(const DfParams& p, cudaStream_t s) {
  require(p.rows >= 1 && p.rows < kDfBlock && p.vocab >= kDfTopK && p.logits && p.top_vals && p.top_ids &&
          p.top_prob && p.top_part, "df_topk: 1..7 rows, vocab >= 16, logits, top_vals, top_ids, top_prob, top_part");
  launch_pdl(df_top1_kernel, dim3(p.rows, kTopBlocks), dim3(256), 0, s, p);
  launch_pdl(df_top2_kernel, dim3(p.rows), dim3(32), 0, s, p);
}

void df_embed(const DfParams& p, cudaStream_t s) {
  require(p.anchor && p.embed && p.embed_norm && p.mask_row && p.dh && p.x && p.norm_w && p.eps > 0.0f &&
          p.eps2 > 0.0f, "df_embed: anchor, embed, embed_norm, mask_row, dh, x, norm_w, eps, eps2");
  launch_pdl(df_embed_kernel, dim3(kDfBlock), dim3(kThreads), 0, s, p);
}
void df_conv(const DfParams& p, cudaStream_t s) {
  require(p.src && p.dst && p.coef && p.base && (p.coef_part == 0 || p.coef_part == 1),
          "df_conv: src, dst, coef, base, coef_part 0 / 1");
  launch_pdl(df_conv_kernel, dim3(kDfBlock), dim3(kThreads), 0, s, p);
}
void df_conv_add_norm(const DfParams& p, cudaStream_t s) {
  require((p.src || p.src_f32) && p.coef && p.base && p.dh && (p.coef_part == 0 || p.coef_part == 1) &&
          (!p.norm_w || (p.x && p.eps > 0.0f)), "df_conv_add_norm: src or src_f32, coef, base, dh (x, eps with norm_w)");
  launch_pdl(df_conv_add_norm_kernel, dim3(kDfBlock), dim3(kThreads), 0, s, p);
}
void df_qk(const DfParams& p, cudaStream_t s) {
  check_block(p, "df_qk: pos, kv, cap, layer");
  require(p.qkv && p.q_norm && p.k_norm && p.q_out && p.theta > 0.0f && p.eps > 0.0f,
          "df_qk: qkv, q_norm, k_norm, q_out, theta, eps");
  launch_pdl(df_qk_kernel, dim3(kDfBlock), dim3(kThreads), 0, s, p);
}
void df_attention(const DfParams& p, cudaStream_t s) {
  check_block(p, "df_attention: pos, kv, cap, layer");
  require(p.q_out && p.attn_out && p.dh && p.window > 0, "df_attention: q_out, attn_out, dh (partials), window");
  static std::atomic<unsigned long long> smem_set{0};
  if (first_use_on_device(smem_set))
    check_cuda(cudaFuncSetAttribute(df_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kAttSmem)),
               "df_attention smem");
  launch_pdl(df_attention_kernel, dim3(kKv, kAttCs), dim3(128), kAttSmem, s, p);
}
void df_silu(const DfParams& p, cudaStream_t s) {
  require(p.gu && p.act && p.inter > 0, "df_silu: gu, act, inter > 0");
  launch_pdl(df_silu_kernel, dim3((kDfBlock * p.inter + kThreads - 1) / kThreads), dim3(kThreads), 0, s, p);
}
void df_ctx_write(const DfParams& p, cudaStream_t s) {
  require(p.rows >= 1 && p.layers >= 1 && p.layers <= kDfMaxLayers && p.kvc && p.kv && p.pos && p.cap > 0 &&
          p.theta > 0.0f && p.eps > 0.0f, "df_ctx_write: rows, layers, kvc, kv, pos, cap, theta, eps");
  for (int l = 0; l < p.layers; ++l) require(p.k_norm_l[l], "df_ctx_write: k_norm_l per layer");
  launch_pdl(df_ctx_write_kernel, dim3(p.rows, p.layers), dim3(kThreads), 0, s, p);
}
void df_select(const DfParams& p, cudaStream_t s) {
  require(p.anchor && p.cand && p.vals && p.hp && p.pred_cb && p.succ_cb && p.drafts && p.softcap > 0.0f,
          "df_select: anchor, cand, vals, hp, pred_cb, succ_cb, drafts, softcap");
  launch_pdl(df_select_kernel, dim3(1), dim3(512), 0, s, p);
}

}  // namespace inkling2
}  // namespace flash_rt
