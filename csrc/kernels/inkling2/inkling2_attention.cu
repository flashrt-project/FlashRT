// SPDX-License-Identifier: Apache-2.0
//
// Batch-one decode attention of a layer visit: the prologue (short convolutions of k / v, L2 clips, value RMS
// norm, log-n scaling, the relative-bias table, the KV write) and split-KV attention with the relative bias and
// the attention sinks (24 query heads over 8 KV heads of dim 128; sliding window 1024 or full). The last block of
// each KV head merges the splits and writes the BF16 output (gated later by the o_proj prologue).

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kHd = 128;
constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = kHeads / kKvHeads;
constexpr int kRel = 16;
constexpr int kExtent = 1024;
constexpr int kWindow = 1024;

__device__ __forceinline__ float block_sum128(float v, float* scratch) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
  __syncthreads();
  const float t = scratch[0] + scratch[1] + scratch[2] + scratch[3];
  __syncthreads();
  return t;
}

__device__ __forceinline__ float l2_clip_scale(float sumsq, float threshold) {
  return fminf(threshold / (sqrtf(sumsq) + 1e-6f), 1.0f);
}

// Fused decode attention: one block per (KV head h, split of kSplitKeys keys), 128 threads.
//   - the split's cached K / V rows go to shared memory with cp.async (one load round);
//   - the 3 query heads of the group: L2 clip (x scale), log-n scale (full layers), BF16; their relative states the
//     same way, and the bias of exactly this split's distances (r @ proj);
//   - the block holding the new position computes its K / V row from the projections (short convolution, K L2 clip,
//     V RMS norm), writes it to the cache and advances the KV head's conv states;
//   - logits and P V from shared memory; the last block of the head merges the splits with the sinks.
constexpr int kSplitKeys = 64;
constexpr int kAttnThreads = 128;
constexpr int kMaxMergeSplits = 16;      // merges of up to 16 splits load their partials at once

// The new K / V row (one head, 128 threads): conv of x over the three previous inputs (s[0] oldest), then the K L2
// clip or the V RMS norm, BF16. Static (sliding) or dynamic (full, dyn = the row's 4 weights of this head) conv.
__device__ __forceinline__ float kv_conv(bool sliding, const uint16_t* w, const uint16_t* dyn, int c, int C,
                                        const float (&s)[3], float x) {
  float y;
  if (sliding) {
    float conv = bf16_to_f32(w[4 * c]) * s[0];
    conv = fmaf(bf16_to_f32(w[4 * c + 1]), s[1], conv);
    conv = fmaf(bf16_to_f32(w[4 * c + 2]), s[2], conv);
    conv = fmaf(bf16_to_f32(w[4 * c + 3]), x, conv);
    y = round_bf16((round_bf16(conv) + x) / sqrtf(5.0f));
  } else {
    const int h = c / kHd;
    const float xs[4] = {x, s[2], s[1], s[0]};
    float out = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float wj = bf16_to_f32(w[j * C + c]) + bf16_to_f32(dyn[h * 4 + j]);
      out = __fadd_rn(out, __fmul_rn(xs[j], wj));
    }
    y = round_bf16((out + x) / sqrtf(9.0f));
  }
  return y;
}

__device__ __forceinline__ float kv_conv_pre(bool sliding, const float (&w)[4], const uint16_t* dyn, int h,
                                            const float (&s)[3], float x) {
  float y;
  if (sliding) {
    float conv = w[0] * s[0];
    conv = fmaf(w[1], s[1], conv);
    conv = fmaf(w[2], s[2], conv);
    conv = fmaf(w[3], x, conv);
    y = round_bf16((round_bf16(conv) + x) / sqrtf(5.0f));
  } else {
    const float xs[4] = {x, s[2], s[1], s[0]};
    float out = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float wj = w[j] + bf16_to_f32(dyn[h * 4 + j]);
      out = __fadd_rn(out, __fmul_rn(xs[j], wj));
    }
    y = round_bf16((out + x) / sqrtf(9.0f));
  }
  return y;
}

__device__ __forceinline__ uint16_t new_kv(bool is_k, bool sliding, const uint16_t* w, const uint16_t* dyn, int c, int C,
                                           const float (&s)[3], float x, float threshold, float* scratch) {
  float y;
  if (sliding) {
    float conv = bf16_to_f32(w[4 * c]) * s[0];
    conv = fmaf(bf16_to_f32(w[4 * c + 1]), s[1], conv);
    conv = fmaf(bf16_to_f32(w[4 * c + 2]), s[2], conv);
    conv = fmaf(bf16_to_f32(w[4 * c + 3]), x, conv);
    y = round_bf16((round_bf16(conv) + x) / sqrtf(5.0f));
  } else {
    const int h = c / kHd;
    const float xs[4] = {x, s[2], s[1], s[0]};
    float out = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float wj = bf16_to_f32(w[j * C + c]) + bf16_to_f32(dyn[h * 4 + j]);
      out = __fadd_rn(out, __fmul_rn(xs[j], wj));
    }
    y = round_bf16((out + x) / sqrtf(9.0f));
  }
  const float ss = block_sum128(y * y, scratch);
  return f32_to_bf16_bits(is_k ? y * l2_clip_scale(ss, threshold) : y * (1.0f / sqrtf(ss / float(kHd) + 1e-6f)));
}

// One (row, split) of decode attention at query position qpos: the split's K / V rows to shared memory (cached rows
// with cp.async; keys at or after `pos` of a verification step from the staged rows of a sliding layer), the row's
// query heads and relative states, logits, softmax and P V into the partial. In a one-row step the block holding
// qpos computes the new K / V row itself (cache write, conv state shift).
struct AttnSmem {
  __align__(16) uint16_t ks[kSplitKeys][kHd];
  __align__(16) uint16_t vs[kSplitKeys][kHd];
  float qs[kGroup][kHd];
  float rs[kGroup][kRel];
  float lg[kGroup][kSplitKeys];
  float scratch[4];
  float red[kGroup][4];
};

// The queries of KV head h for row r at qpos: q = bf16(l2clip(q_lin * q_clip)) (x tau, full layers) [3][128] and the
// relative queries [3][16]; 128 threads.
constexpr int kQPre = kGroup * kHd + kGroup * kRel;
__device__ __forceinline__ void prepare_queries(const AttnDecodeParams& p, int h, int r, int qpos, float* qs, float* rs,
                                                float* scratch) {
  const int t = threadIdx.x;
  float tau = 1.0f;
  if (!p.sliding) tau = 1.0f + p.log_alpha * logf(fmaxf((float(qpos) + 1.0f) / p.log_floor, 1.0f));
  const uint16_t* ql = after_wait(p.q_lin) + size_t(r) * kHeads * kHd;
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    const int qh = h * kGroup + g;
    const float q = bf16_to_f32(ql[qh * kHd + t]) * bf16_to_f32(p.q_clip[qh * kHd + t]);
    const float ss = block_sum128(q * q, scratch);
    float qo = round_bf16(q * l2_clip_scale(ss, p.qk_threshold));
    if (!p.sliding) qo = round_bf16(qo * tau);
    qs[g * kHd + t] = qo;
  }
  {
    const uint16_t* rl = after_wait(p.r_lin) + size_t(r) * kHeads * kRel;
    const int g = t >> 4, i = t & 15;
    float rv = 0.0f;
    if (g < kGroup) rv = bf16_to_f32(rl[(h * kGroup + g) * kRel + i]) * bf16_to_f32(p.rel_clip[(h * kGroup + g) * kRel + i]);
    float s2 = rv * rv;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    if (g < kGroup) {
      float ro = round_bf16(rv * l2_clip_scale(s2, p.rel_threshold));
      if (!p.sliding) ro = round_bf16(ro * tau);
      rs[g * kRel + i] = ro;
    }
  }
}

// Row r's split over its keys ks[0 .. kn) / vs (positions k0 ..): logits with the relative bias, softmax, P V; the
// split's partial (max, sum, acc). The queries in sm.qs / sm.rs.
template <class Sm>
__device__ __forceinline__ void split_rows(const AttnDecodeParams& p, Sm& sm, const uint16_t (*ks)[kHd],
                                           const uint16_t (*vs)[kHd], int h, int split, int r, int qpos, int k0, int kn) {
  const int t = threadIdx.x;
  {
    const int key = t >> 1, half = t & 1;
    float acc[kGroup] = {0.0f, 0.0f, 0.0f};
    const uint4* kr = reinterpret_cast<const uint4*>(&ks[key][half * 64]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 v = kr[i];
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int d = half * 64 + i * 8 + j * 2;
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
          acc[g] = fmaf(bf16lo(w[j]), sm.qs[g][d], acc[g]);
          acc[g] = fmaf(bf16hi(w[j]), sm.qs[g][d + 1], acc[g]);
        }
      }
    }
#pragma unroll
    for (int g = 0; g < kGroup; ++g) acc[g] += __shfl_xor_sync(0xffffffffu, acc[g], 1);
    if (half == 0) {
      const int dist = qpos - (k0 + key);
#pragma unroll
      for (int g = 0; g < kGroup; ++g) {
        float l = -INFINITY;
        if (key < kn) {
          l = acc[g] * (1.0f / sqrtf(float(kHd)));
          if (dist < kExtent) {
            float b = 0.0f;
#pragma unroll
            for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[g][i], bf16_to_f32(p.rel_proj[i * kExtent + dist]), b);
            l += round_bf16(b);
          }
        }
        sm.lg[g][key] = l;
      }
    }
  }
  __syncthreads();
  const int lane = t & 31, warp = t >> 5;
  if (warp < kGroup) {
    const float a = sm.lg[warp][lane], b = sm.lg[warp][lane + 32];
    const float m = warp_max(fmaxf(a, b));
    const float ea = a == -INFINITY ? 0.0f : expf(a - m), eb = b == -INFINITY ? 0.0f : expf(b - m);
    sm.lg[warp][lane] = ea;
    sm.lg[warp][lane + 32] = eb;
    const float s = warp_sum(ea + eb);
    if (lane == 0) { sm.red[warp][0] = m; sm.red[warp][1] = s; }
  }
  __syncthreads();
  float acc[kGroup] = {0.0f, 0.0f, 0.0f};
#pragma unroll 8
  for (int key = 0; key < kn; ++key) {
    const float v = bf16_to_f32(vs[key][t]);
#pragma unroll
    for (int g = 0; g < kGroup; ++g) acc[g] = fmaf(sm.lg[g][key], v, acc[g]);
  }
  const int R = p.rows < 1 ? 1 : p.rows;
  float* out = after_wait(p.partial) + ((size_t(h) * p.max_splits + split) * R + r) * kGroup * (kHd + 2);
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    out[g * (kHd + 2) + 2 + t] = acc[g];
    if (t == 0) { out[g * (kHd + 2)] = sm.red[g][0]; out[g * (kHd + 2) + 1] = sm.red[g][1]; }
  }
}


// Before the dependency wait: the split's cached keys (positions below pos, written by earlier steps)
// to shared memory, and the static weights / conv states of this thread's channel into registers.
struct AttnPre {
  float q_clip[kGroup], rel_clip;
  float kw[4], vw[4];               // conv weights of channel c (sliding: taps; full: the [4][C] column)
  float sk[3], sv[3];               // conv states (oldest first)
};

__device__ __forceinline__ void attend_pre(const AttnDecodeParams& p, AttnSmem& sm, int h, int split, int r, int pos,
                                           int qpos, bool verify, AttnPre& pre) {
  const int t = threadIdx.x;
  const int first = p.sliding ? max(0, qpos - (kWindow - 1)) : 0;
  const int k0 = first + split * kSplitKeys;
  const int kn = min(kSplitKeys, qpos + 1 - k0);
  const uint16_t* kc = p.k_cache;
  const uint16_t* vc = p.v_cache;
  for (int i = t; i < kSplitKeys * 16; i += kAttnThreads) {
    const int key = i >> 4, part16 = i & 15;
    const int kp = k0 + key;
    if (key < kn && kp < pos) {
      const int slot = p.sliding ? kp % kWindow : kp;
      cp_async16(&sm.ks[key][part16 * 8], kc + (size_t(slot) * kKvHeads + h) * kHd + part16 * 8);
      cp_async16(&sm.vs[key][part16 * 8], vc + (size_t(slot) * kKvHeads + h) * kHd + part16 * 8);
    }
  }
#pragma unroll
  for (int g = 0; g < kGroup; ++g) pre.q_clip[g] = bf16_to_f32(p.q_clip[(h * kGroup + (g)) * kHd + t]);
  {
    const int g = t >> 4, i = t & 15;
    pre.rel_clip = g < kGroup ? bf16_to_f32(p.rel_clip[(h * kGroup + g) * kRel + i]) : 0.0f;
  }
  const int c = h * kHd + t, C = kKvHeads * kHd;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    pre.kw[j] = bf16_to_f32(p.sliding ? p.k_conv_w[4 * c + j] : p.k_conv_w[j * C + c]);
    pre.vw[j] = bf16_to_f32(p.sliding ? p.v_conv_w[4 * c + j] : p.v_conv_w[j * C + c]);
  }
  if (!verify) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      pre.sk[j] = bf16_to_f32(p.k_conv_state[j * C + c]);
      pre.sv[j] = bf16_to_f32(p.v_conv_state[j * C + c]);
    }
  }
}

__device__ void attend_split(const AttnDecodeParams& p, AttnSmem& sm, int h, int split, int r, int pos, int qpos,
                             bool verify, const AttnPre* pre = nullptr) {
  const int t = threadIdx.x;
  const int first = p.sliding ? max(0, qpos - (kWindow - 1)) : 0;
  const int k0 = first + split * kSplitKeys;
  const int kn = min(kSplitKeys, qpos + 1 - k0);
  uint16_t* kc = after_wait(p.k_cache);
  uint16_t* vc = after_wait(p.v_cache);
  const uint16_t* skv = verify ? after_wait(p.stage_kv) : nullptr;     // [rows][2][8][128]
  const bool fuse = verify && p.fuse_kv;
  for (int i = t; i < kSplitKeys * 16; i += kAttnThreads) {
    const int key = i >> 4, part16 = i & 15;
    const int kp = k0 + key;
    if (key < kn && (verify || kp < qpos) && !(fuse && kp >= pos) && !(pre && kp < pos)) {
      if (verify && p.sliding && kp >= pos) {
        const uint16_t* src = skv + ((size_t(kp - pos) * 2) * kKvHeads + h) * kHd + part16 * 8;
        cp_async16(&sm.ks[key][part16 * 8], src);
        cp_async16(&sm.vs[key][part16 * 8], src + kKvHeads * kHd);
      } else {
        const int slot = p.sliding ? kp % kWindow : kp;
        cp_async16(&sm.ks[key][part16 * 8], kc + (size_t(slot) * kKvHeads + h) * kHd + part16 * 8);
        cp_async16(&sm.vs[key][part16 * 8], vc + (size_t(slot) * kKvHeads + h) * kHd + part16 * 8);
      }
    }
  }
  const bool newrow = !verify && qpos < k0 + kSplitKeys;     // one-row step: this block computes the new row
  if (p.q_pre) {
    const float* pre = after_wait(p.q_pre) + (size_t(r) * kKvHeads + h) * kQPre;
#pragma unroll
    for (int g = 0; g < kGroup; ++g) sm.qs[g][t] = pre[g * kHd + t];
    if (t < kGroup * kRel) sm.rs[t >> 4][t & 15] = pre[kGroup * kHd + t];
  } else {
    // prepare_queries' and new_kv's arithmetic with their 3 (+ 2) block sums in one reduction round
    float tau = 1.0f;
    if (!p.sliding) tau = 1.0f + p.log_alpha * logf(fmaxf((float(qpos) + 1.0f) / p.log_floor, 1.0f));
    const uint16_t* ql = after_wait(p.q_lin) + size_t(r) * kHeads * kHd;
    float q[kGroup], v5[kGroup + 2];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const int qh = h * kGroup + g;
      q[g] = bf16_to_f32(ql[qh * kHd + t]) * (pre ? pre->q_clip[g] : bf16_to_f32(p.q_clip[qh * kHd + t]));
      v5[g] = q[g] * q[g];
    }
    const int c = h * kHd + t, C = kKvHeads * kHd;
    float xk = 0.0f, xv = 0.0f, yk = 0.0f, yv = 0.0f;
    if (newrow) {
      xk = bf16_to_f32(after_wait(p.k_lin)[c]); xv = bf16_to_f32(after_wait(p.v_lin)[c]);
      if (pre) {
        yk = kv_conv_pre(p.sliding, pre->kw, p.sliding ? nullptr : after_wait(p.k_dyn), h, pre->sk, xk);
        yv = kv_conv_pre(p.sliding, pre->vw, p.sliding ? nullptr : after_wait(p.v_dyn), h, pre->sv, xv);
      } else {
        const uint16_t* kst = after_wait(p.k_conv_state);
        const uint16_t* vst = after_wait(p.v_conv_state);
        const float sk[3] = {bf16_to_f32(kst[c]), bf16_to_f32(kst[C + c]), bf16_to_f32(kst[2 * C + c])};
        const float sv[3] = {bf16_to_f32(vst[c]), bf16_to_f32(vst[C + c]), bf16_to_f32(vst[2 * C + c])};
        yk = kv_conv(p.sliding, p.k_conv_w, p.sliding ? nullptr : after_wait(p.k_dyn), c, C, sk, xk);
        yv = kv_conv(p.sliding, p.v_conv_w, p.sliding ? nullptr : after_wait(p.v_dyn), c, C, sv, xv);
      }
    }
    v5[kGroup] = yk * yk;
    v5[kGroup + 1] = yv * yv;
    float (*sc)[4] = reinterpret_cast<float (*)[4]>(&sm.lg[0][0]);
    const int nsum = newrow ? kGroup + 2 : kGroup;
#pragma unroll
    for (int i = 0; i < kGroup + 2; ++i) {
      if (i >= nsum) break;
      const float w = warp_sum(v5[i]);
      if ((t & 31) == 0) sc[i][t >> 5] = w;
    }
    // the relative queries (warp-level sums of 16) meanwhile
    {
      const uint16_t* rl = after_wait(p.r_lin) + size_t(r) * kHeads * kRel;
      const int g = t >> 4, i = t & 15;
      float rv = 0.0f;
      if (g < kGroup) rv = bf16_to_f32(rl[(h * kGroup + g) * kRel + i]) *
                           (pre ? pre->rel_clip : bf16_to_f32(p.rel_clip[(h * kGroup + g) * kRel + i]));
      float s2 = rv * rv;
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      if (g < kGroup) {
        float ro = round_bf16(rv * l2_clip_scale(s2, p.rel_threshold));
        if (!p.sliding) ro = round_bf16(ro * tau);
        sm.rs[g][i] = ro;
      }
    }
    __syncthreads();
    float ss[kGroup + 2];
#pragma unroll
    for (int i = 0; i < kGroup + 2; ++i) ss[i] = i < nsum ? sc[i][0] + sc[i][1] + sc[i][2] + sc[i][3] : 0.0f;
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      float qo = round_bf16(q[g] * l2_clip_scale(ss[g], p.qk_threshold));
      if (!p.sliding) qo = round_bf16(qo * tau);
      sm.qs[g][t] = qo;
    }
    if (newrow) {
      const uint16_t kb = f32_to_bf16_bits(yk * l2_clip_scale(ss[kGroup], p.qk_threshold));
      const uint16_t vb = f32_to_bf16_bits(yv * (1.0f / sqrtf(ss[kGroup + 1] / float(kHd) + 1e-6f)));
      uint16_t* kst = after_wait(p.k_conv_state);
      uint16_t* vst = after_wait(p.v_conv_state);
      kst[c] = kst[C + c]; kst[C + c] = kst[2 * C + c]; kst[2 * C + c] = f32_to_bf16_bits(xk);
      vst[c] = vst[C + c]; vst[C + c] = vst[2 * C + c]; vst[2 * C + c] = f32_to_bf16_bits(xv);
      const int slot = p.sliding ? qpos % kWindow : qpos;
      kc[(size_t(slot) * kKvHeads + h) * kHd + t] = kb;
      vc[(size_t(slot) * kKvHeads + h) * kHd + t] = vb;
      sm.ks[qpos - k0][t] = kb;
      sm.vs[qpos - k0][t] = vb;
    }
    __syncthreads();          // sc (sm.lg) is reused by split_rows
  }
  if (fuse) {
    // the new rows j of this split (positions pos + j in [k0, k0 + kn)): K / V as kv_rows computes them (conv over the
    // three previous inputs: rows j - 3 .. j - 1, else the read-only conv state); row r's block stores row r
    // all rows' convolutions first, their 2 x n sums of squares in one reduction (each as block_sum128 orders it)
    const int c = h * kHd + t, C = kKvHeads * kHd;
    const int j0 = max(0, k0 - pos), j1 = min(r, k0 + kn - 1 - pos);
    float y[2][kMaxRows], xin[2][kMaxRows];
#pragma unroll
    for (int kv = 0; kv < 2; ++kv) {
      const bool is_k = kv == 0;
      const uint16_t* lin = after_wait(is_k ? p.k_lin : p.v_lin);
      const uint16_t* st = after_wait(is_k ? p.k_conv_state : p.v_conv_state);
#pragma unroll
      for (int q = 0; q < kMaxRows; ++q) {
        const int j = j0 + q;
        y[kv][q] = 0.0f;
        xin[kv][q] = 0.0f;
        if (j > j1) continue;
        float s3[3];
#pragma unroll
        for (int jj = 0; jj < 3; ++jj) {
          const int src = j - 3 + jj;
          s3[jj] = bf16_to_f32(src >= 0 ? lin[size_t(src) * C + c] : st[(3 + src) * C + c]);
        }
        const float x = bf16_to_f32(lin[size_t(j) * C + c]);
        const uint16_t* dyn = p.sliding ? nullptr : after_wait(is_k ? p.k_dyn : p.v_dyn) + size_t(j) * 32;
        y[kv][q] = kv_conv(p.sliding, is_k ? p.k_conv_w : p.v_conv_w, dyn, c, C, s3, x);
        xin[kv][q] = x;
      }
    }
    float (*sc)[4] = reinterpret_cast<float (*)[4]>(&sm.lg[0][0]);      // [2 x kMaxRows][4 warps]
#pragma unroll
    for (int kv = 0; kv < 2; ++kv)
#pragma unroll
      for (int q = 0; q < kMaxRows; ++q) {
        if (j0 + q > j1) break;
        const float w = warp_sum(y[kv][q] * y[kv][q]);
        if ((t & 31) == 0) sc[kv * kMaxRows + q][t >> 5] = w;
      }
    __syncthreads();
#pragma unroll
    for (int kv = 0; kv < 2; ++kv)
#pragma unroll
      for (int q = 0; q < kMaxRows; ++q) {
        const int j = j0 + q;
        if (j > j1) break;
        const float* v4 = sc[kv * kMaxRows + q];
        const float ss = v4[0] + v4[1] + v4[2] + v4[3];
        const float yy = y[kv][q];
        const uint16_t o = f32_to_bf16_bits(kv == 0 ? yy * l2_clip_scale(ss, p.qk_threshold)
                                                    : yy * (1.0f / sqrtf(ss / float(kHd) + 1e-6f)));
        if (kv == 0) sm.ks[pos + j - k0][t] = o;
        else sm.vs[pos + j - k0][t] = o;
        if (j == r) {
          if (p.sliding)
            after_wait(p.stage_kv)[((size_t(j) * 2 + kv) * kKvHeads + h) * kHd + t] = o;
          else
            (kv == 0 ? kc : vc)[(size_t(pos + j) * kKvHeads + h) * kHd + t] = o;
          after_wait(p.stage_in)[(size_t(j) * 2 + kv) * C + c] = f32_to_bf16_bits(xin[kv][q]);
        }
      }
    __syncthreads();
  }
  cp_async_wait_all();
  __syncthreads();
  split_rows(p, sm, sm.ks, sm.vs, h, split, r, qpos, k0, kn);
  __syncthreads();
}

// Row r of head h: its splits merged with the sinks (and the output gate); the row's counter reset.
__device__ __forceinline__ void merge_row(const AttnDecodeParams& p, int h, int r, int pos, int R) {
  const int t = threadIdx.x;
  __threadfence();
  const uint16_t* sinks = after_wait(p.sinks);
  const float* part = after_wait(p.partial);
  uint16_t* o = after_wait(p.out);
  const size_t stride = size_t(R) * kGroup * (kHd + 2);           // one split
  const int qpos = pos + r;
  const int first = p.sliding ? max(0, qpos - (kWindow - 1)) : 0;
  const int nsplit = (qpos - first + kSplitKeys) / kSplitKeys;
  const float* hb = part + size_t(h) * p.max_splits * stride + size_t(r) * kGroup * (kHd + 2);
  if (nsplit <= kMaxMergeSplits) {
    // every load of the three heads issued at once (splits' max / sum to shared memory, this thread's accumulators,
    // the sinks and the gates to registers), then the merge in the one-pass order
    __shared__ float ms[kGroup][2][kMaxMergeSplits];
    for (int i = t; i < kGroup * nsplit * 2; i += kAttnThreads) {
      const int g = i / (nsplit * 2), sp = (i / 2) % nsplit, e = i & 1;
      ms[g][e][sp] = hb[sp * stride + g * (kHd + 2) + e];
    }
    float acc[kGroup][kMaxMergeSplits], sink[kGroup], gate[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
#pragma unroll
      for (int sp = 0; sp < kMaxMergeSplits; ++sp) acc[g][sp] = sp < nsplit ? hb[sp * stride + g * (kHd + 2) + 2 + t] : 0.0f;
      const int qh = h * kGroup + g;
      sink[g] = bf16_to_f32(sinks[size_t(r) * kHeads + qh]);
      gate[g] = p.gate ? bf16_to_f32(after_wait(p.gate)[(size_t(r) * kHeads + qh) * kHd + t]) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const int qh = h * kGroup + g;
      float mx = sink[g];
      for (int sp = 0; sp < nsplit; ++sp) mx = fmaxf(mx, ms[g][0][sp]);
      float den = expf(sink[g] - mx), num = 0.0f;
#pragma unroll
      for (int sp = 0; sp < kMaxMergeSplits; ++sp) {
        if (sp >= nsplit) break;
        const float f = expf(ms[g][0][sp] - mx);
        den = fmaf(ms[g][1][sp], f, den);
        num = fmaf(acc[g][sp], f, num);
      }
      float v = num / den;
      if (p.gate) v = round_bf16(v) * (1.0f / (1.0f + expf(-gate[g])));
      o[(size_t(r) * kHeads + qh) * kHd + t] = f32_to_bf16_bits(v);
    }
    if (t == 0) after_wait(p.counters)[h * R + r] = 0;
    __syncthreads();          // ms is rewritten by the block's next row (attn_rows_kernel merges several)
    return;
  }
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    const int qh = h * kGroup + g;
    const float sink = bf16_to_f32(sinks[size_t(r) * kHeads + qh]);
    const float* base = hb + g * (kHd + 2);
    float mx = sink;
    for (int sp = 0; sp < nsplit; ++sp) mx = fmaxf(mx, base[sp * stride]);
    float den = expf(sink - mx), num = 0.0f;
#pragma unroll 4
    for (int sp = 0; sp < nsplit; ++sp) {
      const float f = expf(base[sp * stride] - mx);
      den = fmaf(base[sp * stride + 1], f, den);
      num = fmaf(base[sp * stride + 2 + t], f, num);
    }
    const size_t at = (size_t(r) * kHeads + qh) * kHd + t;
    float v = num / den;
    if (p.gate) v = round_bf16(v) * (1.0f / (1.0f + expf(-bf16_to_f32(after_wait(p.gate)[at]))));
    o[at] = f32_to_bf16_bits(v);
  }
  if (t == 0) after_wait(p.counters)[h * R + r] = 0;
}

// Fused decode attention: grid (8 KV heads, max_splits), 128 threads. rows 0 / 1: one row at the device position
// (the block holding it computes the new K / V row); a verification step (rows > 1): every row r at pos + r over
// its own window and splits (the one-row arithmetic), new rows from kv_rows (cache / staged). The last block of a
// head merges each row's splits with the sinks.
__global__ void __launch_bounds__(kAttnThreads) attn_decode_kernel(const __grid_constant__ AttnDecodeParams p) {
  __shared__ AttnSmem sm;
  __shared__ int last;
  if (!p.pf_late) prefetch_share(p.prefetch);
  const int t = threadIdx.x;
  const int h = blockIdx.x;
  const int split = blockIdx.y;
  const bool verify = p.stage_kv != nullptr;
  const int R = p.rows < 1 ? 1 : p.rows;
  // the position and the cached keys were written by earlier steps: issued before the wait (the projections)
  const int pos = *p.pos;
  AttnPre pre;
  const int r0 = blockIdx.z, qpos0 = pos + r0;
  const int first0 = p.sliding ? max(0, qpos0 - (kWindow - 1)) : 0;
  const bool active = split < (qpos0 - first0 + kSplitKeys) / kSplitKeys;
  if (active) attend_pre(p, sm, h, split, r0, pos, qpos0, verify, pre);   // (with q_pre only the keys are used)
  pdl_wait();
  if (!p.pf_late) pdl_launch();
  {
    const int r = blockIdx.z;                    // the row
    const int qpos = pos + r;
    const int first = p.sliding ? max(0, qpos - (kWindow - 1)) : 0;
    const int nsplit = (qpos - first + kSplitKeys) / kSplitKeys;
    if (split < nsplit) attend_split(p, sm, h, split, r, pos, qpos, verify, &pre);
  }
  // verification rows (pf_late): the output projection's prefetch and the dependents' launch after the split, so
  // that the projection's early weight loads do not compete with the split's key reads (decode: at the wait)
  if (p.pf_late) {
    prefetch_share(p.prefetch);
    pdl_launch();
  }
  __threadfence();
  __syncthreads();
  // the last of row r's split blocks of this head merges the row
  const int r = blockIdx.z;
  if (t == 0) last = atomicAdd(after_wait(p.counters) + h * R + r, 1) == int(gridDim.y) - 1;
  __syncthreads();
  if (!last) return;
  merge_row(p, h, r, pos, R);
}

// Prefill chunks (rows > kMaxRows): grid (8 KV heads, max_splits, row tiles of kRowTile), 128 threads. The block
// loads the split's keys for all rows of its tile once (sliding windows: the union of the rows' windows), then runs
// each row's split with the per-row arithmetic above; merges as attn_decode_kernel.
constexpr int kRowTile = 16;
struct AttnTileSmem {
  __align__(16) uint16_t ks[kSplitKeys + kRowTile][kHd];
  __align__(16) uint16_t vs[kSplitKeys + kRowTile][kHd];
  float qs[kGroup][kHd];
  float rs[kGroup][kRel];
  float lg[kGroup][kSplitKeys];
  float scratch[4];
  float red[kGroup][4];
};
__global__ void __launch_bounds__(kAttnThreads) attn_rows_kernel(const __grid_constant__ AttnDecodeParams p) {
  __shared__ AttnTileSmem sm;
  __shared__ int last[kRowTile];
  pdl_wait();
  pdl_launch();
  const int t = threadIdx.x, h = blockIdx.x, split = blockIdx.y;
  const int pos = *after_wait(p.pos);
  const int R = p.rows;
  const int r_lo = blockIdx.z * kRowTile, r_hi = min(R, r_lo + kRowTile);
  auto first_of = [&](int q) { return p.sliding ? max(0, q - (kWindow - 1)) : 0; };
  const int kmin = first_of(pos + r_lo) + split * kSplitKeys;
  const int kmax = min(first_of(pos + r_hi - 1) + (split + 1) * kSplitKeys, pos + r_hi);
  if (kmin < kmax) {
    const uint16_t* kc = after_wait(p.k_cache);
    const uint16_t* vc = after_wait(p.v_cache);
    const uint16_t* skv = after_wait(p.stage_kv);
    for (int i = t; i < (kmax - kmin) * 16; i += kAttnThreads) {
      const int key = i >> 4, part16 = i & 15, kp = kmin + key;
      if (p.sliding && kp >= pos) {
        const uint16_t* src = skv + ((size_t(kp - pos) * 2) * kKvHeads + h) * kHd + part16 * 8;
        cp_async16(&sm.ks[key][part16 * 8], src);
        cp_async16(&sm.vs[key][part16 * 8], src + kKvHeads * kHd);
      } else {
        const int slot = p.sliding ? kp % kWindow : kp;
        cp_async16(&sm.ks[key][part16 * 8], kc + (size_t(slot) * kKvHeads + h) * kHd + part16 * 8);
        cp_async16(&sm.vs[key][part16 * 8], vc + (size_t(slot) * kKvHeads + h) * kHd + part16 * 8);
      }
    }
    cp_async_wait_all();
  }
  __syncthreads();
  for (int r = r_lo; r < r_hi; ++r) {
    const int qpos = pos + r, first = first_of(qpos);
    if (split >= (qpos - first + kSplitKeys) / kSplitKeys) continue;
    const int k0 = first + split * kSplitKeys, kn = min(kSplitKeys, qpos + 1 - k0);
    const float* pre = after_wait(p.q_pre) + (size_t(r) * kKvHeads + h) * kQPre;
#pragma unroll
    for (int g = 0; g < kGroup; ++g) sm.qs[g][t] = pre[g * kHd + t];
    if (t < kGroup * kRel) sm.rs[t >> 4][t & 15] = pre[kGroup * kHd + t];
    __syncthreads();
    split_rows(p, sm, sm.ks + (k0 - kmin), sm.vs + (k0 - kmin), h, split, r, qpos, k0, kn);
    __syncthreads();
  }
  __threadfence();
  __syncthreads();
  if (t < r_hi - r_lo) last[t] = atomicAdd(after_wait(p.counters) + h * R + r_lo + t, 1) == int(gridDim.y) - 1;
  __syncthreads();
  for (int r = r_lo; r < r_hi; ++r)
    if (last[r - r_lo]) {
      __threadfence();
      merge_row(p, h, r, pos, R);
    }
}

// New K / V rows of a verification step: grid (rows, 8 heads, 2 = K / V), 128 threads. Row r's conv window: the
// state (read only) and the inputs of rows r - 3 .. r - 1; full layers write the cache at pos + r, sliding layers
// the staged rows; the conv inputs go to stage_in [rows][2][1024] for the commit.
__global__ void __launch_bounds__(kAttnThreads) kv_rows_kernel(const __grid_constant__ AttnDecodeParams p) {
  __shared__ float scratch[4];
  pdl_wait();
  pdl_launch();
  const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const bool is_k = blockIdx.z == 0;
  const int C = kKvHeads * kHd, c = h * kHd + t;
  const int pos = *after_wait(p.pos);
  const uint16_t* lin = after_wait(is_k ? p.k_lin : p.v_lin);
  const uint16_t* st = after_wait(is_k ? p.k_conv_state : p.v_conv_state);
  float s[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int src = r - 3 + j;
    s[j] = bf16_to_f32(src >= 0 ? lin[size_t(src) * C + c] : st[(3 + src) * C + c]);
  }
  const float x = bf16_to_f32(lin[size_t(r) * C + c]);
  const uint16_t* dyn = p.sliding ? nullptr : after_wait(is_k ? p.k_dyn : p.v_dyn) + size_t(r) * 32;
  const uint16_t out = new_kv(is_k, p.sliding, is_k ? p.k_conv_w : p.v_conv_w, dyn, c, C, s, x, p.qk_threshold, scratch);
  if (p.sliding) {
    after_wait(p.stage_kv)[((size_t(r) * 2 + (is_k ? 0 : 1)) * kKvHeads + h) * kHd + t] = out;
  } else {
    uint16_t* cache = after_wait(is_k ? p.k_cache : p.v_cache);
    cache[(size_t(pos + r) * kKvHeads + h) * kHd + t] = out;
  }
  after_wait(p.stage_in)[(size_t(r) * 2 + (is_k ? 0 : 1)) * C + c] = f32_to_bf16_bits(x);
  if (is_k && p.q_pre) {
    __shared__ float qs[kGroup * kHd], rs[kGroup * kRel];
    prepare_queries(p, h, r, pos + r, qs, rs, scratch);
    __syncthreads();
    float* pre = after_wait(p.q_pre) + (size_t(r) * kKvHeads + h) * kQPre;
#pragma unroll
    for (int g = 0; g < kGroup; ++g) pre[g * kHd + t] = qs[g * kHd + t];
    if (t < kGroup * kRel) pre[kGroup * kHd + t] = rs[t];
  }
}

}  // namespace

void kv_rows(const AttnDecodeParams& p, cudaStream_t stream) {
  if (p.rows < 1 || p.rows > kMaxChunk || !p.stage_in || (p.sliding && !p.stage_kv))
    throw std::invalid_argument("kv_rows: 1..kMaxChunk rows, stage_in, and stage_kv for sliding layers");
  launch_pdl(kv_rows_kernel, dim3(p.rows, kKvHeads, 2), dim3(kAttnThreads), 0, stream, p);
}

void attn_decode(const AttnDecodeParams& p, cudaStream_t stream) {
  if (p.max_splits < 1) throw std::invalid_argument("attn_decode: max_splits >= 1");
  if (p.rows > kMaxChunk) throw std::invalid_argument("attn_decode: at most kMaxChunk rows");
  if (p.rows > kMaxRows) {
    if (!p.stage_kv && p.sliding) throw std::invalid_argument("attn_decode: prefill chunks need the staged rows");
    if (!p.q_pre) throw std::invalid_argument("attn_decode: prefill chunks need q_pre (kv_rows)");
    if (p.fuse_kv) throw std::invalid_argument("attn_decode: fuse_kv takes at most kMaxRows rows");
    launch_pdl(attn_rows_kernel, dim3(kKvHeads, p.max_splits, (p.rows + kRowTile - 1) / kRowTile), dim3(kAttnThreads),
               0, stream, p);
    return;
  }
  if (p.q_pre && !p.stage_kv) throw std::invalid_argument("attn_decode: q_pre with the staged rows (stage_kv)");
  if (p.fuse_kv && (!p.stage_kv || !p.stage_in || p.q_pre))
    throw std::invalid_argument("attn_decode: fuse_kv writes stage_kv / stage_in and computes the queries (no q_pre)");
  launch_pdl(attn_decode_kernel, dim3(kKvHeads, p.max_splits, p.rows < 1 ? 1 : p.rows), dim3(kAttnThreads), 0, stream, p);
}

int attn_split_keys() { return kSplitKeys; }

}  // namespace inkling2
}  // namespace flash_rt
