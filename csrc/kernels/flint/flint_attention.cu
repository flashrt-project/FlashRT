// SPDX-License-Identifier: Apache-2.0
//
// Flint decode attention (one query token): GQA 32/4, head_dim 128, BF16 KV,
// per-query-head sink logits, optional sliding window over a ring cache, key
// length read from device memory (graph-safe).
//
// Numerics: S = q.k (bf16 inputs, fp32 accumulate); online softmax in fp32 in
// the exp2 domain; P rounded to bf16 for P.V while the row sum accumulates the
// fp32 P; the sink logit (already in scaled units) only enlarges the
// denominator; the output is rounded once to bf16.

#include "flint_common.cuh"
#include "flint_kernels.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <cfloat>

namespace flash_rt {
namespace flint {

namespace {

constexpr int kD = 128;
constexpr int kGroup = 8;
constexpr int kSlotElems = 1024;  // [K 4x128 | V 4x128]
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Select the bf16 at index i (0..15) of a 32-byte row held as 8 u32.
__device__ __forceinline__ uint32_t pick_bf16(const uint32_t (&row)[8], int i) {
  return (i & 1) ? (row[i >> 1] >> 16) : (row[i >> 1] & 0xffffu);
}

}  // namespace

// grid (n_split, 4 kv heads), 128 threads (4 warps).
// Partials: o_part f32 [n_split][32][128] (unnormalized, relative to m), ml f32 [n_split][32][2] = (m*c, l).
__global__ void __launch_bounds__(128)
decode_attn_kernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ cache,
                   const int* __restrict__ pos_p, int ring, int window, float scale,
                   float* __restrict__ o_part, float* __restrict__ ml_part) {
  const int split = blockIdx.x, n_split = gridDim.x, kvh = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int pos = *pos_p;
  const int kstart = window > 0 ? max(0, pos - window + 1) : 0;
  const int klen = pos + 1 - kstart;
  const int per = (((klen + n_split - 1) / n_split) + 15) & ~15;
  const int b0 = kstart + split * per;
  const int b1 = min(b0 + per, pos + 1);
  const float c = scale * kLog2e;

  // Q fragments: head kvh*8 + g, dims 32t + 4j + {0..3} for k-step j.
  uint32_t qa[8][2];
  {
    const uint16_t* qh = q + (kvh * kGroup + g) * kD + 32 * t;
    const uint4* qp = reinterpret_cast<const uint4*>(qh);
    uint32_t qw[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = qp[i];
      qw[4 * i] = v.x; qw[4 * i + 1] = v.y; qw[4 * i + 2] = v.z; qw[4 * i + 3] = v.w;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) { qa[j][0] = qw[2 * j]; qa[j][1] = qw[2 * j + 1]; }
  }

  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m = -FLT_MAX;  // running max of raw scores (head g)
  float l = 0.f;       // per-lane partial row sum

  const int kv_off_k = kvh * kD, kv_off_v = 512 + kvh * kD;
  for (int tile = b0 + warp * 16; tile < b1; tile += 64) {
    // ---- S = Q K^T for 16 tokens (two n-tiles: tokens tile+g and tile+8+g) ----
    float s[2][4];
    uint32_t kr[2][16];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int p = tile + nt * 8 + g;
      if (p < b1) {
        const int slot = ring > 0 ? (p & (ring - 1)) : p;
        const uint4* kp = reinterpret_cast<const uint4*>(cache + static_cast<size_t>(slot) * kSlotElems + kv_off_k + 32 * t);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v = kp[i];
          kr[nt][4 * i] = v.x; kr[nt][4 * i + 1] = v.y; kr[nt][4 * i + 2] = v.z; kr[nt][4 * i + 3] = v.w;
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) kr[nt][i] = 0u;
      }
    }
    // V rows for tokens 2t, 2t+1, 2t+8, 2t+9 of this tile, dims [16g, 16g+16).
    uint32_t vr[4][8];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = tile + (r >> 1) * 8 + 2 * t + (r & 1);
      if (p < b1) {
        const int slot = ring > 0 ? (p & (ring - 1)) : p;
        const uint4* vp = reinterpret_cast<const uint4*>(cache + static_cast<size_t>(slot) * kSlotElems + kv_off_v + 16 * g);
        const uint4 v0 = vp[0], v1 = vp[1];
        vr[r][0] = v0.x; vr[r][1] = v0.y; vr[r][2] = v0.z; vr[r][3] = v0.w;
        vr[r][4] = v1.x; vr[r][5] = v1.y; vr[r][6] = v1.z; vr[r][7] = v1.w;
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) vr[r][i] = 0u;
      }
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        mma_bf16_16816(s[nt], qa[j][0], 0u, qa[j][1], 0u, kr[nt][2 * j], kr[nt][2 * j + 1]);
    }
    // ---- online softmax (head g; this lane: tokens 2t,2t+1 and 8+2t,8+2t+1) ----
    float sc[4] = {s[0][0], s[0][1], s[1][0], s[1][1]};
    const int tok[4] = {tile + 2 * t, tile + 2 * t + 1, tile + 8 + 2 * t, tile + 9 + 2 * t};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (tok[i] >= b1) sc[i] = -FLT_MAX;
    float tmax = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
    const float m_new = fmaxf(m, tmax);
    const float alpha = exp2f((m - m_new) * c);
    float pr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) pr[i] = (tok[i] < b1) ? exp2f((sc[i] - m_new) * c) : 0.f;
    l = l * alpha + (pr[0] + pr[1] + pr[2] + pr[3]);
    m = m_new;
#pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] *= alpha; o[i][1] *= alpha; }
    const uint32_t pa0 = pack_bf16x2(pr[0], pr[1]);
    const uint32_t pa2 = pack_bf16x2(pr[2], pr[3]);
    // ---- O += P V; n-tile i covers dims 16*col + i ----
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t bb0 = pick_bf16(vr[0], i) | (pick_bf16(vr[1], i) << 16);
      const uint32_t bb1 = pick_bf16(vr[2], i) | (pick_bf16(vr[3], i) << 16);
      mma_bf16_16816(o[i], pa0, 0u, pa2, 0u, bb0, bb1);
    }
  }

  // ---- reduce the 4 lanes of a head (row sum), then combine the 4 warps ----
  l += __shfl_xor_sync(0xffffffffu, l, 1);
  l += __shfl_xor_sync(0xffffffffu, l, 2);
  __shared__ float sm_m[4][kGroup], sm_l[4][kGroup];
  __shared__ float sm_o[4][kGroup][kD];
  if (t == 0) { sm_m[warp][g] = m; sm_l[warp][g] = l; }
  // lane (g,t) holds O[head g][dims 32t + i] (c0, col 2t) and [32t + 16 + i] (c1, col 2t+1)
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    sm_o[warp][g][32 * t + i] = o[i][0];
    sm_o[warp][g][32 * t + 16 + i] = o[i][1];
  }
  __syncthreads();
  // 128 threads: thread -> (head hh = tid/16, dims 8*(tid%16) .. +8)
  const int hh = threadIdx.x >> 4, d0 = (threadIdx.x & 15) * 8;
  float mm = -FLT_MAX;
#pragma unroll
  for (int w = 0; w < 4; ++w) mm = fmaxf(mm, sm_m[w][hh]);
  float ll = 0.f, acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const float f = (sm_m[w][hh] == -FLT_MAX) ? 0.f : exp2f((sm_m[w][hh] - mm) * c);
    ll += sm_l[w][hh] * f;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] += sm_o[w][hh][d0 + i] * f;
  }
  const int head = kvh * kGroup + hh;
  float* op = o_part + (static_cast<size_t>(split) * 32 + head) * kD + d0;
#pragma unroll
  for (int i = 0; i < 8; ++i) op[i] = acc[i];
  if ((threadIdx.x & 15) == 0) {
    ml_part[(split * 32 + head) * 2] = (mm == -FLT_MAX) ? -FLT_MAX : mm * c;
    ml_part[(split * 32 + head) * 2 + 1] = ll;
  }
}

// grid 32 heads, 128 threads (one per dim).
__global__ void __launch_bounds__(128)
decode_attn_combine_kernel(const float* __restrict__ o_part, const float* __restrict__ ml_part,
                           const float* __restrict__ sinks, int n_split, uint16_t* __restrict__ out) {
  const int head = blockIdx.x, d = threadIdx.x;
  const float sk = sinks[head] * kLog2e;
  float M = sk;
  for (int s = 0; s < n_split; ++s) M = fmaxf(M, ml_part[(s * 32 + head) * 2]);
  float den = exp2f(sk - M), num = 0.f;
  for (int s = 0; s < n_split; ++s) {
    const float ms = ml_part[(s * 32 + head) * 2];
    if (ms == -FLT_MAX) continue;
    const float f = exp2f(ms - M);
    den += ml_part[(s * 32 + head) * 2 + 1] * f;
    num += o_part[(static_cast<size_t>(s) * 32 + head) * kD + d] * f;
  }
  out[head * kD + d] = f32_to_bf16_bits(num / den);
}

void decode_attention_bf16(const void* q, const void* cache, const int* pos, int ring, int window,
                           const float* sinks, float scale, int n_split, float* o_part, float* ml_part,
                           void* out, cudaStream_t s) {
  decode_attn_kernel<<<dim3(n_split, 4), 128, 0, s>>>(static_cast<const uint16_t*>(q),
                                                     static_cast<const uint16_t*>(cache), pos, ring,
                                                     window, scale, o_part, ml_part);
  decode_attn_combine_kernel<<<32, 128, 0, s>>>(o_part, ml_part, sinks, n_split, static_cast<uint16_t*>(out));
}


// ===========================================================================
// Fused decode attention (PDL): Q/K RMSNorm + RoPE + current-token KV write +
// split-KV attention with sinks. grid (n_split, 4), WARPS warps per CTA.
//  * n_split == 1: the CTA merges its warps and writes the output directly.
//  * n_split  > 1: partials go to global memory; the last CTA of each KV head
//    (self-resetting counter) merges them, so the kernel is graph-replay safe.
// Before pdl_wait the kernel reads the position (final for the whole step: the
// step's first kernel starts only after the previous argmax completed) and
// prefetches its first K/V tile when that tile lies entirely before the current
// token (older tokens are immutable); the token's own K/V row is written by the
// split that owns it, after the wait.
// ===========================================================================
template <int WARPS, bool CLUSTER>
__global__ void __launch_bounds__(WARPS * 32)
decode_attn_fused_kernel(const uint16_t* __restrict__ qkv, const float* __restrict__ qw, const float* __restrict__ kw,
                         const float* __restrict__ cos_t, const float* __restrict__ sin_t, const int* __restrict__ pos_p,
                         uint16_t* __restrict__ cache, int ring, int window, int rope, float eps,
                         const float* __restrict__ sinks, float scale, float* __restrict__ o_part,
                         float* __restrict__ ml_part, unsigned* __restrict__ counters, uint16_t* __restrict__ out,
                         const unsigned long long* __restrict__ pf_ranges, int n_pf) {
  constexpr int NT = WARPS * 32;
  constexpr int DPT = (kGroup * kD) / NT;          // output dims per thread in the merges
  constexpr int TPH = kD / DPT;                    // threads per head
  const int split = blockIdx.x, n_split = gridDim.x, kvh = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  extern __shared__ __align__(16) uint8_t dsm[];
  uint16_t (*qs)[kD] = reinterpret_cast<uint16_t (*)[kD]>(dsm);                      // [8][128] bf16
  float (*sm_o)[kGroup][kD] = reinterpret_cast<float (*)[kGroup][kD]>(dsm + kGroup * kD * 2);  // [WARPS][8][128]
  float (*sm_m)[kGroup] = reinterpret_cast<float (*)[kGroup]>(dsm + kGroup * kD * 2 + WARPS * kGroup * kD * 4);
  float (*sm_l)[kGroup] = sm_m + WARPS;

  float qwv[4], kwv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) { qwv[j] = qw[lane + 32 * j]; kwv[j] = kw[lane + 32 * j]; }
  const int pos = __ldcg(pos_p);
  const int kstart = window > 0 ? max(0, pos - window + 1) : 0;
  const int klen = pos + 1 - kstart;
  const int per = (((klen + n_split - 1) / n_split) + 15) & ~15;
  const int b0 = kstart + split * per;
  const int b1 = min(b0 + per, pos + 1);
  const float c = scale * kLog2e;
  const int kv_off_k = kvh * kD, kv_off_v = 512 + kvh * kD;

  uint32_t kr[2][16], vr[4][8];
  auto load_tile = [&](int tile) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int p = tile + nt * 8 + g;
      if (p < b1) {
        const int slot = ring > 0 ? (p & (ring - 1)) : p;
        const uint4* kp = reinterpret_cast<const uint4*>(cache + static_cast<size_t>(slot) * kSlotElems + kv_off_k + 32 * t);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v = kp[i];
          kr[nt][4 * i] = v.x; kr[nt][4 * i + 1] = v.y; kr[nt][4 * i + 2] = v.z; kr[nt][4 * i + 3] = v.w;
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) kr[nt][i] = 0u;
      }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = tile + (r >> 1) * 8 + 2 * t + (r & 1);
      if (p < b1) {
        const int slot = ring > 0 ? (p & (ring - 1)) : p;
        const uint4* vp = reinterpret_cast<const uint4*>(cache + static_cast<size_t>(slot) * kSlotElems + kv_off_v + 16 * g);
        const uint4 v0 = vp[0], v1 = vp[1];
        vr[r][0] = v0.x; vr[r][1] = v0.y; vr[r][2] = v0.z; vr[r][3] = v0.w;
        vr[r][4] = v1.x; vr[r][5] = v1.y; vr[r][6] = v1.z; vr[r][7] = v1.w;
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) vr[r][i] = 0u;
      }
    }
  };
  const int tile0 = b0 + warp * 16;
  bool have = false;
  if (tile0 < b1 && tile0 + 16 <= pos) { load_tile(tile0); have = true; }
  // Optional L2 warm-up of the layer's upcoming static weights (o_proj, router, shared expert):
  // this kernel moves little data on short windows, so HBM is otherwise idle here.
  if (threadIdx.x < n_pf) {
    const unsigned long long base = pf_ranges[2 * threadIdx.x], bytes = pf_ranges[2 * threadIdx.x + 1];
    const unsigned nblk = gridDim.x * gridDim.y, bid = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned long long sl = ((bytes / nblk) + 15) & ~15ull;
    const unsigned long long off = sl * bid;
    if (off < bytes) {
      const unsigned long long len = (off + sl <= bytes ? sl : bytes - off) & ~15ull;
      if (len) prefetch_l2_bulk(reinterpret_cast<const void*>(base + off), static_cast<uint32_t>(len));
    }
  }

  pdl_wait();

  auto norm_rope = [&](const uint16_t* src, const float (&w)[4], float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = bf16_to_f32(src[lane + 32 * j]);
    const float ss = warp_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    const float rs = rsqrtf(ss / kD + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = round_bf16(v[j] * rs * w[j]);
    if (rope) {
      const float* ct = cos_t + static_cast<size_t>(pos) * 64;
      const float* st = sin_t + static_cast<size_t>(pos) * 64;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int d = lane + 32 * j;
        const float cs = ct[d], sn = st[d], n1 = v[j], n2 = v[j + 2];
        v[j] = fmaf(n1, cs, -(n2 * sn));
        v[j + 2] = fmaf(n2, cs, n1 * sn);
      }
    }
  };
  for (int h = warp; h < kGroup; h += WARPS) {
    float v[4];
    norm_rope(qkv + (kvh * kGroup + h) * kD, qwv, v);
#pragma unroll
    for (int j = 0; j < 4; ++j) qs[h][lane + 32 * j] = f32_to_bf16_bits(v[j]);
  }
  if (b0 <= pos && pos < b1 && warp >= WARPS - 2) {   // last two warps: often idle in the tile loop
    const int kvsel = warp - (WARPS - 2);            // 0 = K, 1 = V
    const int slot = ring > 0 ? (pos & (ring - 1)) : pos;
    uint16_t* dst = cache + static_cast<size_t>(slot) * kSlotElems + kvsel * 512 + kvh * kD;
    const uint16_t* src = qkv + (32 + kvsel * 4 + kvh) * kD;
    if (kvsel == 0) {
      float v[4];
      norm_rope(src, kwv, v);
#pragma unroll
      for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = f32_to_bf16_bits(v[j]);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = src[lane + 32 * j];
    }
  }
  __syncthreads();

  uint32_t qa[8][2];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    qa[j][0] = *reinterpret_cast<const uint32_t*>(&qs[g][32 * t + 4 * j]);
    qa[j][1] = *reinterpret_cast<const uint32_t*>(&qs[g][32 * t + 4 * j + 2]);
  }
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m = -FLT_MAX, l = 0.f;
  for (int tile = tile0; tile < b1; tile += 16 * WARPS) {
    if (!have) load_tile(tile);
    have = false;
    float s[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) mma_bf16_16816(s[nt], qa[j][0], 0u, qa[j][1], 0u, kr[nt][2 * j], kr[nt][2 * j + 1]);
    }
    float sc[4] = {s[0][0], s[0][1], s[1][0], s[1][1]};
    const int tok[4] = {tile + 2 * t, tile + 2 * t + 1, tile + 8 + 2 * t, tile + 9 + 2 * t};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (tok[i] >= b1) sc[i] = -FLT_MAX;
    float tmax = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
    const float m_new = fmaxf(m, tmax);
    const float alpha = exp2f((m - m_new) * c);
    float pr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) pr[i] = (tok[i] < b1) ? exp2f((sc[i] - m_new) * c) : 0.f;
    l = l * alpha + (pr[0] + pr[1] + pr[2] + pr[3]);
    m = m_new;
#pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] *= alpha; o[i][1] *= alpha; }
    const uint32_t pa0 = pack_bf16x2(pr[0], pr[1]);
    const uint32_t pa2 = pack_bf16x2(pr[2], pr[3]);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t bb0 = pick_bf16(vr[0], i) | (pick_bf16(vr[1], i) << 16);
      const uint32_t bb1 = pick_bf16(vr[2], i) | (pick_bf16(vr[3], i) << 16);
      mma_bf16_16816(o[i], pa0, 0u, pa2, 0u, bb0, bb1);
    }
  }
  l += __shfl_xor_sync(0xffffffffu, l, 1);
  l += __shfl_xor_sync(0xffffffffu, l, 2);
  if (t == 0) { sm_m[warp][g] = m; sm_l[warp][g] = l; }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    sm_o[warp][g][32 * t + i] = o[i][0];
    sm_o[warp][g][32 * t + 16 + i] = o[i][1];
  }
  __syncthreads();
  const int hh = threadIdx.x / TPH, d0 = (threadIdx.x % TPH) * DPT;
  const int head = kvh * kGroup + hh;
  float mm = -FLT_MAX;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][hh]);
  float ll = 0.f, acc[DPT];
#pragma unroll
  for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) {
    const float f = (sm_m[w][hh] == -FLT_MAX) ? 0.f : exp2f((sm_m[w][hh] - mm) * c);
    ll += sm_l[w][hh] * f;
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] += sm_o[w][hh][d0 + i] * f;
  }
  const float sk = sinks[head] * kLog2e;
  if (n_split == 1) {
    // single split: fold the sink and normalize here
    const float m2 = (mm == -FLT_MAX) ? sk : fmaxf(mm * c, sk);
    const float fo = (mm == -FLT_MAX) ? 0.f : exp2f(mm * c - m2);
    const float inv = 1.f / (ll * fo + exp2f(sk - m2));
#pragma unroll
    for (int i = 0; i < DPT; i += 2)
      *reinterpret_cast<uint32_t*>(out + head * kD + d0 + i) = pack_bf16x2(acc[i] * fo * inv, acc[i + 1] * fo * inv);
    pdl_launch();
    return;
  }
  if (CLUSTER) {
    // Cluster of the n_split CTAs of this KV head: publish the CTA partial in shared memory,
    // then rank r merges query head(s) r, r + n_split, ... reading its peers through DSMEM.
    float (*cp_o)[kD] = reinterpret_cast<float (*)[kD]>(&sm_o[0][0][0]);   // reuse: [8][128] partial
    float* cp_m = &sm_m[0][0];
    float* cp_l = &sm_l[0][0];
    __syncthreads();   // everyone finished reading sm_o/sm_m/sm_l of the warp merge
#pragma unroll
    for (int i = 0; i < DPT; ++i) cp_o[hh][d0 + i] = acc[i];
    if ((threadIdx.x % TPH) == 0) { cp_m[hh] = (mm == -FLT_MAX) ? -FLT_MAX : mm * c; cp_l[hh] = ll; }
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    const int rank = static_cast<int>(cl.block_rank());
    for (int h2 = rank; h2 < kGroup; h2 += n_split) {
      const int hd = kvh * kGroup + h2;
      const float sk2 = sinks[hd] * kLog2e;
      for (int d = threadIdx.x; d < kD; d += NT) {
        float M = sk2;
        for (int q = 0; q < n_split; ++q) M = fmaxf(M, *cl.map_shared_rank(cp_m + h2, q));
        float den = exp2f(sk2 - M), num = 0.f;
        for (int q = 0; q < n_split; ++q) {
          const float mq = *cl.map_shared_rank(cp_m + h2, q);
          if (mq == -FLT_MAX) continue;
          const float f = exp2f(mq - M);
          den += *cl.map_shared_rank(cp_l + h2, q) * f;
          num += *cl.map_shared_rank(&cp_o[h2][d], q) * f;
        }
        out[hd * kD + d] = f32_to_bf16_bits(num / den);
      }
    }
    pdl_launch();
    cl.sync();         // keep shared memory alive until every peer has read it
    return;
  }
  {
    float* op = o_part + (static_cast<size_t>(split) * 32 + head) * kD + d0;
#pragma unroll
    for (int i = 0; i < DPT; ++i) op[i] = acc[i];
    if ((threadIdx.x % TPH) == 0) {
      ml_part[(split * 32 + head) * 2] = (mm == -FLT_MAX) ? -FLT_MAX : mm * c;
      ml_part[(split * 32 + head) * 2 + 1] = ll;
    }
  }
  __threadfence();
  __syncthreads();
  pdl_launch();
  __shared__ bool last;
  if (threadIdx.x == 0) last = (atomicAdd(&counters[kvh], 1u) == static_cast<unsigned>(n_split - 1));
  __syncthreads();
  if (!last) return;
  __threadfence();
  float M = sk;
  for (int sp = 0; sp < n_split; ++sp) M = fmaxf(M, __ldcg(ml_part + (sp * 32 + head) * 2));
  float den = exp2f(sk - M), num[DPT];
#pragma unroll
  for (int i = 0; i < DPT; ++i) num[i] = 0.f;
  for (int sp = 0; sp < n_split; ++sp) {
    const float ms = __ldcg(ml_part + (sp * 32 + head) * 2);
    if (ms == -FLT_MAX) continue;
    const float f = exp2f(ms - M);
    den += __ldcg(ml_part + (sp * 32 + head) * 2 + 1) * f;
    const float* op = o_part + (static_cast<size_t>(sp) * 32 + head) * kD + d0;
#pragma unroll
    for (int i = 0; i < DPT; ++i) num[i] += __ldcg(op + i) * f;
  }
  const float inv = 1.f / den;
#pragma unroll
  for (int i = 0; i < DPT; i += 2)
    *reinterpret_cast<uint32_t*>(out + head * kD + d0 + i) = pack_bf16x2(num[i] * inv, num[i + 1] * inv);
  if (threadIdx.x == 0) counters[kvh] = 0u;
}

template <int WARPS>
static void launch_attn_fused(const void* qkv, const float* qw, const float* kw, const float* cos_t, const float* sin_t,
                              const int* pos, void* cache, int ring, int window, int rope, float eps, const float* sinks,
                              float scale, int n_split, float* o_part, float* ml_part, unsigned* counters, void* out,
                              cudaStream_t s, const unsigned long long* pf_ranges, int n_pf) {
  constexpr size_t smem = kGroup * kD * 2 + WARPS * kGroup * kD * 4 + 2 * WARPS * kGroup * 4;
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(decode_attn_fused_kernel<WARPS, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
    cudaFuncSetAttribute(decode_attn_fused_kernel<WARPS, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
    init = true;
  }
  const bool cluster = n_split >= 2 && n_split <= 8 && attn_cluster_enabled();
  if (!cluster) {
    launch_pdl(decode_attn_fused_kernel<WARPS, false>, dim3(n_split, 4), dim3(WARPS * 32), smem, s,
               static_cast<const uint16_t*>(qkv), qw, kw, cos_t, sin_t, pos, static_cast<uint16_t*>(cache), ring, window,
               rope, eps, sinks, scale, o_part, ml_part, counters, static_cast<uint16_t*>(out), pf_ranges, n_pf);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(n_split, 4);
  cfg.blockDim = dim3(WARPS * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = n_split;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = pdl_enabled() ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  cudaLaunchKernelEx(&cfg, decode_attn_fused_kernel<WARPS, true>, static_cast<const uint16_t*>(qkv), qw, kw, cos_t,
                     sin_t, pos, static_cast<uint16_t*>(cache), ring, window, rope, eps, sinks, scale, o_part, ml_part,
                     counters, static_cast<uint16_t*>(out), pf_ranges, n_pf);
}

int decode_attention_fused_bf16(const void* qkv, const float* q_norm_w, const float* k_norm_w, const float* cos_t,
                                const float* sin_t, const int* pos, void* cache, int ring, int window, int rope,
                                float eps, const float* sinks, float scale, int n_split, int warps, float* o_part,
                                float* ml_part, unsigned* counters, void* out, cudaStream_t s,
                                const unsigned long long* pf_ranges, int n_pf) {
#define FLINT_ATTN_W(W)                                                                                              \
  if (warps == W) {                                                                                                  \
    launch_attn_fused<W>(qkv, q_norm_w, k_norm_w, cos_t, sin_t, pos, cache, ring, window, rope, eps, sinks, scale,  \
                         n_split, o_part, ml_part, counters, out, s, pf_ranges, n_pf);                               \
    return 0;                                                                                                        \
  }
  FLINT_ATTN_W(4)
  FLINT_ATTN_W(8)
  FLINT_ATTN_W(16)
#undef FLINT_ATTN_W
  return 1;
}

// ===========================================================================
// Prefill attention (many query tokens), FlashAttention-2 style with GQA packing.
// grid (ceil(n/16), 4 kv heads), 256 threads: warp w handles q head kvh*8 + w for the
// CTA's 16 query tokens; all 8 warps share each K/V tile staged in shared memory.
// Query row r is at position pos0 + r (pos0 = *pos_p). Keys come from the cache
// (slot = p, or p & (ring-1) for sliding layers): causal, plus sliding window.
// ===========================================================================
namespace {

constexpr int PQ = 16, PKV = 32, PSTAGES = 2;

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// row r (key), 16B chunk c (0..15) of a [PKV][128] bf16 tile, XOR swizzled
__device__ __forceinline__ int kv_off(int r, int c) { return r * kD + ((c ^ (r & 7)) << 3); }

}  // namespace

__global__ void __launch_bounds__(256)
prefill_attn_kernel(const uint16_t* __restrict__ q, int n, const uint16_t* __restrict__ cache,
                    const int* __restrict__ pos_p, int ring, int window, const float* __restrict__ sinks,
                    float scale, uint16_t* __restrict__ out) {
  __shared__ __align__(16) uint16_t ks[PSTAGES][PKV * kD];
  __shared__ __align__(16) uint16_t vs[PSTAGES][PKV * kD];
  const int kvh = blockIdx.y, q0 = blockIdx.x * PQ;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int head = kvh * kGroup + warp;
  const int pos0 = *pos_p;
  const int qlast = min(n, q0 + PQ) - 1;                   // last valid query row
  const int kv_end = pos0 + qlast + 1;                     // exclusive
  const int kv_beg = window > 0 ? max(0, pos0 + q0 - window + 1) : 0;
  const float c = scale * kLog2e;

  // Q fragments (rows g, g+8 = tokens q0+g, q0+8+g)
  uint32_t qa[8][4];
  {
    const int r0 = q0 + g, r1 = q0 + g + 8;
    const uint16_t* p0 = q + (static_cast<size_t>(min(r0, n - 1)) * 32 + head) * kD;
    const uint16_t* p1 = q + (static_cast<size_t>(min(r1, n - 1)) * 32 + head) * kD;
#pragma unroll
    for (int k8 = 0; k8 < 8; ++k8) {
      qa[k8][0] = *reinterpret_cast<const uint32_t*>(p0 + 16 * k8 + 2 * t);
      qa[k8][1] = *reinterpret_cast<const uint32_t*>(p1 + 16 * k8 + 2 * t);
      qa[k8][2] = *reinterpret_cast<const uint32_t*>(p0 + 16 * k8 + 8 + 2 * t);
      qa[k8][3] = *reinterpret_cast<const uint32_t*>(p1 + 16 * k8 + 8 + 2 * t);
    }
  }
  const int pq[2] = {pos0 + q0 + g, pos0 + q0 + g + 8};

  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  const float sk = sinks[head] * kLog2e;
  float m2[2] = {sk, sk};                                  // running max (log2 units), sink included
  float l[2] = {t == 0 ? 1.f : 0.f, t == 0 ? 1.f : 0.f};   // sink contributes exp2(sk - m) = 1 once per row

  auto load_tile = [&](int st, int kb) {
    // 32 keys x (16 K chunks + 16 V chunks) = 1024 chunks / 256 threads
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = threadIdx.x + i * 256;
      const int r = idx >> 5, cc = idx & 31, isv = cc >> 4, ch = cc & 15;
      const int p = kb + r;
      const bool ok = p < kv_end;
      const int slot = ring > 0 ? (p & (ring - 1)) : p;
      const uint16_t* src = cache + static_cast<size_t>(ok ? slot : 0) * kSlotElems + isv * 512 + kvh * kD + ch * 8;
      uint16_t* dst = (isv ? vs[st] : ks[st]) + kv_off(r, ch);
      cpa16(smem_addr(dst), src, ok);
    }
    asm volatile("cp.async.commit_group;");
  };

  const int ntiles = (kv_end - kv_beg + PKV - 1) / PKV;
  if (ntiles > 0) load_tile(0, kv_beg);
  for (int it = 0; it < ntiles; ++it) {
    const int kb = kv_beg + it * PKV, st = it & 1;
    if (it + 1 < ntiles) {
      load_tile(st ^ 1, kb + PKV);
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
    // ---- S = Q K^T : 4 n-tiles of 8 keys ----
    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2) {  // pairs of k-steps (32 dims)
        uint32_t b[4];
        const int r = 8 * j + (lane & 7), ch = 4 * k2 + (lane >> 3);
        ldsm_x4(b, smem_addr(ks[st] + kv_off(r, ch)));
        mma_bf16_16816(s[j], qa[2 * k2][0], qa[2 * k2][1], qa[2 * k2][2], qa[2 * k2][3], b[0], b[1]);
        mma_bf16_16816(s[j], qa[2 * k2 + 1][0], qa[2 * k2 + 1][1], qa[2 * k2 + 1][2], qa[2 * k2 + 1][3], b[2], b[3]);
      }
    }
    // ---- mask + online softmax (rows g, g+8) ----
    float rmax[2] = {-FLT_MAX, -FLT_MAX};
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1, key = kb + 8 * j + 2 * t + (e & 1);
        const int qp = pq[h];
        const bool ok = key < kv_end && key <= qp && (window <= 0 || qp - key < window);
        s[j][e] = ok ? s[j][e] * c : -FLT_MAX;
        rmax[h] = fmaxf(rmax[h], s[j][e]);
      }
    float alpha[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 1));
      rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 2));
      const float mn = fmaxf(m2[h], rmax[h]);
      alpha[h] = exp2f(m2[h] - mn);
      m2[h] = mn;
      l[h] *= alpha[h];
    }
    uint32_t pa[2][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float p[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        p[e] = s[j][e] == -FLT_MAX ? 0.f : exp2f(s[j][e] - m2[e >> 1]);
        l[e >> 1] += p[e];
      }
      // k-step kk = j/2 (16 keys = n-tiles 2kk, 2kk+1)
      pa[j >> 1][(j & 1) * 2 + 0] = pack_bf16x2(p[0], p[1]);
      pa[j >> 1][(j & 1) * 2 + 1] = pack_bf16x2(p[2], p[3]);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) { o[i][0] *= alpha[0]; o[i][1] *= alpha[0]; o[i][2] *= alpha[1]; o[i][3] *= alpha[1]; }
    // ---- O += P V : 2 k-steps (16 keys) x 16 n-tiles (8 dims) ----
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
      for (int d2 = 0; d2 < 8; ++d2) {  // pairs of dim n-tiles
        uint32_t b[4];
        const int r = 16 * kk + (lane & 7) + ((lane >> 3) & 1) * 8, ch = 2 * d2 + (lane >> 4);
        ldsm_x4_t(b, smem_addr(vs[st] + kv_off(r, ch)));
        // A layout: a0 row g keys 2t.., a1 row g+8, a2 row g keys 8+2t.., a3 row g+8
        mma_bf16_16816(o[2 * d2], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b[0], b[1]);
        mma_bf16_16816(o[2 * d2 + 1], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b[2], b[3]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = q0 + g + 8 * h;
    if (r >= n) continue;
    const float inv = 1.f / l[h];
    uint16_t* dst = out + (static_cast<size_t>(r) * 32 + head) * kD;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      *reinterpret_cast<uint32_t*>(dst + 8 * i + 2 * t) = pack_bf16x2(o[i][2 * h] * inv, o[i][2 * h + 1] * inv);
  }
}

void prefill_attention_bf16(const void* q, int n, const void* cache, const int* pos, int ring, int window,
                            const float* sinks, float scale, void* out, cudaStream_t s) {
  prefill_attn_kernel<<<dim3((n + PQ - 1) / PQ, 4), 256, 0, s>>>(
      static_cast<const uint16_t*>(q), n, static_cast<const uint16_t*>(cache), pos, ring, window, sinks, scale,
      static_cast<uint16_t*>(out));
}

}  // namespace flint
}  // namespace flash_rt
