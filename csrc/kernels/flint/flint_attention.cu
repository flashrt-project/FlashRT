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
