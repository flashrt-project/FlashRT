// SPDX-License-Identifier: Apache-2.0
//
// Split-KV decode attention over a paged KV cache (vLLM / FlashInfer block
// tables), batched over sequences: the single-token kernel of
// flint_attention.cu with the slot computation replaced by a block-table
// lookup and a sequence grid dimension. Numerics are unchanged: bf16 QK^T and
// PV on tensor cores with fp32 accumulation, fp32 online softmax in base 2
// with the running max in unscaled logits, bf16 P, the sink logit joins the
// denominator in the combine step, one bf16 rounding of the output.
//
// Cache views: K and V as [num_blocks, block_size, kv_heads, 128] bf16 with
// arbitrary strides (block, token, head) and unit stride along the head dim;
// token p of a sequence lives in block block_table[p / block_size], row
// p % block_size. Window w > 0 keeps keys [pos - w + 1, pos].

#include "flint_common.cuh"
#include "flint_kernels.cuh"

#include <cfloat>

namespace flash_rt {
namespace flint {

namespace {

constexpr int kD = 128;
constexpr int kGroup = 8;
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ void mma_bf16_16816p(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                                uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16_16816p(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <bool F16>
__device__ __forceinline__ void mma_kv(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                       uint32_t b0, uint32_t b1) {
  if constexpr (F16) mma_f16_16816p(d, a0, a1, a2, a3, b0, b1);
  else mma_bf16_16816p(d, a0, a1, a2, a3, b0, b1);
}
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  __half2 h = __floats2half2_rn(bf16lo(v), bf16hi(v));
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pick_bf16(const uint32_t (&row)[8], int i) {
  return (i & 1) ? (row[i >> 1] >> 16) : (row[i >> 1] & 0xffffu);
}

// Two E4M3 bytes -> f16x2 (unit KV scale; exact). The FP8 path runs its MMAs in f16.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2p(uint32_t pair) {
  uint32_t h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(static_cast<uint16_t>(pair)));
  return h;
}
// bf16 (as fp32) pair -> two E4M3 bytes (RN, saturating).
__device__ __forceinline__ uint32_t f2_to_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// 32 contiguous KV elements of a row (this lane's slice) as 16 bf16x2 words.
template <bool FP8>
__device__ __forceinline__ void load_kv32(const uint16_t* p, uint32_t (&w)[16]) {
  if constexpr (!FP8) {
    const uint4* kp = reinterpret_cast<const uint4*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = kp[i];
      w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w;
    }
  } else {
    const uint4* kp = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(p));
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint4 v = kp[i];
      const uint32_t b[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        w[8 * i + 2 * j] = e4m3x2_to_f16x2p(b[j] & 0xffffu);
        w[8 * i + 2 * j + 1] = e4m3x2_to_f16x2p(b[j] >> 16);
      }
    }
  }
}
// 16 contiguous V elements (dims 16g..16g+16) as 8 bf16x2 words.
template <bool FP8>
__device__ __forceinline__ void load_v16(const uint16_t* p, uint32_t (&w)[8]) {
  if constexpr (!FP8) {
    const uint4* vp = reinterpret_cast<const uint4*>(p);
    const uint4 v0 = vp[0], v1 = vp[1];
    w[0] = v0.x; w[1] = v0.y; w[2] = v0.z; w[3] = v0.w; w[4] = v1.x; w[5] = v1.y; w[6] = v1.z; w[7] = v1.w;
  } else {
    const uint4 v = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(p));
    const uint32_t b[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      w[2 * j] = e4m3x2_to_f16x2p(b[j] & 0xffffu);
      w[2 * j + 1] = e4m3x2_to_f16x2p(b[j] >> 16);
    }
  }
}

struct PagedKV {
  const uint16_t* k;     // bf16 cache, or e4m3 bytes reinterpreted (FP8): addresses below are in cache elements
  const uint16_t* v;
  long long sb, st, sh;  // element strides: block, token, head
  const int* bt;         // block tables [n_seq, bt_stride]
  int bt_stride, block_size;
  // KVM 2: V as NVFP4 (E2M1 codes, one E4M3 scale per 16 dims) in a shadow cache: codes [block][token][head][64 B],
  // scales [block][token][head][8 B]; byte strides (block, token, head) for each
  const uint8_t* v4;
  const uint8_t* v4s;
  long long c4b, c4t, c4h, s4b, s4t, s4h;
};

template <bool FP8>
__device__ __forceinline__ const uint16_t* kv_elem(const uint16_t* base, long long elem) {
  return FP8 ? reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(base) + elem) : base + elem;
}

struct QkRopeIn {  // fused Q/K RMSNorm + NeoX RoPE + current-token KV write (decode_attention_paged_qkv_bf16)
  const uint16_t* qkv;    // [n_seq, 5120] bf16: q 32x128 | k 4x128 | v 4x128
  long long qkv_stride;   // elements
  const long long* pos;   // [n_seq]
  const float* q_w;       // [128]
  const float* k_w;       // [128]
  const float* cos_sin;   // [max_pos, 128] fp32 (cos | sin), nullptr = no RoPE
  long long cs_stride;
  float eps;
};

// One head: 128 bf16 -> RMSNorm (bf16-rounded) -> NeoX RoPE (lane l: dims l, l+32 | l+64, l+96).
__device__ __forceinline__ void norm_rope_head(const uint16_t* __restrict__ src, const float* __restrict__ w, float eps,
                                               const float* __restrict__ cs, int lane, float (&v)[4]) {
  const int d[4] = {lane, lane + 32, lane + 64, lane + 96};
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = bf16_to_f32(src[d[i]]);
  const float ss = warp_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
  const float rs = rsqrtf(ss / kD + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = round_bf16(v[i] * rs * w[d[i]]);
  if (cs != nullptr) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float c = cs[d[i]], sn = cs[64 + d[i]], n1 = v[i], n2 = v[i + 2];
      v[i] = n1 * c - n2 * sn;
      v[i + 2] = n2 * c + n1 * sn;
    }
  }
}

// Shared body: Q fragments from q_head (8 heads x 128 bf16, global or shared), K/V tiles through the block table.
// NPF: tiles kept in flight per warp beyond the one being consumed (raw K/V bytes in registers).
// When gridDim.x == 1 the CTA normalizes with the sink itself and writes bf16 `out` (no partials, no combine).
// KVM: 0 bf16 K/V, 1 fp8 K/V, 2 fp8 K + NVFP4 V (shadow cache)
template <int KVM, int NPF>
__device__ __forceinline__ void decode_attn_paged_body(const uint16_t* __restrict__ q_head, PagedKV kv, int len,
                                                       int window, float scale, float* __restrict__ o_part,
                                                       float* __restrict__ ml_part, const int* __restrict__ bt,
                                                       const float* __restrict__ sinks, uint16_t* __restrict__ out) {
  constexpr bool FP8 = KVM >= 1;
  constexpr bool V4 = KVM == 2;
  const int split = blockIdx.x, n_split = gridDim.x, kvh = blockIdx.y, seq = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int pos = len - 1;
  const int kstart = (window > 0 && len > 0) ? max(0, pos - window + 1) : 0;
  const int klen = max(0, len - kstart);
  const int per = (((klen + n_split - 1) / n_split) + 15) & ~15;
  const int b0 = kstart + split * per;
  const int b1 = min(b0 + per, len);
  const float c = scale * kLog2e;
  (void)pos;

  uint32_t qa[8][2];
  {
    const uint16_t* qh = q_head + g * kD + 32 * t;
    const uint4* qp = reinterpret_cast<const uint4*>(qh);
    uint32_t qw[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = qp[i];
      qw[4 * i] = v.x; qw[4 * i + 1] = v.y; qw[4 * i + 2] = v.z; qw[4 * i + 3] = v.w;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) { qa[j][0] = qw[2 * j]; qa[j][1] = qw[2 * j + 1]; }
    if constexpr (FP8) {
#pragma unroll
      for (int j = 0; j < 8; ++j) { qa[j][0] = bf16x2_to_f16x2(qa[j][0]); qa[j][1] = bf16x2_to_f16x2(qa[j][1]); }
    }
  }
  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m = -FLT_MAX;
  float l = 0.f;

  auto row_ptr = [&](const uint16_t* base, int p) {
    return kv_elem<FP8>(base, static_cast<long long>(bt[p / kv.block_size]) * kv.sb +
                                  static_cast<long long>(p % kv.block_size) * kv.st +
                                  static_cast<long long>(kvh) * kv.sh);
  };

  // Tile loop, software-pipelined: the raw K/V bytes of the next tile (one DRAM round trip) are requested before
  // the current tile is dequantized and multiplied, so every warp keeps two tiles in flight.
  constexpr int RK = FP8 ? 2 : 4;   // uint4 per K row slice (32 elements)
  constexpr int RV = FP8 ? 1 : 2;   // uint4 per V row slice (16 elements); V4: .x/.y codes, .z scale byte
  uint4 rk[2][RK], rv[4][RV];
  auto fetch = [&](int tile, uint4 (&fk)[2][RK], uint4 (&fv)[4][RV]) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int p = tile + nt * 8 + g;
      if (p < b1) {
        const uint4* kp = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(kv_elem<FP8>(row_ptr(kv.k, p), 32 * t)));
#pragma unroll
        for (int i = 0; i < RK; ++i) fk[nt][i] = kp[i];
      } else {
#pragma unroll
        for (int i = 0; i < RK; ++i) fk[nt][i] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int p = tile + (r >> 1) * 8 + 2 * t + (r & 1);
      if (p < b1) {
        if constexpr (V4) {
          const long long blk = bt[p / kv.block_size], tok = p % kv.block_size;
          const uint8_t* cp = kv.v4 + blk * kv.c4b + tok * kv.c4t + static_cast<long long>(kvh) * kv.c4h + 8 * g;
          const uint8_t* sp = kv.v4s + blk * kv.s4b + tok * kv.s4t + static_cast<long long>(kvh) * kv.s4h + g;
          const uint2 cw = *reinterpret_cast<const uint2*>(cp);
          fv[r][0] = make_uint4(cw.x, cw.y, static_cast<uint32_t>(*sp), 0u);
        } else {
          const uint4* vp = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(kv_elem<FP8>(row_ptr(kv.v, p), 16 * g)));
#pragma unroll
          for (int i = 0; i < RV; ++i) fv[r][i] = vp[i];
        }
      } else {
#pragma unroll
        for (int i = 0; i < RV; ++i) fv[r][i] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  auto unpack_k = [&](const uint4 (&fk)[RK], uint32_t (&w)[16]) {
    if constexpr (!FP8) {
#pragma unroll
      for (int i = 0; i < 4; ++i) { w[4 * i] = fk[i].x; w[4 * i + 1] = fk[i].y; w[4 * i + 2] = fk[i].z; w[4 * i + 3] = fk[i].w; }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint32_t b[4] = {fk[i].x, fk[i].y, fk[i].z, fk[i].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          w[8 * i + 2 * j] = e4m3x2_to_f16x2p(b[j] & 0xffffu);
          w[8 * i + 2 * j + 1] = e4m3x2_to_f16x2p(b[j] >> 16);
        }
      }
    }
  };
  auto unpack_v = [&](const uint4 (&fv)[RV], uint32_t (&w)[8]) {
    if constexpr (V4) {
      // 16 E2M1 values x one E4M3 scale -> 8 f16x2 words (exact products, as in the weight dequantization)
      uint32_t h0[4], h1[4];
      e2m1x8_to_f16x2(fv[0].x, h0);
      e2m1x8_to_f16x2(fv[0].y, h1);
      const uint32_t s2 = e4m3_to_f16x2(fv[0].z & 0xffu);
#pragma unroll
      for (int i = 0; i < 4; ++i) { w[i] = hmul2_u32(h0[i], s2); w[4 + i] = hmul2_u32(h1[i], s2); }
    } else if constexpr (!FP8) {
      w[0] = fv[0].x; w[1] = fv[0].y; w[2] = fv[0].z; w[3] = fv[0].w; w[4] = fv[1].x; w[5] = fv[1].y; w[6] = fv[1].z; w[7] = fv[1].w;
    } else {
      const uint32_t b[4] = {fv[0].x, fv[0].y, fv[0].z, fv[0].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) { w[2 * j] = e4m3x2_to_f16x2p(b[j] & 0xffffu); w[2 * j + 1] = e4m3x2_to_f16x2p(b[j] >> 16); }
    }
  };
  // Slot ring: NPF+1 tiles in flight per warp; after a slot is consumed it is refilled with the tile
  // (NPF+1)*64 ahead, so the loads of the next NPF tiles overlap this tile's dequantization and MMAs.
  constexpr int NS = NPF + 1;
  uint4 rks[NS][2][RK], rvs[NS][4][RV];
  const int tile0 = b0 + warp * 16;
#pragma unroll
  for (int sl = 0; sl < NS; ++sl)
    if (tile0 + sl * 64 < b1) fetch(tile0 + sl * 64, rks[sl], rvs[sl]);
  auto consume = [&](int tile, const uint4 (&fk)[2][RK], const uint4 (&fv)[4][RV]) {
    float s[2][4];
    uint32_t kr[2][16];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) unpack_k(fk[nt], kr[nt]);
    uint32_t vr[4][8];
#pragma unroll
    for (int r = 0; r < 4; ++r) unpack_v(fv[r], vr[r]);
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        mma_kv<FP8>(s[nt], qa[j][0], 0u, qa[j][1], 0u, kr[nt][2 * j], kr[nt][2 * j + 1]);
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
    const uint32_t pa0 = FP8 ? pack_f16x2(pr[0], pr[1]) : pack_bf16x2(pr[0], pr[1]);
    const uint32_t pa2 = FP8 ? pack_f16x2(pr[2], pr[3]) : pack_bf16x2(pr[2], pr[3]);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t bb0 = pick_bf16(vr[0], i) | (pick_bf16(vr[1], i) << 16);
      const uint32_t bb1 = pick_bf16(vr[2], i) | (pick_bf16(vr[3], i) << 16);
      mma_kv<FP8>(o[i], pa0, 0u, pa2, 0u, bb0, bb1);
    }
  };
  for (int tile = tile0; tile < b1; tile += 64 * NS) {
#pragma unroll
    for (int sl = 0; sl < NS; ++sl) {
      const int tt = tile + sl * 64;
      if (tt >= b1) break;
      consume(tt, rks[sl], rvs[sl]);
      if (tt + 64 * NS < b1) fetch(tt + 64 * NS, rks[sl], rvs[sl]);
    }
  }

  l += __shfl_xor_sync(0xffffffffu, l, 1);
  l += __shfl_xor_sync(0xffffffffu, l, 2);
  __shared__ float sm_m[4][kGroup], sm_l[4][kGroup];
  __shared__ float sm_o[4][kGroup][kD];
  if (t == 0) { sm_m[warp][g] = m; sm_l[warp][g] = l; }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    sm_o[warp][g][32 * t + i] = o[i][0];
    sm_o[warp][g][32 * t + 16 + i] = o[i][1];
  }
  __syncthreads();
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
  if (n_split == 1 && out != nullptr) {
    // single split: fold the sink in here (same arithmetic as the combine kernel) and store bf16
    const float sk = sinks[head] * kLog2e;
    const float ms = (mm == -FLT_MAX) ? -FLT_MAX : mm * c;
    const float M = fmaxf(sk, ms);
    const float f = (ms == -FLT_MAX) ? 0.f : exp2f(ms - M);
    const float den = exp2f(sk - M) + ll * f;
    uint16_t* dst = out + (static_cast<size_t>(seq) * 32 + head) * kD + d0;
    uint32_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      packed[i] = f32_to_bf16_bits(acc[2 * i] * f / den) | (static_cast<uint32_t>(f32_to_bf16_bits(acc[2 * i + 1] * f / den)) << 16);
    *reinterpret_cast<uint4*>(dst) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    return;
  }
  const size_t part = (static_cast<size_t>(seq) * n_split + split) * 32 + head;
  float* op = o_part + part * kD + d0;
#pragma unroll
  for (int i = 0; i < 8; ++i) op[i] = acc[i];
  if ((threadIdx.x & 15) == 0) {
    ml_part[part * 2] = (mm == -FLT_MAX) ? -FLT_MAX : mm * c;
    ml_part[part * 2 + 1] = ll;
  }
}


// grid (n_split, 4 kv heads, n_seq), 128 threads (4 warps). q [n_seq, 32, 128] already normed / roped.
// Partials: o_part f32 [n_seq][n_split][32][128], ml_part f32 [n_seq][n_split][32][2] = (m*c, l).
template <int KVM, int NPF>
__global__ void __launch_bounds__(128)
decode_attn_paged_kernel(const uint16_t* __restrict__ q, PagedKV kv, const int* __restrict__ seq_lens, int window,
                         float scale, float* __restrict__ o_part, float* __restrict__ ml_part, int early,
                         const float* __restrict__ sinks, uint16_t* __restrict__ out) {
  pdl_wait();
  if (early) pdl_launch();
  const int seq = blockIdx.z, kvh = blockIdx.y;
  decode_attn_paged_body<KVM, NPF>(q + (static_cast<size_t>(seq) * 32 + kvh * kGroup) * kD, kv, seq_lens[seq], window,
                                   scale, o_part, ml_part, kv.bt + static_cast<size_t>(seq) * kv.bt_stride, sinks, out);
}

// Fused variant: Q/K RMSNorm + RoPE from the projection output; the split holding the current token writes
// its K/V row into the paged cache (older rows are immutable, so no other CTA depends on it).
template <int KVM, int NPF>
__global__ void __launch_bounds__(128)
decode_attn_paged_qkv_kernel(QkRopeIn in, PagedKV kv, const int* __restrict__ seq_lens, int window, float scale,
                             float* __restrict__ o_part, float* __restrict__ ml_part, int early, const float* __restrict__ sinks, uint16_t* __restrict__ out) {
  constexpr bool FP8 = KVM >= 1;
  __shared__ __align__(16) uint16_t q_s[kGroup * kD];
  const int split = blockIdx.x, n_split = gridDim.x, kvh = blockIdx.y, seq = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  pdl_wait();
  if (early) pdl_launch();
  const int len = seq_lens[seq];
  const int pos = len - 1;
  const long long p = in.pos[seq];
  const float* cs = in.cos_sin != nullptr ? in.cos_sin + p * in.cs_stride : nullptr;
  const uint16_t* row = in.qkv + static_cast<size_t>(seq) * in.qkv_stride;
  const int* bt = kv.bt + static_cast<size_t>(seq) * kv.bt_stride;
  // q heads kvh*8 .. +8: warp w handles heads 2w, 2w+1
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int h = 2 * warp + i;
    float v[4];
    norm_rope_head(row + (kvh * kGroup + h) * kD, in.q_w, in.eps, cs, lane, v);
    q_s[h * kD + lane] = f32_to_bf16_bits(v[0]); q_s[h * kD + lane + 32] = f32_to_bf16_bits(v[1]);
    q_s[h * kD + lane + 64] = f32_to_bf16_bits(v[2]); q_s[h * kD + lane + 96] = f32_to_bf16_bits(v[3]);
  }
  // current token's K (normed + roped) and V for this kv head, written by the split that will read it
  {
    const int kstart = (window > 0 && len > 0) ? max(0, pos - window + 1) : 0;
    const int klen = max(0, len - kstart);
    const int per = (((klen + n_split - 1) / n_split) + 15) & ~15;
    const int b0 = kstart + split * per;
    const int b1 = min(b0 + per, len);
    if (len > 0 && pos >= b0 && pos < b1 && warp == 0) {
      float kvv[4];
      norm_rope_head(row + (32 + kvh) * kD, in.k_w, in.eps, cs, lane, kvv);
      const long long base = static_cast<long long>(bt[pos / kv.block_size]) * kv.sb +
                             static_cast<long long>(pos % kv.block_size) * kv.st + static_cast<long long>(kvh) * kv.sh;
      const uint16_t* vsrc = row + (36 + kvh) * kD;
      if constexpr (!FP8) {
        uint16_t* kdst = const_cast<uint16_t*>(kv.k) + base;
        uint16_t* vdst = const_cast<uint16_t*>(kv.v) + base;
        kdst[lane] = f32_to_bf16_bits(kvv[0]); kdst[lane + 32] = f32_to_bf16_bits(kvv[1]);
        kdst[lane + 64] = f32_to_bf16_bits(kvv[2]); kdst[lane + 96] = f32_to_bf16_bits(kvv[3]);
        vdst[lane] = vsrc[lane]; vdst[lane + 32] = vsrc[lane + 32];
        vdst[lane + 64] = vsrc[lane + 64]; vdst[lane + 96] = vsrc[lane + 96];
      } else {  // e4m3, unit scale: K from the bf16-rounded normed value, V from the bf16 projection
        uint8_t* kdst = reinterpret_cast<uint8_t*>(const_cast<uint16_t*>(kv.k)) + base;
        uint8_t* vdst = reinterpret_cast<uint8_t*>(const_cast<uint16_t*>(kv.v)) + base;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = lane + 32 * i;
          kdst[d] = static_cast<uint8_t>(f2_to_e4m3x2(round_bf16(kvv[i]), 0.f) & 0xffu);
          if constexpr (KVM != 2) vdst[d] = static_cast<uint8_t>(f2_to_e4m3x2(bf16_to_f32(vsrc[d]), 0.f) & 0xffu);
        }
        (void)vdst;
        if constexpr (KVM == 2) {
          // NVFP4 shadow: dims d = lane + 32 i; 16-dim block of d = 2 i + (lane >> 4); absmax over the 16 lanes of
          // a block, E4M3 scale = absmax / 6, E2M1 code by nearest level, bytes packed by the even lane.
          const long long blk = bt[pos / kv.block_size], tok = pos % kv.block_size;
          uint8_t* cdst = const_cast<uint8_t*>(kv.v4) + blk * kv.c4b + tok * kv.c4t + static_cast<long long>(kvh) * kv.c4h;
          uint8_t* sdst = const_cast<uint8_t*>(kv.v4s) + blk * kv.s4b + tok * kv.s4t + static_cast<long long>(kvh) * kv.s4h;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int d = lane + 32 * i;
            const float x = bf16_to_f32(vsrc[d]);
            float am = fabsf(x);
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
            const uint32_t sb = f2_to_e4m3x2(am / 6.f, 0.f) & 0xffu;
            const float sf = h2_to_f2(e4m3_to_f16x2(sb)).x;
            const float q = fabsf(x) / (sf > 0.f ? sf : 1.f);
            const int lvl = q < 0.25f ? 0 : q < 0.75f ? 1 : q < 1.25f ? 2 : q < 1.75f ? 3 : q < 2.5f ? 4 : q < 3.5f ? 5 : q < 5.f ? 6 : 7;
            const uint32_t code = static_cast<uint32_t>(lvl) | (x < 0.f ? 8u : 0u);
            const uint32_t hi = __shfl_down_sync(0xffffffffu, code, 1);
            if ((lane & 1) == 0) cdst[d >> 1] = static_cast<uint8_t>(code | (hi << 4));
            if ((lane & 15) == 0) sdst[2 * i + (lane >> 4)] = static_cast<uint8_t>(sb);
          }
        }
      }
    }
  }
  __threadfence_block();
  __syncthreads();
  decode_attn_paged_body<KVM, NPF>(q_s, kv, len, window, scale, o_part, ml_part, bt, sinks, out);
}

// grid (32 heads, n_seq), 128 threads (one per dim).
__global__ void __launch_bounds__(128)
decode_attn_paged_combine_kernel(const float* __restrict__ o_part, const float* __restrict__ ml_part,
                                 const float* __restrict__ sinks, int n_split, uint16_t* __restrict__ out) {
  pdl_wait();  // PDL-launched: never touch o_part/ml_part before the producers complete
  const int head = blockIdx.x, seq = blockIdx.y, d = threadIdx.x;
  const float sk = sinks[head] * kLog2e;
  const size_t base = static_cast<size_t>(seq) * n_split;
  float M = sk;
  for (int s = 0; s < n_split; ++s) M = fmaxf(M, ml_part[((base + s) * 32 + head) * 2]);
  float den = exp2f(sk - M), num = 0.f;
  for (int s = 0; s < n_split; ++s) {
    const float ms = ml_part[((base + s) * 32 + head) * 2];
    if (ms == -FLT_MAX) continue;
    const float f = exp2f(ms - M);
    den += ml_part[((base + s) * 32 + head) * 2 + 1] * f;
    num += o_part[((base + s) * 32 + head) * kD + d] * f;
  }
  out[(static_cast<size_t>(seq) * 32 + head) * kD + d] = f32_to_bf16_bits(num / den);
}

}  // namespace

void decode_attention_paged_bf16(const void* q, const void* k_cache, const void* v_cache, long long stride_block,
                                 long long stride_token, long long stride_head, int block_size,
                                 const int* block_tables, int bt_stride, const int* seq_lens, int n_seq, int window,
                                 const float* sinks, float scale, int n_split, float* o_part, float* ml_part,
                                 void* out, cudaStream_t s, int kv_mode, const V4Shadow* v4) {
  PagedKV kv{static_cast<const uint16_t*>(k_cache), static_cast<const uint16_t*>(v_cache), stride_block,
             stride_token, stride_head, block_tables, bt_stride, block_size, nullptr, nullptr, 0, 0, 0, 0, 0, 0};
  if (v4 != nullptr) {
    kv.v4 = static_cast<const uint8_t*>(v4->codes); kv.v4s = static_cast<const uint8_t*>(v4->scales);
    kv.c4b = v4->cb; kv.c4t = v4->ct; kv.c4h = v4->ch; kv.s4b = v4->sb; kv.s4t = v4->st; kv.s4h = v4->sh;
  }
  const int early = early_trigger_enabled() ? 1 : 0;
  static const int npf = [] { const char* v = std::getenv("FLINT_ATTN_PF"); return v ? std::atoi(v) : 1; }();
  uint16_t* direct = n_split == 1 ? static_cast<uint16_t*>(out) : nullptr;
#define FLINT_ATTN_LAUNCH(KVM_, NPF_)                                                                           \
  launch_pdl(decode_attn_paged_kernel<KVM_, NPF_>, dim3(n_split, 4, n_seq), dim3(128), 0, s,                   \
             static_cast<const uint16_t*>(q), kv, seq_lens, window, scale, o_part, ml_part, early, sinks, direct)
  if (kv_mode == 2) FLINT_ATTN_LAUNCH(2, 1);
  else if (kv_mode == 1) { if (npf >= 2) FLINT_ATTN_LAUNCH(1, 2); else FLINT_ATTN_LAUNCH(1, 1); }
  else { if (npf >= 2) FLINT_ATTN_LAUNCH(0, 2); else FLINT_ATTN_LAUNCH(0, 1); }
#undef FLINT_ATTN_LAUNCH
  if (n_split > 1)
    launch_pdl(decode_attn_paged_combine_kernel, dim3(32, n_seq), dim3(128), 0, s, o_part, ml_part, sinks, n_split,
               static_cast<uint16_t*>(out));
}


void decode_attention_paged_qkv_bf16(const void* qkv, long long qkv_stride, const long long* positions,
                                     const float* q_w, const float* k_w, const float* cos_sin, long long cs_stride,
                                     float eps, const void* k_cache, const void* v_cache, long long stride_block,
                                     long long stride_token, long long stride_head, int block_size,
                                     const int* block_tables, int bt_stride, const int* seq_lens, int n_seq,
                                     int window, const float* sinks, float scale, int n_split, float* o_part,
                                     float* ml_part, void* out, cudaStream_t s, int kv_mode, const V4Shadow* v4) {
  PagedKV kv{static_cast<const uint16_t*>(k_cache), static_cast<const uint16_t*>(v_cache), stride_block,
             stride_token, stride_head, block_tables, bt_stride, block_size, nullptr, nullptr, 0, 0, 0, 0, 0, 0};
  if (v4 != nullptr) {
    kv.v4 = static_cast<const uint8_t*>(v4->codes); kv.v4s = static_cast<const uint8_t*>(v4->scales);
    kv.c4b = v4->cb; kv.c4t = v4->ct; kv.c4h = v4->ch; kv.s4b = v4->sb; kv.s4t = v4->st; kv.s4h = v4->sh;
  }
  QkRopeIn in{static_cast<const uint16_t*>(qkv), qkv_stride, positions, q_w, k_w, cos_sin, cs_stride, eps};
  const int early = early_trigger_enabled() ? 1 : 0;
  static const int npf = [] { const char* v = std::getenv("FLINT_ATTN_PF"); return v ? std::atoi(v) : 1; }();
  uint16_t* direct = n_split == 1 ? static_cast<uint16_t*>(out) : nullptr;
#define FLINT_ATTNQ_LAUNCH(FP8_, NPF_)                                                                          \
  launch_pdl(decode_attn_paged_qkv_kernel<FP8_, NPF_>, dim3(n_split, 4, n_seq), dim3(128), 0, s, in, kv, seq_lens, \
             window, scale, o_part, ml_part, early, sinks, direct)
  if (kv_mode == 2) FLINT_ATTNQ_LAUNCH(2, 1);
  else if (kv_mode == 1) { if (npf >= 2) FLINT_ATTNQ_LAUNCH(1, 2); else FLINT_ATTNQ_LAUNCH(1, 1); }
  else { if (npf >= 2) FLINT_ATTNQ_LAUNCH(0, 2); else FLINT_ATTNQ_LAUNCH(0, 1); }
#undef FLINT_ATTNQ_LAUNCH
  if (n_split > 1)
    launch_pdl(decode_attn_paged_combine_kernel, dim3(32, n_seq), dim3(128), 0, s, o_part, ml_part, sinks, n_split,
               static_cast<uint16_t*>(out));
}

}  // namespace flint
}  // namespace flash_rt
