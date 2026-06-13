// SPDX-License-Identifier: Apache-2.0
//
// Native CUDA block-sparse GQA decode attention for Blackwell (sm_120/121),
// tensor-core (mma m16n8k16) fragment-resident flash-decode variant.
//
// Supersedes the per-key warp-shuffle-reduce kernel (msa_decode_attn.cu) for
// the M3 MSA shape (head_dim=128, GQA group=16). Design:
//   * S = Q@K^T and O = P@V via mma.m16n8k16 with the 16 GQA q-heads as the M
//     dimension; the QK^T accumulator stays in registers.
//   * Online softmax (row max/sum) is done inside the mma accumulator fragment
//     layout via a 4-lane shuffle once per key sub-tile -- this removes the
//     per-key warp-shuffle reduce that left the per-key kernel MIO-stall bound.
//   * The accumulator(P) -> operand-A repack for P@V is a pure within-thread
//     register move (no shared memory, no cross-lane shuffle).
//   * K and V are read straight from global into the mma operands: each element
//     feeds exactly one instruction, so shared-memory staging would add no
//     reuse, only occupancy pressure and a smem-load->use scoreboard stall.
//   * Split-K over key sub-tiles (BLK_TILE keys per CTA) for occupancy; the
//     partials are merged by the log-sum-exp combine pass.
//
// Same contract as msa_decode_sparse_attn_cuda. Requires D=128, Hq/Hkv=16.

#include "msa_decode_attn_mma.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

namespace flashrt_minimax_msa {
namespace {
constexpr int D = 128;
constexpr int NDT = D / 8;        // 16
constexpr int KT = D / 16;        // 8
constexpr int BLK_TILE = 64;      // keys per CTA sub-tile (M3 sweet spot)
constexpr int NNT = BLK_TILE / 8; // 4 key n-tiles (QK^T S)
constexpr int KTB = BLK_TILE / 16;// 2 key k-tiles (PV)

__device__ __forceinline__ uint32_t u32(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t packbf(float lo, float hi) {
  __nv_bfloat16 a = __float2bfloat16(lo), b = __float2bfloat16(hi);
  return (uint32_t)(*reinterpret_cast<unsigned short*>(&a)) |
         ((uint32_t)(*reinterpret_cast<unsigned short*>(&b)) << 16);
}
__device__ __forceinline__ void mma16816(float& c0, float& c1, float& c2,
                                         float& c3, uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void decode_p1_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
    const int* __restrict__ seq_lens, const int64_t* __restrict__ slot_ids,
    const int* __restrict__ topk_idx, float* __restrict__ m_part,
    float* __restrict__ l_part, float* __restrict__ o_part,
    int B, int Hq, int Hkv, int max_slots, int max_len, int block_size,
    int topk, int nsplit, int subsplit, float sm_scale) {
  // No K/V smem staging: each K/V element is used by exactly one mma (reused
  // across the 16 M-rows within the instruction, not across instructions), so
  // staging gives no reuse — only occupancy cost + smem-scoreboard stall.
  // K/V operands are read directly from global (L2-resident block). Only Q is
  // reused across n/k tiles, so it lives in registers (qa).
  const int b = blockIdx.x, kh = blockIdx.y, split = blockIdx.z;
  const int gqa = Hq / Hkv;
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2, tig = lane & 3;
  const int t = split / subsplit;          // which selected block
  const int sub = split % subsplit;        // which sub-tile of the block
  const int seq_len = seq_lens[b] > 0 ? seq_lens[b] : 0;
  long sid = ((slot_ids[b] % max_slots) + max_slots) % max_slots;
  const long kbase = (((sid * 2 + 0) * max_len) * Hkv + kh) * D;
  const long vbase = (((sid * 2 + 1) * max_len) * Hkv + kh) * D;
  const long pstride = (long)Hkv * D;
  const int* idx_row = topk_idx + ((long)kh * B + b) * topk;

  float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.f, l1 = 0.f;
  float oa[NDT][4];
  #pragma unroll
  for (int nn = 0; nn < NDT; ++nn)
    oa[nn][0] = oa[nn][1] = oa[nn][2] = oa[nn][3] = 0.f;

  // resolve this CTA's key sub-range [start, start+n)
  int n = 0, start = 0;
  if (t < topk) {
    const int blk = idx_row[t];
    if (blk >= 0) {
      const int bstart = blk * block_size;
      const int nblock = min(block_size, seq_len - bstart);  // keys in block
      const int sstart = sub * BLK_TILE;
      if (bstart < seq_len && sstart < nblock) {
        start = bstart + sstart;
        n = min(BLK_TILE, nblock - sstart);
      }
    }
  }
  if (n > 0) {
    const long qhead0 = (long)b * Hq + kh * gqa;
    // Q operands (reused across all n/k tiles) -> registers, read once.
    uint32_t qa[KT][4];
    #pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      const int kk = kt * 16 + 2 * tig;
      qa[kt][0] = u32(q + (qhead0 + gid) * D + kk);
      qa[kt][1] = u32(q + (qhead0 + gid + 8) * D + kk);
      qa[kt][2] = u32(q + (qhead0 + gid) * D + kk + 8);
      qa[kt][3] = u32(q + (qhead0 + gid + 8) * D + kk + 8);
    }

    float sc[NNT][4];
    #pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
      const int nkey = nt * 8 + gid;
      // clamp invalid keys to a valid row (masked out later) to avoid OOB.
      const int rkey = start + (nkey < n ? nkey : 0);
      const __nv_bfloat16* kp = kv + kbase + (long)rkey * pstride;
      // batch-load all K operands for this key-tile (global, L2-resident).
      uint32_t kb[KT][2];
      #pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        const int kk = kt * 16 + 2 * tig;
        kb[kt][0] = u32(kp + kk);
        kb[kt][1] = u32(kp + kk + 8);
      }
      #pragma unroll
      for (int kt = 0; kt < KT; ++kt)
        mma16816(c0, c1, c2, c3, qa[kt][0], qa[kt][1], qa[kt][2], qa[kt][3],
                 kb[kt][0], kb[kt][1]);
      const int col = nt * 8 + 2 * tig;
      c0 = (col + 0 < n) ? c0 * sm_scale : -CUDART_INF_F;
      c1 = (col + 1 < n) ? c1 * sm_scale : -CUDART_INF_F;
      c2 = (col + 0 < n) ? c2 * sm_scale : -CUDART_INF_F;
      c3 = (col + 1 < n) ? c3 * sm_scale : -CUDART_INF_F;
      sc[nt][0] = c0; sc[nt][1] = c1; sc[nt][2] = c2; sc[nt][3] = c3;
    }

    float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
    #pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      bm0 = fmaxf(bm0, fmaxf(sc[nt][0], sc[nt][1]));
      bm1 = fmaxf(bm1, fmaxf(sc[nt][2], sc[nt][3]));
    }
    bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 1));
    bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 2));
    bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 1));
    bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 2));
    m0 = bm0; m1 = bm1;   // single sub-tile -> running max = sub-tile max

    float bs0 = 0.f, bs1 = 0.f;
    #pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      float p0 = (sc[nt][0] > -CUDART_INF_F) ? __expf(sc[nt][0] - m0) : 0.f;
      float p1 = (sc[nt][1] > -CUDART_INF_F) ? __expf(sc[nt][1] - m0) : 0.f;
      float p2 = (sc[nt][2] > -CUDART_INF_F) ? __expf(sc[nt][2] - m1) : 0.f;
      float p3 = (sc[nt][3] > -CUDART_INF_F) ? __expf(sc[nt][3] - m1) : 0.f;
      bs0 += p0 + p1; bs1 += p2 + p3;
      sc[nt][0] = p0; sc[nt][1] = p1; sc[nt][2] = p2; sc[nt][3] = p3;
    }
    bs0 += __shfl_xor_sync(0xffffffffu, bs0, 1);
    bs0 += __shfl_xor_sync(0xffffffffu, bs0, 2);
    bs1 += __shfl_xor_sync(0xffffffffu, bs1, 1);
    bs1 += __shfl_xor_sync(0xffffffffu, bs1, 2);
    l0 = bs0; l1 = bs1;

    #pragma unroll
    for (int ndt = 0; ndt < NDT; ++ndt) {
      float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
      const int dcol = ndt * 8 + gid;
      const __nv_bfloat16* vp = kv + vbase + dcol;
      uint32_t vb[KTB][2];
      #pragma unroll
      for (int ktp = 0; ktp < KTB; ++ktp) {
        const int key0 = ktp * 16 + 2 * tig;
        const int r0 = start + (key0 < n ? key0 : 0);
        const int r1 = start + (key0 + 1 < n ? key0 + 1 : 0);
        const int r8 = start + (key0 + 8 < n ? key0 + 8 : 0);
        const int r9 = start + (key0 + 9 < n ? key0 + 9 : 0);
        vb[ktp][0] = packbf(__bfloat162float(vp[(long)r0 * pstride]),
                            __bfloat162float(vp[(long)r1 * pstride]));
        vb[ktp][1] = packbf(__bfloat162float(vp[(long)r8 * pstride]),
                            __bfloat162float(vp[(long)r9 * pstride]));
      }
      #pragma unroll
      for (int ktp = 0; ktp < KTB; ++ktp) {
        const int n0 = 2 * ktp, n1 = 2 * ktp + 1;
        uint32_t a0 = packbf(sc[n0][0], sc[n0][1]);
        uint32_t a1 = packbf(sc[n0][2], sc[n0][3]);
        uint32_t a2 = packbf(sc[n1][0], sc[n1][1]);
        uint32_t a3 = packbf(sc[n1][2], sc[n1][3]);
        mma16816(o0, o1, o2, o3, a0, a1, a2, a3, vb[ktp][0], vb[ktp][1]);
      }
      oa[ndt][0] = o0; oa[ndt][1] = o1; oa[ndt][2] = o2; oa[ndt][3] = o3;
    }
  }

  const long p0idx = (((long)b * Hkv + kh) * gqa + gid) * nsplit + split;
  const long p1idx = (((long)b * Hkv + kh) * gqa + gid + 8) * nsplit + split;
  if (tig == 0) {
    m_part[p0idx] = m0; l_part[p0idx] = l0;
    m_part[p1idx] = m1; l_part[p1idx] = l1;
  }
  #pragma unroll
  for (int ndt = 0; ndt < NDT; ++ndt) {
    const int col = ndt * 8 + 2 * tig;
    o_part[p0idx * D + col + 0] = oa[ndt][0];
    o_part[p0idx * D + col + 1] = oa[ndt][1];
    o_part[p1idx * D + col + 0] = oa[ndt][2];
    o_part[p1idx * D + col + 1] = oa[ndt][3];
  }
}

__global__ void decode_combine_kernel(
    const float* __restrict__ m_part, const float* __restrict__ l_part,
    const float* __restrict__ o_part, __nv_bfloat16* __restrict__ out,
    int B, int Hq, int Hkv, int nsplit) {
  const int b = blockIdx.x, qh = blockIdx.y, d = threadIdx.x;
  const int gqa = Hq / Hkv, kh = qh / gqa, g = qh - kh * gqa;
  const long base = (((long)b * Hkv + kh) * gqa + g) * nsplit;
  float M = -CUDART_INF_F;
  for (int s = 0; s < nsplit; ++s) M = fmaxf(M, m_part[base + s]);
  float denom = 0.f, num = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    const float ms = m_part[base + s];
    if (ms == -CUDART_INF_F) continue;
    const float w = __expf(ms - M);
    denom += w * l_part[base + s];
    num += w * o_part[(base + s) * D + d];
  }
  out[((long)b * Hq + qh) * D + d] =
      __float2bfloat16(denom > 0.f ? num / denom : 0.f);
}

// Paged variant: K/V live in separate [max_slots, Hkv, D] caches and a logical
// token position is mapped to a physical slot via req_to_token. The mma core is
// identical to decode_p1_kernel; only the K/V address resolution differs. The
// sub-tile's physical slots are gathered once into shared memory and reused by
// both QK^T and P@V.
__global__ void decode_p1_paged_kernel(
    const __nv_bfloat16* __restrict__ q,        // [B, Hq, D]
    const __nv_bfloat16* __restrict__ k_cache,  // [max_slots, Hkv, D]
    const __nv_bfloat16* __restrict__ v_cache,  // [max_slots, Hkv, D]
    const int* __restrict__ req_to_token,       // [max_reqs, max_kv_len]
    const int* __restrict__ seq_lens, const int64_t* __restrict__ slot_ids,
    const int* __restrict__ topk_idx, float* __restrict__ m_part,
    float* __restrict__ l_part, float* __restrict__ o_part,
    int B, int Hq, int Hkv, int max_slots, int max_kv_len, int block_size,
    int topk, int nsplit, int subsplit, float sm_scale) {
  __shared__ int sPhys[BLK_TILE];
  const int b = blockIdx.x, kh = blockIdx.y, split = blockIdx.z;
  const int gqa = Hq / Hkv;
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2, tig = lane & 3;
  const int t = split / subsplit, sub = split % subsplit;
  const int seq_len = seq_lens[b] > 0 ? seq_lens[b] : 0;
  const long sidr = ((slot_ids[b] % max_slots) + max_slots) % max_slots;
  const long kvrow = (long)Hkv * D;  // stride between slots in k/v_cache

  float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.f, l1 = 0.f;
  float oa[NDT][4];
  #pragma unroll
  for (int nn = 0; nn < NDT; ++nn)
    oa[nn][0] = oa[nn][1] = oa[nn][2] = oa[nn][3] = 0.f;

  int n = 0, start = 0;
  if (t < topk) {
    const int blk = topk_idx[((long)kh * B + b) * topk + t];
    if (blk >= 0) {
      const int bstart = blk * block_size;
      const int nblock = min(block_size, seq_len - bstart);
      const int sstart = sub * BLK_TILE;
      if (bstart < seq_len && sstart < nblock) {
        start = bstart + sstart;
        n = min(BLK_TILE, nblock - sstart);
      }
    }
  }

  // gather physical slots for this sub-tile's keys (clamp invalid -> 0).
  for (int i = lane; i < BLK_TILE; i += 32) {
    int s = 0;
    if (i < n) {
      s = req_to_token[sidr * max_kv_len + (start + i)];
      s = ((s % max_slots) + max_slots) % max_slots;
    }
    sPhys[i] = s;
  }
  __syncwarp();

  if (n > 0) {
    const long qhead0 = (long)b * Hq + kh * gqa;
    uint32_t qa[KT][4];
    #pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      const int kk = kt * 16 + 2 * tig;
      qa[kt][0] = u32(q + (qhead0 + gid) * D + kk);
      qa[kt][1] = u32(q + (qhead0 + gid + 8) * D + kk);
      qa[kt][2] = u32(q + (qhead0 + gid) * D + kk + 8);
      qa[kt][3] = u32(q + (qhead0 + gid + 8) * D + kk + 8);
    }

    float sc[NNT][4];
    #pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
      const int nkey = nt * 8 + gid;
      const long phys = sPhys[nkey < n ? nkey : 0];
      const __nv_bfloat16* kp = k_cache + phys * kvrow + (long)kh * D;
      uint32_t kb[KT][2];
      #pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        const int kk = kt * 16 + 2 * tig;
        kb[kt][0] = u32(kp + kk);
        kb[kt][1] = u32(kp + kk + 8);
      }
      #pragma unroll
      for (int kt = 0; kt < KT; ++kt)
        mma16816(c0, c1, c2, c3, qa[kt][0], qa[kt][1], qa[kt][2], qa[kt][3],
                 kb[kt][0], kb[kt][1]);
      const int col = nt * 8 + 2 * tig;
      c0 = (col + 0 < n) ? c0 * sm_scale : -CUDART_INF_F;
      c1 = (col + 1 < n) ? c1 * sm_scale : -CUDART_INF_F;
      c2 = (col + 0 < n) ? c2 * sm_scale : -CUDART_INF_F;
      c3 = (col + 1 < n) ? c3 * sm_scale : -CUDART_INF_F;
      sc[nt][0] = c0; sc[nt][1] = c1; sc[nt][2] = c2; sc[nt][3] = c3;
    }

    float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
    #pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      bm0 = fmaxf(bm0, fmaxf(sc[nt][0], sc[nt][1]));
      bm1 = fmaxf(bm1, fmaxf(sc[nt][2], sc[nt][3]));
    }
    bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 1));
    bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 2));
    bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 1));
    bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 2));
    m0 = bm0; m1 = bm1;

    float bs0 = 0.f, bs1 = 0.f;
    #pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
      float p0 = (sc[nt][0] > -CUDART_INF_F) ? __expf(sc[nt][0] - m0) : 0.f;
      float p1 = (sc[nt][1] > -CUDART_INF_F) ? __expf(sc[nt][1] - m0) : 0.f;
      float p2 = (sc[nt][2] > -CUDART_INF_F) ? __expf(sc[nt][2] - m1) : 0.f;
      float p3 = (sc[nt][3] > -CUDART_INF_F) ? __expf(sc[nt][3] - m1) : 0.f;
      bs0 += p0 + p1; bs1 += p2 + p3;
      sc[nt][0] = p0; sc[nt][1] = p1; sc[nt][2] = p2; sc[nt][3] = p3;
    }
    bs0 += __shfl_xor_sync(0xffffffffu, bs0, 1);
    bs0 += __shfl_xor_sync(0xffffffffu, bs0, 2);
    bs1 += __shfl_xor_sync(0xffffffffu, bs1, 1);
    bs1 += __shfl_xor_sync(0xffffffffu, bs1, 2);
    l0 = bs0; l1 = bs1;

    #pragma unroll
    for (int ndt = 0; ndt < NDT; ++ndt) {
      float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
      const int dcol = ndt * 8 + gid;
      uint32_t vb[KTB][2];
      #pragma unroll
      for (int ktp = 0; ktp < KTB; ++ktp) {
        const int k0 = ktp * 16 + 2 * tig;
        const long ph0 = sPhys[k0 < n ? k0 : 0];
        const long ph1 = sPhys[k0 + 1 < n ? k0 + 1 : 0];
        const long ph8 = sPhys[k0 + 8 < n ? k0 + 8 : 0];
        const long ph9 = sPhys[k0 + 9 < n ? k0 + 9 : 0];
        const long off = (long)kh * D + dcol;
        vb[ktp][0] = packbf(__bfloat162float(v_cache[ph0 * kvrow + off]),
                            __bfloat162float(v_cache[ph1 * kvrow + off]));
        vb[ktp][1] = packbf(__bfloat162float(v_cache[ph8 * kvrow + off]),
                            __bfloat162float(v_cache[ph9 * kvrow + off]));
      }
      #pragma unroll
      for (int ktp = 0; ktp < KTB; ++ktp) {
        const int n0 = 2 * ktp, n1 = 2 * ktp + 1;
        uint32_t a0 = packbf(sc[n0][0], sc[n0][1]);
        uint32_t a1 = packbf(sc[n0][2], sc[n0][3]);
        uint32_t a2 = packbf(sc[n1][0], sc[n1][1]);
        uint32_t a3 = packbf(sc[n1][2], sc[n1][3]);
        mma16816(o0, o1, o2, o3, a0, a1, a2, a3, vb[ktp][0], vb[ktp][1]);
      }
      oa[ndt][0] = o0; oa[ndt][1] = o1; oa[ndt][2] = o2; oa[ndt][3] = o3;
    }
  }

  const long p0idx = (((long)b * Hkv + kh) * gqa + gid) * nsplit + split;
  const long p1idx = (((long)b * Hkv + kh) * gqa + gid + 8) * nsplit + split;
  if (tig == 0) {
    m_part[p0idx] = m0; l_part[p0idx] = l0;
    m_part[p1idx] = m1; l_part[p1idx] = l1;
  }
  #pragma unroll
  for (int ndt = 0; ndt < NDT; ++ndt) {
    const int col = ndt * 8 + 2 * tig;
    o_part[p0idx * D + col + 0] = oa[ndt][0];
    o_part[p0idx * D + col + 1] = oa[ndt][1];
    o_part[p1idx * D + col + 0] = oa[ndt][2];
    o_part[p1idx * D + col + 1] = oa[ndt][3];
  }
}
}  // namespace

void msa_decode_sparse_attn_mma_cuda(
    const void* q, const void* kv_cache, const int* seq_lens,
    const int64_t* slot_ids, const int* topk_idx, void* out,
    int B, int Hq, int Hkv, int D, int max_slots, int max_len,
    int block_size, int topk, float sm_scale, cudaStream_t stream) {
  const int gqa = Hq / Hkv;
  const int subsplit = block_size / BLK_TILE;
  const int nsplit = topk * subsplit;
  static float* ws = nullptr;
  static long ws_floats = 0;
  const long n_pl = (long)B * Hkv * gqa * nsplit;
  const long need = n_pl * 2 + n_pl * D;
  if (need > ws_floats) {
    if (ws) cudaFree(ws);
    cudaMalloc(&ws, need * sizeof(float));
    ws_floats = need;
  }
  float* m_part = ws;
  float* l_part = ws + n_pl;
  float* o_part = ws + n_pl * 2;
  dim3 g1(B, Hkv, nsplit);
  decode_p1_kernel<<<g1, 32, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(kv_cache), seq_lens, slot_ids,
      topk_idx, m_part, l_part, o_part, B, Hq, Hkv, max_slots, max_len,
      block_size, topk, nsplit, subsplit, sm_scale);
  dim3 g2(B, Hq);
  decode_combine_kernel<<<g2, D, 0, stream>>>(
      m_part, l_part, o_part, reinterpret_cast<__nv_bfloat16*>(out), B, Hq, Hkv,
      nsplit);
}

void msa_decode_sparse_attn_mma_paged_cuda(
    const void* q, const void* k_cache, const void* v_cache,
    const int* req_to_token, const int* seq_lens, const int64_t* slot_ids,
    const int* topk_idx, void* out, int B, int Hq, int Hkv, int D,
    int max_slots, int max_kv_len, int block_size, int topk, float sm_scale,
    cudaStream_t stream) {
  const int gqa = Hq / Hkv;
  const int subsplit = block_size / BLK_TILE;
  const int nsplit = topk * subsplit;
  static float* ws = nullptr;
  static long ws_floats = 0;
  const long n_pl = (long)B * Hkv * gqa * nsplit;
  const long need = n_pl * 2 + n_pl * D;
  if (need > ws_floats) {
    if (ws) cudaFree(ws);
    cudaMalloc(&ws, need * sizeof(float));
    ws_floats = need;
  }
  float* m_part = ws;
  float* l_part = ws + n_pl;
  float* o_part = ws + n_pl * 2;
  dim3 g1(B, Hkv, nsplit);
  decode_p1_paged_kernel<<<g1, 32, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k_cache),
      reinterpret_cast<const __nv_bfloat16*>(v_cache), req_to_token, seq_lens,
      slot_ids, topk_idx, m_part, l_part, o_part, B, Hq, Hkv, max_slots,
      max_kv_len, block_size, topk, nsplit, subsplit, sm_scale);
  dim3 g2(B, Hq);
  decode_combine_kernel<<<g2, D, 0, stream>>>(
      m_part, l_part, o_part, reinterpret_cast<__nv_bfloat16*>(out), B, Hq, Hkv,
      nsplit);
}

}  // namespace flashrt_minimax_msa
