// SPDX-License-Identifier: Apache-2.0
//
// Prefill-chunk attention on tensor cores (flash): one block per (query head, 64 query rows), 4 warps x 16 rows.
// Per tile of 64 keys: S = Q K^T (mma.sync m16n8k16 BF16, FP32), scaled, plus the relative bias
// bf16(r . rel_proj[:, distance]) for distances < 1024 (CUDA cores, the transposed rel_proj [1024][16]), causal /
// sliding-window mask, online softmax, O += P V (FP16 P from the S fragments, V converted to FP16 in shared
// memory, read by ldmatrix.trans). The sink
// joins the denominator at the end; the output gate as in the decode merge. Queries come prepared from kv_rows
// (q_pre), keys from the cache (sliding layers: the staged rows of the chunk at positions >= pos).
// Numerics differ from the decode kernels' FP32 per-row arithmetic in the accumulation order and the FP16 P.

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kHd = 128, kHeads = 24, kKvHeads = 8, kGroup = 3, kRel = 16, kExtent = 1024, kWindow = 1024;
constexpr int kQPre = kGroup * kHd + kGroup * kRel;
constexpr int kQRows = 64, kKeys = 64, kFlashThreads = 128;
constexpr int kKPitch = kHd + 8;                   // bf16 row pitch of the key / value tiles (ldmatrix: no conflicts)

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  return uint32_t(f32_to_bf16_bits(lo)) | (uint32_t(f32_to_bf16_bits(hi)) << 16);
}

__global__ void __launch_bounds__(kFlashThreads) attn_flash_kernel(const __grid_constant__ AttnDecodeParams p) {
  __shared__ __align__(16) uint16_t ks[kKeys][kKPitch];
  __shared__ __align__(16) uint16_t vs[kKeys][kKPitch];
  pdl_wait();
  pdl_launch();
  const int qh = blockIdx.x, h = qh / kGroup, gq = qh % kGroup;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, g = lane >> 2, c = lane & 3;
  const int pos = *after_wait(p.pos), R = p.rows;
  const int r0 = blockIdx.y * kQRows;                         // first row of the block
  const int wr0 = r0 + warp * 16;                             // first row of the warp
  // the warp's query fragments (8 k-steps of 16 dims) and relative queries (rows g, g + 8)
  uint32_t qa[8][4];
  float rq[2][kRel];
  const float* qpre = after_wait(p.q_pre);
#pragma unroll
  for (int hr = 0; hr < 2; ++hr) {
    const int r = min(wr0 + g + 8 * hr, R - 1);
    const float* q = qpre + (size_t(r) * kKvHeads + h) * kQPre + gq * kHd;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      qa[kk][hr] = pack_bf16(q[16 * kk + 2 * c], q[16 * kk + 2 * c + 1]);
      qa[kk][2 + hr] = pack_bf16(q[16 * kk + 8 + 2 * c], q[16 * kk + 8 + 2 * c + 1]);
    }
    const float* rr = qpre + (size_t(r) * kKvHeads + h) * kQPre + kGroup * kHd + gq * kRel;
#pragma unroll
    for (int i = 0; i < kRel; ++i) rq[hr][i] = rr[i];
  }
  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n)
#pragma unroll
    for (int q = 0; q < 4; ++q) o[n][q] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
  auto first_of = [&](int qp) { return p.sliding ? max(0, qp - (kWindow - 1)) : 0; };
  const int rlast = min(r0 + kQRows, R) - 1;
  const int kbeg = first_of(pos + r0), kend = pos + rlast + 1;
  const uint16_t* kc = after_wait(p.k_cache);
  const uint16_t* vc = after_wait(p.v_cache);
  const uint16_t* skv = after_wait(p.stage_kv);
  const uint16_t* relt = p.rel_t;
  const float scale = 1.0f / sqrtf(float(kHd));
  for (int k0 = kbeg; k0 < kend; k0 += kKeys) {
    __syncthreads();
    for (int i = t; i < kKeys * 16; i += kFlashThreads) {
      const int key = i >> 4, part = i & 15, kp = k0 + key;
      if (kp >= kend) continue;
      if (p.sliding && kp >= pos) {
        const uint16_t* src = skv + ((size_t(kp - pos) * 2) * kKvHeads + h) * kHd + part * 8;
        cp_async16(&ks[key][part * 8], src);
        cp_async16(&vs[key][part * 8], src + kKvHeads * kHd);
      } else {
        const int slot = p.sliding ? kp % kWindow : kp;
        cp_async16(&ks[key][part * 8], kc + (size_t(slot) * kKvHeads + h) * kHd + part * 8);
        cp_async16(&vs[key][part * 8], vc + (size_t(slot) * kKvHeads + h) * kHd + part * 8);
      }
    }
    cp_async_wait_all();
    __syncthreads();
    // V to FP16 for P V (P in FP16 keeps 10 mantissa bits; the BF16 values of V are exact in FP16)
    for (int i = t; i < kKeys * kHd / 2; i += kFlashThreads) {
      uint32_t* w = reinterpret_cast<uint32_t*>(&vs[i / (kHd / 2)][2 * (i % (kHd / 2))]);
      const uint32_t v = *w;
      *w = pack_f16(bf16lo(v), bf16hi(v));
    }
    __syncthreads();
    // S = Q K^T: 8 n-tiles of 8 keys
    float s[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q) s[n][q] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
      for (int n = 0; n < 8; n += 2) {
        uint32_t b[4];
        const int mi = lane >> 3;
        ldsm4(b, &ks[8 * (n + (mi >> 1)) + (lane & 7)][16 * kk + 8 * (mi & 1)]);
        mma_bf16(s[n], qa[kk], b[0], b[1]);
        mma_bf16(s[n + 1], qa[kk], b[2], b[3]);
      }
    }
    // logits: scale, relative bias, mask; online softmax per row
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int hr = q >> 1, r = wr0 + g + 8 * hr, qpos = pos + r, kp = k0 + 8 * n + 2 * c + (q & 1);
        float v = -INFINITY;
        if (r < R && kp <= qpos && kp >= first_of(qpos)) {
          v = s[n][q] * scale;
          const int dist = qpos - kp;
          if (dist < kExtent) {
            const uint4* rt = reinterpret_cast<const uint4*>(relt + size_t(dist) * kRel);
            const uint4 w0 = rt[0], w1 = rt[1];
            const uint32_t ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            float b = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              b = fmaf(rq[hr][2 * i], bf16lo(ww[i]), b);
              b = fmaf(rq[hr][2 * i + 1], bf16hi(ww[i]), b);
            }
            v += round_bf16(b);
          }
        }
        s[n][q] = v;
        mx[hr] = fmaxf(mx[hr], v);
      }
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      mx[hr] = fmaxf(mx[hr], __shfl_xor_sync(0xffffffffu, mx[hr], 1));
      mx[hr] = fmaxf(mx[hr], __shfl_xor_sync(0xffffffffu, mx[hr], 2));
    }
    float corr[2], rs[2] = {0.0f, 0.0f};
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) corr[hr] = mx[hr] == -INFINITY ? 1.0f : __expf(m[hr] - mx[hr]);
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int hr = q >> 1;
        const float e = s[n][q] == -INFINITY ? 0.0f : __expf(s[n][q] - mx[hr]);
        s[n][q] = e;
        rs[hr] += e;
      }
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      rs[hr] += __shfl_xor_sync(0xffffffffu, rs[hr], 1);
      rs[hr] += __shfl_xor_sync(0xffffffffu, rs[hr], 2);
      l[hr] = l[hr] * corr[hr] + rs[hr];
      m[hr] = mx[hr];
    }
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      o[n][0] *= corr[0]; o[n][1] *= corr[0];
      o[n][2] *= corr[1]; o[n][3] *= corr[1];
    }
    // O += P V: 4 k-steps of 16 keys, 16 n-tiles of 8 dims
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const uint32_t pa[4] = {pack_f16(s[2 * kk][0], s[2 * kk][1]), pack_f16(s[2 * kk][2], s[2 * kk][3]),
                              pack_f16(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_f16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int n = 0; n < 16; n += 2) {
        uint32_t b[4];
        const int mi = lane >> 3;
        ldsm4t(b, &vs[16 * kk + 8 * (mi & 1) + (lane & 7)][8 * (n + (mi >> 1))]);
        mma_f16(o[n], pa, b[0], b[1]);
        mma_f16(o[n + 1], pa, b[2], b[3]);
      }
    }
  }
  // the sink, normalization, gate
  const uint16_t* sinks = after_wait(p.sinks);
  uint16_t* out = after_wait(p.out);
#pragma unroll
  for (int hr = 0; hr < 2; ++hr) {
    const int r = wr0 + g + 8 * hr;
    if (r >= R) continue;
    const float sink = bf16_to_f32(sinks[size_t(r) * kHeads + qh]);
    const float mf = fmaxf(m[hr], sink);
    const float f = __expf(m[hr] - mf);
    const float den = l[hr] * f + __expf(sink - mf);
#pragma unroll
    for (int n = 0; n < 16; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const size_t at = (size_t(r) * kHeads + qh) * kHd + 8 * n + 2 * c + e;
        float v = o[n][2 * hr + e] * f / den;
        if (p.gate) v = round_bf16(v) * (1.0f / (1.0f + expf(-bf16_to_f32(after_wait(p.gate)[at]))));
        out[at] = f32_to_bf16_bits(v);
      }
  }
}

}  // namespace

void attn_flash(const AttnDecodeParams& p, cudaStream_t stream) {
  if (p.rows < 1 || p.rows > kMaxChunk || !p.q_pre || !p.rel_t || (p.sliding && !p.stage_kv))
    throw std::invalid_argument("attn_flash: 1..kMaxChunk rows, q_pre, rel_t, stage_kv for sliding layers");
  launch_pdl(attn_flash_kernel, dim3(kHeads, (p.rows + kQRows - 1) / kQRows), dim3(kFlashThreads), 0, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
