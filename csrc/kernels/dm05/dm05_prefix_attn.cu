// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — DM05 Gemma3 prefix attention with a per-row key limit.
//
// One CTA is 64 query rows (4 warps x 16) of one query head; 64-key blocks
// stream from the last to the first, which is FA2's order. Q/K/V tiles are
// 64 x 256 bf16 in XOR-swizzled shared memory, and V of a block is in flight
// while S = Q K^T runs, the next K while P V runs.
//
// The softmax arithmetic is FA2's -- raw scores in the running max, exp2 of
// s * c - m * c with c = scale * log2 e, -inf rows guarded, fp32 row sums, P
// rounded to bf16 -- because the caller calibrated against a build that used
// FA2 for these heads, so reproducing the order is what keeps the change from
// moving that calibration.
//
// See dm05_prefix_attn.cuh for the interface and for why this is opt-in.

#include "dm05_prefix_attn.cuh"

#include "dm05_common.cuh"

#include <stdexcept>

namespace flash_rt {
namespace dm05 {
namespace {

// XOR-swizzle: 16-byte chunk c of row r sits at c ^ (r & 7)
__device__ __forceinline__ unsigned pa_swz(int r, int c) { return (unsigned)(r * 256 + ((c ^ (r & 7)) << 3)); }

__global__ void __launch_bounds__(128)
prefix_attn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                   const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out,
                   const int* __restrict__ lim, int S, int NH, int NKV, float scale_log2) {
  constexpr int HD = 256, BM = 64, BN = 64;
  extern __shared__ __align__(128) uint8_t smem[];
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sk = sq + BM * HD;
  __nv_bfloat16* sv = sk + BN * HD;
  const int qb = blockIdx.x, h = blockIdx.y, hk = h / (NH / NKV);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int q0 = qb * BM;
  const long long qld = (long long)NH * HD, kld = (long long)NKV * HD;
  int kmax = 0;
  for (int r = 0; r < BM; ++r) if (q0 + r < S) kmax = max(kmax, lim[q0 + r]);
  const int nblk = (kmax + BN - 1) / BN;
  auto load = [&](__nv_bfloat16* dst, const __nv_bfloat16* src, long long ld, int r0, int rmax) {
    for (int i = tid; i < 64 * 32; i += 128) {
      const int r = i >> 5, c = i & 31, gr = r0 + r;
      const bool ok = gr < rmax;
      dm05_cp16(dst + pa_swz(r, c), src + (long long)(ok ? gr : 0) * ld + c * 8, ok);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  load(sq, q + h * HD, qld, q0, S);
  if (nblk > 0) load(sk, k + hk * HD, kld, (nblk - 1) * BN, S);
  const int ra = q0 + warp * 16 + g, rb = ra + 8;
  const int la = ra < S ? lim[ra] : 0, lb = rb < S ? lim[rb] : 0;
  float o[32][4];
#pragma unroll
  for (int n = 0; n < 32; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  for (int jb = nblk - 1; jb >= 0; --jb) {
    const int j0 = jb * BN;
    load(sv, v + hk * HD, kld, j0, S);            // V of this block, overlaps with S = Q K^T
    asm volatile("cp.async.wait_group 1;\n" ::);   // Q and K of this block landed
    __syncthreads();
    float sc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll 4
    for (int kk = 0; kk < 16; ++kk) {
      uint32_t a[4];
      {
        const int r = warp * 16 + (lane & 15), c = kk * 2 + (lane >> 4);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                     : "r"((unsigned)__cvta_generic_to_shared(sq + pa_swz(r, c))));
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {     // two key n-tiles per ldmatrix.x4
        const int r = np * 16 + (lane & 7) + ((lane >> 4) << 3), c = kk * 2 + ((lane >> 3) & 1);
        uint32_t b[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                     : "r"((unsigned)__cvta_generic_to_shared(sk + pa_swz(r, c))));
        dm05_mma_bf16_16816(sc[2 * np], a, b[0], b[1]);
        dm05_mma_bf16_16816(sc[2 * np + 1], a, b[2], b[3]);
      }
    }
    // mask by the per-row key limit
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int j = j0 + nt * 8 + 2 * t;
      if (j >= la) sc[nt][0] = -INFINITY;
      if (j + 1 >= la) sc[nt][1] = -INFINITY;
      if (j >= lb) sc[nt][2] = -INFINITY;
      if (j + 1 >= lb) sc[nt][3] = -INFINITY;
    }
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mx0 = fmaxf(mx0, fmaxf(sc[nt][0], sc[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(sc[nt][2], sc[nt][3]));
    }
#pragma unroll
    for (int o2 = 2; o2 >= 1; o2 >>= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o2));
    }
    const float ms0 = (mx0 == -INFINITY) ? 0.f : mx0 * scale_log2;
    const float ms1 = (mx1 == -INFINITY) ? 0.f : mx1 * scale_log2;
    const float a0 = (m0 == -INFINITY) ? 0.f : dm05_exp2((m0 - mx0) * scale_log2);
    const float a1 = (m1 == -INFINITY) ? 0.f : dm05_exp2((m1 - mx1) * scale_log2);
    m0 = mx0; m1 = mx1;
    float rs0 = l0 * a0, rs1 = l1 * a1;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      sc[nt][0] = dm05_exp2(fmaf(sc[nt][0], scale_log2, -ms0)); sc[nt][1] = dm05_exp2(fmaf(sc[nt][1], scale_log2, -ms0));
      sc[nt][2] = dm05_exp2(fmaf(sc[nt][2], scale_log2, -ms1)); sc[nt][3] = dm05_exp2(fmaf(sc[nt][3], scale_log2, -ms1));
      rs0 += sc[nt][0]; rs0 += sc[nt][1]; rs1 += sc[nt][2]; rs1 += sc[nt][3];
    }
    l0 = rs0; l1 = rs1;
#pragma unroll
    for (int n = 0; n < 32; ++n) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
    __syncthreads();                                // all warps done reading K
    if (jb > 0) load(sk, k + hk * HD, kld, (jb - 1) * BN, S);   // next K, overlaps with P V
    if (jb > 0) asm volatile("cp.async.wait_group 1;\n" ::); else asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();                                // V landed
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t a[4];
      __nv_bfloat162 p0 = __floats2bfloat162_rn(sc[2 * kk][0], sc[2 * kk][1]);
      __nv_bfloat162 p1 = __floats2bfloat162_rn(sc[2 * kk][2], sc[2 * kk][3]);
      __nv_bfloat162 p2 = __floats2bfloat162_rn(sc[2 * kk + 1][0], sc[2 * kk + 1][1]);
      __nv_bfloat162 p3 = __floats2bfloat162_rn(sc[2 * kk + 1][2], sc[2 * kk + 1][3]);
      a[0] = *reinterpret_cast<uint32_t*>(&p0); a[1] = *reinterpret_cast<uint32_t*>(&p1);
      a[2] = *reinterpret_cast<uint32_t*>(&p2); a[3] = *reinterpret_cast<uint32_t*>(&p3);
      const int r = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int np = 0; np < 16; ++np) {
        const int c = np * 2 + (lane >> 4);
        uint32_t b[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                     : "r"((unsigned)__cvta_generic_to_shared(sv + pa_swz(r, c))));
        dm05_mma_bf16_16816(o[2 * np], a, b[0], b[1]);
        dm05_mma_bf16_16816(o[2 * np + 1], a, b[2], b[3]);
      }
    }
    __syncthreads();                                // all warps done reading V
  }
#pragma unroll
  for (int o2 = 2; o2 >= 1; o2 >>= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o2);
  }
  const float i0 = (l0 == 0.f || l0 != l0) ? 1.f : dm05_rcp(l0), i1 = (l1 == 0.f || l1 != l1) ? 1.f : dm05_rcp(l1);
#pragma unroll
  for (int n = 0; n < 32; ++n) {
    if (ra < S) *reinterpret_cast<__nv_bfloat162*>(out + ra * qld + h * HD + n * 8 + 2 * t) = __floats2bfloat162_rn(o[n][0] * i0, o[n][1] * i0);
    if (rb < S) *reinterpret_cast<__nv_bfloat162*>(out + rb * qld + h * HD + n * 8 + 2 * t) = __floats2bfloat162_rn(o[n][2] * i1, o[n][3] * i1);
  }
}

}  // namespace

void prefix_attn(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                 __nv_bfloat16* out, const int* lim,
                 int S, int NH, int NKV, float scale, cudaStream_t stream) {
    if (S <= 0 || NH <= 0 || NKV <= 0) return;
    if (!q || !k || !v || !out || !lim)
        throw std::runtime_error("dm05_prefix_attn: null pointer argument");
    if (NH % NKV != 0)
        throw std::runtime_error("dm05_prefix_attn: NH must be a multiple of NKV");
    constexpr size_t sm = (size_t)3 * 64 * 256 * 2;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(prefix_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        attr = true;
    }
    // as fa2_wrapper: softmax_scale * float(M_LOG2E)
    prefix_attn_kernel<<<dim3((S + 63) / 64, NH), 128, sm, stream>>>(
        q, k, v, out, lim, S, NH, NKV, scale * float(1.4426950408889634));
}

}  // namespace dm05
}  // namespace flash_rt
