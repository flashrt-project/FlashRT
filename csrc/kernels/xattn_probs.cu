// SPDX-License-Identifier: Apache-2.0
//
// Cross-attention against a text prompt whose keys collapse to a few distinct
// rows (one per prompt token plus one for the identical padding rows). With
// the text fixed, the attention output projection is linear in the
// probabilities: Wo * sum_j p_hj v_hj = sum_j p_hj (Wo_h v_hj), so the
// projected values U[:, (h, j)] = Wo_h v_hj are static per prompt and the
// cross-attention + output GEMM becomes this kernel (the [L, heads * nk]
// probabilities) followed by an [L, heads * nk] x [heads * nk, D] GEMM.
// The softmax is exact over the distinct keys: a key standing for m identical
// rows enters with weight m (log2_mult = log2 m).
#include "xattn_probs.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

// One warp per (16-row tile, head): the logits are a [16 x 128] x [128 x 32]
// product on the bf16 tensor-core MMA (m16n8k16, fp32 accumulate), read
// straight from global memory in fragment order, and the softmax runs on the
// accumulator fragments (each row's 32 logits live in one lane quad). Rows of a
// tile belong to one batch element (the host checks the layout).
__device__ __forceinline__ uint32_t ld_b32(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

__global__ void __launch_bounds__(128)
xattn_probs_fp8_kernel(const __nv_bfloat16* __restrict__ q, int ld_q,
                       const __nv_bfloat16* __restrict__ keys, int ld_k,
                       const float* __restrict__ log2_mult,
                       const int* __restrict__ row_elem, const int* __restrict__ elem_k0,
                       const int* __restrict__ elem_nk, const int* __restrict__ elem_col0,
                       const float* __restrict__ col_scale,
                       __nv_fp8_e4m3* __restrict__ P, int ld_p, int L, int heads,
                       float scale_log2, float p_mul) {
  const int lane = threadIdx.x & 31;
  const int h = blockIdx.y * 4 + (threadIdx.x >> 5);
  if (h >= heads) return;
  const int r0 = blockIdx.x * 16;
  const int b = row_elem[r0];
  const int k0 = elem_k0[b], nk = elem_nk[b], c0 = elem_col0[b];
  const int g = lane >> 2, t = lane & 3;
  const int ra = r0 + g, rb = r0 + g + 8;
  const __nv_bfloat16* qa = q + (size_t)ra * ld_q + h * 128 + 2 * t;
  const __nv_bfloat16* qb = q + (size_t)rb * ld_q + h * 128 + 2 * t;
  const bool va = ra < L, vb = rb < L;

  float acc[4][4];
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[n][e] = 0.f;
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const int d = kk * 16;
    const uint32_t a0 = va ? ld_b32(qa + d) : 0u, a1 = vb ? ld_b32(qb + d) : 0u;
    const uint32_t a2 = va ? ld_b32(qa + d + 8) : 0u, a3 = vb ? ld_b32(qb + d + 8) : 0u;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int j = n * 8 + g;
      const __nv_bfloat16* kp = keys + (size_t)(k0 + j) * ld_k + h * 128 + d + 2 * t;
      const uint32_t b0 = j < nk ? ld_b32(kp) : 0u, b1 = j < nk ? ld_b32(kp + 8) : 0u;
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(acc[n][0]), "+f"(acc[n][1]), "+f"(acc[n][2]), "+f"(acc[n][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  // acc[n][0..1]: row ra, keys n*8 + 2t + {0,1}; acc[n][2..3]: row rb, same keys
  float lg[2][8];
  float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int j = n * 8 + 2 * t + e;
      const float lm = j < nk ? log2_mult[k0 + j] : 0.f;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float v = j < nk ? acc[n][2 * r + e] * scale_log2 + lm : -INFINITY;
        lg[r][n * 2 + e] = v;
        mx[r] = fmaxf(mx[r], v);
      }
    }
  float z[2] = {0.f, 0.f};
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      lg[r][i] = exp2f(lg[r][i] - mx[r]);
      z[r] += lg[r][i];
    }
    z[r] += __shfl_xor_sync(0xffffffffu, z[r], 1);
    z[r] += __shfl_xor_sync(0xffffffffu, z[r], 2);
    z[r] = p_mul / z[r];
  }
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int j = n * 8 + 2 * t + e;
      if (j >= nk) continue;
      const int c = c0 + h * nk + j;
      const float cs = col_scale != nullptr ? col_scale[c] : 1.f;
      if (va) P[(size_t)ra * ld_p + c] = __nv_fp8_e4m3(lg[0][n * 2 + e] * z[0] * cs);
      if (vb) P[(size_t)rb * ld_p + c] = __nv_fp8_e4m3(lg[1][n * 2 + e] * z[1] * cs);
    }
}

}  // namespace

int xattn_probs_fp8(const void* q, int ld_q, const void* keys, int ld_k,
                    const float* log2_mult, const int* row_elem, const int* elem_k0,
                    const int* elem_nk, const int* elem_col0, const float* col_scale,
                    void* P, int ld_p, int L, int heads, int hd, float scale, float p_mul,
                    cudaStream_t stream) {
  if (hd != 128 || L <= 0 || heads <= 0) return -1;
  dim3 grid((L + 15) / 16, (heads + 3) / 4);
  xattn_probs_fp8_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q), ld_q,
      reinterpret_cast<const __nv_bfloat16*>(keys), ld_k, log2_mult, row_elem, elem_k0,
      elem_nk, elem_col0, col_scale, reinterpret_cast<__nv_fp8_e4m3*>(P), ld_p, L, heads,
      scale * 1.4426950408889634f, p_mul);
  return (int)cudaGetLastError();
}

}  // namespace kernels
}  // namespace flash_rt
