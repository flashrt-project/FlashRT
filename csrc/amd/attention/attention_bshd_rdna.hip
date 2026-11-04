// SPDX-License-Identifier: Apache-2.0
// BSHD BF16 attention for GR00T geometries, reusing the RDNA WMMA layout.
#include "../kernels/wmma_rdna.h"
#include "../kernels/transformer_ops_rdna.h"
#include <cmath>
#include <stdexcept>

namespace {
__device__ float row_sum(float x) {
#pragma unroll
  for (int i = 1; i < 16; i *= 2)
    x += __shfl_xor(x, i, 32);
  return x;
}
__device__ float row_max(float x) {
#pragma unroll
  for (int i = 1; i < 16; i *= 2)
    x = fmaxf(x, __shfl_xor(x, i, 32));
  return x;
}
__device__ RdnaWmmaFragment load_global_fragment(const __bf16 *p, int lane) {
  RdnaWmmaFragment fragment;
#pragma unroll
  for (int i = 0; i < FLASHRT_RDNA_WMMA_LANE_K; ++i)
    fragment[i] = p[rdna_wmma_lane_k(lane) + i];
  return fragment;
}
template <int D>
__global__ __launch_bounds__(32) void bshd_attention(
    const __hip_bfloat16 *q, const __hip_bfloat16 *k, const __hip_bfloat16 *v,
    __hip_bfloat16 *out, int sq, int sk, int hq, int hk, int64_t qb, int64_t qr,
    int64_t qh, int64_t kb, int64_t kr, int64_t kh, int64_t vb, int64_t vr,
    int64_t vh, bool causal, float scale) {
  __shared__ __align__(16) __bf16 pv[16][20];
  __shared__ __align__(16) __bf16 vt[D][20];
  const int lane = threadIdx.x, col = lane & 15;
  const int head = blockIdx.y % hq, batch = blockIdx.y / hq,
            kvhead = head / (hq / hk);
  const int start = blockIdx.x * 16;
  RdnaWmmaAccumulator acc[D / 16];
#pragma unroll
  for (int t = 0; t < D / 16; ++t)
    acc[t] = (RdnaWmmaAccumulator){0, 0, 0, 0, 0, 0, 0, 0};
  float mx[8], denom[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    mx[e] = -3.0e38f;
    denom[e] = 0;
  }
  const int end = causal ? min(sk, start + 16) : sk;
  for (int ks = 0; ks < end; ks += 16) {
    const int count = min(16, sk - ks);
    for (int i = lane; i < D * 16; i += 32) {
      int d = i / 16, key = i % 16;
      vt[d][key] = key < count
                       ? v[batch * vb + (ks + key) * vr + kvhead * vh + d]
                       : __float2bfloat16(0);
    }
    RdnaWmmaAccumulator score = (RdnaWmmaAccumulator){0, 0, 0, 0, 0, 0, 0, 0};
    const auto *qp = reinterpret_cast<const __bf16 *>(
        q + batch * qb + min(start + col, sq - 1) * qr + head * qh);
    const auto *kp = reinterpret_cast<const __bf16 *>(
        k + batch * kb + (ks + min(col, count - 1)) * kr + kvhead * kh);
#pragma unroll
    for (int d = 0; d < D; d += 16)
      score = rdna_wmma_bf16(load_global_fragment(qp + d, lane),
                             load_global_fragment(kp + d, lane), score);
    float correction[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = rdna_wmma_accumulator_row(lane, e);
      const bool valid = col < count && (!causal || ks + col <= start + row);
      const float s = valid ? score[e] * scale : -3.0e38f;
      const float next = row_max(fmaxf(mx[e], s));
      correction[e] = exp2f(mx[e] - next);
      const float p = valid ? exp2f(s - next) : 0;
      denom[e] = denom[e] * correction[e] + row_sum(p);
      mx[e] = next;
      pv[row][col] = __float2bfloat16(p);
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < D / 16; ++t) {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[t][e] *= correction[e];
      acc[t] = rdna_wmma_bf16(rdna_wmma_load_fragment(pv[col], lane),
                              rdna_wmma_load_fragment(vt[t * 16 + col], lane),
                              acc[t]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int t = 0; t < D / 16; ++t) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = start + rdna_wmma_accumulator_row(lane, e);
      if (row < sq)
        out[((static_cast<int64_t>(batch) * sq + row) * hq + head) * D +
            t * 16 + col] = __float2bfloat16(acc[t][e] / denom[e]);
    }
  }
}
} // namespace

void attention_bshd_bf16_rdna(const __hip_bfloat16 *q, const __hip_bfloat16 *k,
                              const __hip_bfloat16 *v, __hip_bfloat16 *out,
                              int b, int sq, int sk, int hq, int hk, int d,
                              int64_t qb, int64_t qr, int64_t qh, int64_t kb,
                              int64_t kr, int64_t kh, int64_t vb, int64_t vr,
                              int64_t vh, bool causal, hipStream_t stream) {
  if (b < 1 || sq < 1 || sk < 1 || hq < 1 || hk < 1 || hq % hk ||
      (causal && sq != sk))
    throw std::invalid_argument("invalid RDNA BSHD attention geometry");
  const dim3 grid((sq + 15) / 16, b * hq);
  const float scale = 1.4426950408889634f / std::sqrt(static_cast<float>(d));
#define RUN(D)                                                                 \
  bshd_attention<D><<<grid, 32, 0, stream>>>(q, k, v, out, sq, sk, hq, hk, qb, \
                                             qr, qh, kb, kr, kh, vb, vr, vh,   \
                                             causal, scale)
  switch (d) {
  case 48:
    RUN(48);
    break;
  case 64:
    RUN(64);
    break;
  case 128:
    RUN(128);
    break;
  default:
    throw std::invalid_argument("unsupported RDNA head width");
  }
#undef RUN
}
