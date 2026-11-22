// SPDX-License-Identifier: Apache-2.0
//
// Batch-one LM head with greedy selection: x = bf16(rms_norm(in) * w) (the final norm), logits = W x in FP32
// (the checkpoint's FP32 forward output), softcapped (tanh(l / cap) * cap), greedy over the unpadded vocabulary
// (first index on ties). Warps stride over the rows two at a time; the last block reduces the block winners.

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kThreads = 256;

// LM head over `rows` rows (one: the decode step): every row's norm and dot products in the one-row order, the
// vocabulary rows read once per block for all rows, greedy per row (first index on ties).
// 8 E4M3 values (two words) as FP32
__device__ __forceinline__ void e4m3x8_to_f32(uint2 q, float (&f)[8]) {
  const uint32_t qq[2] = {q.x, q.y};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      uint32_t h2;
      const uint16_t pair = static_cast<uint16_t>(qq[h] >> (16 * p));
      asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(pair));
      const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&h2));
      f[4 * h + 2 * p] = v.x;
      f[4 * h + 2 * p + 1] = v.y;
    }
  }
}

template <bool FP8>
__global__ void __launch_bounds__(kThreads) lm_head_kernel(const __grid_constant__ LmHeadParams p) {
  extern __shared__ __align__(16) uint16_t xb[];            // [rows][kHidden] the normalized rows (BF16 values)
  __shared__ float scratch[8];
  __shared__ float bv[8][kMaxRows];
  __shared__ int bi[8][kMaxRows];
  __shared__ int last;
  const int R = p.rows < 1 ? 1 : p.rows;
  prefetch_share(p.prefetch);
  pdl_wait();
  pdl_launch();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int r = 0; r < R; ++r) {
    const float* in = after_wait(p.in) + size_t(r) * p.in_stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kHidden; i += kThreads) ss = fmaf(in[i], in[i], ss);
    ss = warp_sum(ss);
    if (lane == 0) scratch[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += scratch[i];
    const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
    __syncthreads();
    for (int i = threadIdx.x; i < kHidden; i += kThreads) xb[r * kHidden + i] = f32_to_bf16_bits(in[i] * inv * bf16_to_f32(p.norm_w[i]));
  }
  __syncthreads();
  if (p.normed_out && blockIdx.x == 0) {
    uint4* o = reinterpret_cast<uint4*>(after_wait(p.normed_out));
    for (int i = threadIdx.x; i < R * kHidden / 8; i += kThreads) o[i] = reinterpret_cast<const uint4*>(xb)[i];
  }
  float best[kMaxRows];
  int bidx[kMaxRows];
#pragma unroll
  for (int r = 0; r < kMaxRows; ++r) { best[r] = -INFINITY; bidx[r] = 1 << 30; }
  const int warps = gridDim.x * 8;
  const int gw = blockIdx.x * 8 + warp;
  float* out = p.logits ? after_wait(p.logits) : nullptr;
  for (int v = gw; v < p.vocab; v += warps) {
    uint4 wv[kHidden / 8 / 32];
    if (FP8) {
      // the drafter's copy: E4M3 codes with a per-row scale, 8 codes per lane and unit as the BF16 path
      const uint2* w = reinterpret_cast<const uint2*>(p.weight_fp8) + size_t(v) * (kHidden / 8);
      uint2 q[kHidden / 8 / 32];
#pragma unroll
      for (int u = 0; u < kHidden / 8 / 32; ++u) q[u] = *reinterpret_cast<const uint2*>(w + lane + 32 * u);
      float acc[kMaxRows];
#pragma unroll
      for (int r = 0; r < kMaxRows; ++r) acc[r] = 0.0f;
#pragma unroll
      for (int u = 0; u < kHidden / 8 / 32; ++u) {
        float f[8];
        e4m3x8_to_f32(q[u], f);
        const int i = lane + 32 * u;
#pragma unroll
        for (int r = 0; r < kMaxRows; ++r) {
          if (r >= R) break;
          const uint4 xv = reinterpret_cast<const uint4*>(xb + r * kHidden)[i];
          const uint32_t xx[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            acc[r] = fmaf(f[2 * j], bf16lo(xx[j]), acc[r]);
            acc[r] = fmaf(f[2 * j + 1], bf16hi(xx[j]), acc[r]);
          }
        }
      }
      const float sc = p.weight_scale[v];
#pragma unroll
      for (int r = 0; r < kMaxRows; ++r) {
        if (r >= R) break;
        const float a = warp_sum(acc[r]) * sc;
        const float c = p.softcap > 0.0f ? tanhf(a / p.softcap) * p.softcap : a;
        if (lane == 0) {
          if (out) out[size_t(r) * p.vocab + v] = c;
          if (c > best[r] || (c == best[r] && v < bidx[r])) { best[r] = c; bidx[r] = v; }
        }
      }
      continue;
    }
    const uint4* w = reinterpret_cast<const uint4*>(p.weight) + size_t(v) * (kHidden / 8);
#pragma unroll
    for (int u = 0; u < kHidden / 8 / 32; ++u) wv[u] = ldg_stream(w + lane + 32 * u);
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
      if (r >= R) break;
      float acc = 0.0f;
#pragma unroll
      for (int u = 0; u < kHidden / 8 / 32; ++u) {
        const int i = lane + 32 * u;
        const uint4 xv = reinterpret_cast<const uint4*>(xb + r * kHidden)[i];
        const uint32_t ww[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
        const uint32_t xx[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc = fmaf(bf16lo(ww[j]), bf16lo(xx[j]), acc);
          acc = fmaf(bf16hi(ww[j]), bf16hi(xx[j]), acc);
        }
      }
      acc = warp_sum(acc);
      const float c = p.softcap > 0.0f ? tanhf(acc / p.softcap) * p.softcap : acc;
      if (lane == 0) {
        if (out) out[size_t(r) * p.vocab + v] = c;
        if (c > best[r] || (c == best[r] && v < bidx[r])) { best[r] = c; bidx[r] = v; }
      }
    }
  }
  if (lane == 0)
    for (int r = 0; r < R; ++r) { bv[warp][r] = best[r]; bi[warp][r] = bidx[r]; }
  __syncthreads();
  float* gv = after_wait(p.block_best);
  int* gi = after_wait(p.block_index);
  if (threadIdx.x < R) {
    const int r = threadIdx.x;
    float b = bv[0][r];
    int ix = bi[0][r];
    for (int w = 1; w < 8; ++w)
      if (bv[w][r] > b || (bv[w][r] == b && bi[w][r] < ix)) { b = bv[w][r]; ix = bi[w][r]; }
    gv[blockIdx.x * R + r] = b;
    gi[blockIdx.x * R + r] = ix;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(after_wait(p.counter), 1) == int(gridDim.x) - 1;
  __syncthreads();
  if (!last || warp >= R) return;
  __threadfence();
  const int r = warp;
  float b = -INFINITY;
  int ix = 1 << 30;
  for (int i = lane; i < int(gridDim.x); i += 32) {
    const float v = gv[i * R + r];
    const int j = gi[i * R + r];
    if (v > b || (v == b && j < ix)) { b = v; ix = j; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, b, o);
    const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
    if (ov > b || (ov == b && oi < ix)) { b = ov; ix = oi; }
  }
  if (lane == 0) after_wait(p.token)[r] = ix;
  __syncthreads();
  if (threadIdx.x == 0) *after_wait(p.counter) = 0;
}

}  // namespace

void lm_head(const LmHeadParams& p, cudaStream_t stream) {
  const int R = p.rows < 1 ? 1 : p.rows;
  if (p.blocks < 1 || R > kMaxRows) throw std::invalid_argument("lm_head: blocks >= 1, at most 8 rows");
  const size_t smem = size_t(R) * kHidden * 2;
  if (p.weight_fp8) {
    if (!p.weight_scale) throw std::invalid_argument("lm_head: FP8 weights need weight_scale");
    if (smem > 48 * 1024)
      check_cuda(cudaFuncSetAttribute(lm_head_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)),
                 "lm_head smem");
    launch_pdl(lm_head_kernel<true>, dim3(p.blocks), dim3(kThreads), smem, stream, p);
    return;
  }
  if (smem > 48 * 1024)
    check_cuda(cudaFuncSetAttribute(lm_head_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)),
               "lm_head smem");
  launch_pdl(lm_head_kernel<false>, dim3(p.blocks), dim3(kThreads), smem, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
