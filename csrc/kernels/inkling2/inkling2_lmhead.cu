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

__device__ __forceinline__ float dot_row(const uint4* w, const float* xs, int lane) {
  float acc = 0.0f;
#pragma unroll
  for (int u = 0; u < kHidden / 8 / 32; ++u) {
    const int i = lane + 32 * u;
    const uint4 v = ldg_stream(w + i);
    const uint32_t ww[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc = fmaf(bf16lo(ww[j]), xs[8 * i + 2 * j], acc);
      acc = fmaf(bf16hi(ww[j]), xs[8 * i + 2 * j + 1], acc);
    }
  }
  return warp_sum(acc);
}

__global__ void __launch_bounds__(kThreads) lm_head_kernel(const __grid_constant__ LmHeadParams p) {
  __shared__ float xs[kHidden];
  __shared__ float scratch[8];
  __shared__ float bv[8];
  __shared__ int bi[8];
  __shared__ int last;
  prefetch_share(p.prefetch);
  pdl_wait();
  pdl_launch();
  const float* in = after_wait(p.in);
  float ss = 0.0f;
  for (int i = threadIdx.x; i < kHidden; i += kThreads) {
    xs[i] = in[i];
    ss = fmaf(xs[i], xs[i], ss);
  }
  ss = warp_sum(ss);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) scratch[warp] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += scratch[i];
  const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
  __syncthreads();
  for (int i = threadIdx.x; i < kHidden; i += kThreads) xs[i] = round_bf16(xs[i] * inv * bf16_to_f32(p.norm_w[i]));
  __syncthreads();
  float best = -INFINITY;
  int bidx = 1 << 30;
  const int warps = gridDim.x * 8;
  const int gw = blockIdx.x * 8 + warp;
  float* out = p.logits ? after_wait(p.logits) : nullptr;
  for (int r = gw; r < p.vocab; r += 2 * warps) {
    const int r2 = r + warps;
    const uint4* w0 = reinterpret_cast<const uint4*>(p.weight) + size_t(r) * (kHidden / 8);
    const float l0 = dot_row(w0, xs, lane);
    float l1 = -INFINITY;
    if (r2 < p.vocab) l1 = dot_row(reinterpret_cast<const uint4*>(p.weight) + size_t(r2) * (kHidden / 8), xs, lane);
    const float c0 = tanhf(l0 / p.softcap) * p.softcap;
    const float c1 = r2 < p.vocab ? tanhf(l1 / p.softcap) * p.softcap : -INFINITY;
    if (lane == 0) {
      if (out) {
        out[r] = c0;
        if (r2 < p.vocab) out[r2] = c1;
      }
      if (c0 > best || (c0 == best && r < bidx)) { best = c0; bidx = r; }
      if (c1 > best || (c1 == best && r2 < bidx)) { best = c1; bidx = r2; }
    }
  }
  if (lane == 0) { bv[warp] = best; bi[warp] = bidx; }
  __syncthreads();
  float* gv = after_wait(p.block_best);
  int* gi = after_wait(p.block_index);
  if (threadIdx.x == 0) {
    float b = bv[0];
    int ix = bi[0];
    for (int w = 1; w < 8; ++w)
      if (bv[w] > b || (bv[w] == b && bi[w] < ix)) { b = bv[w]; ix = bi[w]; }
    gv[blockIdx.x] = b;
    gi[blockIdx.x] = ix;
    __threadfence();
    last = atomicAdd(after_wait(p.counter), 1) == int(gridDim.x) - 1;
  }
  __syncthreads();
  if (!last || warp != 0) return;
  __threadfence();
  float b = -INFINITY;
  int ix = 1 << 30;
  for (int i = lane; i < int(gridDim.x); i += 32) {
    const float v = gv[i];
    const int j = gi[i];
    if (v > b || (v == b && j < ix)) { b = v; ix = j; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, b, o);
    const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
    if (ov > b || (ov == b && oi < ix)) { b = ov; ix = oi; }
  }
  if (lane == 0) {
    *after_wait(p.token) = ix;
    *after_wait(p.counter) = 0;
  }
}

}  // namespace

void lm_head(const LmHeadParams& p, cudaStream_t stream) {
  if (p.blocks < 1) throw std::invalid_argument("lm_head: blocks >= 1");
  launch_pdl(lm_head_kernel, dim3(p.blocks), dim3(kThreads), 0, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
