// SPDX-License-Identifier: Apache-2.0
//
// Batch-one BF16 GEMV over several matrices in one launch (the projections of a layer visit, the gates of the
// loop controllers, the router). y_t = W_t x_t for every task t: W_t [n_t][k_t] BF16 row-major, x_t BF16 or FP32,
// y_t BF16 (rounded once) or FP32, FP32 accumulation. A warp owns one output row; the 8 rows of a block belong to
// one task, whose input the block stages once in shared memory. Before the PDL wait a block prefetches its own
// weight rows and its share of the caller's prefetch ranges into L2.

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kRows = 8;               // output rows (warps) per block
#ifndef INKLING2_ROW_PREFETCH
#define INKLING2_ROW_PREFETCH 0
#endif
constexpr bool kRowPrefetch = INKLING2_ROW_PREFETCH;
constexpr int kThreads = kRows * 32;

struct GemvParams {
  GemvTask task[kMaxGemvTasks];
  int block_start[kMaxGemvTasks + 1];  // first block of each task
  int count;
  int max_k;
  Prefetch prefetch;
};

__device__ __forceinline__ float dot8(const uint4& w, const float* x) {
  float acc = 0.0f;
  const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc = fmaf(bf16lo(ww[i]), x[2 * i], acc);
    acc = fmaf(bf16hi(ww[i]), x[2 * i + 1], acc);
  }
  return acc;
}

__global__ void __launch_bounds__(kThreads) gemv_multi_kernel(const __grid_constant__ GemvParams p) {
  extern __shared__ float xs[];                 // the task input as FP32
  int t = 0;
  while (t + 1 < p.count && int(blockIdx.x) >= p.block_start[t + 1]) ++t;
  const GemvTask& task = p.task[t];
  const int row = (int(blockIdx.x) - p.block_start[t]) * kRows + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int k = task.k;
  const bool live = row < task.n;
  const uint4* w = reinterpret_cast<const uint4*>(task.w) + (live ? size_t(row) * (k / 8) : 0);
  if (kRowPrefetch && lane == 0 && live) prefetch_l2(w, uint32_t(k) * 2u);
  prefetch_share(p.prefetch);
  pdl_wait();
  pdl_launch();
  const void* xin = after_wait(task.x);
  // stage the input with 16-byte loads (k is a multiple of 8): one load round per thread for k <= 2048 x 4
  if (task.x_f32) {
    const float4* x = static_cast<const float4*>(xin);
    for (int i = threadIdx.x; i < k / 4; i += kThreads) reinterpret_cast<float4*>(xs)[i] = x[i];
  } else {
    const uint4* x = static_cast<const uint4*>(xin);
    for (int i = threadIdx.x; i < k / 8; i += kThreads) {
      const uint4 v = x[i];
      float4* o = reinterpret_cast<float4*>(xs + 8 * i);
      o[0] = make_float4(bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y));
      o[1] = make_float4(bf16lo(v.z), bf16hi(v.z), bf16lo(v.w), bf16hi(v.w));
    }
  }
  __syncthreads();
  if (task.prologue == kGemvNorm) {
    __shared__ float part[kRows];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < k; i += kThreads) ss = fmaf(xs[i], xs[i], ss);
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < kRows; ++i) tot += part[i];
    const float inv = 1.0f / sqrtf(tot / float(k) + 1e-6f);
    const uint4* nw = static_cast<const uint4*>(task.aux);
    __syncthreads();
    for (int i = threadIdx.x; i < k / 8; i += kThreads) {
      const uint4 v = nw[i];
      const uint32_t ww[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        xs[8 * i + 2 * j] = round_bf16(xs[8 * i + 2 * j] * inv * bf16lo(ww[j]));
        xs[8 * i + 2 * j + 1] = round_bf16(xs[8 * i + 2 * j + 1] * inv * bf16hi(ww[j]));
      }
    }
    __syncthreads();
  } else if (task.prologue == kGemvGate) {
    const uint4* gate = static_cast<const uint4*>(after_wait(task.aux));
    for (int i = threadIdx.x; i < k / 8; i += kThreads) {
      const uint4 v = gate[i];
      const uint32_t gg[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        xs[8 * i + 2 * j] = round_bf16(xs[8 * i + 2 * j] * (1.0f / (1.0f + expf(-bf16lo(gg[j])))));
        xs[8 * i + 2 * j + 1] = round_bf16(xs[8 * i + 2 * j + 1] * (1.0f / (1.0f + expf(-bf16hi(gg[j])))));
      }
    }
    __syncthreads();
  }
  if (!live) return;
  float acc = 0.0f;
  const int k8 = k / 8;
  constexpr int kUnroll = 9;
  int i = lane;
  for (; i + 32 * (kUnroll - 1) < k8; i += 32 * kUnroll) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = ldg_stream(w + i + 32 * u);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) acc += dot8(v[u], xs + 8 * (i + 32 * u));
  }
  for (; i < k8; i += 32) acc += dot8(ldg_stream(w + i), xs + 8 * i);
  acc = warp_sum(acc);
  if (lane == 0) {
    void* y = after_wait(task.y);
    if (task.y_f32) static_cast<float*>(y)[row] = acc;
    else static_cast<uint16_t*>(y)[row] = f32_to_bf16_bits(acc);
  }
}

}  // namespace

void gemv_multi(const GemvTask* tasks, int count, const Prefetch& prefetch, cudaStream_t stream) {
  if (count < 1 || count > kMaxGemvTasks) throw std::invalid_argument("gemv_multi: 1 to 8 tasks");
  GemvParams p = {};
  p.count = count;
  int blocks = 0;
  for (int t = 0; t < count; ++t) {
    const GemvTask& task = tasks[t];
    if (!task.w || !task.x || !task.y || task.n < 1 || task.k < 8 || task.k % 8)
      throw std::invalid_argument("gemv_multi: task needs pointers, n >= 1 and k a multiple of 8");
    p.task[t] = task;
    p.block_start[t] = blocks;
    blocks += (task.n + kRows - 1) / kRows;
    p.max_k = task.k > p.max_k ? task.k : p.max_k;
  }
  p.block_start[count] = blocks;
  p.prefetch = prefetch;
  const size_t smem = size_t(p.max_k) * sizeof(float);
  if (smem > 48 * 1024)
    check_cuda(cudaFuncSetAttribute(gemv_multi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)),
               "gemv_multi smem");
  launch_pdl(gemv_multi_kernel, dim3(blocks), dim3(kThreads), smem, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
