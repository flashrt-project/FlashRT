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
  // the first round of this warp's weights into registers before the wait (static data)
  constexpr int kUnroll = 9;
  const int k8 = k / 8;
  uint4 v0[kUnroll];
  const bool early = live && lane + 32 * (kUnroll - 1) < k8;
  if (early) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v0[u] = ldg_stream(w + lane + 32 * u);
  }
  // the norm weight (static) before the wait: uint4 t and t + 256 (k <= 4096)
  uint4 nwp[2];
  const bool nwpre = task.prologue == kGemvNorm && k / 8 <= 2 * kThreads;
  if (nwpre) {
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int i = threadIdx.x + m * kThreads;
      nwp[m] = i < k / 8 ? static_cast<const uint4*>(task.aux)[i] : uint4{};
    }
  }
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
    auto apply = [&](int i, const uint4& v) {
      const uint32_t ww[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        xs[8 * i + 2 * j] = round_bf16(xs[8 * i + 2 * j] * inv * bf16lo(ww[j]));
        xs[8 * i + 2 * j + 1] = round_bf16(xs[8 * i + 2 * j + 1] * inv * bf16hi(ww[j]));
      }
    };
    if (nwpre) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int i = threadIdx.x + m * kThreads;
        if (i < k / 8) apply(i, nwp[m]);
      }
    } else {
      for (int i = threadIdx.x; i < k / 8; i += kThreads) apply(i, nw[i]);
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
  int i = lane;
  bool first = early;
  for (; i + 32 * (kUnroll - 1) < k8; i += 32 * kUnroll) {
    uint4 v[kUnroll];
    if (first) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) v[u] = v0[u];
      first = false;
    } else {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) v[u] = ldg_stream(w + i + 32 * u);
    }
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

// Several rows: the inputs staged as BF16 [rows][k] (the values the one-row kernel computes in FP32 are BF16
// values), the same prologue arithmetic per row, R accumulators per lane in the one-row order. kRowsWarps output
// rows per block (fewer blocks staging the inputs); the norm sums over the first kThreads threads as in the one-row
// kernel.
constexpr int kRowsWarps = 8;               // 16 measured slower (one block per SM either way)
constexpr int kRowsThreads = kRowsWarps * 32;
// RT: the largest row count of the launch (RT accumulators per lane)
template <int RT>
__global__ void __launch_bounds__(kRowsThreads) gemv_rows_kernel(const __grid_constant__ GemvParams p) {
  extern __shared__ __align__(16) uint16_t xb[];
  int t = 0;
  while (t + 1 < p.count && int(blockIdx.x) >= p.block_start[t + 1]) ++t;
  const GemvTask& task = p.task[t];
  const int row = (int(blockIdx.x) - p.block_start[t]) * kRowsWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int k = task.k, R = task.rows < 1 ? 1 : task.rows;
  const size_t xs = task.x_stride ? task.x_stride : k;
  const bool live = row < task.n;
  const uint4* w = reinterpret_cast<const uint4*>(task.w) + (live ? size_t(row) * (k / 8) : 0);
  prefetch_share(p.prefetch);
  // the first round of this warp's weights into registers before the wait (static data)
  constexpr int kUnrollR = 9;
  uint4 v0[kUnrollR];
  // (up to 4 rows; beyond, the accumulators and the early round spill)
  const bool early = RT <= 4 && live && lane + 32 * (kUnrollR - 1) < k / 8;
  if (early) {
#pragma unroll
    for (int u = 0; u < kUnrollR; ++u) v0[u] = ldg_stream(w + lane + 32 * u);
  }
  pdl_wait();
  pdl_launch();
  const void* xin = after_wait(task.x);
  // all rows at once: each thread visits the elements t, t + 256, ... of every row in the one-row order
  if (task.prologue == kGemvNorm) {
    __shared__ float rp[kRows][RT];
    float ss[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) ss[r] = 0.0f;
    for (int i = threadIdx.x; threadIdx.x < kThreads && i < k; i += kThreads) {
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= R) break;
        const float v = task.x_f32 ? static_cast<const float*>(xin)[r * xs + i]
                                   : bf16_to_f32(static_cast<const uint16_t*>(xin)[r * xs + i]);
        ss[r] = fmaf(v, v, ss[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      if (r >= R) break;
      const float s = warp_sum(ss[r]);
      if ((threadIdx.x & 31) == 0 && threadIdx.x < kThreads) rp[threadIdx.x >> 5][r] = s;
    }
    __syncthreads();
    float inv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      float tot = 0.0f;
#pragma unroll
      for (int i = 0; i < kRows; ++i) tot += r < R ? rp[i][r] : 0.0f;
      inv[r] = 1.0f / sqrtf(tot / float(k) + 1e-6f);
    }
    const uint16_t* nw = static_cast<const uint16_t*>(task.aux);
    for (int i = threadIdx.x; i < k; i += kRowsThreads) {
      const float wv = bf16_to_f32(nw[i]);
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= R) break;
        const float v = task.x_f32 ? static_cast<const float*>(xin)[r * xs + i]
                                   : bf16_to_f32(static_cast<const uint16_t*>(xin)[r * xs + i]);
        xb[r * k + i] = f32_to_bf16_bits(v * inv[r] * wv);
      }
    }
  } else if (task.prologue == kGemvGate) {
    const uint16_t* gate = static_cast<const uint16_t*>(after_wait(task.aux));
    for (int i = threadIdx.x; i < k; i += kRowsThreads)
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= R) break;
        const float x = bf16_to_f32(static_cast<const uint16_t*>(xin)[r * xs + i]);
        xb[r * k + i] = f32_to_bf16_bits(x * (1.0f / (1.0f + expf(-bf16_to_f32(gate[size_t(r) * k + i])))));
      }
  } else {
    // 8 loads of a thread in flight per round
    const int k8 = k / 8, total = R * k8;
    for (int base = threadIdx.x; base < total; base += 8 * kRowsThreads) {
      uint4 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int i = base + u * kRowsThreads;
        if (i < total) v[u] = reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(xin) + (i / k8) * xs)[i % k8];
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int i = base + u * kRowsThreads;
        if (i < total) reinterpret_cast<uint4*>(xb)[i] = v[u];
      }
    }
  }
  __syncthreads();
  if (!live) return;
  float acc[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r) acc[r] = 0.0f;
  const int k8 = k / 8;
  // every one of the RT rows each step (rows past R compute on stale staging and are not stored): the rows' chains
  // interleave (with a break after each row the compiler serializes them)
  auto step = [&](const uint4& v, int i) {
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint4 xv = reinterpret_cast<const uint4*>(xb + r * k)[i];
      const float xf[8] = {bf16lo(xv.x), bf16hi(xv.x), bf16lo(xv.y), bf16hi(xv.y),
                           bf16lo(xv.z), bf16hi(xv.z), bf16lo(xv.w), bf16hi(xv.w)};
      acc[r] += dot8(v, xf);
    }
  };
  constexpr int kUnroll = kUnrollR;
  int i = lane;
  if (early) {        // the rounds loaded before the wait, in the loop's order
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) step(v0[u], i + 32 * u);
    i += 32 * kUnroll;
  }
  for (; i + 32 * (kUnroll - 1) < k8; i += 32 * kUnroll) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = ldg_stream(w + i + 32 * u);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) step(v[u], i + 32 * u);
  }
  for (; i < k8; i += 32) step(ldg_stream(w + i), i);
  void* y = after_wait(task.y);
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    if (r >= R) break;
    const float s = warp_sum(acc[r]);
    if (lane == 0) {
      if (task.y_f32) static_cast<float*>(y)[size_t(r) * task.n + row] = s;
      else static_cast<uint16_t*>(y)[size_t(r) * task.n + row] = f32_to_bf16_bits(s);
    }
  }
}

struct NormRowsParams {
  NormRowsTask task[kMaxNormTasks];
  int block_start[kMaxNormTasks + 1];
  int count;
};

// One block per (task, row): the staging and the arithmetic of gemv_multi_kernel's kGemvNorm prologue.
__global__ void __launch_bounds__(kThreads) norm_rows_kernel(const __grid_constant__ NormRowsParams p) {
  extern __shared__ float xs[];
  __shared__ float part[kRows];
  int t = 0;
  while (t + 1 < p.count && int(blockIdx.x) >= p.block_start[t + 1]) ++t;
  const NormRowsTask& task = p.task[t];
  const int r = int(blockIdx.x) - p.block_start[t];
  const int k = task.k;
  const size_t stride = task.x_stride ? task.x_stride : k;
  // the norm weight (static) into registers before the wait: elements t, t + 256, .. (k <= 9 x 256)
  constexpr int kNW = 9;
  float wv[kNW];
  const bool wpre = k <= kNW * kThreads;
  if (wpre) {
#pragma unroll
    for (int m = 0; m < kNW; ++m) {
      const int i = threadIdx.x + m * kThreads;
      wv[m] = i < k ? bf16_to_f32(task.w[i]) : 0.0f;
    }
  }
  pdl_wait();
  const void* xin = after_wait(task.x);
  for (int i = threadIdx.x; i < k; i += kThreads)
    xs[i] = task.x_f32 ? static_cast<const float*>(xin)[r * stride + i]
                       : bf16_to_f32(static_cast<const uint16_t*>(xin)[r * stride + i]);
  __syncthreads();
  pdl_launch();                // (once the rows are loaded: the dependents' early weight loads would delay them)
  float ss = 0.0f;
  for (int i = threadIdx.x; i < k; i += kThreads) ss = fmaf(xs[i], xs[i], ss);
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int i = 0; i < kRows; ++i) tot += part[i];
  const float inv = 1.0f / sqrtf(tot / float(k) + 1e-6f);
  uint16_t* out = after_wait(task.out) + size_t(r) * k;
  if (wpre) {
#pragma unroll
    for (int m = 0; m < kNW; ++m) {
      const int i = threadIdx.x + m * kThreads;
      if (i < k) out[i] = f32_to_bf16_bits(xs[i] * inv * wv[m]);
    }
  } else {
    for (int i = threadIdx.x; i < k; i += kThreads) out[i] = f32_to_bf16_bits(xs[i] * inv * bf16_to_f32(task.w[i]));
  }
}

}  // namespace

void norm_rows(const NormRowsTask* tasks, int count, cudaStream_t stream) {
  if (count < 1 || count > kMaxNormTasks) throw std::invalid_argument("norm_rows: 1 to 4 tasks");
  NormRowsParams p = {};
  p.count = count;
  int blocks = 0, max_k = 0;
  for (int t = 0; t < count; ++t) {
    const NormRowsTask& task = tasks[t];
    if (!task.x || !task.w || !task.out || task.k < 8 || task.rows < 1 || task.rows > kMaxChunk)
      throw std::invalid_argument("norm_rows: task needs pointers, k >= 8 and 1 to kMaxChunk rows");
    p.task[t] = task;
    p.block_start[t] = blocks;
    blocks += task.rows;
    max_k = task.k > max_k ? task.k : max_k;
  }
  p.block_start[count] = blocks;
  const size_t smem = size_t(max_k) * sizeof(float);
  if (smem > 48 * 1024)
    check_cuda(cudaFuncSetAttribute(norm_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)),
               "norm_rows smem");
  launch_pdl(norm_rows_kernel, dim3(blocks), dim3(kThreads), smem, stream, p);
}

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
  int max_rows = 1;
  for (int t = 0; t < count; ++t) max_rows = tasks[t].rows > max_rows ? tasks[t].rows : max_rows;
  if (max_rows > 1) {
    blocks = 0;
    for (int t = 0; t < count; ++t) {
      p.block_start[t] = blocks;
      blocks += (tasks[t].n + kRowsWarps - 1) / kRowsWarps;
    }
    p.block_start[count] = blocks;
    size_t smem = 0;
    for (int t = 0; t < count; ++t) {
      const GemvTask& task = tasks[t];
      const int r = task.rows < 1 ? 1 : task.rows;
      if (r > kMaxRows) throw std::invalid_argument("gemv_multi: at most 8 rows");
      if (task.x_f32 && task.prologue != kGemvNorm) throw std::invalid_argument("gemv_multi: FP32 rows need the norm");
      smem = smem > size_t(r) * task.k * 2 ? smem : size_t(r) * task.k * 2;
    }
    {      // every one of the kernel's RT rows is read: the staging sized for RT rows of the largest k
      int rmax = 1, kmax = 0;
      for (int t = 0; t < count; ++t) {
        rmax = tasks[t].rows > rmax ? tasks[t].rows : rmax;
        kmax = tasks[t].k > kmax ? tasks[t].k : kmax;
      }
      smem = smem > size_t(rmax) * kmax * 2 ? smem : size_t(rmax) * kmax * 2;
    }
    // the kernel for the largest row count (RT accumulators per lane)
    auto run = [&](auto kernel) {
      check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)), "gemv_rows smem");
      launch_pdl(kernel, dim3(blocks), dim3(kRowsThreads), smem, stream, p);
    };
    switch (max_rows) {
      case 2: run(gemv_rows_kernel<2>); break;
      case 3: run(gemv_rows_kernel<3>); break;
      case 4: run(gemv_rows_kernel<4>); break;
      case 5: run(gemv_rows_kernel<5>); break;
      case 6: run(gemv_rows_kernel<6>); break;
      case 7: run(gemv_rows_kernel<7>); break;
      default: run(gemv_rows_kernel<8>); break;
    }
    return;
  }
  const size_t smem = size_t(p.max_k) * sizeof(float);
  if (smem > 48 * 1024)
    check_cuda(cudaFuncSetAttribute(gemv_multi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)),
               "gemv_multi smem");
  launch_pdl(gemv_multi_kernel, dim3(blocks), dim3(kThreads), smem, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt
