// SPDX-License-Identifier: Apache-2.0
//
// Shared device helpers for the Inkling-2-Mini SM120 kernels (module flash_rt_inkling2).
//
// Numerics: BF16 storage between operations, FP32 inside them, as the reference (mlx-lm's inkling2
// package); the routed experts quantize their inputs as the checkpoint requires (NVFP4 fouroversix_mae,
// MXFP8 UE8M0). Compiled without fast-math.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <atomic>
#include <cstdlib>
#include <utility>

#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
  __nv_bfloat16 h = __float2bfloat16_rn(f);
  return *reinterpret_cast<uint16_t*>(&h);
}
__device__ __forceinline__ float round_bf16(float f) { return bf16_to_f32(f32_to_bf16_bits(f)); }

template <int W = 32>
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = W / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int W = 32>
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = W / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ldg_stream2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

// ---------------------------------------------------------------------------------------------------------------
// Programmatic dependent launch. A kernel launched with launch_pdl may start while the previous kernel of the
// stream still runs. Before pdl_wait() it reads only static data (weights) and data written two or more launches
// back (final once the previous kernel has started), and writes only memory that the previous kernel neither reads
// nor writes (work counters and scratch reset two or more launches back). Pointers to data produced by the previous
// grid go through after_wait() so no load is hoisted above the wait.
// ---------------------------------------------------------------------------------------------------------------
// L2 prefetches of more than ~64 KiB per instruction are mostly dropped on SM120 (measured: a 40 MB GEMV after
// 1 MiB prefetches runs at the cold rate, after <= 64 KiB prefetches at the L2 rate); ranges go out in 64 KiB pieces.
// Per-device host setup (kernel attributes): true on the first call on the current device for the caller's flag word
// (one bit per device ordinal below 64).
inline bool first_use_on_device(std::atomic<unsigned long long>& done) {
  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "device");
  const unsigned long long bit = 1ull << (dev & 63);
  return (done.fetch_or(bit) & bit) == 0;
}
constexpr uint32_t kPrefetchChunk = 64u << 10;
__device__ __forceinline__ void prefetch_l2(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))),
               "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// The checkpoint's MLP activation: softcapped SwiGLU of the gate / up projections.
__device__ __forceinline__ float softcapped_swiglu(float g, float u, float gcap, float ucap) {
  const float gc = tanhf(g / gcap) * gcap;
  const float uc = tanhf(u / ucap) * ucap;
  return gc * (1.0f / (1.0f + expf(-g))) * uc;
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" :::); }
template <typename T>
__device__ __forceinline__ T* after_wait(T* p) {
  asm volatile("mov.b64 %0, %0;" : "+l"(p)::"memory");
  return p;
}

// The caller's prefetch ranges (Prefetch, inkling2_kernels.cuh) spread over the blocks of the grid that carries
// them: each block takes an equal share, in 4 KiB multiples.
__device__ __forceinline__ void prefetch_share(const Prefetch& pf) {
  const uint32_t blocks = gridDim.x * gridDim.y * gridDim.z;
  const uint32_t b = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  if (threadIdx.x != 0) return;
  for (int i = 0; i < pf.count; ++i) {
    const uint64_t total = pf.bytes[i];
    uint64_t share = (total + blocks - 1) / blocks;
    share = (share + 4095) & ~uint64_t(4095);
    const uint64_t lo = uint64_t(b) * share;
    if (lo >= total) continue;
    uint64_t n = total - lo < share ? total - lo : share;
    n &= ~uint64_t(15);
    const char* base = static_cast<const char*>(pf.ptr[i]) + lo;
    while (n) {
      const uint32_t chunk = n > kPrefetchChunk ? kPrefetchChunk : static_cast<uint32_t>(n);
      prefetch_l2(base, chunk);
      base += chunk;
      n -= chunk;
    }
  }
}

// Extra prefetch-only blocks of a vector kernel: blocks first .. gridDim.x - 1 share the ranges and return true.
__device__ __forceinline__ bool prefetch_block(const Prefetch& pf, int first) {
  if (int(blockIdx.x) < first) return false;
  if (threadIdx.x != 0) return true;
  const uint32_t blocks = gridDim.x - first, b = blockIdx.x - first;
  for (int i = 0; i < pf.count; ++i) {
    const uint64_t total = pf.bytes[i];
    uint64_t share = ((total + blocks - 1) / blocks + 4095) & ~uint64_t(4095);
    const uint64_t lo = uint64_t(b) * share;
    if (lo >= total) continue;
    uint64_t n = (total - lo < share ? total - lo : share) & ~uint64_t(15);
    const char* base = static_cast<const char*>(pf.ptr[i]) + lo;
    for (; n; ) {
      const uint32_t chunk = n > kPrefetchChunk ? kPrefetchChunk : static_cast<uint32_t>(n);
      prefetch_l2(base, chunk);
      base += chunk;
      n -= chunk;
    }
  }
  return true;
}

inline bool pdl_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("INKLING2_PDL");
    return !(v && v[0] == '0');
  }();
  return on;
}

template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args&&... args) {
  if (!pdl_enabled()) {
    kernel<<<grid, block, smem, s>>>(std::forward<Args>(args)...);
    check_cuda(cudaGetLastError(), "kernel launch");
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  check_cuda(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...), "cudaLaunchKernelEx");
}

}  // namespace inkling2
}  // namespace flash_rt
