// See pi05_l2_prefetch.cuh.
#include "kernels/pi05/thor/pi05_l2_prefetch.cuh"
#include <cstdint>

namespace flash_rt {
namespace fp4 {
namespace {

constexpr unsigned CHUNK = 16384;   // bytes per prefetch instruction
constexpr int THREADS = 128;

__global__ void l2_touch_kernel(L2PrefetchRegions r, unsigned* __restrict__ sink) {
  // Real 16-byte loads (ld.global.cg allocates in L2); one XOR per CTA keeps them live.
  const int reg = blockIdx.y;
  if (reg >= r.count) return;
  const unsigned long long bytes = r.bytes[reg] & ~15ull;
  const uint4* p = static_cast<const uint4*>(r.ptr[reg]);
  const unsigned long long n16 = bytes >> 4;
  unsigned acc = 0;
  for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * THREADS + threadIdx.x;
       i < n16; i += static_cast<unsigned long long>(gridDim.x) * THREADS) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) sink[0] = acc;   // practically never true; defeats DCE
}

__global__ void l2_prefetch_kernel(L2PrefetchRegions r) {
  // CTA x covers region r.ptr[blockIdx.y] chunks [x*THREADS, ...)
  const int reg = blockIdx.y;
  if (reg >= r.count) return;
  const unsigned long long bytes = r.bytes[reg];
  const unsigned long long chunk = (static_cast<unsigned long long>(blockIdx.x) * THREADS + threadIdx.x) * CHUNK;
  if (chunk >= bytes) return;
  const unsigned long long len = (bytes - chunk < CHUNK) ? (bytes - chunk) : CHUNK;
  const uint8_t* p = static_cast<const uint8_t*>(r.ptr[reg]) + chunk;
  const unsigned n = static_cast<unsigned>(len) & ~15u;   // multiple of 16
  if (n == 0) return;
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(n) : "memory");
}

template <int HINT, int DEPTH>
__global__ void __launch_bounds__(256) l2_touch_ex_kernel(L2PrefetchRegions r, unsigned* __restrict__ sink, unsigned pace_ns) {
  unsigned acc = 0;
  unsigned long long pol = 0;
  if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  const unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (int reg = 0; reg < r.count; ++reg) {
    const uint4* p = static_cast<const uint4*>(r.ptr[reg]);
    const unsigned long long n16 = (r.bytes[reg] & ~15ull) >> 4;
    unsigned long long i = tid;
    for (; i + (DEPTH - 1) * stride < n16; i += DEPTH * stride) {
      uint4 v[DEPTH];
#pragma unroll
      for (int u = 0; u < DEPTH; ++u) {
        const uint4* a = p + i + static_cast<unsigned long long>(u) * stride;
        if (HINT) {
          asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                       : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(a), "l"(pol));
        } else {
          asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(a));
        }
      }
#pragma unroll
      for (int u = 0; u < DEPTH; ++u) acc ^= v[u].x ^ v[u].w;
      if (pace_ns) __nanosleep(pace_ns);
    }
    for (; i < n16; i += stride) {
      uint4 v;
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
      acc ^= v.x ^ v.w;
    }
  }
  if (acc == 0x9E3779B9u) sink[0] = acc;   // practically never true; defeats DCE
}

// TMA bulk-copy touch: one elected thread per CTA streams 32 KB chunks of the regions into a smem
// ring (STAGES deep) with mbarrier transaction counting; the data is discarded, the lines stay in L2.
// No LSU traffic, so it coexists with the GEMM CTAs on the same SM; ~STAGES*32 KB in flight per CTA.
constexpr int kBulkMaxStages = 16;

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" :: "r"(a), "r"(parity) : "memory");
}
template <int HINT>
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar, unsigned long long pol) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const unsigned b = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  if (HINT) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 :: "r"(d), "l"(src), "r"(bytes), "r"(b), "l"(pol) : "memory");
  } else {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
  }
}

template <int HINT>
__global__ void __launch_bounds__(32) l2_touch_bulk_kernel(L2PrefetchRegions r, int stages, unsigned chunk) {
  extern __shared__ __align__(1024) uint8_t smem_ring[];
  __shared__ __align__(8) uint64_t bars[kBulkMaxStages];
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) mbar_init(&bars[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  unsigned long long pol = 0;
  if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  // Chunks of each region are dealt to the CTAs by stride (no division in the loop); the ring
  // index wraps with a counter and the wait parity flips on every wrap.
  // Stage s has been used `wraps` times (+1 if s < current position); the copy that must finish
  // before stage s is reused is its (wraps-1)-th, i.e. mbarrier phase (wraps-1).
  int s = 0; unsigned wraps = 0;
  const unsigned long long cstride = static_cast<unsigned long long>(gridDim.x) * chunk;
  for (int reg = 0; reg < r.count; ++reg) {
    const uint8_t* base = static_cast<const uint8_t*>(r.ptr[reg]);
    const unsigned long long bytes = r.bytes[reg] & ~15ull;
    for (unsigned long long off = static_cast<unsigned long long>(blockIdx.x) * chunk; off < bytes; off += cstride) {
      const unsigned n = static_cast<unsigned>((bytes - off < chunk) ? (bytes - off) : chunk);
      if (wraps > 0) mbar_wait(&bars[s], (wraps - 1) & 1u);
      mbar_expect_tx(&bars[s], n);
      bulk_g2s<HINT>(smem_ring + static_cast<size_t>(s) * chunk, base + off, n, &bars[s], pol);
      if (++s == stages) { s = 0; ++wraps; }
    }
  }
  // drain: stages [s, stages) were last written in wrap (wraps-1), stages [0, s) in wrap `wraps`
  if (wraps == 0) {
    for (int j = 0; j < s; ++j) mbar_wait(&bars[j], 0u);
  } else {
    for (int j = s; j < stages; ++j) mbar_wait(&bars[j], (wraps - 1) & 1u);
    for (int j = 0; j < s; ++j) mbar_wait(&bars[j], wraps & 1u);
  }
}


__device__ __forceinline__ bool mbar_test(uint64_t* bar, unsigned parity) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  unsigned ok;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  return ok != 0u;
}

// Multi-thread issuing form: one issuing thread per warp (T of them), each with its own
// stages/T ring slots and mbarriers, chunks dealt round-robin over (CTA, thread). A single
// thread only keeps one bulk copy in flight on this part (~0.5 us per request whatever the
// size), so T threads multiply the per-CTA rate. Waits spin on test_wait (dedicated SM).
template <int HINT>
__global__ void __launch_bounds__(256) l2_touch_bulk_mt_kernel(L2PrefetchRegions r, int stages, unsigned chunk, int T) {
  extern __shared__ __align__(1024) uint8_t smem_ring[];
  __shared__ __align__(8) uint64_t bars[kBulkMaxStages];
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) mbar_init(&bars[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if ((threadIdx.x & 31) != 0) return;
  const int t = static_cast<int>(threadIdx.x >> 5);
  if (t >= T) return;
  unsigned long long pol = 0;
  if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  const int sp = stages / T;
  uint64_t* fb = bars + t * sp;
  uint8_t* ring = smem_ring + static_cast<size_t>(t) * sp * chunk;
  int s = 0; unsigned wraps = 0;
  const unsigned long long cstride = static_cast<unsigned long long>(gridDim.x) * T * chunk;
  for (int reg = 0; reg < r.count; ++reg) {
    const uint8_t* base = static_cast<const uint8_t*>(r.ptr[reg]);
    const unsigned long long bytes = r.bytes[reg] & ~15ull;
    for (unsigned long long off = (static_cast<unsigned long long>(blockIdx.x) * T + t) * chunk; off < bytes; off += cstride) {
      const unsigned n = static_cast<unsigned>((bytes - off < chunk) ? (bytes - off) : chunk);
      if (wraps > 0) { while (!mbar_test(&fb[s], (wraps - 1) & 1u)) { } }
      mbar_expect_tx(&fb[s], n);
      bulk_g2s<HINT>(ring + static_cast<size_t>(s) * chunk, base + off, n, &fb[s], pol);
      if (++s == sp) { s = 0; ++wraps; }
    }
  }
  if (wraps == 0) {
    for (int j = 0; j < s; ++j) { while (!mbar_test(&fb[j], 0u)) { } }
  } else {
    for (int j = s; j < sp; ++j) { while (!mbar_test(&fb[j], (wraps - 1) & 1u)) { } }
    for (int j = 0; j < s; ++j) { while (!mbar_test(&fb[j], wraps & 1u)) { } }
  }
}

cudaStream_t g_touch_side = nullptr;
cudaEvent_t g_touch_fork = nullptr;
cudaEvent_t g_touch_join = nullptr;

}  // namespace

int pi05_l2_touch_init() {
  if (g_touch_side != nullptr) return 0;
  if (cudaStreamCreateWithFlags(&g_touch_side, cudaStreamNonBlocking) != cudaSuccess) return -1;
  if (cudaEventCreateWithFlags(&g_touch_fork, cudaEventDisableTiming) != cudaSuccess) return -2;
  if (cudaEventCreateWithFlags(&g_touch_join, cudaEventDisableTiming) != cudaSuccess) return -3;
  return 0;
}

int pi05_l2_touch_regions_ex(const L2PrefetchRegions& regions, cudaStream_t stream, int nctas, int hint, void* sink, int depth, unsigned pace_ns, int nthreads) {
  if (regions.count <= 0 || regions.count > kL2TouchMaxRegions || nctas < 1 || nctas > 64 || sink == nullptr) return -1;
  if (nthreads < 32 || nthreads > 256 || (nthreads & 31)) return -1;
  for (int i = 0; i < regions.count; ++i)
    if ((reinterpret_cast<uintptr_t>(regions.ptr[i]) & 15) != 0) return -1;
  unsigned* s = static_cast<unsigned*>(sink);
  if (hint >= 4) {
    // multi-thread bulk form: `depth` = ring stages (rounded down to a multiple of T), `pace_ns` = chunk bytes,
    // T = nthreads / 32 issuing threads (1..8)
    const unsigned chunk = (pace_ns >= 1024 && pace_ns <= 65536 && (pace_ns & 15u) == 0) ? pace_ns : 32768u;
    int T = nthreads / 32; if (T < 1) T = 1; if (T > 8) T = 8;
    int stages = (depth >= 1 && depth <= kBulkMaxStages && static_cast<size_t>(depth) * chunk <= 200 * 1024) ? depth : 4;
    stages -= stages % T; if (stages < T) return -3;
    const size_t smem = static_cast<size_t>(stages) * chunk;
    static bool attr_set_mt = false;
    if (!attr_set_mt) {
      if (cudaFuncSetAttribute(l2_touch_bulk_mt_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024) != cudaSuccess) return -2;
      if (cudaFuncSetAttribute(l2_touch_bulk_mt_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024) != cudaSuccess) return -2;
      attr_set_mt = true;
    }
    if (hint == 5) l2_touch_bulk_mt_kernel<1><<<nctas, 32 * T, smem, stream>>>(regions, stages, chunk, T);
    else           l2_touch_bulk_mt_kernel<0><<<nctas, 32 * T, smem, stream>>>(regions, stages, chunk, T);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
  }
  if (hint >= 2) {
    // bulk form: `depth` = ring stages (1..8, default 4), `pace_ns` = chunk bytes (multiple of 16, default 32 KB)
    const unsigned chunk = (pace_ns >= 1024 && pace_ns <= 65536 && (pace_ns & 15u) == 0) ? pace_ns : 32768u;
    // ring stages from `depth`; anything that would not fit next to the GEMM smem budget falls back to 4
    const int stages = (depth >= 1 && depth <= kBulkMaxStages && static_cast<size_t>(depth) * chunk <= 200 * 1024) ? depth : 4;
    const size_t smem = static_cast<size_t>(stages) * chunk;
    static bool attr_set = false;
    if (!attr_set) {
      if (cudaFuncSetAttribute(l2_touch_bulk_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024) != cudaSuccess) return -2;
      if (cudaFuncSetAttribute(l2_touch_bulk_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024) != cudaSuccess) return -2;
      attr_set = true;
    }
    if (smem > 200 * 1024) return -3;
    if (hint == 3) l2_touch_bulk_kernel<1><<<nctas, 32, smem, stream>>>(regions, stages, chunk);
    else           l2_touch_bulk_kernel<0><<<nctas, 32, smem, stream>>>(regions, stages, chunk);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
  }
  if (depth <= 4) { if (hint) l2_touch_ex_kernel<1, 4><<<nctas, nthreads, 0, stream>>>(regions, s, pace_ns); else l2_touch_ex_kernel<0, 4><<<nctas, nthreads, 0, stream>>>(regions, s, pace_ns); }
  else            { if (hint) l2_touch_ex_kernel<1, 16><<<nctas, nthreads, 0, stream>>>(regions, s, pace_ns); else l2_touch_ex_kernel<0, 16><<<nctas, nthreads, 0, stream>>>(regions, s, pace_ns); }
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int pi05_l2_touch_fork(const L2PrefetchRegions& regions, cudaStream_t main_stream, int nctas, int hint, void* sink, int depth, unsigned pace_ns, int nthreads) {
  const int rc = pi05_l2_touch_init();
  if (rc != 0) return rc;
  if (cudaEventRecord(g_touch_fork, main_stream) != cudaSuccess) return -4;
  if (cudaStreamWaitEvent(g_touch_side, g_touch_fork, 0) != cudaSuccess) return -5;
  return pi05_l2_touch_regions_ex(regions, g_touch_side, nctas, hint, sink, depth, pace_ns, nthreads);
}

int pi05_l2_touch_join(cudaStream_t main_stream) {
  if (g_touch_side == nullptr) return 0;
  if (cudaEventRecord(g_touch_join, g_touch_side) != cudaSuccess) return -6;
  if (cudaStreamWaitEvent(main_stream, g_touch_join, 0) != cudaSuccess) return -7;
  return 0;
}

int pi05_l2_prefetch_regions(const L2PrefetchRegions& regions, cudaStream_t stream, int mode, void* sink) {
  if (regions.count <= 0 || regions.count > 8) return -1;
  if (mode == 1) {
    const dim3 grid(64, regions.count);   // 64 x 128 threads per region, grid-stride
    l2_touch_kernel<<<grid, THREADS, 0, stream>>>(regions, static_cast<unsigned*>(sink));
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
  }
  unsigned long long maxb = 0;
  for (int i = 0; i < regions.count; ++i) {
    if ((reinterpret_cast<uintptr_t>(regions.ptr[i]) & 15) != 0) return -1;
    if (regions.bytes[i] > maxb) maxb = regions.bytes[i];
  }
  const unsigned chunks = static_cast<unsigned>((maxb + CHUNK - 1) / CHUNK);
  const dim3 grid((chunks + THREADS - 1) / THREADS, regions.count);
  l2_prefetch_kernel<<<grid, THREADS, 0, stream>>>(regions);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt
