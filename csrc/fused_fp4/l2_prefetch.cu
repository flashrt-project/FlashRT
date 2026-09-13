// See l2_prefetch.cuh.
#include "fused_fp4/l2_prefetch.cuh"
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

}  // namespace

int l2_prefetch_regions(const L2PrefetchRegions& regions, cudaStream_t stream, int mode, void* sink) {
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
