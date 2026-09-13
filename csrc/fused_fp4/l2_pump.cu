// See l2_pump.cuh.
#include "fused_fp4/l2_pump.cuh"
#include "fused_fp4/pdl.cuh"
#include <cstdint>

namespace flash_rt {
namespace fp4 {
namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__global__ void __launch_bounds__(kThreads) l2_pump_kernel(L2PumpArgs a) {
  __shared__ int s_go;
  for (int i = 0; i < a.total_units; ++i) {
    const int u = i % a.nunits;
    if (a.progress != nullptr) {
      if (threadIdx.x == 0) {
        const int need = i - a.ahead;
        unsigned long long spins = 0; int go = 1;
        while (ld_acquire(a.progress) < need) {
          __nanosleep(256);
          if (++spins > a.spin_limit) { go = 0; break; }
        }
        s_go = go;
      }
      __syncthreads();
      if (!s_go) return;   // consumer never arrived: stop pumping rather than hang
    }
    for (int c = a.unit_begin[u]; c < a.unit_begin[u + 1]; ++c) {
      const uint8_t* base = static_cast<const uint8_t*>(a.ptr[c]);
      const unsigned long long nb = a.bytes[c];
      for (unsigned long long off = static_cast<unsigned long long>(blockIdx.x * kThreads + threadIdx.x) * a.chunk_bytes;
           off < nb; off += static_cast<unsigned long long>(gridDim.x * kThreads) * a.chunk_bytes) {
        const unsigned long long rem = nb - off;
        const unsigned n = static_cast<unsigned>(rem < a.chunk_bytes ? rem : a.chunk_bytes) & ~15u;
        if (n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(base + off), "r"(n) : "memory");
      }
    }
    __syncthreads();
  }
}

__global__ void l2_pump_progress_kernel(int* progress, int value) {
  flashrt_pdl_wait_and_trigger();
  if (threadIdx.x == 0) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(progress), "r"(value) : "memory");
  }
}

}  // namespace

int l2_pump_launch(const L2PumpArgs& args, cudaStream_t stream, int nctas) {
  if (nctas < 1 || nctas > 8) return -1;
  if (args.nunits <= 0 || args.nunits > kL2PumpMaxUnits || args.total_units <= 0) return -1;
  if (args.unit_begin[args.nunits] > kL2PumpMaxChunks || (args.chunk_bytes & 15u) != 0 || args.chunk_bytes == 0) return -1;
  for (int c = 0; c < args.unit_begin[args.nunits]; ++c)
    if ((reinterpret_cast<uintptr_t>(args.ptr[c]) & 15) != 0) return -1;
  l2_pump_kernel<<<nctas, kThreads, 0, stream>>>(args);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int l2_pump_progress_store(int* progress, int value, cudaStream_t stream) {
  launch_maybe_pdl(l2_pump_progress_kernel, dim3(1), dim3(32), 0, stream, progress, value);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt
