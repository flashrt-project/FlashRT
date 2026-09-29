// FlashRT — shared elementwise activations (native, framework-free hot path).
//
//   silu_bf16     : out[i] = x[i] * sigmoid(x[i])           (in-place, bf16)
//   gelu_erf_bf16 : out[i] = 0.5*x[i]*(1+erf(x[i]/sqrt(2))) (in-place, bf16)
//
// 128-bit vectorized (8 bf16 per thread via uint4) to match the memory
// throughput of torch's elementwise kernels. Bit-exact with F.silu/F.gelu.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace flash_rt {
namespace kernels {

namespace {
constexpr int kThreads = 256;

__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_erf_f32(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// 8 bf16 (16 bytes) per uint4.
__device__ __forceinline__ uint4 load8(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint4*>(p);
}
__device__ __forceinline__ void store8(__nv_bfloat16* p, uint4 v) {
  *reinterpret_cast<uint4*>(p) = v;
}

template <float (*Fn)(float)>
__global__ void act_bf16_kernel(__nv_bfloat16* __restrict__ x, int n) {
  int idx = (blockIdx.x * kThreads + threadIdx.x) * 8;
  if (idx + 7 < n) {
    uint4 v = load8(x + idx);
    __nv_bfloat16* a = reinterpret_cast<__nv_bfloat16*>(&v);
    float r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = __bfloat162float(a[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = __float2bfloat16(Fn(r[i]));
    store8(x + idx, v);
  } else {
    for (int i = idx; i < n; ++i) {
      x[i] = __float2bfloat16(Fn(__bfloat162float(x[i])));
    }
  }
}
}  // namespace

void silu_bf16(__nv_bfloat16* x, int n, cudaStream_t stream) {
  int grid = (n + kThreads * 8 - 1) / (kThreads * 8);
  act_bf16_kernel<silu_f32><<<grid, kThreads, 0, stream>>>(x, n);
}

void gelu_erf_bf16(__nv_bfloat16* x, int n, cudaStream_t stream) {
  int grid = (n + kThreads * 8 - 1) / (kThreads * 8);
  act_bf16_kernel<gelu_erf_f32><<<grid, kThreads, 0, stream>>>(x, n);
}

}  // namespace kernels
}  // namespace flash_rt
