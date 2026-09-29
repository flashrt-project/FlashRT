// FlashRT — Hy-VLA denoise Euler update: x_fp32 += dt * (float)v_bf16,
// optionally emitting a bf16 copy of the updated state (so the action-MLP
// input cast is done by this kernel instead of a framework launch).
// With v == nullptr and dt == 0 the kernel only (re)writes the bf16 copy.

#include "kernels/hyvla_euler.cuh"
#include <cuda_bf16.h>

namespace {

__global__ void hyvla_euler_update_kernel(float* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ v,
                                          __nv_bfloat16* __restrict__ x_bf16,
                                          float dt, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xv = x[i];
    if (v != nullptr) {
      xv = fmaf(dt, __bfloat162float(v[i]), xv);
      x[i] = xv;
    }
    if (x_bf16 != nullptr) {
      x_bf16[i] = __float2bfloat16(xv);
    }
  }
}

}  // namespace

extern "C" void hyvla_euler_update_bf16_fp32(
    void* x, const void* v, void* x_bf16, float dt, long n,
    cudaStream_t stream) {
  if (n <= 0) return;
  const int threads = 256;
  const long blocks = (n + threads - 1) / threads;
  hyvla_euler_update_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                              stream>>>(
      reinterpret_cast<float*>(x),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(x_bf16), dt, n);
}
