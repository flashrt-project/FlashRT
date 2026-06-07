#include <cstddef>

#include <hip/hip_runtime.h>

namespace {

__global__ void vector_add_f32_kernel(const float* a, const float* b, float* out,
                                      std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

}  // namespace

void launch_vector_add_f32(const float* a, const float* b, float* out,
                           std::size_t n, hipStream_t stream) {
  constexpr int kBlock = 256;
  const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  if (grid > 0) {
    hipLaunchKernelGGL(vector_add_f32_kernel, dim3(grid), dim3(kBlock), 0, stream,
                       a, b, out, n);
  }
}
