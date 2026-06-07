#include <cstddef>
#include <cmath>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBlock = 256;

__global__ void amax_f32_kernel(const float* x, float* partial,
                                std::size_t n) {
  extern __shared__ float smem[];
  float local = 0.0f;
  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += blockDim.x * gridDim.x) {
    local = fmaxf(local, fabsf(x[idx]));
  }

  smem[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = smem[0];
  }
}

__global__ void amax_bf16_kernel(const hip_bfloat16* x, float* partial,
                                 std::size_t n) {
  extern __shared__ float smem[];
  float local = 0.0f;
  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < n;
       idx += blockDim.x * gridDim.x) {
    local = fmaxf(local, fabsf(static_cast<float>(x[idx])));
  }

  smem[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = smem[0];
  }
}

__global__ void scale_from_partials_kernel(const float* partial, float* scale,
                                           int partial_count) {
  extern __shared__ float smem[];
  float local = 0.0f;
  for (int idx = threadIdx.x; idx < partial_count; idx += blockDim.x) {
    local = fmaxf(local, partial[idx]);
  }

  smem[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scale[0] = fmaxf(smem[0] / 240.0f, 1.0e-8f);
  }
}

__global__ void quantize_f32_to_fp8_e4m3fnuz_kernel(const float* x,
                                                    const float* scale,
                                                    __hip_fp8_e4m3_fnuz* out,
                                                    std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = __hip_fp8_e4m3_fnuz(x[idx] / scale[0]);
}

__global__ void quantize_bf16_to_fp8_e4m3fnuz_kernel(const hip_bfloat16* x,
                                                     const float* scale,
                                                     __hip_fp8_e4m3_fnuz* out,
                                                     std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(x[idx]) / scale[0]);
}

}  // namespace

void launch_quantize_f32_to_fp8_e4m3fnuz(const float* x, const float* scale,
                                         void* out, std::size_t n,
                                         hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(quantize_f32_to_fp8_e4m3fnuz_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     x, scale,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     n);
}

void launch_quantize_bf16_to_fp8_e4m3fnuz(const void* x, const float* scale,
                                          void* out, std::size_t n,
                                          hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(quantize_bf16_to_fp8_e4m3fnuz_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     scale,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     n);
}

void launch_dynamic_quantize_f32_to_fp8_e4m3fnuz(
    const float* x, void* out, float* scale, float* partial,
    int partial_count, std::size_t n, hipStream_t stream) {
  if (n == 0) {
    return;
  }
  hipLaunchKernelGGL(amax_f32_kernel,
                     dim3(partial_count), dim3(kBlock), kBlock * sizeof(float),
                     stream,
                     x, partial, n);
  hipLaunchKernelGGL(scale_from_partials_kernel,
                     dim3(1), dim3(kBlock), kBlock * sizeof(float),
                     stream,
                     partial, scale, partial_count);
  launch_quantize_f32_to_fp8_e4m3fnuz(x, scale, out, n, stream);
}

void launch_dynamic_quantize_bf16_to_fp8_e4m3fnuz(
    const void* x, void* out, float* scale, float* partial,
    int partial_count, std::size_t n, hipStream_t stream) {
  if (n == 0) {
    return;
  }
  hipLaunchKernelGGL(amax_bf16_kernel,
                     dim3(partial_count), dim3(kBlock), kBlock * sizeof(float),
                     stream,
                     static_cast<const hip_bfloat16*>(x), partial, n);
  hipLaunchKernelGGL(scale_from_partials_kernel,
                     dim3(1), dim3(kBlock), kBlock * sizeof(float),
                     stream,
                     partial, scale, partial_count);
  launch_quantize_bf16_to_fp8_e4m3fnuz(x, scale, out, n, stream);
}
