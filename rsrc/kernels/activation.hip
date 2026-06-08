#include <cmath>
#include <cstddef>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBlock = 256;
constexpr float kGeluTanhA = 0.7978845608028654f;
constexpr float kGeluTanhB = 0.044715f;

__device__ float gelu_tanh(float x) {
  const float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(kGeluTanhA * (x + kGeluTanhB * x3)));
}

__global__ void gelu_tanh_mul_bf16_kernel(const hip_bfloat16* gate,
                                          const hip_bfloat16* up,
                                          hip_bfloat16* out,
                                          std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float g = static_cast<float>(gate[idx]);
  const float u = static_cast<float>(up[idx]);
  out[idx] = hip_bfloat16(gelu_tanh(g) * u);
}

__global__ void gelu_tanh_mul_quantize_fp8_e4m3fnuz_kernel(
    const hip_bfloat16* gate,
    const hip_bfloat16* up,
    const float* scale,
    __hip_fp8_e4m3_fnuz* out,
    std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float g = static_cast<float>(gate[idx]);
  const float u = static_cast<float>(up[idx]);
  out[idx] = __hip_fp8_e4m3_fnuz((gelu_tanh(g) * u) / scale[0]);
}

__global__ void gelu_tanh_merged_bf16_kernel(
    const hip_bfloat16* gate_up,
    hip_bfloat16* out,
    int rows,
    int hidden) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t n = static_cast<std::size_t>(rows) * hidden;
  if (idx >= n) {
    return;
  }
  const int col = static_cast<int>(idx % hidden);
  const int row = static_cast<int>(idx / hidden);
  const std::size_t base = static_cast<std::size_t>(row) * 2 * hidden;
  const float g = static_cast<float>(gate_up[base + col]);
  const float u = static_cast<float>(gate_up[base + hidden + col]);
  out[idx] = hip_bfloat16(gelu_tanh(g) * u);
}

__global__ void gelu_tanh_merged_quantize_fp8_e4m3fnuz_kernel(
    const hip_bfloat16* gate_up,
    const float* scale,
    __hip_fp8_e4m3_fnuz* out,
    int rows,
    int hidden) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t n = static_cast<std::size_t>(rows) * hidden;
  if (idx >= n) {
    return;
  }
  const int col = static_cast<int>(idx % hidden);
  const int row = static_cast<int>(idx / hidden);
  const std::size_t base = static_cast<std::size_t>(row) * 2 * hidden;
  const float g = static_cast<float>(gate_up[base + col]);
  const float u = static_cast<float>(gate_up[base + hidden + col]);
  out[idx] = __hip_fp8_e4m3_fnuz((gelu_tanh(g) * u) / scale[0]);
}

__global__ void gelu_tanh_bf16_kernel(const hip_bfloat16* x,
                                      hip_bfloat16* out,
                                      std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = hip_bfloat16(gelu_tanh(static_cast<float>(x[idx])));
}

__global__ void gelu_tanh_quantize_fp8_e4m3fnuz_kernel(
    const hip_bfloat16* x,
    const float* scale,
    __hip_fp8_e4m3_fnuz* out,
    std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const hip_bfloat16 rounded =
      hip_bfloat16(gelu_tanh(static_cast<float>(x[idx])));
  out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / scale[0]);
}

__global__ void silu_bf16_kernel(const hip_bfloat16* x,
                                 hip_bfloat16* out,
                                 std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const float v = static_cast<float>(x[idx]);
  out[idx] = hip_bfloat16(v / (1.0f + expf(-v)));
}

__global__ void silu_mul_merged_bf16_kernel(
    const hip_bfloat16* gate_up,
    hip_bfloat16* out,
    int rows,
    int hidden) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t n = static_cast<std::size_t>(rows) * hidden;
  if (idx >= n) {
    return;
  }
  const int col = static_cast<int>(idx % hidden);
  const int row = static_cast<int>(idx / hidden);
  const std::size_t base = static_cast<std::size_t>(row) * 2 * hidden;
  const float g = static_cast<float>(gate_up[base + col]);
  const float u = static_cast<float>(gate_up[base + hidden + col]);
  out[idx] = hip_bfloat16((g / (1.0f + expf(-g))) * u);
}

__global__ void silu_mul_merged_quantize_fp8_e4m3fnuz_kernel(
    const hip_bfloat16* gate_up,
    const float* scale,
    __hip_fp8_e4m3_fnuz* out,
    int rows,
    int hidden) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t n = static_cast<std::size_t>(rows) * hidden;
  if (idx >= n) {
    return;
  }
  const int col = static_cast<int>(idx % hidden);
  const int row = static_cast<int>(idx / hidden);
  const std::size_t base = static_cast<std::size_t>(row) * 2 * hidden;
  const float g = static_cast<float>(gate_up[base + col]);
  const float u = static_cast<float>(gate_up[base + hidden + col]);
  const hip_bfloat16 rounded = hip_bfloat16((g / (1.0f + expf(-g))) * u);
  out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / scale[0]);
}

}  // namespace

void launch_gelu_tanh_mul_bf16(const void* gate, const void* up, void* out,
                               std::size_t n, hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(gelu_tanh_mul_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<const hip_bfloat16*>(up),
                     static_cast<hip_bfloat16*>(out),
                     n);
}

void launch_gelu_tanh_mul_quantize_fp8_e4m3fnuz(
    const void* gate, const void* up, const float* scale, void* out,
    std::size_t n, hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(gelu_tanh_mul_quantize_fp8_e4m3fnuz_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<const hip_bfloat16*>(up),
                     scale,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     n);
}

void launch_gelu_tanh_merged_bf16(
    const void* gate_up, void* out, int rows, int hidden, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  const std::size_t n = static_cast<std::size_t>(rows) * hidden;
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(gelu_tanh_merged_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate_up),
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden);
}

void launch_gelu_tanh_merged_quantize_fp8_e4m3fnuz(
    const void* gate_up, const float* scale, void* out, int rows, int hidden,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  const std::size_t n = static_cast<std::size_t>(rows) * hidden;
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(gelu_tanh_merged_quantize_fp8_e4m3fnuz_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate_up),
                     scale,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     rows, hidden);
}

void launch_gelu_tanh_quantize_fp8_e4m3fnuz(
    const void* x, const float* scale, void* out, std::size_t n,
    hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(gelu_tanh_quantize_fp8_e4m3fnuz_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     scale,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     n);
}

void launch_silu_bf16(const void* x, void* out, std::size_t n,
                      hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(silu_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<hip_bfloat16*>(out),
                     n);
}

void launch_silu_mul_merged_bf16(
    const void* gate_up, void* out, int rows, int hidden, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  const std::size_t n = static_cast<std::size_t>(rows) * hidden;
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(silu_mul_merged_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate_up),
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden);
}

void launch_silu_mul_merged_quantize_fp8_e4m3fnuz(
    const void* gate_up, const float* scale, void* out, int rows, int hidden,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  const std::size_t n = static_cast<std::size_t>(rows) * hidden;
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(silu_mul_merged_quantize_fp8_e4m3fnuz_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate_up),
                     scale,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     rows, hidden);
}

void launch_gelu_tanh_bf16(const void* x, void* out, std::size_t n,
                           hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(gelu_tanh_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<hip_bfloat16*>(out),
                     n);
}
