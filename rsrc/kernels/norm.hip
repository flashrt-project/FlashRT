#include <cmath>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBlock = 256;

template <typename T>
__device__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ float to_float<hip_bfloat16>(hip_bfloat16 v) {
  return static_cast<float>(v);
}

template <typename T>
__device__ T from_float(float v);

template <>
__device__ float from_float<float>(float v) {
  return v;
}

template <>
__device__ hip_bfloat16 from_float<hip_bfloat16>(float v) {
  return hip_bfloat16(v);
}

template <typename T>
__global__ void rms_norm_kernel(const T* x, const T* weight, T* out,
                                int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = to_float<T>(x[base + col]);
    sum += v * v;
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = to_float<T>(x[base + col]);
    const float w = to_float<T>(weight[col]);
    out[base + col] = from_float<T>(v * inv * (1.0f + w));
  }
}

__global__ void rms_norm_bf16_f32w_kernel(const hip_bfloat16* x,
                                          const float* weight,
                                          hip_bfloat16* out,
                                          int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    sum += v * v;
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    out[base + col] = hip_bfloat16(v * inv * (1.0f + weight[col]));
  }
}

__global__ void rms_norm_bf16_plain_kernel(const hip_bfloat16* x,
                                           const hip_bfloat16* weight,
                                           hip_bfloat16* out,
                                           int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    sum += v * v;
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(x[idx]);
    const float w = static_cast<float>(weight[col]);
    out[idx] = hip_bfloat16(v * inv * w);
  }
}

__global__ void rms_norm_fp8_kernel(const hip_bfloat16* x,
                                    const hip_bfloat16* weight,
                                    __hip_fp8_e4m3_fnuz* out,
                                    int rows, int hidden, float eps,
                                    const float* scale) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    sum += v * v;
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const float s = scale[0];
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(x[idx]);
    const float w = static_cast<float>(weight[col]);
    out[idx] = __hip_fp8_e4m3_fnuz((v * inv * (1.0f + w)) / s);
  }
}

__global__ void rms_norm_fp8_plain_kernel(const hip_bfloat16* x,
                                          const hip_bfloat16* weight,
                                          __hip_fp8_e4m3_fnuz* out,
                                          int rows, int hidden, float eps,
                                          const float* scale) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    sum += v * v;
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const float s = scale[0];
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(x[idx]);
    const float w = static_cast<float>(weight[col]);
    const hip_bfloat16 rounded = hip_bfloat16(v * inv * w);
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / s);
  }
}

__global__ void layer_norm_bf16_kernel(const hip_bfloat16* x,
                                       const hip_bfloat16* weight,
                                       const hip_bfloat16* bias,
                                       hip_bfloat16* out,
                                       int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += static_cast<float>(x[base + col]);
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = smem[0] / static_cast<float>(hidden);

  float var = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float diff = static_cast<float>(x[base + col]) - mean;
    var += diff * diff;
  }
  smem[threadIdx.x] = var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    const float w = static_cast<float>(weight[col]);
    const float b = static_cast<float>(bias[col]);
    out[base + col] = hip_bfloat16((v - mean) * inv * w + b);
  }
}

__global__ void layer_norm_fp8_kernel(const hip_bfloat16* x,
                                      const hip_bfloat16* weight,
                                      const hip_bfloat16* bias,
                                      __hip_fp8_e4m3_fnuz* out,
                                      const float* scale,
                                      int rows, int hidden, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    sum += static_cast<float>(x[base + col]);
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = smem[0] / static_cast<float>(hidden);

  float var = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float diff = static_cast<float>(x[base + col]) - mean;
    var += diff * diff;
  }
  smem[threadIdx.x] = var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const float s = scale[0];

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(x[idx]);
    const float w = static_cast<float>(weight[col]);
    const float b = static_cast<float>(bias[col]);
    const hip_bfloat16 rounded = hip_bfloat16((v - mean) * inv * w + b);
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / s);
  }
}

__global__ void add_bias_bf16_kernel(hip_bfloat16* x,
                                     const hip_bfloat16* bias,
                                     std::size_t n,
                                     int hidden) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const int col = static_cast<int>(idx % static_cast<std::size_t>(hidden));
  x[idx] = hip_bfloat16(static_cast<float>(x[idx]) +
                        static_cast<float>(bias[col]));
}

__global__ void bias_residual_bf16_kernel(hip_bfloat16* residual,
                                          const hip_bfloat16* x,
                                          const hip_bfloat16* bias,
                                          std::size_t n,
                                          int hidden) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const int col = static_cast<int>(idx % static_cast<std::size_t>(hidden));
  residual[idx] = hip_bfloat16(static_cast<float>(residual[idx]) +
                               static_cast<float>(x[idx]) +
                               static_cast<float>(bias[col]));
}

__global__ void residual_add_bf16_kernel(hip_bfloat16* residual,
                                         const hip_bfloat16* x,
                                         std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  residual[idx] = hip_bfloat16(static_cast<float>(residual[idx]) +
                               static_cast<float>(x[idx]));
}

__global__ void gate_mul_residual_bf16_kernel(hip_bfloat16* residual,
                                              const hip_bfloat16* x,
                                              const hip_bfloat16* gate,
                                              std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  residual[idx] = hip_bfloat16(static_cast<float>(residual[idx]) +
                               static_cast<float>(x[idx]) *
                               static_cast<float>(gate[idx]));
}

__global__ void residual_add_rms_norm_bf16_kernel(
    hip_bfloat16* residual,
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    hip_bfloat16* out,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const hip_bfloat16 updated = hip_bfloat16(
        static_cast<float>(residual[idx]) + static_cast<float>(x[idx]));
    residual[idx] = updated;
    sum += static_cast<float>(updated) * static_cast<float>(updated);
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(residual[idx]);
    const float w = static_cast<float>(weight[col]);
    out[idx] = hip_bfloat16(v * inv * (1.0f + w));
  }
}

__global__ void residual_add_rms_norm_bf16_plain_kernel(
    hip_bfloat16* residual,
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    hip_bfloat16* out,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const hip_bfloat16 updated = hip_bfloat16(
        static_cast<float>(residual[idx]) + static_cast<float>(x[idx]));
    residual[idx] = updated;
    const float v = static_cast<float>(updated);
    sum += v * v;
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(residual[idx]);
    const float w = static_cast<float>(weight[col]);
    out[idx] = hip_bfloat16(v * inv * w);
  }
}

__global__ void residual_add_rms_norm_fp8_kernel(
    hip_bfloat16* residual,
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    __hip_fp8_e4m3_fnuz* out,
    int rows,
    int hidden,
    float eps,
    const float* scale) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const hip_bfloat16 updated = hip_bfloat16(
        static_cast<float>(residual[idx]) + static_cast<float>(x[idx]));
    residual[idx] = updated;
    sum += static_cast<float>(updated) * static_cast<float>(updated);
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const float s = scale[0];
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(residual[idx]);
    const float w = static_cast<float>(weight[col]);
    const hip_bfloat16 rounded = hip_bfloat16(v * inv * (1.0f + w));
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / s);
  }
}

__global__ void residual_add_rms_norm_fp8_plain_kernel(
    hip_bfloat16* residual,
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    __hip_fp8_e4m3_fnuz* out,
    int rows,
    int hidden,
    float eps,
    const float* scale) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const hip_bfloat16 updated = hip_bfloat16(
        static_cast<float>(residual[idx]) + static_cast<float>(x[idx]));
    residual[idx] = updated;
    const float v = static_cast<float>(updated);
    sum += v * v;
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const float s = scale[0];
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(residual[idx]);
    const float w = static_cast<float>(weight[col]);
    const hip_bfloat16 rounded = hip_bfloat16(v * inv * w);
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / s);
  }
}

__global__ void ada_rms_norm_style_bf16_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    const hip_bfloat16* style,
    hip_bfloat16* out,
    hip_bfloat16* gate_out,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    sum += v * v;
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const int style_base = row * 3 * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    const float w = static_cast<float>(weight[col]);
    const float scale = static_cast<float>(style[style_base + col]);
    const float shift = static_cast<float>(style[style_base + hidden + col]);
    const hip_bfloat16 gate = style[style_base + 2 * hidden + col];
    out[base + col] = hip_bfloat16(v * inv * w * (1.0f + scale) + shift);
    gate_out[base + col] = gate;
  }
}

__global__ void ada_rms_norm_style_fp8_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    const hip_bfloat16* style,
    __hip_fp8_e4m3_fnuz* out,
    hip_bfloat16* gate_out,
    int rows,
    int hidden,
    float eps,
    const float* scale) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
    sum += v * v;
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const float s = scale[0];
  const int style_base = row * 3 * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(x[idx]);
    const float w = static_cast<float>(weight[col]);
    const float style_scale = static_cast<float>(style[style_base + col]);
    const float shift = static_cast<float>(style[style_base + hidden + col]);
    const float normed = v * inv * w * (1.0f + style_scale) + shift;
    const hip_bfloat16 rounded = hip_bfloat16(normed);
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / s);
    gate_out[idx] = style[style_base + 2 * hidden + col];
  }
}

__global__ void gate_residual_ada_norm_bf16_kernel(
    hip_bfloat16* residual,
    const hip_bfloat16* x,
    const hip_bfloat16* gate,
    const hip_bfloat16* weight,
    const hip_bfloat16* style,
    hip_bfloat16* out,
    hip_bfloat16* gate_out,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const hip_bfloat16 updated = hip_bfloat16(
        static_cast<float>(residual[idx]) +
        static_cast<float>(x[idx]) * static_cast<float>(gate[idx]));
    residual[idx] = updated;
    sum += static_cast<float>(updated) * static_cast<float>(updated);
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const int style_base = row * 3 * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(residual[idx]);
    const float w = static_cast<float>(weight[col]);
    const float scale = static_cast<float>(style[style_base + col]);
    const float shift = static_cast<float>(style[style_base + hidden + col]);
    out[idx] = hip_bfloat16(v * inv * w * (1.0f + scale) + shift);
    gate_out[idx] = style[style_base + 2 * hidden + col];
  }
}

__global__ void gate_residual_ada_norm_fp8_kernel(
    hip_bfloat16* residual,
    const hip_bfloat16* x,
    const hip_bfloat16* gate,
    const hip_bfloat16* weight,
    const hip_bfloat16* style,
    __hip_fp8_e4m3_fnuz* out,
    hip_bfloat16* gate_out,
    int rows,
    int hidden,
    float eps,
    const float* scale) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const hip_bfloat16 updated = hip_bfloat16(
        static_cast<float>(residual[idx]) +
        static_cast<float>(x[idx]) * static_cast<float>(gate[idx]));
    residual[idx] = updated;
    sum += static_cast<float>(updated) * static_cast<float>(updated);
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const float s = scale[0];
  const int style_base = row * 3 * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const float v = static_cast<float>(residual[idx]);
    const float w = static_cast<float>(weight[col]);
    const float style_scale = static_cast<float>(style[style_base + col]);
    const float shift = static_cast<float>(style[style_base + hidden + col]);
    const float normed = v * inv * w * (1.0f + style_scale) + shift;
    const hip_bfloat16 rounded = hip_bfloat16(normed);
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / s);
    gate_out[idx] = style[style_base + 2 * hidden + col];
  }
}

__global__ void bias_residual_layer_norm_bf16_kernel(
    hip_bfloat16* residual,
    const hip_bfloat16* x,
    const hip_bfloat16* bias_pre,
    const hip_bfloat16* norm_weight,
    const hip_bfloat16* norm_bias,
    hip_bfloat16* out,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const hip_bfloat16 updated = hip_bfloat16(
        static_cast<float>(residual[idx]) +
        static_cast<float>(x[idx]) +
        static_cast<float>(bias_pre[col]));
    residual[idx] = updated;
    sum += static_cast<float>(updated);
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = smem[0] / static_cast<float>(hidden);

  float var = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float diff = static_cast<float>(residual[base + col]) - mean;
    var += diff * diff;
  }
  smem[threadIdx.x] = var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(residual[base + col]);
    const float w = static_cast<float>(norm_weight[col]);
    const float b = static_cast<float>(norm_bias[col]);
    out[base + col] = hip_bfloat16((v - mean) * inv * w + b);
  }
}

__global__ void bias_residual_layer_norm_fp8_kernel(
    hip_bfloat16* residual,
    const hip_bfloat16* x,
    const hip_bfloat16* bias_pre,
    const hip_bfloat16* norm_weight,
    const hip_bfloat16* norm_bias,
    __hip_fp8_e4m3_fnuz* out,
    const float* scale,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = base + col;
    const hip_bfloat16 updated = hip_bfloat16(
        static_cast<float>(residual[idx]) +
        static_cast<float>(x[idx]) +
        static_cast<float>(bias_pre[col]));
    residual[idx] = updated;
    sum += static_cast<float>(updated);
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = smem[0] / static_cast<float>(hidden);

  float var = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float diff = static_cast<float>(residual[base + col]) - mean;
    var += diff * diff;
  }
  smem[threadIdx.x] = var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
  const float s = scale[0];

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(residual[base + col]);
    const float w = static_cast<float>(norm_weight[col]);
    const float b = static_cast<float>(norm_bias[col]);
    const hip_bfloat16 rounded = hip_bfloat16((v - mean) * inv * w + b);
    out[base + col] =
        __hip_fp8_e4m3_fnuz(static_cast<float>(rounded) / s);
  }
}

}  // namespace

void launch_rms_norm_f32(const float* x, const float* weight, float* out,
                         int rows, int hidden, float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(HIP_KERNEL_NAME(rms_norm_kernel<float>),
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     x, weight, out, rows, hidden, eps);
}

void launch_rms_norm_bf16(const void* x, const void* weight, void* out,
                          int rows, int hidden, float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(HIP_KERNEL_NAME(rms_norm_kernel<hip_bfloat16>),
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden, eps);
}

void launch_rms_norm_bf16_f32w(const void* x, const float* weight, void* out,
                               int rows, int hidden, float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(rms_norm_bf16_f32w_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     weight,
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden, eps);
}

void launch_rms_norm_bf16_plain(const void* x, const void* weight, void* out,
                                int rows, int hidden, float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(rms_norm_bf16_plain_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden, eps);
}

void launch_rms_norm_fp8_e4m3fnuz(
    const void* x, const void* weight, void* out,
    int rows, int hidden, float eps, const float* scale,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(rms_norm_fp8_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     rows, hidden, eps, scale);
}

void launch_rms_norm_fp8_e4m3fnuz_plain(
    const void* x, const void* weight, void* out,
    int rows, int hidden, float eps, const float* scale,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(rms_norm_fp8_plain_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     rows, hidden, eps, scale);
}

void launch_layer_norm_bf16(const void* x, const void* weight, const void* bias,
                            void* out, int rows, int hidden, float eps,
                            hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(layer_norm_bf16_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<const hip_bfloat16*>(bias),
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden, eps);
}

void launch_layer_norm_fp8_e4m3fnuz(
    const void* x, const void* weight, const void* bias, void* out,
    const float* scale, int rows, int hidden, float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(layer_norm_fp8_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<const hip_bfloat16*>(bias),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     rows, hidden, eps);
}

void launch_add_bias_bf16(void* x, const void* bias, int rows, int hidden,
                          hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  const std::size_t n = static_cast<std::size_t>(rows) *
                        static_cast<std::size_t>(hidden);
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(add_bias_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(bias),
                     n, hidden);
}

void launch_bias_residual_bf16(void* residual, const void* x, const void* bias,
                               int rows, int hidden, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  const std::size_t n = static_cast<std::size_t>(rows) *
                        static_cast<std::size_t>(hidden);
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(bias_residual_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(bias),
                     n, hidden);
}

void launch_residual_add_bf16(void* residual, const void* x,
                              std::size_t n, hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(residual_add_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     n);
}

void launch_gate_mul_residual_bf16(void* residual, const void* x,
                                   const void* gate, std::size_t n,
                                   hipStream_t stream) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
  hipLaunchKernelGGL(gate_mul_residual_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(gate),
                     n);
}

void launch_residual_add_rms_norm_bf16(
    void* residual,
    const void* x,
    const void* weight,
    void* out,
    int rows,
    int hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(residual_add_rms_norm_bf16_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden, eps);
}

void launch_residual_add_rms_norm_bf16_plain(
    void* residual,
    const void* x,
    const void* weight,
    void* out,
    int rows,
    int hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(residual_add_rms_norm_bf16_plain_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden, eps);
}

void launch_residual_add_rms_norm_fp8_e4m3fnuz(
    void* residual,
    const void* x,
    const void* weight,
    void* out,
    int rows,
    int hidden,
    float eps,
    const float* scale,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(residual_add_rms_norm_fp8_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     rows, hidden, eps, scale);
}

void launch_residual_add_rms_norm_fp8_e4m3fnuz_plain(
    void* residual,
    const void* x,
    const void* weight,
    void* out,
    int rows,
    int hidden,
    float eps,
    const float* scale,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(residual_add_rms_norm_fp8_plain_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     rows, hidden, eps, scale);
}

void launch_ada_rms_norm_style_bf16(
    const void* x,
    const void* weight,
    const void* style,
    void* out,
    void* gate_out,
    int rows,
    int hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(ada_rms_norm_style_bf16_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<const hip_bfloat16*>(style),
                     static_cast<hip_bfloat16*>(out),
                     static_cast<hip_bfloat16*>(gate_out),
                     rows, hidden, eps);
}

void launch_ada_rms_norm_style_fp8_e4m3fnuz(
    const void* x,
    const void* weight,
    const void* style,
    void* out,
    void* gate_out,
    int rows,
    int hidden,
    float eps,
    const float* scale,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(ada_rms_norm_style_fp8_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<const hip_bfloat16*>(style),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     static_cast<hip_bfloat16*>(gate_out),
                     rows, hidden, eps, scale);
}

void launch_gate_residual_ada_norm_bf16(
    void* residual,
    const void* x,
    const void* gate,
    const void* weight,
    const void* style,
    void* out,
    void* gate_out,
    int rows,
    int hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(gate_residual_ada_norm_bf16_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<const hip_bfloat16*>(style),
                     static_cast<hip_bfloat16*>(out),
                     static_cast<hip_bfloat16*>(gate_out),
                     rows, hidden, eps);
}

void launch_gate_residual_ada_norm_fp8_e4m3fnuz(
    void* residual,
    const void* x,
    const void* gate,
    const void* weight,
    const void* style,
    void* out,
    void* gate_out,
    int rows,
    int hidden,
    float eps,
    const float* scale,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(gate_residual_ada_norm_fp8_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<const hip_bfloat16*>(style),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     static_cast<hip_bfloat16*>(gate_out),
                     rows, hidden, eps, scale);
}

void launch_bias_residual_layer_norm_bf16(
    void* residual,
    const void* x,
    const void* bias_pre,
    const void* norm_weight,
    const void* norm_bias,
    void* out,
    int rows,
    int hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(bias_residual_layer_norm_bf16_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(bias_pre),
                     static_cast<const hip_bfloat16*>(norm_weight),
                     static_cast<const hip_bfloat16*>(norm_bias),
                     static_cast<hip_bfloat16*>(out),
                     rows, hidden, eps);
}

void launch_bias_residual_layer_norm_fp8_e4m3fnuz(
    void* residual,
    const void* x,
    const void* bias_pre,
    const void* norm_weight,
    const void* norm_bias,
    void* out,
    const float* scale,
    int rows,
    int hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(bias_residual_layer_norm_fp8_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<hip_bfloat16*>(residual),
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(bias_pre),
                     static_cast<const hip_bfloat16*>(norm_weight),
                     static_cast<const hip_bfloat16*>(norm_bias),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     rows, hidden, eps);
}
