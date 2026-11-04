// SPDX-License-Identifier: Apache-2.0
// Generic affine/rounding profiles absent from the existing Pi0.5 norm ABI.
#include "common_hip_rdna.h"
#include "transformer_ops_rdna.h"
#include <stdexcept>
namespace {
__device__ float rounded(float x) {
  return __bfloat162float(__float2bfloat16(x));
}
__global__ void norm_affine(const __hip_bfloat16 *x, const __hip_bfloat16 *w,
                            const __hip_bfloat16 *bias,
                            const __hip_bfloat16 *residual, __hip_bfloat16 *out,
                            __hip_bfloat16 *sum, int width, float eps,
                            bool center, float offset, bool round_affine) {
  __shared__ float scratch[8];
  const int row = blockIdx.x, tid = threadIdx.x;
  float total = 0;
  for (int c = tid; c < width; c += 256) {
    float a = __bfloat162float(x[row * width + c]);
    if (residual) {
      a = rounded(a + __bfloat162float(residual[row * width + c]));
    }
    total += a;
  }
  const float mean = center ? rdna_block_reduce_sum(total, scratch) / width : 0;
  __syncthreads();
  float squares = 0;
  for (int c = tid; c < width; c += 256) {
    float a = __bfloat162float(x[row * width + c]);
    if (residual)
      a = rounded(a + __bfloat162float(residual[row * width + c]));
    a -= mean;
    squares += a * a;
  }
  const float inv =
      rsqrtf(rdna_block_reduce_sum(squares, scratch) / width + eps);
  for (int c = tid; c < width; c += 256) {
    float a = __bfloat162float(x[row * width + c]);
    if (residual)
      a = rounded(a + __bfloat162float(residual[row * width + c]));
    if (residual)
      sum[row * width + c] = __float2bfloat16(a);
    float y = (a - mean) * inv;
    if (round_affine)
      y = rounded(y);
    if (w) {
      float scale = __bfloat162float(w[c]) + offset;
      if (round_affine)
        scale = rounded(scale);
      y *= scale;
      if (round_affine)
        y = rounded(y);
    }
    if (bias)
      y += __bfloat162float(bias[c]);
    out[row * width + c] = __float2bfloat16(y);
  }
}
template <class T, class U>
__global__ void rotate_half(const __hip_bfloat16 *x, const T *cos, const U *sin,
                            __hip_bfloat16 *out, int64_t size, int heads,
                            int dim, int64_t xs, int64_t xh, int64_t cs,
                            int64_t ss) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
  if (i >= size)
    return;
  int c = i % dim, h = i / dim % heads;
  int64_t row = i / (dim * heads);
  int64_t base = row * xs + h * xh;
  float a = __bfloat162float(x[base + c]),
        b = __bfloat162float(x[base + (c + dim / 2) % dim]);
  if (c < dim / 2)
    b = -b;
  out[i] = __float2bfloat16(a * static_cast<float>(cos[row * cs + c]) +
                            b * static_cast<float>(sin[row * ss + c]));
}
} // namespace
void normalize_bf16_rdna(const __hip_bfloat16 *x, const __hip_bfloat16 *w,
                         const __hip_bfloat16 *bias,
                         const __hip_bfloat16 *residual, __hip_bfloat16 *out,
                         __hip_bfloat16 *sum, int rows, int width, float eps,
                         bool center, float offset, bool round_affine,
                         hipStream_t stream) {
  if (rows < 1 || width < 1 || eps <= 0)
    throw std::invalid_argument("invalid RDNA normalization geometry");
  norm_affine<<<rows, 256, 0, stream>>>(x, w, bias, residual, out, sum, width,
                                        eps, center, offset, round_affine);
}
void rope_bf16_rdna(const __hip_bfloat16 *x, const void *cos, const void *sin,
                    __hip_bfloat16 *out, int64_t size, int heads, int dim,
                    int64_t xs, int64_t xh, int64_t cs, int64_t ss,
                    bool cos_fp32, bool sin_fp32, hipStream_t stream) {
  if (size < 1 || heads < 1 || dim < 1 || dim % 2)
    throw std::invalid_argument("invalid RDNA RoPE geometry");
  dim3 grid((size + 255) / 256);
#define ROTATE(T, U)                                                           \
  rotate_half<T, U><<<grid, 256, 0, stream>>>(                                 \
      x, static_cast<const T *>(cos), static_cast<const U *>(sin), out, size,  \
      heads, dim, xs, xh, cs, ss)
  if (cos_fp32 && sin_fp32) {
    ROTATE(float, float);
  } else if (cos_fp32) {
    ROTATE(float, __hip_bfloat16);
  } else if (sin_fp32) {
    ROTATE(__hip_bfloat16, float);
  } else {
    ROTATE(__hip_bfloat16, __hip_bfloat16);
  }
#undef ROTATE
}
