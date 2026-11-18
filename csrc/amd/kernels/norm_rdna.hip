// ================================================================
// FlashRT AMD -- BF16 normalization kernels (RDNA, wave32)
//
// Covers the normalization surface used by the RDNA Pi0.5 pipeline:
//   LayerNorm, RMSNorm, AdaRMSNorm, residual+RMSNorm, and
//   gated-residual+AdaRMSNorm.
//
// One workgroup owns one row. Threads accumulate FP32 statistics, reduce
// first within each 32-lane wave and then through one LDS value per wave, and
// finally write BF16 output. Fused residual kernels deliberately round the
// residual sum to BF16 before using it for RMS statistics; this matches the
// model's materialized BF16 residual stream and keeps fused/unfused parity.
//
// Pi0.5's hot widths have register-resident specializations:
//   width 1152: LayerNorm, 64 threads, 9 BF16 pairs/thread
//   width 2048: RMSNorm, 64 threads, 16 BF16 pairs/thread
//   width 2048: residual+RMSNorm, 256 threads, 4 BF16 pairs/thread
// Generic 256-thread kernels remain available for all other widths. No
// process-global tuning state or temporary global workspace is required.
// ================================================================

#include "common_hip_rdna.h"
#include <algorithm>

namespace {

constexpr int kNormThreads = 256;
constexpr int kMaxWaves = kNormThreads / FLASHRT_RDNA_WAVE_SIZE;

// ── Fixed-width register-resident kernels ──
//
// The encoder and vision widths are compile-time constants in Pi0.5.  Keep a
// row resident in registers across the reduction instead of loading it again
// for the normalization pass.  Fixed wave32 launch shapes retain useful
// occupancy on gfx1151 without paying for the generic 256-thread path.
template <int Width, int Threads>
__global__ __launch_bounds__(Threads) void layer_norm_wide_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const __hip_bfloat16 *__restrict__ weight,
    const __hip_bfloat16 *__restrict__ bias,
    __hip_bfloat16 *__restrict__ output, float epsilon) {
  static_assert(Width % (2 * Threads) == 0);
  constexpr int kPairsPerThread = Width / (2 * Threads);
  constexpr int kWaves = Threads / FLASHRT_RDNA_WAVE_SIZE;
  __shared__ float wave_sums[kWaves];
  __shared__ float wave_square_sums[kWaves];

  const int row_pair_start = blockIdx.x * (Width / 2);
  const auto *input2 = reinterpret_cast<const __hip_bfloat162 *>(input);
  auto *output2 = reinterpret_cast<__hip_bfloat162 *>(output);
  __hip_bfloat162 values[kPairsPerThread];
  float local_sum = 0.0f;

#pragma unroll
  for (int item = 0; item < kPairsPerThread; ++item) {
    const int pair = row_pair_start + threadIdx.x + item * Threads;
    const __hip_bfloat162 packed = input2[pair];
    const float first = __bfloat162float(packed.x);
    const float second = __bfloat162float(packed.y);
    values[item] = packed;
    local_sum += first + second;
  }

  const float mean = rdna_block_reduce_sum(local_sum, wave_sums) / Width;
  float local_variance = 0.0f;
#pragma unroll
  for (int item = 0; item < kPairsPerThread; ++item) {
    const float first = __bfloat162float(values[item].x) - mean;
    const float second = __bfloat162float(values[item].y) - mean;
    local_variance = fmaf(first, first, local_variance);
    local_variance = fmaf(second, second, local_variance);
  }
  const float variance =
      rdna_block_reduce_sum(local_variance, wave_square_sums) / Width;
  const float inverse_std = rsqrtf(variance + epsilon);
  const auto *weight2 = reinterpret_cast<const __hip_bfloat162 *>(weight);
  const auto *bias2 = reinterpret_cast<const __hip_bfloat162 *>(bias);

#pragma unroll
  for (int item = 0; item < kPairsPerThread; ++item) {
    const int column_pair = threadIdx.x + item * Threads;
    const int pair = row_pair_start + column_pair;
    const __hip_bfloat162 packed_weight = weight2[column_pair];
    const __hip_bfloat162 packed_bias = bias2[column_pair];
    const float first =
        (__bfloat162float(values[item].x) - mean) * inverse_std *
            __bfloat162float(packed_weight.x) +
        __bfloat162float(packed_bias.x);
    const float second =
        (__bfloat162float(values[item].y) - mean) * inverse_std *
            __bfloat162float(packed_weight.y) +
        __bfloat162float(packed_bias.y);
    output2[pair] =
        __hip_bfloat162(__float2bfloat16(first), __float2bfloat16(second));
  }
}

// Fixed-width RMSNorm keeps the input row in registers across its reduction,
// avoiding a global-memory reread without changing the observable arithmetic
// order.
template <int Width, int Threads>
__global__ __launch_bounds__(Threads) void rms_norm_wide_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    __hip_bfloat16 *__restrict__ output, float epsilon) {
  static_assert(Width % (2 * Threads) == 0);
  constexpr int kPairsPerThread = Width / (2 * Threads);
  constexpr int kWaves = Threads / FLASHRT_RDNA_WAVE_SIZE;
  __shared__ float wave_sums[kWaves];

  const int row_pair_start = blockIdx.x * (Width / 2);
  const auto *input2 = reinterpret_cast<const __hip_bfloat162 *>(input);
  auto *output2 = reinterpret_cast<__hip_bfloat162 *>(output);
  __hip_bfloat162 values[kPairsPerThread];
  float local_square_sum = 0.0f;

#pragma unroll
  for (int item = 0; item < kPairsPerThread; ++item) {
    const int pair = row_pair_start + threadIdx.x + item * Threads;
    const __hip_bfloat162 packed = input2[pair];
    const float first = __bfloat162float(packed.x);
    const float second = __bfloat162float(packed.y);
    values[item] = packed;
    local_square_sum = fmaf(first, first, local_square_sum);
    local_square_sum = fmaf(second, second, local_square_sum);
  }

  const float inverse_rms = rsqrtf(
      rdna_block_reduce_sum(local_square_sum, wave_sums) / Width + epsilon);
#pragma unroll
  for (int item = 0; item < kPairsPerThread; ++item) {
    const int pair = row_pair_start + threadIdx.x + item * Threads;
    output2[pair] = __hip_bfloat162(
        __float2bfloat16(__bfloat162float(values[item].x) * inverse_rms),
        __float2bfloat16(__bfloat162float(values[item].y) * inverse_rms));
  }
}

template <int Width, int Threads>
__global__ __launch_bounds__(Threads) void residual_rms_wide_rdna_kernel(
    const __hip_bfloat16 *__restrict__ update,
    const __hip_bfloat16 *__restrict__ residual,
    __hip_bfloat16 *__restrict__ output_sum,
    __hip_bfloat16 *__restrict__ output_norm, float epsilon) {
  static_assert(Width % (2 * Threads) == 0);
  constexpr int kPairsPerThread = Width / (2 * Threads);
  constexpr int kWaves = Threads / FLASHRT_RDNA_WAVE_SIZE;
  __shared__ float wave_sums[kWaves];

  const int row_pair_start = blockIdx.x * (Width / 2);
  const auto *update2 = reinterpret_cast<const __hip_bfloat162 *>(update);
  const auto *residual2 = reinterpret_cast<const __hip_bfloat162 *>(residual);
  auto *sum2 = reinterpret_cast<__hip_bfloat162 *>(output_sum);
  auto *norm2 = reinterpret_cast<__hip_bfloat162 *>(output_norm);
  __hip_bfloat162 values[kPairsPerThread];
  float local_square_sum = 0.0f;

#pragma unroll
  for (int item = 0; item < kPairsPerThread; ++item) {
    const int pair = row_pair_start + threadIdx.x + item * Threads;
    const __hip_bfloat162 update_value = update2[pair];
    const __hip_bfloat162 residual_value = residual2[pair];
    // Match the model's BF16 residual stream: round before RMS statistics.
    const __hip_bfloat162 summed(
        __float2bfloat16(__bfloat162float(update_value.x) +
                         __bfloat162float(residual_value.x)),
        __float2bfloat16(__bfloat162float(update_value.y) +
                         __bfloat162float(residual_value.y)));
    sum2[pair] = summed;
    values[item] = summed;
    const float first = __bfloat162float(summed.x);
    const float second = __bfloat162float(summed.y);
    local_square_sum = fmaf(first, first, local_square_sum);
    local_square_sum = fmaf(second, second, local_square_sum);
  }

  const float inverse_rms = rsqrtf(
      rdna_block_reduce_sum(local_square_sum, wave_sums) / Width + epsilon);
#pragma unroll
  for (int item = 0; item < kPairsPerThread; ++item) {
    const int pair = row_pair_start + threadIdx.x + item * Threads;
    norm2[pair] = __hip_bfloat162(
        __float2bfloat16(__bfloat162float(values[item].x) * inverse_rms),
        __float2bfloat16(__bfloat162float(values[item].y) * inverse_rms));
  }
}

// ── Generic row-per-workgroup kernels ──
// These variants trade the fixed-width register cache for a grid-stride row
// traversal and support widths outside the two production specializations.

__global__ __launch_bounds__(kNormThreads) void layer_norm_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const __hip_bfloat16 *__restrict__ weight,
    const __hip_bfloat16 *__restrict__ bias,
    __hip_bfloat16 *__restrict__ output, int width, float epsilon) {
  // Block reductions broadcast without a trailing barrier; keep the mean and
  // variance scratch separate, as in the fixed-width kernel above.
  __shared__ float wave_sums[kMaxWaves];
  __shared__ float wave_square_sums[kMaxWaves];
  const int row = blockIdx.x;
  const int row_start = row * width;

  float local_sum = 0.0f;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    local_sum += __bfloat162float(input[row_start + column]);
  }
  const float mean = rdna_block_reduce_sum(local_sum, wave_sums) / width;

  float local_variance = 0.0f;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const float centered = __bfloat162float(input[row_start + column]) - mean;
    local_variance += centered * centered;
  }
  const float variance =
      rdna_block_reduce_sum(local_variance, wave_square_sums) / width;
  const float inverse_std = rsqrtf(variance + epsilon);

  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value =
        (__bfloat162float(input[row_start + column]) - mean) * inverse_std;
    output[row_start + column] =
        __float2bfloat16(value * __bfloat162float(weight[column]) +
                         __bfloat162float(bias[column]));
  }
}

__global__ __launch_bounds__(kNormThreads) void rms_norm_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    __hip_bfloat16 *__restrict__ output, int width, float epsilon) {
  __shared__ float wave_sums[kMaxWaves];
  const int row_start = blockIdx.x * width;
  float local_square_sum = 0.0f;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(input[row_start + column]);
    local_square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(
      rdna_block_reduce_sum(local_square_sum, wave_sums) / width + epsilon);
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    output[row_start + column] = __float2bfloat16(
        __bfloat162float(input[row_start + column]) * inverse_rms);
  }
}

// AdaRMS modulation layout is [scale | shift], each `width` elements. The
// normalized value is transformed as x * (1 + scale) + shift.
__global__ __launch_bounds__(kNormThreads) void adarms_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const __hip_bfloat16 *__restrict__ modulation,
    __hip_bfloat16 *__restrict__ output, int width, float epsilon) {
  __shared__ float wave_sums[kMaxWaves];
  const int row_start = blockIdx.x * width;
  float local_square_sum = 0.0f;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(input[row_start + column]);
    local_square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(
      rdna_block_reduce_sum(local_square_sum, wave_sums) / width + epsilon);
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const float normalized =
        __bfloat162float(input[row_start + column]) * inverse_rms;
    const float scale = __bfloat162float(modulation[column]);
    const float shift = __bfloat162float(modulation[width + column]);
    output[row_start + column] =
        __float2bfloat16(normalized * (1.0f + scale) + shift);
  }
}

// Residual is stored before the second pass so normalization observes exactly
// the same rounded BF16 value returned through output_sum.
__global__ __launch_bounds__(kNormThreads) void residual_rms_rdna_kernel(
    const __hip_bfloat16 *__restrict__ update,
    const __hip_bfloat16 *__restrict__ residual,
    __hip_bfloat16 *__restrict__ output_sum,
    __hip_bfloat16 *__restrict__ output_norm, int width, float epsilon) {
  __shared__ float wave_sums[kMaxWaves];
  const int row_start = blockIdx.x * width;
  float local_square_sum = 0.0f;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const int index = row_start + column;
    const __hip_bfloat16 summed = __float2bfloat16(
        __bfloat162float(residual[index]) + __bfloat162float(update[index]));
    output_sum[index] = summed;
    const float value = __bfloat162float(summed);
    local_square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(
      rdna_block_reduce_sum(local_square_sum, wave_sums) / width + epsilon);
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const int index = row_start + column;
    output_norm[index] =
        __float2bfloat16(__bfloat162float(output_sum[index]) * inverse_rms);
  }
}

// Fuses gated residual update and AdaRMSNorm. Gate is shared across rows;
// modulation uses the same [scale | shift] layout as adarms_rdna_kernel.
__global__ __launch_bounds__(kNormThreads) void residual_adarms_rdna_kernel(
    const __hip_bfloat16 *__restrict__ update,
    const __hip_bfloat16 *__restrict__ residual,
    const __hip_bfloat16 *__restrict__ gate,
    const __hip_bfloat16 *__restrict__ modulation,
    __hip_bfloat16 *__restrict__ output_sum,
    __hip_bfloat16 *__restrict__ output_norm, int width, float epsilon) {
  __shared__ float wave_sums[kMaxWaves];
  const int row_start = blockIdx.x * width;
  float local_square_sum = 0.0f;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const int index = row_start + column;
    const __hip_bfloat16 summed = __float2bfloat16(
        __bfloat162float(residual[index]) +
        __bfloat162float(update[index]) * __bfloat162float(gate[column]));
    output_sum[index] = summed;
    const float value = __bfloat162float(summed);
    local_square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(
      rdna_block_reduce_sum(local_square_sum, wave_sums) / width + epsilon);
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const int index = row_start + column;
    const float normalized = __bfloat162float(output_sum[index]) * inverse_rms;
    const float scale = __bfloat162float(modulation[column]);
    const float shift = __bfloat162float(modulation[width + column]);
    output_norm[index] = __float2bfloat16(normalized * (1.0f + scale) + shift);
  }
}

// ── Contiguous FP32 mean-square reduction ──

__device__ __forceinline__ float rms_mean_rdna(const __hip_bfloat16 *input,
                                               int width, float *partial) {
  const int lane = threadIdx.x;
  float sums[4] = {};
  if (width >= 128) {
    for (int index = lane * 4; index < width; index += blockDim.x * 4) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const float x =
            index + item < width ? __bfloat162float(input[index + item]) : 0.0f;
        sums[item] += x * x;
      }
    }
  } else {
    for (int index = lane; index < width; index += blockDim.x * 4) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int column = index + item * blockDim.x;
        const float x = column < width ? __bfloat162float(input[column]) : 0.0f;
        sums[item] += x * x;
      }
    }
  }
  float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
  if (blockDim.x > 32) {
    partial[lane] = sum;
    for (int offset = blockDim.x / 2; offset >= 32; offset /= 2) {
      __syncthreads();
      if (lane < offset) {
        sum += partial[lane + offset];
        partial[lane] = sum;
      }
    }
  }
  if (lane < 32) {
    for (int offset = 1; offset < min(static_cast<int>(blockDim.x), 32);
         offset *= 2)
      sum += __shfl_down(sum, offset, 32);
  }
  if (blockDim.x <= 32)
    return __shfl(sum, 0, 32) * (1.0f / width);
  if (lane == 0)
    partial[0] = sum * (1.0f / width);
  __syncthreads();
  return partial[0];
}

template <bool Adaptive>
__global__ void rms_affine_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const __hip_bfloat16 *__restrict__ weight, const float *__restrict__ gamma,
    const float *__restrict__ beta, __hip_bfloat16 *__restrict__ output,
    int width, float epsilon) {
  __shared__ float partial[512];
  input += static_cast<size_t>(blockIdx.x) * width;
  output += static_cast<size_t>(blockIdx.x) * width;
  // Round the accurate inverse to FP32; the hardware estimate can cross BF16
  // midpoints.
  const float inverse = static_cast<float>(rsqrt(
      static_cast<double>(rms_mean_rdna(input, width, partial) + epsilon)));
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    float value = __bfloat162float(input[column]) * inverse;
    if constexpr (!Adaptive)
      value = __bfloat162float(__float2bfloat16(value));
    value *= __bfloat162float(weight[column]);
    if constexpr (Adaptive)
      value = rdna_mul_f32(1.0f + gamma[column], value) + beta[column];
    output[column] = __float2bfloat16(value);
  }
}

// ATen chooses the reduction width from both row count and vectorized width.
int rms_threads_rdna(int rows, int width) {
  const int vectors = width >= 128 ? width / 4 : width;
  int x = 1, y = 1;
  while (x * 2 <= std::min(vectors, 512))
    x *= 2;
  while (y * 2 <= std::min(rows, 512))
    y *= 2;
  const int block_y = std::min(y, 512 / std::min(x, 32));
  return std::min(x, 512 / block_y);
}

// ── Head RMSNorm, FP32 RoPE and attention layout ──

__device__ __forceinline__ void
norm_rope_head_rdna(const __hip_bfloat16 *input, const __hip_bfloat16 *weight,
                    const float *cosine, const float *sine, float epsilon,
                    float *rotated) {
  const int lane = threadIdx.x;
  float values[4], sums[4];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    values[item] = __bfloat162float(input[lane * 4 + item]);
    sums[item] = values[item] * values[item];
  }
  float mean = ((sums[0] + sums[1]) + sums[2]) + sums[3];
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2)
    mean += __shfl_down(mean, offset, 32);
  mean = rdna_mul_f32(__shfl(mean, 0, 32), 1.0f / 128);
  const float inverse =
      static_cast<float>(rsqrt(static_cast<double>(mean + epsilon)));
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int column = lane * 4 + item;
    float value = __bfloat162float(__float2bfloat16(values[item] * inverse));
    values[item] = __bfloat162float(
        __float2bfloat16(value * __bfloat162float(weight[column])));
  }
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int column = lane * 4 + item;
    float other = __shfl(values[item], (lane + 16) % 32, 32);
    if (column < 64)
      other = -other;
    rotated[item] = rdna_mul_f32(values[item], cosine[column]) +
                    rdna_mul_f32(other, sine[column]);
  }
}

__global__ __launch_bounds__(32) void q_norm_rope_pack_rdna_kernel(
    const __hip_bfloat16 *input, const __hip_bfloat16 *weight,
    const float *cosine, const float *sine, float *output, int rows, int heads,
    int groups, float epsilon) {
  const int row = blockIdx.x / heads, head = blockIdx.x % heads;
  float rotated[4];
  norm_rope_head_rdna(input + (row * heads + head) * 128, weight,
                      cosine + row * 128, sine + row * 128, epsilon, rotated);
  const int base =
      ((head / groups) * rows * groups + row * groups + head % groups) * 128;
#pragma unroll
  for (int item = 0; item < 4; ++item)
    output[base + threadIdx.x * 4 + item] = rotated[item];
}

__global__ __launch_bounds__(32) void k_norm_rope_cache_rdna_kernel(
    const __hip_bfloat16 *key, const __hip_bfloat16 *value,
    const __hip_bfloat16 *weight, const float *cosine, const float *sine,
    const int64_t *row_map, float *key_cache, float *value_cache, int heads,
    float epsilon) {
  const int position = blockIdx.x / heads, head = blockIdx.x % heads;
  const int row = row_map[position];
  const int base = (position * heads + head) * 128;
  float rotated[4] = {};
  if (row >= 0) {
    norm_rope_head_rdna(key + (row * heads + head) * 128, weight,
                        cosine + row * 128, sine + row * 128, epsilon, rotated);
  }
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int column = threadIdx.x * 4 + item;
    key_cache[base + column] = rotated[item];
    value_cache[base + column] =
        row >= 0 ? __bfloat162float(value[(row * heads + head) * 128 + column])
                 : 0.0f;
  }
}

// ── Residual stream and affine RMSNorm with preserved BF16 stores ──

template <bool Routed>
__device__ __forceinline__ __hip_bfloat16
rms_residual_value_rdna(const void *update, const __hip_bfloat16 *residual,
                        const __hip_bfloat16 *shared, int index) {
  float value;
  if constexpr (Routed) {
    value = __bfloat162float(
        __float2bfloat16(static_cast<const float *>(update)[index]));
    value = __bfloat162float(
        __float2bfloat16(value + __bfloat162float(shared[index])));
  } else {
    value =
        __bfloat162float(static_cast<const __hip_bfloat16 *>(update)[index]);
  }
  return __float2bfloat16(value + __bfloat162float(residual[index]));
}

template <bool Routed, bool Adaptive>
__global__ __launch_bounds__(32) void residual_rms_affine_wide_rdna_kernel(
    const void *update, const __hip_bfloat16 *residual,
    const __hip_bfloat16 *shared, const __hip_bfloat16 *weight,
    const float *gamma, const float *beta, __hip_bfloat16 *summed,
    __hip_bfloat16 *output, float epsilon) {
  constexpr int width = 768;
  const int row = blockIdx.x * width, lane = threadIdx.x;
  float values[6][4], sums[4] = {};
#pragma unroll
  for (int index = 0; index < 6; ++index) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int column = lane * 4 + index * 128 + item;
      const auto value = rms_residual_value_rdna<Routed>(update, residual,
                                                         shared, row + column);
      values[index][item] = __bfloat162float(value);
      summed[row + column] = value;
      sums[item] += values[index][item] * values[index][item];
    }
  }
  float mean = ((sums[0] + sums[1]) + sums[2]) + sums[3];
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2)
    mean += __shfl_down(mean, offset, 32);
  // Keep the mean's FP32 rounding before adding epsilon, as in the row
  // reduction.
  mean = rdna_mul_f32(__shfl(mean, 0, 32), 1.0f / width);
  const float inverse =
      static_cast<float>(rsqrt(static_cast<double>(mean + epsilon)));
#pragma unroll
  for (int index = 0; index < 6; ++index) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int column = lane * 4 + index * 128 + item;
      float value = values[index][item] * inverse;
      if constexpr (!Adaptive)
        value = __bfloat162float(__float2bfloat16(value));
      value *= __bfloat162float(weight[column]);
      if constexpr (Adaptive)
        value = rdna_mul_f32(1.0f + gamma[column], value) + beta[column];
      output[row + column] = __float2bfloat16(value);
    }
  }
}

template <bool Routed, bool Adaptive>
__global__ void residual_rms_affine_rdna_kernel(
    const void *update, const __hip_bfloat16 *residual,
    const __hip_bfloat16 *shared, const __hip_bfloat16 *weight,
    const float *gamma, const float *beta, __hip_bfloat16 *summed,
    __hip_bfloat16 *output, int width, float epsilon) {
  extern __shared__ __hip_bfloat16 values[];
  __shared__ float partial[512];
  const int row = blockIdx.x * width;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const auto value =
        rms_residual_value_rdna<Routed>(update, residual, shared, row + column);
    values[column] = value;
    summed[row + column] = value;
  }
  __syncthreads();
  const float inverse = static_cast<float>(rsqrt(
      static_cast<double>(rms_mean_rdna(values, width, partial) + epsilon)));
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    float value = __bfloat162float(values[column]) * inverse;
    if constexpr (!Adaptive)
      value = __bfloat162float(__float2bfloat16(value));
    value *= __bfloat162float(weight[column]);
    if constexpr (Adaptive)
      value = rdna_mul_f32(1.0f + gamma[column], value) + beta[column];
    output[row + column] = __float2bfloat16(value);
  }
}

// ── BF16 residual norm with a rounded FP32 router input ──

__global__ __launch_bounds__(32) void residual_rms_router_rdna_kernel(
    const __hip_bfloat16 *update, const __hip_bfloat16 *residual,
    const __hip_bfloat16 *weight, const float *gamma, const float *beta,
    __hip_bfloat16 *summed, __hip_bfloat16 *output, float *router_input,
    float epsilon) {
  constexpr int width = 768;
  const int row = blockIdx.x * width, lane = threadIdx.x;
  float values[6][4], sums[4] = {};
#pragma unroll
  for (int index = 0; index < 6; ++index) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int column = lane * 4 + index * 128 + item;
      const auto value = rms_residual_value_rdna<false>(update, residual,
                                                        nullptr, row + column);
      values[index][item] = __bfloat162float(value);
      summed[row + column] = value;
      sums[item] += values[index][item] * values[index][item];
    }
  }
  float mean = ((sums[0] + sums[1]) + sums[2]) + sums[3];
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2)
    mean += __shfl_down(mean, offset, 32);
  mean = rdna_mul_f32(__shfl(mean, 0, 32), 1.0f / width);
  const float inverse =
      static_cast<float>(rsqrt(static_cast<double>(mean + epsilon)));
#pragma unroll
  for (int index = 0; index < 6; ++index) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int column = lane * 4 + index * 128 + item;
      float value = values[index][item] * inverse;
      value *= __bfloat162float(weight[column]);
      value = rdna_mul_f32(1.0f + gamma[column], value) + beta[column];
      const auto rounded = __float2bfloat16(value);
      output[row + column] = rounded;
      router_input[row + column] = __bfloat162float(rounded);
    }
  }
}

template <bool Routed, bool Adaptive>
void residual_rms_affine_impl_rdna(const void *update,
                                   const __hip_bfloat16 *residual,
                                   const __hip_bfloat16 *shared,
                                   const __hip_bfloat16 *weight,
                                   const float *gamma, const float *beta,
                                   __hip_bfloat16 *summed,
                                   __hip_bfloat16 *output, int rows, int width,
                                   float epsilon, hipStream_t stream) {
  const int threads = rms_threads_rdna(rows, width);
  if (width == 768 && threads == 32) {
    residual_rms_affine_wide_rdna_kernel<Routed, Adaptive>
        <<<rows, 32, 0, stream>>>(update, residual, shared, weight, gamma, beta,
                                  summed, output, epsilon);
  } else {
    residual_rms_affine_rdna_kernel<Routed, Adaptive>
        <<<rows, threads, width * sizeof(__hip_bfloat16), stream>>>(
            update, residual, shared, weight, gamma, beta, summed, output,
            width, epsilon);
  }
}

} // namespace

// ── Host dispatch ──
// Select only measured fixed-width variants; preserve the generic kernels for
// unsupported widths and for API compatibility with the CDNA backend.

void layer_norm_rdna(const __hip_bfloat16 *input, const __hip_bfloat16 *weight,
                     const __hip_bfloat16 *bias, __hip_bfloat16 *output,
                     int rows, int width, float epsilon, hipStream_t stream) {
  if (width == 1152) {
    layer_norm_wide_rdna_kernel<1152, 64><<<rows, 64, 0, stream>>>(
        input, weight, bias, output, epsilon);
    return;
  }
  layer_norm_rdna_kernel<<<rows, kNormThreads, 0, stream>>>(
      input, weight, bias, output, width, epsilon);
}

void rms_norm_rdna(const __hip_bfloat16 *input, __hip_bfloat16 *output,
                   int rows, int width, float epsilon, hipStream_t stream) {
  if (width == 2048) {
    rms_norm_wide_rdna_kernel<2048, 64><<<rows, 64, 0, stream>>>(
        input, output, epsilon);
    return;
  }
  rms_norm_rdna_kernel<<<rows, kNormThreads, 0, stream>>>(input, output, width,
                                                          epsilon);
}

void adarms_rdna(const __hip_bfloat16 *input, const __hip_bfloat16 *modulation,
                 __hip_bfloat16 *output, int rows, int width, float epsilon,
                 hipStream_t stream) {
  adarms_rdna_kernel<<<rows, kNormThreads, 0, stream>>>(input, modulation,
                                                        output, width, epsilon);
}

void residual_rms_rdna(const __hip_bfloat16 *update,
                       const __hip_bfloat16 *residual,
                       __hip_bfloat16 *output_sum, __hip_bfloat16 *output_norm,
                       int rows, int width, float epsilon, hipStream_t stream) {
  if (width == 2048) {
    residual_rms_wide_rdna_kernel<2048, 256><<<rows, 256, 0, stream>>>(
        update, residual, output_sum, output_norm, epsilon);
    return;
  }
  residual_rms_rdna_kernel<<<rows, kNormThreads, 0, stream>>>(
      update, residual, output_sum, output_norm, width, epsilon);
}

void residual_adarms_rdna(const __hip_bfloat16 *update,
                          const __hip_bfloat16 *residual,
                          const __hip_bfloat16 *gate,
                          const __hip_bfloat16 *modulation,
                          __hip_bfloat16 *output_sum,
                          __hip_bfloat16 *output_norm, int rows, int width,
                          float epsilon, hipStream_t stream) {
  residual_adarms_rdna_kernel<<<rows, kNormThreads, 0, stream>>>(
      update, residual, gate, modulation, output_sum, output_norm, width,
      epsilon);
}

void rms_norm_affine_rdna(const __hip_bfloat16 *input,
                          const __hip_bfloat16 *weight, __hip_bfloat16 *output,
                          int rows, int width, float epsilon,
                          hipStream_t stream) {
  rms_affine_rdna_kernel<false>
      <<<rows, rms_threads_rdna(rows, width), 0, stream>>>(
          input, weight, nullptr, nullptr, output, width, epsilon);
}

void q_norm_rope_pack_rdna(const __hip_bfloat16 *input,
                           const __hip_bfloat16 *weight, const float *cosine,
                           const float *sine, float *output, int rows,
                           int heads, int groups, float epsilon,
                           hipStream_t stream) {
  q_norm_rope_pack_rdna_kernel<<<rows * heads, 32, 0, stream>>>(
      input, weight, cosine, sine, output, rows, heads, groups, epsilon);
}

void k_norm_rope_cache_rdna(const __hip_bfloat16 *key,
                            const __hip_bfloat16 *value,
                            const __hip_bfloat16 *weight, const float *cosine,
                            const float *sine, const int64_t *row_map,
                            float *key_cache, float *value_cache, int prefix,
                            int heads, float epsilon, hipStream_t stream) {
  k_norm_rope_cache_rdna_kernel<<<prefix * heads, 32, 0, stream>>>(
      key, value, weight, cosine, sine, row_map, key_cache, value_cache, heads,
      epsilon);
}

void ada_rms_norm_fp32_rdna(const __hip_bfloat16 *input,
                            const __hip_bfloat16 *weight, const float *gamma,
                            const float *beta, __hip_bfloat16 *output, int rows,
                            int width, float epsilon, hipStream_t stream) {
  rms_affine_rdna_kernel<true>
      <<<rows, rms_threads_rdna(rows, width), 0, stream>>>(
          input, weight, gamma, beta, output, width, epsilon);
}

void residual_rms_affine_rdna(const void *update,
                              const __hip_bfloat16 *residual,
                              const __hip_bfloat16 *shared,
                              const __hip_bfloat16 *weight, const float *gamma,
                              const float *beta, __hip_bfloat16 *summed,
                              __hip_bfloat16 *output, int rows, int width,
                              float epsilon, hipStream_t stream) {
#define FLASHRT_LAUNCH_RESIDUAL_NORM(routed, adaptive)                         \
  residual_rms_affine_impl_rdna<routed, adaptive>(                             \
      update, residual, shared, weight, gamma, beta, summed, output, rows,     \
      width, epsilon, stream)
  if (shared && gamma) {
    FLASHRT_LAUNCH_RESIDUAL_NORM(true, true);
  } else if (shared) {
    FLASHRT_LAUNCH_RESIDUAL_NORM(true, false);
  } else if (gamma) {
    FLASHRT_LAUNCH_RESIDUAL_NORM(false, true);
  } else {
    FLASHRT_LAUNCH_RESIDUAL_NORM(false, false);
  }
#undef FLASHRT_LAUNCH_RESIDUAL_NORM
}

void residual_rms_router_rdna(const __hip_bfloat16 *update,
                              const __hip_bfloat16 *residual,
                              const __hip_bfloat16 *weight, const float *gamma,
                              const float *beta, __hip_bfloat16 *summed,
                              __hip_bfloat16 *output, float *router_input,
                              int rows, float epsilon, hipStream_t stream) {
  residual_rms_router_rdna_kernel<<<rows, 32, 0, stream>>>(
      update, residual, weight, gamma, beta, summed, output, router_input,
      epsilon);
}
