// ================================================================
// FlashRT AMD -- BF16 normalization kernels (RDNA 3.5, wave32)
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
  __shared__ float wave_sums[kMaxWaves];
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
      rdna_block_reduce_sum(local_variance, wave_sums) / width;
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
