// ================================================================
// FlashRT AMD -- fused BF16 decoder GQA (RDNA 3.5, wave32)
// Portions of the split-key online-softmax variant are adapted from Comfy
// Kitchen's Apache-2.0 HIP flash-decode implementation.
//
// BF16 GQA contract:
//   Q [Sq, Hq, 256] BF16, Sq <= 16, Hq <= 16
//   K/V [Skv, 1, 256] BF16, Skv <= 2048
//   O [Sq, Hq, 256] BF16
// K/V have one shared head, while each output workgroup owns one (query, head)
// row. The row-owned comparison path is limited to Skv <= 1024 because it
// materializes scores in LDS; the production split-key path covers 2048.
// The valid mask is the union [0, valid_prefix) U [suffix_start, Skv),
// which represents both compact KV and the padded prefix-plus-decoder layout.
//
// Two implementations share the same raw-pointer ABI:
//
//   split-key (production): eight waves traverse disjoint KV rows. Each lane
//   owns eight contiguous head dimensions, computes its partial QK dot, and
//   updates an online-softmax/PV accumulator. One LDS exchange combines the
//   eight independently normalized wave results. KeysPerIteration controls
//   per-wave load ILP and is fixed when a HIP graph is captured.
//
//   row-owned (comparison/fallback): 256 threads stage Q, write all scores to
//   LDS, normalize them with wave32 reductions, then stream V by output
//   dimension. It uses more barriers but is useful as an independent native
//   implementation and can win in a hot-cache microbenchmark.
//
// Both paths accumulate in FP32 and write BF16 once. They allocate no global
// workspace, launch no reduction kernel, use no atomics, and are safe for HIP
// graph capture and replay.
// ================================================================

#include "../kernels/common_hip_rdna.h"

#include <cfloat>
#include <stdexcept>

namespace {

constexpr int kAttentionThreads = 256;
constexpr int kAttentionDim = 256;
constexpr int kAttentionMaxQuery = 16;
// The row-owned comparison kernel materializes every score in LDS, so keep
// its original 1024-row allocation.  The production split-key kernel streams
// K/V and can cover the CDNA decoder contract without increasing LDS or
// changing the <=1024 hot specialization.
constexpr int kAttentionRowMaxKv = 1024;
constexpr int kAttentionSplitMaxKv = 2048;
constexpr int kAttentionMaxWaves = kAttentionThreads / FLASHRT_RDNA_WAVE_SIZE;

struct alignas(16) Bf16x8Rdna {
  __hip_bfloat162 p0;
  __hip_bfloat162 p1;
  __hip_bfloat162 p2;
  __hip_bfloat162 p3;
};

using Bf16Vec8Rdna = __bf16 __attribute__((ext_vector_type(8)));
using FloatVec8Rdna = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ FloatVec8Rdna
load_bf16x8_rdna(const __hip_bfloat16 *source) {
  const Bf16Vec8Rdna raw = *reinterpret_cast<const Bf16Vec8Rdna *>(source);
  FloatVec8Rdna result;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    result[index] = static_cast<float>(raw[index]);
  }
  return result;
}

__device__ __forceinline__ float attention_wave_sum_rdna(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset, FLASHRT_RDNA_WAVE_SIZE);
  }
  return value;
}

constexpr int kSplitKeyWaves = 8;

// ── Split-key online-softmax kernel ──
// Every wave computes a locally normalized partial. The final correction by
// exp2(local_max - global_max) makes the merge algebraically equivalent to a
// single online-softmax traversal.
template <int KeysPerIteration>
__global__
__launch_bounds__(kSplitKeyWaves *FLASHRT_RDNA_WAVE_SIZE) void attention_decoder_gqa_splitkey_rdna_kernel(
    const __hip_bfloat16 *__restrict__ query,
    const __hip_bfloat16 *__restrict__ key,
    const __hip_bfloat16 *__restrict__ value,
    __hip_bfloat16 *__restrict__ output, int query_rows, int kv_rows,
    int query_heads, int valid_prefix, int suffix_start, float scale_log2) {
  const int head = blockIdx.x;
  const int query_row = blockIdx.y;
  const int wave = threadIdx.x / FLASHRT_RDNA_WAVE_SIZE;
  const int lane = threadIdx.x & (FLASHRT_RDNA_WAVE_SIZE - 1);

  const FloatVec8Rdna query_values = load_bf16x8_rdna(
      query +
      (static_cast<size_t>(query_row) * query_heads + head) * kAttentionDim +
      lane * 8);
  FloatVec8Rdna output_values = {0.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = -FLT_MAX / 4.0f;
  float running_sum = 0.0f;

  constexpr int key_stride = kSplitKeyWaves * KeysPerIteration;
  for (int key_start = wave * KeysPerIteration; key_start < kv_rows;
       key_start += key_stride) {
    FloatVec8Rdna key_values[KeysPerIteration];
    FloatVec8Rdna value_values[KeysPerIteration];
#pragma unroll
    for (int item = 0; item < KeysPerIteration; ++item) {
      const int key_row = key_start + item;
      const int clamped_row = key_row < kv_rows ? key_row : kv_rows - 1;
      const size_t offset =
          static_cast<size_t>(clamped_row) * kAttentionDim + lane * 8;
      key_values[item] = load_bf16x8_rdna(key + offset);
      value_values[item] = load_bf16x8_rdna(value + offset);
    }

#pragma unroll
    for (int item = 0; item < KeysPerIteration; ++item) {
      const int key_row = key_start + item;
      const bool valid = key_row < kv_rows &&
                         (key_row < valid_prefix || key_row >= suffix_start);
      float score = 0.0f;
#pragma unroll
      for (int dim = 0; dim < 8; ++dim) {
        score = fmaf(query_values[dim], key_values[item][dim], score);
      }
      score = attention_wave_sum_rdna(score) * scale_log2;
      const float masked_score = valid ? score : -FLT_MAX / 4.0f;
      const float next_max = fmaxf(running_max, masked_score);
      const float correction = exp2f(running_max - next_max);
      const float probability = valid ? exp2f(score - next_max) : 0.0f;
      running_sum = fmaf(running_sum, correction, probability);
#pragma unroll
      for (int dim = 0; dim < 8; ++dim) {
        output_values[dim] = fmaf(output_values[dim], correction,
                                  probability * value_values[item][dim]);
      }
      running_max = next_max;
    }
  }

  __shared__ float maxima[kSplitKeyWaves];
  __shared__ float denominators[kSplitKeyWaves];
  __shared__ float partial_output[kSplitKeyWaves][kAttentionDim];
  if (lane == 0) {
    maxima[wave] = running_max;
    denominators[wave] = running_sum;
  }
#pragma unroll
  for (int dim = 0; dim < 8; ++dim) {
    partial_output[wave][lane * 8 + dim] = output_values[dim];
  }
  __syncthreads();

  if (wave != 0) {
    return;
  }
  float maximum = maxima[0];
#pragma unroll
  for (int other_wave = 1; other_wave < kSplitKeyWaves; ++other_wave) {
    maximum = fmaxf(maximum, maxima[other_wave]);
  }
  float denominator = 0.0f;
  FloatVec8Rdna accumulated = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int other_wave = 0; other_wave < kSplitKeyWaves; ++other_wave) {
    const float correction = exp2f(maxima[other_wave] - maximum);
    denominator = fmaf(denominators[other_wave], correction, denominator);
#pragma unroll
    for (int dim = 0; dim < 8; ++dim) {
      accumulated[dim] = fmaf(partial_output[other_wave][lane * 8 + dim],
                              correction, accumulated[dim]);
    }
  }
  const float inverse_denominator = 1.0f / denominator;
  __hip_bfloat16 *destination =
      output +
      (static_cast<size_t>(query_row) * query_heads + head) * kAttentionDim +
      lane * 8;
#pragma unroll
  for (int dim = 0; dim < 8; ++dim) {
    destination[dim] = __float2bfloat16(accumulated[dim] * inverse_denominator);
  }
}

// ── Row-owned LDS-score kernel ──
// Phase A stages Q and computes one QK score per thread-owned KV row.
// Phase B normalizes the LDS score vector. Phase C assigns one output
// dimension to each thread and streams the corresponding V column.
__global__
__launch_bounds__(kAttentionThreads) void attention_decoder_gqa_rdna_kernel(
    const __hip_bfloat16 *__restrict__ query,
    const __hip_bfloat16 *__restrict__ key,
    const __hip_bfloat16 *__restrict__ value,
    __hip_bfloat16 *__restrict__ output, int query_rows, int kv_rows,
    int query_heads, int valid_prefix, int suffix_start, float scale) {
  const int head = blockIdx.x;
  const int query_row = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float query_lds[kAttentionDim];
  __shared__ float scores[kAttentionRowMaxKv];
  __shared__ float wave_values[kAttentionMaxWaves];

  query_lds[tid] = __bfloat162float(
      query[(static_cast<size_t>(query_row) * query_heads + head) *
                kAttentionDim +
            tid]);
  __syncthreads();

  constexpr int packets = kAttentionDim / 8;
  for (int kv_row = tid; kv_row < kv_rows; kv_row += kAttentionThreads) {
    const bool valid = kv_row < valid_prefix || kv_row >= suffix_start;
    float score = -FLT_MAX;
    if (valid) {
      const Bf16x8Rdna *key_row = reinterpret_cast<const Bf16x8Rdna *>(
          key + static_cast<size_t>(kv_row) * kAttentionDim);
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      float acc2 = 0.0f;
      float acc3 = 0.0f;
      float acc4 = 0.0f;
      float acc5 = 0.0f;
      float acc6 = 0.0f;
      float acc7 = 0.0f;
#pragma unroll 4
      for (int packet = 0; packet < packets; ++packet) {
        const Bf16x8Rdna keys = key_row[packet];
        const float4 *queries =
            reinterpret_cast<const float4 *>(query_lds + packet * 8);
        const float4 query_lo = queries[0];
        const float4 query_hi = queries[1];
        acc0 = fmaf(__bfloat162float(keys.p0.x), query_lo.x, acc0);
        acc1 = fmaf(__bfloat162float(keys.p0.y), query_lo.y, acc1);
        acc2 = fmaf(__bfloat162float(keys.p1.x), query_lo.z, acc2);
        acc3 = fmaf(__bfloat162float(keys.p1.y), query_lo.w, acc3);
        acc4 = fmaf(__bfloat162float(keys.p2.x), query_hi.x, acc4);
        acc5 = fmaf(__bfloat162float(keys.p2.y), query_hi.y, acc5);
        acc6 = fmaf(__bfloat162float(keys.p3.x), query_hi.z, acc6);
        acc7 = fmaf(__bfloat162float(keys.p3.y), query_hi.w, acc7);
      }
      score =
          (((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) *
          scale;
    }
    scores[kv_row] = score;
  }
  __syncthreads();

  float local_maximum = -FLT_MAX;
  for (int kv_row = tid; kv_row < kv_rows; kv_row += kAttentionThreads) {
    local_maximum = fmaxf(local_maximum, scores[kv_row]);
  }
  const float maximum = rdna_block_reduce_max(local_maximum, wave_values);

  float local_sum = 0.0f;
  for (int kv_row = tid; kv_row < kv_rows; kv_row += kAttentionThreads) {
    const float probability = __expf(scores[kv_row] - maximum);
    scores[kv_row] = probability;
    local_sum += probability;
  }
  const float denominator = rdna_block_reduce_sum(local_sum, wave_values);
  const float inverse_denominator = 1.0f / denominator;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  int kv_row = 0;
  for (; kv_row + 3 < kv_rows; kv_row += 4) {
    const float p0 = scores[kv_row];
    const float p1 = scores[kv_row + 1];
    const float p2 = scores[kv_row + 2];
    const float p3 = scores[kv_row + 3];
    acc0 = fmaf(p0,
                __bfloat162float(
                    value[static_cast<size_t>(kv_row) * kAttentionDim + tid]),
                acc0);
    acc1 =
        fmaf(p1,
             __bfloat162float(
                 value[static_cast<size_t>(kv_row + 1) * kAttentionDim + tid]),
             acc1);
    acc2 =
        fmaf(p2,
             __bfloat162float(
                 value[static_cast<size_t>(kv_row + 2) * kAttentionDim + tid]),
             acc2);
    acc3 =
        fmaf(p3,
             __bfloat162float(
                 value[static_cast<size_t>(kv_row + 3) * kAttentionDim + tid]),
             acc3);
  }
  for (; kv_row < kv_rows; ++kv_row) {
    acc0 = fmaf(scores[kv_row],
                __bfloat162float(
                    value[static_cast<size_t>(kv_row) * kAttentionDim + tid]),
                acc0);
  }
  output[(static_cast<size_t>(query_row) * query_heads + head) * kAttentionDim +
         tid] =
      __float2bfloat16(((acc0 + acc1) + (acc2 + acc3)) * inverse_denominator);
}

} // namespace

// ── Host launchers and shape contracts ──

void attention_decoder_gqa_rdna(const __hip_bfloat16 *query,
                                const __hip_bfloat16 *key,
                                const __hip_bfloat16 *value,
                                __hip_bfloat16 *output, int query_rows,
                                int kv_rows, int query_heads, int head_dim,
                                int valid_prefix, int suffix_start, float scale,
                                hipStream_t stream) {
  if (query_rows < 1 || query_rows > kAttentionMaxQuery || kv_rows < 1 ||
      kv_rows > kAttentionRowMaxKv || query_heads < 1 || query_heads > 16 ||
      head_dim != kAttentionDim || valid_prefix < 1 ||
      valid_prefix > suffix_start || suffix_start > kv_rows) {
    throw std::runtime_error(
        "attention_decoder_gqa_rdna: unsupported Pi0.5 shape");
  }
  const dim3 grid(query_heads, query_rows);
  attention_decoder_gqa_rdna_kernel<<<grid, kAttentionThreads, 0, stream>>>(
      query, key, value, output, query_rows, kv_rows, query_heads, valid_prefix,
      suffix_start, scale);
}

void attention_decoder_gqa_splitkey_rdna(
    const __hip_bfloat16 *query, const __hip_bfloat16 *key,
    const __hip_bfloat16 *value, __hip_bfloat16 *output, int query_rows,
    int kv_rows, int query_heads, int head_dim, int valid_prefix,
    int suffix_start, float scale, int keys_per_iteration, hipStream_t stream) {
  if (query_rows < 1 || query_rows > kAttentionMaxQuery || kv_rows < 1 ||
      kv_rows > kAttentionSplitMaxKv || query_heads < 1 || query_heads > 16 ||
      head_dim != kAttentionDim || valid_prefix < 1 ||
      valid_prefix > suffix_start || suffix_start > kv_rows) {
    throw std::runtime_error(
        "attention_decoder_gqa_splitkey_rdna: unsupported Pi0.5 shape");
  }
  const dim3 grid(query_heads, query_rows);
  const float scale_log2 = scale * 1.4426950408889634f;
#define FLASHRT_LAUNCH_SPLITKEY(keys)                                          \
  attention_decoder_gqa_splitkey_rdna_kernel<keys>                             \
      <<<grid, kSplitKeyWaves * FLASHRT_RDNA_WAVE_SIZE, 0, stream>>>(          \
          query, key, value, output, query_rows, kv_rows, query_heads,         \
          valid_prefix, suffix_start, scale_log2)
  switch (keys_per_iteration) {
  case 1:
    FLASHRT_LAUNCH_SPLITKEY(1);
    break;
  case 2:
    FLASHRT_LAUNCH_SPLITKEY(2);
    break;
  case 4:
    FLASHRT_LAUNCH_SPLITKEY(4);
    break;
  case 8:
    FLASHRT_LAUNCH_SPLITKEY(8);
    break;
  default:
    throw std::runtime_error(
        "attention_decoder_gqa_splitkey_rdna: keys/iteration must be 1/2/4/8");
  }
#undef FLASHRT_LAUNCH_SPLITKEY
}
