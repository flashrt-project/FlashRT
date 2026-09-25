// ================================================================
// FlashRT AMD -- fused BF16 encoder GQA attention (RDNA 3.5, wave32)
//
// BF16 GQA contract:
//   Q [S, Hq, 256] BF16, S <= 4096, Hq <= 16
//   K/V [S, 1, 256] BF16 (only [0, valid_kv_rows) participates)
//   O [S, Hq, 256] BF16, bidirectional and unmasked inside the valid prefix
//
// The kernel is an online-softmax flash-attention implementation built from
// gfx11 wave32 BF16 WMMA instructions. A wave pair owns a 16-row Q tile: its
// leader computes QK and softmax, and the two waves split the 256 output
// columns. Several pairs share one V tile inside a workgroup:
//
//   Phase A: all threads load a 16x256 V tile once and transpose it into LDS.
//   Phase B: pair leaders compute 16x16 QK fragments, update per-row running
//            max/sum, and store BF16 probabilities plus rescale factors.
//   Phase C: both waves consume the shared P/V tiles with WMMA and retain
//            their 16x128 FP32 output half in registers across all KV tiles.
//   Epilogue: divide by the online-softmax denominator and store BF16 output.
//
// The 20-element LDS row pitch is intentional: a conventional 16 or 18
// stride creates costly gfx1151 WMMA bank patterns. The measured 40-byte
// stride removes that conflict without introducing scratch memory.
//
// Dispatch is shape-specialized. The Hq=8 Pi0.5 path keeps its compile-time
// head stride: compact inputs use five wave pairs (80 query rows from one head
// per 320-thread workgroup), while inputs above 640 rows use two pairs (16
// rows from two heads per 128-thread workgroup) to avoid the gfx1151 tail-wave
// residency cliff. Other Hq values use one head per workgroup.
// Reduction order, KV-tile order, and output ownership are fixed; no atomics
// or global workspace are used, so HIP graph replay is deterministic.
// ================================================================

#include "../kernels/common_hip_rdna.h"

#include <cfloat>
#include <stdexcept>

namespace {

using Bf16x16EncoderRdna = __bf16 __attribute__((ext_vector_type(16)));
using Floatx8EncoderRdna = float __attribute__((ext_vector_type(8)));

union Bf16EncoderFragmentRdna {
  Bf16x16EncoderRdna values;
  unsigned long long words[4];
};

constexpr int kEncoderHeads = 8;
constexpr int kEncoderDim = 256;
constexpr int kEncoderWaveQueryTile = 16;
constexpr int kEncoderKeyTile = 16;
constexpr int kEncoderWavesPerQueryTile = 2;
constexpr int kEncoderDimTiles = kEncoderDim / 16;
constexpr int kEncoderDimTilesPerWave =
    kEncoderDimTiles / kEncoderWavesPerQueryTile;
// A 40-byte row stride matches the RDNA 3.5 WMMA LDS access pattern and
// avoids the severe bank conflicts observed with the conventional +2 pad.
constexpr int kEncoderLdsPitch = 20;
constexpr float kEncoderNegInf = -3.0e38f;

struct EncoderAttentionLdsRdna {
  __bf16 value_transposed[kEncoderDim][kEncoderLdsPitch];
  __bf16 probability[5][kEncoderWaveQueryTile][kEncoderLdsPitch];
  float output_scale[5][kEncoderWaveQueryTile];
  float inverse_sum[5][kEncoderWaveQueryTile];
};

__device__ __forceinline__ int encoder_accumulator_row_rdna(int lane,
                                                            int item) {
  return 2 * item + (lane >> 4);
}

__device__ __forceinline__ float encoder_row_reduce_max_rdna(float value) {
#pragma unroll
  for (int offset = 1; offset <= 8; offset <<= 1) {
    value = fmaxf(value, __shfl_xor(value, offset, FLASHRT_RDNA_WAVE_SIZE));
  }
  return value;
}

__device__ __forceinline__ float encoder_row_reduce_sum_rdna(float value) {
#pragma unroll
  for (int offset = 1; offset <= 8; offset <<= 1) {
    value += __shfl_xor(value, offset, FLASHRT_RDNA_WAVE_SIZE);
  }
  return value;
}

__device__ __forceinline__ Bf16x16EncoderRdna
load_encoder_fragment_rdna(const __bf16 *source) {
  Bf16EncoderFragmentRdna fragment;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    fragment.words[index] =
        reinterpret_cast<const unsigned long long *>(source)[index];
  }
  return fragment.values;
}

template <int QueryTilesPerHead, int HeadsPerGroup, bool RuntimeHeadStride>
__global__ __launch_bounds__(QueryTilesPerHead * HeadsPerGroup *
                                 kEncoderWavesPerQueryTile *
                                 FLASHRT_RDNA_WAVE_SIZE,
                             2) void attention_encoder_gqa_rdna_kernel(
    const __hip_bfloat16 *__restrict__ query,
    const __hip_bfloat16 *__restrict__ key,
    const __hip_bfloat16 *__restrict__ value,
    __hip_bfloat16 *__restrict__ output, int query_rows, int valid_kv_rows,
    int query_heads, float scale_log2) {
  __shared__ __align__(16) EncoderAttentionLdsRdna lds;
  constexpr int wave_pairs = QueryTilesPerHead * HeadsPerGroup;
  static_assert(wave_pairs <= 5);
  constexpr int threads =
      wave_pairs * kEncoderWavesPerQueryTile * FLASHRT_RDNA_WAVE_SIZE;
  constexpr int query_tile = QueryTilesPerHead * kEncoderWaveQueryTile;
  const int tid = threadIdx.x;
  const int wave = tid / FLASHRT_RDNA_WAVE_SIZE;
  const int lane = tid & (FLASHRT_RDNA_WAVE_SIZE - 1);
  const int fragment_row = lane & 15;
  const int wave_pair = wave / kEncoderWavesPerQueryTile;
  const int head_in_group = wave_pair / QueryTilesPerHead;
  const int query_tile_in_head = wave_pair % QueryTilesPerHead;
  const int output_wave = wave % kEncoderWavesPerQueryTile;
  const int head = blockIdx.y * HeadsPerGroup + head_in_group;
  const int head_stride = RuntimeHeadStride ? query_heads : kEncoderHeads;
  const int query_start = blockIdx.x * query_tile +
                          query_tile_in_head * kEncoderWaveQueryTile;

  Floatx8EncoderRdna output_accumulator[kEncoderDimTilesPerWave];
#pragma unroll
  for (int tile = 0; tile < kEncoderDimTilesPerWave; ++tile) {
    output_accumulator[tile] =
        (Floatx8EncoderRdna){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  }
  float running_max[8];
  float running_sum[8];
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    running_max[item] = kEncoderNegInf;
    running_sum[item] = 0.0f;
  }

  constexpr int vectors_per_value_row = kEncoderDim / 8;
  for (int key_start = 0; key_start < valid_kv_rows;
       key_start += kEncoderKeyTile) {
    // Stage V once for all wave pairs. Stores transpose [K,D] to [D,K], so
    // every WMMA B fragment below is one aligned, contiguous LDS read.
    const int key_count = min(kEncoderKeyTile, valid_kv_rows - key_start);
    for (int vector = tid; vector < kEncoderKeyTile * vectors_per_value_row;
         vector += threads) {
      const int key_in_tile = vector / vectors_per_value_row;
      const int dim_start = (vector - key_in_tile * vectors_per_value_row) * 8;
      __bf16 loaded[8];
      if (key_in_tile < key_count) {
        const __hip_bfloat16 *source =
            value + static_cast<size_t>(key_start + key_in_tile) * kEncoderDim +
            dim_start;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          loaded[item] = source[item];
        }
      } else {
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          loaded[item] = __float2bfloat16(0.0f);
        }
      }
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        lds.value_transposed[dim_start + item][key_in_tile] = loaded[item];
      }
    }

    // Only the first wave in each pair computes QK and online softmax. The
    // second wave receives probabilities and correction factors through LDS.
    if (output_wave == 0) {
      Floatx8EncoderRdna score_accumulator =
          (Floatx8EncoderRdna){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      const int query_row = min(query_start + fragment_row, query_rows - 1);
      const int key_row = key_start + min(fragment_row, key_count - 1);
      const __bf16 *query_head = reinterpret_cast<const __bf16 *>(
          query + (static_cast<size_t>(query_row) * head_stride + head) *
                      kEncoderDim);
      const __bf16 *key_head = reinterpret_cast<const __bf16 *>(
          key + static_cast<size_t>(key_row) * kEncoderDim);
#pragma unroll
      for (int dim = 0; dim < kEncoderDim; dim += 16) {
        score_accumulator = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            load_encoder_fragment_rdna(query_head + dim),
            load_encoder_fragment_rdna(key_head + dim), score_accumulator);
      }

      float probability[8];
      float next_max[8];
      const bool valid_key = fragment_row < key_count;
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const float score =
            valid_key ? score_accumulator[item] * scale_log2 : kEncoderNegInf;
        next_max[item] =
            encoder_row_reduce_max_rdna(fmaxf(running_max[item], score));
        probability[item] = score;
      }
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const float correction = exp2f(running_max[item] - next_max[item]);
        const float unnormalized =
            valid_key ? exp2f(probability[item] - next_max[item]) : 0.0f;
        const float tile_sum = encoder_row_reduce_sum_rdna(unnormalized);
        running_sum[item] = fmaf(running_sum[item], correction, tile_sum);
        running_max[item] = next_max[item];
        const int row = encoder_accumulator_row_rdna(lane, item);
        lds.probability[wave_pair][row][fragment_row] =
            __float2bfloat16(unnormalized);
        if (fragment_row == 0) {
          lds.output_scale[wave_pair][row] = correction;
        }
      }
    }

    // Publishes V, P, and output_scale to every consumer wave.
    __syncthreads();

#pragma unroll
    for (int tile = 0; tile < kEncoderDimTilesPerWave; ++tile) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int row = encoder_accumulator_row_rdna(lane, item);
        output_accumulator[tile][item] *= lds.output_scale[wave_pair][row];
      }
      const int dim_tile = output_wave * kEncoderDimTilesPerWave + tile;
      output_accumulator[tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
          load_encoder_fragment_rdna(
              &lds.probability[wave_pair][fragment_row][0]),
          load_encoder_fragment_rdna(
              &lds.value_transposed[dim_tile * 16 + fragment_row][0]),
          output_accumulator[tile]);
    }
    // Consumers must finish before the next KV tile overwrites the LDS slots.
    __syncthreads();
  }

  if (output_wave == 0 && fragment_row == 0) {
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int row = encoder_accumulator_row_rdna(lane, item);
      lds.inverse_sum[wave_pair][row] =
          1.0f / fmaxf(running_sum[item], 1.0e-30f);
    }
  }
  __syncthreads();

#pragma unroll
  for (int tile = 0; tile < kEncoderDimTilesPerWave; ++tile) {
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int query_in_tile = encoder_accumulator_row_rdna(lane, item);
      const int query_row = query_start + query_in_tile;
      const int dim =
          (output_wave * kEncoderDimTilesPerWave + tile) * 16 + fragment_row;
      if (query_row < query_rows) {
        output[(static_cast<size_t>(query_row) * head_stride + head) *
                   kEncoderDim +
               dim] =
            __float2bfloat16(output_accumulator[tile][item] *
                             lds.inverse_sum[wave_pair][query_in_tile]);
      }
    }
  }
}

} // namespace

// ── Host dispatch and fixed-shape validation ──

void attention_encoder_gqa_rdna(const __hip_bfloat16 *query,
                                const __hip_bfloat16 *key,
                                const __hip_bfloat16 *value,
                                __hip_bfloat16 *output, int query_rows,
                                int valid_kv_rows, int query_heads,
                                int head_dim, hipStream_t stream) {
  if (query_rows < 1 || query_rows > 4096 || valid_kv_rows < 1 ||
      valid_kv_rows > query_rows || query_heads < 1 || query_heads > 16 ||
      head_dim != kEncoderDim) {
    throw std::runtime_error(
        "attention_encoder_gqa_rdna: unsupported Pi0.5 shape");
  }
  constexpr float scale_log2 = 0.0625f * 1.4426950408889634f;
  if (query_heads == kEncoderHeads && query_rows <= 640) {
    constexpr int query_tiles_per_head = 5;
    constexpr int heads_per_group = 1;
    constexpr int query_tile =
        query_tiles_per_head * kEncoderWaveQueryTile;
    constexpr int threads = query_tiles_per_head * heads_per_group *
                            kEncoderWavesPerQueryTile *
                            FLASHRT_RDNA_WAVE_SIZE;
    const dim3 grid((query_rows + query_tile - 1) / query_tile,
                    kEncoderHeads / heads_per_group);
    attention_encoder_gqa_rdna_kernel<query_tiles_per_head, heads_per_group,
                                      false>
        <<<grid, threads, 0, stream>>>(query, key, value, output, query_rows,
                                      valid_kv_rows, query_heads, scale_log2);
  } else if (query_heads == kEncoderHeads) {
    // Smaller workgroups avoid the tail-wave residency cliff seen with the
    // 320-thread specialization for the padded 712-row model shape.
    constexpr int query_tiles_per_head = 1;
    constexpr int heads_per_group = 2;
    constexpr int query_tile =
        query_tiles_per_head * kEncoderWaveQueryTile;
    constexpr int threads = query_tiles_per_head * heads_per_group *
                            kEncoderWavesPerQueryTile *
                            FLASHRT_RDNA_WAVE_SIZE;
    const dim3 grid((query_rows + query_tile - 1) / query_tile,
                    kEncoderHeads / heads_per_group);
    attention_encoder_gqa_rdna_kernel<query_tiles_per_head, heads_per_group,
                                      false>
        <<<grid, threads, 0, stream>>>(query, key, value, output, query_rows,
                                      valid_kv_rows, query_heads, scale_log2);
  } else if (query_rows <= 640) {
    // The generic-head path keeps one head per workgroup, so every value in
    // the public Hq=[1,16] range is valid without partially populated groups.
    // It is separate from the Hq=8 specialization to keep the current Pi0.5
    // address arithmetic compile-time constant.
    constexpr int query_tiles_per_head = 5;
    constexpr int heads_per_group = 1;
    constexpr int query_tile =
        query_tiles_per_head * kEncoderWaveQueryTile;
    constexpr int threads = query_tiles_per_head * heads_per_group *
                            kEncoderWavesPerQueryTile *
                            FLASHRT_RDNA_WAVE_SIZE;
    const dim3 grid((query_rows + query_tile - 1) / query_tile, query_heads);
    attention_encoder_gqa_rdna_kernel<query_tiles_per_head, heads_per_group,
                                      true>
        <<<grid, threads, 0, stream>>>(query, key, value, output, query_rows,
                                      valid_kv_rows, query_heads, scale_log2);
  } else {
    constexpr int query_tiles_per_head = 1;
    constexpr int heads_per_group = 1;
    constexpr int query_tile =
        query_tiles_per_head * kEncoderWaveQueryTile;
    constexpr int threads = query_tiles_per_head * heads_per_group *
                            kEncoderWavesPerQueryTile *
                            FLASHRT_RDNA_WAVE_SIZE;
    const dim3 grid((query_rows + query_tile - 1) / query_tile, query_heads);
    attention_encoder_gqa_rdna_kernel<query_tiles_per_head, heads_per_group,
                                      true>
        <<<grid, threads, 0, stream>>>(query, key, value, output, query_rows,
                                      valid_kv_rows, query_heads, scale_log2);
  }
}
