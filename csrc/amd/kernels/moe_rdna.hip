// ================================================================
// FlashRT AMD -- BF16 routed MoE for RDNA wave32.
//
// Routes are packed in token order. Gate/up dots accumulate in FP32 over
// increasing K tiles, and the activated, weighted intermediate rounds to
// BF16 before the down dot.
// FP32 down results are summed in routing-slot order without atomics.
// Workspace and weights are owned by the Python provider before capture.
// ================================================================

#include "common_hip_rdna.h"
#include "wmma_rdna.h"
#include <stdexcept>

namespace {

// ── Sigmoid, correction-bias Top-k and BF16 route normalization ──

template <int TopK>
__global__ __launch_bounds__(128) void moe_topk_rdna_kernel(
    const float *__restrict__ logits, const float *__restrict__ bias,
    __hip_bfloat16 *__restrict__ routing, int *__restrict__ selected, int rows,
    int experts, float scale) {
  const int row = blockIdx.x * 4 + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  if (row >= rows)
    return;
  float score =
      lane < experts
          ? 1.0f / (1.0f + rdna_exp_f32(-logits[row * experts + lane]))
          : 0.0f;
  float corrected = lane < experts ? score + bias[lane] : -INFINITY;
  int index = lane;
#pragma unroll
  for (int size = 2; size <= 32; size *= 2) {
#pragma unroll
    for (int distance = size / 2; distance > 0; distance /= 2) {
      const float other = __shfl_xor(corrected, distance, 32);
      const float other_score = __shfl_xor(score, distance, 32);
      const int other_index = __shfl_xor(index, distance, 32);
      const bool higher =
          other > corrected || (other == corrected && other_index < index);
      const bool descending = ((lane & size) == 0) == ((lane & distance) == 0);
      if (higher == descending) {
        corrected = other;
        score = other_score;
        index = other_index;
      }
    }
  }
  // Match the reference's final Top-k bitonic pass when correction scores tie.
  const float next_score = __shfl_down(corrected, 1, 32);
  const bool tied = lane + 1 < TopK && corrected == next_score;
  if (__ballot(tied)) {
    bool valid = lane < TopK;
    if (!valid)
      corrected = 0.0f;
#pragma unroll
    for (int size = 2; size <= 32; size *= 2) {
#pragma unroll
      for (int distance = size / 2; distance > 0; distance /= 2) {
        const float other = __shfl_xor(corrected, distance, 32);
        const float other_score = __shfl_xor(score, distance, 32);
        const int other_index = __shfl_xor(index, distance, 32);
        const bool other_valid =
            __shfl_xor(static_cast<int>(valid), distance, 32);
        const bool first = (lane & distance) == 0;
        const bool swap = ((first ? corrected > other : other > corrected) &&
                           (first ? valid : other_valid)) ||
                          !(first ? other_valid : valid);
        const int pair = lane / (2 * distance) * distance + lane % distance;
        const bool direction = size < 32 && (pair & (size / 2));
        if (swap == direction) {
          corrected = other;
          score = other_score;
          index = other_index;
          valid = other_valid;
        }
      }
    }
  }
  float sum = lane < TopK ? score : 0.0f;
#pragma unroll
  for (int offset = 1; offset < TopK; offset *= 2)
    sum += __shfl_down(sum, offset, 32);
  const float denominator = __shfl(sum, 0, 32) + 1e-20f;
  if (lane < TopK) {
    selected[row * TopK + lane] = index;
    routing[row * TopK + lane] =
        __float2bfloat16((score / denominator) * scale);
  }
}

// ── Deterministic per-expert route packing ──

__global__ __launch_bounds__(32) void moe_pack_rdna_kernel(
    const int *__restrict__ selected, int *__restrict__ counts,
    int *__restrict__ rows, int *__restrict__ slots, int tokens, int top_k) {
  const int expert = blockIdx.x;
  const int lane = threadIdx.x;
  int count = 0;
  for (int start = 0; start < tokens * top_k; start += 32) {
    const int route = start + lane;
    const bool chosen = route < tokens * top_k && selected[route] == expert;
    const unsigned long long mask = __ballot(chosen);
    if (chosen) {
      const int position = count + __popcll(mask & ((1ull << lane) - 1));
      rows[expert * tokens + position] = route / top_k;
      slots[expert * tokens + position] = route % top_k;
    }
    count += __popcll(mask);
  }
  if (lane == 0)
    counts[expert] = count;
}

// ── Cooperative staging of routed input and expert weight tiles ──

template <int Rows, int Columns, bool GateUp>
__device__ __forceinline__ void
stage_moe_rdna(const __hip_bfloat16 *__restrict__ input,
               const __hip_bfloat16 *__restrict__ weight,
               const __hip_bfloat16 *__restrict__ up_weight, __bf16 *input_lds,
               __bf16 *weight_lds, __bf16 *up_lds, const int *rows,
               const int *slots, int count, int row_start, int column_start,
               int columns, int inner, int inner_start, int top_k) {
  constexpr int pitch = 68;
  constexpr int weight_tiles = GateUp ? 2 : 1;
  constexpr int chunks = (Rows + weight_tiles * Columns) * 8;
  for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
    const int tile_row = chunk / 8;
    const int k = (chunk % 8) * 8;
    uint4 value = make_uint4(0, 0, 0, 0);
    __bf16 *destination;
    if (tile_row < Rows) {
      const int route = row_start + tile_row;
      if (route < count && inner_start + k < inner) {
        const int row = rows[route];
        const int slot = GateUp ? 0 : slots[route];
        const size_t offset =
            static_cast<size_t>(row * (GateUp ? 1 : top_k) + slot) * inner;
        value =
            *reinterpret_cast<const uint4 *>(input + offset + inner_start + k);
      }
      destination = input_lds + tile_row * pitch + k;
    } else {
      const int weight_row = (tile_row - Rows) % Columns;
      const bool is_up = tile_row >= Rows + Columns;
      const __hip_bfloat16 *source = is_up ? up_weight : weight;
      if (column_start + weight_row < columns && inner_start + k < inner) {
        value = *reinterpret_cast<const uint4 *>(
            source + static_cast<size_t>(column_start + weight_row) * inner +
            inner_start + k);
      }
      destination = (is_up ? up_lds : weight_lds) + weight_row * pitch + k;
    }
    reinterpret_cast<uint2 *>(destination)[0] = make_uint2(value.x, value.y);
    reinterpret_cast<uint2 *>(destination)[1] = make_uint2(value.z, value.w);
  }
}

// ── FP32 gate/up with one BF16 intermediate store ──

__global__ __launch_bounds__(32) void moe_gate_up_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const __hip_bfloat16 *__restrict__ gate,
    const __hip_bfloat16 *__restrict__ up,
    const __hip_bfloat16 *__restrict__ routing, const int *__restrict__ counts,
    const int *__restrict__ rows, const int *__restrict__ slots,
    __hip_bfloat16 *__restrict__ intermediate, int tokens, int hidden,
    int width, int top_k) {
  const int expert = blockIdx.z;
  const int start = blockIdx.y * 16;
  const int count = counts[expert];
  if (start >= count)
    return;
  constexpr int pitch = 68;
  __shared__ __align__(16) __bf16 a_lds[16 * pitch];
  __shared__ __align__(16) __bf16 g_lds[16 * pitch];
  __shared__ __align__(16) __bf16 u_lds[16 * pitch];
  const int lane = threadIdx.x & 31;
  const int column = blockIdx.x * 16 + (lane & 15);
  RdnaWmmaAccumulator g_total = {}, u_total = {};
  rows += expert * tokens;
  slots += expert * tokens;
  gate += static_cast<size_t>(expert) * width * hidden;
  up += static_cast<size_t>(expert) * width * hidden;
  for (int k = 0; k < hidden; k += 64) {
    stage_moe_rdna<16, 16, true>(input, gate, up, a_lds, g_lds, u_lds, rows,
                                 slots, count, start, blockIdx.x * 16, width,
                                 hidden, k, top_k);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 64; kk += 16) {
      const auto a =
          rdna_wmma_load_fragment(a_lds + (lane & 15) * pitch + kk, lane);
      const auto g =
          rdna_wmma_load_fragment(g_lds + (lane & 15) * pitch + kk, lane);
      const auto u =
          rdna_wmma_load_fragment(u_lds + (lane & 15) * pitch + kk, lane);
      g_total = rdna_wmma_bf16(a, g, g_total);
      u_total = rdna_wmma_bf16(a, u, u_total);
    }
    __syncthreads();
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int position = start + rdna_wmma_accumulator_row(lane, e);
    if (position < count && column < width) {
      const int row = rows[position], slot = slots[position];
      const float route = __bfloat162float(routing[row * top_k + slot]);
      const float exponential = exp2f(-g_total[e] * 1.4426950408889634f);
      const float activated = g_total[e] * (1.0f / (1.0f + exponential));
      intermediate[(static_cast<size_t>(row) * top_k + slot) * width + column] =
          __float2bfloat16(activated * u_total[e] * route);
    }
  }
}

// ── Routed down projection with deterministic per-slot output ──

template <int Columns>
__global__ __launch_bounds__(64) void moe_down_rdna_kernel(
    const __hip_bfloat16 *__restrict__ intermediate,
    const __hip_bfloat16 *__restrict__ down, const int *__restrict__ counts,
    const int *__restrict__ rows, const int *__restrict__ slots,
    float *__restrict__ output, int tokens, int hidden, int width, int top_k) {
  const int expert = blockIdx.z;
  const int start = blockIdx.y * 32;
  const int count = counts[expert];
  if (start >= count)
    return;
  constexpr int pitch = 68;
  __shared__ __align__(16) __bf16 a_lds[32 * pitch];
  __shared__ __align__(16) __bf16 w_lds[Columns * pitch];
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x / 32;
  rows += expert * tokens;
  slots += expert * tokens;
  down += static_cast<size_t>(expert) * hidden * width;
  RdnaWmmaAccumulator totals[Columns / 16] = {};
  for (int k = 0; k < width; k += 64) {
    stage_moe_rdna<32, Columns, false>(
        intermediate, down, nullptr, a_lds, w_lds, nullptr, rows, slots, count,
        start, blockIdx.x * Columns, hidden, width, k, top_k);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 64; kk += 16) {
      const auto a = rdna_wmma_load_fragment(
          a_lds + (wave * 16 + (lane & 15)) * pitch + kk, lane);
#pragma unroll
      for (int n = 0; n < Columns / 16; ++n) {
        const auto w = rdna_wmma_load_fragment(
            w_lds + (n * 16 + (lane & 15)) * pitch + kk, lane);
        totals[n] = rdna_wmma_bf16(a, w, totals[n]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int n = 0; n < Columns / 16; ++n) {
    const int column = blockIdx.x * Columns + n * 16 + (lane & 15);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int position =
          start + wave * 16 + rdna_wmma_accumulator_row(lane, e);
      if (position < count && column < hidden) {
        const int row = rows[position], slot = slots[position];
        output[(static_cast<size_t>(row) * top_k + slot) * hidden + column] =
            totals[n][e];
      }
    }
  }
}

// ── Packed gate/up and down tiles loaded directly into WMMA registers ──

// Gate/up tiles interleave their 16-element fragments; down tiles use N16/K16.
// Alternating fragment sets overlap loads while keeping increasing-K
// accumulation.
__device__ __forceinline__ RdnaWmmaFragment
moe_load_packed_rdna(const __hip_bfloat16 *__restrict__ pointer) {
  return *reinterpret_cast<const RdnaWmmaFragment *>(
      pointer + rdna_wmma_lane_k(threadIdx.x & 31));
}

// ── Lossless weight tiles with shared exponent bits ──

using MoeWeightWords = unsigned int __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned int
moe_permute_bytes_rdna(unsigned int a, unsigned int b, unsigned int selector) {
  return __builtin_amdgcn_perm(b, a, selector);
}

__device__ __forceinline__ RdnaWmmaFragment
moe_decode_lossless_rdna(const unsigned int *words, unsigned int base) {
  const unsigned int exponent = base * 0x01010101;
  MoeWeightWords decoded = {};
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const unsigned int n = words[4 + group], o = n >> 4;
    const unsigned int even =
        (n & 0x07070707) | ((n & 0x08080808) << 4) | exponent;
    const unsigned int odd =
        (o & 0x07070707) | ((o & 0x08080808) << 4) | exponent;
    const unsigned int high0 = moe_permute_bytes_rdna(even, odd, 0x05010400);
    const unsigned int high1 = moe_permute_bytes_rdna(even, odd, 0x07030602);
    decoded[group * 4] =
        moe_permute_bytes_rdna(words[group * 2], high0, 0x05010400);
    decoded[group * 4 + 1] =
        moe_permute_bytes_rdna(words[group * 2], high0, 0x07030602);
    decoded[group * 4 + 2] =
        moe_permute_bytes_rdna(words[group * 2 + 1], high1, 0x05010400);
    decoded[group * 4 + 3] =
        moe_permute_bytes_rdna(words[group * 2 + 1], high1, 0x07030602);
  }
#if defined(FLASHRT_AMD_RDNA4)
  using FullFragment = __bf16 __attribute__((ext_vector_type(16)));
  const auto full = __builtin_bit_cast(FullFragment, decoded);
  return (threadIdx.x & 16) ? full.hi : full.lo;
#else
  return __builtin_bit_cast(RdnaWmmaFragment, decoded);
#endif
}

__device__ __forceinline__ void
moe_load_lossless_rdna(const __hip_bfloat16 *raw, const unsigned int *encoded,
                       unsigned int base, RdnaWmmaFragment &gate,
                       RdnaWmmaFragment &up) {
  if (base == 0) {
    gate = moe_load_packed_rdna(raw);
    up = moe_load_packed_rdna(raw + 16);
    return;
  }
  // Each lane's aligned 48-byte span contains both gate and up fragments.
  const unsigned int *source = encoded;
  const uint4 a = *reinterpret_cast<const uint4 *>(source);
  const uint4 b = *reinterpret_cast<const uint4 *>(source + 4);
  const uint4 c = *reinterpret_cast<const uint4 *>(source + 8);
  const unsigned int gate_words[6] = {a.x, a.y, a.z, a.w, b.x, b.y};
  const unsigned int up_words[6] = {b.z, b.w, c.x, c.y, c.z, c.w};
  gate = moe_decode_lossless_rdna(gate_words, base);
  up = moe_decode_lossless_rdna(up_words, base);
}

// ── Compact tiles combine encoded weights and original BF16 fallback ──

template <int Elements>
__global__ __launch_bounds__(32) void moe_pack_compact_rdna_kernel(
    const __hip_bfloat16 *raw, unsigned char *blob,
    const unsigned int *metadata) {
  const int lane = threadIdx.x;
  if (lane >= Elements / 16)
    return;
  const unsigned int descriptor = metadata[blockIdx.x], base = descriptor >> 24;
  auto target = blob + static_cast<size_t>(descriptor & 0xffffffu) * 128;
  const auto words = *reinterpret_cast<const MoeWeightWords *>(
      raw + static_cast<size_t>(blockIdx.x) * Elements + lane * 16);
  if (base == 0) {
    *reinterpret_cast<MoeWeightWords *>(target + lane * 32) = words;
    return;
  }
  unsigned int lows[4] = {}, highs[2] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned int pair = (words[i] & 255) | ((words[i] >> 8) & 65280);
    lows[i / 2] |= pair << ((i % 2) * 16);
    const unsigned int a = ((words[i] >> 8) & 7) | ((words[i] >> 12) & 8);
    const unsigned int b = ((words[i] >> 24) & 7) | ((words[i] >> 28) & 8);
    highs[i / 4] |= (a | (b << 4)) << ((i % 4) * 8);
  }
  auto output = reinterpret_cast<uint2 *>(target + lane * 24);
  output[0] = make_uint2(lows[0], lows[1]);
  output[1] = make_uint2(lows[2], lows[3]);
  output[2] = make_uint2(highs[0], highs[1]);
}

template <bool Gate>
__device__ __forceinline__ void
moe_load_compact_rdna(const unsigned char *blob, unsigned int descriptor,
                      RdnaWmmaFragment &b, RdnaWmmaFragment &u) {
  const int lane = threadIdx.x & 15;
  const unsigned int base = descriptor >> 24;
  const auto source = blob + static_cast<size_t>(descriptor & 0xffffffu) * 128;
  const auto raw = reinterpret_cast<const __hip_bfloat16 *>(source);
  const auto encoded = reinterpret_cast<const unsigned int *>(source);
  if constexpr (Gate) {
    moe_load_lossless_rdna(raw + lane * 32, encoded + lane * 12, base, b, u);
  } else {
    if (base == 0) {
      b = moe_load_packed_rdna(raw + lane * 16);
      return;
    }
    const auto words = encoded + lane * 6;
    const uint2 a = *reinterpret_cast<const uint2 *>(words);
    const uint2 c = *reinterpret_cast<const uint2 *>(words + 2);
    const uint2 d = *reinterpret_cast<const uint2 *>(words + 4);
    const unsigned int values[6] = {a.x, a.y, c.x, c.y, d.x, d.y};
    b = moe_decode_lossless_rdna(values, base);
  }
}

template <bool Gate, bool CachedHeaders = false, bool Compact = false>
__global__ __launch_bounds__(Gate ? 32 : 64) void moe_packed_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const __hip_bfloat16 *__restrict__ weight,
    const __hip_bfloat16 *__restrict__ routing, const unsigned int *metadata,
    const int *__restrict__ counts, const int *__restrict__ rows,
    const int *__restrict__ slots, __hip_bfloat16 *inter, float *out,
    int tokens, int hidden, int width, int top_k) {
  const int expert = blockIdx.z, lane = threadIdx.x & 31,
            wave = threadIdx.x / 32;
  const int start = blockIdx.y * (Gate ? 16 : 32), count = counts[expert];
  if (start + wave * 16 >= count)
    return;
  rows += expert * tokens;
  slots += expert * tokens;
  const int position = start + wave * 16 + (lane & 15);
  const int row = position < count ? rows[position] : 0;
  const int slot = Gate || position >= count ? 0 : slots[position];
  const int inner = Gate ? hidden : width;
  input += static_cast<size_t>(row * (Gate ? 1 : top_k) + slot) * inner;
  const int column = blockIdx.x * 16 + (lane & 15);
  const size_t tile =
      static_cast<size_t>(expert) * (Gate ? width : hidden) * inner +
      static_cast<size_t>(blockIdx.x) * inner * 16 + (lane & 15) * 16;
  if constexpr (Compact) {
    const size_t tile_base =
        (static_cast<size_t>(expert) * ((Gate ? width : hidden) / 16) +
         blockIdx.x) *
        (inner / 16);
    metadata += tile_base;
  } else {
    weight += tile * (Gate ? 2 : 1);
  }
  // Cache descriptors for each 64-element K span.
  uint4 descriptors = {};
  auto read_headers = [&](int k) {
    descriptors.x = __builtin_amdgcn_readfirstlane(metadata[k / 16]);
    descriptors.y = __builtin_amdgcn_readfirstlane(metadata[k / 16 + 1]);
    descriptors.z = __builtin_amdgcn_readfirstlane(metadata[k / 16 + 2]);
    descriptors.w = __builtin_amdgcn_readfirstlane(metadata[k / 16 + 3]);
  };
  if constexpr (CachedHeaders)
    read_headers(0);
  auto read_descriptor = [&](int k) {
    if constexpr (!CachedHeaders)
      return static_cast<unsigned int>(metadata[k / 16]);
    else {
      switch ((k & 63) / 16) {
      case 0:
        return descriptors.x;
      case 1:
        return descriptors.y;
      case 2:
        return descriptors.z;
      default:
        return descriptors.w;
      }
    }
  };
  auto load_weights = [&](int k, RdnaWmmaFragment &b, RdnaWmmaFragment &u) {
    if constexpr (Compact)
      moe_load_compact_rdna<Gate>(
          reinterpret_cast<const unsigned char *>(weight), read_descriptor(k),
          b, u);
    else {
      b = moe_load_packed_rdna(weight + k * (Gate ? 32 : 16));
      if constexpr (Gate)
        u = moe_load_packed_rdna(weight + k * 32 + 16);
    }
  };
  RdnaWmmaAccumulator total = {}, second = {};
  auto a0 = moe_load_packed_rdna(input);
  RdnaWmmaFragment b0 = {}, u0 = {}, u1 = {};
  load_weights(0, b0, u0);
  for (int k = 0; k + 32 < inner; k += 32) {
    auto a1 = moe_load_packed_rdna(input + k + 16);
    RdnaWmmaFragment b1 = {};
    load_weights(k + 16, b1, u1);
    if constexpr (CachedHeaders) {
      if ((k + 32) % 64 == 0)
        read_headers(k + 32);
    }
    asm volatile("" ::: "memory");
    total = rdna_wmma_bf16(a0, b0, total);
    if constexpr (Gate)
      second = rdna_wmma_bf16(a0, u0, second);
    a0 = moe_load_packed_rdna(input + k + 32);
    load_weights(k + 32, b0, u0);
    asm volatile("" ::: "memory");
    total = rdna_wmma_bf16(a1, b1, total);
    if constexpr (Gate)
      second = rdna_wmma_bf16(a1, u1, second);
  }
  auto a1 = moe_load_packed_rdna(input + inner - 16);
  RdnaWmmaFragment b1 = {};
  load_weights(inner - 16, b1, u1);
  asm volatile("" ::: "memory");
  total = rdna_wmma_bf16(a0, b0, total);
  if constexpr (Gate)
    second = rdna_wmma_bf16(a0, u0, second);
  asm volatile("" ::: "memory");
  total = rdna_wmma_bf16(a1, b1, total);
  if constexpr (Gate)
    second = rdna_wmma_bf16(a1, u1, second);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int p = start + wave * 16 + rdna_wmma_accumulator_row(lane, e);
    if (p < count) {
      const int token = rows[p], s = slots[p];
      if constexpr (Gate) {
        const float route = __bfloat162float(routing[token * top_k + s]);
        const float exponential = exp2f(-total[e] * 1.4426950408889634f);
        const float activated = total[e] * (1.0f / (1.0f + exponential));
        inter[(static_cast<size_t>(token) * top_k + s) * width + column] =
            __float2bfloat16(activated * second[e] * route);
      } else {
        out[(static_cast<size_t>(token) * top_k + s) * hidden + column] =
            total[e];
      }
    }
  }
}

// ── Two-column-tile down projection with shared input fragments ──

__global__ __launch_bounds__(32) void moe_down_compact_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const unsigned char *__restrict__ blob,
    const unsigned int *__restrict__ metadata, const int *__restrict__ counts,
    const int *__restrict__ rows, const int *__restrict__ slots,
    float *__restrict__ out, int tokens, int hidden, int width, int top_k) {
  const int expert = blockIdx.z, lane = threadIdx.x;
  const int start = blockIdx.y * 16, count = counts[expert];
  if (start >= count)
    return;
  rows += expert * tokens;
  slots += expert * tokens;
  const int position = start + (lane & 15);
  const int row = position < count ? rows[position] : 0;
  const int slot = position < count ? slots[position] : 0;
  input += (static_cast<size_t>(row) * top_k + slot) * width;
  metadata += (static_cast<size_t>(expert) * (hidden / 16) + blockIdx.x * 2) *
              (width / 16);
  RdnaWmmaAccumulator total[2] = {};
  uint4 descriptors[2];
  for (int k = 0; k < width; k += 16) {
    if ((k & 63) == 0) {
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const unsigned int *source = metadata + n * (width / 16) + k / 16;
        descriptors[n].x = __builtin_amdgcn_readfirstlane(source[0]);
        descriptors[n].y = __builtin_amdgcn_readfirstlane(source[1]);
        descriptors[n].z = __builtin_amdgcn_readfirstlane(source[2]);
        descriptors[n].w = __builtin_amdgcn_readfirstlane(source[3]);
      }
    }
    const auto a = moe_load_packed_rdna(input + k);
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      unsigned int descriptor;
      switch ((k & 63) / 16) {
      case 0:
        descriptor = descriptors[n].x;
        break;
      case 1:
        descriptor = descriptors[n].y;
        break;
      case 2:
        descriptor = descriptors[n].z;
        break;
      default:
        descriptor = descriptors[n].w;
        break;
      }
      RdnaWmmaFragment b = {}, unused = {};
      moe_load_compact_rdna<false>(blob, descriptor, b, unused);
      total[n] = rdna_wmma_bf16(a, b, total[n]);
    }
  }
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int column = blockIdx.x * 32 + n * 16 + (lane & 15);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int p = start + rdna_wmma_accumulator_row(lane, e);
      if (p < count) {
        const int token = rows[p], s = slots[p];
        out[(static_cast<size_t>(token) * top_k + s) * hidden + column] =
            total[n][e];
      }
    }
  }
}

__global__ void moe_sum_rdna_kernel(const float *__restrict__ routes,
                                    float *__restrict__ output, int size,
                                    int hidden, int top_k) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const int row = index / hidden, column = index % hidden;
    float sum = 0.0f;
    for (int slot = 0; slot < top_k; ++slot)
      sum +=
          routes[(static_cast<size_t>(row) * top_k + slot) * hidden + column];
    output[index] = sum;
  }
}

} // namespace

void moe_forward_rdna(const __hip_bfloat16 *input,
                      const __hip_bfloat16 *routing, const int *selected,
                      const __hip_bfloat16 *gate, const __hip_bfloat16 *up,
                      const __hip_bfloat16 *down, int *counts, int *rows,
                      int *slots, __hip_bfloat16 *intermediate,
                      float *routes_output, float *output, int tokens,
                      int hidden, int experts, int top_k, int width,
                      hipStream_t stream) {
  if (tokens < 1 || hidden < 1 || width < 1 || top_k < 1 || top_k > experts ||
      hidden % 16 || width % 16)
    throw std::invalid_argument(
        "RDNA MoE requires positive dimensions, distinct top-k and 16-aligned "
        "hidden/intermediate widths");
  moe_pack_rdna_kernel<<<experts, 32, 0, stream>>>(selected, counts, rows,
                                                   slots, tokens, top_k);
  moe_gate_up_rdna_kernel<<<
      dim3((width + 15) / 16, (tokens + 15) / 16, experts), 32, 0, stream>>>(
      input, gate, up, routing, counts, rows, slots, intermediate, tokens,
      hidden, width, top_k);
#if defined(FLASHRT_AMD_RDNA3) || defined(FLASHRT_AMD_RDNA4)
  constexpr int down_columns = 32;
#else
  // Narrower output tiles improve occupancy on gfx1151.
  constexpr int down_columns = 16;
#endif
  moe_down_rdna_kernel<down_columns>
      <<<dim3((hidden + down_columns - 1) / down_columns, (tokens + 31) / 32,
              experts),
         64, 0, stream>>>(intermediate, down, counts, rows, slots,
                          routes_output, tokens, hidden, width, top_k);
  moe_sum_rdna_kernel<<<(tokens * hidden + 255) / 256, 256, 0, stream>>>(
      routes_output, output, tokens * hidden, hidden, top_k);
}

void moe_forward_packed_rdna(const __hip_bfloat16 *__restrict__ input,
                             const __hip_bfloat16 *__restrict__ route,
                             const int *__restrict__ selected,
                             const __hip_bfloat16 *__restrict__ gate_up,
                             const __hip_bfloat16 *__restrict__ down,
                             int *counts, int *rows, int *slots,
                             __hip_bfloat16 *inter, float *routes_out,
                             float *out, int tokens, int hidden, int experts,
                             int top_k, int width, hipStream_t stream) {
  if (hidden % 32 || width % 32)
    throw std::invalid_argument(
        "RDNA packed MoE requires 32-aligned hidden/intermediate widths");
  moe_pack_rdna_kernel<<<experts, 32, 0, stream>>>(selected, counts, rows,
                                                   slots, tokens, top_k);
  moe_packed_rdna_kernel<true>
      <<<dim3(width / 16, (tokens + 15) / 16, experts), 32, 0, stream>>>(
          input, gate_up, route, nullptr, counts, rows, slots, inter, nullptr,
          tokens, hidden, width, top_k);
  moe_packed_rdna_kernel<false>
      <<<dim3(hidden / 16, (tokens + 31) / 32, experts), 64, 0, stream>>>(
          inter, down, nullptr, nullptr, counts, rows, slots, nullptr,
          routes_out, tokens, hidden, width, top_k);
  moe_sum_rdna_kernel<<<(tokens * hidden + 255) / 256, 256, 0, stream>>>(
      routes_out, out, tokens * hidden, hidden, top_k);
}

void moe_pack_compact_rdna(const __hip_bfloat16 *raw, unsigned char *blob,
                           const unsigned int *metadata, int tiles, bool gate,
                           hipStream_t stream) {
  if (gate)
    moe_pack_compact_rdna_kernel<512>
        <<<tiles, 32, 0, stream>>>(raw, blob, metadata);
  else
    moe_pack_compact_rdna_kernel<256>
        <<<tiles, 32, 0, stream>>>(raw, blob, metadata);
}

void moe_forward_compact_rdna(
    const __hip_bfloat16 *input, const __hip_bfloat16 *route,
    const int *selected, const unsigned char *gate_blob,
    const unsigned char *down_blob, const unsigned int *gate_metadata,
    const unsigned int *down_metadata, int *counts, int *rows, int *slots,
    __hip_bfloat16 *inter, float *routes_out, float *out, int tokens,
    int hidden, int experts, int top_k, int width, hipStream_t stream) {
  if (hidden % 32 || width % 32)
    throw std::invalid_argument(
        "RDNA packed MoE requires 32-aligned hidden/intermediate widths");
  moe_pack_rdna_kernel<<<experts, 32, 0, stream>>>(selected, counts, rows,
                                                   slots, tokens, top_k);
  const auto gate = reinterpret_cast<const __hip_bfloat16 *>(gate_blob);
  const auto down = reinterpret_cast<const __hip_bfloat16 *>(down_blob);
  if (hidden % 64 == 0)
    moe_packed_rdna_kernel<true, true, true>
        <<<dim3(width / 16, (tokens + 15) / 16, experts), 32, 0, stream>>>(
            input, gate, route, gate_metadata, counts, rows, slots, inter,
            nullptr, tokens, hidden, width, top_k);
  else
    moe_packed_rdna_kernel<true, false, true>
        <<<dim3(width / 16, (tokens + 15) / 16, experts), 32, 0, stream>>>(
            input, gate, route, gate_metadata, counts, rows, slots, inter,
            nullptr, tokens, hidden, width, top_k);
  if (width % 64 == 0) {
#if !defined(FLASHRT_AMD_RDNA3) && !defined(FLASHRT_AMD_RDNA4)
    moe_down_compact_rdna_kernel<<<
        dim3(hidden / 32, (tokens + 15) / 16, experts), 32, 0, stream>>>(
        inter, down_blob, down_metadata, counts, rows, slots, routes_out,
        tokens, hidden, width, top_k);
#else
    moe_packed_rdna_kernel<false, true, true>
        <<<dim3(hidden / 16, (tokens + 31) / 32, experts), 64, 0, stream>>>(
            inter, down, nullptr, down_metadata, counts, rows, slots, nullptr,
            routes_out, tokens, hidden, width, top_k);
#endif
  } else
    moe_packed_rdna_kernel<false, false, true>
        <<<dim3(hidden / 16, (tokens + 31) / 32, experts), 64, 0, stream>>>(
            inter, down, nullptr, down_metadata, counts, rows, slots, nullptr,
            routes_out, tokens, hidden, width, top_k);
  moe_sum_rdna_kernel<<<(tokens * hidden + 255) / 256, 256, 0, stream>>>(
      routes_out, out, tokens * hidden, hidden, top_k);
}

void moe_topk_rdna(const float *logits, const float *bias,
                   __hip_bfloat16 *routing, int *selected, int rows,
                   int experts, int top_k, float scale, hipStream_t stream) {
  if (experts < 1 || experts > 32 || top_k > experts)
    throw std::invalid_argument("RDNA fused routing supports up to 32 experts");
#define FLASHRT_LAUNCH_TOPK(k)                                                 \
  moe_topk_rdna_kernel<k><<<(rows + 3) / 4, 128, 0, stream>>>(                 \
      logits, bias, routing, selected, rows, experts, scale)
  switch (top_k) {
  case 1:
    FLASHRT_LAUNCH_TOPK(1);
    break;
  case 2:
    FLASHRT_LAUNCH_TOPK(2);
    break;
  case 4:
    FLASHRT_LAUNCH_TOPK(4);
    break;
  case 8:
    FLASHRT_LAUNCH_TOPK(8);
    break;
  default:
    throw std::invalid_argument("RDNA fused routing requires Top-k 1/2/4/8");
  }
#undef FLASHRT_LAUNCH_TOPK
}
