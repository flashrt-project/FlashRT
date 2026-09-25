// ================================================================
// FlashRT AMD -- small-M BF16 WMMA GEMM (RDNA 3.5, wave32)
//
// Computes D[M,N] = A[M,K] @ W_nt[N,K]^T + bias[N], where M <= 48.
// W_nt is a persistent row-major transpose of the checkpoint's (K,N) weight.
// The transposed layout makes each output column's K row contiguous, which is
// preferable to asking a wave32 kernel to gather columns from checkpoint KN.
//
// Two measured implementations cover the decoder projection shapes:
//
//   split-K: one workgroup owns a 16-column tile. Waves traverse disjoint K
//   ranges, accumulate 16x16 WMMA fragments in FP32, write one partial per
//   wave to LDS, and wave 0 performs the fixed-order reduction. This keeps
//   enough workgroups in flight for narrow N, especially the 32-column action
//   projection.
//
//   tiled: one workgroup stages a 16x64 A tile and a Columnsx64 W tile with
//   aligned 16-byte global loads. Each wave owns 16 output columns and reuses
//   the staged A tile, reducing repeated A traffic for wider projections.
//
// The gfx11 WMMA fragment and accumulator mapping follows the public wave32
// description used by Comfy Kitchen (Apache-2.0): every lane carries the full
// 16-element BF16 K fragment, and accumulator row is
// 2 * element + lane / 16. The kernel and dispatch below are FlashRT-specific.
//
// Optional bias is accumulated in FP32. The residual variant first rounds the
// GEMM+bias result to BF16, then adds the existing BF16 output, matching the
// unfused projection followed by residual-add dataflow. No atomics or global
// reduction workspace are used; all launch choices are deterministic.
// ================================================================

#include "../kernels/common_hip_rdna.h"

#include <stdexcept>

namespace {

using Bf16x16Rdna = __bf16 __attribute__((ext_vector_type(16)));
using Floatx8Rdna = float __attribute__((ext_vector_type(8)));

union Bf16FragmentRdna {
  Bf16x16Rdna values;
  unsigned long long words[4];
};

// ── WMMA fragment load and cooperative LDS staging ──

__device__ __forceinline__ Bf16x16Rdna
load_lds_bf16x16_rdna(const __bf16 *source) {
  Bf16FragmentRdna fragment;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    fragment.words[index] =
        reinterpret_cast<const unsigned long long *>(source)[index];
  }
  return fragment.values;
}

template <int Rows, int Columns, int InnerTile, int Threads,
          bool MultipleRowTiles>
__device__ __forceinline__ void
stage_wmma_tile_rdna(const __hip_bfloat16 *__restrict__ input,
                     const __hip_bfloat16 *__restrict__ weight_nt,
                     __bf16 *input_lds, __bf16 *weight_lds, int rows,
                     int columns, int inner, int column_start,
                     int row_start, int inner_start) {
  constexpr int chunks_per_row = InnerTile / 8;
  constexpr int chunks = (Rows + Columns) * chunks_per_row;
  constexpr int input_stride = InnerTile + 4;
  constexpr int weight_stride = InnerTile + 4;
  static_assert(chunks % Threads == 0);

#pragma unroll
  for (int item = 0; item < chunks / Threads; ++item) {
    const int chunk = threadIdx.x + item * Threads;
    const int tile_row = chunk / chunks_per_row;
    const int inner_offset = (chunk % chunks_per_row) * 8;
    const int global_inner = inner_start + inner_offset;
    uint4 loaded = make_uint4(0, 0, 0, 0);
    __bf16 *destination;
    if (tile_row < Rows) {
      const int global_row =
          (MultipleRowTiles ? row_start : 0) + tile_row;
      if (global_row < rows && global_inner < inner) {
        loaded = *reinterpret_cast<const uint4 *>(
            input + static_cast<size_t>(global_row) * inner + global_inner);
      }
      destination = input_lds + tile_row * input_stride + inner_offset;
    } else {
      const int weight_row = tile_row - Rows;
      const int global_column = column_start + weight_row;
      if (global_column < columns && global_inner < inner) {
        loaded = *reinterpret_cast<const uint4 *>(
            weight_nt + static_cast<size_t>(global_column) * inner +
            global_inner);
      }
      destination = weight_lds + weight_row * weight_stride + inner_offset;
    }
    reinterpret_cast<uint2 *>(destination)[0] = make_uint2(loaded.x, loaded.y);
    reinterpret_cast<uint2 *>(destination)[1] = make_uint2(loaded.z, loaded.w);
  }
}

// ── LDS-tiled A/W implementation for wider N ──

template <int Columns, int Waves, bool MultipleRowTiles>
__global__
__launch_bounds__(Waves *FLASHRT_RDNA_WAVE_SIZE) void smallm_wmma_bf16_tiled_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const __hip_bfloat16 *__restrict__ weight_nt,
    const __hip_bfloat16 *__restrict__ bias,
    __hip_bfloat16 *__restrict__ output, int rows, int columns, int inner,
    bool accumulate) {
  constexpr int rows_tile = 16;
  constexpr int inner_tile = 64;
  constexpr int stride = inner_tile + 4;
  constexpr int threads = Waves * FLASHRT_RDNA_WAVE_SIZE;
  __shared__ __align__(16) __bf16 input_lds[rows_tile * stride];
  __shared__ __align__(16) __bf16 weight_lds[Columns * stride];

  const int wave = threadIdx.x / FLASHRT_RDNA_WAVE_SIZE;
  const int lane = threadIdx.x & (FLASHRT_RDNA_WAVE_SIZE - 1);
  const int fragment_row = lane & 15;
  const int row_start = MultipleRowTiles ? blockIdx.y * rows_tile : 0;
  const int column_start = blockIdx.x * Columns;
  const int column = column_start + wave * 16 + fragment_row;
  Floatx8Rdna accumulator = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int inner_start = 0; inner_start < inner; inner_start += inner_tile) {
    stage_wmma_tile_rdna<rows_tile, Columns, inner_tile, threads,
                         MultipleRowTiles>(
        input, weight_nt, input_lds, weight_lds, rows, columns, inner,
        column_start, row_start, inner_start);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < inner_tile; k += 16) {
      const Bf16x16Rdna a =
          load_lds_bf16x16_rdna(input_lds + fragment_row * stride + k);
      const Bf16x16Rdna b = load_lds_bf16x16_rdna(
          weight_lds + (wave * 16 + fragment_row) * stride + k);
      accumulator =
          __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, accumulator);
    }
    __syncthreads();
  }

#pragma unroll
  for (int element = 0; element < 8; ++element) {
    const int row = 2 * element + lane / 16;
    const int global_row = (MultipleRowTiles ? row_start : 0) + row;
    if (global_row < rows && column < columns) {
      float result = accumulator[element];
      if (bias != nullptr) {
        result += __bfloat162float(bias[column]);
      }
      if (accumulate) {
        result = __bfloat162float(__float2bfloat16(result)) +
                 __bfloat162float(
                     output[static_cast<size_t>(global_row) * columns +
                            column]);
      }
      output[static_cast<size_t>(global_row) * columns + column] =
          __float2bfloat16(result);
    }
  }
}

template <int Columns, int Waves, bool MultipleRowTiles>
void launch_smallm_wmma_bf16_tiled_rdna(const __hip_bfloat16 *input,
                                        const __hip_bfloat16 *weight_nt,
                                        const __hip_bfloat16 *bias,
                                        __hip_bfloat16 *output, int rows,
                                        int columns, int inner, bool accumulate,
                                        hipStream_t stream) {
  const dim3 grid((columns + Columns - 1) / Columns,
                  MultipleRowTiles ? (rows + 15) / 16 : 1);
  smallm_wmma_bf16_tiled_rdna_kernel<Columns, Waves, MultipleRowTiles>
      <<<grid, Waves * FLASHRT_RDNA_WAVE_SIZE, 0, stream>>>(
          input, weight_nt, bias, output, rows, columns, inner, accumulate);
}

// ── Direct split-K implementation for narrow N ──

template <int Waves, bool MultipleRowTiles>
__global__
__launch_bounds__(Waves *FLASHRT_RDNA_WAVE_SIZE) void smallm_wmma_bf16_rdna_kernel(
    const __hip_bfloat16 *__restrict__ input,
    const __hip_bfloat16 *__restrict__ weight_nt,
    const __hip_bfloat16 *__restrict__ bias,
    __hip_bfloat16 *__restrict__ output, int rows, int columns, int inner,
    bool accumulate) {
  __shared__ float partial[Waves][FLASHRT_RDNA_WAVE_SIZE][8];

  const int wave = threadIdx.x / FLASHRT_RDNA_WAVE_SIZE;
  const int lane = threadIdx.x & (FLASHRT_RDNA_WAVE_SIZE - 1);
  const int fragment_row = lane & 15;
  const int column = blockIdx.x * 16 + fragment_row;
  const int row_start = MultipleRowTiles ? blockIdx.y * 16 : 0;
  const int segment = inner / Waves;
  const int k_begin = wave * segment;
  const int k_end = k_begin + segment;

  Floatx8Rdna accumulator = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  const int candidate_row = row_start + fragment_row;
  const int input_row = candidate_row < rows ? candidate_row : row_start;
  const __bf16 *input_base = reinterpret_cast<const __bf16 *>(
      input + static_cast<size_t>(input_row) * inner);
  const __bf16 *weight_base = reinterpret_cast<const __bf16 *>(
      weight_nt + static_cast<size_t>(column) * inner);

  for (int k = k_begin; k < k_end; k += 16) {
    const Bf16x16Rdna a =
        *reinterpret_cast<const Bf16x16Rdna *>(input_base + k);
    const Bf16x16Rdna b =
        *reinterpret_cast<const Bf16x16Rdna *>(weight_base + k);
    accumulator =
        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, accumulator);
  }

#pragma unroll
  for (int element = 0; element < 8; ++element) {
    partial[wave][lane][element] = accumulator[element];
  }
  __syncthreads();

  if (wave != 0) {
    return;
  }

#pragma unroll
  for (int element = 0; element < 8; ++element) {
    float result = partial[0][lane][element];
#pragma unroll
    for (int other_wave = 1; other_wave < Waves; ++other_wave) {
      result += partial[other_wave][lane][element];
    }
    const int row = 2 * element + lane / 16;
    const int global_row = (MultipleRowTiles ? row_start : 0) + row;
    if (global_row < rows) {
      if (bias != nullptr) {
        result += __bfloat162float(bias[column]);
      }
      if (accumulate) {
        result = __bfloat162float(__float2bfloat16(result)) +
                 __bfloat162float(
                     output[static_cast<size_t>(global_row) * columns +
                            column]);
      }
      output[static_cast<size_t>(global_row) * columns + column] =
          __float2bfloat16(result);
    }
  }
}

template <int Waves, bool MultipleRowTiles>
void launch_smallm_wmma_bf16_rdna(const __hip_bfloat16 *input,
                                  const __hip_bfloat16 *weight_nt,
                                  const __hip_bfloat16 *bias,
                                  __hip_bfloat16 *output, int rows, int columns,
                                  int inner, bool accumulate,
                                  hipStream_t stream) {
  const dim3 grid(columns / 16,
                  MultipleRowTiles ? (rows + 15) / 16 : 1);
  smallm_wmma_bf16_rdna_kernel<Waves, MultipleRowTiles>
      <<<grid, Waves * FLASHRT_RDNA_WAVE_SIZE, 0, stream>>>(
          input, weight_nt, bias, output, rows, columns, inner, accumulate);
}

template <bool MultipleRowTiles>
void dispatch_smallm_wmma_bf16_rdna(const __hip_bfloat16 *input,
                                    const __hip_bfloat16 *weight_nt,
                                    const __hip_bfloat16 *bias,
                                    __hip_bfloat16 *output, int rows,
                                    int columns, int inner, bool accumulate,
                                    hipStream_t stream) {
  if (inner % 64 == 0 &&
      (columns <= 32 || (inner <= 2048 && columns < 4096) ||
       (inner <= 4096 && columns <= 1024))) {
    launch_smallm_wmma_bf16_rdna<4, MultipleRowTiles>(
        input, weight_nt, bias, output, rows, columns, inner, accumulate,
        stream);
  } else if (inner % 64 == 0 && columns >= 64) {
    launch_smallm_wmma_bf16_tiled_rdna<64, 4, MultipleRowTiles>(
        input, weight_nt, bias, output, rows, columns, inner, accumulate,
        stream);
  } else if (inner % 64 == 0 && columns >= 32) {
    launch_smallm_wmma_bf16_tiled_rdna<32, 2, MultipleRowTiles>(
        input, weight_nt, bias, output, rows, columns, inner, accumulate,
        stream);
  } else if (inner % 32 == 0 && inner >= 256) {
    launch_smallm_wmma_bf16_rdna<2, MultipleRowTiles>(
        input, weight_nt, bias, output, rows, columns, inner, accumulate,
        stream);
  } else {
    launch_smallm_wmma_bf16_rdna<1, MultipleRowTiles>(
        input, weight_nt, bias, output, rows, columns, inner, accumulate,
        stream);
  }
}

} // namespace

// ── Shape-driven host dispatch ──
// Thresholds are benchmark-selected for the fixed Pi0.5 BF16 projection set;
// they are not a generic GEMM autotuner. Unsupported layouts stay on
// hipBLASLt in the Python backend.

void smallm_wmma_bf16_rdna_impl(const __hip_bfloat16 *input,
                                const __hip_bfloat16 *weight_nt,
                                const __hip_bfloat16 *bias,
                                __hip_bfloat16 *output, int rows, int columns,
                                int inner, bool accumulate,
                                hipStream_t stream) {
  if (rows < 1 || rows > 48 || columns < 16 || columns % 16 != 0 ||
      inner < 16 || inner % 16 != 0) {
    throw std::runtime_error(
        "smallm_wmma_bf16_rdna: require M in [1,48] and N/K multiples of 16");
  }
  // The M<=16 specialization keeps row-tile arithmetic out of the current
  // Pi0.5 hot kernel. M>16 uses the same 16-row WMMA tile over grid.y.
  if (rows <= 16) {
    dispatch_smallm_wmma_bf16_rdna<false>(
        input, weight_nt, bias, output, rows, columns, inner, accumulate,
        stream);
  } else {
    dispatch_smallm_wmma_bf16_rdna<true>(
        input, weight_nt, bias, output, rows, columns, inner, accumulate,
        stream);
  }
}

void smallm_wmma_bf16_rdna(const __hip_bfloat16 *input,
                           const __hip_bfloat16 *weight_nt,
                           const __hip_bfloat16 *bias, __hip_bfloat16 *output,
                           int rows, int columns, int inner,
                           hipStream_t stream) {
  smallm_wmma_bf16_rdna_impl(input, weight_nt, bias, output, rows, columns,
                             inner, false, stream);
}

void smallm_wmma_bf16_residual_rdna(const __hip_bfloat16 *input,
                                    const __hip_bfloat16 *weight_nt,
                                    const __hip_bfloat16 *bias,
                                    __hip_bfloat16 *output, int rows,
                                    int columns, int inner,
                                    hipStream_t stream) {
  smallm_wmma_bf16_rdna_impl(input, weight_nt, bias, output, rows, columns,
                             inner, true, stream);
}
