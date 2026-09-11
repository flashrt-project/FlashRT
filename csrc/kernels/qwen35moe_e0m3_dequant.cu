// SPDX-License-Identifier: Apache-2.0
//
// Streamed routed-expert block to bf16. See header.

#include "kernels/qwen35moe_e0m3_dequant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kThreads = 256;

// Sign-magnitude: low three bits are the magnitude, bit 3 the sign. Kept as a
// signed integer because the values are exactly the integers 0..7, which is the
// property the quantizer's group scale is chosen against.
__device__ __forceinline__ float decode_nibble(uint8_t code) {
  const float magnitude = static_cast<float>(code & 0x07u);
  return (code & 0x08u) ? -magnitude : magnitude;
}

// One thread per packed byte: two output values that always share a scale,
// because group_size is even.
__global__ void dequant_kernel(const uint8_t* __restrict__ packed,
                               const uint8_t* __restrict__ scale,
                               __nv_bfloat162* __restrict__ out,
                               int rows, int cols, int group_size,
                               float global_scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pairs_per_row = cols >> 1;
  if (index >= rows * pairs_per_row) return;

  const int row = index / pairs_per_row;
  const int pair = index - row * pairs_per_row;
  const int group = (pair << 1) / group_size;
  const int groups_per_row = cols / group_size;

  const __half_raw raw = __nv_cvt_fp8_to_halfraw(
      scale[row * groups_per_row + group], __NV_E4M3);
  const float step =
      __half2float(*reinterpret_cast<const __half*>(&raw)) * global_scale;

  const uint8_t byte = packed[index];
  out[index] = __floats2bfloat162_rn(
      decode_nibble(byte & 0x0Fu) * step,
      decode_nibble(byte >> 4) * step);
}

}  // namespace

int qwen35moe_e0m3_dequant_bf16(const void* packed, const void* scale,
                                void* out, int rows, int cols,
                                int group_size, float global_scale,
                                cudaStream_t stream) {
  if (!packed || !scale || !out) return 1;
  if (rows <= 0 || cols <= 0) return 2;
  if (cols & 1) return 3;
  if (group_size <= 0 || (group_size & 1)) return 4;
  if (cols % group_size) return 5;

  const long long pairs = static_cast<long long>(rows) * (cols >> 1);
  const long long blocks = (pairs + kThreads - 1) / kThreads;
  if (blocks > 2147483647LL) return 6;

  dequant_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(packed),
      reinterpret_cast<const uint8_t*>(scale),
      reinterpret_cast<__nv_bfloat162*>(out),
      rows, cols, group_size, global_scale);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
