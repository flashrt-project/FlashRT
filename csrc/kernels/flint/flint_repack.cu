// SPDX-License-Identifier: Apache-2.0
//
// Lossless GGUF NVFP4 -> Flint runtime layout.
//
// Input: ggml block_nvfp4 superblocks [rows][K/64][36] (4 UE4M3 scale bytes, then
// 32 code bytes; inside each 16-value sub-block byte j holds value j (low nibble)
// and value j+8 (high nibble)). Output: codes [*, K/2] with value 2i in the low
// nibble of byte i, and scales [*, K/16] copied verbatim. Source row r lands at
// destination row (r / group_rows) * group_stride + row_offset + r % group_rows,
// which places separate GGUF tensors into one fused matrix (q|k|v, gate|up,
// per-expert gate|up) without changing a single code or scale bit.

#include "flint_kernels.cuh"

namespace flash_rt {
namespace flint {

__global__ void repack_nvfp4_kernel(const uint8_t* __restrict__ src, long long rows, int nb,
                                    uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
                                    long long group_rows, long long group_stride, long long row_offset) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= rows * nb) return;
  const long long r = idx / nb;
  const int b = static_cast<int>(idx % nb);
  const long long dr = (r / group_rows) * group_stride + row_offset + r % group_rows;
  const uint8_t* blk = src + idx * 36;
  uint32_t qs[8];
  const uint32_t* q32 = reinterpret_cast<const uint32_t*>(blk + 4);  // 36-byte blocks are 4-byte aligned
#pragma unroll
  for (int i = 0; i < 8; ++i) qs[i] = q32[i];
  const uint32_t sc = *reinterpret_cast<const uint32_t*>(blk);
  uint8_t out[32];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    uint8_t q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) q[j] = static_cast<uint8_t>(qs[2 * s + (j >> 2)] >> (8 * (j & 3)));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[8 * s + i] = static_cast<uint8_t>((q[2 * i] & 0x0f) | ((q[2 * i + 1] & 0x0f) << 4));
      out[8 * s + 4 + i] = static_cast<uint8_t>((q[2 * i] >> 4) | (q[2 * i + 1] & 0xf0));
    }
  }
  uint4* dc = reinterpret_cast<uint4*>(codes + dr * (nb * 32LL) + b * 32LL);
  dc[0] = *reinterpret_cast<uint4*>(out);
  dc[1] = *reinterpret_cast<uint4*>(out + 16);
  *reinterpret_cast<uint32_t*>(scales + dr * (nb * 4LL) + b * 4LL) = sc;
}

void repack_nvfp4(const void* src, long long rows, int K, void* codes, void* scales, long long group_rows,
                  long long group_stride, long long row_offset, cudaStream_t s) {
  const int nb = K / 64;
  const long long total = rows * nb;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  repack_nvfp4_kernel<<<static_cast<unsigned>(blocks), threads, 0, s>>>(
      static_cast<const uint8_t*>(src), rows, nb, static_cast<uint8_t*>(codes),
      static_cast<uint8_t*>(scales), group_rows, group_stride, row_offset);
}

}  // namespace flint
}  // namespace flash_rt
