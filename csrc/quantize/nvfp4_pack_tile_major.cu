// SPDX-License-Identifier: Apache-2.0
#include "nvfp4_pack_tile_major.cuh"

namespace flash_rt {
namespace quantize {
namespace {

// One CTA per 4096-byte tile: 64 rows x 64 bytes (a 128-K block at K/2 bytes
// per row). 128 threads x two 16-byte copies.
__global__ void nvfp4_pack_tile_major_kernel(const uint4* __restrict__ src,
                                             uint4* __restrict__ dst,
                                             int k_blocks) {
    const long tile = blockIdx.x;
    const int n_tile = (int)(tile / k_blocks);
    const int kb = (int)(tile - (long)n_tile * k_blocks);

    // Source row stride is K/2 bytes = k_blocks * 4 uint4; this tile's 64-byte
    // segment per row is chunk kb*4 .. kb*4+3.
    const uint4* s = src + (size_t)n_tile * 64 * (k_blocks * 4) + kb * 4;
    uint4* d = dst + tile * 256;

    const int t = threadIdx.x;
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        const int idx = it * 128 + t;        // 0 .. 255
        const int r = idx >> 2;              // row inside the tile
        const int c = idx & 3;               // 16-byte chunk inside the row
        d[idx] = s[(size_t)r * (k_blocks * 4) + c];
    }
}

}  // namespace

int nvfp4_pack_tile_major(const uint8_t* src, uint8_t* dst, int N, int K,
                          cudaStream_t stream) {
    if ((N & 63) || (K & 127)) return -2;
    const int k_blocks = K >> 7;
    const long tiles = (long)(N >> 6) * k_blocks;
    nvfp4_pack_tile_major_kernel<<<(unsigned)tiles, 128, 0, stream>>>(
        reinterpret_cast<const uint4*>(src), reinterpret_cast<uint4*>(dst),
        k_blocks);
    return (int)cudaGetLastError();
}

}  // namespace quantize
}  // namespace flash_rt
