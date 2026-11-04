// SPDX-License-Identifier: Apache-2.0
//
// Tile-major repack of an NVFP4 weight payload.
//
// The SM89 NVFP4 GEMM stages one [64 rows x 128-K block] weight tile per
// k-iteration, i.e. 64-byte row segments at a K/2-byte row stride. On Ada that
// global pattern streams at 188 GB/s (measured, scripts/weight_pattern_bw.cu),
// well under the 274 GB/s device ceiling, and at M<=32 the action stream is
// pure weight bandwidth -- so the address pattern, not the kernel, is the cap.
//
// This is a pure permutation of the same bytes: every 4096-byte tile becomes
// contiguous, after which the identical tile read runs at 274 GB/s. The GEMM's
// shared-memory swizzle, expansion and block-scale fold are unchanged; only
// the global source address changes, so results are bit-identical. Scale
// factors (the swizzled SF image) and the global scale are not touched.
//
// One-time, static cost at weight-preparation time: N*K/2 bytes read + written.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {

// src: [N, K/2] packed e2m1 row-major (low nibble = even k)
// dst: [N/64, K/128, 64, 64] -- tile (n/64, k/128) at a 4096-byte offset,
//      row (n%64) at a 64-byte stride inside it.
// Requires N % 64 == 0 and K % 128 == 0. Returns 0, or -2 on a bad shape.
int nvfp4_pack_tile_major(const uint8_t* src, uint8_t* dst, int N, int K,
                          cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
