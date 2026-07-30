// SPDX-License-Identifier: Apache-2.0
//
// Decode a streamed routed-expert block into bf16.
//
// The edge bundle stores each expert as sign-magnitude 4-bit values with one
// e4m3 scale byte per group of 16 along K, scaled by a per-tensor float the
// bundle keeps beside the blocks. That is not the format the block-scaled 4-bit
// GEMMs read: they decode E2M1, and they want the scale bytes in the SM1xx
// swizzled tile layout. Neither difference is bridgeable by relabelling --
// E2M1's sixteen values and this format's sixteen are different sets.
//
// So the streaming path decodes here and hands bf16 to the existing bf16 GEMM,
// which costs bandwidth on an already-resident block but needs no swizzle, no
// second codebook inside a GEMM, and no architecture beyond SM80. Reading the
// bundle's own linear scale layout is the point: it removes the swizzle step
// rather than implementing it.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// out[r][c] = value(packed) * e4m3(scale[r][c / group_size]) * global_scale
//
// packed  (rows, cols / 2) bytes, low nibble first, each nibble
//         magnitude | sign << 3 with magnitude in 0..7
// scale   (rows, cols / group_size) bytes, each an e4m3 magnitude
// out     (rows, cols) bf16
//
// cols must be even and a multiple of group_size, and group_size must be even
// so that a byte's two values always share one scale.
int qwen35moe_e0m3_dequant_bf16(const void* packed, const void* scale,
                                void* out, int rows, int cols,
                                int group_size, float global_scale,
                                cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
