// SPDX-License-Identifier: Apache-2.0
//
// Fused silu(gate) * up + nvfp4 swizzled quantize.
//
// Replaces the (silu_mul → quantize_bf16_to_nvfp4_swizzled) pair on
// the post-mlp_gate_up path:
//   silu_mul_qwen36_bf16(gate, up, silu_mul_out)
//   quantize_bf16_to_nvfp4_swizzled(silu_mul_out, packed, sf_swz)
// into one launch — 1 launch saved per layer × 36 layers/token,
// plus removes the bf16 round-trip through HBM for the
// (rows × cols) intermediate.
//
// Add-only: keeps the existing silu_mul_qwen36_bf16 and
// quantize_bf16_to_nvfp4_swizzled kernels intact.
//
// Constraints:
//   * cols must be a multiple of 16 (NVFP4 block size)
//   * Same swizzled SF layout as quantize_bf16_to_nvfp4_swizzled
//     (Sm1xx tile-interleaved; matches the production loader).
//   * Math matches the existing silu_mul kernel — fp32 silu, bf16
//     round-trip between silu and the multiply (preserves training
//     rounding pattern).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// gate, up : (rows, cols) bf16
// packed   : (rows, cols/2) u8 — NVFP4 e2m1 packed
// sf_swz   : swizzled UE4M3 SFs, byte count =
//            ceil(rows/128) * ceil(cols/16/4) * 512
int silu_mul_to_nvfp4_swizzled_bf16(
    const void* gate,
    const void* up,
    void*       packed,
    void*       sf_swz,
    int         rows,
    int         cols,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
