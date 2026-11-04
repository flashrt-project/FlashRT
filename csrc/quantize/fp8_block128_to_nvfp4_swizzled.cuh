// SPDX-License-Identifier: Apache-2.0
//
// Single-shot FP8 (block-128) -> NVFP4 (swizzled SF + per-tensor global) for
// weight tensors. Replaces the lossy (fp8_block128_dequantize_to_bf16 +
// quantize_bf16_to_nvfp4_swizzled) two-step that:
//   * truncates dequant precision to BF16 (loses 16 mantissa bits),
//   * computes per-block amax in BF16 (7-bit precision floor),
//   * stores SF without a per-tensor global_scale, so for typical Qwen MTP
//     amax ~ 0.1 the SF byte (= amax/6) lands in UE4M3 denormal/zero region
//     and FP4 effectively only uses [-2, +2] of [-6, +6].
//
// Two-launch design:
//   1. global-amax pass: dequant FP8 -> FP32 inline, atomicMax row reduction
//      into a single FP32 scalar.
//   2. quantize pass: read global_scale = global_amax / (FP4_MAX * SF_MAX)
//      = global_amax / 2688, then per-NVFP4-block compute SF in UE4M3 with
//      proper global division so the byte stays in the well-represented
//      part of UE4M3.
//
// All add-only: existing fp8_block128_dequantize_to_bf16 and
// quantize_bf16_to_nvfp4_swizzled remain untouched.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {

// One-shot weight conversion. All pointers are device-resident.
//
//   w_fp8                : (N, K) e4m3 row-major
//   w_block_scale_fp32   : (N/128, K/128) fp32 (DeepSeek-V3 layout)
//   nvfp4_packed         : (N, K/2) u8 -- output, two e2m1 codes per byte
//   nvfp4_sf_swizzled    : output, layout matches the SM120 NVFP4 W4A16
//                          GEMM SFB expectation (same 512-byte super-atom
//                          permutation as quantize_bf16_to_nvfp4_swizzled).
//   scratch_global_amax  : (1) fp32 -- caller pre-allocates; kernel zeros
//                          and writes max|W|.
//   out_global_scale     : (1) fp32 -- caller pre-allocates; kernel writes
//                          global_scale = max|W| / 2688 (to be passed as
//                          GEMM alpha = act_global * w_global; for per-token
//                          activation quant act_global=1).
//
// Preconditions:
//   * N % 128 == 0, K % 128 == 0 (DeepSeek-V3 block constraint)
//   * K % 16 == 0 (NVFP4 micro-block)
//   * num_blocks_per_row = K/16 may exceed 1024 (e.g. K=17408 -> 1088); the
//     kernel loops thread-over-block to stay within blockDim limits.
void fp8_block128_to_nvfp4_swizzled_bf16(
    const void* w_fp8,
    const float* w_block_scale_fp32,
    uint8_t* nvfp4_packed,
    uint8_t* nvfp4_sf_swizzled,
    float* scratch_global_amax,
    float* out_global_scale,
    int N, int K,
    cudaStream_t stream);

// Per-128 scale-factor variant: the e2m1 codebook is scaled by the amax of
// the full 128-wide K block (which is exactly one FP8 block) instead of one
// 16-wide sub-block, and that single scale is written to all eight sub-block
// SF slots. The payload layout and the GEMM are unchanged, so the existing
// kernel reconstructs the coarser quantization exactly. Built to measure
// whether the per-16 SF image (0.0625 B/param) can be dropped: a later kernel
// only needs one SF byte per 128 block, cutting ~9% of the action weight
// traffic.
// Plain-layout twin of the entry above: same payload and same SF bytes, but
// sf128_plain is [N/128, K/128, 128] u8 (one byte per row and 128-K block),
// the layout the SF128 GEMM variant reads.
void fp8_block128_to_nvfp4_sf128_plain_bf16(
    const void* w_fp8,
    const float* w_block_scale_fp32,
    uint8_t* nvfp4_packed,
    uint8_t* sf128_plain,
    float* scratch_global_amax,
    float* out_global_scale,
    int N, int K,
    cudaStream_t stream);

void fp8_block128_to_nvfp4_sf128_swizzled_bf16(
    const void* w_fp8,
    const float* w_block_scale_fp32,
    uint8_t* nvfp4_packed,
    uint8_t* nvfp4_sf_swizzled,
    float* scratch_global_amax,
    float* out_global_scale,
    int N, int K,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
