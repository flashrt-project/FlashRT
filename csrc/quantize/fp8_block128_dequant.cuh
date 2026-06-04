// SPDX-License-Identifier: Apache-2.0
//
// FP8 block-128 dequantization kernels (Phase 2.2 / Path D).
//
// Two variants for the Qwen3.6-27B FP8 layout (DeepSeek-V3 style):
//
//   1. Weight dequant — input shape (N, K) e4m3 row-major, scale shape
//      (N/128, K/128) fp32. Output (N, K) bf16 row-major.
//          B_bf16[i, j] = e4m3_to_fp32(B_fp8[i, j])
//                       * w_block_scale[i / 128, j / 128]
//
//   2. Per-token activation dequant — input shape (M, K) e4m3 row-major,
//      scale shape (M, K / 128) fp32. Output (M, K) bf16 row-major.
//          A_bf16[i, j] = e4m3_to_fp32(A_fp8[i, j])
//                       * act_block_scale[i, j / 128]
//
// Caller-provided output buffers (FlashRT pre-allocation contract;
// no dynamic allocation inside).
//
// These are NEW kernels per the project rule "kernels only added,
// never deleted/modified". Existing fvk entries are not touched.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// Weight: (N, K) e4m3 + (N/128, K/128) fp32 scale -> (N, K) bf16
// row-major (preserving the natural ckpt layout — no transpose). The
// downstream cuBLASLt GEMM transposes via opT, which is faster than
// a scatter-write transpose here. N and K must both be multiples of
// 128. Caller owns out_bf16 of size N * K * 2 bytes.
void fp8_block128_dequantize_to_bf16(
    const void*  in_fp8,
    const float* w_block_scale,
    void*        out_bf16,
    int N,
    int K,
    cudaStream_t stream);

// Activation: (M, K) e4m3 + (M, K/128) fp32 scale -> (M, K) bf16.
// K must be a multiple of 128. M is unrestricted (per-token scale).
// Caller owns out_bf16 of size M * K * 2 bytes.
void fp8_per_token_block128_dequantize_to_bf16(
    const void*  in_fp8,
    const float* act_block_scale,
    void*        out_bf16,
    int M,
    int K,
    cudaStream_t stream);

// Generic row-wise block-128 dequantization with ceil-div scale layout.
//
//   input : (rows, cols) e4m3 row-major
//   scale : (rows, ceil(cols / 128)) fp32 row-major
//   output: (rows, cols) bf16 row-major
void fp8_row_block128_dequantize_to_bf16(
    const void*  in_fp8,
    const float* block_scale,
    void*        out_bf16,
    int rows,
    int cols,
    cudaStream_t stream);

// Row-wise FP8 dequantization.
//
//   input : (rows, cols) e4m3 row-major
//   scale : (rows) fp32
//   output: (rows, cols) bf16 row-major
void fp8_row_dequantize_to_bf16(
    const void*  in_fp8,
    const float* row_scale,
    void*        out_bf16,
    int rows,
    int cols,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
