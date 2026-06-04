// SPDX-License-Identifier: Apache-2.0
//
// Fast per-token x per-128K-block FP8 e4m3 quantization kernel
// (Qwen3.6 / DeepSeek-V3 layout).
//
// Replaces transformers' triton_fp8_act_quant which has ~30 µs Python
// wrapper overhead per call. With ~400 such calls per Qwen3.6 decode
// step, that's 12 ms/step in pure dispatch — the dominant CPU-side
// bottleneck per torch.profiler measurement.
//
// Math: for input (M, K) bf16:
//   block_amax[m, kb] = max(|input[m, kb*128 : (kb+1)*128]|)
//   scale[m, kb]      = block_amax / 448.0  (e4m3 max = 448)
//   output[m, k]      = clamp(input[m, k] / scale[m, k/128], -448, 448)
//                        cast to e4m3
//   scale stored as fp32; matches act_block_scale layout used by
//   Path B GEMM (fp8_block128_gemm_cutlass_sm120_bf16out).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// Per-token x per-128K-block FP8 e4m3 quantization.
//
//   input  : (M, K)         bf16 row-major  -- K must be multiple of 128
//   output : (M, K)         e4m3 row-major
//   scale  : (M, K/128)     fp32 row-major
//
// Caller-provided output buffers (pre-allocated). M is unrestricted;
// the kernel parallelizes one block per (m, k_block).
void fp8_per_token_block128_quant_bf16(
    const void* input,
    void*       output_fp8,
    float*      output_scale,
    int M, int K,
    cudaStream_t stream);

// Generic row-wise block-128 FP8 e4m3 quantization with ceil-div scale
// layout. Unlike fp8_per_token_block128_quant_bf16, K does not need to
// be a multiple of 128.
//
//   input  : (rows, cols) bf16 row-major
//   output : (rows, cols) e4m3 row-major
//   scale  : (rows, ceil(cols / 128)) fp32 row-major
void fp8_row_block128_quant_bf16(
    const void* input,
    void*       output_fp8,
    float*      output_scale,
    int rows, int cols,
    cudaStream_t stream);

// Row-wise FP8 e4m3 quantization.
//
//   input  : (rows, cols) bf16 row-major
//   output : (rows, cols) e4m3 row-major
//   scale  : (rows) fp32, one scale per row
void fp8_row_quant_bf16(
    const void* input,
    void*       output_fp8,
    float*      output_scale,
    int rows, int cols,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
