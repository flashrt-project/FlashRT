// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS-based block-128 FP8 GEMM for SM120a (RTX 5090 / Blackwell
// consumer).
//
// Native block-scaled FP8 GEMM with DeepSeek-V3 / Qwen3.6 layout:
//   * activation: per-token (1) x per-128 K block scale
//   * weight   : per-128 N x per-128 K block scale
//   * output   : BF16
//
// Replaces the Path D dequantize-then-bf16-GEMM stop-gap with a
// fused Tensor Core kernel from CUTLASS 4.x example 87b
// (87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu). Same Python /
// pybind signature shape as fp8_block128_gemm_descale_bf16out so
// callers can swap with one-line change.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Path B SM120a CUTLASS block-128 FP8 GEMM, BF16 output.
//
// Layout & shapes match Path D's signature:
//   A_fp8      : (M, K)        e4m3 row-major
//   B_fp8      : (N, K)        e4m3 row-major
//   D_bf16     : (M, N)        bf16 row-major
//   act_scale  : (M, K/128)    fp32 row-major
//   w_scale    : (N/128, K/128) fp32 row-major
//
// Constraints: K and N must be multiples of 128. M is unrestricted.
//
// Internally selects a Cooperative or Pingpong CUTLASS schedule
// based on M (Pingpong is faster when M is small, e.g. decode
// step or short prefill). Caller does not provide scratch buffers
// (the kernel is fused, no dequant intermediates needed).
//
// Stream-safe; per-shape arguments + workspace cached internally.
void fp8_block128_gemm_cutlass_sm120_bf16out(
    const void* A_fp8,
    const void* B_fp8,
    void*       D_bf16,
    int M, int N, int K,
    const float* act_block_scale,
    const float* w_block_scale,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
