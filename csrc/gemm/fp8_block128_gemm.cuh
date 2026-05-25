// SPDX-License-Identifier: Apache-2.0
//
// FP8 block-128 GEMM (Phase 2.2 / Path D).
//
// Computes
//
//   D_bf16[M, N] = (A_fp8 ⊙ act_block_scale) @ (B_fp8 ⊙ w_block_scale)^T
//
// using the dequant-then-BF16-GEMM stop-gap path. Replace with the
// native block-FP8 GEMM path once profiling shows this path's 3x
// memory-bandwidth tax dominates tok/s.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// FP8 block-128 GEMM, BF16 output.
//
// Layout & shapes:
//   A_fp8      : (M, K)        e4m3 row-major
//   B_fp8      : (N, K)        e4m3 row-major  (weight; same layout as
//                                                ckpt safetensors)
//   D_bf16     : (M, N)        bf16 row-major  (output)
//   act_scale  : (M, K/128)    fp32 row-major
//   w_scale    : (N/128, K/128) fp32 row-major
//
// Constraints: K and N must be multiples of 128. M is unrestricted
// (per-token activation scale).
//
// Caller-provided scratch buffers (FlashRT pre-allocation contract):
//   scratch_A_bf16 : at least M * K * 2 bytes
//   scratch_B_bf16 : at least N * K * 2 bytes
// Both can be reused across layers; only one weight scratch is live
// at a time even for the worst layer (worst case ~170 MB on Qwen3.6).
//
// Stream-safe; no host syncs. Internal cuBLASLt cache is keyed on
// (M, N, K) so steady-state cost = 3 kernel launches.
void fp8_block128_gemm_descale_bf16out(
    const void* A_fp8,
    const void* B_fp8,
    void*       D_bf16,
    int M, int N, int K,
    const float* act_block_scale,
    const float* w_block_scale,
    void*  scratch_A_bf16,
    void*  scratch_B_bf16,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
