// SPDX-License-Identifier: Apache-2.0
//
// Additive sibling of cutlass_sm120_block128_fp8_gemm.cuh: the same
// SM120a CUTLASS block-128 FP8 GEMM (BF16 output) with a per-column (N,)
// bf16 bias added in the epilogue.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Same layout and shape contract as
// fp8_block128_gemm_cutlass_sm120_bf16out, plus:
//   bias : (N,) bf16 row-major, added to every output column
//          (D = acc * act_scale * w_scale + bias). Must be non-null.
//
// Internally selects the Cooperative or Pingpong CUTLASS schedule by M.
// Stream-safe; per-shape arguments + workspace cached internally.
void fp8_block128_gemm_cutlass_sm120_bf16out_bias(
    const void* A_fp8,
    const void* B_fp8,
    void*       D_bf16,
    const void* bias,
    int M, int N, int K,
    const float* act_block_scale,
    const float* w_block_scale,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
