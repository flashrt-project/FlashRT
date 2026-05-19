// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS NVFP4 W4A16 GEMM with fused per-col bias + GELU(tanh) epilogue,
// BF16 output, SM120a. Header for csrc/gemm/fp4/cutlass_nvfp4_gemm_bias_
// gelu_bf16out_sm120.cu (Recipe C step 1).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// D = GELU_tanh(alpha * (A_fp4 @ B_fp4^T scaled by SFA/SFB) + bias_per_col)
//
//   A_packed : (M, K/2)  uint8   NVFP4 packed (cutlass-swizzled)
//   B_packed : (N, K/2)  uint8   NVFP4 packed (cutlass-swizzled)
//   SFA      : (M*K/16)  e4m3    NVFP4 SF for A
//   SFB      : (N*K/16)  e4m3    NVFP4 SF for B
//   bias_bf16: (N,)      bf16    per-col bias (added before GELU)
//   D_bf16   : (M, N)    bf16    output
//   alpha    : float32           = sf_global_a * sf_global_b
//
// Stream-safe; per-shape workspace cached internally.
void fp4_w4a16_gemm_bias_gelu_bf16out_sm120(
    const void*  A_packed,
    const void*  B_packed,
    const void*  SFA,
    const void*  SFB,
    const void*  bias_bf16,
    void*        D_bf16,
    int M, int N, int K,
    float        alpha,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
