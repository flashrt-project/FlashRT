// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS NVFP4 W4A16 GEMM with fused per-col bias + GELU(tanh) +
// per-block-16 NVFP4 quantization epilogue, FP4 packed output, SM120a.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// D_packed[m, n/2] = pack_FP4(
//   GELU_tanh(alpha * (A_fp4 @ B_fp4^T scaled by SFA/SFB) + bias_per_col)
// ) with per-16-block NVFP4 SFD in cutlass-swizzled UE4M3 layout.
//
//   A_packed : (M, K/2)  uint8   NVFP4 packed (cutlass-swizzled SF)
//   B_packed : (N, K/2)  uint8   NVFP4 packed (cutlass-swizzled SF)
//   SFA      : (M*K/16)  e4m3
//   SFB      : (N*K/16)  e4m3
//   bias_bf16: (N,)      bf16    per-col bias
//   D_packed : (M, N/2)  uint8   NVFP4 packed
//   SFD      : (M*N/16)  e4m3    output SF, cutlass-swizzled layout
//   alpha    : float32           = sf_global_a * sf_global_b
//
// Stream-safe; per-shape workspace cached internally.
void fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
    const void*  A_packed,
    const void*  B_packed,
    const void*  SFA,
    const void*  SFB,
    const void*  bias_bf16,
    void*        D_packed,
    void*        SFD,
    int M, int N, int K,
    float        alpha,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
