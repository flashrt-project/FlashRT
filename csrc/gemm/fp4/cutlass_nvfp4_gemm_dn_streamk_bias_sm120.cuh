// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS NVFP4 W4A16 GEMM_dn with fused per-col bias epilogue, BF16
// output, **StreamK scheduler**, SM120a.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// D = (alpha * (A_fp4 @ B_fp4^T scaled by SFA/SFB) + per_col_bias) → bf16
//
//   A_packed : (M, K/2)  uint8   NVFP4 packed (cutlass-swizzled SF)
//   B_packed : (N, K/2)  uint8   NVFP4 packed
//   SFA      : (M*K/16)  e4m3
//   SFB      : (N*K/16)  e4m3
//   bias_bf16: (N,)      bf16    per-col bias added in epilogue
//   D_bf16   : (M, N)    bf16    output
//   alpha    : float32           = sf_global_a * sf_global_b
//
// Stream-safe; per-shape workspace cached internally. Uses
// StreamKScheduler to recover SM utilization at the motus Wan FFN
// GEMM_dn shape (M=360, N=3072, K=14336): 1.277× over default
// PersistentScheduler standalone.
void fp4_w4a16_gemm_dn_streamk_bias_bf16out_sm120(
    const void*  A_packed,
    const void*  B_packed,
    const void*  SFA,
    const void*  SFB,
    const void*  bias_bf16,
    void*        D_bf16,
    int M, int N, int K,
    float        alpha,
    cudaStream_t stream);

// D = alpha * (A_fp4 @ B_fp4^T scaled by SFA/SFB) -> bf16
//
// Same StreamK schedule as the bias variant, but with a pure linear-combine
// epilogue. This matches Motus down-only sites whose down bias is skipped.
void fp4_w4a16_gemm_dn_streamk_bf16out_sm120(
    const void*  A_packed,
    const void*  B_packed,
    const void*  SFA,
    const void*  SFB,
    void*        D_bf16,
    int M, int N, int K,
    float        alpha,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
