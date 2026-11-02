// SPDX-License-Identifier: Apache-2.0
//
// Small-M NVFP4 block-scaled GEMM (W4A4) for sm_120a, written for the text
// o-projection shape (M=692, N=2560, K=4096) where the configured CUTLASS
// 128-row tile runs 120 CTAs = 3.33 waves.
//
//   A_packed : (M, K/2)   u8   row-major e2m1 (prefill layout)
//   SFA      : swizzled UE4M3 blocks (nvfp4_prefill sf_index layout)
//   B_packed : skinny-packed e2m1 (pack_weight layout)
//   SFB      : skinny-packed UE4M3 (same layout)
//   D_bf16   : (M, N)     bf16 row-major, D = alpha*(A@B) + bias + beta*C
//   alpha    : device float32 scalar
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

int molmo_smallm_gemm_bias_res_bf16out(
    const void* A_packed, const void* SFA, const void* B_packed,
    const void* SFB, const void* bias, const void* residual, void* D,
    int M, int N, int K, const void* alpha, float beta,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
