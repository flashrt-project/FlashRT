// SPDX-License-Identifier: Apache-2.0
#pragma once
// CUTLASS NVFP4 (W4A4) block-scaled GEMM on SM120a with a fused
// per-column bias and optional residual source, BF16 output:
//
//     D = *alpha * (A @ B^T) + bias[n] + beta * C
//
// A: e2m1 [M, K] row-major, B: e2m1 [N, K] (K-major), SFA / SFB: UE4M3
// per-16 scales in the CUTLASS Sm1xx swizzled layout. C / D: BF16 [M, N]
// row-major; C may alias D (in-place residual stream update). With beta = 0
// C may be null. bias: BF16 [N] (required; pass zeros for a bias-free layer).
// alpha is a device fp32 scalar read at run time (captured graphs see
// updates written into it); beta is a host constant.
//
// schedule 0: TMA warp-specialized pingpong, persistent tile scheduler
//             (wide-N projections);
// schedule 1: TMA warp-specialized cooperative, stream-K scheduler
//             (narrow-N, long-K projections that leave SMs idle otherwise).
// Returns 0 on success, a cutlass::Status (as int, > 0) otherwise.
#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

int nvfp4_gemm_bias_res_bf16out_sm120(const void* A, const void* B, const void* SFA, const void* SFB,
                                      const void* bias, const void* C, void* D, int M, int N, int K,
                                      const float* alpha, float beta, int schedule, cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
