// SPDX-License-Identifier: Apache-2.0
#pragma once
// CUTLASS NVFP4 (W4A4) block-scaled GEMM on SM120a (128-deep K tile) with a
// fused per-column bias + tanh-GELU epilogue that emits a static-scale FP8
// E4M3 activation:
//
//     z = bf16(*alpha * (A @ B^T) + bias[n])
//     D = e4m3(bf16(gelu_tanh(z)) * (*inv_scale))
//
// The bf16 roundings replicate a bf16-output GEMM followed by the MolmoAct2
// bias_gelu_quant_fp8 producer (bias = null), so with inv_scale computed by
// molmo_reciprocal_f32 the bytes are identical to that two-kernel chain.
// A: e2m1 [M, K] row-major, B: e2m1 [N, K] (K-major), SFA / SFB: UE4M3
// per-16 scales in the CUTLASS Sm1xx swizzled layout; bias: BF16 [N];
// D: E4M3 [M, N] row-major. alpha / inv_scale are device fp32 scalars.
// Returns 0 on success, a cutlass::Status (as int, > 0) otherwise.
#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

int nvfp4_gemm_bias_gelu_fp8out_k128_sm120(const void* A, const void* B, const void* SFA,
                                           const void* SFB, const void* bias, void* D, int M,
                                           int N, int K, const float* alpha,
                                           const float* inv_scale, cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
