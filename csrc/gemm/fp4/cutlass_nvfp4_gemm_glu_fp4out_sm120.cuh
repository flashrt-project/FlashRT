// SPDX-License-Identifier: Apache-2.0
#pragma once
// CUTLASS NVFP4 (W4A4) GEMMs on SM120a whose epilogue applies the FFN
// activation and writes the next GEMM's NVFP4 operand (e2m1 + UE4M3 block
// scales in the swizzled layout) with an explicit output global scale:
//
//   q = e2m1(y * norm_constant / sf), sf = ue4m3(amax16(y) * norm_constant / 6)
//
// i.e. norm_constant = 1 / gs_out, where gs_out is the global scale the
// consuming GEMM folds into its alpha. Scales (alpha, norm_constant) are
// device fp32 scalars, read at run time, so captured graphs see updates. M must be a multiple of 128 (the
// scale-factor store runs whole tiles; pad the operand rows with zeros).
// Return 0 on success, a cutlass::Status (as int) otherwise.
#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// y = gelu_tanh(*alpha * (A @ B^T) + bias[n]); D_packed [M, N/2], SFD for [M, N].
int nvfp4_gemm_bias_gelu_fp4out_sm120(const void* A, const void* B, const void* SFA, const void* SFB,
                                      const void* bias, void* D_packed, void* SFD, int M, int N, int K,
                                      const float* alpha, const float* norm_constant, cudaStream_t stream);

// GeGLU fold over a row-interleaved gate|up weight (gate row i -> column 2i,
// up row i -> 2i + 1, N_il = 2 * inter): y_i = gelu_tanh(*alpha_gate * g_i) *
// (*alpha_up * u_i). D_packed is full width [M, N_il/2] with each value
// duplicated into its pair (compact the even columns for the consumer); SFD is
// laid out for the compacted [M, inter] operand.
int nvfp4_gemm_geglu_fold_fp4out_sm120(const void* A, const void* B_il, const void* SFA, const void* SFB_il,
                                       void* D_packed, void* SFD, int M, int N_il, int K, const float* alpha_gate,
                                       const float* alpha_up, const float* norm_constant, cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
