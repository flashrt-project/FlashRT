// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace smallM_splitk {

// SplitK FP8 e4m3 -> BF16 GEMM. scratch is fp32 [k_split, M, N] buffer caller
// must allocate (zero-init optional, kernel overwrites). Returns 0 on success.

int splitk_fp8_gemm_16x64x128_w4(const void* A, const void* B, void* D,
                                  int M, int N, int K, int k_split,
                                  float alpha, void* scratch,
                                  cudaStream_t stream);
int splitk_fp8_gemm_16x64x256_w4(const void* A, const void* B, void* D,
                                  int M, int N, int K, int k_split,
                                  float alpha, void* scratch,
                                  cudaStream_t stream);
int splitk_fp8_gemm_32x64x128_w4(const void* A, const void* B, void* D,
                                  int M, int N, int K, int k_split,
                                  float alpha, void* scratch,
                                  cudaStream_t stream);

}  // namespace smallM_splitk
}  // namespace gemm
}  // namespace flash_rt
