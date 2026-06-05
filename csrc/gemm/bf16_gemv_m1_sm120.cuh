// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace gemv_m1 {

// Dedicated M=1 BF16 -> BF16 GEMV for sm_120a decode shapes (the all-BF16
// sibling of the FP8 gemv_fp8_m1_*). Inputs: BF16 A [1,K] row-major, BF16
// B [N,K] row-major (= W.T), BF16 D [1,N]. alpha is applied to the output
// (1.0 for BF16; kept for ABI symmetry with the FP8 GEMV + BIND_GEMV_M1).
// M is ignored (M=1 assumed). Warp-per-output-row; B read in 16-byte coalesced
// uint4 (8 bf16) chunks; warp-shuffle reduction. Unlike the FP8 variant it does
// NOT stage A in shared memory: at BF16 that stage is 2x the bytes and caps
// blocks/SM on K=9728; A is tiny and stays hot in L2, so reading it from global
// keeps smem=0 and occupancy maximal. K assumed a multiple of 8.

#define DECL_BF16_GEMV(NAME) \
  int NAME(const void* A, const void* B, void* D, \
           int M, int N, int K, float alpha, cudaStream_t stream)

DECL_BF16_GEMV(gemv_bf16_m1_w4);
DECL_BF16_GEMV(gemv_bf16_m1_w8);
DECL_BF16_GEMV(gemv_bf16_m1_w16);

#undef DECL_BF16_GEMV

}  // namespace gemv_m1
}  // namespace gemm
}  // namespace flash_rt
