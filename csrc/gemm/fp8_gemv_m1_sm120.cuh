// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace gemv_m1 {

// Dedicated M=1 FP8 e4m3 -> BF16 GEMV for sm_120a decode shapes.
// Inputs: FP8 A [1,K] row-major, FP8 B [N,K] row-major (= W.T), BF16 D [1,N].
// alpha = a_scale * w_scale (per-tensor). M is ignored (M=1 assumed).
// Warp-per-output-row: each warp reduces one B row against A (held in smem),
// 16-byte vectorized coalesced B loads. No MMA / no BLOCK_M padding tax.
// Returns 0 on success.

#define DECL(NAME) \
  int NAME(const void* A, const void* B, void* D, \
           int M, int N, int K, float alpha, cudaStream_t stream)

DECL(gemv_fp8_m1_w4);
DECL(gemv_fp8_m1_w8);
DECL(gemv_fp8_m1_w16);
DECL(gemv_fp8_m1_resadd_w4);  // D[n] += acc*alpha (fused residual)
DECL(gemv_fp8_m1_resadd_w8);

#undef DECL

}  // namespace gemv_m1
}  // namespace gemm
}  // namespace flash_rt
