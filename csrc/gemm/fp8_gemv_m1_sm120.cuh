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

#define DECL_BLOCK128(NAME) \
  int NAME(const void* A, const void* B, void* D, \
           int M, int N, int K, const float* act_scale, \
           const float* w_scale, float alpha, cudaStream_t stream)

// M=1 FP8 e4m3 GEMV with per-token activation scale [K/128] and
// per-weight 128x128 block scale [N/128, K/128]. This matches official
// Qwen3-VL FP8 checkpoints that store `.weight` + `.weight_scale_inv`.
DECL_BLOCK128(gemv_fp8_block128_m1_w4);
DECL_BLOCK128(gemv_fp8_block128_m1_w8);
DECL_BLOCK128(gemv_fp8_block128_m1_w16);

#undef DECL_BLOCK128

}  // namespace gemv_m1
}  // namespace gemm
}  // namespace flash_rt
