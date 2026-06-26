// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

// Native Ada (sm_89) FP8 e4m3 -> BF16 block-128 scaled GEMM.
//
// Computes D_rm[M,N] = (act_fp8 @ w_fp8^T) with DeepSeek-style block-128
// scaling applied in the mainloop:
//   D[m,n] = sum_{kb} act_scale[m, kb] * w_scale[n/128, kb]
//                     * sum_{k in kb} A[m,k] * B[n,k]
//
// Inputs (all device pointers):
//   A         : [M, K]        FP8 e4m3 row-major   (per-token quantized act)
//   B         : [N, K]        FP8 e4m3 row-major   (= W, ckpt weight)
//   act_scale : [M, K/128]    fp32 row-major       (per-token block scale)
//   w_scale   : [N/128, K/128] fp32 row-major      (weight_scale_inv)
//   D         : [M, N]        BF16 row-major
//
// Drop-in replacement for fp8_block128_gemm_descale_bf16out but reads the
// FP8 weight directly (no dequant scratch). K and N must be multiples of 128.
// Returns 0 on success.

#define DECL(NAME)                                                            \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,        \
           const float* act_scale, const float* w_scale, cudaStream_t stream)

DECL(fp8_block128_gemm_bs_sm89_32x128x128_w4);
DECL(fp8_block128_gemm_bs_sm89_64x128x128_w4);
DECL(fp8_block128_gemm_bs_sm89_64x128x128_w8);
DECL(fp8_block128_gemm_bs_sm89_128x128x128_w4);
DECL(fp8_block128_gemm_bs_sm89_128x128x128_w8);
DECL(fp8_block128_gemm_bs_sm89_32x64x128_w4);
DECL(fp8_block128_gemm_bs_sm89_64x64x128_w4);
DECL(fp8_block128_gemm_bs_sm89_128x64x128_w4);
DECL(fp8_block128_gemm_bs_sm89_16x128x128_w4);
DECL(fp8_block128_gemm_bs_sm89_16x64x128_w4);
DECL(fp8_block128_gemm_bs_sm89_32x128x128_w4_s1);
DECL(fp8_block128_gemm_bs_sm89_64x64x128_w4_s1);
DECL(fp8_block128_gemm_bs_sm89_128x128x128_w8_s1);

#undef DECL

// Auto-dispatch over the tuned tile set above based on (M, N, K).
int fp8_block128_gemm_blockscaled_sm89_bf16out(
    const void* A, const void* B, void* D, int M, int N, int K,
    const float* act_scale, const float* w_scale, cudaStream_t stream);

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
