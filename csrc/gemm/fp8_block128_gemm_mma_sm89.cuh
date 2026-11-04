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
DECL(fp8_bs_exact_64x64x128_w4_s1);
DECL(fp8_bs_exact_64x64x128_w4_s2);

#undef DECL

// Residual-fold tile variants (epilogue adds `resid`): D = bf16(acc) + resid.
// resid is [M, N] BF16 row-major, same layout as D. Fuses the residual add
// into the GEMM epilogue (no separate residual_add launch, no D HBM
// round-trip). Additive — the non-resid kernels above are unchanged.
#define DECL_RESID(NAME)                                                       \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,         \
           const float* act_scale, const float* w_scale, const void* resid,    \
           cudaStream_t stream)

DECL_RESID(fp8_block128_gemm_bs_sm89_32x64x128_w4_resid);
DECL_RESID(fp8_block128_gemm_bs_sm89_64x64x128_w4_resid);
DECL_RESID(fp8_block128_gemm_bs_sm89_64x64x128_w4_s1_resid);
DECL_RESID(fp8_block128_gemm_bs_sm89_128x128x128_w8_s1_resid);

#undef DECL_RESID

// GeGLU silu-fold tile variants: fuse gate+up GEMM + silu(gate)*up + per-token
// block-128 FP8 quant into one launch. B = gate_up_w [2*N, K] (gate rows
// [0,N), up rows [N,2N)); w_scale = gate_up_s [2*N/128, K/128]. Output FP8
// [M,N] + scale [M,N/128]. BLOCK_N pinned to 128 (one quant block per CTA).
#define DECL_GEGLU(NAME)                                                     \
  int NAME(const void* A, const void* B, int M, int N, int K,                 \
           const float* act_scale, const float* w_scale, void* output,        \
           float* out_scale, cudaStream_t stream)

DECL_GEGLU(fp8_bs_geglu_silu_fold_sm89_32x128_w4_s2);
DECL_GEGLU(fp8_bs_geglu_silu_fold_sm89_16x128_w4_s2);
DECL_GEGLU(fp8_bs_geglu_silu_fold_sm89_64x128_w4_s2);
DECL_GEGLU(fp8_bs_geglu_silu_fold_sm89_128x128_w8_s1);
DECL_GEGLU(fp8_bs_geglu_silu_fold_sm89_32x128_w4_s1);
DECL_GEGLU(fp8_bs_geglu_silu_fold_sm89_16x128_w4_s1);

#undef DECL_GEGLU

// A-persistent interleaved variant (single B smem region, both gate+up acc in
// registers). Same I/O contract as DECL_GEGLU.
#define DECL_GEGLU_AP(NAME)                                                    \
  int NAME(const void* A, const void* B, int M, int N, int K,                 \
           const float* act_scale, const float* w_scale, void* output,        \
           float* out_scale, cudaStream_t stream)

DECL_GEGLU_AP(fp8_bs_geglu_silu_fold_apersist_sm89_32x128_w4_s1);
DECL_GEGLU_AP(fp8_bs_geglu_silu_fold_apersist_sm89_16x128_w4_s1);
DECL_GEGLU_AP(fp8_bs_geglu_silu_fold_apersist_sm89_32x128_w4_s2);

#undef DECL_GEGLU_AP

// Auto-dispatch over the tuned tile set above based on (M, N, K).
int fp8_block128_gemm_blockscaled_sm89_bf16out(
    const void* A, const void* B, void* D, int M, int N, int K,
    const float* act_scale, const float* w_scale, cudaStream_t stream);

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
