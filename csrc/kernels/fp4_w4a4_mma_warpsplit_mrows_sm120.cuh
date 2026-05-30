// SPDX-License-Identifier: Apache-2.0
//
// Small-M (M = 1..16) warp-split-K NVFP4 W4A4 GEMM for sm_120. Combines
// the warp-split-K fill-the-SM structure (split K across warps in one
// block, partials summed in shared memory -> graph-replay safe) with the
// small-M epilogue (the SM120_16x8x64 MMA atom already computes a 16-row
// tile, so M<=16 output rows cost the SAME weight HBM traffic as M=1).
// Target: the speculative-decode VERIFY path long-K GEMMs (mlp_down
// K=17408 at M=K_draft+1<=16) where the single-warp full_n/smallm kernel
// underfills the SMs. Additive: new file + new entry point.
#pragma once
#include <cuda_runtime.h>
namespace flash_rt {
namespace gemm {
// A_packed (M,K/2), B_packed (N,K/2), D_bf16 (M,N). SFA (M,K/16) swizzled,
// SFB (N,K/16) swizzled. M in 1..16, warps in {2,4,8}, stages in {3,4,6}.
// N%8==0, K%64==0, (K/64)%warps==0. Returns 0 on success.
int fp4_w4a4_mma_sm120_warpsplit_mrows_bf16out(
    const void* A_packed, const void* B_packed, void* D_bf16, int M, int N,
    int K, const void* SFA, const void* SFB, float alpha, int warps,
    int stages, cudaStream_t stream);
}  // namespace gemm
}  // namespace flash_rt
