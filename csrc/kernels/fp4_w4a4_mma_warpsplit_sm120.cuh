// SPDX-License-Identifier: Apache-2.0
//
// Warp-split-K NVFP4 W4A4 M=1 GEMV for sm_120: 8 N-cols/block, `warps` warps
// each streaming K/warps, partials summed in shared memory (intra-block) then
// written bf16. Graph-replay safe (no cross-block/cross-kernel intermediate).
// For long-K/small-N decode shapes (mlp_down, out_proj) the full_n kernel
// underfills. Additive.
#pragma once
#include <cuda_runtime.h>
namespace flash_rt {
namespace gemm {
// A_packed (K/2,), B_packed (N,K/2), D_bf16 (N,). SFA (K/16,), SFB (N,K/16)
// swizzled. warps in {2,4,8}, stages in {3,4,6}. N%8==0, K%64==0,
// (K/64)%warps==0. Returns 0 on success.
int fp4_w4a4_mma_sm120_warpsplit_bf16out(
    const void* A_packed, const void* B_packed, void* D_bf16, int N, int K,
    const void* SFA, const void* SFB, float alpha, int warps, int stages,
    cudaStream_t stream);
}  // namespace gemm
}  // namespace flash_rt
