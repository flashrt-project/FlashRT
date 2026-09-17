// SPDX-License-Identifier: Apache-2.0
//
// Small-M (M = 1..4) warp-split-K NVFP4 W4A4 GEMV/GEMM for sm_120, f32
// output. Next generation of fp4_w4a4_mma_warpsplit_mrows_sm120 (additive:
// that entry is unchanged): the row count is a template parameter instead
// of a kernel argument (a runtime M in the MMA hot loop costs measurable
// time even at M=1), the epilogue writes f32 directly, and the launch can
// join the caller's programmatic-dependent-launch (PDL) chain — hosts that
// overlap every launch lose ground to any kernel that breaks the chain.
// The SM120_16x8x64 blockscaled MMA atom computes a 16-row tile, so rows
// 2..M ride the same weight HBM traffic as M=1: token t occupies A-tile
// row t (smem +t*32B), SFA atom-layout row t (+t*16B), and C-fragment
// lanes 4t..4t+3.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// A_packed (M, K/2) row-major e2m1 pairs, B_packed (N, K/2), D f32 (M, N).
// SFA: atom-layout scales for rows 0..M-1 of problem (M, K); SFB: atom
// layout for (N, K). M in 1..4, warps in {2,4,8}, stages in {3,4,6},
// N % 8 == 0, K % 64 == 0, (K/64) % warps == 0. Returns 0 on success.
int fp4_w4a4_mma_sm120_warpsplit_mrows_f32out(
    const void * A_packed, const void * B_packed, float * D, int M, int N,
    int K, const void * SFA, const void * SFB, float alpha, int warps,
    int stages, bool pdl, cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
