// SPDX-License-Identifier: Apache-2.0
//
// P2: tensor-core NVFP4 W4A4 M=1 GEMM for sm_120 (RTX 5090).
//
// Based on `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.
// m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` (CUTLASS atom
// `SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<e2m1, e2m1, float, ue4m3,
// VS=16>` at cute/arch/mma_sm120.hpp:3187). Hardware does FP4 dequant
// inside the MMA fragment with zero software cost — exactly the lever
// the R2 SIMT kernel could not use.
//
// P2-S1 scope: single (M=16 padded from M=1, N=8, K=64) tile, no K
// accumulation. Gates: cos = 1.000 vs CUTLASS reference at iso shape.
// Subsequent P2-Sx steps add K accumulation, N-tile parallelism,
// cp.async pipelining, and per-shape variant tuning.
//
// Add-only: this header sits alongside fp4_w4a4_matvec_sm120.cuh
// (the R2 SIMT version, kept as an oracle / fallback).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Single-tile NVFP4 W4A4 M=1 MMA, BF16 output, sm_120.
//
// Inputs are caller-prepared in **linear row/col-major byte form**
// (NOT the swizzled SF layout the production loader produces). The
// kernel handles all (thread, value) -> (M, K) fragment composition
// per the cute MMA_Traits layout — caller doesn't need to know about
// register-fragment layout.
//
//   A_packed  : (1, K_TILE=64)  e2m1 packed (32 bytes)
//                              — only row 0 is real; rows 1..15 are
//                                synthesized as zeros inside the
//                                kernel (M=1 padded to MMA_M=16).
//   B_packed  : (N_TILE=8, K_TILE=64) e2m1 packed (256 bytes)
//                              row-major over (n, k)
//   SFA       : (1, K_TILE/16=4) ue4m3 (4 bytes)
//                              — replicated to MMA_M=16 inside.
//   SFB       : (N_TILE=8, K_TILE/16=4) ue4m3 (32 bytes)
//   D_bf16    : (N_TILE=8,) bf16 (16 bytes)
//   alpha     : fp32 final-scale = 1 / (GSw * GSa)
//
// Returns 0 on success, nonzero on caller-side argument error.
//
// One launch per call; lifetime is the caller's stream.
int fp4_w4a4_mma_sm120_single_tile_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

// P2-S2: multi-K accumulation. Same single-warp / single-N-tile shape
// as S1, but loops over K in chunks of K_TILE=64, accumulating into
// the f32 fragment across all K-tiles. Used by gate-S2 to verify
// cos = 1.000 vs reference at K=4096 and K=12288 (production shapes).
//
//   A_packed  : (1, K/2)        e2m1 packed bytes for the full K
//   B_packed  : (N=8, K/2)      e2m1 packed bytes per col, full K
//   SFA       : (1, K/16)       ue4m3 SFs for the full K
//   SFB       : (N=8, K/16)     ue4m3 SFs per col, full K
//   D_bf16    : (N=8,)          bf16 output
//   alpha     : fp32 final scale
//   K         : multiple of 64; production values {4096, 12288}
int fp4_w4a4_mma_sm120_multi_k_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    int          K,
    cudaStream_t stream);

// P2-S3: full-N + multi-K kernel.
//
// gridDim.x = N / 32 blocks, blockDim.x = 128 (4 warps). Each warp
// owns an 8-col N-tile within its block (warp 0 → cols 0..7 of the
// block's range, warp 1 → 8..15, etc.). A and SFA are loaded ONCE
// per block into shared memory (cooperative load across all 4 warps)
// and reused by every warp's K loop. B and SFB are loaded per-warp
// per-K-tile (each warp owns its own 8-col strip).
//
// This is the first kernel that produces a full N-vector output —
// the previous (S2) kernel only computed one (M=1, N=8) tile. Used
// as the production replacement for fp4_w4a4_matvec_sm120_bf16out
// (R2 SIMT) once gates S3-S6 close.
//
// Constraints:
//   * N must be a multiple of 32 (production shapes 1024, 4096,
//     12288 all qualify).
//   * K must be a multiple of 64 (production shapes 4096, 12288 OK).
//   * Pointer alignments same as fp4_w4a4_matvec_sm120 (8-byte for
//     packed bytes, 1-byte for SFs).
int fp4_w4a4_mma_sm120_full_n_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int          N,
    int          K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
