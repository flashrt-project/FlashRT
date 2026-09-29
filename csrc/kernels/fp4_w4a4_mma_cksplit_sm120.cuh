// SPDX-License-Identifier: Apache-2.0
//
// Col-tile x K-group split NVFP4 W4A4 small-M GEMM for sm_120.
//
// Targets the M=41 expert o/dn shapes (N=1024, K=2048) where the CUTLASS
// persistent NVFP4 GEMM launches only N/128 = 8 CTAs (one per N tile) and the
// long per-CTA K loop makes it latency-bound, while the plain warp-split-K
// kernel re-reads the whole A matrix once per 8-col block (A-traffic bound).
//
// A block owns COLS N-columns and splits K into KG groups; the warps are laid
// out as COL_TILES x KG (COL_TILES = COLS/8). Every K group owns a distinct
// K-slice and the COL_TILES warps inside it share the same A/SFA smem tile, so
// A is read exactly once per block per K element (A traffic = (N/COLS)*M*K/2,
// independent of KG). K-group partials are reduced in shared memory, so the
// kernel is graph-replay safe (no cross-block intermediate).
//
// Constraints: 1 <= M <= 48, K % 64 == 0, (K/64) % KG == 0, N % COLS == 0.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Config selectors: cols in {16,32,64}, kg in {2,4}, stages in {2,3,4}.
// Returns 0 on success, nonzero on caller-side error.
int fp4_w4a4_mma_cksplit_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int          M,
    int          N,
    int          K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    int          cols,
    int          kg,
    int          stages,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
