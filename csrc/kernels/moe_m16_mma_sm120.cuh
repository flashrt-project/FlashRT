// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE grouped W4A4 block-scaled mma, M=16 tiled (prefill). See .cu.
// A_tiled / D (num_tiles*16, K/2 | N); SFA_tiled is per-tile super (each 16-row
// tile its own swizzle super, sfa_stride bytes apart); tile_expert (num_tiles,)
// selects the expert per tile. Pad short tiles with zero rows.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

int moe_m16_mma_sm120_bf16(
    const void*  A_tiled,
    const void*  B_stack,
    const void*  SFA_tiled,
    const void*  SFB_stack,
    void*        D,
    const void*  alpha_stack,
    const void*  tile_expert,
    int          num_tiles,
    int          N,
    int          K,
    long         sfa_stride,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
