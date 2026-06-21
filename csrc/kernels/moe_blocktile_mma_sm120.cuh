// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE grouped W4A4 block-scaled GEMM, multi-warp CTA tile. See .cu.
// A_tiled / D (num_tiles*64, K/2 | N); SFA_tiled is the batched-quant swizzle of
// all num_tiles*64 rows (global-row offset); tile_expert (num_tiles,) selects
// the expert per tile. Pad short tiles with zero rows. BM=BN=64, 4 warps; loads
// activation + weight once into smem and shares across warps (traffic 1/BM +
// 1/BN). N must be a multiple of 64.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

int moe_blocktile_mma_sm120_bf16(
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
