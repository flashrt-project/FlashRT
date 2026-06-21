// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini fine-grained MoE grouped NVFP4 W4A16 GEMV for SM120.
//
// Decode routes one token to top-8 of 256 experts; the 8 gate_up GEMVs
// share the token activation, the 8 down GEMVs each have their own. This
// kernel runs all E selected experts in ONE launch (grid.y = expert slot),
// indexing the weight stack by a device-side expert-id buffer so the
// dynamic top-k routing can drive a captured CUDA graph (the buffer is
// re-read each replay). The per-slot output / activation strides let the
// same kernel serve both gate_up (shared activation, a_stride=0) and down
// (per-slot activation).
//
// Inner block-scaled mma + swizzled-SF decode are identical to
// fp4_w4a4_mma_sm120_full_n_bf16out (validated cos=1.0); only the per-slot
// base-pointer arithmetic is added on top.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// Grouped NVFP4 W4A16 GEMV, M=1 per expert, BF16 output, SM120.
//
//   A_stack    : activation(s), NVFP4 packed. slot s reads A_stack + s*a_stride
//                (a_stride=0 => one shared activation for every slot).
//   B_stack    : (num_experts, N, K/2) packed weight stack.
//   D          : (E, N) bf16 output, one row per slot.
//   SFA_stack  : per-slot swizzled SF (sfa_stride=0 => shared).
//   SFB_stack  : (num_experts, sf_bytes) swizzled SF stack.
//   alpha_stack: (num_experts,) fp32 per-expert GEMM alpha.
//   expert_idx : (E,) device int, global expert id for each slot.
//   strides    : byte strides into the stacks (w_stride = N*K/2,
//                sfb_stride = swizzled SF bytes for (N,K)).
//
// Returns 0 on success, nonzero on caller-side argument error.
int moe_grouped_gemv_sm120_bf16(
    const void*  A_stack,
    const void*  B_stack,
    void*        D,
    const void*  SFA_stack,
    const void*  SFB_stack,
    const void*  alpha_stack,
    const void*  expert_idx,
    int          E,
    int          N,
    int          K,
    long         a_stride,
    long         sfa_stride,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
