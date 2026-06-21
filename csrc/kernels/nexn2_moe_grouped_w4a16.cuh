// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE grouped W4A16 GEMV for sm_120 (prefill / M>1 via slots).
//
// One GEMV per (token, expert) assignment ("slot"): D[slot] = A[slot] (bf16)
// @ W[expert_idx[slot]] (NVFP4). It is the grouped sibling of
// nexn2_w4a16_matvec_bf16 -- same hardware-fp4-decode bf16-activation body,
// with per-slot activation + per-expert weight/SF/alpha base pointers added on
// top (the only change over the single-GEMV kernel). The BF16 activation means
// there is no activation scale-factor swizzle to thread through, unlike the
// W4A4 block-scaled mma grouped kernel.
//
// Feed it the routing's S*TOPK assignments (sorted by expert so consecutive
// grid.y blocks hit the same weight -> the expert weight is read ~once from
// HBM, the rest L2). All add-only.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// D(slots,N) where D[s] = A[s](1,K) bf16 . W[expert_idx[s]](N,K) NVFP4 ^T.
//   A_stack    : (slots, K)        bf16 activations (one row per slot)
//   W_stack    : (E, N, K/2)       NVFP4 e2m1 weights
//   SFB_stack  : (E, swizzled)     UE4M3 block scales
//   alpha_stack: (E,)              per-expert global_scale
//   expert_idx : (slots,)          device expert id per slot
//   D          : (slots, N)        bf16
//   a_stride   : element stride between slots in A_stack (= K)
//   w_stride/sfb_stride: byte stride between experts
// K multiple of 16. Returns 0 on success.
int nexn2_moe_grouped_w4a16_bf16(
    const void*  A_stack,
    const void*  W_stack,
    const void*  SFB_stack,
    const void*  alpha_stack,
    const void*  expert_idx,
    void*        D,
    int          slots,
    int          N,
    int          K,
    long         a_stride,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
