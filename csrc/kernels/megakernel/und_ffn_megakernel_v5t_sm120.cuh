// SPDX-License-Identifier: Apache-2.0
//
// Und FFN megakernel V5tuned for sm_120a (RTX 5090).
//
// Fuses the Pi0.5 understanding-module FFN call chain
//   norm + per-tensor FP8 quant of input →
//   GEMM_up (FP8 W·X + bias + GELU + per-tensor FP8 quant of intermediate) →
//   GEMM_dn (FP8 W·Y + bias + residual_add) → bf16 out.
//
// Shape lock: M ≤ 144 (capacity = 9 m-tiles of 16),
//             K_up=512, N_up=2048, K_dn=2048, N_dn=512.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace megakernel {

int und_ffn_v5t_launch_sm120(
    const void* x_in, const void* up_inv_s,
    const void* up_w_NK, const void* up_bias,
    const void* dn_inv_s, const void* dn_w_NK, const void* dn_bias,
    const void* residual_in,
    void* y_out,
    void* x_fp8_scr, void* up_fp8_scr,
    int M, int K_up, int N_up, int K_dn, int N_dn,
    float up_alpha, float dn_alpha,
    float up_act_scale, float dn_act_scale,
    void* barrier_state, cudaStream_t stream);

}  // namespace megakernel
}  // namespace flash_rt
