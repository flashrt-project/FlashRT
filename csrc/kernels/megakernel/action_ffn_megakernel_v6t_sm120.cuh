// SPDX-License-Identifier: Apache-2.0
//
// Action FFN megakernel V6tuned for sm_120a (RTX 5090).
//
// Fuses the Pi0.5 action-expert FFN call chain
//   pre-FFN AdaLN+modulate (handled by fvk.awq_ada_layer_norm_fp8) →
//   GEMM_up (FP8 W·X + bias + GELU + per-tensor FP8 quant of intermediate) →
//   GEMM_dn (FP8 W·Y + bias + gate * (acc) + residual_add) → bf16 out.
//
// Shape lock: M<=32 (action expert tokens), K_up=1024, N_up=4096,
//             K_dn=4096, N_dn=1024.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace megakernel {

// Returns 0 on success.
int action_ffn_v6t_launch_sm120(
    const void* x_fp8_in,
    const void* up_w_NK, const void* up_bias,
    const void* dn_inv_s, const void* dn_w_NK, const void* dn_bias,
    const void* gate, const void* residual,
    void* y_out,
    void* up_fp8_scr,
    int M, int K_up, int N_up, int K_dn, int N_dn,
    float up_alpha, float dn_alpha,
    float dn_act_scale,
    cudaStream_t stream);

}  // namespace megakernel
}  // namespace flash_rt
