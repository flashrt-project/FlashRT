// SPDX-License-Identifier: Apache-2.0
//
// Stage3 understanding-module FFN split megakernel for sm_120a.
//
// Shape lock: M <= 192, K_up=512, N_up=2048, K_dn=2048, N_dn=512.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace megakernel {

int und_ffn_v5split_stage3_launch_sm120(
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
