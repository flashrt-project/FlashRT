// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// One sliding-window Picard sweep of a flow-matching Euler sampler over bf16
// latents (see flow_euler_window.cu): re-runs x_{i+1} = x_i + v_{i-ws} * dsig_i
// from the exact window base s, fills the points past the window and, when
// ws_next >= 0, writes the next sweep's masked inputs. v_ch > 0 reads v as
// token-major [T, v_ch] per point (token stride v_ld, point stride v_pt).
// Returns 0 on success.
int flow_euler_window_bf16(void* xs, const void* v, const float* dsig, int s, int ws,
                           int r, int n, int E, int v_ch, int v_ld, int v_pt,
                           bool extrap, void* nl, const void* mask, int ch_stride,
                           int ws_next, cudaStream_t stream);

// Classifier-free-guidance blend and Euler step from a [2, L, ld] patch
// sequence (conditional, unconditional) into a [C, F, H, W] latent, with the
// bf16 roundings of `u + g * (c - u)` and `x + v * dsig`.
int flow_cfg_euler_patch_bf16(const void* pred, const void* lat, void* out, int C, int F,
                              int H, int W, int ph, int pw, int ld, float g, float dsig,
                              cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
