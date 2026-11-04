// SPDX-License-Identifier: Apache-2.0
//
// Channels-last (NDHWC) conv3d fprop on the FP16-accumulating tensor-core MMA,
// for the Wan VAE encoder's causal convolutions on sm_89.
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace conv3d_sm89 {

// y[n, z, p, q, k] = bias[k] + sum_{t, r, s, c} x[n, z + t, p + r - pad, q + s - pad, c]
//                                             * w[k, t, r, s, c]
// x: [N, D, H, W, C] fp16, w: [Ko, T, R, R, C] fp16, bias: [Ko] bf16 (may be
// null), y: [N, D - T + 1, H + 2 pad - R + 1, W + 2 pad - R + 1, Ko] bf16.
// Stride 1, dilation 1, zero spatial padding `pad`, no temporal padding (the
// caller supplies the causal frames). Products accumulate in FP16 (the
// double-rate HMMA on GeForce Ada), the epilogue adds the bias in FP32.
// C % 8 == 0, Ko % 8 == 0, T in {1, 3}, R == T. Returns 0 on success.
int conv3d_ndhwc_f16acc_bf16out(const void* x, const void* w, const void* bias, void* y,
                                int N, int D, int H, int W, int C, int Ko, int T, int R,
                                int pad, cudaStream_t stream);

// out[n, 0:Tc] = cache[n], out[n, Tc:Tc+Tx] = x[n], as fp16, for NDHWC
// tensors whose per-frame payload is `frame` = H*W*C elements (bf16 inputs).
// Tc may be 0. Returns 0 on success.
int ndhwc_time_cat_bf16_to_f16(const void* cache, int Tc, const void* x, int Tx, void* out,
                               int N, long long frame, cudaStream_t stream);

}  // namespace conv3d_sm89
}  // namespace gemm
}  // namespace flash_rt
