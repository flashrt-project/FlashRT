// SPDX-License-Identifier: Apache-2.0
//
// G7.23 v19 — bare per-tensor static FP8 quant + NCDHW→NDHWC permute.
//
// This is a stripped-down sibling of bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4
// (no RMS_norm, no gamma multiply, no SiLU): just
//     y[b,t,h,w,c] = clamp(x[b,c,t,h,w] / act_scale, ±448).fp8
// Used by the motus VAE 1×1×1 ResidualBlock.shortcut path where the
// input is NCDHW bf16 and the downstream FP8 GEMM (fp8_nn_dev) needs
// (M=B·T·H·W, K=Ci) FP8 row-major — i.e. NDHWC flattened.
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

int bf16_quant_fp8_ncdhw_to_ndhwc(
    const void*  x_bf16,           // [B, C, T, H, W] bf16
    void*        y_fp8,            // [B, T, H, W, C] fp8_e4m3
    int B, int C, int T, int H, int W,
    float act_scale,
    cudaStream_t stream);

int bf16_upsample2x_quant_fp8_nchw_to_nhwc(
    const void*  x_bf16,           // [N, C, H, W] bf16
    void*        y_fp8,            // [N, 2H, 2W, C] fp8_e4m3
    int N, int C, int H, int W,
    float act_scale,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
