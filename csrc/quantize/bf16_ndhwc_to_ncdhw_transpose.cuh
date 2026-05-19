// SPDX-License-Identifier: Apache-2.0
//
// G7.23 — fast 5D BF16 NDHWC -> NCDHW transpose for v17 conv output.
//
// aten::contiguous after .permute(0, 4, 1, 2, 3) does the right thing
// numerically but uses a generic stride-based copy that runs at ~10x
// over peak HBM BW (10.6 ms total for 36 motus VAE conv outputs in the
// G7.23 profile). This kernel is a tiled smem-transpose hand-written for
// the 5D layout swap, achieves near-peak BW.
//
// Layout in : x  [B, T, H, W, C]   bf16  (v17 output)
// Layout out: y  [B, C, T, H, W]   bf16  (consumed by next bf16 / fused_quant)
//
// Tile: each CTA owns one (b, t, h, w_block of 32). Iterates over C in
// chunks of 32. Per chunk: 32x32 smem transpose. Read NDHWC contiguous
// in C (innermost), write NCDHW contiguous in W (innermost). Both
// passes coalesced over the W and C axes respectively.
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

int bf16_ndhwc_to_ncdhw_transpose(
    const void* x_NDHWC,           // [B, T, H, W, C] bf16
    void*       y_NCDHW,           // [B, C, T, H, W] bf16
    int B, int C, int T, int H, int W,
    cudaStream_t stream);

int bf16_ndhwc_to_ncdhw_add_bf16(
    const void* x_NDHWC,           // [B, T, H, W, C] bf16
    const void* residual_NCDHW,    // [B, C, T, H, W] bf16
    void*       y_NCDHW,           // [B, C, T, H, W] bf16
    int B, int C, int T, int H, int W,
    long long rs_b, long long rs_c, long long rs_t,
    long long rs_h, long long rs_w,
    cudaStream_t stream);

int bf16_ndhwc_to_ncdhw_bias_bf16(
    const void* x_NDHWC,           // [B, T, H, W, C] bf16
    const void* bias_C,            // [C] bf16
    void*       y_NCDHW,           // [B, C, T, H, W] bf16
    int B, int C, int T, int H, int W,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
