// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// Wan VAE AvgDown3D (diffusers) on a channels-last bf16 activation x
// [B, C, T, H, W] (memory [B, T, H, W, C]) -> y channels-last
// [B, out_ch, T/ft, H/fs, W/fs]; group size G = C * ft * fs * fs / out_ch must
// be a power of two <= 16. order: fp32 summation order (0 sequential, 1 pairwise
// for G = 4). T, H, W divisible by their factors (the caller pads T).
int bf16_avgdown3d_ndhwc(const void* x, void* y, int B, int C, int T, int H, int W,
                         int ft, int fs, int out_ch, int order, cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
