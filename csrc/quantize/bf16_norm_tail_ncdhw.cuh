#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// The elementwise tail of `WanRMS_norm.forward`, with the normaliser supplied by
// the caller (ATen's `norm(2, dim=1)`, whose reduction order is not reproducible
// from scratch). Computes, for every element,
//
//     nb = bf16(x / nrm)                      one fp32 rounding per step
//     tb = bf16(float(nb) * scale)
//     ub = bf16(float(tb) * float(gamma[c]))
//     y  = bf16(float(ub) + float(bias[c]))   (bias may be null)
//
// which is what the reference does including its intermediate roundings.
//
//   x_bf16     [B,C,T,H,W] contiguous bf16
//   nrm_fp32   [B,1,T,H,W] contiguous fp32, already clamped away from zero
//   gamma_bf16 [C,1,1,1] (contiguous; only the channel stride matters)
//   bias_bf16  same layout as gamma, or null
//   y_bf16     [B,C,T,H,W] contiguous bf16, may not alias x
int bf16_norm_tail_ncdhw(
    const void* x_bf16,
    const void* nrm_fp32,
    const void* gamma_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int B, int C, int T, int H, int W,
    float scale,
    cudaStream_t stream);

// Same contract for a channels-last activation (memory order [B,T,H,W,C],
// i.e. torch.channels_last_3d); nrm is one fp32 value per (b,t,h,w), y has
// the layout of x.
int bf16_norm_tail_ndhwc(
    const void* x_bf16,
    const void* nrm_fp32,
    const void* gamma_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int B, int C, int T, int H, int W,
    float scale,
    cudaStream_t stream);

// bf16_norm_tail_ndhwc with eight channels per thread (C % 8 == 0, 16-byte
// aligned x / y / gamma); silu != 0 also applies ATen's bf16 SiLU to the
// result, bit for bit.
int bf16_norm_tail_ndhwc_v8(
    const void* x_bf16,
    const void* nrm_fp32,
    const void* gamma_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int B, int C, int T, int H, int W,
    float scale, int silu,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
