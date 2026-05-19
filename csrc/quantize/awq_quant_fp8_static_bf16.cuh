// SPDX-License-Identifier: Apache-2.0
//
// G7.14 — Fused (per-K AWQ activation scale + per-tensor static FP8
// e4m3 quantize) for AWQ/SmoothQuant FP8 PTQ sites in motus action
// expert + und expert. Replaces the chain:
//
//   x_scaled = flat * inv_s                // 1 launch (torch.mul, K-broadcast bf16)
//   quantize_fp8_static(x_scaled, x_fp8,   // 1 launch (per-tensor fp8 quant)
//                       act_scale)
//
// with a single fused kernel. 1800 calls/replay (60 O-proj × 10 + 60
// FFN × 2 × 10) saved one launch each. Steady state only; calibration
// keeps the 2-launch chain (runs once per session).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// in_bf16  : (M, K) bf16 row-major (linear input, post-G3b transpose)
// inv_s_bf16: (K,) bf16, broadcast over M (AWQ per-input-channel scale)
// out_fp8  : (M, K) fp8_e4m3 row-major
// act_scale: device fp32 scalar.
//   out[m, k] = clamp((in[m, k] * inv_s[k]) / *act_scale, ±448).to(fp8)
void awq_quant_fp8_static_bf16(
    const void*  in_bf16,
    const void*  inv_s_bf16,
    void*        out_fp8,
    const float* act_scale,
    long long M, int K,
    cudaStream_t stream);

void awq_quant2_fp8_static_bf16(
    const void*  in0_bf16,
    const void*  inv_s0_bf16,
    void*        out0_fp8,
    const float* act_scale0,
    long long M0, int K0,
    const void*  in1_bf16,
    const void*  inv_s1_bf16,
    void*        out1_fp8,
    const float* act_scale1,
    long long M1, int K1,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
