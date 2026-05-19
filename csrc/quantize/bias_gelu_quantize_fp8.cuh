// SPDX-License-Identifier: Apache-2.0
//
// G7.10 — Fused (add_bias + GELU(tanh) + per-tensor FP8 e4m3 quantize)
// for the FFN intermediate activation between up_proj and down_proj
// FP8 GEMMs. Replaces the chain:
//
//   add_bias_bf16(up_out, bias)         // 1 launch
//   gelu_inplace(up_out)                // 1 launch
//   quantize_fp8_static(up_out, up_fp8, // 1 launch
//                       down_act_scale)
//
// with a single fused kernel. Used in steady state only; calibration
// keeps the 3-launch chain (runs once per session).
//
// GELU formula: tanh approx (matches Wan upstream and existing
// flash_rt gelu_inplace kernel):
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// in_bf16  : (M, N) bf16 row-major (FP8 GEMM up_proj output)
// bias     : (N,)  bf16 (or nullptr; then no bias add)
// out_fp8  : (M, N) fp8_e4m3 row-major
// act_scale: device fp32 scalar; out = clamp(gelu(in+bias)/scale, +/-448).to(fp8)
void bias_gelu_quantize_fp8_static_bf16(
    const void*  in_bf16,
    const void*  bias_bf16,    // may be nullptr
    void*        out_fp8,
    const float* act_scale,
    long long M, int N,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
