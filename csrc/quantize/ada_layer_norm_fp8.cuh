// SPDX-License-Identifier: Apache-2.0
//
// G7.17 — Fused AdaLayerNorm + per-tensor FP8 e4m3 quantize.
//
// Replaces the motus 2-launch chain
//   ada_layer_norm_bf16(x, scale, shift) -> bf16 modulated
//   quantize_fp8_static(modulated)       -> fp8
// with a single kernel. Eliminates the bf16 (B*L, D) intermediate
// (~15 MB at T=2520 D=3072 for Wan video), which is the dominant
// cost — memory traffic round-trip dwarfs launch overhead.
//
// Math (per row, given mean μ and variance σ² of x):
//   norm[i] = (x[i] - μ) / sqrt(σ² + eps)
//   mod[i]  = norm[i] * (1 + scale[i]) + shift[i]
//   out[i]  = clamp(mod[i] / *act_scale, ±448).to(fp8_e4m3)
//
// Layout: x, scale, shift are (B*L, D) bf16 row-major (or scale/shift
// broadcast over rows when their leading dim == 1). out is (B*L, D)
// fp8_e4m3 row-major. act_scale is a device fp32 scalar.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

void ada_layer_norm_fp8(
    const void*  x_bf16,
    const void*  scale_bf16,
    const void*  shift_bf16,
    void*        out_fp8,
    const float* act_scale,
    int seq_len, int dim, float eps,
    cudaStream_t stream);

void ada_layer_norm_fp8_modfp8(
    const void*  x_bf16,
    const void*  scale_fp8,
    const void*  shift_fp8,
    const float* scale_deq,
    const float* shift_deq,
    void*        out_fp8,
    const float* act_scale,
    int seq_len, int dim, float eps,
    cudaStream_t stream);

void awq_ada_layer_norm_fp8(
    const void*  x_bf16,
    const void*  scale_bf16,
    const void*  shift_bf16,
    const void*  inv_s_bf16,
    void*        out_fp8,
    const float* act_scale,
    int seq_len, int dim, float eps,
    cudaStream_t stream);

void ada_layer_norm_nvfp4_swizzled(
    const void* x_bf16,
    const void* scale_bf16,
    const void* shift_bf16,
    void* packed_u8,
    void* sf_swizzled_u8,
    int seq_len, int dim, float eps,
    cudaStream_t stream);

void ada_layer_norm_nvfp4_swizzled_modfp8(
    const void* x_bf16,
    const void* scale_fp8,
    const void* shift_fp8,
    const float* scale_deq,
    const float* shift_deq,
    void* packed_u8,
    void* sf_swizzled_u8,
    int seq_len, int dim, float eps,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
