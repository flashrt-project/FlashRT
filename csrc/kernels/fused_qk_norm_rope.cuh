// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — Fused Q/K RMSNorm + RoPE kernel.
//
// Single-kernel warp-per-head Q/K normalization + RoPE rotation.
// Replaces qkv_split + rms_norm + rope_apply (3 kernel launches → 1).
//
// Pipeline:
//   1. Read Q and K from strided Dq buffer
//   2. Apply per-head RMSNorm (Q and K separately)
//   3. Apply RoPE — warp-shuffle RMS reduction, register-based
//   4. Write RoPE results to q_temp/k_temp output buffers

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

void fused_qk_norm_rope_v4_bf16(
    const __nv_bfloat16* dq,         // [BS, QKVD] = [BS, NH*HD + 2*NKV*HD]
    const __nv_bfloat16* q_weight,   // [HD]  Q norm weight
    const __nv_bfloat16* k_weight,   // [HD]  K norm weight
    const __nv_bfloat16* cos,        // [BS, rope_dim]  RoPE cos
    const __nv_bfloat16* sin,        // [BS, rope_dim]  RoPE sin
    __nv_bfloat16* q_temp,           // RoPE output
    __nv_bfloat16* k_temp,           // RoPE output
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
