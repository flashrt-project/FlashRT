// SPDX-License-Identifier: Apache-2.0
//
// Fused q_norm + RoPE + Q_buf write
// Fused k_norm + RoPE + KV_cache write (V copy inline)
//
// Replaces the per-decode-layer chain
//   rms_norm(q_pre, q_norm_w, q_norm_out)
//   _rope_apply_inline(q_norm_out -> q_rot)        [6 aten ops]
//   Q_buf.copy_(q_rot)
//   rms_norm(k_pre, k_norm_w, k_norm_out)
//   _rope_apply_inline(k_norm_out -> k_rot)        [6 aten ops]
//   K_cache[L, cur_pos].copy_(k_rot)
//   V_cache[L, cur_pos].copy_(v_slice)
// with two kernel launches, saving ~14 launches / layer × 36 layers.
//
// Add-only — does NOT modify the existing rms_norm or rope kernels;
// they remain available for the prefill / non-fused paths.
//
// Constraints:
//   * head_dim must be 128 (Qwen3-8B value); kernel hardcodes this.
//   * S = 1 (decode hot path); prefill keeps the existing chain.
//   * cos/sin tensors are (head_dim/2,) = (64,) BF16 — same shape used
//     by `_rope_apply_inline`.
//
// Math (Qwen3 RMSNorm, no 1+w; full-RoPE):
//   x_normed[d] = x[d] * rsqrt(sum_sq / head_dim + eps) * w[d]
//   rotate_half(x)[d] = -x[d + half]   if d < half
//                     =  x[d - half]   if d >= half
//   x_out[d] = x_normed[d] * cos[d % half] + rotate_half(x_normed)[d] * sin[d % half]

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Fused q_norm + RoPE + Q_buf write (S=1 decode).
//
//   q_pre     : (n_q_heads, 128) bf16  — output of fused QKV GEMM (q part)
//   q_norm_w  : (128,)            bf16
//   cos       : (64,)             bf16 — half-head_dim
//   sin       : (64,)             bf16
//   q_buf_dst : (n_q_heads, 128) bf16  — staging for FA2
//
// Returns 0 on success.
int qwen3_q_norm_rope_qstage_bf16(
    const void* q_pre,
    const void* q_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    int         n_q_heads,
    float       eps,
    cudaStream_t stream);

// Fused k_norm + RoPE + K_cache write + V_cache write (S=1 decode).
//
//   k_pre        : (n_kv_heads, 128) bf16
//   v_pre        : (n_kv_heads, 128) bf16
//   k_norm_w     : (128,)             bf16
//   cos / sin    : (64,)              bf16
//   k_cache_dst  : pointer to K_cache[L, cur_pos] = base of (n_kv, 128)
//   v_cache_dst  : pointer to V_cache[L, cur_pos] = base of (n_kv, 128)
//
// Returns 0 on success.
int qwen3_k_norm_rope_kvwrite_bf16(
    const void* k_pre,
    const void* v_pre,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
