// SPDX-License-Identifier: Apache-2.0
//
// G7.19 — Fused split + WanRMSNorm + 3D RoPE for the Wan video QKV.
//
// Replaces motus's per-layer chain inside wan_layer.self_attn:
//   q_lin, k_lin, v_lin = packed_qkv[..., 0:D], packed_qkv[..., D:2D], packed_qkv[..., 2D:3D]   (3 narrows)
//   q = norm_q(q_lin)        (RMSNorm with weight, 1 launch)
//   k = norm_k(k_lin)        (1 launch)
//   q_rope = rope_apply(q)   (1 launch via G7.15/16 fused RoPE)
//   k_rope = rope_apply(k)   (1 launch)
//   v = v_lin (view)
// → 5 launches + 4 bf16 (B*L_v, dim) global memory round-trips.
//
// Replaces with one kernel that streams packed_qkv once, outputs
// q_rope (B, L_v, N, D_h) bf16 and k_rope (B, L_v, N, D_h) bf16. V is
// just a view of packed_qkv slice — caller handles it (zero kernels).
//
// Math per (b, t, c):
//   dim = N * D_h
//   q[c] = packed_qkv[b, t, c]
//   k[c] = packed_qkv[b, t, dim + c]
//   rms_q = sum_c(q[c]^2) / dim;  inv_q = rsqrt(rms_q + eps)
//   rms_k = sum_c(k[c]^2) / dim;  inv_k = rsqrt(rms_k + eps)
//   q_normed[c] = q[c] * inv_q * norm_q_w[c]
//   k_normed[c] = k[c] * inv_k * norm_k_w[c]
//   For each (n, c_pair) where c_pair < D_h/2:
//     re_q = q_normed[n*D_h + 2*c_pair]
//     im_q = q_normed[n*D_h + 2*c_pair + 1]
//     re_k = k_normed[...]
//     im_k = k_normed[...]
//     RoPE complex multiply with freqs[t * (D_h/2) + c_pair]
//     Write q_rope[b, t, n, 2*c_pair / 2*c_pair+1]
//     Write k_rope[...]

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

// packed_qkv : (B, L_v, 3*dim) bf16 row-major (output of fused QKV GEMM)
// norm_q_w   : (dim,)  bf16
// norm_k_w   : (dim,)  bf16
// freqs_re   : (seq_len, D_h/2) fp32
// freqs_im   : (seq_len, D_h/2) fp32
// q_rope_out : (B, L_v, N, D_h) bf16  — first contiguous dim D_h
// k_rope_out : (B, L_v, N, D_h) bf16
// dim = N * D_h.
// seq_len: number of rows for which RoPE is applied (typically T = L_v).
//          Rows beyond seq_len are passthrough (no rope mult).
void qkv_split_norm_rope_bf16(
    const void*  packed_qkv,
    const void*  norm_q_w,
    const void*  norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void*        q_rope_out,
    void*        k_rope_out,
    int B, int L_v, int N, int D_h, int seq_len, float eps,
    cudaStream_t stream);

// Variant for Motus video QKV when the fused FP8 GEMM skipped bias.
// Adds q/k bias before RMSNorm and writes a biased V output.
void qkv_split_bias_norm_rope_v_bf16(
    const void*  packed_qkv,
    const void*  qkv_bias,
    const void*  norm_q_w,
    const void*  norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void*        q_rope_out,
    void*        k_rope_out,
    void*        v_out,
    int B, int L_v, int N, int D_h, int seq_len, float eps,
    cudaStream_t stream);

// Same math as qkv_split_bias_norm_rope_v_bf16, but writes directly into
// preallocated joint Q/K/V workspace at [video_offset:video_offset + L_v].
void qkv_split_bias_norm_rope_v_cat_bf16(
    const void*  packed_qkv,
    const void*  qkv_bias,
    const void*  norm_q_w,
    const void*  norm_k_w,
    const float* freqs_re,
    const float* freqs_im,
    void*        q_cat_out,
    void*        k_cat_out,
    void*        v_cat_out,
    int B, int total_L, int video_offset, int L_v,
    int N, int D_h, int seq_len, float eps,
    cudaStream_t stream);

// Two-stream no-RoPE variant for Motus action+und QKV.
// Each packed input is (B, L, 3*N*D_h); output Q/K are (B, L, N, D_h).
void qkv_split_norm2_bf16(
    const void* packed_a,
    const void* norm_a_q_w,
    const void* norm_a_k_w,
    void* q_a_out,
    void* k_a_out,
    int B, int L_a, int N, int D_h, float eps_a,
    const void* packed_u,
    const void* norm_u_q_w,
    const void* norm_u_k_w,
    void* q_u_out,
    void* k_u_out,
    int L_u, float eps_u,
    cudaStream_t stream);

// Two-stream no-RoPE variant that writes action+und Q/K/V directly into
// preallocated joint workspace after the video segment.
void qkv_split_norm2_cat_bf16(
    const void* packed_a,
    const void* norm_a_q_w,
    const void* norm_a_k_w,
    const void* packed_u,
    const void* norm_u_q_w,
    const void* norm_u_k_w,
    void* q_cat_out, void* k_cat_out, void* v_cat_out,
    int B, int total_L, int L_v, int L_a, int L_u,
    int N, int D_h, float eps_a, float eps_u,
    cudaStream_t stream);

// Three-stream Motus joint variant. Video QKV gets bias + Q/K RMSNorm + RoPE;
// action and und QKV get Q/K RMSNorm without RoPE. All three segments write
// directly into the preallocated joint Q/K/V workspace.
void qkv_split_joint3_cat_bf16(
    const void* packed_v,
    const void* qkv_v_bias,
    const void* norm_v_q_w,
    const void* norm_v_k_w,
    const float* freqs_re,
    const float* freqs_im,
    const void* packed_a,
    const void* norm_a_q_w,
    const void* norm_a_k_w,
    const void* packed_u,
    const void* norm_u_q_w,
    const void* norm_u_k_w,
    void* q_cat_out, void* k_cat_out, void* v_cat_out,
    int B, int total_L, int L_v, int L_a, int L_u,
    int N, int D_h, int seq_len,
    float eps_v, float eps_a, float eps_u,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
