// SPDX-License-Identifier: Apache-2.0
//
// Generic Gated DeltaNet / WY chunk primitives.
//
// These kernels are model-agnostic over the GQA/GVA head layout used by
// linear-attention variants:
//   k_l2:     (S, Hk, D) bf16
//   beta/g:   (S, Hv) bf16
//   K_pack:   (ceil(S/64), Hk, 64, D) bf16 workspace
//   KKt_base: (ceil(S/64), Hk, 64, 64) fp32 workspace
//   A:        (ceil(S/64), Hv, 64, 64) fp32 output
//
// qk_group maps value heads to key heads: key_head = value_head / qk_group.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace linear_attention {

void gdn_wy_kkt_b64_bf16_cublaslt(
    const void* k_l2,
    const void* beta,
    const void* g_cumsum,
    void*       k_pack,
    void*       kkt_base,
    void*       A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

// KKT variant that consumes a pre-packed key buffer:
//   k_pack: (ceil(S / 64), num_k_heads, 64, head_dim)
// This lets model-specific q/k normalization pack K once while k_l2 is still
// hot, avoiding a separate pack_k_chunks_kernel launch and HBM pass.
void gdn_wy_kkt_b64_bf16_cublaslt_packed_k(
    const void* k_pack,
    const void* beta,
    const void* g_cumsum,
    void*       kkt_base,
    void*       A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

// KKT-only variant for fused downstream gating+solve.
void gdn_wy_kkt_b64_bf16_cublaslt_packed_k_only(
    const void* k_pack,
    void*       kkt_base,
    int S,
    int num_k_heads,
    int head_dim,
    cudaStream_t stream);

// Same cublasLt K @ K^T path as gdn_wy_kkt_b64_bf16_cublaslt, but leaves the
// cumulative gate out of A:
//   A[i,j] = beta[i] * dot(k[i], k[j]), i > j
// The FlashQLA-style fused GDR applies the gate by diagonal similarity later:
//   Ai_gated[i,j] * beta[j] == exp(g[i]-g[j]) * Ai_nogate[i,j] * beta[j].
void gdn_wy_kkt_b64_bf16_cublaslt_nogate(
    const void* k_l2,
    const void* beta,
    void*       k_pack,
    void*       kkt_base,
    void*       A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_recompute_wu_b64_bf16_cublaslt(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai,
    void*       Ai_pack,
    void*       rhs_w,
    void*       rhs_u,
    void*       w_pack,
    void*       u_pack,
    void*       w,
    void*       u,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_recompute_wu_b64_bf16_cublaslt_packed(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai,
    void*       Ai_pack,
    void*       rhs_w,
    void*       rhs_u,
    void*       w_pack,
    void*       u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai_pack,
    void*       rhs_w,
    void*       rhs_u,
    void*       w_pack,
    void*       u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

// Bridge path for the FlashQLA-style no-gate representation. Consumes
// Ai_pack from gdn_wy_kkt_b64_bf16_cublaslt_nogate + solve, then reproduces
// the same gated w_pack/u_pack consumed by the existing chunk_h kernels:
//   rhs_u = v * beta * exp(-g)
//   rhs_w = k * beta
//   u,w = exp(g_i) * Ai_no_gate @ rhs
// This is additive and not wired by default; the fused GDR kernel should
// eventually consume Ai_no_gate directly and avoid materializing w/u.
void gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs_nogate(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai_pack,
    void*       rhs_w,
    void*       rhs_u,
    void*       w_pack,
    void*       u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_solve_tril_b64_f32_parallel(
    const void* A,
    void*       Ai,
    int S,
    int num_v_heads,
    cudaStream_t stream);

void gdn_wy_solve_tril_b64_f32_parallel_pack(
    const void* A,
    void*       Ai,
    void*       Ai_pack,
    int S,
    int num_v_heads,
    cudaStream_t stream);

void gdn_wy_solve_tril_b64_f32_fused_pack(
    const void* A,
    void*       Ai,
    void*       Ai_pack,
    int S,
    int num_v_heads,
    cudaStream_t stream);

void gdn_wy_solve_tril_b64_f32_fused_pack_only(
    const void* A,
    void*       Ai_pack,
    int S,
    int num_v_heads,
    cudaStream_t stream);

// Fuses apply_kkt_gating_kernel + solve_tril_b64_f32_fused_pack_only.
// Consumes cublasLt KKT output laid out as
//   kkt_base: (ceil(S/64), num_k_heads, 64, 64) fp32
// and writes the solved gated inverse directly to
//   Ai_pack:  (ceil(S/64), num_v_heads, 64, 64) bf16.
void gdn_wy_solve_tril_b64_from_kkt_pack_only(
    const void* kkt_base,
    const void* beta,
    const void* g_cumsum,
    void*       Ai_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_output_o_b64_bf16_cublaslt(
    const void* q_l2,
    const void* k_l2,
    const void* v_new,
    const void* h0,
    const void* g_cumsum,
    void*       q_pack,
    void*       k_pack_hv,
    void*       v_pack,
    void*       qk_base,
    void*       local_a_pack,
    void*       qh_pack,
    void*       local_pack,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_output_o_b64_bf16_cublaslt_packed_k(
    const void* q_l2,
    const void* k_pack_hv,
    const void* v_new,
    const void* h0,
    const void* g_cumsum,
    void*       q_pack,
    void*       v_pack,
    void*       qk_base,
    void*       local_a_pack,
    void*       qh_pack,
    void*       local_pack,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_output_o_b64_bf16_cublaslt_packed_kv(
    const void* q_l2,
    const void* k_pack_hv,
    const void* v_pack,
    const void* h0,
    const void* g_cumsum,
    void*       q_pack,
    void*       qk_base,
    void*       local_a_pack,
    void*       qh_pack,
    void*       local_pack,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv(
    const void* q_pack,
    const void* k_pack_hv,
    const void* v_pack,
    const void* h0,
    const void* g_cumsum,
    void*       qk_base,
    void*       local_a_pack,
    void*       qh_pack,
    void*       local_pack,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_chunk_h_b64_bf16_cublaslt(
    const void* k_l2,
    const void* u,
    const void* w,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       w_pack,
    void*       u_pack,
    void*       wh_pack,
    void*       decayed_v_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_chunk_h_b64_bf16_cublaslt_f32state(
    const void* k_l2,
    const void* u,
    const void* w,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       w_pack,
    void*       u_pack,
    void*       wh_pack,
    void*       decayed_v_pack,
    void*       state_f32,
    void*       delta_f32,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm(
    const void* k_l2,
    const void* u,
    const void* w,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       w_pack,
    void*       u_pack,
    void*       wh_pack,
    void*       decayed_v_pack,
    void*       state_f32,
    void*       chunk_f32,
    void*       acc_f32,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm_packed_wu(
    const void* k_l2,
    const void* w_pack,
    const void* u_pack,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       decayed_v_pack,
    void*       state_f32,
    void*       chunk_f32,
    void*       acc_f32,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

void gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu(
    const void* k_l2,
    const void* w_pack,
    const void* u_pack,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       wh_pack,
    void*       decayed_v_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

// FLA-style hand-tuned mma.sync + cp.async kernel.
// Drop-in replacement for gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu using
// the SAME packed (NT, num_v_heads, 64, head_dim) w/u layout produced by
// the recompute_wu_packed_rhs / packed kernels. head_dim must be 128.
// Inputs:
//   k_l2:     (S, num_k_heads, head_dim) bf16 raw
//   w, u:     (NT, num_v_heads, 64, head_dim) bf16 packed-per-chunk
//   g_cumsum: (S, num_v_heads) bf16 chunk-local cumsum
//   state:    (num_v_heads, head_dim, head_dim) bf16, IN/OUT (in-place update)
// Outputs (raw v_new + two nullable packed side outputs for downstream
// cublasLt packed_qkv output_o):
//   h_out:        (NT, num_v_heads, head_dim, head_dim) bf16 chunk prologue states
//   v_new:        (S, num_v_heads, head_dim) bf16 raw v_new (BEFORE decay). Nullable.
//   v_new_packed: (NT, num_v_heads, 64, head_dim) bf16 packed-per-chunk view
//                 of the same raw v_new. Nullable.
//   k_pack_hv:    (NT, num_v_heads, 64, head_dim) bf16 GQA-expanded packed k
//                 (matches the layout output_o_*_packed_qkv expects). Nullable.
//
// At least one of v_new / v_new_packed must be non-null for the downstream
// attention output to have access to v_new. No scratch buffers; smem-resident
// pipeline. NT = ceil(S / 64).
void gdn_wy_chunk_h_b64_bf16_mma_fla(
    const void* k_l2,
    const void* w,
    const void* u,
    const void* g_cumsum,
    void*       state,
    void*       h_out,
    void*       v_new,
    void*       v_new_packed,
    void*       k_pack_hv,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

// FLA-style hand-tuned recompute_wu kernel. Fuses pack_recompute_rhs +
// the 2 cublasLt mmas (Ai @ rhs_u, Ai @ rhs_w) into a single CTA kernel.
// head_dim must be 128. Writes only w_pack and u_pack (no rhs_w/rhs_u
// intermediates).
//   k_l2:     (S, num_k_heads, head_dim) bf16
//   v:        (S, num_v_heads, head_dim) bf16
//   beta:     (S, num_v_heads) bf16
//   g_cumsum: (S, num_v_heads) bf16
//   Ai_pack:  (NT, num_v_heads, 64, 64) bf16
//   w_pack:   (NT, num_v_heads, 64, head_dim) bf16
//   u_pack:   (NT, num_v_heads, 64, head_dim) bf16
void gdn_wy_recompute_wu_b64_bf16_mma_fla(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai_pack,
    void*       w_pack,
    void*       u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

// FLA-style hand-tuned output_o kernel. Drop-in replacement for
// gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv.
//   q_pack:    (NT, num_v_heads, 64, head_dim) bf16 packed q (post norm_pack_q)
//   k_pack_hv: (NT, num_v_heads, 64, head_dim) bf16 GQA-expanded packed k
//   v_pack:    (NT, num_v_heads, 64, head_dim) bf16 packed v_new (un-decayed)
//   h:         (NT, num_v_heads, head_dim, head_dim) bf16 chunk prologue states
//   g_cumsum:  (S, num_v_heads) bf16
//   out:       (S, num_v_heads, head_dim) bf16
// head_dim must be 128.
void gdn_wy_output_o_b64_bf16_mma_fla(
    const void* q_pack,
    const void* k_pack_hv,
    const void* v_pack,
    const void* h,
    const void* g_cumsum,
    void*       out,
    int S,
    int num_v_heads,
    int head_dim,
    float scale,
    cudaStream_t stream);

// Variant that reads raw k_l2 (S, num_k_heads, head_dim) directly instead of
// a GQA-expanded packed K side buffer. This avoids chunk_h writing k_pack_hv
// solely for output_o and keeps the same math as the packed entry.
void gdn_wy_output_o_b64_bf16_mma_fla_rawk(
    const void* q_pack,
    const void* k_l2,
    const void* v_pack,
    const void* h,
    const void* g_cumsum,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    float scale,
    cudaStream_t stream);

}  // namespace linear_attention
}  // namespace kernels
}  // namespace flash_rt
