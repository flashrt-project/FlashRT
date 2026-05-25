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

void gdn_wy_solve_tril_b64_f32_parallel(
    const void* A,
    void*       Ai,
    int S,
    int num_v_heads,
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

}  // namespace linear_attention
}  // namespace kernels
}  // namespace flash_rt
