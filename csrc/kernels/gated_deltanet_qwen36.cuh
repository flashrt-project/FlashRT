// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet (linear attention) kernels for Qwen3.6 — Phase 3.3 / 3.4.
// Implements the same math as transformers/models/qwen3_5/modeling_qwen3_5.py
// torch_recurrent_gated_delta_rule and torch_chunk_gated_delta_rule, fused
// into single-launch kernels for SM120a / RTX 5090.
//
// Qwen3.6 layer config (linear-attn):
//   * num_v_heads = 48, num_k_heads = 16   (Q/K broadcast 3x)
//   * head_k_dim  = head_v_dim = 128
//   * use_qk_l2norm_in_kernel = True
//   * scale = 1 / sqrt(head_k_dim) applied to Q
//
// State per layer: (B, num_v_heads, head_k_dim, head_v_dim) bf16.
// Per-token operations are O(NH_v * HD_k * HD_v) = ~786 K bf16 ops/layer.
//
// All ops accumulate in fp32 internally; state stored as bf16 on disk
// (fp32 state was prototyped but bf16 cos vs HF ref >= 0.999 in practice).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Single-token decode using recurrent state. Updates ``state`` in place
// and writes the per-head output to ``out``.
//
// Tensor layouts (all bf16 row-major):
//   q     : (B, num_v_heads, head_k_dim)        already broadcast from
//                                                num_k_heads via repeat_interleave
//   k     : (B, num_v_heads, head_k_dim)        same broadcast
//   v     : (B, num_v_heads, head_v_dim)
//   g     : (B, num_v_heads)                    log-decay gate
//   beta  : (B, num_v_heads)                    sigmoid'd update rate
//   state : (B, num_v_heads, head_k_dim, head_v_dim)  in/out
//   out   : (B, num_v_heads, head_v_dim)
//
// Constraints: head_k_dim == head_v_dim == 128 (Qwen3.6); kernel is
// templated on it for register-allocation efficiency, but exposes a
// runtime check for correctness.
//
// If ``use_qk_l2norm`` is true, q and k are L2-normalized along the
// head dim before the recurrent update (matches HF's
// use_qk_l2norm_in_kernel=True).
void gated_deltanet_recurrent_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream);

// In/out-state variant: reads col from state_in, writes updated col
// to state_out (different buffer). Caller chains state_in[k+1] :=
// state_out[k] to support per-step state save without an extra
// .copy_(state_save, state) launch per step.
void gated_deltanet_recurrent_inout_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    const void* state_in,
    void*       state_out,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream);

// FP32-state variant: identical math to the BF16-state path, but the
// persistent recurrent state is stored as fp32. q/k/v/g/beta and the
// per-step ``out`` stay bf16. Used by the Thor K-row to avoid the
// per-iteration bf16 round-trip on state, which is what makes BF16-
// state K-row diverge from per-token at K beyond ~22 on Thor.
void gated_deltanet_recurrent_qwen36_f32state_bf16io(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state_f32,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream);

// Multi-token recurrent scan for prefill chunks. Reads initial state,
// loops S tokens inside one launch, writes all S outputs and final state.
// It intentionally does not materialize per-token state snapshots, so
// speculative verify with partial-reject recovery keeps using the
// recurrent_inout variant above.
void gated_deltanet_chunk_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int S, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream);

// Shared-memory state variant for profiling/tuning long prefill.
// Same ABI and math as gated_deltanet_chunk_qwen36_bf16.
void gated_deltanet_chunk_smem_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int S, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream);

// Split linear-attention conv output and broadcast Q/K heads.
// conv_out: (S, 10240) = Q(16*128), K(16*128), V(48*128)
// q48/k48/v48: contiguous (S, 48, 128), with Q/K head h sourced
// from floor(h / 3).
void qwen36_lin_split_qkv_broadcast_bf16(
    const void* conv_out,
    void*       q48,
    void*       k48,
    void*       v48,
    int S,
    cudaStream_t stream);

// Split linear-attention conv output for the chunk/WY GQA path.
// conv_out: (S, 10240) = Q(16*128), K(16*128), V(48*128)
// q16/k16:  contiguous (S, 16, 128)
// v48:      contiguous (S, 48, 128)
void qwen36_lin_split_qkv_gqa_bf16(
    const void* conv_out,
    void*       q16,
    void*       k16,
    void*       v48,
    int S,
    cudaStream_t stream);

// Split full-attention q_proj output:
// q_proj: (S, 24, 512) = [q_pre(256), gate(256)] per head.
// q_pre:  (S, 24, 256), contiguous.
// gate:   (S, 24*256), contiguous.
void qwen36_split_q_gate_bf16(
    const void* q_proj,
    void*       q_pre,
    void*       gate,
    int S,
    cudaStream_t stream);

// Fused Gated DeltaNet gate preparation:
//   beta = sigmoid(b)
//   g    = neg_exp_A_log[h] * log1p(exp(a + dt_bias[h]))
// Inputs a/b are (S, 48) bf16, per-head constants are (48) fp32,
// outputs beta/g are (S, 48) bf16.
void qwen36_gdn_gating_bf16(
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       g_out,
    void*       beta_out,
    int S,
    int num_heads,
    cudaStream_t stream);

void qwen36_gdn_gating_strided_bf16(
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       g_out,
    void*       beta_out,
    int S,
    int num_heads,
    int a_stride,
    int b_stride,
    cudaStream_t stream);

// Long-prefill fused path: read conv_out directly, broadcast Q/K,
// compute g/beta from a/b, and run the shared-memory Gated DeltaNet
// recurrent scan in one kernel.
void qwen36_gdn_chunk_from_conv_smem_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       state,
    void*       out,
    int S,
    int num_v_heads,
    bool use_qk_l2norm,
    cudaStream_t stream);

void qwen36_gdn_chunk_from_conv_smem_strided_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_A_log,
    const float* dt_bias,
    void*       state,
    void*       out,
    int S,
    int num_v_heads,
    int a_stride,
    int b_stride,
    bool use_qk_l2norm,
    cudaStream_t stream);

// Chunk/WY Gated DeltaNet building blocks. These are the native
// FlashRT replacement path for the Python/Triton FLA chunk implementation.
// First specialization targets Qwen3.6 shapes:
//   q/k:      (S, 16, 128) bf16
//   g/beta:   (S, 48) bf16
//   q/k out:  (S, 16, 128) bf16
//   g_cumsum: (S, 48) bf16
//   A/Ai:     (ceil(S / 64), 48, 64, 64) fp32
void qwen36_gdn_wy_norm_cumsum_bf16(
    const void* q16,
    const void* k16,
    const void* g,
    void*       q16_l2,
    void*       k16_l2,
    void*       g_cumsum,
    int S,
    cudaStream_t stream);

void qwen36_gdn_wy_norm_cumsum_pack_q_bf16(
    const void* q16,
    const void* k16,
    const void* g,
    void*       q16_l2,
    void*       k16_l2,
    void*       q_pack_hv,
    void*       g_cumsum,
    int S,
    cudaStream_t stream);

void qwen36_gdn_wy_norm_cumsum_pack_qk_bf16(
    const void* q16,
    const void* k16,
    const void* g,
    void*       q16_l2,
    void*       k16_l2,
    void*       q_pack_hv,
    void*       k_pack_hk,
    void*       g_cumsum,
    int S,
    cudaStream_t stream);

void qwen36_gdn_wy_kkt_b64_bf16(
    const void* k16_l2,
    const void* beta,
    const void* g_cumsum,
    void*       A,
    int S,
    cudaStream_t stream);

void qwen36_gdn_wy_solve_tril_b64_f32(
    const void* A,
    void*       Ai,
    int S,
    cudaStream_t stream);

void qwen36_gdn_wy_recompute_wu_b64_bf16(
    const void* k16_l2,
    const void* v48,
    const void* beta,
    const void* g_cumsum,
    const void* Ai,
    void*       w48,
    void*       u48,
    int S,
    cudaStream_t stream);

void qwen36_gdn_wy_chunk_h_b64_bf16(
    const void* k16_l2,
    const void* u48,
    const void* w48,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    int S,
    cudaStream_t stream);

void qwen36_gdn_wy_output_o_b64_bf16(
    const void* q16_l2,
    const void* k16_l2,
    const void* v_new,
    const void* h0,
    const void* g_cumsum,
    void*       out,
    int S,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
