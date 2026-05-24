// Small Qwen3.6 hot-path helpers that do not belong to GEMM/attention files.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

void qwen36_embedding_lookup_bf16(
    const int64_t* token_ids,
    const __nv_bfloat16* embed,
    __nv_bfloat16* out,
    int rows,
    int hidden,
    cudaStream_t stream);

void qwen36_partial_rope_qk_bf16(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int rows,
    int q_heads,
    int k_heads,
    int head_dim,
    int rope_dim,
    cudaStream_t stream);

void qwen36_tq_prepare_scalars(
    const __half* k_norm,
    const __half* k_rnorm,
    const __half* v_norm,
    float* norm_k,
    float* coef_rnorm,
    float* norm_v,
    int n,
    float coef,
    cudaStream_t stream);

void qwen36_argmax_bf16(
    const __nv_bfloat16* logits,
    int64_t* argmax_out,
    int rows,
    int vocab,
    cudaStream_t stream);

void qwen36_spec_accept_greedy_bf16(
    const __nv_bfloat16* logits,
    const int64_t* drafts,
    int64_t* argmax_out,
    int* accept_n,
    int rows,
    int vocab,
    int spec_k,
    cudaStream_t stream);

void qwen36_spec_accept_partitioned_bf16(
    const __nv_bfloat16* logits,
    const int64_t* drafts,
    int64_t* argmax_out,
    int* accept_n,
    float* partial_vals,
    int* partial_idx,
    int rows,
    int vocab,
    int spec_k,
    int parts,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
