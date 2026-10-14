#ifndef OPENDM_DM05_FA2_FP8_WRAPPER_H
#define OPENDM_DM05_FA2_FP8_WRAPPER_H
#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif
void dm05_fa2_fp8_fwd_bf16_causal(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k, int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream,
    void* o_fp8_ptr, const float* o_fp8_scale_ptr,
    int o_fp8_batch_stride, int o_fp8_row_stride, int o_fp8_head_stride, int o_bf16_skip);

void dm05_fa2_fp8_fwd_bf16_noncausal(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k, int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream,
    void* o_fp8_ptr, const float* o_fp8_scale_ptr,
    int o_fp8_batch_stride, int o_fp8_row_stride, int o_fp8_head_stride, int o_bf16_skip);

#ifdef __cplusplus
}
#endif
#endif
