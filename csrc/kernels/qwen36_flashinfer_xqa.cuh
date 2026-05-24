#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

void qwen36_flashinfer_xqa_bf16_fp8kv_spec(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* page_table,
    const uint32_t* seq_lens,
    const uint32_t* mask,
    void* out,
    uint32_t* semaphores,
    void* scratch,
    int max_seq_len,
    int q_seq_len,
    int sm_count,
    float q_scale,
    float kv_scale,
    bool enable_pdl,
    int64_t k_stride_page,
    int64_t k_stride_token,
    int64_t k_stride_head,
    cudaStream_t stream);
