#pragma once

#include <cuda_runtime.h>

namespace flash_rt::attention::sage2 {

int qk_int8_sv_f8_bf16_nhd_d128(
    const void* q_int8,
    const void* k_int8,
    const void* v_fp8,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    const void* v_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    float softmax_scale,
    cudaStream_t stream);

int qk_int8_sv_f16_bf16_nhd_d128(
    const void* q_int8,
    const void* k_int8,
    const void* v_half,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    float softmax_scale,
    cudaStream_t stream);

}  // namespace flash_rt::attention::sage2
