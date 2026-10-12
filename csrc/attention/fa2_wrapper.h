#ifndef FLASHRT_ATTENTION_FA2_WRAPPER_H
#define FLASHRT_ATTENTION_FA2_WRAPPER_H

#include <cuda_runtime_api.h>

#if defined(_WIN32)
#if defined(FLASHRT_FA2_NATIVE_BUILD)
#define FLASHRT_FA2_NATIVE_API __declspec(dllexport)
#elif defined(FLASHRT_FA2_NATIVE_SHARED)
#define FLASHRT_FA2_NATIVE_API __declspec(dllimport)
#else
#define FLASHRT_FA2_NATIVE_API
#endif
#elif defined(__GNUC__) && defined(FLASHRT_FA2_NATIVE_BUILD)
#define FLASHRT_FA2_NATIVE_API __attribute__((visibility("default")))
#else
#define FLASHRT_FA2_NATIVE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_fp16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_bf16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_bf16_window(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int window_left, int window_right, int num_sms,
    cudaStream_t stream);

FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_bf16_seqused(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, const void* seqused_k_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_bf16_seqused_splitkv(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, const void* seqused_k_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

// Short-query FA2 tile entry (bf16). head_dim in (0, 128]; seqused_k_ptr
// may be null. Dispatches to the narrow-query-tile traits rather than the
// vendored <128,64,4> default. See fa2_tile_inst/flash_fwd_smallq_bf16_sm80.cu.
FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_bf16_tile(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, const void* seqused_k_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_bf16_causal(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

// FlashRT local patch 4: siblings of fvk_attention_fa2_fwd_bf16{,_causal} that also
// write a static-FP8 (e4m3) copy of the attention output, so a consumer GEMM reads its
// quantized activation straight from the attention kernel instead of from a separate
// quantize pass. `o_fp8_scale_ptr` is the device scalar quantize_fp8_static would read;
// the FP8 values match that kernel bit for bit. Pass o_fp8_ptr = nullptr to get exactly
// the behaviour of the entry without the suffix. `o_bf16_skip != 0` additionally drops
// the bf16 store for callers whose only consumer is the FP8 copy.
FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_bf16_fp8out(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream,
    void* o_fp8_ptr, const void* o_fp8_scale_ptr,
    int o_fp8_batch_stride, int o_fp8_row_stride, int o_fp8_head_stride,
    int o_bf16_skip);

FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_bf16_causal_fp8out(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream,
    void* o_fp8_ptr, const void* o_fp8_scale_ptr,
    int o_fp8_batch_stride, int o_fp8_row_stride, int o_fp8_head_stride,
    int o_bf16_skip);

#ifdef FLASHRT_ENABLE_CHAMELEON
FLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_fp16_causal(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);
#endif

#ifdef __cplusplus
}
#endif

#endif  // FLASHRT_ATTENTION_FA2_WRAPPER_H
