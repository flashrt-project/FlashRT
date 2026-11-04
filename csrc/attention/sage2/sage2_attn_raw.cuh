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
#ifdef FLASHRT_ENABLE_LINGBOT_VA_SM89

// Same contract, but with explicit batch strides in elements. The engine's
// KV cache is a fixed-capacity [B, total, H, D] buffer whose live window is a
// prefix [0, W); the dense batch stride the plain entry assumes would be W*H*D.
int qk_int8_sv_f16_bf16_nhd_d128_strided(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_batch_stride,
    int o_batch_stride,
    cudaStream_t stream);

// V-fp8 variant with explicit batch strides and a caller-fixed V padded-K
// stride, so the per-channel transposed V cache can have a constant layout
// while the live window length changes. scale_per_head: q_scale/k_scale are
// [num_heads] static per-head scales instead of the per-warp-block layout.
int qk_int8_sv_f8_bf16_nhd_d128_strided(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    float pad_extra = 0.0f,
    bool pad_exact = false,
    void* out_fp8 = nullptr,
    float* out_fp8_scale = nullptr,
    bool scale_per_head = false);

// Same contract with a 32-row / 16-row-per-warp query tile: for S_q <= 32 the
// 128-row tile spends 4-8x the MMA and softmax work the query needs. Rejects
// seqlen_q > 32 with -3.
int qk_int8_sv_f8_bf16_nhd_d128_strided_sq(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    float pad_extra = 0.0f,
    bool pad_exact = false,
    void* out_fp8 = nullptr,
    float* out_fp8_scale = nullptr,
    bool scale_per_head = false);

// Same contract with the 128-row query tile split over eight 16-row warps
// instead of four 32-row warps. Bit-identical output.
int qk_int8_sv_f8_bf16_nhd_d128_strided_w16(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    float pad_extra = 0.0f,
    bool pad_exact = false,
    void* out_fp8 = nullptr,
    float* out_fp8_scale = nullptr,
    bool scale_per_head = false);

// Same contract with a 64-row query tile over four 16-row warps: twice the
// CTAs of the 128-row tiles, so the batch-2 (CFG) video call fills 96 CTAs
// instead of 48 on 34 SMs. Bit-identical output.
int qk_int8_sv_f8_bf16_nhd_d128_strided_q64(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    float pad_extra = 0.0f,
    bool pad_exact = false,
    void* out_fp8 = nullptr,
    float* out_fp8_scale = nullptr,
    bool scale_per_head = false);

// The 128x32 core (tile 0), the 32x16 short-query core (tile 1, four times
// the CTAs for a single-batch call) or the 128x16 eight-warp core (tile 2), also
// writing the log2-domain log-sum-exp per (head, row) into `lse`
// ([B, H, S_q], fp32). Plain bf16 output only.
// The strided QK-int8 / PV-fp8 core with K on a ring: key j of the call is K
// row (ring_origin[0] + ring_start + j) mod ring_R ([B, rows, H, D],
// k_batch_stride = rows * H * D); the writer mirrors rows [0, 64) at
// [ring_R, ring_R + 64), so rows >= ring_R + 64. V is the usual [B, D, H,
// v_padded_k] window starting at the V pointer. lse != nullptr selects the LSE
// variant (tile 0, 1, 2 as qk_int8_sv_f8_bf16_nhd_d128_strided_lse); otherwise
// tile 0-3 as sage2_attn_self_f8_tile. No pad_extra.
int qk_int8_sv_f8_bf16_nhd_d128_strided_ring(
    const void* q_int8, const void* k_int8, const void* v_fp8, void* out_bf16,
    float* lse, const void* q_scale, const void* k_scale, const void* v_scale,
    int batch, int seqlen_q, int seqlen_k, int num_heads, float softmax_scale,
    int q_batch_stride, int k_batch_stride, int v_padded_k, int o_batch_stride,
    cudaStream_t stream, void* out_fp8, float* out_fp8_scale, bool scale_per_head,
    int tile, const int* ring_origin, int ring_R, int ring_start);

int qk_int8_sv_f8_bf16_nhd_d128_strided_lse(
    const void* q_int8,
    const void* k_int8,
    const void* v_fp8,
    void* out_bf16,
    float* lse,
    const void* q_scale,
    const void* k_scale,
    const void* v_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    float softmax_scale,
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    bool scale_per_head = false,
    int tile = 0);

// Private 16-key segment per group of 16 query rows, merged with a shared
// segment's normalised output and LSE (sage2_tail_merge.cu). Emits the FP8
// block-128 activation (one scale per (row, head)) and optionally bf16.
int sage2_tail_merge_fp8(const void* q8, const void* k8, const void* v16,
                         const void* o_shared, const float* lse_shared,
                         const float* q_scale, const float* k_scale, bool per_head,
                         float sm_scale, int groups, int group_rows, int num_heads,
                         void* out_fp8, float* out_scale, void* out_bf16,
                         cudaStream_t stream, int rs_reduce = 0);
#endif  // FLASHRT_ENABLE_LINGBOT_VA_SM89

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

int qk_int8_sv_f16_bf16_gqa_nhd_d256(
    const void* q_int8,
    const void* k_int8,
    const void* v_half,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream);

int qk_int8_sv_f8_bf16_gqa_nhd_d256(
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
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream);

}  // namespace flash_rt::attention::sage2
