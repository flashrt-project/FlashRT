#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

// ── INT8 KV cache for Qwen3-VL q=1 decode attention (Orin/Ampere) ──
// KV rows are quantized per-(position, kv-head): one bf16 scale per 128-elem
// row (scale = amax/127). Decode attention reads int8 K/V + scales (half the
// HBM bytes of bf16 KV) with a flash-decoding split over KV chunks.

// Quantize n_rows contiguous 128-elem bf16 rows to int8 + per-row bf16 scale.
// Layout-agnostic: used for the post-prefill bulk pass (n_rows = L*S*NKV, the
// whole contiguous cache prefix per K/V) and the per-step row pass (n_rows =
// NKV at one (layer, pos)).
void qwen3_kv_rows_quant_int8(
    const __nv_bfloat16* src, int8_t* dst, __nv_bfloat16* scales,
    int n_rows, cudaStream_t stream);

// Flash-decoding partial pass: q=1, GQA (16 Q / 8 KV heads, head_dim 128).
//   q        : [16,128] bf16 (Q_buf row)
//   k8/v8    : [kv_len, 8, 128] int8 (layer base)
//   ks/vs    : [kv_len, 8] bf16 row scales
//   part_o   : [n_chunks, 16, 128] fp32
//   part_m/l : [n_chunks, 16] fp32
// grid (8 kv-heads, n_chunks); one block = 2 q-heads x 128 dims.
void qwen3_attn_decode_int8kv_partial(
    const __nv_bfloat16* q, const int8_t* k8, const int8_t* v8,
    const __nv_bfloat16* ks, const __nv_bfloat16* vs,
    float* part_o, float* part_m, float* part_l,
    int kv_len, int n_chunks, float softmax_scale, cudaStream_t stream);

// Combine pass: merge chunk partials into O (bf16 [16,128]).
void qwen3_attn_decode_int8kv_combine(
    const float* part_o, const float* part_m, const float* part_l,
    __nv_bfloat16* out, int n_chunks, cudaStream_t stream);

}  // namespace flash_rt::kernels
