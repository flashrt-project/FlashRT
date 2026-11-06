// SPDX-License-Identifier: Apache-2.0
//
// Fused q_norm + RoPE + Q_buf write
// Fused k_norm + RoPE + KV_cache write (V copy inline)
//
// Replaces the per-decode-layer chain
//   rms_norm(q_pre, q_norm_w, q_norm_out)
//   _rope_apply_inline(q_norm_out -> q_rot)        [6 aten ops]
//   Q_buf.copy_(q_rot)
//   rms_norm(k_pre, k_norm_w, k_norm_out)
//   _rope_apply_inline(k_norm_out -> k_rot)        [6 aten ops]
//   K_cache[L, cur_pos].copy_(k_rot)
//   V_cache[L, cur_pos].copy_(v_slice)
// with two kernel launches, saving ~14 launches / layer × 36 layers.
//
// Add-only — does NOT modify the existing rms_norm or rope kernels;
// they remain available for the prefill / non-fused paths.
//
// Constraints:
//   * head_dim must be 128 (Qwen3-8B value); kernel hardcodes this.
//   * S = 1 (decode hot path); prefill keeps the existing chain.
//   * cos/sin tensors are (head_dim/2,) = (64,) BF16 — same shape used
//     by `_rope_apply_inline`.
//
// Math (Qwen3 RMSNorm, no 1+w; full-RoPE):
//   x_normed[d] = x[d] * rsqrt(sum_sq / head_dim + eps) * w[d]
//   rotate_half(x)[d] = -x[d + half]   if d < half
//                     =  x[d - half]   if d >= half
//   x_out[d] = x_normed[d] * cos[d % half] + rotate_half(x_normed)[d] * sin[d % half]

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Fused q_norm + RoPE + Q_buf write (S=1 decode).
//
//   q_pre     : (n_q_heads, 128) bf16  — output of fused QKV GEMM (q part)
//   q_norm_w  : (128,)            bf16
//   cos       : (64,)             bf16 — half-head_dim
//   sin       : (64,)             bf16
//   q_buf_dst : (n_q_heads, 128) bf16  — staging for FA2
//
// Returns 0 on success.
int qwen3_q_norm_rope_qstage_bf16(
    const void* q_pre,
    const void* q_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    int         n_q_heads,
    float       eps,
    cudaStream_t stream);

// Fused q_norm + RoPE + Q_buf write and k_norm + RoPE + KV_cache write in
// one launch. Decode-only S=1 path, head_dim hardcoded at 128.
int qwen3_qk_norm_rope_kvwrite_bf16(
    const void* q_pre,
    const void* k_pre,
    const void* v_pre,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         n_q_heads,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream);

// Batched fused q_norm + RoPE + Q_buf write and k_norm + RoPE + KV_cache
// write for prefill (S>=1). q_pre / k_pre / v_pre may be row-strided views
// into a larger qkv tensor; row strides are in elements, not bytes. cos/sin
// are (S, 64) bf16 and outputs are contiguous per-position rows in Q_buf /
// K_cache / V_cache.
int qwen3_qk_norm_rope_kvwrite_batched_bf16(
    const void* q_pre,
    const void* k_pre,
    const void* v_pre,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         seq_len,
    int         q_pre_row_elems,
    int         k_pre_row_elems,
    int         v_pre_row_elems,
    int         q_dst_row_elems,
    int         kv_dst_row_elems,
    int         n_q_heads,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream);

// Fused k_norm + RoPE + K_cache write + V_cache write (S=1 decode).
//
//   k_pre        : (n_kv_heads, 128) bf16
//   v_pre        : (n_kv_heads, 128) bf16
//   k_norm_w     : (128,)             bf16
//   cos / sin    : (64,)              bf16
//   k_cache_dst  : pointer to K_cache[L, cur_pos] = base of (n_kv, 128)
//   v_cache_dst  : pointer to V_cache[L, cur_pos] = base of (n_kv, 128)
//
// Returns 0 on success.
int qwen3_k_norm_rope_kvwrite_bf16(
    const void* k_pre,
    const void* v_pre,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream);

// Prefill (S>1) batched variants. grid = (n_heads, S); one block per
// (row, head). q/k/v read with `in_row_stride` (= qkv_N for the fused QKV
// output) so the strided q/k/v slices are consumed in place — no
// contiguous copy. cos/sin are (S, 64); outputs are written with the
// destination row stride (q_buf: n_q*128; K/V cache: cache_row_stride).
// Folds the per-layer rms_norm + multi-op RoPE + Q/K/V copies into 2
// launches. head_dim hardcoded 128. Returns 0 on success.
int qwen3_q_norm_rope_qstage_prefill_bf16(
    const void* q_pre,
    const void* q_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    int         n_q_heads,
    int         S,
    int         in_row_stride,
    int         out_row_stride,
    float       eps,
    cudaStream_t stream);

int qwen3_k_norm_rope_kvwrite_prefill_bf16(
    const void* k_pre,
    const void* v_pre,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         n_kv_heads,
    int         S,
    int         in_row_stride,
    int         cache_row_stride,
    float       eps,
    cudaStream_t stream);

// v3: warp-per-row (32 threads) with vectorized contiguous-4 (uint2) access —
// one 256B coalesced transaction per warp. Same semantics; NOT bit-identical
// (~1 ULP). head_dim=128 only.
int qwen3_q_norm_rope_qstage_prefill_v3_bf16(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, int n_q_heads, int S, int in_row_stride,
    int out_row_stride, float eps, cudaStream_t stream);

int qwen3_k_norm_rope_kvwrite_prefill_v3_bf16(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride,
    float eps, cudaStream_t stream);

// v3 + FOLDED per-token e4m3 emit (STEP D fp8 prefill attention). Same RMS+RoPE
// as v3_bf16, but also writes e4m3 Q/K (NHD [S,H,128]) + per-(token,head) scale
// ([H,S] head-major), with no extra HBM round-trip. The K variant also casts V
// to fp16 (NHD [S,Hkv,128]), folding the prefill V cast. bf16 dst still written.
int qwen3_q_norm_rope_qstage_prefill_v3_fp8(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, void* q8_dst, void* q_scale_dst,
    int n_q_heads, int S, int in_row_stride, int out_row_stride, float eps,
    cudaStream_t stream);

int qwen3_k_norm_rope_kvwrite_prefill_v3_fp8(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    void* k8_dst, void* k_scale_dst, void* v_fp16_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride, float eps,
    cudaStream_t stream);

// Direct-e4m3 variants: no per-token scale (post-norm Q/K sit inside e4m3's
// dynamic range, so the cast is direct) and V is emitted as e4m3 too — the
// operand set of the all-fp8 attention (fmha_fp8_causal_gqa_nhd_d128).
int qwen3_q_norm_rope_qstage_prefill_v3_fp8_direct(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, void* q8_dst,
    int n_q_heads, int S, int in_row_stride, int out_row_stride, float eps,
    cudaStream_t stream);

int qwen3_k_norm_rope_kvwrite_prefill_v3_fp8_direct(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    void* k8_dst, void* v8_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride, float eps,
    cudaStream_t stream);

// Single-launch form of the two direct producers above for the all-fp8 route
// (grid.y selects Q vs K/V; no cache-shaped bf16 scratch is written).
int qwen3_qk_norm_rope_prefill_v3_fp8_direct(
    const void* q_pre, const void* k_pre, const void* v_pre,
    const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin,
    void* q8_dst, void* k8_dst, void* v8_dst,
    int n_q_heads, int n_kv_heads, int S, int in_row_stride, float eps,
    cudaStream_t stream);

// Device-position variant: cache slot = K_cache_base + (*cur_pos) * row_elems,
// so one captured graph serves every decode position (host bumps *cur_pos
// before each replay). row_elems = n_kv * 128. Returns 0 on success.
int qwen3_k_norm_rope_kvwrite_devpos_bf16(
    const void* k_pre,
    const void* v_pre,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       k_cache_base,
    void*       v_cache_base,
    const void* cur_pos,
    int         row_elems,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
