// SPDX-License-Identifier: Apache-2.0
//
// Alpamayo 1.5 (Thor sm_110) model-local kernels.
//
// Each kernel reproduces the rounding sequence of the official PyTorch module
// graph under torch.autocast(bfloat16), so the fused path stays numerically
// interchangeable with the reference:
//
//   vision RoPE      q/k BF16 -> FP32 rotate-half RoPE with FP32 tables ->
//                    one BF16 rounding (apply_rotary_pos_emb_vision).
//   expert RMSNorm   FP32 residual (+ optional BF16 delta) -> FP32 RMSNorm ->
//                    FP32 weight product -> one BF16 rounding (the autocast
//                    cast at the next linear).
//   expert QK prep   BF16 q/k -> Qwen3 head RMSNorm (BF16 rounding of the
//                    normalized value, then of the weight product) -> FP32
//                    RoPE with FP32 tables -> one BF16 rounding (autocast cast
//                    at SDPA); V is copied unchanged.
//   VLM RMSNorm      optional BF16 residual add (rounded), FP32 statistics,
//                    BF16 rounding of the normalized value, BF16 weight
//                    product (Qwen3VLTextRMSNorm on BF16 activations).
//   VLM QK prep      Qwen3 head RMSNorm as above -> BF16 RoPE with BF16
//                    tables (each product and the sum rounded to BF16) -> K/V
//                    written to the cache row selected by an optional device
//                    slot (CUDA-graph friendly decode).
//
//   W8A16 GEMM       opt-in FP8 decode-weight tier (not part of the BF16
//                    contract): y = bf16(scale[n] * sum_k e4m3(w[n,k]) * x[m,k])
//                    with exact e4m3 -> bf16 widening and FP32 tensor-core
//                    accumulation.
//
// All FP32 products and sums use explicit round-to-nearest intrinsics so no
// FMA contraction changes the result relative to PyTorch's elementwise ops.
// The module is built with --ftz=true, so FP32 denormals flush to zero (PyTorch
// eager keeps them); reductions use a different order than torch.mean.
// Vectorized 8-byte accesses: tensor pointers must be 8-byte aligned (checked).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// qkv  : [S, 3, H, D] bf16 (q, k rotated in place; v untouched)
// cos/sin: [S, D] fp32
// Returns 0 on success, nonzero on invalid arguments.
int alpamayo15_vision_rope_qk_bf16(void* qkv, const float* cos, const float* sin,
                                   int S, int H, int D, cudaStream_t stream);

// residual: [rows, dim] fp32 (updated in place when delta != nullptr)
// delta   : [rows, dim] bf16 or nullptr
// weight  : [dim] bf16
// out     : [rows, dim] bf16
int alpamayo15_res_rms_f32_bf16(float* residual, const void* delta, const void* weight,
                                void* out, int rows, int dim, float eps, cudaStream_t stream);

// qkv  : [rows, (Hq + 2*Hk) * 128] bf16 (q | k | v per row)
// q_norm_w, k_norm_w : [128] bf16
// cos/sin : [rows, 128] fp32
// q_out : [rows, Hq, 128] bf16
// k_out / v_out : K/V of row r written at cache row
//   (r / rows_per_batch) * batch_stride + r % rows_per_batch + slot[r / rows_per_batch]
//   (slot = 0 when slot_ptr is null), [Hk, 128] bf16 per row, rows k_row_stride elements apart
int alpamayo15_expert_qk_prep_bf16(const void* qkv, const void* q_norm_w, const void* k_norm_w,
                                   const float* cos, const float* sin,
                                   void* q_out, void* k_out, void* v_out, const long long* slot_ptr,
                                   int rows, int Hq, int Hk, int k_row_stride,
                                   int rows_per_batch, long long batch_stride, float eps,
                                   cudaStream_t stream);

// residual: [rows, dim] bf16, updated in place (residual += delta) when delta != nullptr
// out = RMSNorm(residual) * weight, [rows, dim] bf16
int alpamayo15_rms_bf16(void* residual, const void* delta, const void* weight, void* out,
                        int rows, int dim, float eps, cudaStream_t stream);

// qkv : [rows, (Hq + 2*Hk) * 128] bf16; cos/sin : [rows, 128] bf16
// q_out : [rows, Hq, 128] bf16
// K/V of row r go to cache row (r / rows_per_batch) * batch_stride + r % rows_per_batch
// + slot, rows kv_row_stride elements apart, where slot = *slot_ptr (int64 device
// scalar) when slot_ptr != nullptr, else 0.
int alpamayo15_vlm_qk_prep_bf16(const void* qkv, const void* q_norm_w, const void* k_norm_w,
                                const void* cos, const void* sin, void* q_out,
                                void* k_cache, void* v_cache, const long long* slot_ptr,
                                int rows, int Hq, int Hk, int kv_row_stride,
                                int rows_per_batch, long long batch_stride, float eps,
                                cudaStream_t stream);


// Skinny GEMM y[M, N] = x[M, K] @ dequant(w)^T for M <= 16 (decode projections).
// w_frag : e4m3 codes in mma.m16n8k16 A-fragment order, 16-byte aligned:
//          T = ceil(N / 16) row tiles (rows >= N zero), K / 32 column chunks,
//          32 lanes x 16 bytes each; lane L = 4g + c, byte j = 8kt + 2ri + e holds
//          w[16t + g + 8(ri & 1), 32kc + 16kt + 8(ri >> 1) + 2c + e]
// scale  : [N] fp32 per output row (fp8_fragment_layout pads it to T * 16 with zeros)
// x      : [M, K] bf16 row-major (4-byte aligned); y : [M, N] bf16 row-major
// K must be a multiple of 32.
int alpamayo15_w8a16_gemm_bf16(const void* w_frag, const float* scale, const void* x, void* y,
                               int M, int N, int K, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
