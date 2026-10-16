// SPDX-License-Identifier: Apache-2.0
//
// FlashDrive (Alpamayo 1.5) backend kernels for Thor (sm_110).
//
// FlashDrive keeps keys *un-rotated* in its static cache (layout [1, Hk, S, 128]
// per layer) and rotates the whole cache on every forward with a fixed position
// table. The backend keeps that cache as the source of truth and maintains a
// rotated attention copy, so every attention reads keys equal to FlashDrive's
// per-forward rotation:
//
//   qk prep       per projection row: Qwen3 head RMSNorm (BF16 rounding of the
//                 normalized value, then of the weight product); q rotated with
//                 BF16 tables (each product and the sum rounded to BF16, as
//                 FlashDrive's apply_rotary_pos_emb_single on BF16 tables); the
//                 normalized un-rotated k written to the FlashDrive cache, its
//                 rotation to the attention copy, v to the FlashDrive cache.
//   rope rows     rotate cached rows into an attention copy:
//                 kBf16Rule  BF16 tables, BF16 rounding of each product and of the
//                            sum (the VLM's view of the cache);
//                 kFp32Rule  FP32 tables, FP32 products and sum, one BF16
//                            rounding (the action expert's view: FP32 tables
//                            under autocast, cast to BF16 at attention).
//
// Row addressing: cache row of work row r = (row_map ? row_map[r] : row0 + r)
// + (slot_ptr ? *slot_ptr : 0); an optional device bound (n_ptr) skips rows
// r >= *n_ptr (CUDA-graph friendly variable lengths). All strides are in
// elements; 8-byte aligned pointers and strides that are multiples of 4.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// q/k/v      : projection outputs, row-major, leading dims q_ld / k_ld / v_ld (elements); bf16, or fp16 when
//              f16_in=1 (Marlin W4A8 outputs, rounded to bf16 on load exactly as .to(torch.bfloat16))
// q_norm_w/k_norm_w : [128] bf16
// cos/sin    : [rows, 128] position table rows of the work rows: bf16 (fp32_rule=0, the VLM) or fp32
//              (fp32_rule=1, the action expert: FP32 products and sum, one BF16 rounding)
// q_out      : [rows, Hq, 128] bf16, rotated
// k_raw      : FlashDrive cache keys, element (c, h, d) at c * kr_rs + h * kr_hs + d (normalized, un-rotated)
// k_rope     : attention copy, element (c, h, d) at c * k_rs + h * k_hs + d (rotated)
// v_out      : FlashDrive cache values, element (c, h, d) at c * v_rs + h * v_hs + d
int flashdrive_qk_prep_bf16(const void* q, const void* k, const void* v, int q_ld, int k_ld, int v_ld,
                            const void* q_norm_w, const void* k_norm_w, const void* cos, const void* sin, int fp32_rule,
                            int f16_in, void* q_out, void* k_raw, long long kr_rs, long long kr_hs,
                            void* k_rope, long long k_rs, long long k_hs,
                            void* v_out, long long v_rs, long long v_hs,
                            const long long* row_map, const long long* slot_ptr,
                            int rows, int Hq, int Hk, float eps, cudaStream_t stream);

// Consumers of FP16 Marlin outputs that apply FlashDrive's .to(torch.bfloat16) cast on load (identical results to
// the cast followed by the BF16 kernel):
//   rms_f16delta  residual [rows, dim] bf16 += bf16(delta fp16), out = Qwen3 RMSNorm(residual) * weight (the
//                 rounding contract of alpamayo15_rms_bf16)
//   silu_mul      out [n] bf16 = bf16(silu(bf16(gate))) * bf16(up), each product rounded to bf16 (silu_mul_qwen36_bf16)
int flashdrive_rms_f16delta_bf16(void* residual, const void* delta, const void* weight, void* out, int rows, int dim,
                                 float eps, cudaStream_t stream);
int flashdrive_silu_mul_f16in_bf16(const void* gate, const void* up, void* out, long long n, cudaStream_t stream);

// dst rows = rotation of src rows (same cache rows), Hk heads of 128.
// cos/sin: position table indexed by cache row ([S, 128], bf16 for fp32_rule=0, fp32 for fp32_rule=1).
// n_ptr  : optional int32 device bound on the work rows processed (r < *n_ptr).
int flashdrive_rope_rows(const void* src, long long s_rs, long long s_hs,
                         void* dst, long long d_rs, long long d_hs,
                         const void* cos, const void* sin, int fp32_rule,
                         const long long* row_map, long long row0, const long long* slot_ptr,
                         const int* n_ptr, int rows, int Hk, cudaStream_t stream);

// W4A8 activation preparation for FlashDrive's ParoQuant linears (bit-exact with the paroquant rotation op and
// vLLM's per-token INT8 quantization; see the .cu). For each of n_sets (1-3) rotation sets sharing input x
// ([rows, K] bf16, leading dim ld_x, K % 128 == 0): theta [8, K/2] fp16, pairs [8, K] int16, scales [K] fp16,
// igs = input_global_scale (fp32 device scalar); workspaces xr [rows, K] fp16 and part [rows, K/128] fp32;
// outputs xq [rows, K] int8 and a_scale [rows] fp32 (Marlin a_scales). Two launches (rotate, quantize).
int flashdrive_w4a8_act_prep(const void* x, int ld_x, int rows, int K, int n_sets,
                             const void* const* theta, const void* const* pairs, const void* const* scales,
                             const float* const* igs, void* const* xr, float* const* part, void* const* xq,
                             float* const* a_scale, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
