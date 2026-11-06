// SPDX-License-Identifier: Apache-2.0
#pragma once
// Small-row attention that consumes the K-split FP32 partials of a skinny
// NVFP4 projection GEMM directly (no Q/K/V store pass), for action-head
// transformers with a few dozen query rows (DiT self / cross attention).
//
//   q = (sum_s partials[s][row][h*hd + d]) * a_gs * w_gs + bias      (cols [0, H*hd))
//   self  (k_ext == nullptr): K / V from the same partials, cols [H*hd, 2H*hd) and
//                             [2H*hd, 3H*hd), kv_seq == rows
//   cross (k_ext != nullptr): K / V BF16 [kv_seq, kv_ld] external rows, head h
//                             at column h*hd
//   out[row, h*hd + d] = softmax(q K^T * scale) V   (BF16 [rows, H*hd])
//
// out4 / out_sf (optional): emit the result as the next skinny GEMM's NVFP4
// operand instead (the BF16-rounded values quantized per 16 columns with a
// unit global scale, row-major e2m1 [rows, H*hd/2] + UE4M3 [rows, H*hd/16]),
// bit-identical to [BF16 out -> nvfp4_skinny gemm_bf16_act with a_gs = 1].
// FP32 softmax / accumulation, online over 64-key chunks. kv_static: the
// external K / V are not written by any kernel of the current PDL chain, so
// they are fetched before the grid-dependency wait. hd == 48 or 64.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace nvfp4_skinny {

int partials_attention(const float* partials, int splits, const float* a_gs, const float* w_gs,
                       const __nv_bfloat16* bias, int n_proj, const __nv_bfloat16* k_ext,
                       const __nv_bfloat16* v_ext, int kv_ld, int kv_seq, bool kv_static, __nv_bfloat16* out,
                       int rows, int heads, int hd, float scale, bool pdl, cudaStream_t stream,
                       uint8_t* out4 = nullptr, uint8_t* out_sf = nullptr);

// Self-attention with its head's Q/K/V projection in the same launch: A/SFA is
// the [rows, K] NVFP4 operand, W/SFB the fragment-major [3 * heads * hd, K]
// projection weights (nvfp4_skinny_pack_weight layout). Equivalent to
// nvfp4_skinny gemm (single split) followed by partials_attention.
// hd == 64, K == 768, rows <= 48.
int qkv_self_attention(const void* A, const void* SFA, const void* W, const void* SFB, const float* a_gs,
                       const float* w_gs, const __nv_bfloat16* bias, __nv_bfloat16* out, int rows, int heads,
                       int hd, int K, float scale, bool pdl, cudaStream_t stream, uint8_t* out4 = nullptr,
                       uint8_t* out_sf = nullptr);

// Cross attention with its Q projection in the same launch (K/V from the
// external cache, as partials_attention with kv_static). splits is the K-split
// count of the standalone projection config it replaces (3: KC = 256).
int q_cross_attention(const void* A, const void* SFA, const void* W, const void* SFB, const float* a_gs,
                      const float* w_gs, const __nv_bfloat16* bias, const __nv_bfloat16* k_ext,
                      const __nv_bfloat16* v_ext, int kv_ld, int kv_seq, __nv_bfloat16* out, int rows, int heads,
                      int hd, int K, int splits, float scale, bool pdl, cudaStream_t stream,
                      uint8_t* out4 = nullptr, uint8_t* out_sf = nullptr);

}  // namespace nvfp4_skinny
}  // namespace flash_rt
