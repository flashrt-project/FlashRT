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
// FP32 softmax / accumulation, online over 64-key chunks. kv_static: the
// external K / V are not written by any kernel of the current PDL chain, so
// they are fetched before the grid-dependency wait. hd == 48 or 64.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace nvfp4_skinny {

int partials_attention(const float* partials, int splits, const float* a_gs, const float* w_gs,
                       const __nv_bfloat16* bias, int n_proj, const __nv_bfloat16* k_ext,
                       const __nv_bfloat16* v_ext, int kv_ld, int kv_seq, bool kv_static, __nv_bfloat16* out,
                       int rows, int heads, int hd, float scale, bool pdl, cudaStream_t stream);

}  // namespace nvfp4_skinny
}  // namespace flash_rt
