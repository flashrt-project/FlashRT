// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Cross-attention probabilities over a small set of distinct text keys,
// written as FP8 e4m3 operands for a projected-value GEMM (see
// xattn_probs.cu). q: [L, ld_q] bf16 (already normalised), head h at columns
// [h*hd, (h+1)*hd); keys: [nk_total, ld_k] bf16; log2_mult: [nk_total] log2 of
// each key's multiplicity. Row r belongs to batch element row_elem[r], whose
// keys are keys[elem_k0[b] .. elem_k0[b] + elem_nk[b]) (nk <= 32; every aligned
// 16-row tile must lie in one element) and whose probabilities go to
// P[r, c = elem_col0[b] + h*nk + j] as e4m3(p * p_mul * col_scale[c])
// (col_scale may be null). Other columns of P are not written. hd must be 128.
// Returns 0 on success.
int xattn_probs_fp8(const void* q, int ld_q, const void* keys, int ld_k,
                    const float* log2_mult, const int* row_elem, const int* elem_k0,
                    const int* elem_nk, const int* elem_col0, const float* col_scale,
                    void* P, int ld_p, int L, int heads, int hd, float scale, float p_mul,
                    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
