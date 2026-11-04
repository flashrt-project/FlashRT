// SPDX-License-Identifier: Apache-2.0
//
// Block-128-scaled FP8 GEMM on the vendored CUTLASS 2.x SM89 pipeline.
//
// CUTLASS 2.x applies one scalar alpha per GEMM, so its stock FP8 kernel can
// only run per-tensor scales -- and this model's closed-loop delivery gate
// requires the block-128 weight format. This kernel keeps the delivered
// weights and activation quantization byte-for-byte and grafts the two
// accumulator block-scale fold from the hand-written SM89 kernel into the
// CUTLASS mainloop: each 128-wide K iteration accumulates into a temporary
// fragment, which is then added to the running accumulator scaled by
// act_scale[row, kb] * w_scale[col/128, kb].
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace bs_cutlass_sm89 {

// A: [M,K] e4m3 row-major, act_scale [M, K/128]
// B: [N,K] e4m3 row-major, w_scale [N/128, K/128]
// D: [M,N] bf16 row-major
// Returns 0 on success, negative on an unsupported shape or launch error.
int fp8_bs_gemm_cutlass_sm89(const void* A, const void* B, void* D,
                             int M, int N, int K,
                             const float* act_scale, const float* w_scale,
                             cudaStream_t stream);

// Same GEMM with each 128-K block summed by the double-rate FP16-accumulating
// e4m3 MMA and folded into FP32. The block sum must fit FP16: B must be
// quantized to |w| <= 1 per 128x128 tile (w_scale [N/128, K/128] = tile
// amax), so that with e4m3 activations |sum| <= 128 * 448 = 57344.
// N % 128 == 0, M % 16 == 0; act_scale must be readable up to the next
// multiple of 64 rows.
int fp8_bs_gemm_cutlass_f16blk_sm89(const void* A, const void* B, void* D,
                                    int M, int N, int K,
                                    const float* act_scale, const float* w_scale,
                                    cudaStream_t stream);

// fp8_bs_gemm_cutlass_f16blk_sm89 on a 64-deep K tile with the fold every two
// iterations (two CTAs per SM). Numerically equivalent (one bf16 ulp).
int fp8_bs_gemm_cutlass_f16blk_k64_sm89(const void* A, const void* B, void* D,
                                        int M, int N, int K,
                                        const float* act_scale, const float* w_scale,
                                        cudaStream_t stream);

// fp8_bs_gemm_cutlass_f16blk_k64_sm89 with bias + gelu-tanh + the FP8
// block-128 quantiser fused into the epilogue (fp8_bs_gemm_gelu_sm89.cu):
// writes the FP8 activation [M, N] and its scales [M, N/128], bit-identical to
// the GEMM followed by gelu_tanh_bias_to_fp8_block128_warp_bf16.
int fp8_bs_gemm_cutlass_f16blk_k64_gelu_fp8_sm89(const void* A, const void* B,
                                                 const void* bias, void* out,
                                                 float* out_scale, int M, int N, int K,
                                                 const float* act_scale,
                                                 const float* w_scale,
                                                 cudaStream_t stream);

// fp8_bs_gemm_cutlass_f16blk_k64_sm89 whose epilogue updates the residual
// stream in place: h = bf16(h + (bf16(y + bias)) * (ss_row + tp)) with the
// arithmetic of wan_gated_residual_sstp_bias_bf16, or h = bf16(h + bf16(y +
// bias)) (wan_gated_residual_bias_bf16) when ss_row and tp are null. bias may
// be null. tp_bcast reads tp's first row for every row.
int fp8_bs_gemm_cutlass_f16blk_k64_residual_sm89(const void* A, const void* B, int M, int N,
                                                 int K, const float* act_scale,
                                                 const float* w_scale, void* residual,
                                                 const void* bias, const float* ss_row,
                                                 const float* tp, int tp_bcast,
                                                 cudaStream_t stream);

// Stage-depth variants (2 and 4 stages) of the same kernel, for the
// weight-bound M=128 video shapes.
int fp8_bs_gemm_cutlass_sm89_s2(const void* A, const void* B, void* D,
                                int M, int N, int K,
                                const float* act_scale, const float* w_scale,
                                cudaStream_t stream);
int fp8_bs_gemm_cutlass_sm89_s4(const void* A, const void* B, void* D,
                                int M, int N, int K,
                                const float* act_scale, const float* w_scale,
                                cudaStream_t stream);

}  // namespace bs_cutlass_sm89
}  // namespace gemm
}  // namespace flash_rt
