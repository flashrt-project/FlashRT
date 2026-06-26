// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace gemv_m1_sm89 {

// M=1 FP8 e4m3 -> BF16 GEMV with per-token activation block scale [K/128] and
// per-weight 128x128 block scale [N/128, K/128]. Matches official Qwen3-VL FP8
// checkpoints that store `.weight` + `.weight_scale_inv`. Decode-shape sibling
// of the M>1 fp8_block128_gemm_mma_sm89 kernel. Warp-per-output-row, A staged
// in smem, 16-byte coalesced B loads. Returns 0 on success.
#define DECL_BLOCK128(NAME) \
  int NAME(const void* A, const void* B, void* D, \
           int M, int N, int K, const float* act_scale, \
           const float* w_scale, float alpha, cudaStream_t stream)

DECL_BLOCK128(gemv_fp8_block128_m1_w4);
DECL_BLOCK128(gemv_fp8_block128_m1_w8);
DECL_BLOCK128(gemv_fp8_block128_m1_w16);

#undef DECL_BLOCK128

}  // namespace gemv_m1_sm89
}  // namespace gemm
}  // namespace flash_rt
