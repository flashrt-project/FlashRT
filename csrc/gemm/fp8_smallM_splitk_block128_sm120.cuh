// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace smallM_splitk {

// Block-128-scaled split-K FP8 e4m3 -> BF16 GEMM.
// A/B are block-128 FP8 quantized; AS=(M, K/128) and WS=(N/128, K/128) are
// fp32 per-128 scale tensors. The per-128-K-block scale is applied inside the
// partial kernel so block-128 accuracy is preserved. scratch is an fp32
// [k_split, M, N] buffer the caller must allocate. Returns 0 on success.
int splitk_b128_fp8_gemm_32x64x128_w4(const void* A, const void* B,
                                      const float* AS, const float* WS,
                                      void* D, int M, int N, int K, int Kb,
                                      int k_split, void* scratch,
                                      cudaStream_t stream);

}  // namespace smallM_splitk
}  // namespace gemm
}  // namespace flash_rt
