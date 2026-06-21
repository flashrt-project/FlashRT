// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini W4A16 dense GEMM (BF16 activation x NVFP4 weight), sm120. See .cu.
// y[M,N] = x[M,K] @ W[N,K]^T. x bf16; W NVFP4 packed (N, K/2) + swizzled UE4M3
// SFB + per-tensor alpha. Weight dequanted to bf16 in smem, bf16 m16n8k16 mma.
// K must be a multiple of 16. Precise (bf16 activation) + 4-bit weight read.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

int w4a16_gemm_sm120_bf16(
    const void*  X,
    const void*  W,
    const void*  SFB,
    void*        Y,
    int          M,
    int          N,
    int          K,
    float        alpha,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
