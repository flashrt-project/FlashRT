// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini W16A16 dense GEMM (BF16 activation x BF16 weight), sm120. See .cu.
// y[M,N] = x[M,K] @ W[N,K]^T, both bf16, fp32 register accumulate. Deterministic
// (single pass over K, no split-K / atomics) and precise (fp32 accumulate ==
// the .float() path's argmax) -- the precision-preserving fast replacement for
// the fp32/TF32 cutlass dense projections at prefill M. K multiple of 16.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

int nexn2_w16a16_gemm_bf16(
    const void*  X,
    const void*  W,
    void*        Y,
    int          M,
    int          N,
    int          K,
    float        alpha,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
