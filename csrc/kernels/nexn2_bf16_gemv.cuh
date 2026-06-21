// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini M=1 BF16 GEMV for SM120, memory-level-parallel.
//
// The qwen36 bf16_matvec is 1 warp/output with a single int4 load in flight
// per K-loop iteration (serial acc chain) -> ~40% of HBM BW on the Nex-N2
// decode shapes. This variant keeps the 1-warp-per-output layout but issues
// UNROLL int4 loads before consuming them, so each warp keeps several HBM
// requests in flight and the loop becomes bandwidth-bound. Same numerics
// (fp32 accumulate, bf16 out), drop-in for bf16_matvec_qwen36_bf16.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// y(1,N) = x(1,K) . W(N,K)^T, bf16, fp32 accumulate. K must be a multiple
// of 8 (int4 vectorization). Returns 0 on success, nonzero on arg error.
int nexn2_bf16_matvec_bf16(
    const void*  x,
    const void*  W,
    void*        out,
    int          N,
    int          K,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
