// ============================================================================
//  FlashRT — NVFP4 GEMM with fp32 per-column bias and fp16 output
//  (SM100/SM110).
//
//  D_f16[M, N] = A @ B^T + bias[N]. For hosts that keep biases in fp32 but
//  consume the projection in fp16 (e.g. attention inputs cast for flash
//  attention): the fp16 conversion happens once in the epilogue from the
//  fp32 accumulator, matching an fp32-output GEMM followed by an fp16 cast
//  bit for bit.
// ============================================================================
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// A: [M, K] NVFP4 packed row-major + SFA (tile-interleaved).
// B: [N, K] NVFP4 packed column-major + SFB.
// bias_f32: [N] fp32, broadcast over rows. D_f16: [M, N] fp16 row-major.
// Returns 0 on success; CUTLASS status | stage flag otherwise.
int gemm_bias_f16out(
    const void * A_packed, const void * SFA,
    const void * B_packed, const void * SFB,
    const void * bias_f32, void * D_f16,
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
