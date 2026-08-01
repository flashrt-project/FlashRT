#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// A few rows of activation against the same 4-bit weight the decode GEMV reads.
//
// A speculative verify has to be the same function as the decode step it
// verifies, or the tokens it keeps are not the ones plain greedy emits. Two
// things were in the way. The verify read the dense weights at BF16 while
// decode reads them at 4 bits -- four times the traffic, and a different
// answer: measured logit cosine 0.988 between the two forwards. And the
// general W4A16 GEMM, which does read 4 bits, is not the same arithmetic
// either, and at these shapes it is 7.5 to 9.3 times off the GEMV: 250 us
// against 33 for an 8192x2048 projection, and flat in M, so it is not reading
// the weight at bandwidth at all.
//
// This is the decode GEMV with M rows of activation. The weight stream, the
// lane-to-block mapping, the unroll and the reduction order are unchanged --
// only the number of activation rows staged in shared memory and the number of
// accumulators a warp carries. The weight is what costs, and it is read once
// regardless of M, so a window of four verifies for what one costs.
//
// Because each output row accumulates in exactly the order the GEMV uses, the
// result at M=1 is bit-identical to it, and rows of a larger M agree with what
// the GEMV would have produced for each row on its own.
//
//   x     (M, K) bf16, row-major
//   W     (N, K/2) NVFP4 e2m1 nibbles
//   SFB   swizzled UE4M3 block scales, as bf16_weight_to_nvfp4_swizzled writes
//   out   (M, N) bf16
//   alpha weight per-tensor global scale
//
// K must be a multiple of 16, M at most 8. Returns 0 on success.
int w4a16_mrows_edge_sm120_bf16(
    const void*  x_bf16,
    const void*  W_packed,
    const void*  SFB,
    void*        out,
    int          M,
    int          N,
    int          K,
    float        alpha,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
