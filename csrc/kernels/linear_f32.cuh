// Generic fp32-weight linear projections for small shapes.
//
// Two tiny projections with 32-bit float weights and a fused bias, differing in
// the activation and output dtypes. Both accumulate in fp32 with no TF32, which
// is what a reference `torch.addmm` with allow_tf32=False computes, and both are
// one pass per output element -- the shapes they serve are a handful of rows
// against a few hundred to a few thousand columns, where a library GEMM's setup
// costs more than the arithmetic.
//
// Weights are stored (N, K), an nn.Linear weight as it is held.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// out[n_row, n] = bf16( sum_k x[n_row, k] * W[n, k] + b[n] ), x and W fp32.
// One thread per output element.
void linear_f32_in_bf16_out(const float* x, const float* W, const float* b,
                            __nv_bfloat16* out, int rows, int N, int K,
                            cudaStream_t stream);

// out[t, n] = sum_k fp32(x[t, k]) * W[n, k] + b[n], x bf16, W and b fp32 and the
// output fp32. One warp per output element, so the k loop is split across 32
// lanes and the partial sums are reduced in registers -- a different summation
// order from the sequential one above, and the reason this is not the same
// kernel with a cast on each end.
void linear_bf16_in_f32_out(const __nv_bfloat16* x, const float* W, const float* b,
                            float* out, int rows, int N, int K,
                            cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
