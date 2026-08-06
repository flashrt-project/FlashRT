// Generic bf16 row-major matmul (small-M) — model-neutral.
//
//   out[m, n] = sum_k x[m, k] * W[n, k],   m in [0, M), n in [0, N)
//
// Two implementations share this header:
//   - bf16_matmul_bf16: stream-invariant, deterministic warp-per-output
//     kernel (fixed K-order fp32 fma, no cuBLAS handle, CUDA Graph safe).
//     Reads W exactly once and broadcasts across the M output rows.
//   - bf16_matmul_cublaslt_bf16: cuBLASLt BF16 GEMM with a per-(M,N,K)
//     cached plan, for the larger-M / throughput path.
//
// These were historically named under Qwen3.6 (bf16_matmul_qwen36*), but they
// are model-neutral by shape and semantics and are reused by Qwen3, Qwen3-VL
// and the Qwen3-VL SM89 FP8 path. The legacy name bf16_matmul_qwen36_bf16 is
// kept as a thin wrapper over bf16_matmul_bf16 (see bf16_matmul_qwen36.cu) so
// existing call sites and bindings keep working.
//
// Shapes:
//   x   : (M, K)      bf16, row-major
//   W   : (N, K)      bf16, row-major
//   out : (M, N)      bf16, row-major

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt::kernels {

void bf16_matmul_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M,
    int N,
    int K,
    cudaStream_t stream);

// ``max_algos`` bounds how many cuBLASLt candidates the first call for a shape
// times before it commits to one. 0 (the default, and what every existing call
// site passes) keeps the current behaviour: the count comes from
// FLASHRT_BF16_CUBLASLT_AUTOTUNE_ALGOS, defaulting to 8.
//
// A caller passes 1 to take the heuristic's own first choice and skip the
// timing loop. That matters beyond speed: timing is noisy, so different
// processes pick different algorithms, different algorithms reduce in
// different orders, and a model whose output is compared token for token then
// disagrees with itself across runs. Plans are cached per (M, N, K, max_algos),
// so one caller asking for a deterministic pick does not decide the algorithm
// for another that did not.
void bf16_matmul_cublaslt_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    int max_algos = 0);

}  // namespace flash_rt::kernels
