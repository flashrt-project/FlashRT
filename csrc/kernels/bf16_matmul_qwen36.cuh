// bf16 row-major matmul (small-M) for Qwen3.6 — stream-invariant,
// deterministic. Designed as an add-only sibling of
// bf16_matvec_qwen36 to support the S=K verify path:
//
//   D[m, n] = sum_k x[m, k] * W[n, k],   m in [0, M), n in [0, N)
//
// Replaces a Python K-loop of bf16_matvec_qwen36_bf16 calls (which
// reads the full W weight K times) with a single launch that reads
// W exactly once and broadcasts across the M output rows. Eliminates
// the (M-1) × |W| redundant weight read at the lin-attn unquantized
// projections (in_proj_qkv 100MB, in_proj_z 60MB, out_proj 60MB),
// which is the dominant cost of the verify forward at S>=2 (profile:
// verify K=4 = 45ms vs S=1 = 28ms; the 17ms delta is this read).
//
// Stream-invariance and CUDA Graph compatibility match the matvec:
// each thread sums in fixed K-order with fp32 fma, no cuBLAS handle.
//
// Shapes:
//   x   : (M, K)      bf16, row-major
//   W   : (N, K)      bf16, row-major
//   out : (M, N)      bf16, row-major
//
// Launch: (ceil(N/8), M) blocks of 256 threads. One block computes
// 8 output elements (= 8 warps × 1 element/warp) for one (m, n-tile).

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt::kernels {

void bf16_matmul_qwen36_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M,
    int N,
    int K,
    cudaStream_t stream);

void bf16_matmul_qwen36_ab96_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_ab,
    __nv_bfloat16* out_ab,
    int M,
    cudaStream_t stream);

void bf16_matmul_qwen36_ab96_m4_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_ab,
    __nv_bfloat16* out_ab,
    int M,
    cudaStream_t stream);

void bf16_matmul_qwen36_ab96_m4_pair_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_ab,
    __nv_bfloat16* out_ab,
    int M,
    cudaStream_t stream);

void bf16_matmul_qwen36_ab96_lt_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_ab,
    __nv_bfloat16* out_ab,
    int M,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
