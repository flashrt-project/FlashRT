// Qwen3.6 AB96 bf16 matmul kernels (K=5120, N=96) plus the legacy
// bf16_matmul_qwen36_bf16 name.
//
// bf16_matmul_qwen36_bf16 is the historical name for the model-neutral small-M
// matmul. The implementation moved to bf16_matmul_bf16.cu (symbol
// bf16_matmul_bf16) as part of the generic-helper ownership cleanup (#112);
// this declaration is kept as a thin wrapper so existing Qwen3.6 call sites and
// the existing binding stay unchanged. The generic cuBLASLt BF16 GEMM
// (bf16_matmul_cublaslt_bf16) also moved to bf16_matmul_bf16.cuh.
//
//   D[m, n] = sum_k x[m, k] * W[n, k],   m in [0, M), n in [0, N)
//
// The AB96 kernels below remain Qwen3.6-specific: they hardcode N=96 / K=5120
// for the lin-attn unquantized projections (in_proj_qkv/z, out_proj) and the
// MTP tile shapes, so they stay in this Qwen3.6-named file.

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
