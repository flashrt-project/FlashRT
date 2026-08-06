// ============================================================================
//  FlashRT — NVFP4 GEMMs for the SigLIP FFN (SM100/SM110 block-scaled).
//
//  Two epilogue-fused kernels covering the vision-tower MLP:
//
//    Up:   D_fp4[M, N] = blockscale( gelu_tanh(A @ B^T + bias[N]) )
//          (packed e2m1 + UE4M3 SFA output feeds the Down GEMM directly,
//           replacing the fp16 hidden round-trip + separate quantize)
//    Down: D_fp16[M, N] = A @ B^T + bias[N] + C[M, N]
//          (residual accumulate, C may alias D)
//
//  A: NVFP4 packed activation [M, K] row-major + SFA.
//  B: NVFP4 packed weight [N, K] (column-major B tag) + SFB.
//  bias: fp16 [N].
//
//  Additive: separate instantiations; no shared GEMM files are modified.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

int cutlass_fp4_gemm_bias_gelu_fp4out(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_fp16,
    void*       D_packed,
    void*       D_SFD,
    int M, int N, int K,
    cudaStream_t stream);

int cutlass_fp4_gemm_bias_res_fp16(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_fp16,
    void const* C_fp16,          // residual source (may alias D)
    void*       D_fp16,
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
