// ============================================================================
//  FlashRT — NVFP4 GEMMs with bf16 fused-bias epilogues (SM100/SM110).
//
//  bf16 companions of the fp16 fused-epilogue NVFP4 GEMMs, for pipelines
//  whose activations and biases are bf16 (GR00T N1.7 DiT). All three share
//  the proven skinny-M block-scaled mainloop (tile 128x64x256, cluster
//  1x1x1). A is row-major [M, K] packed e2m1, B is column-major [N, K]
//  packed e2m1 (A @ B^T, nn.Linear convention); SFA/SFB use the CUTLASS
//  Sm1xx tile-interleaved UE4M3 layout.
//
//  Additive: new symbols only; no existing kernel is modified.
// ============================================================================
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// D_bf16[M,N] = A @ B^T + bias[N]
int cutlass_fp4_gemm_bias_bf16(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_bf16,
    void* D_bf16,
    int M, int N, int K, cudaStream_t stream);

// D_bf16[M,N] = A @ B^T + bias[N] + C_bf16[M,N]   (residual; C may alias D)
int cutlass_fp4_gemm_bias_res_bf16(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_bf16,
    void const* C_bf16, void* D_bf16,
    int M, int N, int K, cudaStream_t stream);

// D_fp4[M,N], SFD = blockscale(gelu_tanh(A @ B^T + bias[N]))
// SFD is written in the SFA tile-interleaved layout over (M, N) so the
// output can feed the K side of a following NVFP4 GEMM directly.
int cutlass_fp4_gemm_bias_gelu_fp4out_bf16(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_bf16,
    void* D_packed, void* D_SFD,
    int M, int N, int K, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
