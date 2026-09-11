// ============================================================================
//  FlashRT — NVFP4 GEMM pair for a SigLIP-style vision-tower FFN with fp32
//  bias/residual boundaries (SM100/SM110).
//
//  Variant of cutlass_fp4_gemm_siglip_ffn_sm100 for hosts that keep the FFN
//  bias and residual tensors in fp32 (rather than fp16): the Up projection
//  fuses bias + tanh-GELU and emits FP4 (e2m1) packed output + SFD, and the
//  Down projection fuses bias + fp32 residual add with fp32 output. The
//  CUTLASS workspace is cached per shape instead of allocated per call, so
//  steady-state calls are graph-capture safe.
//
//    Up:   D_fp4[M, N] = blockscale( gelu_tanh(A @ B^T + bias[N]) )
//    Down: D_f32[M, N] = A @ B^T + bias[N] + beta * C_f32[M, N]
// ============================================================================
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// A: [M, K] NVFP4 packed row-major + SFA (tile-interleaved).
// B: [N, K] NVFP4 packed column-major + SFB.
// bias_f32: [N] fp32, broadcast over rows.
// D_packed: [M, N] NVFP4 packed row-major; D_SFD: SFD tile-interleaved.
// Returns 0 on success; CUTLASS status | stage flag otherwise.
int siglip_ffn_up_gelu_fp4out(
    const void * A_packed, const void * SFA,
    const void * B_packed, const void * SFB,
    const void * bias_f32,
    void * D_packed, void * D_SFD,
    int M, int N, int K,
    cudaStream_t stream);

// C_f32/D_f32: [M, N] fp32 row-major (may alias). beta scales C.
int siglip_ffn_down_bias_res_f32(
    const void * A_packed, const void * SFA,
    const void * B_packed, const void * SFB,
    const void * bias_f32,
    const void * C_f32, void * D_f32,
    int M, int N, int K,
    cudaStream_t stream, float beta);

// Down configuration with beta = 0: D = A@B + bias (no residual read).
int gemm_bias_f32out(
    const void * A_packed, const void * SFA,
    const void * B_packed, const void * SFB,
    const void * bias_f32, void * D_f32,
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
