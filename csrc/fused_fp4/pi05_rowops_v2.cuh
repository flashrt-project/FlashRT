// ============================================================================
//  FlashRT — warp-per-row activation kernels for the Pi0.5 Thor encoder and
//  SigLIP paths (additive; the originals stay the default).
//
//  Every kernel here processes one row per warp with 16-byte loads, warp
//  shuffle reductions only (no shared memory, no __syncthreads), and the
//  hardware cvt for e2m1x2 / e4m3x2. The originals are one CTA per row with
//  block reductions and an 8-compare e2m1 chain; Nsight showed them
//  instruction-bound at ~50 instructions per element.
//
//  Semantics match the originals up to (a) fp32 reduction order and (b) the
//  e2m1 rounding of exact ties (hardware round-to-nearest-even vs the
//  original chain's round-toward-zero at the midpoints). Acceptance is by
//  kernel-level parity and the end-to-end cosine gates.
//
//  Requirements: D % 16 == 0, D/16 <= 128 (4 blocks of 16 per lane),
//  16-byte aligned rows. Nonzero return = not launched.
// ============================================================================
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// residual += x (fp16 write-back); out = rms_norm(residual) [* inv_s] -> NVFP4 + SFA
int rowops_residual_rms_mul_fp4_sfa_v2(
    __half* residual, const __half* x, const __half* inv_s,
    void* packed, void* sfa, int S, int D, cudaStream_t stream);

// out = rms_norm(x) [* inv_s] -> NVFP4 + SFA (no residual; pair with a beta=1 GEMM)
int rowops_rms_mul_fp4_sfa_v2(
    const __half* x, const __half* inv_s, void* packed, void* sfa, int S, int D,
    cudaStream_t stream);

// out = NVFP4(src) + SFA
int rowops_quantize_fp4_sfa_v2(
    const __half* src, void* packed, void* sfa, int N, int D,
    cudaStream_t stream);

// residual += x; out_fp8 = e4m3(rms_norm(residual) / descale)
int rowops_residual_rms_fp8_v2(
    __half* residual, const __half* x, void* out_fp8, int S, int D,
    const float* descale, cudaStream_t stream);

// out_fp8 = e4m3(rms_norm(x) / descale)
int rowops_rms_fp8_v2(
    const __half* x, void* out_fp8, int S, int D, const float* descale,
    cudaStream_t stream);

// out = NVFP4(LayerNorm(x; gamma, beta) [* inv_s]) + SFA
int rowops_layer_norm_mul_fp4_sfa_v2(
    const __half* x, const __half* gamma, const __half* beta,
    const __half* inv_s, void* packed, void* sfa, int S, int D, float eps,
    cudaStream_t stream);

// out_fp8 = e4m3(LayerNorm(x; gamma, beta))
int rowops_layer_norm_fp8_v2(
    const __half* x, const __half* gamma, const __half* beta,
    void* out_fp8, int S, int D, float eps, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
