// ============================================================================
//  FlashRT — vectorized SigLIP LayerNorm kernels (additive).
//
//  Register-resident variants of the two SigLIP per-layer LayerNorms:
//    * layer_norm_fp8_vec_fp16      (LN1 -> FP8 activations, attention input)
//    * layer_norm_mul_fp4_sfa_vec_fp16
//                                   (LN2 [x AWQ inv_s] -> NVFP4 + SFA, FFN
//                                    input; inv_s == nullptr for the plain
//                                    path)
//
//  The originals stream the row from memory once per stage (mean, variance,
//  normalize) with 2/4-byte accesses. These variants load the row once with
//  16-byte accesses into registers (one 16-element quant block per thread)
//  and run all three stages from registers. Reduction order differs from
//  the originals, so results match to floating-point rounding (ulp-level),
//  not bit-exactly; acceptance is by kernel-level parity and the end-to-end
//  cosine gates.
//
//  Requires dim % 16 == 0 and dim/16 <= 128 (one block per thread at 128
//  threads). Returns nonzero without launching otherwise so callers can
//  fall back to the original kernels.
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// LayerNorm(gamma, beta) -> FP8 e4m3 out [S, D].
int layer_norm_fp8_vec_fp16(
    const __half* x, const __half* gamma, const __half* beta,
    void* out_fp8, int seq_len, int dim, float eps, cudaStream_t stream);

// LayerNorm(gamma, beta) [* inv_s] -> NVFP4 packed + SFA tile-interleaved.
// Semantics match layer_norm_mul_fp4_sfa_fp16 (fp16 rounding before the
// quantizer) up to reduction order.
int layer_norm_mul_fp4_sfa_vec_fp16(
    const __half* x, const __half* gamma, const __half* beta,
    const __half* inv_s,
    void* packed, void* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
