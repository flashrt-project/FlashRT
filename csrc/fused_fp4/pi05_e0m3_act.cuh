// ============================================================================
//  FlashRT — E0M3 (uniform INT4) decoder activation quantizers, optional
//  per-16 Hadamard rotation (RHT).
//
//  E0M3 variants of the four decoder activation quantize exits:
//    * pi05_adarms_fp4_sfa_native_fp16        -> pi05_adarms_e0m3_sfa_fp16
//    * pi05_gate_res_adarms_fp4_sfa_native_.. -> pi05_gate_res_adarms_e0m3_..
//    * quantize_fp4_dynamic_sfa_fp16_vec      -> quantize_e0m3_dynamic_sfa_vec
//    * gate_geglu_fp4_sfa_vec_fp16 (GELU mul) -> gate_geglu_e0m3_sfa_vec_fp16
//
//  Identical math to the originals up to the element encoder: per-16 amax,
//  UE4M3 block scale (amax/7 for the uniform +-7 grid), SFA tile-interleaved
//  scale write. With rht=1 each 16-value block is transformed by the
//  symmetric 16x16 Hadamard matrix scaled by 1/4 (orthonormal) BEFORE
//  quantization; applying the same rotation to the weights offline leaves
//  the GEMM mathematically unchanged while gaussianizing the per-block
//  distribution for the uniform grid.
//
//  Additive: does NOT modify norm_silu_fp4_sfa.cu, quantize_fp4_sfa_vec.cu,
//  or silu_mul_fp4_sfa_vec.cu.
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// AdaRMS(x, style) -> E0M3 packed + SFA, copies style gate (layer-0 entry).
void pi05_adarms_e0m3_sfa_fp16(
    const __half* x, const __half* style,
    uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, int use_rht, cudaStream_t stream);

// residual += x * prev_gate; AdaRMS(residual, style) -> E0M3 packed + SFA.
void pi05_gate_res_adarms_e0m3_sfa_fp16(
    const __half* x, const __half* prev_gate, __half* residual,
    const __half* style,
    uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, int use_rht, cudaStream_t stream);

// GELU(gate) * up on merged [S, 2H] -> E0M3 packed + SFA over H.
int gate_geglu_e0m3_sfa_vec_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, int use_rht, cudaStream_t stream);

}  // namespace fused_fp4

namespace fp4 {

// Vectorized fp16 [N, D] -> E0M3 packed + SFA/SFB (attention-context exit).
int quantize_e0m3_dynamic_sfa_fp16_vec(
    const void* src_fp16, void* dst_packed, void* dst_sfa,
    int N, int D, bool is_sfb, int use_rht, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
