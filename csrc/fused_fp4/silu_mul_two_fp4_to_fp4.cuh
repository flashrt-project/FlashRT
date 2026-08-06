// ============================================================================
//  P1 split-GU FFN combiner: silu(dequant(gate_fp4)) * dequant(up_fp4)
//  → quant_block(...) → fp4 packed + SFA tile-interleaved.
//
//  Inputs are produced by separate fp4out NVFP4 GEMMs (gate_proj, up_proj).
//  Output feeds the next-stage Down NVFP4 GEMM as A.
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flash_rt {
namespace fused_fp4 {

// silu(gate) * up — per-block FP4 quant — for [seq_len, H] hidden buffers.
// gate_packed/gate_sfa  : FP4 + SFA from gate_proj GEMM (shape S × H)
// up_packed/up_sfa      : FP4 + SFA from up_proj   GEMM (shape S × H)
// out_packed/out_sfa    : FP4 + SFA, ready for Down GEMM A operand
void silu_mul_two_fp4_to_fp4(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream);

// Same as above + per-input-channel multiply by inv_s (AWQ Down activation
// scaling). Applied to the silu_mul fp32 result BEFORE per-block FP4 quant
// so the per-block amax reflects the post-AWQ distribution. inv_s is
// fp16 [H], shared across all rows.
void silu_mul_two_mul_fp4_to_fp4(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    const __half*  inv_s,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream);

// Explicit LUT implementation of the AWQ combiner. The LUT contains the
// exact device-computed GELU value for each (UE4M3 scale byte, FP4 code).
void silu_mul_two_mul_fp4_to_fp4_lut(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    const __half*  inv_s,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream);

// Explicit SM110 native E2M1 conversion experiment. Gate evaluation is the
// same LUT path; input decode and output encode use CUDA FP4 instructions.
void silu_mul_two_mul_fp4_to_fp4_lut_native(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    const __half*  inv_s,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
