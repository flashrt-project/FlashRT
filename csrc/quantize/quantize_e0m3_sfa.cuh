// ============================================================================
//  FlashRT — E0M3 (uniform INT4) quantize + CUTLASS SFA/SFB tile-interleave.
//
//  E0M3 is the sign-magnitude uniform 4-bit codebook (-7 .. +7) decoded by
//  the SM110 tcgen05 block-scaled MMA when the instruction-descriptor format
//  field is 0 (see cutlass_fp4_gemm_e0m3w_sm100.cuh). Packed layout and the
//  per-16 UE4M3 scale-factor layout are identical to the NVFP4 (E2M1)
//  quantizers, so E0M3 tensors are drop-in operands for the runtime-datatype
//  GEMM. The uniform grid removes E2M1's non-uniform rounding bins, which
//  lowers weight quantization error.
//
//  Additive: does NOT modify quantize_fp4_sfa.* or any existing kernel.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// fp16 [N, D] → packed [N, D/2] (e0m3) + SFA/SFB tile-interleaved UE4M3
// scales. Per-16 block scale = amax / 7 (E0M3 max magnitude), rounded to
// UE4M3; elements round to the nearest integer step and clamp to ±7.
//   is_sfb = false → SFA layout (A operand)
//   is_sfb = true  → SFB layout (B operand / weights)
// Returns 0 on success.
int quantize_e0m3_dynamic_sfa_fp16(
    const void* src_fp16,
    void* dst_packed,
    void* dst_sfa,
    int N, int D, bool is_sfb,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
