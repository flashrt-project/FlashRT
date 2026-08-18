// ============================================================================
//  FlashRT — fused SiLU(gate) * up (bf16 in) + NVFP4 quantize + SFA write.
//
//  GR00T N1.6 Qwen3 FFN hand-off for the NVFP4 tier: replaces the torch
//  silu + mul + separate quantize chain (three elementwise passes over
//  [Se, Dff]) with one kernel emitting packed e2m1 + tile-interleaved UE4M3
//  scales. Value path mirrors torch: silu in fp32, rounded to bf16, then
//  bf16 multiply.
//
//  Additive: new symbols only.
// ============================================================================
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// packed/sfa = quantize(bf16(silu(gate)) * up)
// gate/up: bf16 [N, D] row-major. is_sfb selects SFB (weights) layout.
int silu_mul_fp4_sfa_bf16(
    const void* gate, const void* up, void* packed, void* sfa,
    int N, int D, bool is_sfb, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
