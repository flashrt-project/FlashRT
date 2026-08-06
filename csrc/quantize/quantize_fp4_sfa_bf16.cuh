// ============================================================================
//  FlashRT — bf16-input fused NVFP4 quantize + CUTLASS SFA/SFB scale write.
//
//  bf16 companion of quantize_fp4_dynamic_sfa_fp16 for pipelines whose
//  activations are bf16 (GR00T N1.7 DiT). Additive: new symbols only.
// ============================================================================
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// Quantize a bf16 [N, D] row-major tensor to packed e2m1 [N, D/2] plus
// UE4M3 per-16-element scales written directly at the CUTLASS
// tile-interleaved SFA/SFB offsets. Vectorized (16-byte loads, 8-byte
// packed stores). Returns 0 on success, -1 on unsupported shape or
// misaligned buffers, -2 when built without CUTLASS.
int quantize_fp4_dynamic_sfa_bf16_vec(
    const void* src_bf16, void* dst_packed, void* dst_sfa,
    int N, int D, bool is_sfb, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
