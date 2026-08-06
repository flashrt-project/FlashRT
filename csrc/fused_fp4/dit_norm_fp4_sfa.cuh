// ============================================================================
//  FlashRT — fused DiT norms (bf16 in) + NVFP4 quantize + SFA write.
//
//  GR00T N1.7 DiT front-ends for NVFP4 GEMMs: the AdaLN-modulated norm1
//  (QKV / cross-Q input) and the pre-FFN no-affine LayerNorm emit packed
//  e2m1 + tile-interleaved UE4M3 scales directly, replacing a bf16 norm
//  kernel followed by a separate quantize kernel. The normalized value is
//  rounded through bf16 before quantization so the output is bit-identical
//  to the two-step reference chain.
//
//  Additive: new symbols only.
// ============================================================================
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// packed/sfa = quantize(LN_no_affine(x[row]) * (1 + scale) + shift)
// x: bf16 [S, D]; scale/shift: bf16 [D] (per-layer AdaLN modulators).
int ada_layer_norm_fp4_sfa_bf16(
    const void* x, const void* scale, const void* shift,
    void* packed, void* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream);

// packed/sfa = quantize(LN_no_affine(x[row]))
int layer_norm_no_affine_fp4_sfa_bf16(
    const void* x, void* packed, void* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
