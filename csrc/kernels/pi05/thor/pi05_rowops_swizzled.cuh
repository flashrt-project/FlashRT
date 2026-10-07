// ============================================================================
//  FlashRT — v5 warp-per-row activation kernels for the Pi0.5 Thor encoder,
//  SigLIP and decoder paths (additive; v2 stays the default).
//
//  Per-column fp32 tables (gamma / beta / inverse scales) use the lane-major
//  order produced by the frontend's _swz32 (see pi05_rowops_v5.cu).
//
//  Same row decomposition and reduction orders as pi05_rowops_v2, with the
//  per-element instruction stream cut down (see pi05_rowops_v5.cu). Per-column
//  tables (gamma / beta / AWQ inverse scales) are fp32 here.
//
//  Requirements: D % 16 == 0, D/16 <= 128, 16-byte aligned rows and tables.
//  Nonzero return = not launched.
// ============================================================================
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

// out_fp8 = e4m3(rms_norm(x) / descale)
int pi05_row_rms_fp8_swizzled(const __half* x, void* out_fp8, int S, int D, const float* descale,
                      cudaStream_t stream);

// out = NVFP4(src) + SFA
int pi05_row_quantize_fp4_sfa_swizzled(const __half* src, void* packed, void* sfa, int N, int D,
                               cudaStream_t stream);

// out = rms_norm(x) [* inv_s (fp32)] -> NVFP4 + SFA
int pi05_row_rms_mul_fp4_sfa_swizzled(const __half* x, const float* inv_s, void* packed, void* sfa,
                              int S, int D, cudaStream_t stream);

// out_fp8 = e4m3(LayerNorm(x; gamma, beta))  (fp32 tables)
int pi05_row_layer_norm_fp8_swizzled(const __half* x, const float* gamma, const float* beta,
                             void* out_fp8, int S, int D, float eps, cudaStream_t stream);

// out = NVFP4(LayerNorm(x; gamma, beta) [* inv_s]) + SFA  (fp32 tables)
int pi05_row_layer_norm_mul_fp4_sfa_swizzled(const __half* x, const float* gamma, const float* beta,
                                     const float* inv_s, void* packed, void* sfa, int S, int D,
                                     float eps, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
