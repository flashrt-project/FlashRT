// SPDX-License-Identifier: Apache-2.0
#pragma once
// Fused consumers for the skinny NVFP4 GEMM family on sm_120a (LayerNorm /
// AdaLN transformer blocks). Each one sums the GEMM's K-split partials,
// applies alpha = (*a_gs) * (*w_gs) and the bias, and finishes the block step
// in the same launch; whenever its output feeds another NVFP4 GEMM it emits
// e2m1 + UE4M3 block scales (global scale *out_gs) directly.
//
// Rounding follows the unfused BF16 chain: the GEMM result and the bias add
// are each rounded to BF16, the residual stream stays BF16. Row kernels run
// one CTA per row with dim / 4 threads (dim % 128 == 0, dim <= 4096); every
// thread owns 4 consecutive columns, so an NVFP4 block is one lane quad.
// Scale / shift (AdaLN modulation, out = norm(x) * (1 + scale) + shift) are
// optional BF16 vectors of dim; out_bf16, when given, replaces the NVFP4
// output (the block's last norm feeding a BF16 projection).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace nvfp4_skinny {

// LayerNorm (no affine) of x with optional AdaLN modulation.
int layer_norm_fp4(const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
                   uint8_t* out, uint8_t* out_sf, const float* out_gs, __nv_bfloat16* out_bf16, int rows,
                   int dim, float eps, bool pdl, cudaStream_t stream);

// residual += bf16(bf16(alpha * sum partials) + bias), then the LayerNorm
// (no affine) of the updated row with optional modulation.
int sum_bias_residual_norm(const float* partials, int splits, const float* a_gs, const float* w_gs,
                           const __nv_bfloat16* bias, __nv_bfloat16* residual, const __nv_bfloat16* scale,
                           const __nv_bfloat16* shift, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                           __nv_bfloat16* out_bf16, int rows, int dim, float eps, bool pdl, cudaStream_t stream);

// out = nvfp4(gelu_tanh(bf16(bf16(alpha * sum partials) + bias))).
int sum_bias_gelu_fp4(const float* partials, int splits, const float* a_gs, const float* w_gs,
                      const __nv_bfloat16* bias, uint8_t* out, uint8_t* out_sf, const float* out_gs, int rows,
                      int n, bool pdl, cudaStream_t stream);

}  // namespace nvfp4_skinny
}  // namespace flash_rt
