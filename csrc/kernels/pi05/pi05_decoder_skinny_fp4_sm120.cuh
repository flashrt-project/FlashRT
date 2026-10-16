// SPDX-License-Identifier: Apache-2.0
#pragma once
// Pi0.5 action-decoder consumers for the skinny NVFP4 (W4A4) family on sm_120a.
//
// The decoder GEMMs run on the shared skinny NVFP4 family
// (csrc/gemm/fp4/nvfp4_skinny_gemm_sm120.cuh). These consumers sum its K-split
// partials and finish the Pi0.5 action-expert block in the same launch,
// emitting the next GEMM's operand as NVFP4 (e2m1 + UE4M3 block scales,
// global scale *out_gs): the gated residual + adaptive RMS norm, the GeGLU,
// and the action input projection + first adaptive norm. alpha is
// (*a_gs) * (*w_gs) as in the FP8 family's consumers, whose RoPE, attention
// and final-norm consumers the NVFP4 tier reuses unchanged.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace pi05_dec_skinny_fp4 {

// Gated residual + adaptive RMS norm (as the FP8 family's residual_ada_norm)
// emitting the normed row as NVFP4 with global scale *out_gs. dim == 1024.
int residual_ada_norm(const float* partials, int splits, const float* a_gs, const float* w_gs,
                      __nv_bfloat16* residual, const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                      const __nv_bfloat16* style, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                      __nv_bfloat16* gate_out, int rows, int dim, float eps, bool pdl, cudaStream_t stream);

// GeGLU over merged [gate | up] partials straight to NVFP4 (global scale *out_gs).
int gate_gelu(const float* partials, int splits, const float* a_gs, const float* w_gs, uint8_t* out,
              uint8_t* out_sf, int rows, int half, const float* out_gs, bool pdl, cudaStream_t stream);

// Action input projection (K = 32, N = 1024) + bias into the residual stream,
// then the first layer's adaptive RMS norm straight to NVFP4.
int action_in_norm(const __nv_bfloat16* noise, const __nv_bfloat16* w_in, const __nv_bfloat16* b_in,
                   __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* style, uint8_t* out,
                   uint8_t* out_sf, __nv_bfloat16* gate_out, const float* out_gs, int rows, float eps, bool pdl,
                   cudaStream_t stream);

}  // namespace pi05_dec_skinny_fp4
}  // namespace flash_rt
