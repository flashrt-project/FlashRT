// SPDX-License-Identifier: Apache-2.0
//
// Gated GeGLU activation, in bf16 and straight into a static FP8 GEMM input.
// Both round the activation to bf16 before the product and the product again;
// see the .cu for why activation.cu's gate_geglu is not a substitute.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_rt {
namespace kernels {

// out[i] = bf16( bf16(gelu_tanh(gate[i])) * up[i] )
void gelu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, long long n, cudaStream_t stream);

// out[i] = fp8_e4m3( clamp( bf16( bf16(gelu_tanh(gate[i])) * up[i] ) / act_scale, +-448 ) )
// `n` is the element count; it must be a multiple of 8 (16-byte accesses).
void gelu_mul_quant_fp8_static_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                    const float* act_scale, __nv_fp8_e4m3* out,
                                    long long n, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
