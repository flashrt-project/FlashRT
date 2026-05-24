// SiLU-gate elementwise multiply: out = silu(gate) * up.
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
//
// Qwen3.6 SwiGLU MLP needs this exactly; the existing `gate_silu_mul`
// in csrc/kernels/activation.cu is misnamed -- it actually computes
// GELU(gate)*up via the tanh approximation. This kernel is the proper
// SiLU variant, used by Qwen36's _layer_forward_{lin,full} MLP.
//
// Shape: gate, up, out are bf16 device pointers, n total elements
// (flat). For Qwen3.6 MLP intermediate: n = 1 * 17408 per decode step.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt::kernels {

void silu_mul_qwen36_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream);

void sigmoid_mul_qwen36_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* x,
    __nv_bfloat16* out,
    int n,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
