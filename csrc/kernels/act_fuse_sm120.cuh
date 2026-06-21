// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini fused activation-gate kernels (bf16, fp32-internal, EXACT math).
//
// The MoE / shared-expert SwiGLU and the full-attn output gate were each ~4
// torch ops (2 .float() casts + the activation + the mul + .to(bf16)). These
// fuse them into one launch with the same fp32 math (cos 1.0, no precision
// change). NOTE: the existing `gate_silu_mul` binding is misnamed -- it is the
// tanh-approx GELU, not SiLU -- so it cannot be reused for Nex-N2's SiLU.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// out = silu(g) * u   (silu(x) = x / (1 + exp(-x))), bf16 io, fp32 internal.
int silu_mul_sm120_bf16(const void* g, const void* u, void* out, int n,
                        cudaStream_t stream);

// out = x * sigmoid(gate), bf16 io, fp32 internal.
int sigmoid_mul_sm120_bf16(const void* x, const void* gate, void* out, int n,
                           cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
