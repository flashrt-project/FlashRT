// SPDX-License-Identifier: Apache-2.0
//
// Gated GeGLU activation straight into a clipped static NVFP4 GEMM input.
//
//   out = quant_nvfp4( bf16( bf16(gelu_tanh(gate)) * up ) )
//
// Two separate (rows, cols) bf16 inputs, per-16-element-group e2m1 values with a
// UE4M3 block scale in the CUTLASS sm120 swizzled layout, and a per-group clip:
// the group's amax is clamped to `cap` before the scale is formed, which is how
// a frozen calibration amax bounds a group that drifts above it at runtime.
//
// The two roundings in the product are HF's, not an approximation of them: the
// activation is rounded to bf16 and the product is rounded again, which is what
// running the gated multiply and the quantizer as two kernels produces. Skipping
// the first rounding changes roughly 2% of the output bytes.
//
// Unlike silu_mul_to_nvfp4_swizzled this is weight-agnostic about the activation
// function and takes the clip factor; the NVFP4 encoding itself is identical to
// quantize_bf16_to_nvfp4_swizzled_clipped.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

// gate, up : (rows, cols) bf16, row-major
// cap      : cols/16 fp32 per-group clip values (device); a null or non-positive
//            entry disables the clip for that group
// packed   : (rows, cols/2) uint8, e2m1 values
// sf_swz   : the swizzled UE4M3 scale factors, (rows rounded up to 128) x
//            ((cols/16 + 3) / 4) x 512 bytes
// cols must be a multiple of 16.
void gelu_mul_quant_nvfp4_swizzled(const void* gate, const void* up, const float* cap,
                                   uint8_t* packed, uint8_t* sf_swz,
                                   int rows, int cols, cudaStream_t stream);

// Bytes the caller must allocate for sf_swz.
long long gelu_mul_quant_nvfp4_sf_bytes(int rows, int cols);

}  // namespace kernels
}  // namespace flash_rt
