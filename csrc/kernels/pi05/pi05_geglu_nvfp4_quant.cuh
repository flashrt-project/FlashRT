// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {

// out = nvfp4(gelu_tanh(gate) * up) over a merged [rows, 2*half] BF16
// buffer (gate columns first). fp4_out is [rows, half/2] packed e2m1, sf_out
// the swizzled per-16 UE4M3 scales sized like quantize_bf16_to_nvfp4_swizzled
// for (rows, half); the global scale is 1. The product is rounded to BF16
// before quantization, as the unfused GeGLU + quantize pair does.
int pi05_geglu_merged_to_nvfp4_swizzled(const __nv_bfloat16* merged, uint8_t* fp4_out,
                                   uint8_t* sf_out, int rows, int half,
                                   cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
