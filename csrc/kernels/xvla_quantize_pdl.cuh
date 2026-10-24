// FlashRT — opt-in NVFP4 activation quantize v3 (X-VLA).
//
// The existing quantize_bf16_to_nvfp4_swizzled_v2 entry is unchanged.  This
// distinctly named sibling uses the thread-per-16-element-block kernel and
// falls back to the unchanged v2 entry for shapes that need the row path.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

void quantize_bf16_to_nvfp4_swizzled_v3(
    const __nv_bfloat16* input, uint8_t* fp4_data, uint8_t* scale_factors,
    int rows, int cols, cudaStream_t stream);
