#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace pi05 {
// x: mutable BF16 [rows, cols]; bias, weight: BF16 [cols].
// out: INT8 [rows, cols]; scales: FP32 [rows]. All on the same device.
// Updates x[:image_rows] in place with BF16(x+bias) before normalization.
// Does not modify the language rows x[image_rows:].
void projector_bias_rms_norm_int8(
    __nv_bfloat16* x, const __nv_bfloat16* bias,
    const __nv_bfloat16* weight, int8_t* out, float* scales,
    int rows, int cols, int image_rows, float eps, cudaStream_t stream);
}  // namespace pi05
}  // namespace flash_rt
