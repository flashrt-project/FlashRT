// FlashRT — opt-in FP8 per-token block quantize v2 (X-VLA).
//
// The existing fp8_per_token_block128_quant_bf16 entry is unchanged.  This
// sibling maps one warp to each (row, K-block) so four contiguous bf16 per
// thread replace the 128-thread shared-memory reduction.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

void fp8_per_token_block128_quant_bf16_v2(
    const void* input, void* output_fp8, float* output_scale, int M, int K,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
