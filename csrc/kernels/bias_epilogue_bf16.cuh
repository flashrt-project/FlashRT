// Shared BF16 bias consumers. Implemented by bias_epilogue_bf16.cu.
// Targets exposing these functions must own the implementing compile unit.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
void qkv_split_bias_bf16(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    int rows, int hq, int hk, int hv, cudaStream_t stream);
}  // namespace kernels
}  // namespace flash_rt
