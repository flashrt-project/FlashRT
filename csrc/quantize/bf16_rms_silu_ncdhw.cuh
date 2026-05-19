#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

int bf16_rms_silu_ncdhw(
    const void* x_bf16,
    const void* gamma_bf16,
    void* y_bf16,
    const void* prev_cache_bf16,
    void* next_cache_bf16,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream);

int bf16_rms_norm_ncdhw(
    const void* x_bf16,
    const void* gamma_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream);

int bf16_pack_t1_cache3_nchw_channels_last(
    const void* prev_cache_bf16,
    const void* cur_bf16,
    void* out_bf16,
    int C, int H, int W,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
