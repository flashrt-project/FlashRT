#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Fused per-head RMSNorm + rotate-half RoPE (bf16, in-place). See .cu.
// Launched once for Q (NHQ heads) and once for K (NHKV heads) for GQA.
// Returns 0 on success, -1 unsupported HD, -cudaError otherwise.
int qk_norm_rope_rotate_half_bf16(
    void* x, const void* w, const void* cos_t, const void* sin_t,
    int S, int NH, int HD, float eps, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
