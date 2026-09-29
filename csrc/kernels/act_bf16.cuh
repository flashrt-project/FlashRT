// FlashRT — shared elementwise activations (silu / erf-gelu), bf16 in-place.
//
// Model-agnostic kernels (no model-specific tensors, dims, or control flow),
// hence the generic file/binding names and the unconditional compile.
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

void silu_bf16(__nv_bfloat16* x, int n, cudaStream_t stream);

void gelu_erf_bf16(__nv_bfloat16* x, int n, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
