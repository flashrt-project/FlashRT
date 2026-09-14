// Stochastic denoising step: x += a + sigma * eps (bf16 state, fp32 math).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {

// x[i] = bf16(float(x[i]) + fma(*sigma, float(eps[i]), float(a[i]))) for
// i < n. With *sigma == 0 the result is bit-identical to residual_add
// (the fma with a zero multiplier returns float(a[i]) exactly). ``sigma``
// is a device pointer so a captured graph can change it per call.
int pi05_sde_residual_add(__nv_bfloat16* x, const __nv_bfloat16* a, const __nv_bfloat16* eps,
                     const float* sigma, int n, cudaStream_t stream);

}  // namespace flash_rt
