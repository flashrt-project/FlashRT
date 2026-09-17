// Stochastic denoising step for the flow-matching sampler.
#include "pi05_sde_step.cuh"

namespace flash_rt {

__global__ void sde_residual_add_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a,
                                        const __nv_bfloat16* __restrict__ eps, const float* __restrict__ sigma,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = *sigma;
    const float inc = fmaf(s, __bfloat162float(eps[i]), __bfloat162float(a[i]));
    x[i] = __float2bfloat16(__bfloat162float(x[i]) + inc);
}

int pi05_sde_residual_add(__nv_bfloat16* x, const __nv_bfloat16* a, const __nv_bfloat16* eps,
                     const float* sigma, int n, cudaStream_t stream) {
    if (n <= 0) return 0;
    sde_residual_add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, a, eps, sigma, n);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace flash_rt
