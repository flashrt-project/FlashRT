// SPDX-License-Identifier: Apache-2.0
//
// See header. The kernel is compile-time specialized on the row count and
// optionally joins the caller's programmatic-dependent-launch (PDL) chain:
// hosts that overlap every launch (llama.cpp CUDA backend) lose measurable
// time to any kernel that breaks the chain.

#include "f32_act_to_nvfp4_swizzled_mrows_sm120.cuh"

namespace flash_rt {
namespace quantize {

namespace {

__device__ __forceinline__ void pdl_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}
__device__ __forceinline__ void pdl_lc() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int THREADS, int MT>
__global__ void f32_act_to_nvfp4_kernel(
        const float * __restrict__ x,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sfa,
        int D, long long x_srow) {
    pdl_lc(); pdl_sync();
#pragma unroll
    for (int r = blockIdx.x; r < MT; r += gridDim.x)   // launch with grid = MT
        f32_act_to_nvfp4_row<THREADS>(x + (size_t) r * x_srow, dst_packed, dst_sfa, D, r);
}

template <int MT>
int launch(const float * x, void * dst_packed, void * dst_sfa,
           int D, long long x_srow, bool pdl, cudaStream_t stream) {
    const dim3 grid(MT), block(256);
    if (pdl) {
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0;
        cfg.stream = stream; cfg.attrs = &attr; cfg.numAttrs = 1;
        return (int) cudaLaunchKernelEx(&cfg, f32_act_to_nvfp4_kernel<256, MT>,
            x, (uint2 *) dst_packed, (uint8_t *) dst_sfa, D, x_srow);
    }
    f32_act_to_nvfp4_kernel<256, MT><<<grid, block, 0, stream>>>(
        x, (uint2 *) dst_packed, (uint8_t *) dst_sfa, D, x_srow);
    return (int) cudaGetLastError();
}

}  // namespace

int f32_act_to_nvfp4_swizzled_mrows(
    const float * x, void * dst_packed, void * dst_sfa,
    int D, int M, long long x_srow, bool pdl, cudaStream_t stream) {
    if (D % 16 != 0 || M < 1 || M > 4) return -1;
    switch (M) {
        case 1: return launch<1>(x, dst_packed, dst_sfa, D, x_srow, pdl, stream);
        case 2: return launch<2>(x, dst_packed, dst_sfa, D, x_srow, pdl, stream);
        case 3: return launch<3>(x, dst_packed, dst_sfa, D, x_srow, pdl, stream);
        default: return launch<4>(x, dst_packed, dst_sfa, D, x_srow, pdl, stream);
    }
}

}  // namespace quantize
}  // namespace flash_rt
