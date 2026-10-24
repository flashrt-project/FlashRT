// FlashRT — safe block reductions for the opt-in PDL kernels.
//
// The existing common.cuh block_reduce_sum/max are unchanged.  These
// distinctly named variants add the trailing barrier required when two
// reductions use the same shared[0] back to back under programmatic
// dependent launch; they are used only by the X-VLA opt-in entries.
#pragma once

#include "common.cuh"

__device__ __forceinline__ float block_reduce_sum_safe(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    const float result = shared[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float block_reduce_max_safe(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_max(val);

    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    const float result = shared[0];
    __syncthreads();
    return result;
}
