// ================================================================
// FlashRT AMD — Common HIP kernel utilities (CDNA, wave64)
// Shared helpers used across all AMD kernel files.
//
// Mirrors csrc/kernels/common.cuh with the CDNA differences:
//   - wavefront size is 64 (not 32); reductions use __shfl_down
//     (HIP has no *_sync variants — waves execute in lockstep)
//   - dtypes come from <hip/hip_fp16.h> / <hip/hip_bf16.h>
//     (__hip_bfloat16 provides the CUDA-compatible conversion API)
// ================================================================
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define FVK_WAVE_SIZE 64

// ── Generic dtype conversion (template) ──

template<typename T> __device__ __forceinline__ float to_f32(T x);
template<> __device__ __forceinline__ float to_f32<__half>(__half x) { return __half2float(x); }
template<> __device__ __forceinline__ float to_f32<__hip_bfloat16>(__hip_bfloat16 x) { return __bfloat162float(x); }

template<typename T> __device__ __forceinline__ T from_f32(float x);
template<> __device__ __forceinline__ __half from_f32<__half>(float x) { return __float2half(x); }
template<> __device__ __forceinline__ __hip_bfloat16 from_f32<__hip_bfloat16>(float x) { return __float2bfloat16(x); }

// Packed 2-element type: __half2 for FP16, __hip_bfloat162 for BF16
template<typename T> struct packed2;
template<> struct packed2<__half> { using type = __half2; };
template<> struct packed2<__hip_bfloat16> { using type = __hip_bfloat162; };

template<typename T> __device__ __forceinline__ typename packed2<T>::type make_packed2(T a, T b);
template<> __device__ __forceinline__ __half2 make_packed2<__half>(__half a, __half b) { return __halves2half2(a, b); }
template<> __device__ __forceinline__ __hip_bfloat162 make_packed2<__hip_bfloat16>(__hip_bfloat16 a, __hip_bfloat16 b) { return __halves2bfloat162(a, b); }

// ── Wave-level reductions (wave64, no shared memory) ──

__device__ __forceinline__ float wave_reduce_sum(float val) {
    #pragma unroll
    for (int off = FVK_WAVE_SIZE / 2; off > 0; off >>= 1)
        val += __shfl_down(val, off);
    return val;
}

__device__ __forceinline__ float wave_reduce_max(float val) {
    #pragma unroll
    for (int off = FVK_WAVE_SIZE / 2; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down(val, off));
    return val;
}

// ── Block-level reduction (LDS for inter-wave) ──
// Result is broadcast: every thread in the block receives the full sum.
// `shared` must hold at least ceil(blockDim.x / 64) floats.

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x & (FVK_WAVE_SIZE - 1);
    const int wave = threadIdx.x >> 6;
    const int nwaves = (blockDim.x + FVK_WAVE_SIZE - 1) >> 6;

    val = wave_reduce_sum(val);
    if (lane == 0) shared[wave] = val;
    __syncthreads();

    if (wave == 0) {
        float v = (lane < nwaves) ? shared[lane] : 0.0f;
        v = wave_reduce_sum(v);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}
