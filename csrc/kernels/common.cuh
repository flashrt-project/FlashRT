// ================================================================
// FlashRT — Common CUDA kernel utilities
// Shared helpers used across all kernel files.
//
// All elementwise kernels are dtype-generic via C++ templates.
// Supported types: __half (FP16), __nv_bfloat16 (BF16).
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// ── Explicit-instantiation guard for MSVC ──
// MSVC 19.50 (Visual Studio 2022/2026) refuses explicit __global__
// template instantiations whose signatures contain const-qualified
// pointers + typedef aliases like __half (matched by GCC/Clang).
// Linux keeps the explicit instantiations (cross-TU safe, earlier
// link errors); MSVC relies on implicit instantiation triggered by
// the host wrapper kernel-launches living in the same .cu file.
#ifdef _MSC_VER
  #define FVK_KERNEL_INSTANTIATE(decl) /* implicit on MSVC */
#else
  #define FVK_KERNEL_INSTANTIATE(decl) template decl;
#endif

// ── Generic dtype conversion (template) ──
// Every kernel uses these instead of hardcoded bf16_to_f32 / f32_to_bf16.

template<typename T> __device__ __forceinline__ float to_f32(T x);
template<> __device__ __forceinline__ float to_f32<__half>(__half x) { return __half2float(x); }
template<> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename T> __device__ __forceinline__ T from_f32(float x);
template<> __device__ __forceinline__ __half from_f32<__half>(float x) { return __float2half(x); }
template<> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Packed 2-element type: half2 for FP16, __nv_bfloat162 for BF16
template<typename T> struct packed2;
template<> struct packed2<__half> { using type = __half2; };
template<> struct packed2<__nv_bfloat16> { using type = __nv_bfloat162; };

template<typename T> __device__ __forceinline__ typename packed2<T>::type make_packed2(T a, T b);
template<> __device__ __forceinline__ __half2 make_packed2<__half>(__half a, __half b) { return __halves2half2(a, b); }
template<> __device__ __forceinline__ __nv_bfloat162 make_packed2<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __halves2bfloat162(a, b); }

// ── Legacy aliases (for gradual migration) ──

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ── Warp-level reductions (no shared memory needed) ──

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;
}

// ── Block-level reduction (shared memory for inter-warp) ──

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
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
    return shared[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
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
    return shared[0];
}
