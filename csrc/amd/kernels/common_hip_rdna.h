// ================================================================
// FlashRT AMD -- common HIP helpers for RDNA (wave32).
//
// Keep this separate from common_hip.h: that header is part of the
// CDNA/gfx950 implementation and intentionally hard-codes wave64.
// The helpers here define the numerical and synchronization conventions used
// by every `_rdna` kernel: FP16/BF16 values are promoted to FP32 for math,
// wave reductions use the native 32-lane shuffle width, and block reductions
// use only one barrier before broadcasting the per-wave partials.
// ================================================================
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

constexpr int FLASHRT_RDNA_WAVE_SIZE = 32;

// ── Low-precision conversion helpers ──

template <typename T> __device__ __forceinline__ float rdna_to_f32(T value);

template <> __device__ __forceinline__ float rdna_to_f32<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float
rdna_to_f32<__hip_bfloat16>(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T> __device__ __forceinline__ T rdna_from_f32(float value);

template <>
__device__ __forceinline__ __half rdna_from_f32<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __hip_bfloat16
rdna_from_f32<__hip_bfloat16>(float value) {
  return __float2bfloat16(value);
}

// ── Wave32 reductions ──
// The full-wave variants return the complete result in lane 0. The 8-lane
// variants reduce lanes [0,8), which is enough for the maximum eight wave
// partials produced by the 256-thread generic kernels.

__device__ __forceinline__ float rdna_wave_reduce_sum(float value) {
#pragma unroll
  for (int offset = FLASHRT_RDNA_WAVE_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset, FLASHRT_RDNA_WAVE_SIZE);
  }
  return value;
}

__device__ __forceinline__ float rdna_wave_reduce_max(float value) {
#pragma unroll
    for (int offset = FLASHRT_RDNA_WAVE_SIZE / 2; offset > 0; offset >>= 1) {
        value = fmaxf(
            value, __shfl_down(value, offset, FLASHRT_RDNA_WAVE_SIZE));
    }
    return value;
}

__device__ __forceinline__ float rdna_wave_reduce_sum_8(float value) {
  value += __shfl_down(value, 4, FLASHRT_RDNA_WAVE_SIZE);
  value += __shfl_down(value, 2, FLASHRT_RDNA_WAVE_SIZE);
  value += __shfl_down(value, 1, FLASHRT_RDNA_WAVE_SIZE);
  return value;
}

__device__ __forceinline__ float rdna_wave_reduce_max_8(float value) {
  value = fmaxf(value, __shfl_down(value, 4, FLASHRT_RDNA_WAVE_SIZE));
  value = fmaxf(value, __shfl_down(value, 2, FLASHRT_RDNA_WAVE_SIZE));
  value = fmaxf(value, __shfl_down(value, 1, FLASHRT_RDNA_WAVE_SIZE));
  return value;
}

// ── Block reductions ──
// Returns the block-wide result to every thread. Each wave repeats the tiny
// reduction over the per-wave partials after the first barrier. This avoids a
// second block barrier solely for broadcasting wave 0's result.
__device__ __forceinline__ float rdna_block_reduce_sum(float value,
                                                       float *wave_sums) {
  const int lane = threadIdx.x & (FLASHRT_RDNA_WAVE_SIZE - 1);
  const int wave = threadIdx.x / FLASHRT_RDNA_WAVE_SIZE;
  const int wave_count =
      (blockDim.x + FLASHRT_RDNA_WAVE_SIZE - 1) / FLASHRT_RDNA_WAVE_SIZE;

  value = rdna_wave_reduce_sum(value);
  if (wave_count == 1) {
    return __shfl(value, 0, FLASHRT_RDNA_WAVE_SIZE);
  }
  if (lane == 0) {
    wave_sums[wave] = value;
  }
  __syncthreads();

  float partial = lane < wave_count ? wave_sums[lane] : 0.0f;
  partial = rdna_wave_reduce_sum_8(partial);
  return __shfl(partial, 0, FLASHRT_RDNA_WAVE_SIZE);
}

__device__ __forceinline__ float rdna_block_reduce_max(
    float value, float* wave_maxima) {
    const int lane = threadIdx.x & (FLASHRT_RDNA_WAVE_SIZE - 1);
    const int wave = threadIdx.x / FLASHRT_RDNA_WAVE_SIZE;
    const int wave_count =
        (blockDim.x + FLASHRT_RDNA_WAVE_SIZE - 1) / FLASHRT_RDNA_WAVE_SIZE;

    value = rdna_wave_reduce_max(value);
    if (wave_count == 1) {
        return __shfl(value, 0, FLASHRT_RDNA_WAVE_SIZE);
    }
    if (lane == 0) {
        wave_maxima[wave] = value;
    }
    __syncthreads();

    float partial = lane < wave_count ? wave_maxima[lane] : -INFINITY;
    partial = rdna_wave_reduce_max_8(partial);
    return __shfl(partial, 0, FLASHRT_RDNA_WAVE_SIZE);
}
