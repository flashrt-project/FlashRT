// SPDX-License-Identifier: Apache-2.0
//
// F32 activation rows -> NVFP4 packed + swizzled SFA (Sm1xx atom layout),
// M rows in one launch (M = 1..4, speculative-decode verify batches).
// Framework-free: raw pointers + cudaStream_t; the device body is exposed
// so host-adapter kernels can inline the quantization into fused producers.
// Additive: new file + new entry point (the single-row bf16 weight/act
// quantizers are separate, older entries).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {

// ---- device body (shared with fused producers) ----------------------------

__device__ __forceinline__ int nvfp4_sfa_offset_128x64(int row, int k, int dim) {
    const int row_block    = row >> 7;
    const int row_in_block = row & 127;
    const int k_block      = k >> 6;
    const int k_in_block   = k & 63;
    const int k_blocks     = (dim + 63) >> 6;
    return row_block * k_blocks * 512 + k_block * 512 +
        (row_in_block & 31) * 16 + (row_in_block >> 5) * 4 +
        (k_in_block >> 4);
}

// activation-quant boundary convention (<=; the weight packers use strict <)
__device__ __forceinline__ uint8_t nvfp4_act_f32_to_e2m1(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

// quantize one f32 row of length D into packed e2m1 + SFA; `row` selects the
// SFA atom-layout row and the packed output row (row-major, D/16 uint2).
// Callable from any single participating block of THREADS threads.
template <int THREADS>
__device__ __forceinline__ void f32_act_to_nvfp4_row(
        const float * __restrict__ x,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sfa,
        int D, int row = 0) {
    const int n_blocks = D / 16;
    uint2 * dst_row = dst_packed + (size_t) row * n_blocks;
    for (int b = threadIdx.x; b < n_blocks; b += THREADS) {
        float vals[16];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[i] = x[b * 16 + i];
            const float a = fabsf(vals[i]);
            if (a > amax) amax = a;
        }
        float desired = amax / 6.f;
        if (desired < 1e-12f) desired = 1e-12f;
        __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
        const float bs_dq = static_cast<float>(bs_q);
        dst_sfa[nvfp4_sfa_offset_128x64(row, b * 16, D)] = *reinterpret_cast<uint8_t *>(&bs_q);
        const float inv_bs = 1.f / bs_dq;
        uint2 out;
        uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint8_t lo = nvfp4_act_f32_to_e2m1(vals[2 * p]     * inv_bs);
            const uint8_t hi = nvfp4_act_f32_to_e2m1(vals[2 * p + 1] * inv_bs);
            ob[p] = static_cast<uint8_t>(lo | (hi << 4));
        }
        dst_row[b] = out;
    }
}

// ---- host entry ------------------------------------------------------------

// Quantize M f32 rows (row t at x + t*x_srow) into packed [M, D/2] + SFA in
// the atom layout for problem rows 0..M-1. M in 1..4 (compile-time
// specialized; a runtime M in the hot loop costs measurable time).
// pdl: join the caller's programmatic-dependent-launch chain.
// Returns 0 on success.
int f32_act_to_nvfp4_swizzled_mrows(
    const float * x, void * dst_packed, void * dst_sfa,
    int D, int M, long long x_srow, bool pdl, cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
