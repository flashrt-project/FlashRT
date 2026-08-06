#pragma once

// NVFP4 element and scale-factor conversions.
//
// Moved out of quantize.cu unchanged so a translation unit that produces the
// same wire format without pulling in the whole quantiser -- the gated
// qwen3_5_moe grouped quantiser is the first -- encodes it with the same code
// rather than a second copy of these thresholds. quantize.cu includes this
// header where the definitions used to be, so its own kernels are unaffected.
//
// Everything here is a device-side __forceinline__ helper: including this
// header adds no symbol and no code to a TU that does not call it.
//
//  FP4 E2M1 values: +/-{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
//  UE4M3 (unsigned E4M3): one scale factor per 16-element block

#include <cuda_runtime.h>

#include <cstdint>

// FP4 E2M1 value table (magnitude only, 3 bits):
//   0b000 = 0.0   (E=0, M=0)
//   0b001 = 0.5   (E=0, M=1, subnormal)
//   0b010 = 1.0   (E=1, M=0)
//   0b011 = 1.5   (E=1, M=1)
//   0b100 = 2.0   (E=2, M=0)
//   0b101 = 3.0   (E=2, M=1)
//   0b110 = 4.0   (E=3, M=0)
//   0b111 = 6.0   (E=3, M=1)

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint8_t mag;
    if      (a < 0.25f)  mag = 0;  // -> 0.0
    else if (a < 0.75f)  mag = 1;  // -> 0.5
    else if (a < 1.25f)  mag = 2;  // -> 1.0
    else if (a < 1.75f)  mag = 3;  // -> 1.5
    else if (a < 2.5f)   mag = 4;  // -> 2.0
    else if (a < 3.5f)   mag = 5;  // -> 3.0
    else if (a < 5.0f)   mag = 6;  // -> 4.0
    else                 mag = 7;  // -> 6.0
    return sign | mag;
}

// Branchless equivalent of float_to_fp4_e2m1 — bit-identical, but the 8-way
// if-else (which diverges across a warp and serializes) becomes a sum of
// threshold comparisons (predicated, no divergence). Used by the prefetch _v2
// quant/norm kernels where the encode is the hot per-element op.
__device__ __forceinline__ uint8_t float_to_fp4_e2m1_branchless(float v) {
    float a = fabsf(v);
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    uint8_t mag = (uint8_t)((a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f)
                          + (a >= 1.75f) + (a >= 2.5f)  + (a >= 3.5f)
                          + (a >= 5.0f));
    return sign | mag;
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t v) {
    float mag;
    switch (v & 0x7u) {
        case 0: mag = 0.0f; break;
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        default: mag = 6.0f; break;
    }
    return (v & 0x8u) ? -mag : mag;
}

// Convert float to UE4M3 (unsigned, 4-bit exponent, 3-bit mantissa)
// Rounds UP (ceil) so that scale >= true_amax / 6.0 (avoids FP4 overflow)
// UE4M3: bias=7, normal = 2^(E-7) * (1 + M/8), subnormal = 2^(-6) * M/8
// Range: [~0.002, 240]
__device__ __forceinline__ uint8_t float_to_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    if (v > 240.0f) return 0xFE;  // max finite: E=14, M=7 -> 2^7 * 1.875 = 240

    uint32_t bits = __float_as_uint(v);
    int float_exp = ((bits >> 23) & 0xFF) - 127;  // unbiased float exponent
    uint32_t frac = bits & 0x7FFFFF;               // 23-bit float mantissa

    int ue_exp = float_exp + 7;  // UE4M3 bias = 7

    if (ue_exp <= 0) {
        // Subnormal in UE4M3: value = 2^(-6) * M/8
        float scaled = v * 512.0f;  // v / (2^(-6) / 8)
        int m = (int)ceilf(scaled);
        if (m > 7) return (1 << 3) | 0;  // smallest normal: E=1, M=0
        if (m < 1) m = 1;
        return (uint8_t)m;
    }
    if (ue_exp >= 15) return 0xFE;  // clamp to max

    // Extract top 3 mantissa bits, round up
    int m = (int)(frac >> 20);  // top 3 of 23 bits
    if (frac & 0xFFFFF) m++;    // ceil: round up if remaining bits nonzero
    if (m >= 8) { m = 0; ue_exp++; }
    if (ue_exp >= 15) return 0xFE;

    return (uint8_t)((ue_exp << 3) | m);
}

__device__ __forceinline__ float ue4m3_to_float(uint8_t v) {
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}
