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
    // Threshold count, predicated rather than an 8-way if-else chain (which
    // diverges across a warp and serializes the hot per-element encode).
    // Same codes for every input: thresholds are the round-half-up
    // midpoints, and NaN (no comparison true) still maps to magnitude 7.
    const float a = fabsf(v);
    const uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    const uint8_t mag = (uint8_t)((a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f)
                                + (a >= 1.75f) + (a >= 2.5f)  + (a >= 3.5f)
                                + (a >= 5.0f) + 7 * (a != a));
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

// Two values -> one packed byte (lo in the low nibble) with the codes of
// float_to_fp4_e2m1_branchless: one hardware round-to-nearest-even
// conversion, plus the four midpoints (0.25, 1.25, 2.5, 5) where RNE picks the
// even code and the threshold encoders round half up, plus their (v < 0) sign
// rule. Equal to two branchless calls on every non-NaN fp32 input (checked
// exhaustively under --ftz=true); sm_100a / sm_120a only.
__device__ __forceinline__ uint8_t float2_to_fp4_e2m1x2_exact(float lo, float hi) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM100_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM121_ALL)
    uint16_t r;
    asm("{\n.reg .b8 b;\ncvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\ncvt.u16.u8 %0, b;\n}"
        : "=h"(r) : "f"(lo), "f"(hi));
    const float al = fabsf(lo), ah = fabsf(hi);
    uint32_t c = r;
    c += ((al == 0.25f) | (al == 1.25f) | (al == 2.5f) | (al == 5.0f))
         + (((ah == 0.25f) | (ah == 1.25f) | (ah == 2.5f) | (ah == 5.0f)) << 4);
    c = (c & 0x77u) | (lo < 0.0f ? 0x8u : 0u) | (hi < 0.0f ? 0x80u : 0u);
    return static_cast<uint8_t>(c);
#else
    return static_cast<uint8_t>((float_to_fp4_e2m1_branchless(hi) << 4)
                                | (float_to_fp4_e2m1_branchless(lo) & 0x0F));
#endif
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
    // exact: normals (1 + m/8) * 2^(e-7) assembled as fp32 bits, subnormals
    // m * 2^-9 (what the ldexpf forms compute, without the libm calls)
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    if (e == 0) return (float)m * 0.001953125f;
    return __uint_as_float((uint32_t)(e + 120) << 23 | (uint32_t)m << 20);
}
