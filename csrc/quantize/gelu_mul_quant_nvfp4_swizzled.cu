// SPDX-License-Identifier: Apache-2.0
// Gated GeGLU activation straight into a clipped static NVFP4 GEMM input.
// See the header for the contract.

#include "gelu_mul_quant_nvfp4_swizzled.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdexcept>

namespace flash_rt {
namespace kernels {
namespace {

// The conversions below match quantize.cu's; the fused form has to produce the
// same scale factors and packed bytes as the unfused chain it replaces.
__device__ __forceinline__ float gelu_tanh_f32(float x) {
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float v) {
    const float a = fabsf(v);
    const uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    return sign | (uint8_t)((a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) +
                            (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f));
}

__device__ __forceinline__ uint8_t float_to_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    if (v > 240.0f) return 0xFE;
    const uint32_t bits = __float_as_uint(v);
    const int float_exp = ((bits >> 23) & 0xFF) - 127;
    const uint32_t frac = bits & 0x7FFFFF;
    int ue_exp = float_exp + 7;
    if (ue_exp <= 0) {
        int m = (int)ceilf(v * 512.0f);
        if (m > 7) return (1 << 3) | 0;
        if (m < 1) m = 1;
        return (uint8_t)m;
    }
    if (ue_exp >= 15) return 0xFE;
    int m = (int)(frac >> 20);
    if (frac & 0xFFFFF) m++;
    if (m >= 8) { m = 0; ue_exp++; }
    if (ue_exp >= 15) return 0xFE;
    return (uint8_t)((ue_exp << 3) | m);
}

__device__ __forceinline__ float ue4m3_to_float(uint8_t v) {
    const int e = (v >> 3) & 0xF, m = v & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

union GeluMulV8 { uint4 u; __nv_bfloat16 h[8]; };

__global__ void gelu_mul_quant_nvfp4_kernel(const __nv_bfloat16* __restrict__ gate,
                                            const __nv_bfloat16* __restrict__ up,
                                            const float* __restrict__ cap,
                                            uint8_t* __restrict__ packed,
                                            uint8_t* __restrict__ sf_swz,
                                            int rows, int cols) {
    const int nb = cols / 16, ncb = (nb + 3) / 4;
    const long long groups = (long long)rows * nb;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x; g < groups;
         g += (long long)gridDim.x * blockDim.x) {
        const int row = (int)(g / nb), b = (int)(g % nb);
        const uint4* gp = reinterpret_cast<const uint4*>(gate) + g * 2;
        const uint4* upp = reinterpret_cast<const uint4*>(up) + g * 2;
        float h[16], a = 0.f;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            GeluMulV8 gv, uv;
            gv.u = gp[k];
            uv.u = upp[k];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                // the activation rounds to bf16 before the product, and the
                // product rounds again -- HF's roundings, not an approximation
                const float act = __bfloat162float(
                    __float2bfloat16(gelu_tanh_f32(__bfloat162float(gv.h[j]))));
                const float v = __bfloat162float(
                    __float2bfloat16(act * __bfloat162float(uv.h[j])));
                h[k * 8 + j] = v;
                a = fmaxf(a, fabsf(v));
            }
        }
        const float clip = cap ? cap[b] : 0.0f;
        if (clip > 0.0f && isfinite(clip)) a = fminf(a, clip);
        const uint8_t ue = float_to_ue4m3_ceil(a / 6.0f);
        const int rb = row / 128, ri = row % 128;
        sf_swz[(rb * ncb + b / 4) * 512 + (ri % 32) * 16 + (ri / 32) * 4 + b % 4] = ue;
        const float scale = ue4m3_to_float(ue);
        const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t l = float_to_fp4_e2m1(h[2 * j] * inv);
            const uint8_t u = float_to_fp4_e2m1(h[2 * j + 1] * inv);
            lo |= (uint32_t)(((u << 4) | (l & 0x0F)) & 0xFF) << (8 * j);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t l = float_to_fp4_e2m1(h[8 + 2 * j] * inv);
            const uint8_t u = float_to_fp4_e2m1(h[8 + 2 * j + 1] * inv);
            hi |= (uint32_t)(((u << 4) | (l & 0x0F)) & 0xFF) << (8 * j);
        }
        uint2 o;
        o.x = lo;
        o.y = hi;
        reinterpret_cast<uint2*>(packed)[g] = o;
    }
}

}  // namespace

long long gelu_mul_quant_nvfp4_sf_bytes(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return 0;
    return (long long)((rows + 127) / 128) * (((cols / 16) + 3) / 4) * 512;
}

void gelu_mul_quant_nvfp4_swizzled(const void* gate, const void* up, const float* cap,
                                   uint8_t* packed, uint8_t* sf_swz,
                                   int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    if (!gate || !up || !packed || !sf_swz)
        throw std::runtime_error("gelu_mul_quant_nvfp4_swizzled: null pointer argument");
    if (cols % 16 != 0)
        throw std::runtime_error("gelu_mul_quant_nvfp4_swizzled: cols must be a multiple of 16");
    const long long groups = (long long)rows * (cols / 16);
    int grid = (int)((groups + 255) / 256);
    if (grid > 65535) grid = 65535;
    gelu_mul_quant_nvfp4_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up), cap, packed, sf_swz, rows, cols);
}

}  // namespace kernels
}  // namespace flash_rt
