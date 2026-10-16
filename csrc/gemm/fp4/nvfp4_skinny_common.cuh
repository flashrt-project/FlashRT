// SPDX-License-Identifier: Apache-2.0
#pragma once
// Device helpers shared by the skinny NVFP4 (W4A4) GEMM family and the fused
// consumers that produce its NVFP4 operands on sm_120a. Header-only; every
// helper is __forceinline__ so a translation unit that does not call one gets
// no code for it.
//
// NVFP4 numerics used throughout: for a 16-element block of x (already divided
// by the per-tensor global scale gs), sf = ue4m3_rn(amax / 6) for activations
// (act_scale_code) and q = e2m1_rn_satfinite(x / sf). Weights are quantized
// once at setup with sf = ue4m3_ceil(amax / 6) or a squared-error search below
// it (nvfp4_skinny_gemm_sm120.cu).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flash_rt {
namespace nvfp4_skinny {

// D += A(e2m1, 16x64) * B(e2m1, 64x8), per-16 UE4M3 scales, FP32 accumulate.
__device__ __forceinline__ void mma_nvf4(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z),
          "h"(z));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ldg_stream8(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

// Programmatic dependent launch: let the next kernel start its prologue, then
// wait for the producer's writes.
__device__ __forceinline__ void pdl_prologue() {
    asm volatile("griddepcontrol.launch_dependents;\n" ::);
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

__device__ __forceinline__ float tanh_gelu(float g) {
    return g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Smallest UE4M3 value >= v (v >= 0), saturating at 448.
__device__ __forceinline__ uint8_t ue4m3_ceil(float v) {
    if (!(v > 0.f)) return 0;
    if (v >= 448.f) return 0x7E;
    const __nv_fp8_e4m3 q(v);
    uint8_t b = q.__x;
    if (static_cast<float>(q) < v) ++b;
    return b;
}

// Activation block scale: the UE4M3 value nearest to amax / 6 (saturating at
// 448). A block maximum that rounds above 6 after scaling saturates to 6; on
// real decoder and DiT activations this gives a lower error than rounding the
// scale up.
__device__ __forceinline__ uint8_t act_scale_code(float v) {
    if (!(v > 0.f)) return 0;
    if (v >= 448.f) return 0x7E;
    const __nv_fp8_e4m3 q(v);
    return q.__x;
}

__device__ __forceinline__ float ue4m3_value(uint8_t b) {
    __nv_fp8_e4m3 q;
    q.__x = b;
    return static_cast<float>(q);
}

// Four floats -> four e2m1 (round to nearest even, saturating), element 0 in the low nibble.
__device__ __forceinline__ uint16_t e2m1x4(float v0, float v1, float v2, float v3) {
    uint16_t r;
    asm("{\n"
        ".reg .b8 b0, b1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "mov.b16 %0, {b0, b1};\n"
        "}"
        : "=h"(r)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3));
    return r;
}

// Quantize one 16-element block v[] (already divided by the global scale).
__device__ __forceinline__ void quant_block16(const float* v, uint8_t* out8, uint8_t* sf) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const uint8_t s = act_scale_code(amax * (1.0f / 6.0f));
    const float d = ue4m3_value(s);
    const float inv = d > 0.f ? 1.0f / d : 0.f;
    uint16_t h[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        h[q] = e2m1x4(v[4 * q] * inv, v[4 * q + 1] * inv, v[4 * q + 2] * inv, v[4 * q + 3] * inv);
    *reinterpret_cast<uint2*>(out8) = make_uint2(h[0] | (static_cast<uint32_t>(h[1]) << 16),
                                                 h[2] | (static_cast<uint32_t>(h[3]) << 16));
    *sf = s;
}

// Quad-cooperative quantization: the four lanes of an aligned lane quad own
// four consecutive values each of one 16-element block. Every lane writes its
// two bytes; the lane passed write_sf writes the block scale.
__device__ __forceinline__ void quant_quad4(float v0, float v1, float v2, float v3, uint8_t* out2,
                                            uint8_t* sf, bool write_sf) {
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const uint8_t s = act_scale_code(amax * (1.0f / 6.0f));
    const float d = ue4m3_value(s);
    const float inv = d > 0.f ? 1.0f / d : 0.f;
    *reinterpret_cast<uint16_t*>(out2) = e2m1x4(v0 * inv, v1 * inv, v2 * inv, v3 * inv);
    if (write_sf) *sf = s;
}

// Sum of the K-split partials for one element pair, scaled by alpha and
// rounded to BF16 (the value an unfused GEMM would have written).
__device__ __forceinline__ float2 sum_pair(const float* __restrict__ partials, int splits, size_t elements,
                                           size_t pair, float alpha) {
    float2 s0 = make_float2(0.f, 0.f), s1 = make_float2(0.f, 0.f);
    int s = 0;
    for (; s + 1 < splits; s += 2) {
        const float2 p0 = *reinterpret_cast<const float2*>(partials + s * elements + 2 * pair);
        const float2 p1 = *reinterpret_cast<const float2*>(partials + (s + 1) * elements + 2 * pair);
        s0.x += p0.x; s0.y += p0.y; s1.x += p1.x; s1.y += p1.y;
    }
    if (s < splits) {
        const float2 p0 = *reinterpret_cast<const float2*>(partials + s * elements + 2 * pair);
        s0.x += p0.x; s0.y += p0.y;
    }
    return __bfloat1622float2(__floats2bfloat162_rn((s0.x + s1.x) * alpha, (s0.y + s1.y) * alpha));
}

__device__ __forceinline__ float2 bf2(uint32_t bits) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
}

__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Host-side launch with the programmatic-stream-serialization attribute.
template <typename... Args>
inline int launch_ex(const void* fn, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    void* pargs[] = {(void*)&args...};
    return static_cast<int>(cudaLaunchKernelExC(&cfg, fn, pargs));
}

}  // namespace nvfp4_skinny
}  // namespace flash_rt
