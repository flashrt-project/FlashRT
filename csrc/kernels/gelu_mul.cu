// SPDX-License-Identifier: Apache-2.0
//
// Gated GeGLU activation, in bf16 and straight into a static FP8 GEMM input.
//
//   bf16 out :  out = bf16( bf16(gelu_tanh(gate)) * up )
//   FP8 out  :  out = fp8( clamp( bf16( bf16(gelu_tanh(gate)) * up ) / s, +-448 ) )
//
// The activation is rounded to bf16 and the product is rounded again, which is
// what running the pieces as separate kernels produces. activation.cu's
// gate_geglu computes the same mathematical function with one rounding and the
// sigmoid form of gelu_tanh, so it is not a substitute for a caller calibrated
// against these roundings -- the two disagree on the elements near a bf16 or
// e4m3 boundary.
//
// The FP8 form writes only the e4m3 tensor; the bf16 product it is derived from
// never reaches memory, which is the point of fusing it.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace flash_rt {
namespace kernels {
namespace {

__device__ __forceinline__ float gelu_mul_tanh(float x) {
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// the two roundings, in one place so both kernels below cannot drift apart
__device__ __forceinline__ float gelu_mul_product(float g, float u) {
    const float act = __bfloat162float(__float2bfloat16(gelu_mul_tanh(g)));
    return __bfloat162float(__float2bfloat16(act * u));
}

union GeluMulPack { uint4 u; __nv_bfloat16 h[8]; };

__global__ void gelu_mul_bf16_kernel(const __nv_bfloat16* __restrict__ gate,
                                     const __nv_bfloat16* __restrict__ up,
                                     __nv_bfloat16* __restrict__ out, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        out[i] = __float2bfloat16(gelu_mul_product(__bfloat162float(gate[i]),
                                                   __bfloat162float(up[i])));
}

__global__ void gelu_mul_quant_fp8_kernel(const __nv_bfloat16* __restrict__ gate,
                                          const __nv_bfloat16* __restrict__ up,
                                          const float* __restrict__ scale,
                                          __nv_fp8_e4m3* __restrict__ out, long long n8) {
    const float inv = 1.0f / scale[0];
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n8;
         i += (long long)gridDim.x * blockDim.x) {
        GeluMulPack gv, uv;
        gv.u = reinterpret_cast<const uint4*>(gate)[i];
        uv.u = reinterpret_cast<const uint4*>(up)[i];
        __align__(8) __nv_fp8_e4m3 q[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = gelu_mul_product(__bfloat162float(gv.h[j]),
                                             __bfloat162float(uv.h[j]));
            q[j] = __nv_fp8_e4m3(fminf(fmaxf(v * inv, -448.f), 448.f));
        }
        reinterpret_cast<uint2*>(out)[i] = *reinterpret_cast<uint2*>(q);
    }
}

}  // namespace

void gelu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, long long n, cudaStream_t stream) {
    if (n <= 0) return;
    if (!gate || !up || !out)
        throw std::runtime_error("gelu_mul_bf16: null pointer argument");
    long long grid = (n + 255) / 256;
    if (grid > 65535) grid = 65535;
    gelu_mul_bf16_kernel<<<(int)grid, 256, 0, stream>>>(gate, up, out, n);
}

void gelu_mul_quant_fp8_static_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                    const float* act_scale, __nv_fp8_e4m3* out,
                                    long long n, cudaStream_t stream) {
    if (n <= 0) return;
    if (!gate || !up || !act_scale || !out)
        throw std::runtime_error("gelu_mul_quant_fp8_static_bf16: null pointer argument");
    if (n % 8 != 0)
        throw std::runtime_error("gelu_mul_quant_fp8_static_bf16: element count must be a multiple of 8");
    long long grid = (n / 8 + 255) / 256;
    if (grid > 65535) grid = 65535;
    gelu_mul_quant_fp8_kernel<<<(int)grid, 256, 0, stream>>>(gate, up, act_scale, out, n / 8);
}

}  // namespace kernels
}  // namespace flash_rt
