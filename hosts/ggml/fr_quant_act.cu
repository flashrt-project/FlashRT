// fp32 activation quantizer: [M, K] fp32 row-major -> NVFP4 packed + SFA
// scales at the CUTLASS Sm1xx block-scaled atom-layout offsets.
//
// Vendored from FlashRT's quantize_fp4_sfa_bf16.cu (Apache-2.0) with the
// source element type changed from bf16 to fp32. One thread quantizes one
// 16-element block: four 16-byte loads, one 8-byte packed store, one SFA
// byte at the tile-interleaved offset.

#include "fr_kernels.h"

#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace ggml_cuda_flashrt {

namespace {

using CfgVec = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e2m1(float x) {
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

template <class LayoutSF>
__global__ void kernel_quantize_f32(
        const float4 * __restrict__ src,   // fp32 [M, K] as float4 (4 elements)
        uint2 * __restrict__ dst_packed,   // [M, K/2] bytes as uint2 (1 block)
        uint8_t * __restrict__ dst_sfa,
        LayoutSF layout,
        int M, int K4) {                   // K4 = K / 4 float4 chunks per row
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int n_blocks = K4 >> 2;          // 16 elements per block
    if (row >= M || block_idx >= n_blocks) return;

    float vals[16];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float4 raw = src[row * K4 + 4 * block_idx + c];
        vals[4 * c + 0] = raw.x;
        vals[4 * c + 1] = raw.y;
        vals[4 * c + 2] = raw.z;
        vals[4 * c + 3] = raw.w;
    }

    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float a = fabsf(vals[i]);
        if (a > amax) amax = a;
    }

    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(desired);
    const float bs_dq = static_cast<float>(bs_q);

    dst_sfa[layout(row, block_idx * 16, 0)] = *reinterpret_cast<uint8_t *>(&bs_q);

    const float inv_bs = 1.f / bs_dq;
    uint2 out;
    uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint8_t lo = fp32_to_e2m1(vals[2 * p]     * inv_bs);
        const uint8_t hi = fp32_to_e2m1(vals[2 * p + 1] * inv_bs);
        ob[p] = static_cast<uint8_t>(lo | (hi << 4));
    }
    dst_packed[row * n_blocks + block_idx] = out;
}

} // namespace

int quantize_act_f32(const float * src, void * dst_packed, void * dst_sfa,
                     int M, int K, cudaStream_t stream) {
    if (K % 16 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(src) & 15) ||
        (reinterpret_cast<uintptr_t>(dst_packed) & 7)) return -1;

    const int n_blocks = K / 16;
    const int threads = 128;
    dim3 grid((n_blocks + threads - 1) / threads, M);

    auto shape = cute::make_shape(M, 1, K, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFA(shape);

    kernel_quantize_f32<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const float4 *>(src),
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sfa),
        layout, M, K >> 2);

    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

} // namespace ggml_cuda_flashrt
