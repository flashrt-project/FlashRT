// SPDX-License-Identifier: Apache-2.0
// GeGLU over a merged [gate | up] BF16 buffer straight into NVFP4 (e2m1
// with per-16 UE4M3 block scales in the cuBLASLt swizzled layout, global
// scale 1): out = quant(gelu_tanh(g) * u). One pass over the merged input,
// one CTA per (row, 2048-column chunk of the output) so long rows do not
// serialize on a single CTA the way the row-per-CTA quantizer does.
#include "pi05_geglu_nvfp4_quant.cuh"
#include "nvfp4_convert.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {
namespace {

constexpr int CHUNK = 2048;      // output columns per CTA
constexpr int THREADS = 256;     // 8 columns per thread = half an NVFP4 block

__device__ __forceinline__ float tanh_gelu(float g) {
    return g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
}

__global__ void __launch_bounds__(THREADS)
geglu_merged_to_nvfp4_swizzled_kernel(const __nv_bfloat16* __restrict__ merged,
                                      uint8_t* __restrict__ fp4_out,
                                      uint8_t* __restrict__ sf_out,
                                      int half, int n_col_blocks) {
    const int row = blockIdx.x;
    const int c0 = blockIdx.y * CHUNK + threadIdx.x * 8;
    if (c0 >= half) return;
    const __nv_bfloat16* g = merged + static_cast<size_t>(row) * 2 * half + c0;
    const __nv_bfloat16* u = g + half;
    float v[8];
    {
        const uint4 gp = *reinterpret_cast<const uint4*>(g);
        const uint4 up = *reinterpret_cast<const uint4*>(u);
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gp);
        const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&up);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 gf = __bfloat1622float2(g2[i]);
            const float2 uf = __bfloat1622float2(u2[i]);
            // gate_geglu_merged rounds the product to BF16 before the FP8/FP4
            // quantizer of the unfused path sees it; keep that rounding.
            v[2 * i] = __bfloat162float(__float2bfloat16(tanh_gelu(gf.x) * uf.x));
            v[2 * i + 1] = __bfloat162float(__float2bfloat16(tanh_gelu(gf.y) * uf.y));
        }
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
    // the block of 16 spans this thread and its neighbour (lane ^ 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    const uint8_t ue = float_to_ue4m3_ceil(amax / 6.0f);
    const float scale = ue4m3_to_float(ue);
    const float inv = scale > 0.f ? 1.0f / scale : 0.f;
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) packed |= static_cast<uint32_t>(float_to_fp4_e2m1_branchless(v[i] * inv) & 0xF) << (4 * i);
    *reinterpret_cast<uint32_t*>(fp4_out + (static_cast<size_t>(row) * half + c0) / 2) = packed;
    if ((threadIdx.x & 1) == 0) {
        const int b = c0 >> 4;                 // NVFP4 block index along the row
        const int rb = row >> 7, ri = row & 127;
        const int cb = b >> 2, ci = b & 3;
        sf_out[(static_cast<size_t>(rb) * n_col_blocks + cb) * 512 + (ri & 31) * 16 + (ri >> 5) * 4 + ci] = ue;
    }
}

}  // namespace

int pi05_geglu_merged_to_nvfp4_swizzled(const __nv_bfloat16* merged, uint8_t* fp4_out, uint8_t* sf_out,
                                   int rows, int half, cudaStream_t stream) {
    if (rows < 1 || half % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const int n_blocks = half / 16;
    const int n_col_blocks = (n_blocks + 3) / 4;
    const dim3 grid(rows, (half + CHUNK - 1) / CHUNK);
    geglu_merged_to_nvfp4_swizzled_kernel<<<grid, THREADS, 0, stream>>>(merged, fp4_out, sf_out, half, n_col_blocks);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace quantize
}  // namespace flash_rt
