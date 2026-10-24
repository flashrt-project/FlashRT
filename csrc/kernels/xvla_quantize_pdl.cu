// FlashRT — opt-in NVFP4 activation quantize v3 (X-VLA).
//
// Bit-identical to quantize_bf16_to_nvfp4_swizzled_v2 for the eligible shapes:
// the same per-16-element block max, UE4M3 ceiling, e2m1 encode and swizzled
// scale address, with one thread per block instead of one row per CTA.  The
// existing v2 entry is not modified; ineligible shapes fall back to it.
#include "xvla_quantize_pdl.cuh"
#include "pdl_switch.cuh"
#include "quantize.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "nvfp4_convert.cuh"

__global__ void quantize_bf16_to_nvfp4_swizzled_v3_kernel(
    const __nv_bfloat16* __restrict__ input, uint8_t* __restrict__ fp4_data,
    uint8_t* __restrict__ scale_factors, int blocks_per_row, int total_blocks,
    int n_col_blocks) {
    flashrt_pdl_wait_and_trigger();
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_blocks) return;
    const int row = g / blocks_per_row;
    const int blk = g - row * blocks_per_row;

    const uint4* src = reinterpret_cast<const uint4*>(
        input + (size_t)row * blocks_per_row * 16 + (size_t)blk * 16);
    const uint4 lo = src[0], hi = src[1];
    const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    float vals[16];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 xv =
            *reinterpret_cast<const __nv_bfloat162*>(&words[j]);
        vals[2 * j] = __bfloat162float(xv.x);
        vals[2 * j + 1] = __bfloat162float(xv.y);
        amax = fmaxf(amax, fmaxf(fabsf(vals[2 * j]), fabsf(vals[2 * j + 1])));
    }

    const uint8_t ue = float_to_ue4m3_ceil(amax / 6.0f);
    const int rb = row >> 7, ri = row & 127;
    scale_factors[(rb * n_col_blocks + (blk >> 2)) * 512 + (ri % 32) * 16 +
                  (ri / 32) * 4 + (blk & 3)] = ue;

    const float s = ue4m3_to_float(ue);
    const float inv = (s > 0.f) ? 1.f / s : 0.f;
    uint64_t enc = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint8_t byte =
            (uint8_t)((float_to_fp4_e2m1_branchless(vals[2 * j + 1] * inv) << 4) |
                      (float_to_fp4_e2m1_branchless(vals[2 * j] * inv) & 0x0F));
        enc |= (uint64_t)byte << (8 * j);
    }
    *reinterpret_cast<uint64_t*>(fp4_data + (size_t)g * 8) = enc;
}

void quantize_bf16_to_nvfp4_swizzled_v3(
    const __nv_bfloat16* input, uint8_t* fp4_data, uint8_t* scale_factors,
    int rows, int cols, cudaStream_t stream) {
    const bool eligible =
        rows > 0 && cols > 0 && (cols & 15) == 0 &&
        ((reinterpret_cast<uintptr_t>(input) |
          reinterpret_cast<uintptr_t>(fp4_data)) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(fp4_data) & 7) == 0;
    if (!eligible) {
        quantize_bf16_to_nvfp4_swizzled_v2(input, fp4_data, scale_factors,
                                           rows, cols, stream);
        return;
    }
    const int num_blocks = (cols + 15) / 16;
    const int n_col_blocks = (num_blocks + 3) / 4;
    const int blocks_per_row = cols >> 4;
    const int total_blocks = rows * blocks_per_row;
    flash_rt::pdl::launch(quantize_bf16_to_nvfp4_swizzled_v3_kernel,
                          dim3((total_blocks + 255) / 256), dim3(256), 0,
                          stream, input, fp4_data, scale_factors,
                          blocks_per_row, total_blocks, n_col_blocks);
}
