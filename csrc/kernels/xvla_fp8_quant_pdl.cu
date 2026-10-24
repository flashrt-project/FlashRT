// FlashRT — opt-in FP8 per-token block quantize v2 (X-VLA).
//
// Bit-identical to fp8_per_token_block128_quant_bf16: the maximum over the
// same 128-element block, the same scale = max(amax/448, 1e-12), the same
// e4m3 clamp.  Only the thread mapping changes (one warp per (m, kb), four
// elements per thread); the existing entry is untouched.
#include "xvla_fp8_quant_pdl.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace flash_rt {
namespace quantize {
namespace {

constexpr int kBlock = 128;
constexpr float kFp8Max = 448.0f;

__global__ void fp8_per_token_block_quant_v2_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int M, int K) {
    const int kb = blockIdx.x;
    const int m = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (m >= M) return;
    if (kb * kBlock >= K) return;

    const int t = threadIdx.x & 31;
    const int k = kb * kBlock + t * 4;
    const int row = m * K;

    float v[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int kk = k + i;
        v[i] = (kk < K) ? static_cast<float>(input[row + kk]) : 0.0f;
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }

    float s = 0.0f;
    if (t == 0) {
        s = fmaxf(amax / kFp8Max, 1.0e-12f);
        scale[m * (K / kBlock) + kb] = s;
    }
    s = __shfl_sync(0xffffffff, s, 0);
    const float inv_s = 1.0f / s;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int kk = k + i;
        if (kk < K) {
            const float q = fminf(fmaxf(v[i] * inv_s, -kFp8Max), kFp8Max);
            output[row + kk] = __nv_fp8_e4m3(q);
        }
    }
}

}  // namespace

void fp8_per_token_block128_quant_bf16_v2(
    const void* input, void* output_fp8, float* output_scale, int M, int K,
    cudaStream_t stream) {
    if ((K % kBlock) != 0) {
        throw std::runtime_error(
            "fp8_per_token_block128_quant_bf16_v2 requires K multiple of 128");
    }
    constexpr int kWarpsPerBlock = 4;
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(K / kBlock, (M + kWarpsPerBlock - 1) / kWarpsPerBlock);
    fp8_per_token_block_quant_v2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<__nv_fp8_e4m3*>(output_fp8), output_scale, M, K);
}

}  // namespace quantize
}  // namespace flash_rt
