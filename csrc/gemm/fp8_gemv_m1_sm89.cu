// SPDX-License-Identifier: Apache-2.0
//
// M=1 FP8 e4m3 -> BF16 block-128 scaled GEMV for SM89 Qwen3-VL decode.
// Header: fp8_gemv_m1_sm89.cuh.
//
// Split out of the SM120 per-tensor GEMV (fp8_gemv_m1_sm120) so the SM89
// block-128 decode path owns its own file: per-token activation scale
// [K/128] and DeepSeek-style weight block scale [N/128, K/128], applied in
// the warp reduction. Warp-per-output-row, A held in smem, 16-byte coalesced
// B loads. No MMA / no padding tax (M=1).

#include "fp8_gemv_m1_sm89.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace gemv_m1_sm89 {

namespace {

// minBlocks=8 only fits blocks of <=192 threads: 8 x 256 exceeds the 1536
// threads/SM cap on sm_87/89/110/120/121, so ptxas discards the whole hint.
// Clamp it to what the arch can hold, keeping the honored W=4 hint intact.
constexpr int kMaxThreadsPerSM = 1536;
template <int W>
constexpr int kMinBlocks = (kMaxThreadsPerSM / (W * 32)) < 8
                               ? (kMaxThreadsPerSM / (W * 32))
                               : 8;

template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, kMinBlocks<WARPS_PER_BLOCK>)
void gemv_fp8_block128_m1_kernel(
    const __nv_fp8_e4m3* __restrict__ A,   // [K]
    const __nv_fp8_e4m3* __restrict__ B,   // [N, K]
    __nv_bfloat16* __restrict__ D,         // [N]
    int N, int K,
    const float* __restrict__ act_scale,   // [K/128]
    const float* __restrict__ w_scale,     // [N/128, K/128]
    float alpha)
{
    extern __shared__ __nv_fp8_e4m3 sA[];
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5;
    const int threads = WARPS_PER_BLOCK * 32;
    const int K16     = K >> 4;
    const int K128    = K >> 7;

    uint4* sA16 = reinterpret_cast<uint4*>(sA);
    const uint4* A16 = reinterpret_cast<const uint4*>(A);
    for (int i = tid; i < K16; i += threads) sA16[i] = A16[i];
    __syncthreads();

    const int n = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    const uint4* Brow = reinterpret_cast<const uint4*>(B) + (size_t)n * K16;
    const __nv_fp8_e4m3* sAf = sA;
    const float* w_scale_row = w_scale + (size_t)(n >> 7) * K128;
    float acc = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        const int kb = i >> 3;
        const float s = act_scale[kb] * w_scale_row[kb] * alpha;
        uint4 bpack = Brow[i];
        const __nv_fp8_e4m3* bp =
            reinterpret_cast<const __nv_fp8_e4m3*>(&bpack);
        const __nv_fp8_e4m3* ap = sAf + (i << 4);
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc += float(ap[j]) * float(bp[j]) * s;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) D[n] = __float2bfloat16(acc);
}

template <int W>
int launch_block128_(const void* A, const void* B, void* D,
                     int /*M*/, int N, int K,
                     const float* act_scale, const float* w_scale,
                     float alpha, cudaStream_t stream) {
    dim3 grid((N + W - 1) / W);
    dim3 block(W * 32);
    size_t smem = (size_t)K * sizeof(__nv_fp8_e4m3);
    gemv_fp8_block128_m1_kernel<W><<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<__nv_bfloat16*>(D),
        N, K, act_scale, w_scale, alpha);
    return 0;
}

// BF16-input variant: skips activation FP8 quantization.
// A is BF16, B is FP8 with block-128 weight scale. No act_scale needed.
template <int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, kMinBlocks<WARPS_PER_BLOCK>)
void gemv_fp8_block128_m1_bf16in_kernel(
    const __nv_bfloat16* __restrict__ A,   // [K] BF16
    const __nv_fp8_e4m3* __restrict__ B,   // [N, K] FP8
    __nv_bfloat16* __restrict__ D,         // [N]
    int N, int K,
    const float* __restrict__ w_scale)     // [N/128, K/128]
{
    extern __shared__ __nv_bfloat16 sA_bf16[];
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5;
    const int threads = WARPS_PER_BLOCK * 32;
    const int K16     = K >> 4;
    const int K128    = K >> 7;

    // Load BF16 activation (2 bytes each) via uint32 pairs.
    uint* sU = reinterpret_cast<uint*>(sA_bf16);
    const uint* AU = reinterpret_cast<const uint*>(A);
    const int Khalf = K >> 1;
    for (int i = tid; i < Khalf; i += threads) sU[i] = AU[i];
    __syncthreads();

    const int n = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    const uint4* Brow = reinterpret_cast<const uint4*>(B) + (size_t)n * K16;
    const float* w_scale_row = w_scale + (size_t)(n >> 7) * K128;
    float acc = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        const int kb = i >> 3;
        const float ws = w_scale_row[kb];
        uint4 bpack = Brow[i];
        const __nv_fp8_e4m3* bp =
            reinterpret_cast<const __nv_fp8_e4m3*>(&bpack);
        const __nv_bfloat16* ap = sA_bf16 + (i << 4);
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            dot += __bfloat162float(ap[j]) * float(bp[j]);
        }
        acc += dot * ws;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) D[n] = __float2bfloat16(acc);
}

template <int W>
int launch_block128_bf16in_(const void* A, const void* B, void* D,
                            int /*M*/, int N, int K,
                            const float* w_scale, cudaStream_t stream) {
    dim3 grid((N + W - 1) / W);
    dim3 block(W * 32);
    size_t smem = (size_t)K * sizeof(__nv_bfloat16);
    gemv_fp8_block128_m1_bf16in_kernel<W><<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<__nv_bfloat16*>(D),
        N, K, w_scale);
    return 0;
}

}  // namespace

#define DEFINE_BLOCK128(NAME, W)                                               \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,         \
           const float* act_scale, const float* w_scale, float alpha,          \
           cudaStream_t stream) {                                              \
    return launch_block128_<W>(A, B, D, M, N, K, act_scale, w_scale, alpha,    \
                               stream);                                        \
  }

DEFINE_BLOCK128(gemv_fp8_block128_m1_w4, 4)
DEFINE_BLOCK128(gemv_fp8_block128_m1_w8, 8)
DEFINE_BLOCK128(gemv_fp8_block128_m1_w16, 16)

#undef DEFINE_BLOCK128

#define DEFINE_BLOCK128_BF16IN(NAME, W)                                        \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,         \
           const float* w_scale, cudaStream_t stream) {                        \
    return launch_block128_bf16in_<W>(A, B, D, M, N, K, w_scale, stream);      \
  }

DEFINE_BLOCK128_BF16IN(gemv_fp8_block128_m1_bf16in_w8, 8)
DEFINE_BLOCK128_BF16IN(gemv_fp8_block128_m1_bf16in_w16, 16)

#undef DEFINE_BLOCK128_BF16IN

}  // namespace gemv_m1_sm89
}  // namespace gemm
}  // namespace flash_rt
