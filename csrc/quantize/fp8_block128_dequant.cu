// SPDX-License-Identifier: Apache-2.0
//
// FP8 block-128 dequantization kernels — implementation.
// Header: fp8_block128_dequant.cuh.
//
// Phase 2.2 / Path D for Qwen3.6 FP8 ckpt: dequantize on the fly so a
// stock cuBLASLt BF16 GEMM can compute the matmul. Slow (3x bandwidth
// vs in-place block-FP8 GEMM) but correct on SM120 / cuBLAS 13 where
// BLK128x128_32F dispatch is unavailable in the deployed cuBLASLt path.
//
// Implementation choices:
//   * One fp8 element per thread (kept simple; vectorization can come
//     after profiling shows dequant is rate-limiting). Memory-bound;
//     coalesced reads along the K axis.
//   * Block is (32 along K, 8 along N) = 256 threads. Each block
//     covers a (8, 32) tile, well below the 128-element scale block
//     so no thread crosses a scale boundary; one scale lookup per
//     thread is fine.
//   * Output is bf16 (CUDA_R_16BF) which the downstream cuBLASLt
//     BF16 GEMM expects.

#include "fp8_block128_dequant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr int kBlock = 128;     // scale block extent (DeepSeek-V3)
constexpr int kBkX   = 32;      // threads along the K axis
constexpr int kBkY   = 8;       // threads along the N (or M) axis

// ----------------------------------------------------------------------
// Weight: (N, K) e4m3 + (N/128, K/128) fp32 scale -> (N, K) bf16.
// ----------------------------------------------------------------------
// Reads (N, K) row-major fp8, writes (N, K) row-major bf16 (in place
// shape, just dtype change). Both reads and writes are coalesced along
// the K axis (innermost). The downstream GEMM applies opT to handle
// the act @ w^T product, which is much cheaper than a scatter-write
// transpose here.
__global__ void weight_dequant_kernel(
    const __nv_fp8_e4m3* __restrict__ in_fp8,
    const float* __restrict__ w_block_scale,
    __nv_bfloat16* __restrict__ out_bf16,
    int N, int K)
{
    const int j = blockIdx.x * kBkX + threadIdx.x;   // K axis
    const int i = blockIdx.y * kBkY + threadIdx.y;   // N axis
    if (i >= N || j >= K) return;

    const int scale_k = K / kBlock;
    const int sb = (i / kBlock) * scale_k + (j / kBlock);
    const float s = w_block_scale[sb];

    const float v = static_cast<float>(in_fp8[i * K + j]) * s;
    out_bf16[i * K + j] = __float2bfloat16(v);
}

// ----------------------------------------------------------------------
// Activation: (M, K) e4m3 + (M, K/128) fp32 scale -> (M, K) bf16.
// ----------------------------------------------------------------------
__global__ void per_token_dequant_kernel(
    const __nv_fp8_e4m3* __restrict__ in_fp8,
    const float* __restrict__ act_block_scale,
    __nv_bfloat16* __restrict__ out_bf16,
    int M, int K)
{
    const int j = blockIdx.x * kBkX + threadIdx.x;   // K axis
    const int i = blockIdx.y * kBkY + threadIdx.y;   // M (token) axis
    if (i >= M || j >= K) return;

    const int scale_k = K / kBlock;
    const int sb = i * scale_k + (j / kBlock);
    const float s = act_block_scale[sb];

    const float v = static_cast<float>(in_fp8[i * K + j]) * s;
    out_bf16[i * K + j] = __float2bfloat16(v);
}

}  // namespace

void fp8_block128_dequantize_to_bf16(
    const void* in_fp8,
    const float* w_block_scale,
    void* out_bf16,
    int N, int K,
    cudaStream_t stream)
{
    dim3 grid((K + kBkX - 1) / kBkX, (N + kBkY - 1) / kBkY);
    dim3 block(kBkX, kBkY);
    weight_dequant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(in_fp8),
        w_block_scale,
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        N, K);
}

void fp8_per_token_block128_dequantize_to_bf16(
    const void* in_fp8,
    const float* act_block_scale,
    void* out_bf16,
    int M, int K,
    cudaStream_t stream)
{
    dim3 grid((K + kBkX - 1) / kBkX, (M + kBkY - 1) / kBkY);
    dim3 block(kBkX, kBkY);
    per_token_dequant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(in_fp8),
        act_block_scale,
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        M, K);
}

}  // namespace quantize
}  // namespace flash_rt
