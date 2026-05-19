// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace smallM_hand {

// Hand-tuned FP8 e4m3 -> BF16 GEMM for sm_120a small-M motus shapes.
// Inputs: FP8 A [M,K] row-major, FP8 B [N,K] row-major (= W.T), BF16 D [M,N].
// alpha = a_scale * w_scale (per-tensor).
// Returns 0 on success.

#define DECL(NAME) \
  int NAME(const void* A, const void* B, void* D, \
           int M, int N, int K, float alpha, cudaStream_t stream)

// 2-stage pipeline.
DECL(fp8_gemm_16x64x128_w4);
DECL(fp8_gemm_16x128x128_w4);
DECL(fp8_gemm_16x256x128_w8);
DECL(fp8_gemm_32x64x128_w4);
DECL(fp8_gemm_32x128x128_w4);
DECL(fp8_gemm_32x128x128_w8);

// 3-stage pipeline.
DECL(fp8_gemm_16x64x128_w4_s3);
DECL(fp8_gemm_16x128x128_w4_s3);
DECL(fp8_gemm_32x64x128_w4_s3);
DECL(fp8_gemm_32x128x128_w4_s3);

// BLOCK_K=256.
DECL(fp8_gemm_16x64x256_w4);
DECL(fp8_gemm_16x128x256_w4);
DECL(fp8_gemm_32x64x256_w4);
DECL(fp8_gemm_32x128x256_w4);

// BLOCK_N=192 (for N=9216 shapes).
DECL(fp8_gemm_16x192x128_w4);
DECL(fp8_gemm_16x192x128_w8);
DECL(fp8_gemm_32x192x128_w4);

// 4-stage pipeline.
DECL(fp8_gemm_16x64x128_w4_s4);
DECL(fp8_gemm_32x64x128_w4_s4);

// Wider BLOCK_N=384 (needs N % 384).
DECL(fp8_gemm_16x384x128_w8);
DECL(fp8_gemm_32x384x128_w8);

// 8-warp variants of 16x64 / 32x64.
DECL(fp8_gemm_16x64x128_w8);
DECL(fp8_gemm_32x64x128_w8);

// 5-stage pipeline.
DECL(fp8_gemm_32x64x128_w4_s5);

// BLOCK_K=64.
DECL(fp8_gemm_16x64x64_w4);
DECL(fp8_gemm_16x128x64_w4);
DECL(fp8_gemm_32x64x64_w4);
DECL(fp8_gemm_32x128x64_w4);
DECL(fp8_gemm_16x64x64_w4_s3);
DECL(fp8_gemm_16x64x64_w4_s4);

// Big-smem BLOCK_N variants.
DECL(fp8_gemm_16x384x128_w4_big);
DECL(fp8_gemm_32x384x128_w4_big);
DECL(fp8_gemm_16x512x128_w8_big);
DECL(fp8_gemm_16x256x128_w4_big);
DECL(fp8_gemm_32x256x128_w4_big);

// BLOCK_M=64 / 128 — wave reduction for M=138 shapes (und_qkv main target).
DECL(fp8_gemm_64x64x128_w4);
DECL(fp8_gemm_64x128x128_w4);
DECL(fp8_gemm_64x128x128_w8);
DECL(fp8_gemm_128x64x128_w4);
DECL(fp8_gemm_128x128x128_w4);
DECL(fp8_gemm_128x128x128_w8);
DECL(fp8_gemm_64x256x128_w4_big);
DECL(fp8_gemm_64x256x128_w8_big);
DECL(fp8_gemm_128x256x128_w8_big);

#undef DECL

}  // namespace smallM_hand
}  // namespace gemm
}  // namespace flash_rt
