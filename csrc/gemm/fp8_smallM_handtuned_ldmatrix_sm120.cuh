// SPDX-License-Identifier: Apache-2.0
//
// Hand-tuned FP8 e4m3 GEMM v2 — adds 128B swizzle smem layout + ldmatrix.x4
// loads to clear bank conflicts that bottleneck v1 (`fp8_smallM_handtuned`).
//
// All variants restricted to BLOCK_K = 128 (natural 128B swizzle stride).

#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {
namespace smallM_ld {

#define DECL(NAME) \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K, \
           float alpha, cudaStream_t stream)

DECL(ld_fp8_gemm_16x64x128_w4);
DECL(ld_fp8_gemm_16x128x128_w4);
DECL(ld_fp8_gemm_16x256x128_w8);
DECL(ld_fp8_gemm_32x64x128_w4);
DECL(ld_fp8_gemm_32x128x128_w4);
DECL(ld_fp8_gemm_32x128x128_w8);

DECL(ld_fp8_gemm_16x64x128_w4_s3);
DECL(ld_fp8_gemm_16x128x128_w4_s3);
DECL(ld_fp8_gemm_32x64x128_w4_s3);
DECL(ld_fp8_gemm_32x128x128_w4_s3);

DECL(ld_fp8_gemm_16x192x128_w4);
DECL(ld_fp8_gemm_32x192x128_w4);

DECL(ld_fp8_gemm_16x64x128_w4_s4);
DECL(ld_fp8_gemm_16x64x128_w4_s5);
DECL(ld_fp8_gemm_32x64x128_w4_s4);
DECL(ld_fp8_gemm_32x64x128_w4_s5);
DECL(ld_fp8_gemm_16x128x128_w4_s4);
DECL(ld_fp8_gemm_32x128x128_w4_s4);

DECL(ld_fp8_gemm_16x64x256_w4);
DECL(ld_fp8_gemm_16x128x256_w4);
DECL(ld_fp8_gemm_32x64x256_w4);
DECL(ld_fp8_gemm_32x128x256_w4);
DECL(ld_fp8_gemm_16x64x256_w4_s3);

DECL(ld_fp8_gemm_16x64x64_w4);
DECL(ld_fp8_gemm_16x128x64_w4);
DECL(ld_fp8_gemm_32x64x64_w4);
DECL(ld_fp8_gemm_16x64x64_w4_s3);
DECL(ld_fp8_gemm_16x64x64_w4_s4);

// und_qkv attack variants (M=188, K=512)
DECL(ld_fp8_gemm_64x64x128_w4);
DECL(ld_fp8_gemm_64x128x128_w4);
DECL(ld_fp8_gemm_64x64x256_w4);
DECL(ld_fp8_gemm_64x128x256_w4);
DECL(ld_fp8_gemm_64x64x256_w4_s3);
DECL(ld_fp8_gemm_32x64x256_w4_s3);
DECL(ld_fp8_gemm_32x128x256_w4_s3);
DECL(ld_fp8_gemm_128x64x128_w4);
DECL(ld_fp8_gemm_128x128x128_w4);

#undef DECL

}  // namespace smallM_ld
}  // namespace gemm
}  // namespace flash_rt
