// SPDX-License-Identifier: Apache-2.0
#pragma once
// Split-K NVFP4-weight / FP8-activation block-128 scaled GEMM -> BF16.
// Device body: fp4_bs_gemm_splitk_device.cuh.
//
// The k-loop is split across blockIdx.z; each split writes an fp32 partial to
// Dpart (splits x [M,N]) and a reduction kernel sums them and rounds once to
// bf16. Adding CTAs is what recovers the narrow-N shapes (N=3072 gives only 48
// CTAs at BN=64, too few to hide the shared-memory expansion); measured
// 1.15-1.18x over the unsplit FP4 kernel at M=16, N=K=3072.
//
// A   : [M, K] FP8 e4m3, act_scale [M, K/128] fp32
// Bq  : [N, K/2] packed NVFP4 e2m1
// Bsf : UE4M3 scale factors, CUTLASS Sm1xxBlockScaledConfig<16> order
// gscale : [1] fp32 per-tensor global scale
// w_scale: [N/128, K/128] fp32 from the FP8 block-128 quantization
// D   : [M, N] BF16
// Dpart: [splits, M, N] fp32 workspace, caller-owned

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

int fp4_bs_gemm_splitk_sm89(const void* A, const void* Bq, const void* Bsf,
                            const float* gscale, const float* act_scale,
                            const float* w_scale, void* D, void* Dpart,
                            int M, int N, int K, int splits, cudaStream_t s);

// Tile-major Bq variant: Bq is the [N/64, K/128, 64, 64] image from
// nvfp4_pack_tile_major. Same geometry, same numerics.
int fp4_bs_gemm_splitk_sm89_tm(const void* A, const void* Bq, const void* Bsf,
                               const float* gscale, const float* act_scale,
                               const float* w_scale, void* D, void* Dpart,
                               int M, int N, int K, int splits,
                               cudaStream_t s);

// Per-128 SF variants (Bsf = [N/128, K/128, 128] u8).
int fp4_bs_gemm_splitk_sm89_sf128(const void* A, const void* Bq, const void* Bsf,
                                  const float* gscale, const float* act_scale,
                                  const float* w_scale, void* D, void* Dpart,
                                  int M, int N, int K, int splits,
                                  cudaStream_t s);
int fp4_bs_gemm_splitk_sm89_tm_sf128(const void* A, const void* Bq, const void* Bsf,
                                     const float* gscale, const float* act_scale,
                                     const float* w_scale, void* D, void* Dpart,
                                     int M, int N, int K, int splits,
                                     cudaStream_t s);

// Split-K tile-major SF128 with the attn2-out residual+bias folded into the
// reduce kernel (D is not written; residual is updated in place).
int fp4_bs_gemm_splitk_sm89_tm_sf128_res(
    const void* A, const void* Bq, const void* Bsf, const float* gscale,
    const float* act_scale, const float* w_scale, void* Dpart,
    const void* residual, const void* bias, int M, int N, int K, int splits,
    cudaStream_t s);

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
