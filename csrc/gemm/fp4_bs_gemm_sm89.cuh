// SPDX-License-Identifier: Apache-2.0
#pragma once
// SM89 NVFP4-weight / FP8-activation block-128 scaled GEMM -> BF16.
// Device kernel: csrc/gemm/fp4_bs_gemm_device.cuh.

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

// A   : [M, K] FP8 e4m3, act_scale [M, K/128] fp32
// Bq  : [N, K/2] packed NVFP4 e2m1
// Bsf : UE4M3 scale factors, CUTLASS Sm1xxBlockScaledConfig<16> order
// gscale : [1] fp32 per-tensor global scale
// w_scale: [N/128, K/128] fp32 from the FP8 block-128 quantization of the same
//          weight; the kernel reconstructs the *normalized* FP8 weight, so this
//          fold is unchanged from the FP8 GEMM.
// D   : [M, N] BF16
int fp4_bs_gemm_sm89_x64(const void* A, const void* Bq, const void* Bsf,
                         const float* gscale, const float* act_scale,
                         const float* w_scale, void* D,
                         int M, int N, int K, cudaStream_t s);

// Tile variants, exported for tuning at the production shapes.
int fp4_bs_gemm_sm89_x64_s1(const void* A, const void* Bq, const void* Bsf,
                            const float* gscale, const float* act_scale,
                            const float* w_scale, void* D,
                            int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_x128(const void* A, const void* Bq, const void* Bsf,
                          const float* gscale, const float* act_scale,
                          const float* w_scale, void* D,
                          int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_x32(const void* A, const void* Bq, const void* Bsf,
                         const float* gscale, const float* act_scale,
                         const float* w_scale, void* D,
                         int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_x64_s4(const void* A, const void* Bq, const void* Bsf,
                            const float* gscale, const float* act_scale,
                            const float* w_scale, void* D,
                            int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_x64_s3(const void* A, const void* Bq, const void* Bsf,
                            const float* gscale, const float* act_scale,
                            const float* w_scale, void* D,
                            int M, int N, int K, cudaStream_t s);
// Wider-M tiles: at M=256 the video stream needs BM>=64 or the shared-memory
// expansion is re-run once per m-tile.
int fp4_bs_gemm_sm89_m64(const void* A, const void* Bq, const void* Bsf,
                         const float* gscale, const float* act_scale,
                         const float* w_scale, void* D,
                         int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_m64n128(const void* A, const void* Bq, const void* Bsf,
                             const float* gscale, const float* act_scale,
                             const float* w_scale, void* D,
                             int M, int N, int K, cudaStream_t s);

// Tile-major packed-weight (Bq) variants: same kernel, same tiles, same
// numerics; Bq is the [N/64, K/128, 64, 64] image written by
// nvfp4_pack_tile_major, which makes each on-GEMM 4 KB weight tile contiguous.
// Per-128 SF variants: Bsf is a [N/128, K/128, 128] u8 image, one UE4M3
// byte per (row, 128-K block). Same payload, tiles and numerics as the
// per-16 packing with the scale duplicated, with 8x fewer SF bytes read.
int fp4_bs_gemm_sm89_x64_sf128(const void* A, const void* Bq, const void* Bsf,
                               const float* gscale, const float* act_scale,
                               const float* w_scale, void* D,
                               int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_x64_s1_sf128(const void* A, const void* Bq, const void* Bsf,
                                  const float* gscale, const float* act_scale,
                                  const float* w_scale, void* D,
                                  int M, int N, int K, cudaStream_t s);

int fp4_bs_gemm_sm89_x64_tm(const void* A, const void* Bq, const void* Bsf,
                            const float* gscale, const float* act_scale,
                            const float* w_scale, void* D,
                            int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_x64_s1_tm(const void* A, const void* Bq, const void* Bsf,
                               const float* gscale, const float* act_scale,
                               const float* w_scale, void* D,
                               int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_x128_tm(const void* A, const void* Bq, const void* Bsf,
                             const float* gscale, const float* act_scale,
                             const float* w_scale, void* D,
                             int M, int N, int K, cudaStream_t s);
int fp4_bs_gemm_sm89_x32_tm(const void* A, const void* Bq, const void* Bsf,
                            const float* gscale, const float* act_scale,
                            const float* w_scale, void* D,
                            int M, int N, int K, cudaStream_t s);

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
