// SPDX-License-Identifier: Apache-2.0
#pragma once
// M=128 NVFP4-weight / FP8-activation block-128 scaled GEMM -> BF16 for the
// Wan2.2 video stream. Device body: fp4_stream_m128_sm89.cu.
//
// The weight operand is the same fragment-major NVFP4 payload the M=16
// stream kernel reads (nvfp4_pack_frag_m16), decoded in registers by the same
// per-(row, K-block) e4m3 LUT. At M=128 every decoded B fragment feeds
// M/16 MMAs, so the activation tile (staged per 128-K block in shared memory
// and read with ldmatrix) and the tensor core, not the decode, set the pace.
//
// F16 block accumulation (f16acc = 1): each 128-K block's product runs on the
// double-rate `mma ... .f16.e4m3.e4m3.f16` form and is folded into the FP32
// running accumulator with the block scales. To keep every block sum inside
// the FP16 range, the LUT of each (weight row, K-block) is pre-shifted by the
// power of two 2^-e that brings its largest decoded magnitude to <= 0.875;
// the fold multiplies by 2^e. With |a| <= 448 a 128-term block sum is then
// bounded by 128 * 448 * 0.875 = 50176. The shift is exact: the smallest
// non-zero e2m1 code is 1/12 of the largest, so every shifted LUT entry stays
// an e4m3 normal. f16acc = 0 runs the FP32-accumulating MMA instead.
//
// A    : [M, K] FP8 e4m3 row-major, act_scale [M, K/128] fp32; M % 64 == 0
// Wf   : fragment-major payload from nvfp4_pack_frag_m16 ([N/8, K/128, 32, 16B])
// Bsf  : UE4M3 SF128 scale factors, [N/128, K/128, 128] u8
// gscale : [1] fp32 per-tensor global scale
// w_scale: [N/128, K/128] fp32 from the FP8 block-128 quantization
// D    : [M, N] BF16

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

// cfg selects the tile: 0 = 128 rows x 16 cols per warp, 8 warps;
// 1 = 64 rows x 16 cols per warp, 8 warps; 2 = 128 rows x 8 cols per warp,
// 8 warps; 3 = 128 rows x 16 cols per warp, 4 warps.
int fp4_stream_m128_sm89(const void* A, const void* Wf, const void* Bsf,
                         const float* gscale, const float* act_scale,
                         const float* w_scale, void* D, int M, int N, int K,
                         int cfg, int f16acc, cudaStream_t s);

// LUT image for fp4_stream_m128_lut_sm89: one 96-byte record per (n8 tile,
// K-block) -- 8 rows x {P0, P1} e4m3 LUT words, then 8 rows x fp32 fold
// factor -- computed from (Bsf, gscale, w_scale) with the same expressions the
// kernel above evaluates per block. lut: [N/8, K/128, 96] u8.
int nvfp4_m128_lut_build(const void* Bsf, const float* gscale, const float* w_scale,
                         void* lut, int N, int K, cudaStream_t s);

// Same GEMM (F16 block accumulation) reading the precomputed LUT image and
// staging A, the weight fragments and the LUT records through one cp.async
// ring. Bit-identical to fp4_stream_m128_sm89(..., f16acc = 1). M % 128 == 0.
int fp4_stream_m128_lut_sm89(const void* A, const void* Wf, const void* Lut,
                             const float* act_scale, void* D, int M, int N, int K,
                             int cfg, cudaStream_t s);

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
