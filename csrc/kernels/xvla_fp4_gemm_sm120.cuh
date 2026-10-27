// FlashRT — X-VLA small-M NVFP4 W4A4 GEMMs (SM120).
//
// The X-VLA RTX pipeline runs every NVFP4 GEMM at M = 262 (action
// transformer), 100 (BART encoder) or 30 (pruned last action block). The
// CUTLASS SM120 block-scaled collective only admits 128x128 output tiles (the
// UE4M3 scale-factor TMA atom is 128 rows wide), so at these M the grid is
// 8-24 CTAs on a 36-SM part and a third of every 128-row tile is padding.
//
// These entries keep the same operands -- packed e2m1 rows plus the
// cutlass-swizzled UE4M3 scale atoms that `quantize_bf16_to_nvfp4_swizzled*`
// and `bf16_weight_to_nvfp4_swizzled` produce -- and the same arithmetic:
// `mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`, each output
// accumulated over K in 64-wide chunks in order, then the cutlass epilogue
// (`fma(alpha, acc, bias)`). Outputs are bit-identical to
// `fp4_w4a16_gemm_sm120_bf16out_{bias,pingpong_bias}` and
// `fp4_w4a16_gemm_bias_gelu_fp4out_sm120`. What changes is the tiling: M tiles
// of 32-112 rows chosen per shape, a TMA producer warp (128B/64B swizzled
// data, 1D bulk copies of whole scale atoms) and ldmatrix consumer warps.
//
// Requirements: N % 128 == 0, K % 256 == 0; the A scale buffer covers
// ceil(M / 128) * 128 rows (as `nvfp4_sf_swizzled_bytes` allocates). Rows of D
// at and beyond M are not written. Weights (B, SFB) for the first pipeline
// stages are fetched before the programmatic-dependency wait.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace xvla {

// D[M,N] (bf16) = alpha * A[M,K] . B[N,K]^T + bias[N]. Returns 0 on success.
int fp4_gemm_bias_bf16out(const void* A_packed, const void* B_packed, void* D,
                          const void* bias, int M, int N, int K,
                          const void* SFA, const void* SFB, float alpha,
                          cudaStream_t stream);

// D_packed[M,N] (e2m1) + SFD (swizzled UE4M3, 16-column blocks) =
//   NVFP4(GELU_tanh(alpha * A . B^T + bias)), the cutlass
//   LinCombPerColBiasEltActBlockScaleFactor<GELU_taylor> epilogue.
int fp4_gemm_bias_gelu_fp4out(const void* A_packed, const void* B_packed,
                              const void* SFA, const void* SFB, const void* bias,
                              void* D_packed, void* SFD, int M, int N, int K,
                              float alpha, cudaStream_t stream);

}  // namespace xvla
}  // namespace flash_rt
