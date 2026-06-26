// SPDX-License-Identifier: Apache-2.0
//
// Native Ada (sm_89) FP8 e4m3 -> BF16 block-128 scaled GEMM.
// Header: fp8_block128_gemm_mma_sm89.cuh.
//
// Adapted from csrc/gemm/fp8_smallM_handtuned_sm120.cu (same cp.async
// pipeline + m16n8k32 MMA tiling). Two sm_89-specific changes vs that file:
//   1. MMA uses the plain Ada FP8 op `mma.sync.aligned.m16n8k32.row.col.
//      f32.e4m3.e4m3.f32` (no `.kind::f8f6f4`, which is sm_120a-only).
//   2. Per-tensor `alpha` is replaced by DeepSeek-style block-128 scaling:
//      BLOCK_K is pinned to 128 so each K-iteration is exactly one scale
//      block. Each k-iter accumulates into a temp, then folds
//      act_scale[row,kb] * w_scale[n/128,kb] into the running accumulator.
//
// This reads the FP8 weight directly (no dequant-to-bf16 scratch), cutting
// per-linear weight traffic ~5x vs fp8_block128_gemm_descale_bf16out while
// keeping the per-token activation scale (no precision downgrade).

#include "fp8_block128_gemm_mma_sm89.cuh"
// Device-side kernel body. Shared verbatim with the standalone micro-bench
// (benchmarks/sm89_fp8_block128_gemm), so the bench's `--mode baseline` runs
// this exact kernel and cannot drift behind production.
#include "fp8_bs_gemm_device.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

namespace {

template <int BM, int BN, int W, int STAGES, int MIN_BLK>
int launch_(const void* A, const void* B, void* D,
            int M, int N, int K, const float* act_scale,
            const float* w_scale, cudaStream_t s)
{
    constexpr int BK = 128;
    constexpr int SCALE_KTILE = 8;
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(W * 32, 1, 1);
    // Swizzled A/B cp.async stages (no pad) + staged scale tile.
    int smem_bytes = STAGES * (BM + BN) * BK
                   + (BM * SCALE_KTILE + SCALE_KTILE) * (int)sizeof(float);
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fp8_bs_gemm_kernel<BM, BN, W, STAGES, MIN_BLK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp8_bs_gemm_kernel<BM, BN, W, STAGES, MIN_BLK><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        act_scale, w_scale,
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, K);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
}

}  // namespace

#define DEFINE(NAME, BM, BN, W, S, MB)                                        \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,        \
           const float* act_scale, const float* w_scale, cudaStream_t s) {    \
    return launch_<BM, BN, W, S, MB>(A, B, D, M, N, K, act_scale, w_scale, s);\
  }

DEFINE(fp8_block128_gemm_bs_sm89_32x128x128_w4,   32, 128, 4, 2, 4)
DEFINE(fp8_block128_gemm_bs_sm89_64x128x128_w4,   64, 128, 4, 2, 4)
DEFINE(fp8_block128_gemm_bs_sm89_64x128x128_w8,   64, 128, 8, 2, 4)
DEFINE(fp8_block128_gemm_bs_sm89_128x128x128_w4, 128, 128, 4, 2, 2)
DEFINE(fp8_block128_gemm_bs_sm89_128x128x128_w8, 128, 128, 8, 2, 2)
DEFINE(fp8_block128_gemm_bs_sm89_32x64x128_w4,    32,  64, 4, 2, 4)
DEFINE(fp8_block128_gemm_bs_sm89_64x64x128_w4,    64,  64, 4, 2, 4)
DEFINE(fp8_block128_gemm_bs_sm89_128x64x128_w4,  128,  64, 4, 2, 2)
DEFINE(fp8_block128_gemm_bs_sm89_16x128x128_w4,   16, 128, 4, 2, 4)
DEFINE(fp8_block128_gemm_bs_sm89_16x64x128_w4,    16,  64, 4, 2, 4)
DEFINE(fp8_block128_gemm_bs_sm89_32x128x128_w4_s1, 32, 128, 4, 1, 4)
DEFINE(fp8_block128_gemm_bs_sm89_64x64x128_w4_s1,  64,  64, 4, 1, 4)
DEFINE(fp8_block128_gemm_bs_sm89_128x128x128_w8_s1, 128, 128, 8, 1, 2)

#undef DEFINE

int fp8_block128_gemm_blockscaled_sm89_bf16out(
    const void* A, const void* B, void* D, int M, int N, int K,
    const float* act_scale, const float* w_scale, cudaStream_t stream)
{
    if ((N % 128) != 0)
        throw std::runtime_error(
            "fp8_block128_gemm_blockscaled_sm89_bf16out requires N multiple of 128");
    if ((K % 128) != 0)
        throw std::runtime_error(
            "fp8_block128_gemm_blockscaled_sm89_bf16out requires K multiple of 128");
    // Tuned on 4090 over Qwen3-VL-8B-FP8 layer shapes (qkv 6144, o 4096,
    // gate/up 12288, down 4096x12288) at S=79..256. BLOCK_M=32 keeps grid
    // occupancy high at small M; BLOCK_N=64 wins until M crosses ~128, then
    // the wider BLOCK_N=128 amortizes better. Tiny-N (<2048) prefers BLOCK_N=64.
    //
    // ViT prefill is a different regime: full-res FlashRT.png runs M=6256.
    // On these large-M shapes the language-prefill heuristic is wrong for
    // the small-N linears:
    //   - patch_embed / proj   (N=1152, K≈1152..1536) prefer 32x128
    //   - fc2 / merger-fc2     (N=1152, K>=4096)      prefer 64x64
    // Keep the original small-M path intact and only branch once the grid is
    // already abundant (M>=2048), so text prefill / decode remain unchanged.
    if (N < 2048)
    {
        if (M >= 2048) {
            if (K >= 4096)
                return fp8_block128_gemm_bs_sm89_64x64x128_w4(
                    A, B, D, M, N, K, act_scale, w_scale, stream);
            return fp8_block128_gemm_bs_sm89_32x128x128_w4(
                A, B, D, M, N, K, act_scale, w_scale, stream);
        }
        return fp8_block128_gemm_bs_sm89_16x64x128_w4(
            A, B, D, M, N, K, act_scale, w_scale, stream);
    }
    if (M < 128)
        return fp8_block128_gemm_bs_sm89_32x64x128_w4(
            A, B, D, M, N, K, act_scale, w_scale, stream);
    // Language prefill (M>=128, N>=2048) is limited by low eligible warps on
    // Ada. A single cp.async stage reduces shared-memory pressure and wins on
    // Qwen3-VL 2B/8B prefill shapes. Keep a short-prefill exception for the
    // wide 8B MLP, where the 8-warp tile remains slightly faster.
    if (N >= 8192 && K == 4096 && M < 1024)
        return fp8_block128_gemm_bs_sm89_128x128x128_w8_s1(
            A, B, D, M, N, K, act_scale, w_scale, stream);
    if (N == 2048 && M < 1024)
        return fp8_block128_gemm_bs_sm89_64x64x128_w4(
            A, B, D, M, N, K, act_scale, w_scale, stream);
    return fp8_block128_gemm_bs_sm89_64x64x128_w4_s1(
        A, B, D, M, N, K, act_scale, w_scale, stream);
}

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
