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

template <int BM, int BN, int W, int STAGES, int MIN_BLK, bool EXACT = false>
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
            (const void*)&fp8_bs_gemm_kernel<BM, BN, W, STAGES, MIN_BLK, false, EXACT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp8_bs_gemm_kernel<BM, BN, W, STAGES, MIN_BLK, false, EXACT><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        act_scale, w_scale,
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, K);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
}

// Residual-fold launch: D = bf16(acc) + resid, fusing the residual add into the
// GEMM epilogue (no separate residual_add launch, no D HBM round-trip). resid
// is [M, N] BF16 row-major, same layout as D. See fp8_bs_gemm_device.cuh.
template <int BM, int BN, int W, int STAGES, int MIN_BLK>
int launch_resid_(const void* A, const void* B, void* D,
                  int M, int N, int K, const float* act_scale,
                  const float* w_scale, const void* resid, cudaStream_t s)
{
    constexpr int BK = 128;
    constexpr int SCALE_KTILE = 8;
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(W * 32, 1, 1);
    int smem_bytes = STAGES * (BM + BN) * BK
                   + (BM * SCALE_KTILE + SCALE_KTILE) * (int)sizeof(float);
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fp8_bs_gemm_kernel<BM, BN, W, STAGES, MIN_BLK, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp8_bs_gemm_kernel<BM, BN, W, STAGES, MIN_BLK, true><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        act_scale, w_scale,
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, K,
        reinterpret_cast<const __nv_bfloat16*>(resid));
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
}

}  // namespace

#define DEFINE(NAME, BM, BN, W, S, MB)                                        \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,        \
           const float* act_scale, const float* w_scale, cudaStream_t s) {    \
    return launch_<BM, BN, W, S, MB>(A, B, D, M, N, K, act_scale, w_scale, s);\
  }

// Residual-fold variants (suffix _resid). D = bf16(acc) + resid. Only the
// tiles the prefill down-proj actually selects are defined; additive — the
// non-resid DEFINE list above is unchanged.
#define DEFINE_RESID(NAME, BM, BN, W, S, MB)                                  \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,        \
           const float* act_scale, const float* w_scale, const void* resid,   \
           cudaStream_t s) {                                                  \
    return launch_resid_<BM, BN, W, S, MB>(A, B, D, M, N, K, act_scale,       \
                                           w_scale, resid, s);                \
  }

// GeGLU silu-fold launch: fuses gate+up GEMM + silu(gate)*up + per-token
// block-128 FP8 quant into one launch (no [M,2N] BF16 transient). B is
// gate_up_w [2*N, K] (gate rows [0,N), up rows [N,2N)); w_scale is gate_up_s
// [2*N/128, K/128]. Outputs FP8 [M,N] + scale [M,N/128]. See device header.
template <int BM, int BN, int W, int STAGES, int MIN_BLK>
int launch_geglu_silu_fold_(const void* A, const void* B,
                            int M, int N, int K, const float* act_scale,
                            const float* w_scale, void* output, float* out_scale,
                            cudaStream_t s)
{
    constexpr int BK = 128;
    constexpr int SCALE_KTILE = 8;
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;        // over output N (== inter), NOT 2*N
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(W * 32, 1, 1);
    // A/B cp.async stages + gate_smem (BM*BN bf16) + scales + amax scratch.
    int smem_bytes = STAGES * (BM + BN) * BK
                   + (BM * BN) * (int)sizeof(__nv_bfloat16)
                   + (BM * SCALE_KTILE + 2 * SCALE_KTILE) * (int)sizeof(float)
                   + (W * BM + BM) * (int)sizeof(float);
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fp8_bs_geglu_silu_fold_kernel<BM, BN, W, STAGES, MIN_BLK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp8_bs_geglu_silu_fold_kernel<BM, BN, W, STAGES, MIN_BLK><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        act_scale, w_scale,
        reinterpret_cast<__nv_fp8_e4m3*>(output),
        out_scale, M, N, K);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
}

// A-persistent interleaved variant: stage A once, reuse ONE B smem region for
// gate then up within each k-iter (both gate+up acc live in regs, true
// interleaved per K-tile). Single B region -> 3 CTA/SM at s1 (vs interleaved's
// 2, vs two-pass's 3). See fp8_bs_geglu_silu_fold_apersist_kernel.
template <int BM, int BN, int W, int STAGES, int MIN_BLK>
int launch_geglu_silu_fold_apersist_(const void* A, const void* B,
                                     int M, int N, int K, const float* act_scale,
                                     const float* w_scale, void* output,
                                     float* out_scale, cudaStream_t s)
{
    constexpr int BK = 128;
    constexpr int SCALE_KTILE = 8;
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(W * 32, 1, 1);
    // Same smem layout as the two-pass variant (gate_smem region kept for layout
    // parity though apersist doesn't use it as a handoff — gate stays in regs).
    int smem_bytes = STAGES * (BM + BN) * BK
                   + (BM * BN) * (int)sizeof(__nv_bfloat16)
                   + (BM * SCALE_KTILE + 2 * SCALE_KTILE) * (int)sizeof(float)
                   + (W * BM + BM) * (int)sizeof(float);
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fp8_bs_geglu_silu_fold_apersist_kernel<BM, BN, W, STAGES, MIN_BLK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp8_bs_geglu_silu_fold_apersist_kernel<BM, BN, W, STAGES, MIN_BLK><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        act_scale, w_scale,
        reinterpret_cast<__nv_fp8_e4m3*>(output),
        out_scale, M, N, K);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
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

// Guard-free specialisation for shapes divisible by the tile (video stream:
// M=256, N and K multiples of 128). Same arithmetic, no per-chunk bounds
// checks and no epilogue guards.
#define DEFINE_EXACT(NAME, BM, BN, W, S, MB)                                  \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,        \
           const float* act_scale, const float* w_scale, cudaStream_t s) {    \
    return launch_<BM, BN, W, S, MB, true>(A, B, D, M, N, K, act_scale,       \
                                           w_scale, s);                       \
  }
DEFINE_EXACT(fp8_bs_exact_64x64x128_w4_s1, 64, 64, 4, 1, 4)
DEFINE_EXACT(fp8_bs_exact_64x64x128_w4_s2, 64, 64, 4, 2, 4)
#undef DEFINE_EXACT

// Residual-fold variants for the down-proj prefill tiles (see dispatcher
// below): the 2B/8B down-proj selects 64x64_s1 (8B) / 64x64 (2B small-M) /
// 32x64 (small-M) at the S ranges Phase-0 measured. Defined additively; the
// baseline kernels above are untouched.
DEFINE_RESID(fp8_block128_gemm_bs_sm89_32x64x128_w4_resid,    32,  64, 4, 2, 4)
DEFINE_RESID(fp8_block128_gemm_bs_sm89_64x64x128_w4_resid,    64,  64, 4, 2, 4)
DEFINE_RESID(fp8_block128_gemm_bs_sm89_64x64x128_w4_s1_resid, 64,  64, 4, 1, 4)
DEFINE_RESID(fp8_block128_gemm_bs_sm89_128x128x128_w8_s1_resid, 128, 128, 8, 1, 2)

#undef DEFINE_RESID

// GeGLU silu-fold tile variants (BLOCK_N pinned to 128 = one quant block).
#define DEFINE_GEGLU(NAME, BM, BN, W, S, MB)                                 \
  int NAME(const void* A, const void* B, int M, int N, int K,                 \
           const float* act_scale, const float* w_scale, void* output,        \
           float* out_scale, cudaStream_t s) {                                \
    return launch_geglu_silu_fold_<BM, BN, W, S, MB>(                         \
        A, B, M, N, K, act_scale, w_scale, output, out_scale, s);             \
  }
DEFINE_GEGLU(fp8_bs_geglu_silu_fold_sm89_32x128_w4_s2, 32, 128, 4, 2, 4)
DEFINE_GEGLU(fp8_bs_geglu_silu_fold_sm89_16x128_w4_s2, 16, 128, 4, 2, 4)
DEFINE_GEGLU(fp8_bs_geglu_silu_fold_sm89_64x128_w4_s2, 64, 128, 4, 2, 4)
DEFINE_GEGLU(fp8_bs_geglu_silu_fold_sm89_128x128_w8_s1, 128, 128, 8, 1, 2)
// Low-smem variants (STAGES=1) to recover occupancy lost to gate_smem on sm89:
// the s2 dual-buffer + gate_smem pushes dynamic smem >48KB → Block Limit Shared
// Mem = 1 (8% occupancy, ncu-confirmed). s1 trades cp.async overlap for 3-4x
// the CTA density. Primary candidates for the prefill M>=128 regime.
DEFINE_GEGLU(fp8_bs_geglu_silu_fold_sm89_32x128_w4_s1, 32, 128, 4, 1, 4)
DEFINE_GEGLU(fp8_bs_geglu_silu_fold_sm89_16x128_w4_s1, 16, 128, 4, 1, 4)
#undef DEFINE_GEGLU

// A-persistent interleaved variant (single B smem region, gate+up acc both in
// regs). launch wrapper shares the smem formula with the two-pass variant.
#define DEFINE_GEGLU_AP(NAME, BM, BN, W, S, MB)                              \
  int NAME(const void* A, const void* B, int M, int N, int K,                 \
           const float* act_scale, const float* w_scale, void* output,        \
           float* out_scale, cudaStream_t s) {                                \
    return launch_geglu_silu_fold_apersist_<BM, BN, W, S, MB>(                \
        A, B, M, N, K, act_scale, w_scale, output, out_scale, s);             \
  }
DEFINE_GEGLU_AP(fp8_bs_geglu_silu_fold_apersist_sm89_32x128_w4_s1, 32, 128, 4, 1, 2)
DEFINE_GEGLU_AP(fp8_bs_geglu_silu_fold_apersist_sm89_16x128_w4_s1, 16, 128, 4, 1, 2)
DEFINE_GEGLU_AP(fp8_bs_geglu_silu_fold_apersist_sm89_32x128_w4_s2, 32, 128, 4, 2, 2)
#undef DEFINE_GEGLU_AP

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
    // Narrow-M regime (M<=32). Measured on RTX 4060 Ti over the Wan2.2-TI2V-5B
    // production shapes (N=K=3072) with the shipped kernels, 300 iters, events:
    // 32x64_w4 0.0187 / 0.0192 ms and 16x64_w4 0.0130 / 0.0164 ms at M=16 / M=32
    // -- 1.44x and 1.17x in favour of the 16x64 tile. The heuristic above was
    // fitted on Qwen3-VL-8B shapes and never saw a narrow-M wide-N site; the
    // 32-row tile halves grid_m (24 blocks at M=16) and leaves the SMs idle.
    // Scoped to M<=32 so nothing upstream of the old M<128 branch moves.
    if (M <= 32)
        return fp8_block128_gemm_bs_sm89_16x64x128_w4(
            A, B, D, M, N, K, act_scale, w_scale, stream);
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
    // Small-M regime (M<256) for N<8192 linears (qkv/o/down): at M=128 the
    // 64x64/s1 grid underfills the SMs (8B qkv 64x64_s1 = 192 blocks = 1.5/SM;
    // 2B qkv = 128 blocks = 1/SM), so achieved occupancy is grid-limited well
    // below the theoretical cap. The smaller 32x64 tile doubles grid_m (8B qkv
    // -> 384 blocks = 3/SM; 2B qkv -> 256 = 2/SM) and wins despite a lower
    // per-block warp cap — ncu shows 8B qkv M=128: 32x64 51.6us vs 64x64_s1
    // 61.9us (-17%). Graph-captured e2e confirms: 2B S=128 -13.5%, 8B S=128
    // -12.0%, 2B S=192 -5.5%, 8B S=192 -2.0% (gain shrinks as M approaches the
    // 256 crossover, beyond which 64x64/s1 wins — see layer-regime micro-bench).
    // Wide-MLP gate_up keeps its existing s1 tile (8B via 128x128_w8_s1 above;
    // 2B via the default 64x64_s1 below) — it is best at all M.
    if (M < 256 && N < 8192)
        return fp8_block128_gemm_bs_sm89_32x64x128_w4(
            A, B, D, M, N, K, act_scale, w_scale, stream);
    if (N == 2048 && M < 1024)
        return fp8_block128_gemm_bs_sm89_64x64x128_w4(
            A, B, D, M, N, K, act_scale, w_scale, stream);
    // Video stream (M=256 and any M multiple of the 64-row tile): the guard-free
    // specialisation is bit-identical and 1.0-1.4% faster on all four production
    // shapes (M % 64 == 0 and N % 128 == 0 are checked by the caller above).
    if ((M % 64) == 0)
        return fp8_bs_exact_64x64x128_w4_s1(
            A, B, D, M, N, K, act_scale, w_scale, stream);
    return fp8_block128_gemm_bs_sm89_64x64x128_w4_s1(
        A, B, D, M, N, K, act_scale, w_scale, stream);
}

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
