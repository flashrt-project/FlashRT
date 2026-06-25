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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

namespace {

__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    // Ada (sm_89) FP8 tensor-core op — NO .kind::f8f6f4 qualifier.
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async_16(uint32_t smem, const uint8_t* src) {
    int b = (src == nullptr) ? 0 : 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem), "l"(src), "r"(b));
}

__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// BLOCK_K is pinned to 128 (one DeepSeek scale block per K-iteration).
//  - A: [M, K] row-major FP8 e4m3, act_scale [M, K/128] fp32
//  - B: [N, K] row-major FP8 e4m3, w_scale [N/128, K/128] fp32
//  - D: [M, N] row-major BF16
//  - BLOCK_N must keep each warp's 8-wide N-atoms inside one 128 scale block.
template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int STAGES,
          int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_bs_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    const float* __restrict__ act_scale,   // [M, K/128]
    const float* __restrict__ w_scale,     // [N/128, K/128]
    __nv_bfloat16* __restrict__ D,
    int M, int N, int K)
{
    constexpr int BLOCK_K    = 128;
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int M_ATOMS    = BLOCK_M / 16;
    constexpr int N_ATOMS    = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int K_ATOMS    = BLOCK_K / 32;        // = 4
    constexpr int SMEM_K_PAD = BLOCK_K + 16;        // bank-conflict padding

    static_assert(BLOCK_M % 16 == 0, "BLOCK_M multiple of 16");
    static_assert(BLOCK_N % 8 == 0,  "BLOCK_N multiple of 8");
    static_assert(BLOCK_N <= 128, "one CTA must fit one N scale block");
    static_assert((BLOCK_N / 8) % NUM_WARPS == 0, "N-atoms split across warps");

    extern __shared__ uint8_t smem_raw[];
    uint8_t* A_smem = smem_raw;
    uint8_t* B_smem = A_smem + STAGES * BLOCK_M * SMEM_K_PAD;

    const int cta_m = blockIdx.x;
    const int cta_n = blockIdx.y;
    const int m_base = cta_m * BLOCK_M;
    const int n_base = cta_n * BLOCK_N;

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;

    const int K128 = K >> 7;                        // # scale blocks along K

    auto issue_load = [&](int stage, int k_base) {
        constexpr int A_TOTAL_16B = BLOCK_M * BLOCK_K / 16;
        constexpr int A_ITERS = (A_TOTAL_16B + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= A_TOTAL_16B) break;
            int row_a = idx / (BLOCK_K / 16);
            int koff_a = (idx % (BLOCK_K / 16)) * 16;
            int m_glob = m_base + row_a;
            int k_glob = k_base + koff_a;
            const uint8_t* a_src = nullptr;
            if (m_glob < M && k_glob < K) {
                a_src = reinterpret_cast<const uint8_t*>(&A[m_glob * K + k_glob]);
            }
            cp_async_16(
                to_smem(&A_smem[stage * BLOCK_M * SMEM_K_PAD
                                + row_a * SMEM_K_PAD + koff_a]),
                a_src);
        }
        constexpr int B_TOTAL_16B = BLOCK_N * BLOCK_K / 16;
        constexpr int B_ITERS = (B_TOTAL_16B + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= B_TOTAL_16B) break;
            int row_b = idx / (BLOCK_K / 16);
            int koff_b = (idx % (BLOCK_K / 16)) * 16;
            int n_glob = n_base + row_b;
            int k_glob = k_base + koff_b;
            const uint8_t* b_src = nullptr;
            if (n_glob < N && k_glob < K) {
                b_src = reinterpret_cast<const uint8_t*>(&B[n_glob * K + k_glob]);
            }
            cp_async_16(
                to_smem(&B_smem[stage * BLOCK_N * SMEM_K_PAD
                                + row_b * SMEM_K_PAD + koff_b]),
                b_src);
        }
    };

    // Running (scaled) accumulators across all K-blocks.
    float acc[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[mi][ni][j] = 0.0f;

    const int K_ITERS = (K + BLOCK_K - 1) / BLOCK_K;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        int kb = s * BLOCK_K;
        if (kb < K) issue_load(s, kb);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    int compute_stage = 0;
    for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
        int issue_iter = k_iter + (STAGES - 1);
        int issue_stage = issue_iter % STAGES;
        if (issue_iter < K_ITERS) issue_load(issue_stage, issue_iter * BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
        __syncthreads();

        // This k_iter is exactly one scale block (kb = k_iter).
        const int kb = k_iter;
        // w_scale is constant across this CTA's BLOCK_N if it fits one
        // 128 block; index per warp's N base to stay correct for BLOCK_N>128.
        float tacc[M_ATOMS][N_ATOMS_PW][4];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi)
            #pragma unroll
            for (int ni = 0; ni < N_ATOMS_PW; ++ni)
                #pragma unroll
                for (int j = 0; j < 4; ++j) tacc[mi][ni][j] = 0.0f;

        #pragma unroll
        for (int ka = 0; ka < K_ATOMS; ++ka) {
            int kA0 = ka * 32 + 4 * l;
            int kA2 = ka * 32 + 4 * l + 16;
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int rA0 = mi * 16 + h;
                int rA1 = mi * 16 + h + 8;
                uint32_t A0 = *reinterpret_cast<const uint32_t*>(
                    &A_smem[compute_stage * BLOCK_M * SMEM_K_PAD + rA0 * SMEM_K_PAD + kA0]);
                uint32_t A1 = *reinterpret_cast<const uint32_t*>(
                    &A_smem[compute_stage * BLOCK_M * SMEM_K_PAD + rA1 * SMEM_K_PAD + kA0]);
                uint32_t A2 = *reinterpret_cast<const uint32_t*>(
                    &A_smem[compute_stage * BLOCK_M * SMEM_K_PAD + rA0 * SMEM_K_PAD + kA2]);
                uint32_t A3 = *reinterpret_cast<const uint32_t*>(
                    &A_smem[compute_stage * BLOCK_M * SMEM_K_PAD + rA1 * SMEM_K_PAD + kA2]);
                #pragma unroll
                for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
                    int co_n = warp_id * N_ATOMS_PW * 8 + ni * 8 + h;
                    uint32_t B0 = *reinterpret_cast<const uint32_t*>(
                        &B_smem[compute_stage * BLOCK_N * SMEM_K_PAD + co_n * SMEM_K_PAD + kA0]);
                    uint32_t B1 = *reinterpret_cast<const uint32_t*>(
                        &B_smem[compute_stage * BLOCK_N * SMEM_K_PAD + co_n * SMEM_K_PAD + kA2]);
                    mma_m16n8k32_e4m3(
                        tacc[mi][ni][0], tacc[mi][ni][1], tacc[mi][ni][2], tacc[mi][ni][3],
                        A0, A1, A2, A3, B0, B1);
                }
            }
        }

        // Fold block scales: D += act_scale[row,kb] * w_scale[ncol/128,kb] * tacc
        // All current SM89 prefill tiles use BLOCK_N <= 128, so a CTA stays
        // inside one 128-column weight-scale block even for 64-column half
        // tiles. Load that scale once per K block instead of once per
        // M/N atom.
        float ws_cta = w_scale[(size_t)(n_base >> 7) * K128 + kb];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi) {
            int row0 = m_base + mi * 16 + h;
            int row1 = row0 + 8;
            float as0 = (row0 < M) ? act_scale[row0 * K128 + kb] : 0.0f;
            float as1 = (row1 < M) ? act_scale[row1 * K128 + kb] : 0.0f;
            #pragma unroll
            for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
                acc[mi][ni][0] += tacc[mi][ni][0] * (as0 * ws_cta);
                acc[mi][ni][1] += tacc[mi][ni][1] * (as0 * ws_cta);
                acc[mi][ni][2] += tacc[mi][ni][2] * (as1 * ws_cta);
                acc[mi][ni][3] += tacc[mi][ni][3] * (as1 * ws_cta);
            }
        }
        compute_stage = (compute_stage + 1) % STAGES;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // Epilogue: write BF16. m16n8 layout: thread (h,l) -> rows {h,h+8},
    // cols {2*l, 2*l+1}.
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int row = (j < 2) ? row0 : row1;
                int col = n_pair_base + (j & 1);
                if (row < M && col < N) {
                    D[row * N + col] = __float2bfloat16(acc[mi][ni][j]);
                }
            }
        }
    }
}

template <int BM, int BN, int W, int STAGES, int MIN_BLK>
int launch_(const void* A, const void* B, void* D,
            int M, int N, int K, const float* act_scale,
            const float* w_scale, cudaStream_t s)
{
    constexpr int BK = 128;
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(W * 32, 1, 1);
    int smem_bytes = STAGES * (BM + BN) * (BK + 16);
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
    if (M >= 1024) {
        if (N >= 8192 && K == 4096)
            return fp8_block128_gemm_bs_sm89_128x128x128_w8(
                A, B, D, M, N, K, act_scale, w_scale, stream);
        if (N == 4096 && K >= 8192)
            return fp8_block128_gemm_bs_sm89_64x64x128_w4(
                A, B, D, M, N, K, act_scale, w_scale, stream);
    }
    if (M < 128)
        return fp8_block128_gemm_bs_sm89_32x64x128_w4(
            A, B, D, M, N, K, act_scale, w_scale, stream);
    return fp8_block128_gemm_bs_sm89_32x128x128_w4(
        A, B, D, M, N, K, act_scale, w_scale, stream);
}

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
