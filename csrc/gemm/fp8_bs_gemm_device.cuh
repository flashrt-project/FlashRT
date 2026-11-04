// SPDX-License-Identifier: Apache-2.0
#pragma once

// Shared device-side implementation of the SM89 FP8 block-128 scaled GEMM
// kernel. This header is the single source of truth for the kernel body: both
// the production launcher (fp8_block128_gemm_mma_sm89.cu) and the standalone
// micro-benchmark (benchmarks/sm89_fp8_block128_gemm) include it, so the
// bench's `--mode baseline` runs the *exact* production kernel and cannot
// drift behind it. When experimenting, copy this kernel into the bench's
// candidate slot and edit there; once an experiment is accepted and folded
// back here, the bench baseline tracks it automatically.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

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

// True when the adjacent output column pair {c, c+1} is fully in bounds, so a
// 32-bit bfloat162 store is valid. n_pair_base is even (=...+2*l) and N is a
// multiple of 128, so &D[row*N + c] is 4-byte aligned for the vector store.
__device__ __forceinline__ bool col_pair_ok(int c, int N) {
    return c + 1 < N;
}

// ldmatrix.x4: load four 8x8 b16 fragments from smem into 4 registers/lane in
// one instruction, replacing 4 scalar 32-bit LDS to offload the LSU pipe
// (NCU on the scalar path: LSU 67.7%, 54.7M shared loads = 27% of all insts).
__device__ __forceinline__ void ldmatrix_x4_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3, uint32_t smem_addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(smem_addr));
}

// SiLU in fp32. Matches quantize::silu_f32 (fp8_per_token_block_quant.cu:416)
// so the GeGLU silu-fold epilogue reproduces silu_mul_merged's math exactly.
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// BLOCK_K is pinned to 128 (one DeepSeek scale block per K-iteration).
//  - A: [M, K] row-major FP8 e4m3, act_scale [M, K/128] fp32
//  - B: [N, K] row-major FP8 e4m3, w_scale [N/128, K/128] fp32
//  - D: [M, N] row-major BF16
//  - BLOCK_N must keep each warp's 8-wide N-atoms inside one 128 scale block.
//
// RESID (opt-in epilogue fold): when true, the BF16 store adds a per-element
// residual `resid[M, N]` (same BF16 layout as D): D = bf16(acc + resid).
// This folds what would otherwise be a separate residual_add launch + an
// extra D round-trip through HBM, mirroring #134's residual-fold epilogue.
// When RESID=false, `resid` is unused and the `if constexpr (RESID)` branch
// is dead-stripped at compile time, so the baseline kernel is byte-identical.
template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int STAGES,
          int MIN_BLOCKS_PER_SM, bool RESID = false, bool EXACT = false>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_bs_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    const float* __restrict__ act_scale,   // [M, K/128]
    const float* __restrict__ w_scale,     // [N/128, K/128]
    __nv_bfloat16* __restrict__ D,
    int M, int N, int K,
    const __nv_bfloat16* __restrict__ resid = nullptr)  // [M, N] BF16, used iff RESID
{
    constexpr int BLOCK_K    = 128;
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int M_ATOMS    = BLOCK_M / 16;
    constexpr int N_ATOMS    = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int N_PAIRS_PW = N_ATOMS_PW / 2;      // ldmatrix pairs 2 N-atoms
    constexpr int K_ATOMS    = BLOCK_K / 32;        // = 4
    constexpr int NUM_CHUNKS_PER_ROW = BLOCK_K / 16;  // 8 chunks of 16 bytes
    // 128B swizzle: chunk_sw = chunk ^ (row & SWIZZLE_MASK). Removes the old
    // SMEM_K_PAD and the bank conflicts; applied identically on cp.async store
    // and ldmatrix load so the round-trip is bit-exact.
    constexpr int SWIZZLE_MASK = NUM_CHUNKS_PER_ROW - 1;  // = 7

    static_assert(BLOCK_M % 16 == 0, "BLOCK_M multiple of 16");
    static_assert(BLOCK_N % 8 == 0,  "BLOCK_N multiple of 8");
    static_assert(BLOCK_N <= 128, "one CTA must fit one N scale block");
    static_assert((BLOCK_N / 8) % NUM_WARPS == 0, "N-atoms split across warps");
    static_assert(N_ATOMS_PW >= 2 && N_ATOMS_PW % 2 == 0,
                  "ldmatrix pairs 2 N-atoms: N_ATOMS_PW must be even >= 2");

    // Stage the per-CTA activation/weight scales in shared memory with a
    // coalesced load, so the per-k_iter scale fold reads smem instead of
    // row-strided scalar global loads (NCU's top global-load bottleneck).
    // Only SCALE_KTILE scale-block columns are staged at a time, re-staged on
    // each k-tile boundary, so the smem footprint is K-independent (~2 KB) and
    // occupancy does not regress on large-K shapes (e.g. down, K128=96).
    constexpr int SCALE_KTILE = 8;
    constexpr int A_TILE = BLOCK_M * BLOCK_K;       // swizzled, no pad
    constexpr int B_TILE = BLOCK_N * BLOCK_K;

    extern __shared__ uint8_t smem_raw[];
    uint8_t* A_smem = smem_raw;
    uint8_t* B_smem = A_smem + STAGES * A_TILE;
    float* as_smem = reinterpret_cast<float*>(B_smem + STAGES * B_TILE);
    float* ws_smem = as_smem + BLOCK_M * SCALE_KTILE;

    const int cta_m = blockIdx.x;
    const int cta_n = blockIdx.y;
    const int m_base = cta_m * BLOCK_M;
    const int n_base = cta_n * BLOCK_N;

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;
    // ldmatrix.x4 lane -> fragment partition.
    const int frag_group = lane / 8;       // 0..3 (TL,TR,BL,BR)
    const int row_in_frag = lane % 8;      // row within an 8x8 fragment
    const int row_block = frag_group / 2;  // top(0)/bottom(1) 8 rows
    const int col_block = frag_group % 2;  // left(0)/right(1) 16-byte chunk

    const int K128 = K >> 7;                        // # scale blocks along K

    // Coalesced staging of one SCALE_KTILE-wide scale block into smem.
    auto stage_scales = [&](int kb0) {
        const int as_total = BLOCK_M * SCALE_KTILE;
        for (int idx = t; idx < as_total; idx += THREADS) {
            int r = idx / SCALE_KTILE;
            int kc = idx - r * SCALE_KTILE;
            int row = m_base + r;
            int kb = kb0 + kc;
            as_smem[idx] = (row < M && kb < K128)
                ? act_scale[(size_t)row * K128 + kb] : 0.0f;
        }
        for (int kc = t; kc < SCALE_KTILE; kc += THREADS) {
            int kb = kb0 + kc;
            ws_smem[kc] = (kb < K128)
                ? w_scale[(size_t)(n_base >> 7) * K128 + kb] : 0.0f;
        }
        __syncthreads();
    };

    auto issue_load = [&](int stage, int k_base) {
        constexpr int A_CHUNKS = BLOCK_M * NUM_CHUNKS_PER_ROW;
        constexpr int A_ITERS = (A_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= A_CHUNKS) break;
            int row_a = idx / NUM_CHUNKS_PER_ROW;
            int chunk_a = idx % NUM_CHUNKS_PER_ROW;
            int m_glob = m_base + row_a;
            int k_glob = k_base + chunk_a * 16;
            const uint8_t* a_src;
            if constexpr (EXACT) {
                a_src = reinterpret_cast<const uint8_t*>(&A[(size_t)m_glob * K + k_glob]);
            } else {
                a_src = nullptr;
                if (m_glob < M && k_glob < K) {
                    a_src = reinterpret_cast<const uint8_t*>(&A[(size_t)m_glob * K + k_glob]);
                }
            }
            int csw = chunk_a ^ (row_a & SWIZZLE_MASK);
            cp_async_16(
                to_smem(&A_smem[stage * A_TILE + row_a * BLOCK_K + csw * 16]),
                a_src);
        }
        constexpr int B_CHUNKS = BLOCK_N * NUM_CHUNKS_PER_ROW;
        constexpr int B_ITERS = (B_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= B_CHUNKS) break;
            int row_b = idx / NUM_CHUNKS_PER_ROW;
            int chunk_b = idx % NUM_CHUNKS_PER_ROW;
            int n_glob = n_base + row_b;
            int k_glob = k_base + chunk_b * 16;
            const uint8_t* b_src;
            if constexpr (EXACT) {
                b_src = reinterpret_cast<const uint8_t*>(&B[(size_t)n_glob * K + k_glob]);
            } else {
                b_src = nullptr;
                if (n_glob < N && k_glob < K) {
                    b_src = reinterpret_cast<const uint8_t*>(&B[(size_t)n_glob * K + k_glob]);
                }
            }
            int csw = chunk_b ^ (row_b & SWIZZLE_MASK);
            cp_async_16(
                to_smem(&B_smem[stage * B_TILE + row_b * BLOCK_K + csw * 16]),
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
        // Re-stage the next SCALE_KTILE-wide scale block on each tile boundary.
        if ((kb % SCALE_KTILE) == 0) stage_scales(kb);
        // w_scale is constant across this CTA's BLOCK_N if it fits one
        // 128 block; index per warp's N base to stay correct for BLOCK_N>128.
        float tacc[M_ATOMS][N_ATOMS_PW][4];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi)
            #pragma unroll
            for (int ni = 0; ni < N_ATOMS_PW; ++ni)
                #pragma unroll
                for (int j = 0; j < 4; ++j) tacc[mi][ni][j] = 0.0f;

        uint8_t* A_stage = A_smem + compute_stage * A_TILE;
        uint8_t* B_stage = B_smem + compute_stage * B_TILE;
        #pragma unroll
        for (int ka = 0; ka < K_ATOMS; ++ka) {
            // ldmatrix.x4 loads the m16xk32 A fragment (4 regs/lane) per m-atom.
            uint32_t A_regs[M_ATOMS][4];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int row = mi * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * ka + col_block;
                int csw = chunk ^ (row & SWIZZLE_MASK);
                ldmatrix_x4_b16(A_regs[mi][0], A_regs[mi][1], A_regs[mi][2], A_regs[mi][3],
                                to_smem(&A_stage[row * BLOCK_K + csw * 16]));
            }
            // ldmatrix.x4 loads two N-atoms (n16xk32) per pair.
            uint32_t B_regs[N_PAIRS_PW][4];
            #pragma unroll
            for (int np = 0; np < N_PAIRS_PW; ++np) {
                int nrow = warp_id * N_ATOMS_PW * 8 + np * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * ka + col_block;
                int csw = chunk ^ (nrow & SWIZZLE_MASK);
                ldmatrix_x4_b16(B_regs[np][0], B_regs[np][1], B_regs[np][2], B_regs[np][3],
                                to_smem(&B_stage[nrow * BLOCK_K + csw * 16]));
            }
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                #pragma unroll
                for (int np = 0; np < N_PAIRS_PW; ++np) {
                    int ni0 = np * 2, ni1 = np * 2 + 1;
                    // ldm fragment -> mma A operand: a0=d0,a1=d2,a2=d1,a3=d3.
                    mma_m16n8k32_e4m3(
                        tacc[mi][ni0][0], tacc[mi][ni0][1], tacc[mi][ni0][2], tacc[mi][ni0][3],
                        A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                        B_regs[np][0], B_regs[np][1]);
                    mma_m16n8k32_e4m3(
                        tacc[mi][ni1][0], tacc[mi][ni1][1], tacc[mi][ni1][2], tacc[mi][ni1][3],
                        A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                        B_regs[np][2], B_regs[np][3]);
                }
            }
        }

        // Fold block scales: D += act_scale[row,kb] * w_scale[ncol/128,kb] * tacc
        // Scales come from the smem stage (coalesced load above), indexed by
        // the column within the current SCALE_KTILE tile. BLOCK_N <= 128 keeps
        // the CTA inside one 128-column weight-scale block.
        int kbt = kb % SCALE_KTILE;
        float ws_cta = ws_smem[kbt];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi) {
            int row0 = m_base + mi * 16 + h;
            int row1 = row0 + 8;
            float as0 = as_smem[(mi * 16 + h) * SCALE_KTILE + kbt];
            float as1 = as_smem[(mi * 16 + h + 8) * SCALE_KTILE + kbt];
            #pragma unroll
            for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
                acc[mi][ni][0] += tacc[mi][ni][0] * (as0 * ws_cta);
                acc[mi][ni][1] += tacc[mi][ni][1] * (as0 * ws_cta);
                acc[mi][ni][2] += tacc[mi][ni][2] * (as1 * ws_cta);
                acc[mi][ni][3] += tacc[mi][ni][3] * (as1 * ws_cta);
            }
        }
        // Do not let the next cp.async overwrite this shared-memory stage
        // before all warps finish reading it.
        __syncthreads();
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
            // acc[0,1] = row0 cols {2l,2l+1}; acc[2,3] = row1 cols {2l,2l+1}.
            // RESID epilogue fold: add the BF16 residual in-register before the
            // bf16 store, so the residual read is fused into the GEMM epilogue
            // and never lands as a separate D HBM round-trip + launch.
            if constexpr (RESID) {
                if (row0 < M && col_pair_ok(n_pair_base, N)) {
                    __nv_bfloat162 r = *reinterpret_cast<const __nv_bfloat162*>(
                        &resid[(size_t)row0 * N + n_pair_base]);
                    *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)row0 * N + n_pair_base]) =
                        __floats2bfloat162_rn(acc[mi][ni][0] + __low2float(r),
                                               acc[mi][ni][1] + __high2float(r));
                } else if (row0 < M) {
                    if (n_pair_base < N)
                        D[(size_t)row0 * N + n_pair_base] = __float2bfloat16(
                            acc[mi][ni][0] + __bfloat162float(resid[(size_t)row0 * N + n_pair_base]));
                    if (n_pair_base + 1 < N)
                        D[(size_t)row0 * N + n_pair_base + 1] = __float2bfloat16(
                            acc[mi][ni][1] + __bfloat162float(resid[(size_t)row0 * N + n_pair_base + 1]));
                }
                if (row1 < M && col_pair_ok(n_pair_base, N)) {
                    __nv_bfloat162 r = *reinterpret_cast<const __nv_bfloat162*>(
                        &resid[(size_t)row1 * N + n_pair_base]);
                    *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)row1 * N + n_pair_base]) =
                        __floats2bfloat162_rn(acc[mi][ni][2] + __low2float(r),
                                               acc[mi][ni][3] + __high2float(r));
                } else if (row1 < M) {
                    if (n_pair_base < N)
                        D[(size_t)row1 * N + n_pair_base] = __float2bfloat16(
                            acc[mi][ni][2] + __bfloat162float(resid[(size_t)row1 * N + n_pair_base]));
                    if (n_pair_base + 1 < N)
                        D[(size_t)row1 * N + n_pair_base + 1] = __float2bfloat16(
                            acc[mi][ni][3] + __bfloat162float(resid[(size_t)row1 * N + n_pair_base + 1]));
                }
            } else if constexpr (EXACT) {
                static_assert(!RESID, "EXACT is only defined for the plain epilogue");
                *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)row0 * N + n_pair_base]) =
                    __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
                *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)row1 * N + n_pair_base]) =
                    __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
            } else {
                // Emit one 32-bit bfloat162 store per row instead of two scalar
                // 16-bit stores (NCU's top store-pattern bottleneck after C1).
                // Tail (odd last column) falls back to scalar stores.
                if (row0 < M && col_pair_ok(n_pair_base, N)) {
                    *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)row0 * N + n_pair_base]) =
                        __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
                } else if (row0 < M) {
                    if (n_pair_base < N)     D[(size_t)row0 * N + n_pair_base]   = __float2bfloat16(acc[mi][ni][0]);
                    if (n_pair_base + 1 < N) D[(size_t)row0 * N + n_pair_base+1] = __float2bfloat16(acc[mi][ni][1]);
                }
                if (row1 < M && col_pair_ok(n_pair_base, N)) {
                    *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)row1 * N + n_pair_base]) =
                        __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
                } else if (row1 < M) {
                    if (n_pair_base < N)     D[(size_t)row1 * N + n_pair_base]   = __float2bfloat16(acc[mi][ni][2]);
                    if (n_pair_base + 1 < N) D[(size_t)row1 * N + n_pair_base+1] = __float2bfloat16(acc[mi][ni][3]);
                }
            }
        }
    }
}

// ============================================================================
// GeGLU silu-fold megakernel (Phase 2): fuses gate GEMM + up GEMM +
// silu(gate)*up + per-token block-128 FP8 quant into ONE launch, writing FP8
// output + scale directly — eliminating the [M, 2*N] BF16 transient that the
// baseline gate_up GEMM would write and silu_mul_merged_to_fp8 would read back.
//
//   gate_up_w : [2*N, K] FP8 row-major  (gate rows [0,N); up rows [N,2N))
//   gate_up_s : [2*N/128, K/128] fp32   (up row = gate row + N/128)
//   A         : [M, K] FP8 (per-token quantized),  act_scale [M, K/128]
//   output    : [M, N] FP8,  scale [M, N/128]
//
// Two-pass per CTA (mirrors sm100 flashrt_megakernel_geglu's "gate stays in
// smem"): pass 1 accumulates gate over full K and stores silu(gate) as BF16
// into a smem gate buffer; pass 2 reuses the same A/B smem staging, accumulates
// up, then the epilogue reads gate from smem, forms v = bf16(bf16(silu(gate))*up)
// (matching silu_mul_merged's two bf16 roundings), reduces |v| over the 128-col
// quant block per row, and quantizes to FP8. No grid_barrier (single CTA owns
// its full quant block: BLOCK_N == 128 == one scale block). GEMM body reuses
// the same cp.async + ldmatrix.x4 + mma.m16n8k32 tiles as fp8_bs_gemm_kernel.
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int STAGES,
          int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_bs_geglu_silu_fold_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,        // gate_up_w [2*N, K]
    const float* __restrict__ act_scale,        // [M, K/128]
    const float* __restrict__ w_scale,          // gate_up_s [2*N/128, K/128]
    __nv_fp8_e4m3* __restrict__ output,         // [M, N]
    float* __restrict__ out_scale,              // [M, N/128]
    int M, int N, int K)
{
    static_assert(BLOCK_N == 128,
        "GeGLU silu-fold requires BLOCK_N==128 (one quant block per CTA)");
    constexpr int BLOCK_K    = 128;
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int M_ATOMS    = BLOCK_M / 16;
    constexpr int N_ATOMS    = BLOCK_N / 8;       // 16
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int N_PAIRS_PW = N_ATOMS_PW / 2;
    constexpr int K_ATOMS    = BLOCK_K / 32;      // 4
    constexpr int NUM_CHUNKS_PER_ROW = BLOCK_K / 16;
    constexpr int SWIZZLE_MASK = NUM_CHUNKS_PER_ROW - 1;
    constexpr int SCALE_KTILE = 8;
    constexpr int A_TILE = BLOCK_M * BLOCK_K;
    constexpr int B_TILE = BLOCK_N * BLOCK_K;

    static_assert(BLOCK_M % 16 == 0, "BLOCK_M multiple of 16");
    static_assert(N_ATOMS_PW >= 2 && N_ATOMS_PW % 2 == 0,
                  "ldmatrix pairs 2 N-atoms: N_ATOMS_PW must be even >= 2");

    extern __shared__ uint8_t smem_raw[];
    uint8_t* A_smem = smem_raw;
    uint8_t* B_smem = A_smem + STAGES * A_TILE;
    // gate_smem: silu(gate) as BF16, [BLOCK_M, BLOCK_N]. One CTA-tile, written
    // by pass 1 epilogue, read by pass 2 epilogue. The sm100 geglu's "gate
    // stays in smem" handoff, without tcgen05/EVT.
    __nv_bfloat16* gate_smem = reinterpret_cast<__nv_bfloat16*>(
        B_smem + STAGES * B_TILE);
    float* as_smem = reinterpret_cast<float*>(gate_smem + BLOCK_M * BLOCK_N);
    float* wsg_smem = as_smem + BLOCK_M * SCALE_KTILE;   // gate w_scale row
    float* wsu_smem = wsg_smem + SCALE_KTILE;            // up   w_scale row
    // amax partials: 4 warps × BLOCK_M rows. Cross-warp reduce per row.
    float* amax_smem = wsu_smem + SCALE_KTILE;

    const int cta_m = blockIdx.x;
    const int cta_n = blockIdx.y;
    const int m_base = cta_m * BLOCK_M;
    const int n_base = cta_n * BLOCK_N;          // n0, < N (output col block)

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;
    const int frag_group = lane / 8;
    const int row_in_frag = lane % 8;
    const int row_block = frag_group / 2;
    const int col_block = frag_group % 2;

    const int K128 = K >> 7;
    const int N128 = N >> 7;                       // gate w_scale blocks
    // gate B-rows [n_base, n_base+BLOCK_N); up B-rows [n_base+N, n_base+N+BLOCK_N)
    const int gate_b_row0 = n_base;
    const int up_b_row0   = n_base + N;
    const int gate_ws_row = (n_base >> 7);          // gate w_scale block row
    const int up_ws_row   = gate_ws_row + N128;     // up w_scale block row

    // ---- scale staging (shared by both passes; re-staged per SCALE_KTILE) ----
    auto stage_scales = [&](int kb0) {
        const int as_total = BLOCK_M * SCALE_KTILE;
        for (int idx = t; idx < as_total; idx += THREADS) {
            int r = idx / SCALE_KTILE;
            int kc = idx - r * SCALE_KTILE;
            int row = m_base + r;
            int kb = kb0 + kc;
            as_smem[idx] = (row < M && kb < K128)
                ? act_scale[(size_t)row * K128 + kb] : 0.0f;
        }
        for (int kc = t; kc < SCALE_KTILE; kc += THREADS) {
            int kb = kb0 + kc;
            wsg_smem[kc] = (kb < K128)
                ? w_scale[(size_t)gate_ws_row * K128 + kb] : 0.0f;
            wsu_smem[kc] = (kb < K128)
                ? w_scale[(size_t)up_ws_row * K128 + kb] : 0.0f;
        }
        __syncthreads();
    };

    // ---- cp.async A + (gate or up) B tile staging ----
    // b_row0 selects which 128-row band of B [2*N, K] to stage.
    auto issue_load = [&](int stage, int k_base, int b_row0) {
        constexpr int A_CHUNKS = BLOCK_M * NUM_CHUNKS_PER_ROW;
        constexpr int A_ITERS = (A_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= A_CHUNKS) break;
            int row_a = idx / NUM_CHUNKS_PER_ROW;
            int chunk_a = idx % NUM_CHUNKS_PER_ROW;
            int m_glob = m_base + row_a;
            int k_glob = k_base + chunk_a * 16;
            const uint8_t* a_src = nullptr;
            if (m_glob < M && k_glob < K) {
                a_src = reinterpret_cast<const uint8_t*>(&A[(size_t)m_glob * K + k_glob]);
            }
            int csw = chunk_a ^ (row_a & SWIZZLE_MASK);
            cp_async_16(
                to_smem(&A_smem[stage * A_TILE + row_a * BLOCK_K + csw * 16]),
                a_src);
        }
        constexpr int B_CHUNKS = BLOCK_N * NUM_CHUNKS_PER_ROW;
        constexpr int B_ITERS = (B_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= B_CHUNKS) break;
            int row_b = idx / NUM_CHUNKS_PER_ROW;
            int chunk_b = idx % NUM_CHUNKS_PER_ROW;
            int n_glob = b_row0 + row_b;
            int k_glob = k_base + chunk_b * 16;
            const uint8_t* b_src = nullptr;
            if (n_glob < 2 * N && k_glob < K) {
                b_src = reinterpret_cast<const uint8_t*>(&B[(size_t)n_glob * K + k_glob]);
            }
            int csw = chunk_b ^ (row_b & SWIZZLE_MASK);
            cp_async_16(
                to_smem(&B_smem[stage * B_TILE + row_b * BLOCK_K + csw * 16]),
                b_src);
        }
    };

    // ---- one GEMM pass over full K, accumulating into `acc` with the given
    // w_scale smem row (gate or up). b_row0 selects the B band. ----
    auto run_pass = [&](float (*acc)[N_ATOMS_PW][4], int b_row0,
                        const float* ws_smem_pass) {
        const int K_ITERS = (K + BLOCK_K - 1) / BLOCK_K;
        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            int kb = s * BLOCK_K;
            if (kb < K) issue_load(s, kb, b_row0);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        int compute_stage = 0;
        for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
            int issue_iter = k_iter + (STAGES - 1);
            int issue_stage = issue_iter % STAGES;
            if (issue_iter < K_ITERS) issue_load(issue_stage, issue_iter * BLOCK_K, b_row0);
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
            __syncthreads();

            const int kb = k_iter;
            if ((kb % SCALE_KTILE) == 0) stage_scales(kb);

            float tacc[M_ATOMS][N_ATOMS_PW][4];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi)
                #pragma unroll
                for (int ni = 0; ni < N_ATOMS_PW; ++ni)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) tacc[mi][ni][j] = 0.0f;

            uint8_t* A_stage = A_smem + compute_stage * A_TILE;
            uint8_t* B_stage = B_smem + compute_stage * B_TILE;
            #pragma unroll
            for (int ka = 0; ka < K_ATOMS; ++ka) {
                uint32_t A_regs[M_ATOMS][4];
                #pragma unroll
                for (int mi = 0; mi < M_ATOMS; ++mi) {
                    int row = mi * 16 + row_block * 8 + row_in_frag;
                    int chunk = 2 * ka + col_block;
                    int csw = chunk ^ (row & SWIZZLE_MASK);
                    ldmatrix_x4_b16(A_regs[mi][0], A_regs[mi][1], A_regs[mi][2], A_regs[mi][3],
                                    to_smem(&A_stage[row * BLOCK_K + csw * 16]));
                }
                uint32_t B_regs[N_PAIRS_PW][4];
                #pragma unroll
                for (int np = 0; np < N_PAIRS_PW; ++np) {
                    int nrow = warp_id * N_ATOMS_PW * 8 + np * 16 + row_block * 8 + row_in_frag;
                    int chunk = 2 * ka + col_block;
                    int csw = chunk ^ (nrow & SWIZZLE_MASK);
                    ldmatrix_x4_b16(B_regs[np][0], B_regs[np][1], B_regs[np][2], B_regs[np][3],
                                    to_smem(&B_stage[nrow * BLOCK_K + csw * 16]));
                }
                #pragma unroll
                for (int mi = 0; mi < M_ATOMS; ++mi) {
                    #pragma unroll
                    for (int np = 0; np < N_PAIRS_PW; ++np) {
                        int ni0 = np * 2, ni1 = np * 2 + 1;
                        mma_m16n8k32_e4m3(
                            tacc[mi][ni0][0], tacc[mi][ni0][1], tacc[mi][ni0][2], tacc[mi][ni0][3],
                            A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                            B_regs[np][0], B_regs[np][1]);
                        mma_m16n8k32_e4m3(
                            tacc[mi][ni1][0], tacc[mi][ni1][1], tacc[mi][ni1][2], tacc[mi][ni1][3],
                            A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                            B_regs[np][2], B_regs[np][3]);
                    }
                }
            }

            int kbt = kb % SCALE_KTILE;
            float ws_cta = ws_smem_pass[kbt];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int row0 = m_base + mi * 16 + h;
                int row1 = row0 + 8;
                float as0 = as_smem[(mi * 16 + h) * SCALE_KTILE + kbt];
                float as1 = as_smem[(mi * 16 + h + 8) * SCALE_KTILE + kbt];
                #pragma unroll
                for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
                    acc[mi][ni][0] += tacc[mi][ni][0] * (as0 * ws_cta);
                    acc[mi][ni][1] += tacc[mi][ni][1] * (as0 * ws_cta);
                    acc[mi][ni][2] += tacc[mi][ni][2] * (as1 * ws_cta);
                    acc[mi][ni][3] += tacc[mi][ni][3] * (as1 * ws_cta);
                }
            }
            __syncthreads();
            compute_stage = (compute_stage + 1) % STAGES;
        }
        asm volatile("cp.async.wait_all;\n" ::);
    };

    // =================== Pass 1: gate GEMM → silu(gate) in smem ===================
    float gate_acc[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) gate_acc[mi][ni][j] = 0.0f;

    run_pass(gate_acc, gate_b_row0, wsg_smem);

    // Pass 1 epilogue: store silu(gate_acc) as BF16 into gate_smem[BM, BN].
    // Thread (h,l) owns rows {mi*16+h, mi*16+h+8}, cols {ni*8+2l, ni*8+2l+1}
    // within its warp's N band. Replicate silu_mul_merged's first bf16 rounding
    // (bf16(silu(g))) so the fused path matches the split kernel's precision.
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int n_pair_base = warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            // gate_smem is [BLOCK_M, BLOCK_N]; local col = n_pair_base.
            if (row0 < M) {
                __nv_bfloat162 gs = __floats2bfloat162_rn(
                    silu_f32(gate_acc[mi][ni][0]), silu_f32(gate_acc[mi][ni][1]));
                *reinterpret_cast<__nv_bfloat162*>(
                    &gate_smem[(row0 - m_base) * BLOCK_N + n_pair_base]) = gs;
                __nv_bfloat162 gs2 = __floats2bfloat162_rn(
                    silu_f32(gate_acc[mi][ni][2]), silu_f32(gate_acc[mi][ni][3]));
                *reinterpret_cast<__nv_bfloat162*>(
                    &gate_smem[(row1 - m_base) * BLOCK_N + n_pair_base]) = gs2;
            }
        }
    }
    __syncthreads();   // gate_smem visible to pass 2 epilogue in all warps
    // gate_acc registers now free; reused for up_acc.

    // =================== Pass 2: up GEMM → up_acc ===================
    float up_acc[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) up_acc[mi][ni][j] = 0.0f;

    run_pass(up_acc, up_b_row0, wsu_smem);

    // =================== Pass 2 epilogue: silu(gate)*up + quant → FP8 ===================
    // v = bf16(bf16(silu(gate)) * up), matching silu_mul_merged's two bf16
    // roundings (silu(gate) was already bf16-rounded into gate_smem in pass 1;
    // here we bf16-round the product). Then per-row amax over the 128-col block
    // and quantize.
    constexpr float kFp8Max = 448.0f;
    // Each thread owns 8 cols (4 n-atoms × 2) for 2 rows per m-atom. Compute |v|
    // and a per-warp partial amax per row (the warp owns 32 of the row's 128 cols).
    // amax_smem[warp_id][row_in_cta] holds the warp's row-amax partial.
    float v[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        int rloc0 = mi * 16 + h;          // local row in [0, BLOCK_M)
        int rloc1 = rloc0 + 8;
        float amax0 = 0.0f, amax1 = 0.0f;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int n_pair_base = warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            // gate value (already bf16(silu(gate))) from smem; up from registers.
            if (row0 < M) {
                __nv_bfloat162 g = *reinterpret_cast<const __nv_bfloat162*>(
                    &gate_smem[rloc0 * BLOCK_N + n_pair_base]);
                float gf0 = __low2float(g), gf1 = __high2float(g);
                v[mi][ni][0] = __bfloat162float(__float2bfloat16(gf0 * up_acc[mi][ni][0]));
                v[mi][ni][1] = __bfloat162float(__float2bfloat16(gf1 * up_acc[mi][ni][1]));
                amax0 = fmaxf(amax0, fmaxf(fabsf(v[mi][ni][0]), fabsf(v[mi][ni][1])));
            } else {
                v[mi][ni][0] = 0.0f; v[mi][ni][1] = 0.0f;
            }
            if (row1 < M) {
                __nv_bfloat162 g = *reinterpret_cast<const __nv_bfloat162*>(
                    &gate_smem[rloc1 * BLOCK_N + n_pair_base]);
                float gf0 = __low2float(g), gf1 = __high2float(g);
                v[mi][ni][2] = __bfloat162float(__float2bfloat16(gf0 * up_acc[mi][ni][2]));
                v[mi][ni][3] = __bfloat162float(__float2bfloat16(gf1 * up_acc[mi][ni][3]));
                amax1 = fmaxf(amax1, fmaxf(fabsf(v[mi][ni][2]), fabsf(v[mi][ni][3])));
            } else {
                v[mi][ni][2] = 0.0f; v[mi][ni][3] = 0.0f;
            }
        }
        // Warp-shuffle reduce the 4 lanes (l=0..3) that share row0 / row1.
        for (int off = 2; off > 0; off >>= 1) {
            amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffff, amax0, off));
            amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffff, amax1, off));
        }
        if (l == 0) {
            amax_smem[warp_id * BLOCK_M + rloc0] = amax0;
            amax_smem[warp_id * BLOCK_M + rloc1] = amax1;
        }
    }
    __syncthreads();

    // Cross-warp reduce: each warp wrote its row-amax partial. Final reduce per
    // row done by warp 0 lanes, broadcast via smem.
    #pragma unroll
    for (int rloc = t; rloc < BLOCK_M; rloc += THREADS) {
        int row = m_base + rloc;
        if (row >= M) continue;
        float amax = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
            amax = fmaxf(amax, amax_smem[w * BLOCK_M + rloc]);
        float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
        amax_smem[rloc] = sc;     // reuse slot to broadcast final scale
        // Each active thread owns a distinct rloc in this strided loop, so each
        // writes its own row's scale — no race. (The earlier `warp_id==0 &&
        // lane==0` guard let only thread 0 write, leaving rows 1..BLOCK_M-1
        // unwritten → garbage out_scale, correct-but-unscaled fp8 output.)
        out_scale[(size_t)row * (N >> 7) + (n_base >> 7)] = sc;
    }
    __syncthreads();

    // Quantize + store FP8. Thread re-reads its v[] and the row's scale.
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        int rloc0 = mi * 16 + h;
        int rloc1 = rloc0 + 8;
        float sc0 = (row0 < M) ? amax_smem[rloc0] : 1.0f;
        float sc1 = (row1 < M) ? amax_smem[rloc1] : 1.0f;
        float inv0 = 1.0f / sc0, inv1 = 1.0f / sc1;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            if (row0 < M && col_pair_ok(n_pair_base, N)) {
                float q0 = fminf(fmaxf(v[mi][ni][0] * inv0, -kFp8Max), kFp8Max);
                float q1 = fminf(fmaxf(v[mi][ni][1] * inv0, -kFp8Max), kFp8Max);
                // pack two fp8 e4m3 into a 16-bit store
                __nv_fp8_e4m3 p0(q0), p1(q1);
                uint16_t pack = (uint16_t)(*reinterpret_cast<const uint8_t*>(&p1)) << 8
                              | (uint16_t)(*reinterpret_cast<const uint8_t*>(&p0));
                *reinterpret_cast<uint16_t*>(&output[(size_t)row0 * N + n_pair_base]) = pack;
            } else if (row0 < M) {
                if (n_pair_base < N) {
                    float q = fminf(fmaxf(v[mi][ni][0] * inv0, -kFp8Max), kFp8Max);
                    output[(size_t)row0 * N + n_pair_base] = __nv_fp8_e4m3(q);
                }
                if (n_pair_base + 1 < N) {
                    float q = fminf(fmaxf(v[mi][ni][1] * inv0, -kFp8Max), kFp8Max);
                    output[(size_t)row0 * N + n_pair_base + 1] = __nv_fp8_e4m3(q);
                }
            }
            if (row1 < M && col_pair_ok(n_pair_base, N)) {
                float q2 = fminf(fmaxf(v[mi][ni][2] * inv1, -kFp8Max), kFp8Max);
                float q3 = fminf(fmaxf(v[mi][ni][3] * inv1, -kFp8Max), kFp8Max);
                __nv_fp8_e4m3 p2(q2), p3(q3);
                uint16_t pack = (uint16_t)(*reinterpret_cast<const uint8_t*>(&p3)) << 8
                              | (uint16_t)(*reinterpret_cast<const uint8_t*>(&p2));
                *reinterpret_cast<uint16_t*>(&output[(size_t)row1 * N + n_pair_base]) = pack;
            } else if (row1 < M) {
                if (n_pair_base < N) {
                    float q = fminf(fmaxf(v[mi][ni][2] * inv1, -kFp8Max), kFp8Max);
                    output[(size_t)row1 * N + n_pair_base] = __nv_fp8_e4m3(q);
                }
                if (n_pair_base + 1 < N) {
                    float q = fminf(fmaxf(v[mi][ni][3] * inv1, -kFp8Max), kFp8Max);
                    output[(size_t)row1 * N + n_pair_base + 1] = __nv_fp8_e4m3(q);
                }
            }
        }
    }
}

// ============================================================================
// GeGLU silu-fold, A-persistent two-pass variant.
//
// Same fusion as fp8_bs_geglu_silu_fold_kernel (gate+up GEMM + silu(gate)*up +
// per-token block-128 FP8 quant, one launch, no [M,2N] BF16 transient), but a
// different smem/register strategy that fixes the two-pass weaknesses the ncu
// diagnosis isolated:
//
//   two-pass loss = (a) A re-loaded twice (2*M*K HBM) + (b) 2x pipeline drain.
//   interleaved loss = 2x B smem (both gate+up staged) -> 1 CTA/SM occupancy.
//
// A-persistent: stage A into smem ONCE (reused by both the gate pass and the up
// pass), but keep only ONE B smem region that is filled with B_gate for the
// gate pass and then RE-FILLED with B_up for the up pass (sequential, not
// simultaneous). So:
//   - A loaded once from HBM (the interleaved HBM win), held in smem across
//     both passes -> no A re-load. act_scale also staged once.
//   - B smem = a single STAGES*BN*BK region (NOT doubled) -> fits 4 CTA/SM.
//   - only ONE accumulator live at a time (gate_acc -> store silu(gate) to a
//     small smem gate buffer -> reuse regs for up_acc) -> ~70 regs (two-pass
//     register profile), not the ~140 of true interleaved.
//
// The catch: A must fit in smem for the whole K-walk (A is [BM, K], not
// [BM, BK]), so this only works when K is small enough that BM*K fp8 + the rest
// stays under the smem budget — i.e. the Qwen3-VL gate_up shapes where K=hidden
// (2B K=2048, 8B K=4096). For BM=32: A_persist = 32*K = 64KB (8B) / 32KB (2B).
// 8B 64KB alone already exceeds a CTA's smem, so A-persistent is only viable
// for the 2B shape (K=2048) at BM<=32, OR by staging A in K-chunks and walking
// gate+up together within each K-chunk (chunked-interleaved). The chunked form
// is implemented here: A is staged per BLOCK_K tile like the baseline, but BOTH
// the gate MMA and the up MMA for that K-tile run before the tile is evicted —
// i.e. gate and up advance K-tile-by-K-tile together (true interleaved per
// K-tile, the sm100 geglu pattern), yet B is staged in ONE region reused for
// gate-then-up WITHIN the k-iter (load B_gate, gate-MMA, load B_up into the
// SAME region, up-MMA). That keeps B smem single (no 2x) AND loads A once AND
// holds both gate_acc+up_acc in regs (interleaved) — but pays by serializing
// the two B loads within a k-iter (no overlap between B_gate and B_up loads).
//
// Net vs two-pass: A loaded once (saves M*K HBM), one pipeline drain (K_ITERS
// stalls not 2*K_ITERS), but B_gate/B_up loads are serial within each k-iter.
// Net vs interleaved: B smem halved (2 CTA/SM recoverable to 3-4), but loses
// B_gate||B_up load overlap. On sm89 (HBM-bound, no TMA) the smem/occupancy
// recovery usually dominates, so this is the predicted winner.
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int STAGES,
          int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_bs_geglu_silu_fold_apersist_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,        // gate_up_w [2*N, K]
    const float* __restrict__ act_scale,        // [M, K/128]
    const float* __restrict__ w_scale,          // gate_up_s [2*N/128, K/128]
    __nv_fp8_e4m3* __restrict__ output,         // [M, N]
    float* __restrict__ out_scale,              // [M, N/128]
    int M, int N, int K)
{
    static_assert(BLOCK_N == 128,
        "GeGLU silu-fold requires BLOCK_N==128 (one quant block per CTA)");
    constexpr int BLOCK_K    = 128;
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int M_ATOMS    = BLOCK_M / 16;
    constexpr int N_ATOMS    = BLOCK_N / 8;       // 16
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int N_PAIRS_PW = N_ATOMS_PW / 2;
    constexpr int K_ATOMS    = BLOCK_K / 32;      // 4
    constexpr int NUM_CHUNKS_PER_ROW = BLOCK_K / 16;
    constexpr int SWIZZLE_MASK = NUM_CHUNKS_PER_ROW - 1;
    constexpr int SCALE_KTILE = 8;
    constexpr int A_TILE = BLOCK_M * BLOCK_K;
    constexpr int B_TILE = BLOCK_N * BLOCK_K;

    static_assert(BLOCK_M % 16 == 0, "BLOCK_M multiple of 16");
    static_assert(N_ATOMS_PW >= 2 && N_ATOMS_PW % 2 == 0,
                  "ldmatrix pairs 2 N-atoms: N_ATOMS_PW must be even >= 2");

    extern __shared__ uint8_t smem_raw[];
    uint8_t* A_smem = smem_raw;                   // STAGES * A_TILE
    uint8_t* B_smem = A_smem + STAGES * A_TILE;   // STAGES * B_TILE (reused gate/up)
    // gate_smem: silu(gate) BF16, [BM, BN]. Written by gate epilogue, read by
    // the final silu(gate)*up epilogue (NOT per k-iter — only once at the end).
    __nv_bfloat16* gate_smem = reinterpret_cast<__nv_bfloat16*>(
        B_smem + STAGES * B_TILE);
    float* as_smem = reinterpret_cast<float*>(gate_smem + BLOCK_M * BLOCK_N);
    float* wsg_smem = as_smem + BLOCK_M * SCALE_KTILE;   // gate w_scale row
    float* wsu_smem = wsg_smem + SCALE_KTILE;            // up   w_scale row
    float* amax_smem = wsu_smem + SCALE_KTILE;

    const int cta_m = blockIdx.x;
    const int cta_n = blockIdx.y;
    const int m_base = cta_m * BLOCK_M;
    const int n_base = cta_n * BLOCK_N;
    const int gate_b_row0 = n_base;
    const int up_b_row0   = n_base + N;

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;
    const int frag_group = lane / 8;
    const int row_in_frag = lane % 8;
    const int row_block = frag_group / 2;
    const int col_block = frag_group % 2;

    const int K128 = K >> 7;
    const int N128 = N >> 7;
    const int gate_ws_row = (n_base >> 7);
    const int up_ws_row   = gate_ws_row + N128;

    auto stage_scales = [&](int kb0) {
        const int as_total = BLOCK_M * SCALE_KTILE;
        for (int idx = t; idx < as_total; idx += THREADS) {
            int r = idx / SCALE_KTILE;
            int kc = idx - r * SCALE_KTILE;
            int row = m_base + r;
            int kb = kb0 + kc;
            as_smem[idx] = (row < M && kb < K128)
                ? act_scale[(size_t)row * K128 + kb] : 0.0f;
        }
        for (int kc = t; kc < SCALE_KTILE; kc += THREADS) {
            int kb = kb0 + kc;
            wsg_smem[kc] = (kb < K128)
                ? w_scale[(size_t)gate_ws_row * K128 + kb] : 0.0f;
            wsu_smem[kc] = (kb < K128)
                ? w_scale[(size_t)up_ws_row * K128 + kb] : 0.0f;
        }
        __syncthreads();
    };

    // Stage A + one B band (gate or up) into smem. b_row0 picks the band.
    auto issue_load = [&](int stage, int k_base, int b_row0) {
        constexpr int A_CHUNKS = BLOCK_M * NUM_CHUNKS_PER_ROW;
        constexpr int A_ITERS = (A_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= A_CHUNKS) break;
            int row_a = idx / NUM_CHUNKS_PER_ROW;
            int chunk_a = idx % NUM_CHUNKS_PER_ROW;
            int m_glob = m_base + row_a;
            int k_glob = k_base + chunk_a * 16;
            const uint8_t* a_src = nullptr;
            if (m_glob < M && k_glob < K) {
                a_src = reinterpret_cast<const uint8_t*>(&A[(size_t)m_glob * K + k_glob]);
            }
            int csw = chunk_a ^ (row_a & SWIZZLE_MASK);
            cp_async_16(
                to_smem(&A_smem[stage * A_TILE + row_a * BLOCK_K + csw * 16]),
                a_src);
        }
        constexpr int B_CHUNKS = BLOCK_N * NUM_CHUNKS_PER_ROW;
        constexpr int B_ITERS = (B_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= B_CHUNKS) break;
            int row_b = idx / NUM_CHUNKS_PER_ROW;
            int chunk_b = idx % NUM_CHUNKS_PER_ROW;
            int n_glob = b_row0 + row_b;
            int k_glob = k_base + chunk_b * 16;
            const uint8_t* b_src = nullptr;
            if (n_glob < 2 * N && k_glob < K) {
                b_src = reinterpret_cast<const uint8_t*>(&B[(size_t)n_glob * K + k_glob]);
            }
            int csw = chunk_b ^ (row_b & SWIZZLE_MASK);
            cp_async_16(
                to_smem(&B_smem[stage * B_TILE + row_b * BLOCK_K + csw * 16]),
                b_src);
        }
    };

    // MMA pass over the staged A/B tiles for one k-iter, accumulating into `acc`
    // with the given w_scale smem row.
    auto mma_tile = [&](float (*acc)[N_ATOMS_PW][4], int compute_stage,
                        const float* ws_smem_pass) {
        const int kb = (compute_stage);  // caller passes k_iter; recompute below
        (void)kb;
        float tacc[M_ATOMS][N_ATOMS_PW][4];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi)
            #pragma unroll
            for (int ni = 0; ni < N_ATOMS_PW; ++ni)
                #pragma unroll
                for (int j = 0; j < 4; ++j) tacc[mi][ni][j] = 0.0f;

        uint8_t* A_stage = A_smem + compute_stage * A_TILE;
        uint8_t* B_stage = B_smem + compute_stage * B_TILE;
        #pragma unroll
        for (int ka = 0; ka < K_ATOMS; ++ka) {
            uint32_t A_regs[M_ATOMS][4];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int row = mi * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * ka + col_block;
                int csw = chunk ^ (row & SWIZZLE_MASK);
                ldmatrix_x4_b16(A_regs[mi][0], A_regs[mi][1], A_regs[mi][2], A_regs[mi][3],
                                to_smem(&A_stage[row * BLOCK_K + csw * 16]));
            }
            uint32_t B_regs[N_PAIRS_PW][4];
            #pragma unroll
            for (int np = 0; np < N_PAIRS_PW; ++np) {
                int nrow = warp_id * N_ATOMS_PW * 8 + np * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * ka + col_block;
                int csw = chunk ^ (nrow & SWIZZLE_MASK);
                ldmatrix_x4_b16(B_regs[np][0], B_regs[np][1], B_regs[np][2], B_regs[np][3],
                                to_smem(&B_stage[nrow * BLOCK_K + csw * 16]));
            }
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                #pragma unroll
                for (int np = 0; np < N_PAIRS_PW; ++np) {
                    int ni0 = np * 2, ni1 = np * 2 + 1;
                    mma_m16n8k32_e4m3(
                        tacc[mi][ni0][0], tacc[mi][ni0][1], tacc[mi][ni0][2], tacc[mi][ni0][3],
                        A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                        B_regs[np][0], B_regs[np][1]);
                    mma_m16n8k32_e4m3(
                        tacc[mi][ni1][0], tacc[mi][ni1][1], tacc[mi][ni1][2], tacc[mi][ni1][3],
                        A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                        B_regs[np][2], B_regs[np][3]);
                }
            }
        }
        return tacc;  // caller folds scales into acc
    };

    // Running accumulators for gate and up, both live across the whole K-loop
    // (true interleaved: both gate and up advance K-tile-by-K-tile together).
    float gate_acc[M_ATOMS][N_ATOMS_PW][4];
    float up_acc[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                gate_acc[mi][ni][j] = 0.0f;
                up_acc[mi][ni][j] = 0.0f;
            }

    const int K_ITERS = (K + BLOCK_K - 1) / BLOCK_K;
    // Prefetch STAGES-1 A tiles (A is shared by both passes — issued once).
    // B is NOT prefetched here; within each k-iter we issue B_gate then B_up
    // into the SAME smem region after the previous iter's B is consumed.
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        int kb = s * BLOCK_K;
        if (kb < K) issue_load(s, kb, gate_b_row0);   // first prefetch = gate B
        asm volatile("cp.async.commit_group;\n" ::);
    }

    int compute_stage = 0;
    for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
        int issue_iter = k_iter + (STAGES - 1);
        int issue_stage = issue_iter % STAGES;
        // Issue the NEXT A tile + the NEXT gate-B tile (the up-B for this k_iter
        // is loaded inside the gate-MMA sync below, reusing B_smem after gate
        // MMA reads finish).
        if (issue_iter < K_ITERS) issue_load(issue_stage, issue_iter * BLOCK_K, gate_b_row0);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
        __syncthreads();

        const int kb = k_iter;
        if ((kb % SCALE_KTILE) == 0) stage_scales(kb);

        // ---- gate MMA on the staged A + B_gate ----
        {
            float tacc[M_ATOMS][N_ATOMS_PW][4];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi)
                #pragma unroll
                for (int ni = 0; ni < N_ATOMS_PW; ++ni)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) tacc[mi][ni][j] = 0.0f;
            uint8_t* A_stage = A_smem + compute_stage * A_TILE;
            uint8_t* B_stage = B_smem + compute_stage * B_TILE;
            #pragma unroll
            for (int ka = 0; ka < K_ATOMS; ++ka) {
                uint32_t A_regs[M_ATOMS][4];
                #pragma unroll
                for (int mi = 0; mi < M_ATOMS; ++mi) {
                    int row = mi * 16 + row_block * 8 + row_in_frag;
                    int chunk = 2 * ka + col_block;
                    int csw = chunk ^ (row & SWIZZLE_MASK);
                    ldmatrix_x4_b16(A_regs[mi][0], A_regs[mi][1], A_regs[mi][2], A_regs[mi][3],
                                    to_smem(&A_stage[row * BLOCK_K + csw * 16]));
                }
                uint32_t B_regs[N_PAIRS_PW][4];
                #pragma unroll
                for (int np = 0; np < N_PAIRS_PW; ++np) {
                    int nrow = warp_id * N_ATOMS_PW * 8 + np * 16 + row_block * 8 + row_in_frag;
                    int chunk = 2 * ka + col_block;
                    int csw = chunk ^ (nrow & SWIZZLE_MASK);
                    ldmatrix_x4_b16(B_regs[np][0], B_regs[np][1], B_regs[np][2], B_regs[np][3],
                                    to_smem(&B_stage[nrow * BLOCK_K + csw * 16]));
                }
                #pragma unroll
                for (int mi = 0; mi < M_ATOMS; ++mi) {
                    #pragma unroll
                    for (int np = 0; np < N_PAIRS_PW; ++np) {
                        int ni0 = np * 2, ni1 = np * 2 + 1;
                        mma_m16n8k32_e4m3(
                            tacc[mi][ni0][0], tacc[mi][ni0][1], tacc[mi][ni0][2], tacc[mi][ni0][3],
                            A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                            B_regs[np][0], B_regs[np][1]);
                        mma_m16n8k32_e4m3(
                            tacc[mi][ni1][0], tacc[mi][ni1][1], tacc[mi][ni1][2], tacc[mi][ni1][3],
                            A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                            B_regs[np][2], B_regs[np][3]);
                    }
                }
            }
            int kbt = kb % SCALE_KTILE;
            float ws_cta = wsg_smem[kbt];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int row0 = m_base + mi * 16 + h;
                int row1 = row0 + 8;
                float as0 = as_smem[(mi * 16 + h) * SCALE_KTILE + kbt];
                float as1 = as_smem[(mi * 16 + h + 8) * SCALE_KTILE + kbt];
                #pragma unroll
                for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
                    gate_acc[mi][ni][0] += tacc[mi][ni][0] * (as0 * ws_cta);
                    gate_acc[mi][ni][1] += tacc[mi][ni][1] * (as0 * ws_cta);
                    gate_acc[mi][ni][2] += tacc[mi][ni][2] * (as1 * ws_cta);
                    gate_acc[mi][ni][3] += tacc[mi][ni][3] * (as1 * ws_cta);
                }
            }
        }
        __syncthreads();   // B_smem safe to overwrite with B_up

        // ---- load B_up into the SAME B_smem region, then up MMA ----
        // (A is still resident in A_smem[compute_stage]; not reloaded from HBM.)
        {
            // issue B_up into B_smem[compute_stage] (A_smem left untouched)
            constexpr int B_CHUNKS = BLOCK_N * NUM_CHUNKS_PER_ROW;
            constexpr int B_ITERS = (B_CHUNKS + THREADS - 1) / THREADS;
            int k_base = k_iter * BLOCK_K;
            #pragma unroll
            for (int it = 0; it < B_ITERS; ++it) {
                int idx = it * THREADS + t;
                if (idx >= B_CHUNKS) break;
                int row_b = idx / NUM_CHUNKS_PER_ROW;
                int chunk_b = idx % NUM_CHUNKS_PER_ROW;
                int n_glob = up_b_row0 + row_b;
                int k_glob = k_base + chunk_b * 16;
                const uint8_t* b_src = nullptr;
                if (n_glob < 2 * N && k_glob < K) {
                    b_src = reinterpret_cast<const uint8_t*>(&B[(size_t)n_glob * K + k_glob]);
                }
                int csw = chunk_b ^ (row_b & SWIZZLE_MASK);
                cp_async_16(
                    to_smem(&B_smem[compute_stage * B_TILE + row_b * BLOCK_K + csw * 16]),
                    b_src);
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
            __syncthreads();

            float tacc[M_ATOMS][N_ATOMS_PW][4];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi)
                #pragma unroll
                for (int ni = 0; ni < N_ATOMS_PW; ++ni)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) tacc[mi][ni][j] = 0.0f;
            uint8_t* A_stage = A_smem + compute_stage * A_TILE;
            uint8_t* B_stage = B_smem + compute_stage * B_TILE;
            #pragma unroll
            for (int ka = 0; ka < K_ATOMS; ++ka) {
                uint32_t A_regs[M_ATOMS][4];
                #pragma unroll
                for (int mi = 0; mi < M_ATOMS; ++mi) {
                    int row = mi * 16 + row_block * 8 + row_in_frag;
                    int chunk = 2 * ka + col_block;
                    int csw = chunk ^ (row & SWIZZLE_MASK);
                    ldmatrix_x4_b16(A_regs[mi][0], A_regs[mi][1], A_regs[mi][2], A_regs[mi][3],
                                    to_smem(&A_stage[row * BLOCK_K + csw * 16]));
                }
                uint32_t B_regs[N_PAIRS_PW][4];
                #pragma unroll
                for (int np = 0; np < N_PAIRS_PW; ++np) {
                    int nrow = warp_id * N_ATOMS_PW * 8 + np * 16 + row_block * 8 + row_in_frag;
                    int chunk = 2 * ka + col_block;
                    int csw = chunk ^ (nrow & SWIZZLE_MASK);
                    ldmatrix_x4_b16(B_regs[np][0], B_regs[np][1], B_regs[np][2], B_regs[np][3],
                                    to_smem(&B_stage[nrow * BLOCK_K + csw * 16]));
                }
                #pragma unroll
                for (int mi = 0; mi < M_ATOMS; ++mi) {
                    #pragma unroll
                    for (int np = 0; np < N_PAIRS_PW; ++np) {
                        int ni0 = np * 2, ni1 = np * 2 + 1;
                        mma_m16n8k32_e4m3(
                            tacc[mi][ni0][0], tacc[mi][ni0][1], tacc[mi][ni0][2], tacc[mi][ni0][3],
                            A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                            B_regs[np][0], B_regs[np][1]);
                        mma_m16n8k32_e4m3(
                            tacc[mi][ni1][0], tacc[mi][ni1][1], tacc[mi][ni1][2], tacc[mi][ni1][3],
                            A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                            B_regs[np][2], B_regs[np][3]);
                    }
                }
            }
            int kbt = kb % SCALE_KTILE;
            float ws_cta = wsu_smem[kbt];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int row0 = m_base + mi * 16 + h;
                int row1 = row0 + 8;
                float as0 = as_smem[(mi * 16 + h) * SCALE_KTILE + kbt];
                float as1 = as_smem[(mi * 16 + h + 8) * SCALE_KTILE + kbt];
                #pragma unroll
                for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
                    up_acc[mi][ni][0] += tacc[mi][ni][0] * (as0 * ws_cta);
                    up_acc[mi][ni][1] += tacc[mi][ni][1] * (as0 * ws_cta);
                    up_acc[mi][ni][2] += tacc[mi][ni][2] * (as1 * ws_cta);
                    up_acc[mi][ni][3] += tacc[mi][ni][3] * (as1 * ws_cta);
                }
            }
        }
        __syncthreads();
        compute_stage = (compute_stage + 1) % STAGES;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // ============ Epilogue: silu(gate)*up + per-row amax + quant ============
    // gate_acc and up_acc both live in registers. Replicate silu_mul_merged's
    // two bf16 roundings: bf16(silu(gate)) then bf16(silu_bf * up).
    constexpr float kFp8Max = 448.0f;
    float v[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        int rloc0 = mi * 16 + h;
        int rloc1 = rloc0 + 8;
        float amax0 = 0.0f, amax1 = 0.0f;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            if (row0 < M) {
                float gf0 = __bfloat162float(__float2bfloat16(silu_f32(gate_acc[mi][ni][0])));
                float gf1 = __bfloat162float(__float2bfloat16(silu_f32(gate_acc[mi][ni][1])));
                v[mi][ni][0] = __bfloat162float(__float2bfloat16(gf0 * up_acc[mi][ni][0]));
                v[mi][ni][1] = __bfloat162float(__float2bfloat16(gf1 * up_acc[mi][ni][1]));
                amax0 = fmaxf(amax0, fmaxf(fabsf(v[mi][ni][0]), fabsf(v[mi][ni][1])));
            } else { v[mi][ni][0] = 0.0f; v[mi][ni][1] = 0.0f; }
            if (row1 < M) {
                float gf0 = __bfloat162float(__float2bfloat16(silu_f32(gate_acc[mi][ni][2])));
                float gf1 = __bfloat162float(__float2bfloat16(silu_f32(gate_acc[mi][ni][3])));
                v[mi][ni][2] = __bfloat162float(__float2bfloat16(gf0 * up_acc[mi][ni][2]));
                v[mi][ni][3] = __bfloat162float(__float2bfloat16(gf1 * up_acc[mi][ni][3]));
                amax1 = fmaxf(amax1, fmaxf(fabsf(v[mi][ni][2]), fabsf(v[mi][ni][3])));
            } else { v[mi][ni][2] = 0.0f; v[mi][ni][3] = 0.0f; }
        }
        for (int off = 2; off > 0; off >>= 1) {
            amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffff, amax0, off));
            amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffff, amax1, off));
        }
        if (l == 0) {
            amax_smem[warp_id * BLOCK_M + rloc0] = amax0;
            amax_smem[warp_id * BLOCK_M + rloc1] = amax1;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int rloc = t; rloc < BLOCK_M; rloc += THREADS) {
        int row = m_base + rloc;
        if (row >= M) continue;
        float amax = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
            amax = fmaxf(amax, amax_smem[w * BLOCK_M + rloc]);
        float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
        amax_smem[rloc] = sc;
        // Each active thread owns a distinct rloc — write its row's scale
        // directly (no single-thread guard; see two-pass variant for the bug
        // this fixes).
        out_scale[(size_t)row * (N >> 7) + (n_base >> 7)] = sc;
    }
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        int rloc0 = mi * 16 + h;
        int rloc1 = rloc0 + 8;
        float sc0 = (row0 < M) ? amax_smem[rloc0] : 1.0f;
        float sc1 = (row1 < M) ? amax_smem[rloc1] : 1.0f;
        float inv0 = 1.0f / sc0, inv1 = 1.0f / sc1;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            if (row0 < M && col_pair_ok(n_pair_base, N)) {
                float q0 = fminf(fmaxf(v[mi][ni][0] * inv0, -kFp8Max), kFp8Max);
                float q1 = fminf(fmaxf(v[mi][ni][1] * inv0, -kFp8Max), kFp8Max);
                __nv_fp8_e4m3 p0(q0), p1(q1);
                uint16_t pack = (uint16_t)(*reinterpret_cast<const uint8_t*>(&p1)) << 8
                              | (uint16_t)(*reinterpret_cast<const uint8_t*>(&p0));
                *reinterpret_cast<uint16_t*>(&output[(size_t)row0 * N + n_pair_base]) = pack;
            } else if (row0 < M) {
                if (n_pair_base < N) output[(size_t)row0 * N + n_pair_base] = __nv_fp8_e4m3(fminf(fmaxf(v[mi][ni][0] * inv0, -kFp8Max), kFp8Max));
                if (n_pair_base + 1 < N) output[(size_t)row0 * N + n_pair_base + 1] = __nv_fp8_e4m3(fminf(fmaxf(v[mi][ni][1] * inv0, -kFp8Max), kFp8Max));
            }
            if (row1 < M && col_pair_ok(n_pair_base, N)) {
                float q2 = fminf(fmaxf(v[mi][ni][2] * inv1, -kFp8Max), kFp8Max);
                float q3 = fminf(fmaxf(v[mi][ni][3] * inv1, -kFp8Max), kFp8Max);
                __nv_fp8_e4m3 p2(q2), p3(q3);
                uint16_t pack = (uint16_t)(*reinterpret_cast<const uint8_t*>(&p3)) << 8
                              | (uint16_t)(*reinterpret_cast<const uint8_t*>(&p2));
                *reinterpret_cast<uint16_t*>(&output[(size_t)row1 * N + n_pair_base]) = pack;
            } else if (row1 < M) {
                if (n_pair_base < N) output[(size_t)row1 * N + n_pair_base] = __nv_fp8_e4m3(fminf(fmaxf(v[mi][ni][2] * inv1, -kFp8Max), kFp8Max));
                if (n_pair_base + 1 < N) output[(size_t)row1 * N + n_pair_base + 1] = __nv_fp8_e4m3(fminf(fmaxf(v[mi][ni][3] * inv1, -kFp8Max), kFp8Max));
            }
        }
    }
    (void)gate_smem;  // apersist keeps gate in regs; gate_smem unused (kept for layout parity)
    (void)mma_tile;   // helper retained for future chunked variant; unused in this path
}

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
