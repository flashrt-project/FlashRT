// SPDX-License-Identifier: Apache-2.0
//
// Standalone SM89 FP8 block-128 GEMM benchmark for Qwen3-VL prefill.
// Keep this file independent from the FlashRT Python extension so kernel
// iteration stays fast and reproducible.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,          \
                         __LINE__, cudaGetErrorString(_err));                \
            std::exit(1);                                                     \
        }                                                                    \
    } while (0)

namespace {

__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
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

// True when the adjacent column pair {c, c+1} is fully in bounds, so a
// 32-bit bfloat162 store is valid. n_pair_base is even (=...+2*l) and N is a
// multiple of 128, so &D[row*N + c] is 4-byte aligned for the vector store.
__device__ __forceinline__ bool col_pair_ok(int c, int N) {
    return c + 1 < N;
}

// ldmatrix.x4: load four 8x8 b16 fragments from smem into 4 registers/lane in
// one instruction. The C4 candidate uses it to replace 4 scalar 32-bit LDS,
// offloading the saturated LSU pipe (NCU: LSU 67.7%, 54.7M shared loads).
__device__ __forceinline__ void ldmatrix_x4_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3, uint32_t smem_addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(smem_addr));
}

template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int STAGES,
          int MIN_BLOCKS_PER_SM, bool CANDIDATE>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_bs_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    const float* __restrict__ act_scale,
    const float* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ D,
    int M, int N, int K)
{
    constexpr int BLOCK_K    = 128;
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int M_ATOMS    = BLOCK_M / 16;
    constexpr int N_ATOMS    = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int K_ATOMS    = BLOCK_K / 32;
    constexpr int SMEM_K_PAD = BLOCK_K + 16;

    static_assert(BLOCK_M % 16 == 0, "BLOCK_M multiple of 16");
    static_assert(BLOCK_N % 8 == 0,  "BLOCK_N multiple of 8");
    static_assert(BLOCK_N <= 128, "one CTA must fit one N scale block");
    static_assert((BLOCK_N / 8) % NUM_WARPS == 0, "N-atoms split across warps");

    extern __shared__ uint8_t smem_raw[];
    uint8_t* A_smem = smem_raw;
    uint8_t* B_smem = A_smem + STAGES * BLOCK_M * SMEM_K_PAD;
    // Candidate C1: stage activation/weight scales in shared memory with a
    // coalesced load, so the per-k_iter scale fold reads smem instead of
    // row-strided scalar global loads. To keep the smem footprint independent
    // of K (occupancy-neutral), stage only SCALE_KTILE scale-block columns at
    // a time (act_scale[m_base:m_base+BLOCK_M, kb0:kb0+SCALE_KTILE]).
    constexpr int SCALE_KTILE = 8;
    float* as_smem = reinterpret_cast<float*>(
        B_smem + STAGES * BLOCK_N * SMEM_K_PAD);
    float* ws_smem = as_smem + (CANDIDATE ? BLOCK_M * SCALE_KTILE : 0);

    const int cta_m = blockIdx.x;
    const int cta_n = blockIdx.y;
    const int m_base = cta_m * BLOCK_M;
    const int n_base = cta_n * BLOCK_N;

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;

    const int K128 = K >> 7;

    // Coalesced staging of one SCALE_KTILE-wide scale block into smem.
    // Layout: as_smem[local_row * SCALE_KTILE + (kb - kb0)] for the rows this
    // CTA owns; ws_smem[kb - kb0] for the CTA's single N scale block.
    auto stage_scales = [&](int kb0) {
        if constexpr (CANDIDATE) {
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
        }
    };

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

        const int kb = k_iter;
        // Re-stage the next SCALE_KTILE-wide scale block when crossing a tile
        // boundary (candidate only; no-op for baseline).
        if ((kb % SCALE_KTILE) == 0) stage_scales(kb);
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
                        tacc[mi][ni][0], tacc[mi][ni][1],
                        tacc[mi][ni][2], tacc[mi][ni][3],
                        A0, A1, A2, A3, B0, B1);
                }
            }
        }

        int kbt = kb % SCALE_KTILE;   // column within the staged scale tile
        float ws_cta = CANDIDATE ? ws_smem[kbt]
                                  : w_scale[(size_t)(n_base >> 7) * K128 + kb];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi) {
            int row0 = m_base + mi * 16 + h;
            int row1 = row0 + 8;
            float as0, as1;
            if constexpr (CANDIDATE) {
                // Staged in smem with bounds applied during load.
                as0 = as_smem[(mi * 16 + h) * SCALE_KTILE + kbt];
                as1 = as_smem[(mi * 16 + h + 8) * SCALE_KTILE + kbt];
            } else {
                as0 = (row0 < M) ? act_scale[row0 * K128 + kb] : 0.0f;
                as1 = (row1 < M) ? act_scale[row1 * K128 + kb] : 0.0f;
            }
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

    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            // m16n8 layout: acc[0,1] are row0 cols {2l,2l+1}; acc[2,3] are
            // row1 cols {2l,2l+1}. The two columns are adjacent, so candidate
            // emits one 32-bit bfloat162 store per row instead of two scalar
            // 16-bit stores (NCU's top store-pattern bottleneck). Tail (odd
            // last column) keeps scalar stores.
            if constexpr (CANDIDATE) {
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
            } else {
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
}

// ---------------------------------------------------------------------------
// C4 candidate: ldmatrix.x4 + 128B-swizzle smem, block-scaled (C1 staging) +
// pair-store (C2). Structure ported from the sm120 ldmatrix kernel; MMA uses
// the sm89 e4m3 variant (no kind::f8f6f4). Swizzle removes the SMEM_K_PAD and
// bank conflicts; ldmatrix collapses 4 scalar LDS into one to offload the LSU.
//   Restrictions: BLOCK_K=128, N_ATOMS_PW even (>=2), BLOCK_M%16==0.
template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int STAGES, int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_bs_gemm_ld_kernel(
    const __nv_fp8_e4m3* __restrict__ A, const __nv_fp8_e4m3* __restrict__ B,
    const float* __restrict__ act_scale, const float* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ D, int M, int N, int K)
{
    constexpr int BLOCK_K   = 128;
    constexpr int THREADS   = NUM_WARPS * 32;
    constexpr int M_ATOMS   = BLOCK_M / 16;
    constexpr int N_ATOMS   = BLOCK_N / 8;
    constexpr int N_ATOMS_PW= N_ATOMS / NUM_WARPS;
    constexpr int N_PAIRS_PW= N_ATOMS_PW / 2;
    constexpr int K_ATOMS   = BLOCK_K / 32;            // = 4
    constexpr int NUM_CHUNKS_PER_ROW = BLOCK_K / 16;   // = 8 chunks of 16B
    // 128B swizzle: chunk_sw = chunk ^ (row & SWIZZLE_MASK). For 8 chunks/row
    // the period is 8, mask 7 (>8 chunks would need mask 7 too on 4090).
    constexpr int SWIZZLE_MASK = (NUM_CHUNKS_PER_ROW <= 8) ? (NUM_CHUNKS_PER_ROW - 1) : 7;
    constexpr int SCALE_KTILE = 8;
    constexpr int A_TILE = BLOCK_M * BLOCK_K;          // no pad (swizzle)
    constexpr int B_TILE = BLOCK_N * BLOCK_K;

    static_assert(N_ATOMS_PW >= 2 && N_ATOMS_PW % 2 == 0, "N_ATOMS_PW even>=2");

    extern __shared__ uint8_t smem_raw[];
    uint8_t* A_smem = smem_raw;
    uint8_t* B_smem = A_smem + STAGES * A_TILE;
    float* as_smem = reinterpret_cast<float*>(B_smem + STAGES * B_TILE);
    float* ws_smem = as_smem + BLOCK_M * SCALE_KTILE;

    const int m_base = blockIdx.x * BLOCK_M;
    const int n_base = blockIdx.y * BLOCK_N;
    const int t = threadIdx.x, warp_id = t / 32, lane = t % 32;
    const int frag_group = lane / 8, row_in_frag = lane % 8;
    const int row_block = frag_group / 2, col_block = frag_group % 2;
    const int h = lane / 4, l = lane % 4;
    const int K128 = K >> 7;

    auto stage_scales = [&](int kb0) {
        for (int idx = t; idx < BLOCK_M * SCALE_KTILE; idx += THREADS) {
            int r = idx / SCALE_KTILE, kc = idx - r * SCALE_KTILE;
            int row = m_base + r, kb = kb0 + kc;
            as_smem[idx] = (row < M && kb < K128) ? act_scale[(size_t)row * K128 + kb] : 0.0f;
        }
        for (int kc = t; kc < SCALE_KTILE; kc += THREADS) {
            int kb = kb0 + kc;
            ws_smem[kc] = (kb < K128) ? w_scale[(size_t)(n_base >> 7) * K128 + kb] : 0.0f;
        }
        __syncthreads();
    };

    auto issue_load = [&](int stage, int k_base) {
        constexpr int A_CHUNKS = BLOCK_M * NUM_CHUNKS_PER_ROW;
        #pragma unroll
        for (int it = 0; it * THREADS < A_CHUNKS; ++it) {
            int idx = it * THREADS + t; if (idx >= A_CHUNKS) break;
            int row = idx / NUM_CHUNKS_PER_ROW, chunk = idx % NUM_CHUNKS_PER_ROW;
            int m_g = m_base + row, k_g = k_base + chunk * 16;
            const uint8_t* src = (m_g < M && k_g < K)
                ? reinterpret_cast<const uint8_t*>(&A[(size_t)m_g * K + k_g]) : nullptr;
            int csw = chunk ^ (row & SWIZZLE_MASK);
            cp_async_16(to_smem(&A_smem[stage * A_TILE + row * BLOCK_K + csw * 16]), src);
        }
        constexpr int B_CHUNKS = BLOCK_N * NUM_CHUNKS_PER_ROW;
        #pragma unroll
        for (int it = 0; it * THREADS < B_CHUNKS; ++it) {
            int idx = it * THREADS + t; if (idx >= B_CHUNKS) break;
            int row = idx / NUM_CHUNKS_PER_ROW, chunk = idx % NUM_CHUNKS_PER_ROW;
            int n_g = n_base + row, k_g = k_base + chunk * 16;
            const uint8_t* src = (n_g < N && k_g < K)
                ? reinterpret_cast<const uint8_t*>(&B[(size_t)n_g * K + k_g]) : nullptr;
            int csw = chunk ^ (row & SWIZZLE_MASK);
            cp_async_16(to_smem(&B_smem[stage * B_TILE + row * BLOCK_K + csw * 16]), src);
        }
    };

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
        if (s * BLOCK_K < K) issue_load(s, s * BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    int compute_stage = 0;
    for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
        int issue_iter = k_iter + (STAGES - 1), issue_stage = issue_iter % STAGES;
        if (issue_iter < K_ITERS) issue_load(issue_stage, issue_iter * BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
        __syncthreads();

        const int kb = k_iter;
        if ((kb % SCALE_KTILE) == 0) stage_scales(kb);

        uint8_t* A_stage = A_smem + compute_stage * A_TILE;
        uint8_t* B_stage = B_smem + compute_stage * B_TILE;
        float tacc[M_ATOMS][N_ATOMS_PW][4];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi)
            #pragma unroll
            for (int ni = 0; ni < N_ATOMS_PW; ++ni)
                #pragma unroll
                for (int j = 0; j < 4; ++j) tacc[mi][ni][j] = 0.0f;

        #pragma unroll
        for (int k_a = 0; k_a < K_ATOMS; ++k_a) {
            uint32_t A_regs[M_ATOMS][4];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int row = mi * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * k_a + col_block;
                int csw = chunk ^ (row & SWIZZLE_MASK);
                ldmatrix_x4_b16(A_regs[mi][0], A_regs[mi][1], A_regs[mi][2], A_regs[mi][3],
                                to_smem(&A_stage[row * BLOCK_K + csw * 16]));
            }
            uint32_t B_regs[N_PAIRS_PW][4];
            #pragma unroll
            for (int np = 0; np < N_PAIRS_PW; ++np) {
                int nrow = warp_id * N_ATOMS_PW * 8 + np * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * k_a + col_block;
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

        int kbt = kb % SCALE_KTILE;
        float ws_cta = ws_smem[kbt];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi) {
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

    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h, row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int c = n_base + warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            if (row0 < M && col_pair_ok(c, N))
                *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)row0 * N + c]) =
                    __floats2bfloat162_rn(acc[mi][ni][0], acc[mi][ni][1]);
            else if (row0 < M) {
                if (c < N)     D[(size_t)row0 * N + c]   = __float2bfloat16(acc[mi][ni][0]);
                if (c + 1 < N) D[(size_t)row0 * N + c+1] = __float2bfloat16(acc[mi][ni][1]);
            }
            if (row1 < M && col_pair_ok(c, N))
                *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)row1 * N + c]) =
                    __floats2bfloat162_rn(acc[mi][ni][2], acc[mi][ni][3]);
            else if (row1 < M) {
                if (c < N)     D[(size_t)row1 * N + c]   = __float2bfloat16(acc[mi][ni][2]);
                if (c + 1 < N) D[(size_t)row1 * N + c+1] = __float2bfloat16(acc[mi][ni][3]);
            }
        }
    }
}

template <bool CANDIDATE>
void launch_64x64_s1(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
                     const float* act_scale, const float* w_scale,
                     __nv_bfloat16* D, int M, int N, int K,
                     cudaStream_t stream) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 128;
    constexpr int W = 4;
    constexpr int S = 1;
    constexpr int MB = 4;
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    dim3 block(W * 32, 1, 1);
    constexpr int SCALE_KTILE = 8;
    if (CANDIDATE) {
#ifdef USE_LDMATRIX
        // C4: swizzled smem (no pad) + ldmatrix + C1 scales + C2 store.
        int smem_bytes = S * (BM + BN) * BK
                       + (BM * SCALE_KTILE + SCALE_KTILE) * (int)sizeof(float);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute((const void*)&fp8_bs_gemm_ld_kernel<BM, BN, W, S, MB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        fp8_bs_gemm_ld_kernel<BM, BN, W, S, MB>
            <<<grid, block, smem_bytes, stream>>>(A, B, act_scale, w_scale, D, M, N, K);
        return;
#else
        int smem_bytes = S * (BM + BN) * (BK + 16)
                       + (BM * SCALE_KTILE + SCALE_KTILE) * (int)sizeof(float);
        fp8_bs_gemm_kernel<BM, BN, W, S, MB, true>
            <<<grid, block, smem_bytes, stream>>>(A, B, act_scale, w_scale, D, M, N, K);
        return;
#endif
    }
    int smem_bytes = S * (BM + BN) * (BK + 16);
    fp8_bs_gemm_kernel<BM, BN, W, S, MB, false>
        <<<grid, block, smem_bytes, stream>>>(A, B, act_scale, w_scale, D, M, N, K);
}

__global__ void init_fp8_kernel(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B,
                                float* act_scale, float* w_scale,
                                int M, int N, int K) {
    size_t total_a = static_cast<size_t>(M) * K;
    size_t total_b = static_cast<size_t>(N) * K;
    size_t total_as = static_cast<size_t>(M) * (K >> 7);
    size_t total_ws = static_cast<size_t>(N >> 7) * (K >> 7);
    size_t total = total_a + total_b + total_as + total_ws;

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x) {
        if (idx < total_a) {
            int v = static_cast<int>((idx * 17 + 13) & 31) - 16;
            A[idx] = __nv_fp8_e4m3(static_cast<float>(v) * 0.03125f);
        } else if (idx < total_a + total_b) {
            size_t j = idx - total_a;
            int v = static_cast<int>((j * 11 + 7) & 31) - 16;
            B[j] = __nv_fp8_e4m3(static_cast<float>(v) * 0.03125f);
        } else if (idx < total_a + total_b + total_as) {
            size_t j = idx - total_a - total_b;
            act_scale[j] = 0.75f + static_cast<float>((j * 5) & 15) * 0.0025f;
        } else {
            size_t j = idx - total_a - total_b - total_as;
            w_scale[j] = 0.80f + static_cast<float>((j * 3) & 15) * 0.0025f;
        }
    }
}

__global__ void fill_bf16_kernel(__nv_bfloat16* D, size_t n, float value) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += blockDim.x * gridDim.x) {
        D[idx] = __float2bfloat16(value);
    }
}

__global__ void flush_l2_kernel(float* buf, size_t n) {
    float x = 0.0f;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += blockDim.x * gridDim.x) {
        x += buf[idx];
        buf[idx] = x + 1.0f;
    }
}

__global__ void check_samples_kernel(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    const float* act_scale,
    const float* w_scale,
    const __nv_bfloat16* D,
    float* abs_err,
    float* rel_err,
    int M, int N, int K, int samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= samples) return;
    int m = (idx * 9973 + 17) % M;
    int n = (idx * 7919 + 23) % N;
    int K128 = K >> 7;
    float ref = 0.0f;
    for (int kb = 0; kb < K128; ++kb) {
        float part = 0.0f;
        int k_base = kb << 7;
        #pragma unroll 4
        for (int kk = 0; kk < 128; ++kk) {
            float av = static_cast<float>(A[static_cast<size_t>(m) * K + k_base + kk]);
            float bv = static_cast<float>(B[static_cast<size_t>(n) * K + k_base + kk]);
            part += av * bv;
        }
        ref += part * act_scale[m * K128 + kb] * w_scale[(n >> 7) * K128 + kb];
    }
    float got = __bfloat162float(D[static_cast<size_t>(m) * N + n]);
    float err = fabsf(got - ref);
    abs_err[idx] = err;
    rel_err[idx] = err / fmaxf(fabsf(ref), 1.0e-6f);
}

struct Shape {
    int M = 1581;
    int N = 12288;
    int K = 4096;
};

struct Args {
    Shape shape;
    std::string shape_name = "gate";
    std::string mode = "both";
    int warmup = 10;
    int iters = 50;
    int check_samples = 256;
    int flush_l2_mb = 256;
};

void usage(const char* argv0) {
    std::printf(
        "usage: %s [--shape gate|up|down|qkv] [--M m --N n --K k]\n"
        "          [--mode baseline|candidate|both] [--warmup n] [--iters n]\n"
        "          [--check-samples n] [--flush-l2-mb n]\n",
        argv0);
}

Args parse_args(int argc, char** argv) {
    Args args;
    bool custom_m = false, custom_n = false, custom_k = false;
    for (int i = 1; i < argc; ++i) {
        auto need_value = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--shape") == 0) {
            args.shape_name = need_value("--shape");
        } else if (std::strcmp(argv[i], "--mode") == 0) {
            args.mode = need_value("--mode");
        } else if (std::strcmp(argv[i], "--M") == 0) {
            args.shape.M = std::atoi(need_value("--M"));
            custom_m = true;
        } else if (std::strcmp(argv[i], "--N") == 0) {
            args.shape.N = std::atoi(need_value("--N"));
            custom_n = true;
        } else if (std::strcmp(argv[i], "--K") == 0) {
            args.shape.K = std::atoi(need_value("--K"));
            custom_k = true;
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            args.warmup = std::atoi(need_value("--warmup"));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            args.iters = std::atoi(need_value("--iters"));
        } else if (std::strcmp(argv[i], "--check-samples") == 0) {
            args.check_samples = std::atoi(need_value("--check-samples"));
        } else if (std::strcmp(argv[i], "--flush-l2-mb") == 0) {
            args.flush_l2_mb = std::atoi(need_value("--flush-l2-mb"));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            std::exit(2);
        }
    }

    if (!custom_m && !custom_n && !custom_k) {
        if (args.shape_name == "gate" || args.shape_name == "up") {
            args.shape = {1581, 12288, 4096};
        } else if (args.shape_name == "down") {
            args.shape = {1581, 4096, 12288};
        } else if (args.shape_name == "qkv") {
            args.shape = {1581, 6144, 4096};
        } else {
            std::fprintf(stderr, "unknown shape: %s\n", args.shape_name.c_str());
            std::exit(2);
        }
    }
    if ((args.shape.N % 128) != 0 || (args.shape.K % 128) != 0) {
        std::fprintf(stderr, "N and K must be multiples of 128\n");
        std::exit(2);
    }
    if (args.mode != "baseline" && args.mode != "candidate" && args.mode != "both") {
        std::fprintf(stderr, "mode must be baseline, candidate, or both\n");
        std::exit(2);
    }
    return args;
}

struct TimeStats {
    float median = 0.0f;
    float mean = 0.0f;
    float min = 0.0f;
};

template <bool CANDIDATE>
TimeStats run_timing(const Args& args,
                     const __nv_fp8_e4m3* A,
                     const __nv_fp8_e4m3* B,
                     const float* act_scale,
                     const float* w_scale,
                     __nv_bfloat16* D,
                     float* flush,
                     size_t flush_elems,
                     cudaStream_t stream) {
    for (int i = 0; i < args.warmup; ++i) {
        launch_64x64_s1<CANDIDATE>(A, B, act_scale, w_scale, D,
                                   args.shape.M, args.shape.N, args.shape.K,
                                   stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> times;
    times.reserve(args.iters);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < args.iters; ++i) {
        if (flush_elems != 0) {
            flush_l2_kernel<<<1024, 256, 0, stream>>>(flush, flush_elems);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch_64x64_s1<CANDIDATE>(A, B, act_scale, w_scale, D,
                                   args.shape.M, args.shape.N, args.shape.K,
                                   stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    TimeStats stats;
    stats.median = times[times.size() / 2];
    stats.min = times.front();
    double sum = 0.0;
    for (float t : times) sum += t;
    stats.mean = static_cast<float>(sum / times.size());
    return stats;
}

struct CheckStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float max_rel = 0.0f;
    float mean_rel = 0.0f;
};

CheckStats check_output(const Args& args,
                        const __nv_fp8_e4m3* A,
                        const __nv_fp8_e4m3* B,
                        const float* act_scale,
                        const float* w_scale,
                        const __nv_bfloat16* D,
                        cudaStream_t stream) {
    int samples = args.check_samples;
    float* abs_dev = nullptr;
    float* rel_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&abs_dev, sizeof(float) * samples));
    CUDA_CHECK(cudaMalloc(&rel_dev, sizeof(float) * samples));
    check_samples_kernel<<<(samples + 255) / 256, 256, 0, stream>>>(
        A, B, act_scale, w_scale, D, abs_dev, rel_dev,
        args.shape.M, args.shape.N, args.shape.K, samples);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> abs_host(samples);
    std::vector<float> rel_host(samples);
    CUDA_CHECK(cudaMemcpy(abs_host.data(), abs_dev, sizeof(float) * samples,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rel_host.data(), rel_dev, sizeof(float) * samples,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(abs_dev));
    CUDA_CHECK(cudaFree(rel_dev));

    CheckStats stats;
    double abs_sum = 0.0;
    double rel_sum = 0.0;
    for (int i = 0; i < samples; ++i) {
        stats.max_abs = std::max(stats.max_abs, abs_host[i]);
        stats.max_rel = std::max(stats.max_rel, rel_host[i]);
        abs_sum += abs_host[i];
        rel_sum += rel_host[i];
    }
    stats.mean_abs = static_cast<float>(abs_sum / samples);
    stats.mean_rel = static_cast<float>(rel_sum / samples);
    return stats;
}

void print_result(const char* name, const TimeStats& t, const CheckStats& c) {
    std::printf("%-10s median_ms=%8.4f mean_ms=%8.4f min_ms=%8.4f "
                "max_abs=%9.6f mean_abs=%9.6f max_rel=%9.6f mean_rel=%9.6f\n",
                name, t.median, t.mean, t.min,
                c.max_abs, c.mean_abs, c.max_rel, c.mean_rel);
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int M = args.shape.M;
    int N = args.shape.N;
    int K = args.shape.K;
    int K128 = K >> 7;

    std::printf("shape=%s M=%d N=%d K=%d warmup=%d iters=%d flush_l2_mb=%d\n",
                args.shape_name.c_str(), M, N, K,
                args.warmup, args.iters, args.flush_l2_mb);

    __nv_fp8_e4m3* A = nullptr;
    __nv_fp8_e4m3* B = nullptr;
    __nv_bfloat16* D = nullptr;
    float* act_scale = nullptr;
    float* w_scale = nullptr;
    float* flush = nullptr;

    CUDA_CHECK(cudaMalloc(&A, static_cast<size_t>(M) * K * sizeof(*A)));
    CUDA_CHECK(cudaMalloc(&B, static_cast<size_t>(N) * K * sizeof(*B)));
    CUDA_CHECK(cudaMalloc(&D, static_cast<size_t>(M) * N * sizeof(*D)));
    CUDA_CHECK(cudaMalloc(&act_scale, static_cast<size_t>(M) * K128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w_scale, static_cast<size_t>(N >> 7) * K128 * sizeof(float)));

    size_t flush_elems = static_cast<size_t>(args.flush_l2_mb) * 1024 * 1024 / sizeof(float);
    if (flush_elems != 0) {
        CUDA_CHECK(cudaMalloc(&flush, flush_elems * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(flush, 0, flush_elems * sizeof(float), stream));
    }

    init_fp8_kernel<<<4096, 256, 0, stream>>>(A, B, act_scale, w_scale, M, N, K);
    fill_bf16_kernel<<<4096, 256, 0, stream>>>(D, static_cast<size_t>(M) * N, 0.0f);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (args.mode == "baseline" || args.mode == "both") {
        TimeStats t = run_timing<false>(args, A, B, act_scale, w_scale, D,
                                        flush, flush_elems, stream);
        CheckStats c = check_output(args, A, B, act_scale, w_scale, D, stream);
        print_result("baseline", t, c);
    }
    if (args.mode == "candidate" || args.mode == "both") {
        TimeStats t = run_timing<true>(args, A, B, act_scale, w_scale, D,
                                       flush, flush_elems, stream);
        CheckStats c = check_output(args, A, B, act_scale, w_scale, D, stream);
        print_result("candidate", t, c);
    }

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(D));
    CUDA_CHECK(cudaFree(act_scale));
    CUDA_CHECK(cudaFree(w_scale));
    if (flush) CUDA_CHECK(cudaFree(flush));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
