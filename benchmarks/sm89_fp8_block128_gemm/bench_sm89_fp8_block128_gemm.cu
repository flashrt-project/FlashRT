// SPDX-License-Identifier: Apache-2.0
//
// Standalone SM89 FP8 block-128 GEMM benchmark for Qwen3-VL prefill.
// Keep this file independent from the FlashRT Python extension so kernel
// iteration stays fast and reproducible.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// Production kernel body, shared verbatim. `--mode baseline` runs THIS exact
// kernel, so the bench baseline cannot drift behind production. The candidate
// (EXPERIMENT slot, see below) is seeded from the same source.
#include "fp8_bs_gemm_device.cuh"

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

// ---------------------------------------------------------------------------
// EXPERIMENT slot. By default the candidate IS the production kernel (included
// from the shared header below), so a freshly-synced bench reports ~0% delta
// between baseline and candidate -- a built-in honesty check that the harness
// is faithful. To iterate on a structural change, build with -DEXPERIMENT and
// edit `fp8_bs_gemm_kernel_cand` below; once an experiment is accepted, fold
// it into csrc/gemm/fp8_bs_gemm_device.cuh and the baseline tracks it again.
// ---------------------------------------------------------------------------
#ifdef EXPERIMENT
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

// BLOCK_K is pinned to 128 (one DeepSeek scale block per K-iteration).
//  - A: [M, K] row-major FP8 e4m3, act_scale [M, K/128] fp32
//  - B: [N, K] row-major FP8 e4m3, w_scale [N/128, K/128] fp32
//  - D: [M, N] row-major BF16
//  - BLOCK_N must keep each warp's 8-wide N-atoms inside one 128 scale block.
template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int STAGES,
          int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_bs_gemm_kernel_cand(
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
            int n_glob = n_base + row_b;
            int k_glob = k_base + chunk_b * 16;
            const uint8_t* b_src = nullptr;
            if (n_glob < N && k_glob < K) {
                b_src = reinterpret_cast<const uint8_t*>(&B[(size_t)n_glob * K + k_glob]);
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
#endif  // EXPERIMENT

template <bool CANDIDATE>
void launch_64x64_s1(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
                     const float* act_scale, const float* w_scale,
                     __nv_bfloat16* D, int M, int N, int K,
                     cudaStream_t stream) {
    namespace prod = flash_rt::gemm::block128_sm89;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 128;
    constexpr int W = 4;
    constexpr int S = 1;
    constexpr int MB = 4;
    constexpr int SCALE_KTILE = 8;
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    dim3 block(W * 32, 1, 1);
    // Swizzled smem (no pad) + staged scale tile -- same layout production uses.
    int smem_bytes = S * (BM + BN) * BK
                   + (BM * SCALE_KTILE + SCALE_KTILE) * (int)sizeof(float);
    if (CANDIDATE) {
#ifdef EXPERIMENT
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute((const void*)&fp8_bs_gemm_kernel_cand<BM, BN, W, S, MB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        fp8_bs_gemm_kernel_cand<BM, BN, W, S, MB>
            <<<grid, block, smem_bytes, stream>>>(A, B, act_scale, w_scale, D, M, N, K);
        return;
#else
        // No experiment compiled in: candidate == production baseline.
#endif
    }
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute((const void*)&prod::fp8_bs_gemm_kernel<BM, BN, W, S, MB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    prod::fp8_bs_gemm_kernel<BM, BN, W, S, MB>
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
