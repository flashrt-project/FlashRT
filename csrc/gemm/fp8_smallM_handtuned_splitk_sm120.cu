// SPDX-License-Identifier: Apache-2.0
//
// SplitK variant of hand-tuned FP8 GEMM for sm_120a small-N motus shapes
// (action_o, und_o, und_ffn_dn). Splits K-axis across k_split CTAs to break
// 13-29% HBM efficiency floor on shapes where N is too small to fill SMs.
//
// Architecture:
//   1. partial_kernel: grid = (M/BM, N/BN, k_split). Each CTA computes its
//      slice of K, writes partial fp32 result to scratch[k_split][M][N].
//   2. reduce_kernel: sum scratch across k_split dim, multiply by alpha,
//      cast to BF16 final output.
//
// Two kernel launches but per-launch overhead amortized across more SMs.
// Critical for: action_o (N=1024 -> only 16 CTAs at BN=64), und_o (N=512),
// und_ffn_dn (N=512).

#include "fp8_smallM_handtuned_splitk_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace smallM_splitk {

namespace {

__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
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

// SplitK partial GEMM kernel. Each CTA writes partial fp32 to scratch.
//   scratch layout: [k_split][M][N] fp32, row-major.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS>
__global__ __launch_bounds__(NUM_WARPS * 32, 4)
void fp8_gemm_splitk_partial_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    float* __restrict__ scratch,  // [k_split][M][N] fp32
    int M, int N, int K, int k_split)
{
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int M_ATOMS    = BLOCK_M / 16;
    constexpr int N_ATOMS    = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int K_ATOMS    = BLOCK_K / 32;
    constexpr int SMEM_K_PAD = BLOCK_K + 16;

    extern __shared__ uint8_t smem_raw[];
    uint8_t* A_smem = smem_raw;
    uint8_t* B_smem = A_smem + 2 * BLOCK_M * SMEM_K_PAD;

    const int cta_m  = blockIdx.x;
    const int cta_n  = blockIdx.y;
    const int cta_ks = blockIdx.z;  // which K-split this CTA covers
    const int m_base = cta_m * BLOCK_M;
    const int n_base = cta_n * BLOCK_N;

    // K range this CTA covers.
    const int K_per_split_rounded = ((K / k_split) + BLOCK_K - 1) / BLOCK_K * BLOCK_K;
    const int k_start = cta_ks * K_per_split_rounded;
    int k_end_temp = k_start + K_per_split_rounded;
    if (k_end_temp > K) k_end_temp = K;
    const int k_end = k_end_temp;
    if (k_start >= K) {
        // Empty CTA — still write zeros to scratch slot.
        // (We could skip, but the reduce kernel reads all slots.)
        int t = threadIdx.x;
        float* scratch_slot = scratch + (size_t)cta_ks * M * N + m_base * N + n_base;
        // Zero only this CTA's slot.
        for (int idx = t; idx < BLOCK_M * BLOCK_N; idx += THREADS) {
            int mi = idx / BLOCK_N;
            int ni = idx % BLOCK_N;
            if (m_base + mi < M && n_base + ni < N) {
                scratch_slot[mi * N + ni] = 0.0f;
            }
        }
        return;
    }

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;

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

    issue_load(0, k_start);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;
    for (int k_base = k_start; k_base < k_end; k_base += BLOCK_K) {
        int next_stage = compute_stage ^ 1;
        if (k_base + BLOCK_K < k_end) issue_load(next_stage, k_base + BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        #pragma unroll
        for (int k_iter = 0; k_iter < K_ATOMS; ++k_iter) {
            int kA0 = k_iter * 32 + 4 * l;
            int kA2 = k_iter * 32 + 4 * l + 16;
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
                        acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                        A0, A1, A2, A3, B0, B1);
                }
            }
        }
        compute_stage ^= 1;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // Write partial fp32 result to scratch [cta_ks][M][N].
    float* scratch_slot = scratch + (size_t)cta_ks * M * N;
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
                    scratch_slot[row * N + col] = acc[mi][ni][j];
                }
            }
        }
    }
}

// Reduce kernel: sum scratch[k_split][M][N] -> D[M][N] BF16 with alpha.
__global__ void fp8_gemm_splitk_reduce_kernel(
    const float* __restrict__ scratch,
    __nv_bfloat16* __restrict__ D,
    int M, int N, int k_split, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    float sum = 0.0f;
    size_t per_split = (size_t)M * N;
    for (int s = 0; s < k_split; ++s) {
        sum += scratch[s * per_split + idx];
    }
    D[idx] = __float2bfloat16(sum * alpha);
}

template <int BM, int BN, int BK, int W>
int launch_(const void* A, const void* B, void* D,
            int M, int N, int K, int k_split, float alpha,
            void* scratch, cudaStream_t s)
{
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, k_split);
    dim3 block(W * 32, 1, 1);
    int smem_bytes = 2 * (BM + BN) * (BK + 16);
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fp8_gemm_splitk_partial_kernel<BM, BN, BK, W>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp8_gemm_splitk_partial_kernel<BM, BN, BK, W><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<float*>(scratch),
        M, N, K, k_split);
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp8_gemm_splitk_reduce_kernel<<<blocks, threads, 0, s>>>(
        reinterpret_cast<const float*>(scratch),
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, k_split, alpha);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
}

}  // namespace

#define DEFINE(NAME, BM, BN, BK, W)                                              \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,           \
           int k_split, float alpha, void* scratch, cudaStream_t stream) {       \
    return launch_<BM, BN, BK, W>(A, B, D, M, N, K, k_split, alpha,              \
                                   scratch, stream);                              \
  }

DEFINE(splitk_fp8_gemm_16x64x128_w4,  16,  64, 128, 4)
DEFINE(splitk_fp8_gemm_16x64x256_w4,  16,  64, 256, 4)
DEFINE(splitk_fp8_gemm_32x64x128_w4,  32,  64, 128, 4)

#undef DEFINE

}  // namespace smallM_splitk
}  // namespace gemm
}  // namespace flash_rt
