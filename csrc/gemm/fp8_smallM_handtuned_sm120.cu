// SPDX-License-Identifier: Apache-2.0
//
// Hand-tuned FP8 e4m3 -> BF16 GEMM for sm_120a small-M motus shapes.
// Inline-PTX m16n8k32 mma + 2-stage cp.async pipeline, no cutlass collective
// builder overhead. Modeled after V5split kernel_A pattern; epilogue is just
// alpha * acc -> BF16 (no bias / GELU / quant).
//
// Motivation: cutlass scaffold has ~8 us launch/setup floor on sm_120 for
// small-M kernels, even with smallest tiles. cuBLASLt nvjet ~5 us. To break
// below cuBLASLt, must avoid the scaffold entirely.
//
// Per-tensor scale (A_scale, W_scale as float scalars folded into alpha).
// Returns 0 on success.

#include "fp8_smallM_handtuned_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace smallM_hand {

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

// Generic FP8 GEMM kernel parameterized on tile shape + pipeline stages.
//  - A: [M, K] row-major FP8 e4m3
//  - B: [N, K] row-major FP8 e4m3  (= W.T col-major layout)
//  - D: [M, N] row-major BF16
//  - alpha = a_scale * w_scale (per-tensor)
//  - STAGES = pipeline depth (2 or 3)
//  - MIN_BLOCKS_PER_SM = launch_bounds hint (1 for big-smem variants)
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS,
          int STAGES = 2, int MIN_BLOCKS_PER_SM = 4>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    __nv_bfloat16* __restrict__ D,
    int M, int N, int K,
    float alpha)
{
    static_assert(BLOCK_K % 32 == 0, "BLOCK_K must be multiple of 32");
    static_assert(BLOCK_N % 8 == 0,  "BLOCK_N must be multiple of 8");
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be multiple of 16 (mma m=16)");
    static_assert((BLOCK_N / 8) % NUM_WARPS == 0, "N-atoms must split evenly across warps");

    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int M_ATOMS    = BLOCK_M / 16;        // m-atom rows per CTA
    constexpr int N_ATOMS    = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int K_ATOMS    = BLOCK_K / 32;
    constexpr int SMEM_K_PAD = BLOCK_K + 16;        // +16 byte padding to avoid bank conflict

    // smem layout: [stage][row][col_padded]; STAGES stages.
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

    auto issue_load = [&](int stage, int k_base) {
        // Load A [BLOCK_M, BLOCK_K] FP8 = BLOCK_M * BLOCK_K bytes.
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
        // Load B [BLOCK_N, BLOCK_K] FP8 (B is [N, K] row-major).
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

    // Per-warp accumulators: M_ATOMS * N_ATOMS_PW * 4 fp32
    float acc[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[mi][ni][j] = 0.0f;
        }
    }

    // Prefetch STAGES-1 chunks before main loop (deep pipeline).
    const int K_ITERS = (K + BLOCK_K - 1) / BLOCK_K;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        int kb = s * BLOCK_K;
        if (kb < K) issue_load(s, kb);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    int compute_stage = 0;
    for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
        int k_base = k_iter * BLOCK_K;
        // Issue next load STAGES-1 ahead.
        int issue_iter = k_iter + (STAGES - 1);
        int issue_stage = issue_iter % STAGES;
        if (issue_iter < K_ITERS) issue_load(issue_stage, issue_iter * BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
        // Wait until STAGES-1 prior loads are still in flight, current ready.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
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
        compute_stage = (compute_stage + 1) % STAGES;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // Epilogue: alpha * acc -> BF16 -> HBM.
    // m16n8 layout: thread (h, l): rows {h, h+8}, cols {2*l, 2*l+1}.
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
                    float v = acc[mi][ni][j] * alpha;
                    D[row * N + col] = __float2bfloat16(v);
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int W, int STAGES = 2, int MIN_BLK = 4>
int launch_(const void* A, const void* B, void* D,
            int M, int N, int K, float alpha, cudaStream_t s)
{
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(W * 32, 1, 1);
    int smem_bytes = STAGES * (BM + BN) * (BK + 16);
    // sm_120 default dynamic smem is 48 KB; opt-in to higher (up to ~228 KB).
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fp8_gemm_kernel<BM, BN, BK, W, STAGES, MIN_BLK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp8_gemm_kernel<BM, BN, BK, W, STAGES, MIN_BLK><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, K, alpha);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
}

}  // namespace

// Variant instantiations.
#define DEFINE(NAME, BM, BN, BK, W, S)                                           \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,           \
           float alpha, cudaStream_t stream) {                                   \
    return launch_<BM, BN, BK, W, S, 4>(A, B, D, M, N, K, alpha, stream);        \
  }
// Big-smem variant — uses MIN_BLOCKS_PER_SM=1 to relax register pressure.
#define DEFINE_BIG(NAME, BM, BN, BK, W, S)                                       \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,           \
           float alpha, cudaStream_t stream) {                                   \
    return launch_<BM, BN, BK, W, S, 1>(A, B, D, M, N, K, alpha, stream);        \
  }

// 2-stage pipeline baseline.
DEFINE(fp8_gemm_16x64x128_w4,   16,  64, 128, 4, 2)
DEFINE(fp8_gemm_16x128x128_w4,  16, 128, 128, 4, 2)
DEFINE(fp8_gemm_16x256x128_w8,  16, 256, 128, 8, 2)
DEFINE(fp8_gemm_32x64x128_w4,   32,  64, 128, 4, 2)
DEFINE(fp8_gemm_32x128x128_w4,  32, 128, 128, 4, 2)
DEFINE(fp8_gemm_32x128x128_w8,  32, 128, 128, 8, 2)

// 3-stage pipeline (better cp.async overlap).
DEFINE(fp8_gemm_16x64x128_w4_s3,   16,  64, 128, 4, 3)
DEFINE(fp8_gemm_16x128x128_w4_s3,  16, 128, 128, 4, 3)
DEFINE(fp8_gemm_32x64x128_w4_s3,   32,  64, 128, 4, 3)
DEFINE(fp8_gemm_32x128x128_w4_s3,  32, 128, 128, 4, 3)

// BLOCK_K=256 (fewer K-iters, bigger cp.async chunks per iter).
DEFINE(fp8_gemm_16x64x256_w4,   16,  64, 256, 4, 2)
DEFINE(fp8_gemm_16x128x256_w4,  16, 128, 256, 4, 2)
DEFINE(fp8_gemm_32x64x256_w4,   32,  64, 256, 4, 2)
DEFINE(fp8_gemm_32x128x256_w4,  32, 128, 256, 4, 2)

// Wider BLOCK_N for big-N shapes (action_qkv, und_qkv: N=9216).
DEFINE(fp8_gemm_16x192x128_w4,  16, 192, 128, 4, 2)
DEFINE(fp8_gemm_16x192x128_w8,  16, 192, 128, 8, 2)
DEFINE(fp8_gemm_32x192x128_w4,  32, 192, 128, 4, 2)

// 4-stage pipeline.
DEFINE(fp8_gemm_16x64x128_w4_s4,   16,  64, 128, 4, 4)
DEFINE(fp8_gemm_32x64x128_w4_s4,   32,  64, 128, 4, 4)

// Wider BLOCK_N=384 (needs N % 384, 8-warp config).
DEFINE(fp8_gemm_16x384x128_w8,     16, 384, 128, 8, 2)
DEFINE(fp8_gemm_32x384x128_w8,     32, 384, 128, 8, 2)

// More warps, smaller BLOCK_N (more N-tiles parallelism per CTA).
DEFINE(fp8_gemm_16x64x128_w8,      16,  64, 128, 8, 2)
DEFINE(fp8_gemm_32x64x128_w8,      32,  64, 128, 8, 2)

// 32x64x128 with 8-stage pipeline (deep cp.async overlap for K-bound shapes).
DEFINE(fp8_gemm_32x64x128_w4_s5,   32,  64, 128, 4, 5)

// BLOCK_K=64 variants — better pipeline overlap for K-small shapes (K=512).
DEFINE(fp8_gemm_16x64x64_w4,       16,  64,  64, 4, 2)
DEFINE(fp8_gemm_16x128x64_w4,      16, 128,  64, 4, 2)
DEFINE(fp8_gemm_32x64x64_w4,       32,  64,  64, 4, 2)
DEFINE(fp8_gemm_32x128x64_w4,      32, 128,  64, 4, 2)
DEFINE(fp8_gemm_16x64x64_w4_s3,    16,  64,  64, 4, 3)
DEFINE(fp8_gemm_16x64x64_w4_s4,    16,  64,  64, 4, 4)

// Big-smem (BLOCK_N=384/512) — MIN_BLOCKS_PER_SM=1 relaxes register pressure.
// Targets multi-wave shapes (und_qkv: 9216 N) to reduce wave count.
DEFINE_BIG(fp8_gemm_16x384x128_w4_big,  16, 384, 128, 4, 2)
DEFINE_BIG(fp8_gemm_32x384x128_w4_big,  32, 384, 128, 4, 2)
DEFINE_BIG(fp8_gemm_16x512x128_w8_big,  16, 512, 128, 8, 2)
DEFINE_BIG(fp8_gemm_16x256x128_w4_big,  16, 256, 128, 4, 2)
DEFINE_BIG(fp8_gemm_32x256x128_w4_big,  32, 256, 128, 4, 2)

// BLOCK_M=64 / 128 variants — for M=138 shapes to drop wave count to 1.
// und_qkv (M=138 N=9216): BLOCK_M=64 -> 3 M-tiles, BLOCK_M=128 -> 2 M-tiles.
// Combined with BLOCK_N=128: 216 / 144 total CTAs => 1-1.3 waves on 170 SMs.
DEFINE(fp8_gemm_64x64x128_w4,    64,  64, 128, 4, 2)
DEFINE(fp8_gemm_64x128x128_w4,   64, 128, 128, 4, 2)
DEFINE(fp8_gemm_64x128x128_w8,   64, 128, 128, 8, 2)
DEFINE(fp8_gemm_128x64x128_w4,  128,  64, 128, 4, 2)
DEFINE(fp8_gemm_128x128x128_w4, 128, 128, 128, 4, 2)
DEFINE(fp8_gemm_128x128x128_w8, 128, 128, 128, 8, 2)
DEFINE_BIG(fp8_gemm_64x256x128_w4_big,  64, 256, 128, 4, 2)
DEFINE_BIG(fp8_gemm_64x256x128_w8_big,  64, 256, 128, 8, 2)
DEFINE_BIG(fp8_gemm_128x256x128_w8_big, 128, 256, 128, 8, 2)

#undef DEFINE
#undef DEFINE_BIG

}  // namespace smallM_hand
}  // namespace gemm
}  // namespace flash_rt
