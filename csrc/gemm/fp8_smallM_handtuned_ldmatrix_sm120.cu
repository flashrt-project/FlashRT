// SPDX-License-Identifier: Apache-2.0
//
// Hand-tuned FP8 e4m3 -> BF16 GEMM v2 for sm_120a small-M motus shapes.
// Adds 128B swizzle smem layout + ldmatrix.x4.m8n8.b16 reads to clear the
// 4-way smem bank conflict that bottlenecks v1 (`fp8_smallM_handtuned`).
//
// Restrictions for this version:
//   - BLOCK_K = 128 (natural 128B swizzle row stride)
//   - N_ATOMS_PER_WARP must be even (paired into one ldmatrix.x4 each)
//
// MMA path identical to v1 (inline PTX m16n8k32 e4m3 e4m3 f32).

#include "fp8_smallM_handtuned_ldmatrix_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace smallM_ld {

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

__device__ __forceinline__ void ldmatrix_x4_b16(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    uint32_t smem_addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(smem_addr));
}

// 128B swizzle: byte_addr_swizzled = row*128 + (chunk16 XOR (row & 7))*16
// where chunk16 = byte_col / 16 in [0, 7] (one chunk = 16 bytes).
// Applied identically on cp.async store and ldmatrix load to round-trip cleanly.

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS,
          int STAGES = 2, int MIN_BLOCKS_PER_SM = 4>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp8_gemm_ld_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    __nv_bfloat16* __restrict__ D,
    int M, int N, int K, float alpha)
{
    static_assert(BLOCK_K == 64 || BLOCK_K == 128 || BLOCK_K == 256
                  || BLOCK_K == 512,
                  "BLOCK_K must be 64/128/256/512 (BK=512 uses repeated 128B swizzle period)");
    constexpr int NUM_CHUNKS_PER_ROW = BLOCK_K / 16;       // 4, 8, 16, or 32
    // Swizzle mask: 128B period (8 chunks). For BK=256/512 the pattern
    // repeats every 8 chunks; ldmatrix.x4 reads within one period at a
    // time so bank-conflict-free remains.
    constexpr int SWIZZLE_MASK = (NUM_CHUNKS_PER_ROW <= 8)
                                 ? (NUM_CHUNKS_PER_ROW - 1) : 7;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int M_ATOMS = BLOCK_M / 16;
    constexpr int N_ATOMS = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be multiple of 16");
    static_assert(BLOCK_N % 8 == 0,  "BLOCK_N must be multiple of 8");
    static_assert(N_ATOMS_PW >= 2 && N_ATOMS_PW % 2 == 0,
                  "N atoms per warp must be even >=2 (paired for ldmatrix.x4)");
    constexpr int N_PAIRS_PW = N_ATOMS_PW / 2;
    constexpr int K_ATOMS = BLOCK_K / 32;  // = 4 for BLOCK_K=128
    constexpr int A_TILE_BYTES = BLOCK_M * BLOCK_K;
    constexpr int B_TILE_BYTES = BLOCK_N * BLOCK_K;

    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint8_t* A_smem = smem_raw;
    uint8_t* B_smem = A_smem + STAGES * A_TILE_BYTES;

    const int cta_m  = blockIdx.x;
    const int cta_n  = blockIdx.y;
    const int m_base = cta_m * BLOCK_M;
    const int n_base = cta_n * BLOCK_N;

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;

    // Lane partition for ldmatrix.x4 addressing (lane -> fragment).
    const int frag_group = lane / 8;      // 0..3 (TL,TR,BL,BR per ldmatrix)
    const int row_in_frag = lane % 8;     // row within fragment 0..7
    const int row_block = frag_group / 2; // top(0) / bot(1)
    const int col_block = frag_group % 2; // left(0) / right(1)

    // Lane partition for mma epilogue write.
    const int h = lane / 4;  // 0..7
    const int l = lane % 4;  // 0..3

    auto issue_load = [&](int stage, int k_base) {
        // A tile: BLOCK_M rows x BLOCK_K bytes, each thread issues 16-byte chunks.
        constexpr int A_CHUNKS = BLOCK_M * (BLOCK_K / 16);
        constexpr int A_ITERS  = (A_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= A_CHUNKS) break;
            int row_a   = idx / (BLOCK_K / 16);
            int chunk_a = idx % (BLOCK_K / 16);
            int m_g = m_base + row_a;
            int k_g = k_base + chunk_a * 16;
            const uint8_t* src = nullptr;
            if (m_g < M && k_g < K) {
                src = reinterpret_cast<const uint8_t*>(&A[m_g * K + k_g]);
            }
            int chunk_sw = chunk_a ^ (row_a & SWIZZLE_MASK);
            uint32_t dst = to_smem(
                &A_smem[stage * A_TILE_BYTES + row_a * BLOCK_K + chunk_sw * 16]);
            cp_async_16(dst, src);
        }
        // B tile: BLOCK_N rows x BLOCK_K bytes.
        constexpr int B_CHUNKS = BLOCK_N * (BLOCK_K / 16);
        constexpr int B_ITERS  = (B_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= B_CHUNKS) break;
            int row_b   = idx / (BLOCK_K / 16);
            int chunk_b = idx % (BLOCK_K / 16);
            int n_g = n_base + row_b;
            int k_g = k_base + chunk_b * 16;
            const uint8_t* src = nullptr;
            if (n_g < N && k_g < K) {
                src = reinterpret_cast<const uint8_t*>(&B[n_g * K + k_g]);
            }
            int chunk_sw = chunk_b ^ (row_b & SWIZZLE_MASK);
            uint32_t dst = to_smem(
                &B_smem[stage * B_TILE_BYTES + row_b * BLOCK_K + chunk_sw * 16]);
            cp_async_16(dst, src);
        }
    };

    // Per-warp accumulators: M_ATOMS rows of mma x N_ATOMS_PW cols x 4 fp32.
    float acc[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi)
    #pragma unroll
    for (int ni = 0; ni < N_ATOMS_PW; ++ni)
    #pragma unroll
    for (int j = 0; j < 4; ++j) acc[mi][ni][j] = 0.0f;

    // Prefetch STAGES-1 chunks.
    const int K_ITERS = (K + BLOCK_K - 1) / BLOCK_K;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s * BLOCK_K < K) issue_load(s, s * BLOCK_K);
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

        uint8_t* A_stage = A_smem + compute_stage * A_TILE_BYTES;
        uint8_t* B_stage = B_smem + compute_stage * B_TILE_BYTES;

        // K-atom inner loop. Per k_a: ldmatrix A (per m-atom) and B (per N-pair).
        #pragma unroll
        for (int k_a = 0; k_a < K_ATOMS; ++k_a) {
            uint32_t A_regs[M_ATOMS][4];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int row_in_tile = mi * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * k_a + col_block;
                int chunk_sw = chunk ^ (row_in_tile & SWIZZLE_MASK);
                uint32_t addr = to_smem(
                    &A_stage[row_in_tile * BLOCK_K + chunk_sw * 16]);
                ldmatrix_x4_b16(
                    A_regs[mi][0], A_regs[mi][1], A_regs[mi][2], A_regs[mi][3],
                    addr);
                // ldmatrix output mapping vs mma m16n8k32 A operand:
                //   ldm d0=TL → mma a0
                //   ldm d1=TR → mma a2
                //   ldm d2=BL → mma a1
                //   ldm d3=BR → mma a3
            }

            uint32_t B_regs[N_PAIRS_PW][4];
            #pragma unroll
            for (int np = 0; np < N_PAIRS_PW; ++np) {
                int n_base_pair = warp_id * N_ATOMS_PW * 8 + np * 16;
                int n_row_in_tile = n_base_pair + row_block * 8 + row_in_frag;
                int chunk = 2 * k_a + col_block;
                int chunk_sw = chunk ^ (n_row_in_tile & SWIZZLE_MASK);
                uint32_t addr = to_smem(
                    &B_stage[n_row_in_tile * BLOCK_K + chunk_sw * 16]);
                ldmatrix_x4_b16(
                    B_regs[np][0], B_regs[np][1], B_regs[np][2], B_regs[np][3],
                    addr);
                // ldm output mapping for paired N-atoms:
                //   d0 = TL = N-atom0's b0 (rows 0-7, K-cols 0-15)
                //   d1 = TR = N-atom0's b1 (rows 0-7, K-cols 16-31)
                //   d2 = BL = N-atom1's b0 (rows 8-15, K-cols 0-15)
                //   d3 = BR = N-atom1's b1 (rows 8-15, K-cols 16-31)
            }

            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                #pragma unroll
                for (int np = 0; np < N_PAIRS_PW; ++np) {
                    int ni0 = np * 2;
                    int ni1 = np * 2 + 1;
                    // N-atom 0: B = (b0=B_regs[np][0], b1=B_regs[np][1])
                    mma_m16n8k32_e4m3(
                        acc[mi][ni0][0], acc[mi][ni0][1],
                        acc[mi][ni0][2], acc[mi][ni0][3],
                        A_regs[mi][0], A_regs[mi][2],
                        A_regs[mi][1], A_regs[mi][3],
                        B_regs[np][0], B_regs[np][1]);
                    // N-atom 1: B = (b0=B_regs[np][2], b1=B_regs[np][3])
                    mma_m16n8k32_e4m3(
                        acc[mi][ni1][0], acc[mi][ni1][1],
                        acc[mi][ni1][2], acc[mi][ni1][3],
                        A_regs[mi][0], A_regs[mi][2],
                        A_regs[mi][1], A_regs[mi][3],
                        B_regs[np][2], B_regs[np][3]);
                }
            }
        }
        compute_stage = (compute_stage + 1) % STAGES;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // Epilogue: write 4 fp32 acc per lane to D[BF16].
    // mma m16n8 output per lane (h=lane/4, l=lane%4):
    //   d0,d1 -> row h,         cols 2l, 2l+1
    //   d2,d3 -> row h+8,       cols 2l, 2l+1
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int col_base = n_base + warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            if (row0 < M) {
                if (col_base     < N)
                    D[row0 * N + col_base    ] = __float2bfloat16(acc[mi][ni][0] * alpha);
                if (col_base + 1 < N)
                    D[row0 * N + col_base + 1] = __float2bfloat16(acc[mi][ni][1] * alpha);
            }
            if (row1 < M) {
                if (col_base     < N)
                    D[row1 * N + col_base    ] = __float2bfloat16(acc[mi][ni][2] * alpha);
                if (col_base + 1 < N)
                    D[row1 * N + col_base + 1] = __float2bfloat16(acc[mi][ni][3] * alpha);
            }
        }
    }
}

template <int BM, int BN, int BK, int W, int STAGES = 2, int MIN_BLK = 4>
int launch_(const void* A, const void* B, void* D,
            int M, int N, int K, float alpha, cudaStream_t s)
{
    if (K % BK != 0) return 2;
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(W * 32, 1, 1);
    int smem_bytes = STAGES * (BM + BN) * BK;
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fp8_gemm_ld_kernel<BM, BN, BK, W, STAGES, MIN_BLK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp8_gemm_ld_kernel<BM, BN, BK, W, STAGES, MIN_BLK><<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, K, alpha);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
}

}  // namespace

#define DEFINE(NAME, BM, BN, BK, W, S)                                           \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,           \
           float alpha, cudaStream_t stream) {                                   \
    return launch_<BM, BN, BK, W, S, 4>(A, B, D, M, N, K, alpha, stream);        \
  }
#define DEFINE_BIG(NAME, BM, BN, BK, W, S)                                       \
  int NAME(const void* A, const void* B, void* D, int M, int N, int K,           \
           float alpha, cudaStream_t stream) {                                   \
    return launch_<BM, BN, BK, W, S, 1>(A, B, D, M, N, K, alpha, stream);        \
  }

DEFINE(ld_fp8_gemm_16x64x128_w4,    16,  64, 128, 4, 2)
DEFINE(ld_fp8_gemm_16x128x128_w4,   16, 128, 128, 4, 2)
DEFINE(ld_fp8_gemm_16x256x128_w8,   16, 256, 128, 8, 2)
DEFINE(ld_fp8_gemm_32x64x128_w4,    32,  64, 128, 4, 2)
DEFINE(ld_fp8_gemm_32x128x128_w4,   32, 128, 128, 4, 2)
DEFINE(ld_fp8_gemm_32x128x128_w8,   32, 128, 128, 8, 2)

DEFINE(ld_fp8_gemm_16x64x128_w4_s3, 16,  64, 128, 4, 3)
DEFINE(ld_fp8_gemm_16x128x128_w4_s3,16, 128, 128, 4, 3)
DEFINE(ld_fp8_gemm_32x64x128_w4_s3, 32,  64, 128, 4, 3)
DEFINE(ld_fp8_gemm_32x128x128_w4_s3,32, 128, 128, 4, 3)

DEFINE(ld_fp8_gemm_16x192x128_w4,   16, 192, 128, 4, 2)
DEFINE(ld_fp8_gemm_32x192x128_w4,   32, 192, 128, 4, 2)

DEFINE(ld_fp8_gemm_16x64x128_w4_s4, 16,  64, 128, 4, 4)
DEFINE(ld_fp8_gemm_16x64x128_w4_s5, 16,  64, 128, 4, 5)
DEFINE(ld_fp8_gemm_32x64x128_w4_s4, 32,  64, 128, 4, 4)
DEFINE(ld_fp8_gemm_32x64x128_w4_s5, 32,  64, 128, 4, 5)
DEFINE(ld_fp8_gemm_16x128x128_w4_s4,16, 128, 128, 4, 4)
DEFINE(ld_fp8_gemm_32x128x128_w4_s4,32, 128, 128, 4, 4)

// BK=256 variants — large K-tile, fewer K-iters, more compute per CTA.
DEFINE(ld_fp8_gemm_16x64x256_w4,    16,  64, 256, 4, 2)
DEFINE(ld_fp8_gemm_16x128x256_w4,   16, 128, 256, 4, 2)
DEFINE(ld_fp8_gemm_32x64x256_w4,    32,  64, 256, 4, 2)
DEFINE(ld_fp8_gemm_32x128x256_w4,   32, 128, 256, 4, 2)
DEFINE(ld_fp8_gemm_16x64x256_w4_s3, 16,  64, 256, 4, 3)

// BK=64 variants — small K-tile, more K-iters, finer pipeline grain.
DEFINE(ld_fp8_gemm_16x64x64_w4,     16,  64, 64,  4, 2)
DEFINE(ld_fp8_gemm_16x128x64_w4,    16, 128, 64,  4, 2)
DEFINE(ld_fp8_gemm_32x64x64_w4,     32,  64, 64,  4, 2)
DEFINE(ld_fp8_gemm_16x64x64_w4_s3,  16,  64, 64,  4, 3)
DEFINE(ld_fp8_gemm_16x64x64_w4_s4,  16,  64, 64,  4, 4)

// und_qkv (M=188, N=9216, K=512) untried variants: bigger BM reduces
// CTA count (188/64=3 m_tiles vs 188/32=6); bigger BK reduces K_iter
// overhead; BK=512 single-iter eliminates pipeline overhead at K=512.
DEFINE(ld_fp8_gemm_64x64x128_w4,    64,  64, 128, 4, 2)
DEFINE(ld_fp8_gemm_64x128x128_w4,   64, 128, 128, 4, 2)
DEFINE(ld_fp8_gemm_64x64x256_w4,    64,  64, 256, 4, 2)
DEFINE(ld_fp8_gemm_64x128x256_w4,   64, 128, 256, 4, 2)
DEFINE(ld_fp8_gemm_64x64x256_w4_s3, 64,  64, 256, 4, 3)
DEFINE(ld_fp8_gemm_32x64x256_w4_s3, 32,  64, 256, 4, 3)
DEFINE(ld_fp8_gemm_32x128x256_w4_s3,32, 128, 256, 4, 3)
DEFINE(ld_fp8_gemm_128x64x128_w4,   128, 64, 128, 4, 2)
DEFINE(ld_fp8_gemm_128x128x128_w4,  128, 128, 128, 4, 2)

// BK=512 single-iter benched 14.35us, worse than BK=256 pipelined 12.35us.
// Kept the relaxed static_assert (BK=512 now allowed) but no variants
// instantiated — they lose to existing BK=256 cp.async pipeline.

#undef DEFINE
#undef DEFINE_BIG

}  // namespace smallM_ld
}  // namespace gemm
}  // namespace flash_rt
