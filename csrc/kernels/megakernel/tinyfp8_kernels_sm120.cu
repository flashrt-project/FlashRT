// Tile-templated FP8 GEMM for sm_120, multi-shape support.
// Variants:
//   M8_N128_K64   : action FFN_up  (M=8, K=1024, N=4096) — 32 CTAs
//   M8_N64_K64    : action FFN_dn  (M=8, K=4096, N=1024) — 16 CTAs
//   M8_N32_K64    : test more CTAs                       — 32 CTAs
//   M144_N128_K64 : und FFN  (M=138, ...)                — n_tiles
//
// Designed to be competitive with cuBLASLt nvjet at small M on sm_120.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tinyfp8v2 {

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

__device__ __forceinline__ void cp_async_16(
    uint32_t smem_int_ptr, const uint8_t* src)
{
    int src_bytes = (src == nullptr) ? 0 : 16;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ uint32_t to_smem_int(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Template: tile parameters compile-time configurable.
// Constraints:
//   BLOCK_N % 8 == 0 (mma m16n8 atom)
//   BLOCK_K % 32 == 0 (mma m16n8k32 atom)
//   BLOCK_M divisible by 16 (mma m16 atom; pad zeros if M < BLOCK_M)
//   N_ATOMS = BLOCK_N / 8 must be divisible by NUM_WARPS (atoms per warp)
//   STAGES: cp.async stages, 2 default, 3 for longer K hiding
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS, int STAGES = 2>
__device__ __forceinline__ void cta_gemm_fp8(
    const __nv_fp8_e4m3* __restrict__ A_fp8,
    const __nv_fp8_e4m3* __restrict__ B_fp8,
    __nv_bfloat16* __restrict__ D_bf16,
    int M, int N, int K,
    int m_base, int n_base,
    float alpha)
{
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int N_ATOMS    = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int K_ATOMS    = BLOCK_K / 32;
    constexpr int SMEM_K_PAD = BLOCK_K + 16;
    constexpr int M_ROWS_AT  = (BLOCK_M < 16) ? 16 : BLOCK_M;  // pad to atom

    __shared__ __align__(16) uint8_t A_smem[STAGES][M_ROWS_AT * SMEM_K_PAD];
    __shared__ __align__(16) uint8_t B_smem[STAGES][BLOCK_N * SMEM_K_PAD];

    const int t       = threadIdx.x;
    const int warp_id = t / 32;
    const int lane    = t % 32;
    const int l       = lane % 4;
    const int h       = lane / 4;

    auto issue_load = [&](int stage, int k_base) {
        // A: M_ROWS_AT × BLOCK_K bytes. Each thread loads 16 bytes (cp.async.16).
        // Row-major: each row needs BLOCK_K/16 = 4 threads.
        constexpr int A_ROWS_PER_ITER = THREADS / (BLOCK_K / 16);
        constexpr int A_ITERS = (M_ROWS_AT + A_ROWS_PER_ITER - 1) / A_ROWS_PER_ITER;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            int row_a = idx / (BLOCK_K / 16);
            int koff_a = (idx & (BLOCK_K/16 - 1)) * 16;
            if (row_a < M_ROWS_AT) {
                const uint8_t* a_src = nullptr;
                int m_glob = m_base + row_a;
                int k_glob = k_base + koff_a;
                if (m_glob < M && k_glob < K) {
                    a_src = reinterpret_cast<const uint8_t*>(
                        &A_fp8[m_glob * K + k_glob]);
                }
                cp_async_16(
                    to_smem_int(&A_smem[stage][row_a * SMEM_K_PAD + koff_a]),
                    a_src);
            }
        }
        // B: BLOCK_N × BLOCK_K bytes
        constexpr int B_ITERS = (BLOCK_N * BLOCK_K / 16 + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            int row_b = idx / (BLOCK_K / 16);
            int koff_b = (idx & (BLOCK_K/16 - 1)) * 16;
            if (row_b < BLOCK_N) {
                const uint8_t* b_src = nullptr;
                int n_glob = n_base + row_b;
                int k_glob = k_base + koff_b;
                if (n_glob < N && k_glob < K) {
                    b_src = reinterpret_cast<const uint8_t*>(
                        &B_fp8[n_glob * K + k_glob]);
                }
                cp_async_16(
                    to_smem_int(&B_smem[stage][row_b * SMEM_K_PAD + koff_b]),
                    b_src);
            }
        }
    };

    float acc[N_ATOMS_PW][4] = {0};

    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;
    for (int k_base = 0; k_base < K; k_base += BLOCK_K) {
        int next_stage = compute_stage ^ 1;
        if (k_base + BLOCK_K < K) {
            issue_load(next_stage, k_base + BLOCK_K);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        #pragma unroll
        for (int k_iter = 0; k_iter < K_ATOMS; ++k_iter) {
            int kA0 = k_iter * 32 + 4 * l;
            int kA2 = k_iter * 32 + 4 * l + 16;

            int rA0 = h;
            int rA1 = h + 8;
            uint32_t A0 = *reinterpret_cast<const uint32_t*>(
                &A_smem[compute_stage][rA0 * SMEM_K_PAD + kA0]);
            uint32_t A1 = *reinterpret_cast<const uint32_t*>(
                &A_smem[compute_stage][rA1 * SMEM_K_PAD + kA0]);
            uint32_t A2 = *reinterpret_cast<const uint32_t*>(
                &A_smem[compute_stage][rA0 * SMEM_K_PAD + kA2]);
            uint32_t A3 = *reinterpret_cast<const uint32_t*>(
                &A_smem[compute_stage][rA1 * SMEM_K_PAD + kA2]);

            #pragma unroll
            for (int n_atom = 0; n_atom < N_ATOMS_PW; ++n_atom) {
                int co_n = warp_id * N_ATOMS_PW * 8 + n_atom * 8 + h;
                uint32_t B0 = *reinterpret_cast<const uint32_t*>(
                    &B_smem[compute_stage][co_n * SMEM_K_PAD + kA0]);
                uint32_t B1 = *reinterpret_cast<const uint32_t*>(
                    &B_smem[compute_stage][co_n * SMEM_K_PAD + kA2]);
                mma_m16n8k32_e4m3(
                    acc[n_atom][0], acc[n_atom][1], acc[n_atom][2], acc[n_atom][3],
                    A0, A1, A2, A3, B0, B1);
            }
        }
        compute_stage = next_stage;
    }

    asm volatile("cp.async.wait_all;\n" ::);

    // Epilogue
    #pragma unroll
    for (int n_atom = 0; n_atom < N_ATOMS_PW; ++n_atom) {
        int co_pair = n_base + warp_id * N_ATOMS_PW * 8 + n_atom * 8 + 2 * l;
        int row0    = m_base + h;
        int row1    = m_base + h + 8;

        if (row0 < M && co_pair + 1 < N) {
            __nv_bfloat162 packAB;
            packAB.x = __float2bfloat16(acc[n_atom][0] * alpha);
            packAB.y = __float2bfloat16(acc[n_atom][1] * alpha);
            *reinterpret_cast<__nv_bfloat162*>(
                &D_bf16[row0 * N + co_pair]) = packAB;
        }
        if (row1 < M && co_pair + 1 < N) {
            __nv_bfloat162 packCD;
            packCD.x = __float2bfloat16(acc[n_atom][2] * alpha);
            packCD.y = __float2bfloat16(acc[n_atom][3] * alpha);
            *reinterpret_cast<__nv_bfloat162*>(
                &D_bf16[row1 * N + co_pair]) = packCD;
        }
    }
}

// Per-config kernel
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32, 8)
gemm_kernel(
    const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
    __nv_bfloat16* D, int M, int N, int K, float alpha)
{
    int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    int m_idx = blockIdx.y;
    int n_idx = blockIdx.x;
    cta_gemm_fp8<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS>(
        A, B, D, M, N, K,
        m_idx * BLOCK_M, n_idx * BLOCK_N, alpha);
}

// extern launchers per config
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS>
int launch_template(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t stream)
{
    int m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(n_tiles, m_tiles);
    dim3 block(NUM_WARPS * 32);
    gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, K, alpha);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (100 + (int)e);
}

// Variants we care about for action / und FFN
extern "C" {

// action FFN_up: M=8, K=1024, N=4096 — large N → many tiles even at BLOCK_N=128
int gemm_M8_N128_K64(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 128, 64, 4>(A, B, D, M, N, K, a, s);
}

// action FFN_dn: M=8, K=4096, N=1024 — smaller N → need smaller BLOCK_N for SM saturation
int gemm_M8_N64_K64(const void* A, const void* B, void* D,
                    int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 64, 64, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M8_N32_K64(const void* A, const void* B, void* D,
                    int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 32, 64, 4>(A, B, D, M, N, K, a, s);
}

// Even smaller BLOCK_N to push FFN_dn toward BW floor
// (N=1024/16 = 64 CTAs × M-tiles)
int gemm_M8_N16_K64(const void* A, const void* B, void* D,
                    int M, int N, int K, float a, cudaStream_t s) {
    // BLOCK_N=16 = 2 N-atoms; with NUM_WARPS=4 → 0.5 atoms/warp not OK
    // Use NUM_WARPS=2 → 1 atom/warp
    return launch_template<8, 16, 64, 2>(A, B, D, M, N, K, a, s);
}

// Larger BLOCK_K for FFN_dn (K=4096) — fewer K iters, more cp.async amortization
int gemm_M8_N32_K128(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 32, 128, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M8_N32_K256(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 32, 256, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M8_N64_K128(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 64, 128, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M8_N16_K128(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 16, 128, 2>(A, B, D, M, N, K, a, s);
}

int gemm_M8_N32_K512(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 32, 512, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M8_N16_K256(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 16, 256, 2>(A, B, D, M, N, K, a, s);
}

int gemm_M8_N16_K512(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<8, 16, 512, 2>(A, B, D, M, N, K, a, s);
}

// Larger BLOCK_M variants for M=138 und expert (reduce m-tile redundant B loads)
int gemm_M16_N128_K64(const void* A, const void* B, void* D,
                      int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<16, 128, 64, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M16_N64_K64(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<16, 64, 64, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M16_N32_K64(const void* A, const void* B, void* D,
                     int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<16, 32, 64, 4>(A, B, D, M, N, K, a, s);
}

// Stage3 action branch: M=21. BLOCK_M=32 avoids reloading B across
// three M8 tiles while keeping the N tile narrow enough for occupancy.
int gemm_M32_N32_K128(const void* A, const void* B, void* D,
                      int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<32, 32, 128, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M32_N32_K512(const void* A, const void* B, void* D,
                      int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<32, 32, 512, 4>(A, B, D, M, N, K, a, s);
}

// und FFN: M=138 — pad to 144 (multi 16)
int gemm_M144_N128_K64(const void* A, const void* B, void* D,
                       int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<144, 128, 64, 4>(A, B, D, M, N, K, a, s);
}

int gemm_M144_N64_K64(const void* A, const void* B, void* D,
                      int M, int N, int K, float a, cudaStream_t s) {
    return launch_template<144, 64, 64, 4>(A, B, D, M, N, K, a, s);
}

}  // extern C

}  // namespace tinyfp8v2

namespace tinyfp8_3stage {

__device__ __forceinline__ void mma_m16n8k32_e4m3_3s(
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

__device__ __forceinline__ void cp_async_16_3s(
    uint32_t smem_int_ptr, const uint8_t* src)
{
    int src_bytes = (src == nullptr) ? 0 : 16;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ uint32_t to_smem_int_3s(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32, 8)
gemm_kernel_3stage(
    const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
    __nv_bfloat16* D, int M, int N, int K, float alpha)
{
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int N_ATOMS    = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int K_ATOMS    = BLOCK_K / 32;
    constexpr int SMEM_K_PAD = BLOCK_K + 16;
    constexpr int M_ROWS_AT  = (BLOCK_M < 16) ? 16 : BLOCK_M;
    constexpr int STAGES     = 3;

    __shared__ __align__(16) uint8_t A_smem[STAGES][M_ROWS_AT * SMEM_K_PAD];
    __shared__ __align__(16) uint8_t B_smem[STAGES][BLOCK_N * SMEM_K_PAD];

    const int t       = threadIdx.x;
    const int warp_id = t / 32;
    const int lane    = t % 32;
    const int l       = lane % 4;
    const int h       = lane / 4;

    const int n_base = blockIdx.x * BLOCK_N;
    const int m_base = blockIdx.y * BLOCK_M;

    auto issue_load = [&](int stage, int k_base) {
        constexpr int A_ROWS_PER_ITER = THREADS / (BLOCK_K / 16);
        constexpr int A_ITERS =
            (M_ROWS_AT + A_ROWS_PER_ITER - 1) / A_ROWS_PER_ITER;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            int row_a = idx / (BLOCK_K / 16);
            int koff_a = (idx & (BLOCK_K / 16 - 1)) * 16;
            if (row_a < M_ROWS_AT) {
                const uint8_t* a_src = nullptr;
                int m_glob = m_base + row_a;
                int k_glob = k_base + koff_a;
                if (m_glob < M && k_glob < K) {
                    a_src = reinterpret_cast<const uint8_t*>(
                        &A[m_glob * K + k_glob]);
                }
                cp_async_16_3s(
                    to_smem_int_3s(&A_smem[stage][row_a * SMEM_K_PAD + koff_a]),
                    a_src);
            }
        }
        constexpr int B_ITERS =
            (BLOCK_N * BLOCK_K / 16 + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            int row_b = idx / (BLOCK_K / 16);
            int koff_b = (idx & (BLOCK_K / 16 - 1)) * 16;
            if (row_b < BLOCK_N) {
                const uint8_t* b_src = nullptr;
                int n_glob = n_base + row_b;
                int k_glob = k_base + koff_b;
                if (n_glob < N && k_glob < K) {
                    b_src = reinterpret_cast<const uint8_t*>(
                        &B[n_glob * K + k_glob]);
                }
                cp_async_16_3s(
                    to_smem_int_3s(&B_smem[stage][row_b * SMEM_K_PAD + koff_b]),
                    b_src);
            }
        }
    };

    float acc[N_ATOMS_PW][4] = {0};
    int n_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    if (n_k_tiles >= 2) {
        issue_load(1, BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    int stage = 0;
    for (int k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        int prefetch_tile = k_tile + 2;
        int prefetch_stage = (stage + 2) % STAGES;
        if (prefetch_tile < n_k_tiles) {
            issue_load(prefetch_stage, prefetch_tile * BLOCK_K);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        asm volatile("cp.async.wait_group 2;\n" ::);
        __syncthreads();

        #pragma unroll
        for (int k_iter = 0; k_iter < K_ATOMS; ++k_iter) {
            int kA0 = k_iter * 32 + 4 * l;
            int kA2 = k_iter * 32 + 4 * l + 16;
            int rA0 = h;
            int rA1 = h + 8;
            uint32_t A0 = *reinterpret_cast<const uint32_t*>(
                &A_smem[stage][rA0 * SMEM_K_PAD + kA0]);
            uint32_t A1 = *reinterpret_cast<const uint32_t*>(
                &A_smem[stage][rA1 * SMEM_K_PAD + kA0]);
            uint32_t A2 = *reinterpret_cast<const uint32_t*>(
                &A_smem[stage][rA0 * SMEM_K_PAD + kA2]);
            uint32_t A3 = *reinterpret_cast<const uint32_t*>(
                &A_smem[stage][rA1 * SMEM_K_PAD + kA2]);

            #pragma unroll
            for (int n_atom = 0; n_atom < N_ATOMS_PW; ++n_atom) {
                int co_n = warp_id * N_ATOMS_PW * 8 + n_atom * 8 + h;
                uint32_t B0 = *reinterpret_cast<const uint32_t*>(
                    &B_smem[stage][co_n * SMEM_K_PAD + kA0]);
                uint32_t B1 = *reinterpret_cast<const uint32_t*>(
                    &B_smem[stage][co_n * SMEM_K_PAD + kA2]);
                mma_m16n8k32_e4m3_3s(
                    acc[n_atom][0], acc[n_atom][1],
                    acc[n_atom][2], acc[n_atom][3],
                    A0, A1, A2, A3, B0, B1);
            }
        }
        stage = (stage + 1) % STAGES;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    #pragma unroll
    for (int n_atom = 0; n_atom < N_ATOMS_PW; ++n_atom) {
        int co_pair = n_base + warp_id * N_ATOMS_PW * 8 + n_atom * 8 + 2 * l;
        int row0 = m_base + h;
        int row1 = m_base + h + 8;
        if (row0 < M && co_pair + 1 < N) {
            __nv_bfloat162 packAB;
            packAB.x = __float2bfloat16(acc[n_atom][0] * alpha);
            packAB.y = __float2bfloat16(acc[n_atom][1] * alpha);
            *reinterpret_cast<__nv_bfloat162*>(&D[row0 * N + co_pair]) = packAB;
        }
        if (row1 < M && co_pair + 1 < N) {
            __nv_bfloat162 packCD;
            packCD.x = __float2bfloat16(acc[n_atom][2] * alpha);
            packCD.y = __float2bfloat16(acc[n_atom][3] * alpha);
            *reinterpret_cast<__nv_bfloat162*>(&D[row1 * N + co_pair]) = packCD;
        }
    }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS>
int launch3(const void* A, const void* B, void* D,
            int M, int N, int K, float alpha, cudaStream_t stream)
{
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_WARPS * 32);
    gemm_kernel_3stage<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS>
        <<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const __nv_fp8_e4m3*>(B),
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, K, alpha);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (100 + (int)e);
}

int gemm3_M16_N64_K128(const void* A, const void* B, void* D,
                       int M, int N, int K, float alpha, cudaStream_t s) {
    return launch3<16, 64, 128, 4>(A, B, D, M, N, K, alpha, s);
}

}  // namespace tinyfp8_3stage

namespace flash_rt {
namespace megakernel {

// Five 2-stage tile variants used by tiny_fp8_dispatch at motus production.
// Each: D = (A_fp8 @ B_fp8^T) * alpha → bf16, where A is (M, K) and B is
// stored (N, K) row-major.

int tinyfp8_gemm_M8_N32_K128_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s) {
  return tinyfp8v2::gemm_M8_N32_K128(A, B, D, M, N, K, alpha, s);
}
int tinyfp8_gemm_M8_N32_K256_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s) {
  return tinyfp8v2::gemm_M8_N32_K256(A, B, D, M, N, K, alpha, s);
}
int tinyfp8_gemm_M8_N32_K512_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s) {
  return tinyfp8v2::gemm_M8_N32_K512(A, B, D, M, N, K, alpha, s);
}
int tinyfp8_gemm_M16_N32_K64_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s) {
  return tinyfp8v2::gemm_M16_N32_K64(A, B, D, M, N, K, alpha, s);
}
int tinyfp8_gemm_M16_N64_K64_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s) {
  return tinyfp8v2::gemm_M16_N64_K64(A, B, D, M, N, K, alpha, s);
}
int tinyfp8_gemm_M32_N32_K128_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s) {
  return tinyfp8v2::gemm_M32_N32_K128(A, B, D, M, N, K, alpha, s);
}
int tinyfp8_gemm_M32_N32_K512_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s) {
  return tinyfp8v2::gemm_M32_N32_K512(A, B, D, M, N, K, alpha, s);
}
int tinyfp8_gemm3_M16_N64_K128_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s) {
  return tinyfp8_3stage::gemm3_M16_N64_K128(A, B, D, M, N, K, alpha, s);
}

}  // namespace megakernel
}  // namespace flash_rt
