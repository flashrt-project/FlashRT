// SPDX-License-Identifier: Apache-2.0
//
// Stage3 und FFN split megakernel. Same FP8 W4A8 math as V5t, but
// split into up/intermediate and down/residual kernels so M=188 avoids
// the high-smem monolithic grid barrier.
//
// Base tile pattern adapted from the V5 und FFN kernel.
//   Stage3 M=188 (capacity 192 = 12 x BLOCK_M=16)
//   K_up=512, N_up=2048
//   K_dn=2048, N_dn=512
//   BLOCK_M=16, BLOCK_N_UP=64, BLOCK_N_DN=32
//   BLOCK_K_UP=64 (8 iters), BLOCK_K_DN=128 (16 iters)
//   NUM_CTAS=384 = 12 m-tiles x 32 n-tiles for GEMM_up
//   GEMM_dn = 12 x 16 = 192 tiles
//
// Phases:
//   1. quant input (each CTA owns its M-K slice)
//   2. GEMM_up + bias + gelu + quant (fused epilogue) -> up_fp8
//   3. GEMM_dn + bias_add (fused epilogue) -> y_out

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace und_ffn_v5split_stage3 {

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
__device__ __forceinline__ float gelu_tanh(float x) {
    const float beta = 0.7978845608028654f;
    const float kappa = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + __tanhf(beta * (x + kappa * x3)));
}
__device__ __forceinline__ void grid_barrier(uint32_t* state, int n_ctas) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t my_phase = atomicAdd(&state[1], 0u);
        const uint32_t old = atomicAdd(&state[0], 1u);
        if (old + 1u == static_cast<uint32_t>(n_ctas)) {
            atomicExch(&state[0], 0u);
            atomicAdd(&state[1], 1u);
        } else {
            while (atomicAdd(&state[1], 0u) == my_phase) {
                __nanosleep(64);
            }
        }
    }
    __syncthreads();
}

// Stage3 Robotwin uses M=188, so capacity is 12 x 16 = 192 rows.
// Keep the V4b split design to avoid the high-smem monolithic grid barrier.
#ifndef V5S_BLOCK_K_UP
#define V5S_BLOCK_K_UP 64
#endif
#ifndef V5S_BLOCK_K_DN
#define V5S_BLOCK_K_DN 128
#endif
#ifndef V5S_BLOCK_N_UP
#define V5S_BLOCK_N_UP 64
#endif
#ifndef V5S_BLOCK_N_DN
#define V5S_BLOCK_N_DN 32
#endif
#ifndef V5S_NUM_WARPS
#define V5S_NUM_WARPS 4
#endif
#ifndef V5S_LAUNCH_MIN
#define V5S_LAUNCH_MIN 8
#endif

constexpr int THREADS = 128;
constexpr int BLOCK_M = 16;
constexpr int BLOCK_K_UP = V5S_BLOCK_K_UP;
constexpr int BLOCK_K_DN = V5S_BLOCK_K_DN;
constexpr int BLOCK_N_UP = V5S_BLOCK_N_UP;
constexpr int BLOCK_N_DN = V5S_BLOCK_N_DN;
constexpr int NUM_WARPS = V5S_NUM_WARPS;
constexpr int M_TILES = 12;
constexpr int NUM_CTAS = M_TILES * (2048 / BLOCK_N_UP);
constexpr int M_ROWS_AT = 16;
constexpr int SMEM_K_PAD_UP = BLOCK_K_UP + 16;
constexpr int SMEM_K_PAD_DN = BLOCK_K_DN + 16;

// -- Phase 1: quant on (M, K_up=512). NUM_CTAS CTAs distribute M x K.
__device__ __forceinline__ void phase_quant_input(
    const __nv_bfloat16* __restrict__ x_bf16,
    const __nv_bfloat16* __restrict__ inv_s,
    float act_scale,
    __nv_fp8_e4m3* __restrict__ x_fp8,
    int M, int K, int cta)
{
    // Total elements: M x K. Distribute across NUM_CTAS CTAs.
    int total = M * K;
    int per_cta = (total + NUM_CTAS - 1) / NUM_CTAS;
    int base = cta * per_cta;
    int t = threadIdx.x;
    float inv_act = 1.0f / act_scale;
    for (int idx = base + t; idx < base + per_cta && idx < total; idx += blockDim.x) {
        int m = idx / K;
        int k = idx % K;
        if (m >= M) continue;
        float xv = __bfloat162float(x_bf16[m * K + k]);
        float sv = __bfloat162float(inv_s[k]);
        float q = xv * sv * inv_act;
        q = fmaxf(-448.0f, fminf(448.0f, q));
        x_fp8[m * K + k] = __nv_fp8_e4m3(q);
    }
}

// -- Phase 2: GEMM_up + bias + gelu + quant fused epilogue --
// 12 m-tiles x 32 n-tiles = 384 tiles, each CTA does 1 tile.
__device__ __forceinline__ void phase_gemm_up_fused_eup(
    const __nv_fp8_e4m3* __restrict__ x_fp8,
    const __nv_fp8_e4m3* __restrict__ up_w_NK,
    const __nv_bfloat16* __restrict__ up_bias,
    const __nv_bfloat16* __restrict__ dn_inv_s,
    float up_alpha, float dn_act_scale_inv,
    __nv_fp8_e4m3* __restrict__ up_fp8,
    int M, int N_up, int K_up, int m_base, int n_base,
    uint8_t* A_smem, uint8_t* B_smem)
{
    constexpr int BLOCK_K = BLOCK_K_UP;
    constexpr int SMEM_K_PAD = SMEM_K_PAD_UP;
    constexpr int N_ATOMS = BLOCK_N_UP / 8;             // 8
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;     // 2
    constexpr int K_ATOMS = BLOCK_K / 32;                // 2

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;

    auto issue_load = [&](int stage, int k_base) {
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
                if (m_glob < M && k_glob < K_up) {
                    a_src = reinterpret_cast<const uint8_t*>(
                        &x_fp8[m_glob * K_up + k_glob]);
                }
                cp_async_16(
                    to_smem(&A_smem[stage * M_ROWS_AT * SMEM_K_PAD + row_a * SMEM_K_PAD + koff_a]),
                    a_src);
            }
        }
        constexpr int B_TOTAL = BLOCK_N_UP * BLOCK_K / 16;
        constexpr int B_ITERS = (B_TOTAL + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            int row_b = idx / (BLOCK_K / 16);
            int koff_b = (idx & (BLOCK_K/16 - 1)) * 16;
            if (row_b < BLOCK_N_UP) {
                const uint8_t* b_src = nullptr;
                int n_glob = n_base + row_b;
                int k_glob = k_base + koff_b;
                if (n_glob < N_up && k_glob < K_up) {
                    b_src = reinterpret_cast<const uint8_t*>(
                        &up_w_NK[n_glob * K_up + k_glob]);
                }
                cp_async_16(
                    to_smem(&B_smem[stage * BLOCK_N_UP * SMEM_K_PAD + row_b * SMEM_K_PAD + koff_b]),
                    b_src);
            }
        }
    };

    float acc[N_ATOMS_PW][4] = {0};
    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;
    for (int k_base = 0; k_base < K_up; k_base += BLOCK_K) {
        int next_stage = compute_stage ^ 1;
        if (k_base + BLOCK_K < K_up) issue_load(next_stage, k_base + BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        #pragma unroll
        for (int k_iter = 0; k_iter < K_ATOMS; ++k_iter) {
            int kA0 = k_iter * 32 + 4 * l;
            int kA2 = k_iter * 32 + 4 * l + 16;
            int rA0 = h, rA1 = h + 8;
            uint32_t A0 = *reinterpret_cast<const uint32_t*>(
                &A_smem[compute_stage * M_ROWS_AT * SMEM_K_PAD + rA0 * SMEM_K_PAD + kA0]);
            uint32_t A1 = *reinterpret_cast<const uint32_t*>(
                &A_smem[compute_stage * M_ROWS_AT * SMEM_K_PAD + rA1 * SMEM_K_PAD + kA0]);
            uint32_t A2 = *reinterpret_cast<const uint32_t*>(
                &A_smem[compute_stage * M_ROWS_AT * SMEM_K_PAD + rA0 * SMEM_K_PAD + kA2]);
            uint32_t A3 = *reinterpret_cast<const uint32_t*>(
                &A_smem[compute_stage * M_ROWS_AT * SMEM_K_PAD + rA1 * SMEM_K_PAD + kA2]);
            #pragma unroll
            for (int n_atom = 0; n_atom < N_ATOMS_PW; ++n_atom) {
                int co_n = warp_id * N_ATOMS_PW * 8 + n_atom * 8 + h;
                uint32_t B0 = *reinterpret_cast<const uint32_t*>(
                    &B_smem[compute_stage * BLOCK_N_UP * SMEM_K_PAD + co_n * SMEM_K_PAD + kA0]);
                uint32_t B1 = *reinterpret_cast<const uint32_t*>(
                    &B_smem[compute_stage * BLOCK_N_UP * SMEM_K_PAD + co_n * SMEM_K_PAD + kA2]);
                mma_m16n8k32_e4m3(
                    acc[n_atom][0], acc[n_atom][1], acc[n_atom][2], acc[n_atom][3],
                    A0, A1, A2, A3, B0, B1);
            }
        }
        compute_stage = next_stage;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // Fused epilogue: alpha + bias + gelu + quant_fp8
    #pragma unroll
    for (int n_atom = 0; n_atom < N_ATOMS_PW; ++n_atom) {
        int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + n_atom * 8 + 2 * l;
        int row0 = m_base + h;
        int row1 = m_base + h + 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = (j < 2) ? row0 : row1;
            int col = n_pair_base + (j & 1);
            if (row < M && col < N_up) {
                float v = acc[n_atom][j] * up_alpha;
                v += __bfloat162float(up_bias[col]);
                v = gelu_tanh(v);
                float sv = __bfloat162float(dn_inv_s[col]);
                float q = v * sv * dn_act_scale_inv;
                q = fmaxf(-448.0f, fminf(448.0f, q));
                up_fp8[row * N_up + col] = __nv_fp8_e4m3(q);
            }
        }
    }
}

// -- Phase 3: GEMM_dn + bias_add fused (3-stage cp.async) --
__device__ __forceinline__ void phase_gemm_dn_fused_bias(
    const __nv_fp8_e4m3* __restrict__ up_fp8,
    const __nv_fp8_e4m3* __restrict__ dn_w_NK,
    const __nv_bfloat16* __restrict__ dn_bias,
    const __nv_bfloat16* __restrict__ residual_in,
    float dn_alpha,
    __nv_bfloat16* __restrict__ y_out,
    int M, int N_dn, int K_dn, int m_base, int n_base,
    uint8_t* A_smem, uint8_t* B_smem)
{
    constexpr int BLOCK_K = BLOCK_K_DN;
    constexpr int SMEM_K_PAD = SMEM_K_PAD_DN;
    constexpr int N_ATOMS = BLOCK_N_DN / 8;             // 4
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;     // 1
    constexpr int K_ATOMS = BLOCK_K / 32;                // 4

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;

    constexpr int STAGES = 3;
    auto issue_load = [&](int stage, int k_base) {
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
                if (m_glob < M && k_glob < K_dn) {
                    a_src = reinterpret_cast<const uint8_t*>(
                        &up_fp8[m_glob * K_dn + k_glob]);
                }
                cp_async_16(
                    to_smem(&A_smem[stage * M_ROWS_AT * SMEM_K_PAD + row_a * SMEM_K_PAD + koff_a]),
                    a_src);
            }
        }
        constexpr int B_TOTAL = BLOCK_N_DN * BLOCK_K / 16;
        constexpr int B_ITERS = (B_TOTAL + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            int row_b = idx / (BLOCK_K / 16);
            int koff_b = (idx & (BLOCK_K/16 - 1)) * 16;
            if (row_b < BLOCK_N_DN) {
                const uint8_t* b_src = nullptr;
                int n_glob = n_base + row_b;
                int k_glob = k_base + koff_b;
                if (n_glob < N_dn && k_glob < K_dn) {
                    b_src = reinterpret_cast<const uint8_t*>(
                        &dn_w_NK[n_glob * K_dn + k_glob]);
                }
                cp_async_16(
                    to_smem(&B_smem[stage * BLOCK_N_DN * SMEM_K_PAD + row_b * SMEM_K_PAD + koff_b]),
                    b_src);
            }
        }
    };

    float acc[N_ATOMS_PW][4] = {0};
    int n_k_tiles = (K_dn + BLOCK_K - 1) / BLOCK_K;
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
            int rA0 = h, rA1 = h + 8;
            uint32_t A0 = *reinterpret_cast<const uint32_t*>(
                &A_smem[stage * M_ROWS_AT * SMEM_K_PAD + rA0 * SMEM_K_PAD + kA0]);
            uint32_t A1 = *reinterpret_cast<const uint32_t*>(
                &A_smem[stage * M_ROWS_AT * SMEM_K_PAD + rA1 * SMEM_K_PAD + kA0]);
            uint32_t A2 = *reinterpret_cast<const uint32_t*>(
                &A_smem[stage * M_ROWS_AT * SMEM_K_PAD + rA0 * SMEM_K_PAD + kA2]);
            uint32_t A3 = *reinterpret_cast<const uint32_t*>(
                &A_smem[stage * M_ROWS_AT * SMEM_K_PAD + rA1 * SMEM_K_PAD + kA2]);
            #pragma unroll
            for (int n_atom = 0; n_atom < N_ATOMS_PW; ++n_atom) {
                int co_n = warp_id * N_ATOMS_PW * 8 + n_atom * 8 + h;
                uint32_t B0 = *reinterpret_cast<const uint32_t*>(
                    &B_smem[stage * BLOCK_N_DN * SMEM_K_PAD + co_n * SMEM_K_PAD + kA0]);
                uint32_t B1 = *reinterpret_cast<const uint32_t*>(
                    &B_smem[stage * BLOCK_N_DN * SMEM_K_PAD + co_n * SMEM_K_PAD + kA2]);
                mma_m16n8k32_e4m3(
                    acc[n_atom][0], acc[n_atom][1], acc[n_atom][2], acc[n_atom][3],
                    A0, A1, A2, A3, B0, B1);
            }
        }
        stage = (stage + 1) % STAGES;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    #pragma unroll
    for (int n_atom = 0; n_atom < N_ATOMS_PW; ++n_atom) {
        int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + n_atom * 8 + 2 * l;
        int row0 = m_base + h, row1 = m_base + h + 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = (j < 2) ? row0 : row1;
            int col = n_pair_base + (j & 1);
            if (row < M && col < N_dn) {
                float v = acc[n_atom][j] * dn_alpha;
                v += __bfloat162float(dn_bias[col]);
                v += __bfloat162float(residual_in[row * N_dn + col]);
                y_out[row * N_dn + col] = __float2bfloat16(v);
            }
        }
    }
}

// -- V4b: split into 2 kernels. kernel_A = phase 1 (quant) + phase 2
// (GEMM_up + fused epilogue). kernel_B = phase 3 (GEMM_dn + fused bias).
// Removes the phase-2->phase-3 grid_barrier; CUDA graph manages the
// inter-kernel dep.

__global__ void __launch_bounds__(THREADS, V5S_LAUNCH_MIN)
kernel_A_und(
    const __nv_bfloat16* __restrict__ x_in,
    const __nv_bfloat16* __restrict__ up_inv_s,
    const __nv_fp8_e4m3* __restrict__ up_w_NK,
    const __nv_bfloat16* __restrict__ up_bias,
    const __nv_bfloat16* __restrict__ dn_inv_s,
    __nv_fp8_e4m3* __restrict__ x_fp8_scr,
    __nv_fp8_e4m3* __restrict__ up_fp8_scr,
    int M, int K_up, int N_up,
    float up_alpha, float dn_act_scale_inv, float up_act_scale,
    uint32_t* __restrict__ barrier_state)
{
    constexpr int A_SMEM_BYTES = 2 * M_ROWS_AT * SMEM_K_PAD_UP;
    constexpr int B_SMEM_BYTES = 2 * BLOCK_N_UP * SMEM_K_PAD_UP;
    __shared__ __align__(16) uint8_t A_smem[A_SMEM_BYTES];
    __shared__ __align__(16) uint8_t B_smem[B_SMEM_BYTES];

    const int cta = blockIdx.x;

    phase_quant_input(x_in, up_inv_s, up_act_scale, x_fp8_scr,
                      M, K_up, cta);
    grid_barrier(barrier_state, NUM_CTAS);

    int n_tiles_up = N_up / BLOCK_N_UP;
    int m_tiles_up = (M + BLOCK_M - 1) / BLOCK_M;
    int total_tiles = m_tiles_up * n_tiles_up;
    if (cta < total_tiles) {
        int m_idx = cta / n_tiles_up;
        int n_idx = cta % n_tiles_up;
        phase_gemm_up_fused_eup(
            x_fp8_scr, up_w_NK, up_bias, dn_inv_s,
            up_alpha, dn_act_scale_inv,
            up_fp8_scr,
            M, N_up, K_up,
            m_idx * BLOCK_M, n_idx * BLOCK_N_UP,
            A_smem, B_smem);
    }
}

__global__ void __launch_bounds__(THREADS, V5S_LAUNCH_MIN)
kernel_B_und(
    const __nv_fp8_e4m3* __restrict__ up_fp8_scr,
    const __nv_fp8_e4m3* __restrict__ dn_w_NK,
    const __nv_bfloat16* __restrict__ dn_bias,
    const __nv_bfloat16* __restrict__ residual_in,
    float dn_alpha,
    __nv_bfloat16* __restrict__ y_out,
    int M, int K_dn, int N_dn)
{
    constexpr int A_SMEM_BYTES = 3 * M_ROWS_AT * SMEM_K_PAD_DN;
    constexpr int B_SMEM_BYTES = 3 * BLOCK_N_DN * SMEM_K_PAD_DN;
    __shared__ __align__(16) uint8_t A_smem[A_SMEM_BYTES];
    __shared__ __align__(16) uint8_t B_smem[B_SMEM_BYTES];

    const int cta = blockIdx.x;
    int n_tiles_dn = N_dn / BLOCK_N_DN;
    int m_tiles_dn = (M + BLOCK_M - 1) / BLOCK_M;
    int total_tiles = m_tiles_dn * n_tiles_dn;
    if (cta < total_tiles) {
        int m_idx = cta / n_tiles_dn;
        int n_idx = cta % n_tiles_dn;
            phase_gemm_dn_fused_bias(
            up_fp8_scr, dn_w_NK, dn_bias, residual_in, dn_alpha,
            y_out,
            M, N_dn, K_dn,
            m_idx * BLOCK_M, n_idx * BLOCK_N_DN,
            A_smem, B_smem);
    }
}

constexpr int NUM_CTAS_B_UND = M_TILES * (512 / BLOCK_N_DN);

extern "C" int und_ffn_v5split_stage3_launch(
    const void* x_in, const void* up_inv_s,
    const void* up_w_NK, const void* up_bias,
    const void* dn_inv_s, const void* dn_w_NK, const void* dn_bias,
    const void* residual_in,
    void* y_out,
    void* x_fp8_scr, void* up_fp8_scr,
    int M, int K_up, int N_up, int K_dn, int N_dn,
    float up_alpha, float dn_alpha,
    float up_act_scale, float dn_act_scale,
    void* barrier_state, cudaStream_t stream)
{
    if (M > 192) return -1;
    if (K_up != 512 || N_up != 2048 || K_dn != 2048 || N_dn != 512) return -2;

    kernel_A_und<<<dim3(NUM_CTAS), dim3(THREADS), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_in),
        reinterpret_cast<const __nv_bfloat16*>(up_inv_s),
        reinterpret_cast<const __nv_fp8_e4m3*>(up_w_NK),
        reinterpret_cast<const __nv_bfloat16*>(up_bias),
        reinterpret_cast<const __nv_bfloat16*>(dn_inv_s),
        reinterpret_cast<__nv_fp8_e4m3*>(x_fp8_scr),
        reinterpret_cast<__nv_fp8_e4m3*>(up_fp8_scr),
        M, K_up, N_up,
        up_alpha, 1.0f / dn_act_scale, up_act_scale,
        reinterpret_cast<uint32_t*>(barrier_state));

    kernel_B_und<<<dim3(NUM_CTAS_B_UND), dim3(THREADS), 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(up_fp8_scr),
        reinterpret_cast<const __nv_fp8_e4m3*>(dn_w_NK),
        reinterpret_cast<const __nv_bfloat16*>(dn_bias),
        reinterpret_cast<const __nv_bfloat16*>(residual_in),
        dn_alpha,
        reinterpret_cast<__nv_bfloat16*>(y_out),
        M, K_dn, N_dn);

    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (100 + (int)e);
}

}  // namespace

namespace flash_rt {
namespace megakernel {

int und_ffn_v5split_stage3_launch_sm120(
    const void* x_in, const void* up_inv_s,
    const void* up_w_NK, const void* up_bias,
    const void* dn_inv_s, const void* dn_w_NK, const void* dn_bias,
    const void* residual_in,
    void* y_out,
    void* x_fp8_scr, void* up_fp8_scr,
    int M, int K_up, int N_up, int K_dn, int N_dn,
    float up_alpha, float dn_alpha,
    float up_act_scale, float dn_act_scale,
    void* barrier_state, cudaStream_t stream)
{
    return und_ffn_v5split_stage3::und_ffn_v5split_stage3_launch(
        x_in, up_inv_s,
        up_w_NK, up_bias,
        dn_inv_s, dn_w_NK, dn_bias,
        residual_in,
        y_out, x_fp8_scr, up_fp8_scr,
        M, K_up, N_up, K_dn, N_dn,
        up_alpha, dn_alpha, up_act_scale, dn_act_scale,
        barrier_state, stream);
}

}  // namespace megakernel
}  // namespace flash_rt
