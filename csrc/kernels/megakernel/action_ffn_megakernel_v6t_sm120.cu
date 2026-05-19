// Action FFN megakernel V6: kernel_B epilogue scope EXPAND — absorb gated
// residual (post-FFN element-wise ops `ffn_out * a_mod[5] + residual_in`).
//
// Builds on V4b (2-kernel split: A = quant + GEMM_up + fused epilogue,
// B = GEMM_dn). V6 changes ONLY kernel_B's epilogue:
//
//   V4b epilogue:  y_out = (acc * dn_alpha + dn_bias).to(bf16)
//   V6 epilogue:   y_out = residual + (acc * dn_alpha + dn_bias) * gate
//
// Pre-FFN chain in motus.process_ffn():
//   ffn_input = norm2(x).float() * (1 + a_mod[4]) + a_mod[3]   ← V6 doesn't touch
//   ffn_out   = action_block.ffn(ffn_input)                    ← V6 megakernel
//   action_tokens = x + ffn_out * a_mod[5]                     ← V6 ABSORBS this
//
// Save: 2 elementwise launches (mul-gate + add-residual) per call.
// 300 calls/inf × ~3-5 μs each × ~50-60% transfer ≈ -1.5~2 ms wall target.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace action_ffn_v6t {

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

// V6t: parameterized tile knobs (compile-time -D macros)
#ifndef V6T_BLOCK_K_UP
#define V6T_BLOCK_K_UP 128
#endif
#ifndef V6T_BLOCK_K_DN
#define V6T_BLOCK_K_DN 256
#endif
#ifndef V6T_BLOCK_N_UP
#define V6T_BLOCK_N_UP 32
#endif
#ifndef V6T_BLOCK_N_DN
#define V6T_BLOCK_N_DN 32
#endif
#ifndef V6T_STAGES_UP
#define V6T_STAGES_UP 2
#endif
#ifndef V6T_STAGES_DN
#define V6T_STAGES_DN 3
#endif
#ifndef V6T_NUM_WARPS
#define V6T_NUM_WARPS 4
#endif
#ifndef V6T_LAUNCH_MIN
#define V6T_LAUNCH_MIN 8
#endif

constexpr int NUM_WARPS = V6T_NUM_WARPS;
constexpr int THREADS = NUM_WARPS * 32;
constexpr int NUM_CTAS_A = 4096 / V6T_BLOCK_N_UP;
constexpr int NUM_CTAS_B = 1024 / V6T_BLOCK_N_DN;
constexpr int M_ROWS_AT = 16;
constexpr int BLOCK_K_UP = V6T_BLOCK_K_UP;
constexpr int BLOCK_K_DN = V6T_BLOCK_K_DN;
constexpr int BLOCK_N_UP = V6T_BLOCK_N_UP;
constexpr int BLOCK_N_DN = V6T_BLOCK_N_DN;
constexpr int STAGES_UP = V6T_STAGES_UP;
constexpr int STAGES_DN = V6T_STAGES_DN;
constexpr int SMEM_K_PAD_UP = BLOCK_K_UP + 16;
constexpr int SMEM_K_PAD_DN = BLOCK_K_DN + 16;

// ── Kernel A (V6): consumes pre-quantized x_fp8 from upstream (e.g. produced
// by ada_layer_norm_fp8). NO internal phase 1 quant; NO grid_barrier.
// up_inv_s / up_act_scale params are no longer used but kept for ABI
// compatibility with V5 install pattern (caller may pass nullptr).
__global__ void __launch_bounds__(THREADS, V6T_LAUNCH_MIN)
kernel_A(
    const __nv_fp8_e4m3* __restrict__ x_fp8_in,
    const __nv_fp8_e4m3* __restrict__ up_w_NK,
    const __nv_bfloat16* __restrict__ up_bias,
    const __nv_bfloat16* __restrict__ dn_inv_s,
    __nv_fp8_e4m3* __restrict__ up_fp8_scr,
    int M, int K_up, int N_up,
    float up_alpha, float dn_act_scale_inv)
{
    constexpr int A_SMEM = STAGES_UP * M_ROWS_AT * SMEM_K_PAD_UP;
    constexpr int B_SMEM = STAGES_UP * BLOCK_N_UP * SMEM_K_PAD_UP;
    __shared__ __align__(16) uint8_t A_smem[A_SMEM];
    __shared__ __align__(16) uint8_t B_smem[B_SMEM];

    const int cta = blockIdx.x;
    const int m_base = blockIdx.y * M_ROWS_AT;
    const int t = threadIdx.x;

    int n_tiles_up = N_up / BLOCK_N_UP;
    if (cta >= n_tiles_up) return;
    int n_base = cta * BLOCK_N_UP;

    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;
    constexpr int N_ATOMS_PW = (BLOCK_N_UP / 8 + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int K_ATOMS = BLOCK_K_UP / 32;

    auto issue = [&](int stage, int kb) {
        constexpr int A_ROWS_PER_ITER = THREADS / (BLOCK_K_UP / 16);
        constexpr int A_ITERS = (M_ROWS_AT + A_ROWS_PER_ITER - 1) / A_ROWS_PER_ITER;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            int ra = idx / (BLOCK_K_UP / 16);
            int ko = (idx & (BLOCK_K_UP/16 - 1)) * 16;
            if (ra < M_ROWS_AT) {
                const uint8_t* a_src = nullptr;
                int rg = m_base + ra;
                if (rg < M && kb + ko < K_up)
                    a_src = reinterpret_cast<const uint8_t*>(&x_fp8_in[rg * K_up + kb + ko]);
                cp_async_16(
                    to_smem(&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_UP + ra * SMEM_K_PAD_UP + ko]),
                    a_src);
            }
        }
        constexpr int B_TOTAL = BLOCK_N_UP * BLOCK_K_UP / 16;
        constexpr int B_ITERS = (B_TOTAL + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            int rb = idx / (BLOCK_K_UP / 16);
            int ko = (idx & (BLOCK_K_UP/16 - 1)) * 16;
            if (rb < BLOCK_N_UP) {
                const uint8_t* b_src = nullptr;
                int ng = n_base + rb;
                if (ng < N_up && kb + ko < K_up)
                    b_src = reinterpret_cast<const uint8_t*>(&up_w_NK[ng * K_up + kb + ko]);
                cp_async_16(
                    to_smem(&B_smem[stage * BLOCK_N_UP * SMEM_K_PAD_UP + rb * SMEM_K_PAD_UP + ko]),
                    b_src);
            }
        }
    };

    float acc[N_ATOMS_PW][4] = {0};
    int issue_stage = 0, k_issue = 0;
    #pragma unroll
    for (int s = 0; s < STAGES_UP - 1; ++s) {
        if (k_issue < K_up) issue(issue_stage, k_issue);
        asm volatile("cp.async.commit_group;\n" ::);
        issue_stage = (issue_stage + 1) % STAGES_UP;
        k_issue += BLOCK_K_UP;
    }
    int stage = 0;
    for (int kb = 0; kb < K_up; kb += BLOCK_K_UP) {
        if (k_issue < K_up) issue(issue_stage, k_issue);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES_UP - 1));
        __syncthreads();
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS; ++ki) {
            int kA0 = ki * 32 + 4 * l;
            int kA2 = ki * 32 + 4 * l + 16;
            int rA0 = h, rA1 = h + 8;
            uint32_t A0 = *(uint32_t*)&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_UP + rA0 * SMEM_K_PAD_UP + kA0];
            uint32_t A1 = *(uint32_t*)&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_UP + rA1 * SMEM_K_PAD_UP + kA0];
            uint32_t A2 = *(uint32_t*)&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_UP + rA0 * SMEM_K_PAD_UP + kA2];
            uint32_t A3 = *(uint32_t*)&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_UP + rA1 * SMEM_K_PAD_UP + kA2];
            #pragma unroll
            for (int na = 0; na < N_ATOMS_PW; ++na) {
                int co_n = warp_id * N_ATOMS_PW * 8 + na * 8 + h;
                uint32_t B0 = *(uint32_t*)&B_smem[stage * BLOCK_N_UP * SMEM_K_PAD_UP + co_n * SMEM_K_PAD_UP + kA0];
                uint32_t B1 = *(uint32_t*)&B_smem[stage * BLOCK_N_UP * SMEM_K_PAD_UP + co_n * SMEM_K_PAD_UP + kA2];
                mma_m16n8k32_e4m3(acc[na][0], acc[na][1], acc[na][2], acc[na][3],
                                  A0, A1, A2, A3, B0, B1);
            }
        }
        stage = (stage + 1) % STAGES_UP;
        issue_stage = (issue_stage + 1) % STAGES_UP;
        k_issue += BLOCK_K_UP;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    #pragma unroll
    for (int na = 0; na < N_ATOMS_PW; ++na) {
        int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + na * 8 + 2 * l;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = (j < 2) ? h : (h + 8);
            int col = n_pair_base + (j & 1);
            int rg = m_base + row;
            if (rg < M && col < N_up) {
                float v = acc[na][j] * up_alpha;
                v += __bfloat162float(up_bias[col]);
                v = gelu_tanh(v);
                float sv = __bfloat162float(dn_inv_s[col]);
                float q = v * sv * dn_act_scale_inv;
                q = fmaxf(-448.0f, fminf(448.0f, q));
                up_fp8_scr[rg * N_up + col] = __nv_fp8_e4m3(q);
            }
        }
    }
}

// ── Kernel B (V6): GEMM_dn + fused bias + GATE * (...) + RESIDUAL ──
__global__ void __launch_bounds__(THREADS, V6T_LAUNCH_MIN)
kernel_B(
    const __nv_fp8_e4m3* __restrict__ up_fp8,
    const __nv_fp8_e4m3* __restrict__ dn_w_NK,
    const __nv_bfloat16* __restrict__ dn_bias,
    const __nv_bfloat16* __restrict__ gate,        // ★ NEW: (M, N_dn) bf16
    const __nv_bfloat16* __restrict__ residual,    // ★ NEW: (M, N_dn) bf16
    float dn_alpha,
    __nv_bfloat16* __restrict__ y_out,
    int M, int K_dn, int N_dn)
{
    constexpr int A_SMEM = STAGES_DN * M_ROWS_AT * SMEM_K_PAD_DN;
    constexpr int B_SMEM = STAGES_DN * BLOCK_N_DN * SMEM_K_PAD_DN;
    __shared__ __align__(16) uint8_t A_smem[A_SMEM];
    __shared__ __align__(16) uint8_t B_smem[B_SMEM];

    const int cta = blockIdx.x;
    const int m_base = blockIdx.y * M_ROWS_AT;
    if (cta >= N_dn / BLOCK_N_DN) return;
    int n_base = cta * BLOCK_N_DN;

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;
    constexpr int N_ATOMS_PW = (BLOCK_N_DN / 8 + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int K_ATOMS = BLOCK_K_DN / 32;

    auto issue = [&](int stage, int kb) {
        constexpr int A_ROWS_PER_ITER = THREADS / (BLOCK_K_DN / 16);
        constexpr int A_ITERS = (M_ROWS_AT + A_ROWS_PER_ITER - 1) / A_ROWS_PER_ITER;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            int ra = idx / (BLOCK_K_DN / 16);
            int ko = (idx & (BLOCK_K_DN/16 - 1)) * 16;
            if (ra < M_ROWS_AT) {
                const uint8_t* a_src = nullptr;
                int rg = m_base + ra;
                if (rg < M && kb + ko < K_dn)
                    a_src = reinterpret_cast<const uint8_t*>(&up_fp8[rg * K_dn + kb + ko]);
                cp_async_16(
                    to_smem(&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_DN + ra * SMEM_K_PAD_DN + ko]),
                    a_src);
            }
        }
        constexpr int B_TOTAL = BLOCK_N_DN * BLOCK_K_DN / 16;
        constexpr int B_ITERS = (B_TOTAL + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = it * THREADS + t;
            int rb = idx / (BLOCK_K_DN / 16);
            int ko = (idx & (BLOCK_K_DN/16 - 1)) * 16;
            if (rb < BLOCK_N_DN) {
                const uint8_t* b_src = nullptr;
                int ng = n_base + rb;
                if (ng < N_dn && kb + ko < K_dn)
                    b_src = reinterpret_cast<const uint8_t*>(&dn_w_NK[ng * K_dn + kb + ko]);
                cp_async_16(
                    to_smem(&B_smem[stage * BLOCK_N_DN * SMEM_K_PAD_DN + rb * SMEM_K_PAD_DN + ko]),
                    b_src);
            }
        }
    };

    float acc[N_ATOMS_PW][4] = {0};
    int issue_stage = 0, k_issue = 0;
    #pragma unroll
    for (int s = 0; s < STAGES_DN - 1; ++s) {
        if (k_issue < K_dn) issue(issue_stage, k_issue);
        asm volatile("cp.async.commit_group;\n" ::);
        issue_stage = (issue_stage + 1) % STAGES_DN;
        k_issue += BLOCK_K_DN;
    }
    int stage = 0;
    for (int kb = 0; kb < K_dn; kb += BLOCK_K_DN) {
        if (k_issue < K_dn) issue(issue_stage, k_issue);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES_DN - 1));
        __syncthreads();
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS; ++ki) {
            int kA0 = ki * 32 + 4 * l;
            int kA2 = ki * 32 + 4 * l + 16;
            int rA0 = h, rA1 = h + 8;
            uint32_t A0 = *(uint32_t*)&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_DN + rA0 * SMEM_K_PAD_DN + kA0];
            uint32_t A1 = *(uint32_t*)&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_DN + rA1 * SMEM_K_PAD_DN + kA0];
            uint32_t A2 = *(uint32_t*)&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_DN + rA0 * SMEM_K_PAD_DN + kA2];
            uint32_t A3 = *(uint32_t*)&A_smem[stage * M_ROWS_AT * SMEM_K_PAD_DN + rA1 * SMEM_K_PAD_DN + kA2];
            #pragma unroll
            for (int na = 0; na < N_ATOMS_PW; ++na) {
                int co_n = warp_id * N_ATOMS_PW * 8 + na * 8 + h;
                uint32_t B0 = *(uint32_t*)&B_smem[stage * BLOCK_N_DN * SMEM_K_PAD_DN + co_n * SMEM_K_PAD_DN + kA0];
                uint32_t B1 = *(uint32_t*)&B_smem[stage * BLOCK_N_DN * SMEM_K_PAD_DN + co_n * SMEM_K_PAD_DN + kA2];
                mma_m16n8k32_e4m3(acc[na][0], acc[na][1], acc[na][2], acc[na][3],
                                  A0, A1, A2, A3, B0, B1);
            }
        }
        stage = (stage + 1) % STAGES_DN;
        issue_stage = (issue_stage + 1) % STAGES_DN;
        k_issue += BLOCK_K_DN;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // ★ V6 epilogue: (acc + bias) * gate + residual
    #pragma unroll
    for (int na = 0; na < N_ATOMS_PW; ++na) {
        int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + na * 8 + 2 * l;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = (j < 2) ? h : (h + 8);
            int col = n_pair_base + (j & 1);
            int rg = m_base + row;
            if (rg < M && col < N_dn) {
                float v = acc[na][j] * dn_alpha;
                v += __bfloat162float(dn_bias[col]);
                float g = __bfloat162float(gate[rg * N_dn + col]);
                float r = __bfloat162float(residual[rg * N_dn + col]);
                v = v * g + r;
                y_out[rg * N_dn + col] = __float2bfloat16(v);
            }
        }
    }
}

extern "C" int action_ffn_v6t_launch(
    const void* x_fp8_in,                       // V6: pre-quantized FP8 input
    const void* up_w_NK, const void* up_bias,
    const void* dn_inv_s, const void* dn_w_NK, const void* dn_bias,
    const void* gate, const void* residual,
    void* y_out,
    void* up_fp8_scr,                            // V6: no x_fp8_scr needed
    int M, int K_up, int N_up, int K_dn, int N_dn,
    float up_alpha, float dn_alpha,
    float dn_act_scale,
    cudaStream_t stream)
{
    if (M <= 0 || M > 32) return -1;
    if (K_up != 1024 || N_up != 4096 || K_dn != 4096 || N_dn != 1024) return -2;
    int m_tiles = (M + M_ROWS_AT - 1) / M_ROWS_AT;

    kernel_A<<<dim3(NUM_CTAS_A, m_tiles), dim3(THREADS), 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(x_fp8_in),
        reinterpret_cast<const __nv_fp8_e4m3*>(up_w_NK),
        reinterpret_cast<const __nv_bfloat16*>(up_bias),
        reinterpret_cast<const __nv_bfloat16*>(dn_inv_s),
        reinterpret_cast<__nv_fp8_e4m3*>(up_fp8_scr),
        M, K_up, N_up,
        up_alpha, 1.0f / dn_act_scale);

    kernel_B<<<dim3(NUM_CTAS_B, m_tiles), dim3(THREADS), 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(up_fp8_scr),
        reinterpret_cast<const __nv_fp8_e4m3*>(dn_w_NK),
        reinterpret_cast<const __nv_bfloat16*>(dn_bias),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        dn_alpha,
        reinterpret_cast<__nv_bfloat16*>(y_out),
        M, K_dn, N_dn);

    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (100 + (int)e);
}

}  // namespace

namespace flash_rt {
namespace megakernel {

int action_ffn_v6t_launch_sm120(
    const void* x_fp8_in,
    const void* up_w_NK, const void* up_bias,
    const void* dn_inv_s, const void* dn_w_NK, const void* dn_bias,
    const void* gate, const void* residual,
    void* y_out,
    void* up_fp8_scr,
    int M, int K_up, int N_up, int K_dn, int N_dn,
    float up_alpha, float dn_alpha,
    float dn_act_scale,
    cudaStream_t stream)
{
    return action_ffn_v6t::action_ffn_v6t_launch(
        x_fp8_in,
        up_w_NK, up_bias,
        dn_inv_s, dn_w_NK, dn_bias,
        gate, residual,
        y_out, up_fp8_scr,
        M, K_up, N_up, K_dn, N_dn,
        up_alpha, dn_alpha, dn_act_scale,
        stream);
}

}  // namespace megakernel
}  // namespace flash_rt
