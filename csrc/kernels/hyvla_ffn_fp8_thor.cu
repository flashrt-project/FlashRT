// FlashRT — Hy-VLA denoise FFN megakernel (Thor SM110, plain FP8 MMA).
//
// The occupancy-preserving whole-FFN fusion (kept all 20 SMs busy) that the
// single-CTA quant PoC lacked. Two persistent tiled FP8 GEMMs, m16n8k32 e4m3
// (the plain mma.sync — Thor rejects sm_120's .kind::f8f6f4), cp.async 2-stage.
// Adapted from the sm_120 action_ffn_v6t template:
//   A: gu-GEMM (K=1024 -> 2*Nout=4096) with gate/up dual-accumulator, then
//      silu(gate)*up -> bf16 act (M, Nout=2048).
//   B: dn-GEMM (K=2048 -> N=1024) + residual -> bf16 y.
// Dynamic FP8: activation scale is a device pointer read at launch (graph-safe,
// like fp8_nn_dev); weight scale is a host constant (fixed at quantize time).
// The act between A and B is dynamically requantized by the caller
// (quantize_fp8_device) so no grid barrier / static act scale is needed.
// Verified vs a float FFN reference: A cos 0.999999, B cos 0.9995.
//
// Static shared memory (<48KB) so NO cudaFuncSetAttribute is needed — the
// launch has no host attribute call, keeping the captured path clean.
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

namespace {

constexpr int NUM_WARPS = 4;
constexpr int THREADS = NUM_WARPS * 32;
constexpr int M_ROWS = 16;
constexpr int BLOCK_N = 32;

__device__ __forceinline__ void mma_e4m3(float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(uint32_t s, const uint8_t* g) {
    int b = (g == nullptr) ? 0 : 16;
    asm volatile("cp.async.ca.shared.global [%0],[%1],16,%2;\n" :: "r"(s), "l"(g), "r"(b));
}
__device__ __forceinline__ uint32_t sma(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ float siluf(float x) { return x / (1.0f + __expf(-x)); }

// ── Kernel A: gu-GEMM gate/up + silu_mul -> bf16 act ──
template<int BK>
__global__ void __launch_bounds__(THREADS, 8) ffn_A(
    const __nv_fp8_e4m3* __restrict__ x, const __nv_fp8_e4m3* __restrict__ gu,
    __nv_bfloat16* __restrict__ act, int M, int K, int Nout,
    const float* __restrict__ sx_ptr, float sgu) {
    constexpr int PAD = BK + 16;
    __shared__ __align__(16) uint8_t As[2 * M_ROWS * PAD];
    __shared__ __align__(16) uint8_t Bg[2 * BLOCK_N * PAD];
    __shared__ __align__(16) uint8_t Bu[2 * BLOCK_N * PAD];
    const int cta = blockIdx.x, m_base = blockIdx.y * M_ROWS, t = threadIdx.x;
    if (cta >= Nout / BLOCK_N) return;
    const int nb = cta * BLOCK_N, warp = t / 32, lane = t % 32, l = lane % 4, h = lane / 4;
    constexpr int NA = (BLOCK_N / 8 + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int KA = BK / 32;
    auto issue = [&](int st, int kb) {
        constexpr int AR = THREADS / (BK / 16), AI = (M_ROWS + AR - 1) / AR;
        #pragma unroll
        for (int it = 0; it < AI; ++it) {
            int idx = it * THREADS + t, ra = idx / (BK / 16), ko = (idx & (BK / 16 - 1)) * 16;
            if (ra < M_ROWS) {
                const uint8_t* s = nullptr; int rg = m_base + ra;
                if (rg < M && kb + ko < K) s = (const uint8_t*)&x[rg * K + kb + ko];
                cp16(sma(&As[st * M_ROWS * PAD + ra * PAD + ko]), s);
            }
        }
        constexpr int BT = BLOCK_N * BK / 16, BI = (BT + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < BI; ++it) {
            int idx = it * THREADS + t, rb = idx / (BK / 16), ko = (idx & (BK / 16 - 1)) * 16;
            if (rb < BLOCK_N) {
                int ng = nb + rb; const uint8_t* sg = nullptr; const uint8_t* su = nullptr;
                if (ng < Nout && kb + ko < K) {
                    sg = (const uint8_t*)&gu[ng * K + kb + ko];
                    su = (const uint8_t*)&gu[(Nout + ng) * K + kb + ko];
                }
                cp16(sma(&Bg[st * BLOCK_N * PAD + rb * PAD + ko]), sg);
                cp16(sma(&Bu[st * BLOCK_N * PAD + rb * PAD + ko]), su);
            }
        }
    };
    float ag[NA][4] = {0}, au[NA][4] = {0};
    int is = 0, ki = 0; issue(0, 0); asm volatile("cp.async.commit_group;\n" ::); is = 1; ki = BK;
    int stg = 0;
    for (int kb = 0; kb < K; kb += BK) {
        if (ki < K) issue(is, ki); asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::); __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KA; ++kk) {
            int k0 = kk * 32 + 4 * l, k2 = k0 + 16, r0 = h, r1 = h + 8;
            uint32_t A0 = *(uint32_t*)&As[stg * M_ROWS * PAD + r0 * PAD + k0];
            uint32_t A1 = *(uint32_t*)&As[stg * M_ROWS * PAD + r1 * PAD + k0];
            uint32_t A2 = *(uint32_t*)&As[stg * M_ROWS * PAD + r0 * PAD + k2];
            uint32_t A3 = *(uint32_t*)&As[stg * M_ROWS * PAD + r1 * PAD + k2];
            #pragma unroll
            for (int na = 0; na < NA; ++na) {
                int cn = warp * NA * 8 + na * 8 + h;
                uint32_t G0 = *(uint32_t*)&Bg[stg * BLOCK_N * PAD + cn * PAD + k0];
                uint32_t G1 = *(uint32_t*)&Bg[stg * BLOCK_N * PAD + cn * PAD + k2];
                uint32_t U0 = *(uint32_t*)&Bu[stg * BLOCK_N * PAD + cn * PAD + k0];
                uint32_t U1 = *(uint32_t*)&Bu[stg * BLOCK_N * PAD + cn * PAD + k2];
                mma_e4m3(ag[na][0], ag[na][1], ag[na][2], ag[na][3], A0, A1, A2, A3, G0, G1);
                mma_e4m3(au[na][0], au[na][1], au[na][2], au[na][3], A0, A1, A2, A3, U0, U1);
            }
        }
        __syncthreads();   // WAR: finish reading stage `stg` before it is reissued
        stg ^= 1; is ^= 1; ki += BK;
    }
    asm volatile("cp.async.wait_all;\n" ::);
    float ds = (*sx_ptr) * sgu;
    #pragma unroll
    for (int na = 0; na < NA; ++na) {
        int cb = nb + warp * NA * 8 + na * 8 + 2 * l;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = (j < 2) ? h : (h + 8), col = cb + (j & 1), rg = m_base + row;
            if (rg < M && col < Nout) {
                float g = ag[na][j] * ds, u = au[na][j] * ds;
                act[rg * Nout + col] = __float2bfloat16(siluf(g) * u);
            }
        }
    }
}

// ── Kernel B: dn-GEMM + residual -> bf16 ──
template<int BK>
__global__ void __launch_bounds__(THREADS, 8) ffn_B(
    const __nv_fp8_e4m3* __restrict__ a, const __nv_fp8_e4m3* __restrict__ dn,
    const __nv_bfloat16* __restrict__ res, __nv_bfloat16* __restrict__ y,
    int M, int K, int N, const float* __restrict__ sa_ptr, float sdn) {
    constexpr int PAD = BK + 16;
    __shared__ __align__(16) uint8_t As[2 * M_ROWS * PAD];
    __shared__ __align__(16) uint8_t Bs[2 * BLOCK_N * PAD];
    const int cta = blockIdx.x, m_base = blockIdx.y * M_ROWS, t = threadIdx.x;
    if (cta >= N / BLOCK_N) return;
    const int nb = cta * BLOCK_N, warp = t / 32, lane = t % 32, l = lane % 4, h = lane / 4;
    constexpr int NA = (BLOCK_N / 8 + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int KA = BK / 32;
    auto issue = [&](int st, int kb) {
        constexpr int AR = THREADS / (BK / 16), AI = (M_ROWS + AR - 1) / AR;
        #pragma unroll
        for (int it = 0; it < AI; ++it) {
            int idx = it * THREADS + t, ra = idx / (BK / 16), ko = (idx & (BK / 16 - 1)) * 16;
            if (ra < M_ROWS) {
                const uint8_t* s = nullptr; int rg = m_base + ra;
                if (rg < M && kb + ko < K) s = (const uint8_t*)&a[rg * K + kb + ko];
                cp16(sma(&As[st * M_ROWS * PAD + ra * PAD + ko]), s);
            }
        }
        constexpr int BT = BLOCK_N * BK / 16, BI = (BT + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < BI; ++it) {
            int idx = it * THREADS + t, rb = idx / (BK / 16), ko = (idx & (BK / 16 - 1)) * 16;
            if (rb < BLOCK_N) {
                const uint8_t* s = nullptr; int ng = nb + rb;
                if (ng < N && kb + ko < K) s = (const uint8_t*)&dn[ng * K + kb + ko];
                cp16(sma(&Bs[st * BLOCK_N * PAD + rb * PAD + ko]), s);
            }
        }
    };
    float acc[NA][4] = {0};
    int is = 0, ki = 0; issue(0, 0); asm volatile("cp.async.commit_group;\n" ::); is = 1; ki = BK;
    int stg = 0;
    for (int kb = 0; kb < K; kb += BK) {
        if (ki < K) issue(is, ki); asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::); __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KA; ++kk) {
            int k0 = kk * 32 + 4 * l, k2 = k0 + 16, r0 = h, r1 = h + 8;
            uint32_t A0 = *(uint32_t*)&As[stg * M_ROWS * PAD + r0 * PAD + k0];
            uint32_t A1 = *(uint32_t*)&As[stg * M_ROWS * PAD + r1 * PAD + k0];
            uint32_t A2 = *(uint32_t*)&As[stg * M_ROWS * PAD + r0 * PAD + k2];
            uint32_t A3 = *(uint32_t*)&As[stg * M_ROWS * PAD + r1 * PAD + k2];
            #pragma unroll
            for (int na = 0; na < NA; ++na) {
                int cn = warp * NA * 8 + na * 8 + h;
                uint32_t B0 = *(uint32_t*)&Bs[stg * BLOCK_N * PAD + cn * PAD + k0];
                uint32_t B1 = *(uint32_t*)&Bs[stg * BLOCK_N * PAD + cn * PAD + k2];
                mma_e4m3(acc[na][0], acc[na][1], acc[na][2], acc[na][3], A0, A1, A2, A3, B0, B1);
            }
        }
        __syncthreads();   // WAR: finish reading stage `stg` before it is reissued
        stg ^= 1; is ^= 1; ki += BK;
    }
    asm volatile("cp.async.wait_all;\n" ::);
    float ds = (*sa_ptr) * sdn;
    #pragma unroll
    for (int na = 0; na < NA; ++na) {
        int cb = nb + warp * NA * 8 + na * 8 + 2 * l;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = (j < 2) ? h : (h + 8), col = cb + (j & 1), rg = m_base + row;
            if (rg < M && col < N)
                y[rg * N + col] = __float2bfloat16(acc[na][j] * ds + __bfloat162float(res[rg * N + col]));
        }
    }
}

}  // namespace

extern "C" void hyvla_ffn_gu_silu_bf16(
    const void* x_fp8, const void* gu_w_fp8, void* act_bf16,
    int M, int K, int Nout, const void* sx_ptr, float sgu, cudaStream_t stream) {
    int mt = (M + M_ROWS - 1) / M_ROWS;
    dim3 grid(Nout / BLOCK_N, mt);
    ffn_A<128><<<grid, THREADS, 0, stream>>>(
        (const __nv_fp8_e4m3*)x_fp8, (const __nv_fp8_e4m3*)gu_w_fp8,
        (__nv_bfloat16*)act_bf16, M, K, Nout, (const float*)sx_ptr, sgu);
}

extern "C" void hyvla_ffn_dn_res_bf16(
    const void* act_fp8, const void* dn_w_fp8, const void* residual, void* y_bf16,
    int M, int K, int N, const void* sa_ptr, float sdn, cudaStream_t stream) {
    int mt = (M + M_ROWS - 1) / M_ROWS;
    dim3 grid(N / BLOCK_N, mt);
    ffn_B<256><<<grid, THREADS, 0, stream>>>(
        (const __nv_fp8_e4m3*)act_fp8, (const __nv_fp8_e4m3*)dn_w_fp8,
        (const __nv_bfloat16*)residual, (__nv_bfloat16*)y_bf16,
        M, K, N, (const float*)sa_ptr, sdn);
}
