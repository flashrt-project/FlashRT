// SPDX-License-Identifier: Apache-2.0
//
// See header. Kernel body proven in the llama.cpp SM120 adapter (bit-exact
// duplicated-token replay across M variants, perplexity-neutral).

#include "fp4_w4a4_mma_warpsplit_mrows_f32out_sm120.cuh"

#include <cstdint>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace flash_rt {
namespace gemm {

namespace {

#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || !defined(__CUDA_ARCH__)
#define FR_WS_SM120A_OK 1
#endif

__device__ __forceinline__ void pdl_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}
__device__ __forceinline__ void pdl_lc() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

using AtomType = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;

__device__ __forceinline__ uint32_t fa(const uint8_t * s, int t0, int t1, int r) {
    int ro = ((r & 1) ? (t1 + 8) : t1) * 32;
    return *reinterpret_cast<const uint32_t *>(s + ro + t0 * 4 + ((r >> 1) & 1) * 16);
}
__device__ __forceinline__ uint32_t fb(const uint8_t * s, int t0, int t1, int r) {
    return *reinterpret_cast<const uint32_t *>(s + t1 * 32 + t0 * 4 + r * 16);
}
__device__ __forceinline__ uint32_t fsa(const uint8_t * p, int u) {
    return *reinterpret_cast<const uint32_t *>(p + u * 4);
}
__device__ __forceinline__ void cpa(uint8_t * d, const uint8_t * s) {
    uint32_t i = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n" :: "r"(i), "l"(s));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void waitg() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

template <int STAGES, int WARPS, int MT = 1>
__global__ void warpsplit_kernel_f32out(
        const uint8_t * __restrict__ A, const uint8_t * __restrict__ B,
        const uint8_t * __restrict__ SFA, const uint8_t * __restrict__ SFB,
        float * __restrict__ D, float alpha, int N, int K) {
    constexpr int M = MT;
#if defined(FR_WS_SM120A_OK)
    pdl_lc(); pdl_sync();
    __shared__ uint8_t sA[WARPS][STAGES][16 * 32];
    __shared__ uint8_t sSFA[WARPS][STAGES][16 * 4];
    __shared__ uint8_t sB[WARPS][STAGES][8 * 32];
    __shared__ uint8_t sSFB[WARPS][STAGES][8 * 4];
    __shared__ float s_red[WARPS][4 * 8];

    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int my_n = blockIdx.x * 8;
    const int KI = K / 64, KIw = KI / WARPS;
    const int kt0 = warp * KIw;
    const int KH = K / 2, ncs = (K / 16 + 3) / 4;
    int t0 = lane & 3, t1 = lane >> 2, sau = (lane & 1) * 8 + (lane >> 2), sbu = lane >> 2;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    uint8_t (*mA)[16 * 32] = sA[warp];
    uint8_t (*mSFA)[16 * 4] = sSFA[warp];
    uint8_t (*mB)[8 * 32] = sB[warp];
    uint8_t (*mSFB)[8 * 4] = sSFB[warp];

    if (lane >= 1 && lane < 16) {
#pragma unroll
        for (int st = 0; st < STAGES; ++st) {
            int4 * av = reinterpret_cast<int4 *>(mA[st]); int4 z{0, 0, 0, 0};
            av[lane * 2] = z; av[lane * 2 + 1] = z;
        }
        if (lane < 4) for (int st = 0; st < STAGES; ++st)
            for (int i = 4 + lane; i < 64; i += 4) mSFA[st][i] = 0;
    }
    __syncwarp();   // M>1: row-1 cp.async below must not race the zero-init
    auto ld = [&](int bf, int kt) {
        int bo = kt * 32;
        if (lane < 8) cpa(mA[bf] + lane * 4, A + bo + lane * 4);
        if (lane == 0) cpa(mSFA[bf], SFA + kt * 512);
#pragma unroll
        for (int rr = 1; rr < MT; ++rr) {   // extra token rows: act tile + atom-layout scales (row r -> +r*16)
            if (lane < 8) cpa(mA[bf] + rr * 32 + lane * 4, A + (size_t) rr * KH + bo + lane * 4);
            if (lane == 0) cpa(mSFA[bf] + rr * 4, SFA + kt * 512 + rr * 16);
        }
        for (int c = 0; c < 2; ++c) { int ch = lane + c * 32, col = ch >> 3, off = ch & 7;
            cpa(mB[bf] + ch * 4, B + (size_t)(my_n + col) * KH + bo + off * 4); }
        if (lane < 8) { int col = my_n + lane, rb = col >> 7, ri = col & 127;
            int si = rb * ncs + kt, ib = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
            cpa(mSFB[bf] + lane * 4, SFB + (size_t)si * 512 + ib); }
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) { if (st < KIw) ld(st, kt0 + st); commit(); }
    for (int j = 0; j < KIw; ++j) {
        int cb = j % STAGES, jp = j + STAGES - 1;
        if (jp < KIw) ld(jp % STAGES, kt0 + jp);
        commit(); waitg<STAGES - 1>(); __syncwarp();
        uint32_t a0 = fa(mA[cb], t0, t1, 0), a1 = fa(mA[cb], t0, t1, 1);
        uint32_t a2 = fa(mA[cb], t0, t1, 2), a3 = fa(mA[cb], t0, t1, 3);
        uint32_t b0 = fb(mB[cb], t0, t1, 0), b1 = fb(mB[cb], t0, t1, 1);
        uint32_t sfa_v = fsa(mSFA[cb], sau), sfb_v = fsa(mSFB[cb], sbu);
        float d0, d1, d2, d3;
        AtomType::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa_v, sfb_v);
        c0 = d0; c1 = d1; c2 = d2; c3 = d3;
    }
    // m16n8 C fragment: {c0,c1} hold row (lane>>2) -> token t lives in lanes 4t..4t+3.
    int q = lane >> 2, r = lane & 3;
    if (q < M) { s_red[warp][q * 8 + r * 2] = c0; s_red[warp][q * 8 + r * 2 + 1] = c1; }
    __syncthreads();
    if (warp == 0 && lane < 8) {
        int col = my_n + lane;
        if (col < N) {
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                float acc = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) acc += s_red[w][t * 8 + lane];
                D[(size_t) t * N + col] = acc * alpha;
            }
        }
    }
#endif // FR_WS_SM120A_OK
}

template <int STAGES, int WARPS, int MT>
int launch(const uint8_t * A, const uint8_t * B, const uint8_t * SFA,
           const uint8_t * SFB, float * D, float alpha, int N, int K,
           bool pdl, cudaStream_t stream) {
    const dim3 grid(N / 8), block(WARPS * 32);
    if (pdl) {
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0;
        cfg.stream = stream; cfg.attrs = &attr; cfg.numAttrs = 1;
        return (int) cudaLaunchKernelEx(&cfg, warpsplit_kernel_f32out<STAGES, WARPS, MT>,
            A, B, SFA, SFB, D, alpha, N, K);
    }
    warpsplit_kernel_f32out<STAGES, WARPS, MT><<<grid, block, 0, stream>>>(
        A, B, SFA, SFB, D, alpha, N, K);
    return (int) cudaGetLastError();
}

template <int STAGES, int WARPS>
int launch_m(const uint8_t * A, const uint8_t * B, const uint8_t * SFA,
             const uint8_t * SFB, float * D, float alpha, int M, int N, int K,
             bool pdl, cudaStream_t stream) {
    switch (M) {
        case 1: return launch<STAGES, WARPS, 1>(A, B, SFA, SFB, D, alpha, N, K, pdl, stream);
        case 2: return launch<STAGES, WARPS, 2>(A, B, SFA, SFB, D, alpha, N, K, pdl, stream);
        case 3: return launch<STAGES, WARPS, 3>(A, B, SFA, SFB, D, alpha, N, K, pdl, stream);
        default: return launch<STAGES, WARPS, 4>(A, B, SFA, SFB, D, alpha, N, K, pdl, stream);
    }
}

}  // namespace

int fp4_w4a4_mma_sm120_warpsplit_mrows_f32out(
    const void * A_packed, const void * B_packed, float * D, int M, int N,
    int K, const void * SFA, const void * SFB, float alpha, int warps,
    int stages, bool pdl, cudaStream_t stream) {
    if (M < 1 || M > 4 || N % 8 != 0 || K % 64 != 0) return -1;
    if ((K / 64) % warps != 0) return -1;
    const uint8_t * A = (const uint8_t *) A_packed;
    const uint8_t * B = (const uint8_t *) B_packed;
    const uint8_t * sfa = (const uint8_t *) SFA;
    const uint8_t * sfb = (const uint8_t *) SFB;
    const int cfg = stages * 10 + warps;
    switch (cfg) {
        case 34: return launch_m<3, 4>(A, B, sfa, sfb, D, alpha, M, N, K, pdl, stream);
        case 44: return launch_m<4, 4>(A, B, sfa, sfb, D, alpha, M, N, K, pdl, stream);
        case 64: return launch_m<6, 4>(A, B, sfa, sfb, D, alpha, M, N, K, pdl, stream);
        case 38: return launch_m<3, 8>(A, B, sfa, sfb, D, alpha, M, N, K, pdl, stream);
        case 48: return launch_m<4, 8>(A, B, sfa, sfb, D, alpha, M, N, K, pdl, stream);
        case 32: return launch_m<3, 2>(A, B, sfa, sfb, D, alpha, M, N, K, pdl, stream);
        case 62: return launch_m<6, 2>(A, B, sfa, sfb, D, alpha, M, N, K, pdl, stream);
        case 42: return launch_m<4, 2>(A, B, sfa, sfb, D, alpha, M, N, K, pdl, stream);
        default: return -1;
    }
}

}  // namespace gemm
}  // namespace flash_rt
