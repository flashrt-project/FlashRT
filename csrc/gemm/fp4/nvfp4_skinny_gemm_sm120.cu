// SPDX-License-Identifier: Apache-2.0
// Skinny NVFP4 (W4A4) GEMM family on sm_120a. See the header.
#include "nvfp4_skinny_gemm_sm120.cuh"
#include "nvfp4_skinny_common.cuh"

namespace flash_rt {
namespace nvfp4_skinny {
namespace {

// One CTA: BN weight rows x KC of K for MT 16-row tiles (blockIdx.z picks the row group).
template <int BN, int KC, int NW, int MT, bool ABF16, bool PDL>
__global__ void __launch_bounds__(NW * 32)
gemm_kernel(const void* __restrict__ A_, const uint8_t* __restrict__ SFA, const float* __restrict__ a_gs,
            const uint8_t* __restrict__ W, const uint8_t* __restrict__ SFB, float* __restrict__ partials, int M,
            int N, int K) {
    constexpr int THREADS = NW * 32;
    constexpr int NT = BN / 8;
    constexpr int NTW = NT / NW;
    constexpr int KCH = KC / 128;
    constexpr int ROWS = 16 * MT;
    constexpr int AB = KC / 2;        // e2m1 bytes per row
    constexpr int ASTR = AB + 16;
    constexpr int SB = KC / 16;       // scale bytes per row
    constexpr int SSTR = SB + 4;
    static_assert(NT % NW == 0, "n8 tiles must split across warps");
    static_assert(KC % 128 == 0, "K chunk must be a multiple of 128");
    __shared__ __align__(16) uint8_t sA[ROWS * ASTR];
    __shared__ __align__(16) uint8_t sS[ROWS * SSTR];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, t0 = lane & 3, t1 = lane >> 2;
    const int n0 = blockIdx.x * BN;
    const int kb = blockIdx.y * KC;
    const int m0 = blockIdx.z * ROWS;
    const int kchunks = K / 128;

    uint4 w[KCH][NTW];
    uint2 sb[KCH][NTW];
#pragma unroll
    for (int c = 0; c < KCH; ++c)
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const size_t tile = static_cast<size_t>(n0 / 8 + warp * NTW + t) * kchunks + kb / 128 + c;
            w[c][t] = ldg_stream(W + (tile * 32 + lane) * 16);
            sb[c][t] = ldg_stream8(SFB + (tile * 8 + t1) * 8);
        }

    if constexpr (PDL) pdl_prologue();

    if constexpr (!ABF16) {
        const uint8_t* A = static_cast<const uint8_t*>(A_);
        for (int c = tid; c < ROWS * (AB / 16); c += THREADS) {
            const int row = c / (AB / 16);
            const int off = (c % (AB / 16)) * 16;
            const bool v = m0 + row < M;
            cp_async_16(sA + row * ASTR + off, A + static_cast<size_t>(v ? m0 + row : 0) * (K / 2) + kb / 2 + off, v);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        for (int c = tid; c < ROWS * (SB / 4); c += THREADS) {
            const int row = c / (SB / 4);
            const int off = (c % (SB / 4)) * 4;
            uint32_t s = 0;
            if (m0 + row < M)
                s = *reinterpret_cast<const uint32_t*>(SFA + static_cast<size_t>(m0 + row) * (K / 16) + kb / 16 + off);
            *reinterpret_cast<uint32_t*>(sS + row * SSTR + off) = s;
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
    } else {
        const __nv_bfloat16* A = static_cast<const __nv_bfloat16*>(A_);
        const float inv_gs = 1.0f / (*a_gs);
        for (int b = tid; b < ROWS * SB; b += THREADS) {
            const int row = b / SB;
            const int blk = b % SB;
            uint8_t* dst = sA + row * ASTR + blk * 8;
            if (m0 + row < M) {
                const __nv_bfloat16* src = A + static_cast<size_t>(m0 + row) * K + kb + blk * 16;
                const uint4 p0 = *reinterpret_cast<const uint4*>(src);
                const uint4 p1 = *reinterpret_cast<const uint4*>(src + 8);
                const uint32_t* h0 = reinterpret_cast<const uint32_t*>(&p0);
                const uint32_t* h1 = reinterpret_cast<const uint32_t*>(&p1);
                float v[16];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f0 = bf2(h0[i]);
                    const float2 f1 = bf2(h1[i]);
                    v[2 * i] = f0.x * inv_gs; v[2 * i + 1] = f0.y * inv_gs;
                    v[8 + 2 * i] = f1.x * inv_gs; v[8 + 2 * i + 1] = f1.y * inv_gs;
                }
                quant_block16(v, dst, sS + row * SSTR + blk);
            } else {
                *reinterpret_cast<uint2*>(dst) = make_uint2(0, 0);
                sS[row * SSTR + blk] = 0;
            }
        }
    }
    __syncthreads();

    float acc[MT][NTW][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int t = 0; t < NTW; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][t][i] = 0.f;

    const int sau = (lane & 1) * 8 + (lane >> 2);   // scale-A row owned by this lane (within a 16-row tile)
#pragma unroll
    for (int c = 0; c < KCH; ++c) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint8_t* ra = sA + (mt * 16 + t1) * ASTR + c * 64 + s * 32 + t0 * 4;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ra);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ra + 8 * ASTR);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ra + 16);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ra + 8 * ASTR + 16);
                const uint32_t sfa = *reinterpret_cast<const uint32_t*>(sS + (mt * 16 + sau) * SSTR + c * 8 + s * 4);
#pragma unroll
                for (int t = 0; t < NTW; ++t) {
                    const uint32_t b0 = s == 0 ? w[c][t].x : w[c][t].z;
                    const uint32_t b1 = s == 0 ? w[c][t].y : w[c][t].w;
                    const uint32_t sfb = s == 0 ? sb[c][t].x : sb[c][t].y;
                    mma_nvf4(acc[mt][t], a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }
        }
    }

    float* base = partials + static_cast<size_t>(blockIdx.y) * M * N;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const int col = n0 + (warp * NTW + t) * 8 + 2 * t0;
            const int row0 = m0 + mt * 16 + t1;
            if (row0 < M)
                *reinterpret_cast<float2*>(base + static_cast<size_t>(row0) * N + col) =
                    make_float2(acc[mt][t][0], acc[mt][t][1]);
            if (row0 + 8 < M)
                *reinterpret_cast<float2*>(base + static_cast<size_t>(row0 + 8) * N + col) =
                    make_float2(acc[mt][t][2], acc[mt][t][3]);
        }
}

// One thread per (8-column tile, 128-wide K chunk, lane): gathers the lane's
// 16 bytes; lanes 0..7 also gather the column scales.
__global__ void pack_weight_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sf,
                                   uint8_t* __restrict__ wp, uint8_t* __restrict__ sfp, int N, int K) {
    const int kchunks = K / 128;
    const size_t total = static_cast<size_t>(N / 8) * kchunks * 32;
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= total) return;
    const int lane = static_cast<int>(g & 31);
    const size_t tile = g >> 5;
    const int kc = static_cast<int>(tile % kchunks);
    const int nt = static_cast<int>(tile / kchunks);
    const int t0 = lane & 3, t1 = lane >> 2;
    const int col = nt * 8 + t1;
    const uint8_t* src = w + static_cast<size_t>(col) * (K / 2) + kc * 64;
    uint32_t piece[4];
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int r = 0; r < 2; ++r)
            piece[2 * s + r] = *reinterpret_cast<const uint32_t*>(src + s * 32 + r * 16 + t0 * 4);
    *reinterpret_cast<uint4*>(wp + g * 16) = make_uint4(piece[0], piece[1], piece[2], piece[3]);
    if (lane < 8) {
        const uint8_t* ssrc = sf + static_cast<size_t>(nt * 8 + lane) * (K / 16) + kc * 8;
        *reinterpret_cast<uint2*>(sfp + (tile * 8 + lane) * 8) = *reinterpret_cast<const uint2*>(ssrc);
    }
}

__device__ __forceinline__ float e2m1_value(float x) {
    // value of the e2m1 code that cvt.rn.satfinite selects for x
    const float a = fminf(fabsf(x), 6.0f);
    const float q = a < 2.0f ? rintf(a * 2.0f) * 0.5f : (a < 4.0f ? rintf(a) : rintf(a * 0.5f) * 2.0f);
    return copysignf(q, x);
}

// One thread per 16-element block of a BF16 [N, K] weight. With mse_steps > 0
// the block scale is also tried mse_steps UE4M3 codes below the ceil scale
// (clipping the block maximum) and the code with the smallest squared error
// is kept.
__global__ void quantize_weight_kernel(const __nv_bfloat16* __restrict__ w, const float* __restrict__ gs,
                                       uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf, int N, int K,
                                       int mse_steps) {
    const size_t blocks = static_cast<size_t>(N) * (K / 16);
    const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const float inv_gs = 1.0f / (*gs);
    const __nv_bfloat16* src = w + b * 16;
    float v[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = __bfloat162float(src[i]) * inv_gs;
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const uint8_t base = ue4m3_ceil(amax * (1.0f / 6.0f));
    uint8_t best = base;
    if (mse_steps > 0 && base > 1) {
        float best_err = 3.4e38f;
        for (int k = 0; k <= mse_steps && base - k >= 1; ++k) {
            const uint8_t code = static_cast<uint8_t>(base - k);
            const float d = ue4m3_value(code);
            const float inv = 1.0f / d;
            float err = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float e = v[i] - e2m1_value(v[i] * inv) * d;
                err += e * e;
            }
            if (err < best_err) { best_err = err; best = code; }
        }
    }
    const float d = ue4m3_value(best);
    const float inv = d > 0.f ? 1.0f / d : 0.f;
    uint16_t h[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
        h[q] = e2m1x4(v[4 * q] * inv, v[4 * q + 1] * inv, v[4 * q + 2] * inv, v[4 * q + 3] * inv);
    *reinterpret_cast<uint2*>(out + b * 8) = make_uint2(h[0] | (static_cast<uint32_t>(h[1]) << 16),
                                                        h[2] | (static_cast<uint32_t>(h[3]) << 16));
    out_sf[b] = best;
}

struct Cfg { int bn, kc, nw; };
constexpr Cfg kCfgs[] = {
    {64, 256, 4},  {64, 512, 4},  {128, 256, 4}, {128, 512, 4},
    {64, 1024, 4}, {128, 1024, 4}, {128, 256, 8}, {256, 256, 8},
    {64, 128, 4},  {32, 512, 4},  {32, 1024, 4}, {256, 512, 8},
    {32, 256, 4},  {32, 128, 4},  {64, 128, 8},  {128, 128, 8},
    {32, 2048, 4}, {64, 2048, 8}, {128, 2048, 8}, {256, 1024, 8},
    {64, 1024, 8}, {32, 1024, 2},
};
constexpr int kNumCfgs = sizeof(kCfgs) / sizeof(kCfgs[0]);

template <int BN, int KC, int NW, int MT, bool ABF16>
int launch_mt(const void* A, const uint8_t* SFA, const float* a_gs, const uint8_t* W, const uint8_t* SFB,
              float* partials, int M, int N, int K, bool pdl, cudaStream_t stream) {
    const dim3 grid(N / BN, K / KC, (M + 16 * MT - 1) / (16 * MT));
    const void* fn = pdl ? (const void*)gemm_kernel<BN, KC, NW, MT, ABF16, true>
                         : (const void*)gemm_kernel<BN, KC, NW, MT, ABF16, false>;
    return launch_ex(fn, grid, dim3(NW * 32), stream, pdl, A, SFA, a_gs, W, SFB, partials, M, N, K);
}

template <int BN, int KC, int NW, bool ABF16>
int launch_gemm(const void* A, const void* SFA, const float* a_gs, const void* W, const void* SFB,
                float* partials, int M, int N, int K, bool pdl, cudaStream_t stream) {
    if (M < 1 || N % BN != 0 || K % KC != 0) return static_cast<int>(cudaErrorInvalidValue);
    const uint8_t* sfa = static_cast<const uint8_t*>(SFA);
    const uint8_t* w = static_cast<const uint8_t*>(W);
    const uint8_t* sfb = static_cast<const uint8_t*>(SFB);
    if (M <= 16) return launch_mt<BN, KC, NW, 1, ABF16>(A, sfa, a_gs, w, sfb, partials, M, N, K, pdl, stream);
    if (M <= 32) return launch_mt<BN, KC, NW, 2, ABF16>(A, sfa, a_gs, w, sfb, partials, M, N, K, pdl, stream);
    return launch_mt<BN, KC, NW, 3, ABF16>(A, sfa, a_gs, w, sfb, partials, M, N, K, pdl, stream);
}

template <bool ABF16>
int dispatch_gemm(const void* A, const void* SFA, const float* a_gs, const void* W, const void* SFB,
                  float* partials, int M, int N, int K, int cfg, bool pdl, cudaStream_t stream) {
#define CASE(I, BN, KC, NW) \
    case I: return launch_gemm<BN, KC, NW, ABF16>(A, SFA, a_gs, W, SFB, partials, M, N, K, pdl, stream)
    switch (cfg) {
        CASE(0, 64, 256, 4);
        CASE(1, 64, 512, 4);
        CASE(2, 128, 256, 4);
        CASE(3, 128, 512, 4);
        CASE(4, 64, 1024, 4);
        CASE(5, 128, 1024, 4);
        CASE(6, 128, 256, 8);
        CASE(7, 256, 256, 8);
        CASE(8, 64, 128, 4);
        CASE(9, 32, 512, 4);
        CASE(10, 32, 1024, 4);
        CASE(11, 256, 512, 8);
        CASE(12, 32, 256, 4);
        CASE(13, 32, 128, 4);
        CASE(14, 64, 128, 8);
        CASE(15, 128, 128, 8);
        CASE(16, 32, 2048, 4);
        CASE(17, 64, 2048, 8);
        CASE(18, 128, 2048, 8);
        CASE(19, 256, 1024, 8);
        CASE(20, 64, 1024, 8);
        CASE(21, 32, 1024, 2);
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
#undef CASE
}

}  // namespace

int config_count() { return kNumCfgs; }

int config_k_chunk(int cfg) { return (cfg >= 0 && cfg < kNumCfgs) ? kCfgs[cfg].kc : 0; }

bool config_supports(int cfg, int N, int K) {
    return cfg >= 0 && cfg < kNumCfgs && N % kCfgs[cfg].bn == 0 && K % kCfgs[cfg].kc == 0;
}

int gemm(const void* A, const void* SFA, const void* W, const void* SFB, float* partials, int M, int N, int K,
         int cfg, bool pdl, cudaStream_t stream) {
    return dispatch_gemm<false>(A, SFA, nullptr, W, SFB, partials, M, N, K, cfg, pdl, stream);
}

int gemm_bf16_act(const __nv_bfloat16* A, const float* a_gs, const void* W, const void* SFB, float* partials,
                  int M, int N, int K, int cfg, bool pdl, cudaStream_t stream) {
    return dispatch_gemm<true>(A, nullptr, a_gs, W, SFB, partials, M, N, K, cfg, pdl, stream);
}

int pack_weight(const uint8_t* w, const uint8_t* sf, uint8_t* w_packed, uint8_t* sf_packed, int N, int K,
                cudaStream_t stream) {
    if (N % 8 != 0 || K % 128 != 0 || N < 8) return static_cast<int>(cudaErrorInvalidValue);
    const size_t total = static_cast<size_t>(N / 8) * (K / 128) * 32;
    pack_weight_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(w, sf, w_packed, sf_packed, N, K);
    return static_cast<int>(cudaGetLastError());
}

int quantize_weight(const __nv_bfloat16* w, const float* gs, uint8_t* out, uint8_t* out_sf, int N, int K,
                    int mse_steps, cudaStream_t stream) {
    if (N < 1 || K % 16 != 0 || mse_steps < 0 || mse_steps > 16) return static_cast<int>(cudaErrorInvalidValue);
    const size_t blocks = static_cast<size_t>(N) * (K / 16);
    quantize_weight_kernel<<<static_cast<unsigned>((blocks + 255) / 256), 256, 0, stream>>>(w, gs, out, out_sf, N, K,
                                                                                            mse_steps);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nvfp4_skinny
}  // namespace flash_rt
