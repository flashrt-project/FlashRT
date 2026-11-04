// SPDX-License-Identifier: Apache-2.0
//
// M=128 NVFP4-weight GEMM for the Wan2.2 video stream; see
// fp4_stream_m128_sm89.cuh for the contract and the F16 block-sum bound.
//
// A CTA owns 16*MT rows of A and WARPS column units of NU n8 tiles. Per
// 128-K block the CTA stages its A rows (and their activation scales) in a
// STAGES-deep cp.async ring; each warp keeps a DEPTH-deep register ring of its
// fragment-major weight loads, decodes them through the per-row LUT, and runs
// MT x NU MMAs per k32 slice against ldmatrix'd A fragments.

#include "fp4_stream_m128_sm89.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <type_traits>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

namespace {

__device__ __forceinline__ void mma_f32_(float* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16_(uint32_t* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4_(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                         uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void cp16_(uint32_t smem, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(g));
}

__device__ __forceinline__ void cp4_(uint32_t smem, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem), "l"(g));
}

__device__ __forceinline__ void cp_commit_() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N_>
__device__ __forceinline__ void cp_wait_() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_));
}

// ue4m3_sf_to_fp32 of the shared-memory kernels, built from the bits.
__device__ __forceinline__ float ue4m3_sf_(uint32_t v) {
    const uint32_t e = (v >> 3) & 0xF, m = v & 0x7;
    return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
}

__device__ __forceinline__ uint4 ldg_stream_(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t prmt_(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}

__device__ __forceinline__ uint32_t cvt2_(float a, float b) {
    return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE,
                                              __NV_E4M3);
}

__device__ __forceinline__ uint32_t dec4_(uint32_t w, uint32_t P0, uint32_t P1,
                                          uint32_t N0, uint32_t N1) {
    const uint32_t sel = w & 0x7777u;
    const uint32_t p = prmt_(P0, P1, sel);
    const uint32_t n = prmt_(N0, N1, sel);
    return prmt_(p, n, ((w >> 1) & 0x4444u) | 0x3210u);
}

__device__ __forceinline__ uint32_t comp_(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int MT, int NU, int WARPS, int DEPTH, int STAGES, bool F16>
__global__ __launch_bounds__(WARPS * 32, 1)
void fp4_stream_m128_kernel(const uint8_t* __restrict__ A,
                            const uint4* __restrict__ Wf,
                            const uint8_t* __restrict__ Bsf,
                            const float* __restrict__ gscale,
                            const float* __restrict__ act_scale,
                            const float* __restrict__ w_scale,
                            __nv_bfloat16* __restrict__ D,
                            int N, int K) {
    static_assert(DEPTH % 2 == 0, "the accumulator double buffer alternates per k-block");
    constexpr int ROWS = 16 * MT;
    constexpr int A_BYTES = ROWS * 128;
    constexpr int STAGE_BYTES = A_BYTES + ROWS * 4;
    constexpr int CW = F16 ? 2 : 4;          // 32-bit words per MMA accumulator
    extern __shared__ __align__(128) uint8_t smem[];
    const int K128 = K >> 7;
    const int NT = N >> 3;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * ROWS;
    const int unit = blockIdx.x * WARPS + warp;
    const bool active = unit * NU < NT;
    const float gs = gscale[0];
    // ldmatrix.x4 lane groups -> (rows 0-7, k 0-15), (rows 8-15, k 0-15),
    // (rows 0-7, k 16-31), (rows 8-15, k 16-31): the m16n8k32 A fragment order
    // a0..a3, so the four results land in the MMA's register quad directly.
    const int arow = (((lane >> 3) & 1) << 3) + (lane & 7);
    const int acol = lane >> 4;
    const uint32_t s_base = (uint32_t)__cvta_generic_to_shared(smem);

    // Per-thread A staging addresses, fixed for the whole K walk: each
    // thread moves CPT 16-byte chunks per block at constant smem offsets.
    // WARPS=8/16 divide the 1024 chunks exactly; the 6-warp CTA (96-column
    // tile: 32 CTAs at N=3072, 96 at N=9216, i.e. 94% wave fill on 34 SMs
    // instead of 71%) does not, so its tail chunk is predicated.
    constexpr int A_CHUNKS = ROWS * 8;
    constexpr int CPT = (A_CHUNKS + WARPS * 32 - 1) / (WARPS * 32);
    static_assert(CPT * WARPS * 32 >= A_CHUNKS, "A tile must fit over the CTA");
    const uint8_t* a_src[CPT];
    uint32_t a_dst[CPT];
    bool a_ok[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
        const int c = threadIdx.x + i * WARPS * 32;
        a_ok[i] = c < A_CHUNKS;
        const int row = (a_ok[i] ? c : 0) >> 3, ch = c & 7;
        a_src[i] = A + (size_t)(m0 + row) * K + ch * 16;
        a_dst[i] = row * 128 + ((ch ^ (row & 7)) << 4);
    }
    const bool s_thr = threadIdx.x < ROWS;
    const float* s_src = act_scale + (size_t)(m0 + (s_thr ? threadIdx.x : 0)) * K128;
    const uint32_t s_dst = A_BYTES + threadIdx.x * 4;

    auto load_stage = [&](uint32_t stage_off, int kb) {
        const uint32_t sb = s_base + stage_off;
#pragma unroll
        for (int i = 0; i < CPT; ++i)
            if (a_ok[i]) cp16_(sb + a_dst[i], a_src[i] + kb * 128);
        if (s_thr) cp4_(sb + s_dst, s_src + kb);
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < K128) load_stage(s * STAGE_BYTES, s);
        cp_commit_();
    }
    // ldmatrix lane offsets per k32 slice; the m-tile step is a constant 2 KB
    uint32_t ld_off[4];
#pragma unroll
    for (int ka = 0; ka < 4; ++ka)
        ld_off[ka] = arow * 128 + (((2 * ka + acol) ^ (arow & 7)) << 4);

    const int tile0 = unit * NU;
    const uint4* wp[NU];
    const uint8_t* sfp[NU];
#pragma unroll
    for (int u = 0; u < NU; ++u) {
        const int tl = active ? tile0 + u : 0;
        const int n = tl * 8 + g;
        wp[u] = Wf + (size_t)tl * K128 * 32 + lane;
        sfp[u] = Bsf + (size_t)(n >> 7) * K128 * 128 + (n & 127);
    }
    const float* wsp = w_scale + (size_t)(((active ? tile0 : 0) * 8) >> 7) * K128;

    uint4 ring[DEPTH][NU];
    uint32_t rsf[DEPTH][NU];
    float rws[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
        if (active) {
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                ring[d][u] = ldg_stream_(wp[u] + d * 32);
                rsf[d][u] = __ldg(sfp[u] + d * 128);
            }
            rws[d] = __ldg(wsp + d);
        }
    }

    float acc[MT][NU][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int u = 0; u < NU; ++u)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][u][j] = 0.f;

    // Double-buffered block accumulators: block kb's fold runs after block
    // kb+1's MMAs are issued, so its ALU work hides their latency.
    uint32_t c[2][MT][NU][CW];
    float asv[2][MT][2];
    float cfv[2][NU][2];

    auto fold = [&](auto bc, int mlo, int mhi) {
        constexpr int b = decltype(bc)::value;
#pragma unroll
        for (int m = mlo; m < mhi; ++m)
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                float v[4];
                if (F16) {
                    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&c[b][m][u][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&c[b][m][u][1]));
                    v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] = __uint_as_float(c[b][m][u][j]);
                }
                acc[m][u][0] += v[0] * (asv[b][m][0] * cfv[b][u][0]);
                acc[m][u][1] += v[1] * (asv[b][m][0] * cfv[b][u][1]);
                acc[m][u][2] += v[2] * (asv[b][m][1] * cfv[b][u][0]);
                acc[m][u][3] += v[3] * (asv[b][m][1] * cfv[b][u][1]);
            }
    };

    uint32_t st_rd = 0, st_wr = (STAGES - 1) * STAGE_BYTES;
    for (int kb0 = 0; kb0 < K128; kb0 += DEPTH) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            const int kb = kb0 + d;
            const int b = d & 1;
            cp_wait_<STAGES - 2>();
            __syncthreads();
            {
                const int nk = kb + STAGES - 1;
                if (nk < K128) load_stage(st_wr, nk);
                st_wr = (st_wr + STAGE_BYTES == STAGES * STAGE_BYTES) ? 0 : st_wr + STAGE_BYTES;
                cp_commit_();
            }
            if (!active) continue;
            uint4 wv[NU];
            uint32_t sfv[NU];
#pragma unroll
            for (int u = 0; u < NU; ++u) { wv[u] = ring[d][u]; sfv[u] = rsf[d][u]; }
            const float wsv = rws[d];
            {
                const int nk = kb + DEPTH;
                if (nk < K128) {
#pragma unroll
                    for (int u = 0; u < NU; ++u) {
                        ring[d][u] = ldg_stream_(wp[u] + (size_t)nk * 32);
                        rsf[d][u] = __ldg(sfp[u] + (size_t)nk * 128);
                    }
                    rws[d] = __ldg(wsp + nk);
                }
            }
            const uint32_t sb = s_base + st_rd;
            const float* scs = reinterpret_cast<const float*>(smem + st_rd + A_BYTES);
            st_rd = (st_rd + STAGE_BYTES == STAGES * STAGE_BYTES) ? 0 : st_rd + STAGE_BYTES;

            uint32_t P0[NU], P1[NU], N0[NU], N1[NU];
            const float ws_inv = (wsv > 0.0f) ? (1.0f / wsv) : 0.0f;
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                const float s = ue4m3_sf_(sfv[u]) * gs * ws_inv;
                float sh = 1.0f;
                uint32_t e = 0;
                if (F16) {
                    const float amax = fminf(6.0f * s, 448.0f);
                    if (amax > 0.875f) {
                        // e = frexp exponent of amax / 0.875: amax * 2^-e <= 0.875
                        e = ((__float_as_uint(amax * 1.14285714f) >> 23) & 0xFF) - 126u;
                        sh = __uint_as_float((127u - e) << 23);
                    }
                }
                const float v1 = fminf(0.5f * s, 448.0f) * sh, v2 = fminf(s, 448.0f) * sh;
                const float v3 = fminf(1.5f * s, 448.0f) * sh, v4 = fminf(2.0f * s, 448.0f) * sh;
                const float v5 = fminf(3.0f * s, 448.0f) * sh, v6 = fminf(4.0f * s, 448.0f) * sh;
                const float v7 = fminf(6.0f * s, 448.0f) * sh;
                P0[u] = cvt2_(0.0f, v1) | (cvt2_(v2, v3) << 16);
                P1[u] = cvt2_(v4, v5) | (cvt2_(v6, v7) << 16);
                N0[u] = P0[u] ^ 0x80808080u;
                N1[u] = P1[u] ^ 0x80808080u;
                if (F16) {
                    // lane (g, t) holds row g's shift; output columns 2t, 2t+1
                    // belong to the rows held by lanes 8t and 8t + 4
                    const uint32_t e0 = __shfl_sync(0xffffffffu, e, 8 * t);
                    const uint32_t e1 = __shfl_sync(0xffffffffu, e, 8 * t + 4);
                    cfv[b][u][0] = wsv * __uint_as_float((127u + e0) << 23);
                    cfv[b][u][1] = wsv * __uint_as_float((127u + e1) << 23);
                } else {
                    cfv[b][u][0] = wsv;
                    cfv[b][u][1] = wsv;
                }
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                asv[b][m][0] = scs[m * 16 + g];
                asv[b][m][1] = scs[m * 16 + g + 8];
            }
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int u = 0; u < NU; ++u)
#pragma unroll
                    for (int j = 0; j < CW; ++j) c[b][m][u][j] = 0u;
#pragma unroll
            for (int ka = 0; ka < 4; ++ka) {
                uint32_t b0[NU], b1[NU];
#pragma unroll
                for (int u = 0; u < NU; ++u) {
                    const uint32_t w = comp_(wv[u], ka);
                    b0[u] = dec4_(w, P0[u], P1[u], N0[u], N1[u]);
                    b1[u] = dec4_(w >> 16, P0[u], P1[u], N0[u], N1[u]);
                }
                const int chunk = 2 * ka + acol;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    uint32_t a0, a1, a2, a3;
                    ldsm_x4_(a0, a1, a2, a3, sb + ld_off[ka] + m * 2048);
#pragma unroll
                    for (int u = 0; u < NU; ++u) {
                        if (F16) mma_f16_(c[b][m][u], a0, a1, a2, a3, b0[u], b1[u]);
                        else mma_f32_(reinterpret_cast<float*>(c[b][m][u]), a0, a1, a2, a3, b0[u], b1[u]);
                    }
                }
                // a quarter of the previous block's fold per k32 slice, so
                // its ALU work issues between this block's MMAs
                if (kb > 0) {
                    constexpr int MQ = (MT + 3) / 4;
                    if (b) fold(std::integral_constant<int, 0>{}, ka * MQ, (ka + 1) * MQ < MT ? (ka + 1) * MQ : MT);
                    else fold(std::integral_constant<int, 1>{}, ka * MQ, (ka + 1) * MQ < MT ? (ka + 1) * MQ : MT);
                }
            }
        }
    }
    cp_wait_<0>();

    if (!active) return;
    // K128 % DEPTH == 0 with DEPTH even: the last block always used buffer 1
    fold(std::integral_constant<int, 1>{}, 0, MT);
#pragma unroll
    for (int u = 0; u < NU; ++u) {
        const int col = (tile0 + u) * 8 + 2 * t;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r = m0 + m * 16 + g;
            *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)r * N + col]) =
                __floats2bfloat162_rn(acc[m][u][0], acc[m][u][1]);
            *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)(r + 8) * N + col]) =
                __floats2bfloat162_rn(acc[m][u][2], acc[m][u][3]);
        }
    }
}

template <int MT, int NU, int WARPS, int DEPTH, int STAGES, bool F16>
int launch_(const void* A, const void* Wf, const void* Bsf, const float* gscale,
            const float* act_scale, const float* w_scale, void* D, int M, int N,
            int K, cudaStream_t s) {
    auto kfn = fp4_stream_m128_kernel<MT, NU, WARPS, DEPTH, STAGES, F16>;
    constexpr int smem = STAGES * (16 * MT * 128 + 16 * MT * 4);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute((const void*)kfn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    const int units = (N / 8) / NU;
    dim3 grid((units + WARPS - 1) / WARPS, M / (16 * MT));
    kfn<<<grid, WARPS * 32, smem, s>>>(
        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint4*>(Wf),
        reinterpret_cast<const uint8_t*>(Bsf), gscale, act_scale, w_scale,
        reinterpret_cast<__nv_bfloat16*>(D), N, K);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// LUT-image variant. The per-(weight row, K-block) decode LUT and fold factor
// depend only on static weight data (SF128 byte, global scale, w_scale), so
// nvfp4_m128_lut_build computes them once with the exact expressions of the
// kernel above and stores them as a 96-byte record per (n8 tile, K-block):
// 8 rows x {P0, P1} (the positive e4m3 LUT) then 8 rows x fp32 fold factor.
// The GEMM then stages A, the weight fragments and the LUT records through one
// STAGES-deep cp.async ring, and its CTA is UNITS 16-column units x MSPLIT
// row slices (MT = 8 / MSPLIT m16 tiles per warp). Per output element the
// arithmetic -- decoded B bytes, F16 block sums, fold order -- is that of
// fp4_stream_m128_kernel, so the result is bit-identical; only the LUT
// construction (~60 ALU ops per row and block) leaves the main loop.

__global__ void nvfp4_m128_lut_kernel(const uint8_t* __restrict__ Bsf,
                                      const float* __restrict__ gscale,
                                      const float* __restrict__ w_scale,
                                      uint8_t* __restrict__ lut, int N, int K128) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // (tile, kb, row)
    if (idx >= N * K128) return;
    const int r = idx & 7, rest = idx >> 3;
    const int kb = rest % K128, tl = rest / K128;
    const int n = tl * 8 + r;
    const float gs = gscale[0];
    const float wsv = w_scale[(size_t)(n >> 7) * K128 + kb];
    const uint32_t sfv = Bsf[(size_t)(n >> 7) * K128 * 128 + (size_t)kb * 128 + (n & 127)];
    const float ws_inv = (wsv > 0.0f) ? (1.0f / wsv) : 0.0f;
    const float s = ue4m3_sf_(sfv) * gs * ws_inv;
    float sh = 1.0f;
    uint32_t e = 0;
    const float amax = fminf(6.0f * s, 448.0f);
    if (amax > 0.875f) {
        e = ((__float_as_uint(amax * 1.14285714f) >> 23) & 0xFF) - 126u;
        sh = __uint_as_float((127u - e) << 23);
    }
    const float v1 = fminf(0.5f * s, 448.0f) * sh, v2 = fminf(s, 448.0f) * sh;
    const float v3 = fminf(1.5f * s, 448.0f) * sh, v4 = fminf(2.0f * s, 448.0f) * sh;
    const float v5 = fminf(3.0f * s, 448.0f) * sh, v6 = fminf(4.0f * s, 448.0f) * sh;
    const float v7 = fminf(6.0f * s, 448.0f) * sh;
    uint8_t* o = lut + ((size_t)tl * K128 + kb) * 96;
    reinterpret_cast<uint32_t*>(o)[r * 2] = cvt2_(0.0f, v1) | (cvt2_(v2, v3) << 16);
    reinterpret_cast<uint32_t*>(o)[r * 2 + 1] = cvt2_(v4, v5) | (cvt2_(v6, v7) << 16);
    reinterpret_cast<float*>(o + 64)[r] = wsv * __uint_as_float((127u + e) << 23);
}

template <int UNITS, int MSPLIT, int STAGES>
struct LutGeo {
    static constexpr int WARPS = UNITS * MSPLIT;
    static constexpr int MT = 8 / MSPLIT;
    static constexpr int AS_OFF = 128 * 128;                 // 128 fp32 act scales
    static constexpr int W_OFF = AS_OFF + 128 * 4;           // UNITS*2 tiles x 512 B
    static constexpr int LUT_OFF = W_OFF + UNITS * 2 * 512;  // UNITS*2 tiles x 96 B
    static constexpr int STAGE = ((LUT_OFF + UNITS * 2 * 96) + 127) / 128 * 128;
    static constexpr int SMEM = STAGE * STAGES;
};

template <int UNITS, int MSPLIT, int STAGES, int MINB = 1>
__global__ __launch_bounds__(UNITS * MSPLIT * 32, MINB)
void fp4_stream_m128_lut_kernel(const uint8_t* __restrict__ A,
                                const uint4* __restrict__ Wf,
                                const uint8_t* __restrict__ Lut,
                                const float* __restrict__ act_scale,
                                __nv_bfloat16* __restrict__ D, int N, int K) {
    using G = LutGeo<UNITS, MSPLIT, STAGES>;
    constexpr int MT = G::MT, NU = 2;
    constexpr int THREADS = G::WARPS * 32;
    static_assert(THREADS >= 128, "one thread per act-scale row");
    extern __shared__ __align__(128) uint8_t smem[];
    const int K128 = K >> 7;
    const int NT = N >> 3;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wn = warp % UNITS, wm = warp / UNITS;
    const int m0 = blockIdx.y * 128;
    const int unit0 = blockIdx.x * UNITS;
    const int tile0 = (unit0 + wn) * NU;
    const bool active = tile0 < NT;
    const int arow = (((lane >> 3) & 1) << 3) + (lane & 7);
    const int acol = lane >> 4;
    const uint32_t s_base = (uint32_t)__cvta_generic_to_shared(smem);

    // Per-thread copy plan, fixed for the K walk: A rows (swizzled 16-B
    // chunks), the CTA's weight fragments (512 B per n8 tile and block) and
    // its LUT records (96 B per n8 tile and block).
    constexpr int A_CH = 128 * 8;
    constexpr int W_CH = UNITS * NU * 32;
    constexpr int L_CH = UNITS * NU * 6;
    constexpr int TOT = A_CH + W_CH + L_CH;
    constexpr int CPT = (TOT + THREADS - 1) / THREADS;
    const uint8_t* src[CPT];
    uint32_t dst[CPT];
    uint32_t step[CPT];
    bool ok[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
        const int c = threadIdx.x + i * THREADS;
        ok[i] = c < TOT;
        if (c < A_CH) {
            const int row = c >> 3, ch = c & 7;
            src[i] = A + (size_t)(m0 + row) * K + ch * 16;
            dst[i] = row * 128 + ((ch ^ (row & 7)) << 4);
            step[i] = 128;
        } else if (c < A_CH + W_CH) {
            const int w = c - A_CH;
            const int tl_local = w >> 5, l = w & 31;
            int tl = unit0 * NU + tl_local;
            if (tl >= NT) { tl = 0; ok[i] = false; }
            src[i] = reinterpret_cast<const uint8_t*>(Wf + (size_t)tl * K128 * 32 + l);
            dst[i] = G::W_OFF + tl_local * 512 + l * 16;
            step[i] = 512;
        } else {
            const int w = ok[i] ? c - A_CH - W_CH : 0;
            const int tl_local = w / 6, l = w % 6;
            int tl = unit0 * NU + tl_local;
            if (tl >= NT) { tl = 0; ok[i] = false; }
            src[i] = Lut + (size_t)tl * K128 * 96 + l * 16;
            dst[i] = G::LUT_OFF + tl_local * 96 + l * 16;
            step[i] = 96;
        }
    }
    const bool s_act = threadIdx.x < 128;
    const float* s_act_src = act_scale + (size_t)(m0 + (s_act ? threadIdx.x : 0)) * K128;

    auto load_stage = [&](int slot, int kb) {
        const uint32_t sb = s_base + slot * G::STAGE;
#pragma unroll
        for (int i = 0; i < CPT; ++i)
            if (ok[i]) cp16_(sb + dst[i], src[i] + (size_t)kb * step[i]);
        if (s_act) cp4_(sb + G::AS_OFF + threadIdx.x * 4, s_act_src + kb);
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < K128) load_stage(s, s);
        cp_commit_();
    }
    uint32_t ld_off[4];
#pragma unroll
    for (int ka = 0; ka < 4; ++ka)
        ld_off[ka] = (wm * MT * 16 + arow) * 128 + (((2 * ka + acol) ^ (arow & 7)) << 4);

    float acc[MT][NU][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int u = 0; u < NU; ++u)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][u][j] = 0.f;

    uint32_t c[2][MT][NU][2];
    float asv[2][MT][2];
    float cfv[2][NU][2];

    auto fold = [&](auto bc, int mlo, int mhi) {
        constexpr int b = decltype(bc)::value;
#pragma unroll
        for (int m = mlo; m < mhi; ++m)
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&c[b][m][u][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&c[b][m][u][1]));
                acc[m][u][0] += lo.x * (asv[b][m][0] * cfv[b][u][0]);
                acc[m][u][1] += lo.y * (asv[b][m][0] * cfv[b][u][1]);
                acc[m][u][2] += hi.x * (asv[b][m][1] * cfv[b][u][0]);
                acc[m][u][3] += hi.y * (asv[b][m][1] * cfv[b][u][1]);
            }
    };

    // One 128-K block into accumulator buffer b; folds buffer 1-b (the
    // previous block) a quarter per k32 slice so its ALU work interleaves
    // with this block's MMAs.
    auto block = [&](auto bc, int kb) {
        constexpr int b = decltype(bc)::value;
        const int slot = kb % STAGES;
        cp_wait_<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kb + STAGES - 1;
            if (nk < K128) load_stage(nk % STAGES, nk);
            cp_commit_();
        }
        if (!active) return;
        const uint8_t* st = smem + slot * G::STAGE;
        const uint32_t sb = s_base + slot * G::STAGE;
        uint4 wv[NU];
        uint32_t P0[NU], P1[NU], N0[NU], N1[NU];
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            wv[u] = *reinterpret_cast<const uint4*>(st + G::W_OFF + (wn * NU + u) * 512 + lane * 16);
            const uint8_t* lt = st + G::LUT_OFF + (wn * NU + u) * 96;
            const uint2 p = *reinterpret_cast<const uint2*>(lt + g * 8);
            // fold factors of the output columns 2t, 2t+1 (rows 2t, 2t+1 of the tile)
            const float2 cf = *reinterpret_cast<const float2*>(lt + 64 + t * 8);
            P0[u] = p.x;
            P1[u] = p.y;
            N0[u] = p.x ^ 0x80808080u;
            N1[u] = p.y ^ 0x80808080u;
            cfv[b][u][0] = cf.x;
            cfv[b][u][1] = cf.y;
        }
        const float* scs = reinterpret_cast<const float*>(st + G::AS_OFF);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            asv[b][m][0] = scs[wm * MT * 16 + m * 16 + g];
            asv[b][m][1] = scs[wm * MT * 16 + m * 16 + g + 8];
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int u = 0; u < NU; ++u) c[b][m][u][0] = c[b][m][u][1] = 0u;
#pragma unroll
        for (int ka = 0; ka < 4; ++ka) {
            uint32_t b0[NU], b1[NU];
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                const uint32_t w = comp_(wv[u], ka);
                b0[u] = dec4_(w, P0[u], P1[u], N0[u], N1[u]);
                b1[u] = dec4_(w >> 16, P0[u], P1[u], N0[u], N1[u]);
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                uint32_t a0, a1, a2, a3;
                ldsm_x4_(a0, a1, a2, a3, sb + ld_off[ka] + m * 2048);
#pragma unroll
                for (int u = 0; u < NU; ++u) mma_f16_(c[b][m][u], a0, a1, a2, a3, b0[u], b1[u]);
            }
            if (kb > 0) {
                constexpr int MQ = (MT + 3) / 4;
                const int lo = ka * MQ, hi = (ka + 1) * MQ < MT ? (ka + 1) * MQ : MT;
                if (lo < MT) fold(std::integral_constant<int, 1 - b>{}, lo, hi);
            }
        }
    };

    // K128 is even (K % 256 == 0), so the walk ends on buffer 1
    for (int kb = 0; kb < K128; kb += 2) {
        block(std::integral_constant<int, 0>{}, kb);
        block(std::integral_constant<int, 1>{}, kb + 1);
    }
    cp_wait_<0>();
    if (!active) return;
    fold(std::integral_constant<int, 1>{}, 0, MT);
#pragma unroll
    for (int u = 0; u < NU; ++u) {
        const int col = (tile0 + u) * 8 + 2 * t;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r = m0 + wm * MT * 16 + m * 16 + g;
            *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)r * N + col]) =
                __floats2bfloat162_rn(acc[m][u][0], acc[m][u][1]);
            *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)(r + 8) * N + col]) =
                __floats2bfloat162_rn(acc[m][u][2], acc[m][u][3]);
        }
    }
}

template <int UNITS, int MSPLIT, int STAGES, int MINB = 1>
int launch_lut_(const void* A, const void* Wf, const void* Lut, const float* act_scale,
                void* D, int M, int N, int K, cudaStream_t s) {
    using G = LutGeo<UNITS, MSPLIT, STAGES>;
    auto kfn = fp4_stream_m128_lut_kernel<UNITS, MSPLIT, STAGES, MINB>;
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute((const void*)kfn,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 G::SMEM) != cudaSuccess)
            return -4;
        attr = true;
    }
    dim3 grid((N / 16 + UNITS - 1) / UNITS, M / 128);
    kfn<<<grid, G::WARPS * 32, G::SMEM, s>>>(
        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint4*>(Wf),
        reinterpret_cast<const uint8_t*>(Lut), act_scale,
        reinterpret_cast<__nv_bfloat16*>(D), N, K);
    return (int)cudaGetLastError();
}

}  // namespace

int nvfp4_m128_lut_build(const void* Bsf, const float* gscale, const float* w_scale,
                         void* lut, int N, int K, cudaStream_t s) {
    if ((N % 128) != 0 || (K % 128) != 0) return -1;
    const int tot = N * (K / 128);
    nvfp4_m128_lut_kernel<<<(tot + 255) / 256, 256, 0, s>>>(
        reinterpret_cast<const uint8_t*>(Bsf), gscale, w_scale,
        reinterpret_cast<uint8_t*>(lut), N, K / 128);
    return (int)cudaGetLastError();
}

int fp4_stream_m128_lut_sm89(const void* A, const void* Wf, const void* Lut,
                             const float* act_scale, void* D, int M, int N, int K,
                             int cfg, cudaStream_t s) {
    if ((M % 128) != 0 || (N % 16) != 0 || (K % 256) != 0) return -1;
    // cfg 0: 96-column CTA, 12 warps (6 units x 2 row halves), 4 stages --
    // 32/96/150 CTAs at N = 3072/9216/14336 on 34 SMs. cfg 1: same, 3 stages.
    // cfg 2: 128-column CTA, 16 warps, 3 stages. cfg 3: 96-column CTA, 2 stages.
    // cfg 4: 64-column CTA, 4 warps over all 128 rows (MT = 8), 2 stages, two
    // CTAs per SM -- the wide-N / short-K ffn up (N=14336, K=3072) at M=128:
    // 146.0 -> 133.7 us; slower at the other three video shapes.
    switch (cfg) {
        case 0: return launch_lut_<6, 2, 4>(A, Wf, Lut, act_scale, D, M, N, K, s);
        case 1: return launch_lut_<6, 2, 3>(A, Wf, Lut, act_scale, D, M, N, K, s);
        case 2: return launch_lut_<8, 2, 3>(A, Wf, Lut, act_scale, D, M, N, K, s);
        case 3: return launch_lut_<6, 2, 2>(A, Wf, Lut, act_scale, D, M, N, K, s);
        case 4: return launch_lut_<4, 1, 2, 2>(A, Wf, Lut, act_scale, D, M, N, K, s);
    }
    return -3;
}

int fp4_stream_m128_sm89(const void* A, const void* Wf, const void* Bsf,
                         const float* gscale, const float* act_scale,
                         const float* w_scale, void* D, int M, int N, int K,
                         int cfg, int f16acc, cudaStream_t s) {
    if ((N % 16) != 0 || (K % 256) != 0) return -1;
#define FP4M_CASE(ID, MT, NU, WP)                                                   \
    if (cfg == ID) {                                                                \
        if ((M % (16 * MT)) != 0) return -2;                                        \
        return f16acc ? launch_<MT, NU, WP, 2, 3, true>(A, Wf, Bsf, gscale,         \
                                                        act_scale, w_scale, D, M,   \
                                                        N, K, s)                    \
                      : launch_<MT, NU, WP, 2, 3, false>(A, Wf, Bsf, gscale,        \
                                                         act_scale, w_scale, D, M,  \
                                                         N, K, s);                  \
    }
    FP4M_CASE(0, 8, 2, 8)
    FP4M_CASE(1, 4, 2, 8)
    FP4M_CASE(2, 8, 1, 8)
    FP4M_CASE(3, 8, 2, 4)
    // 96-column CTA. 34 SMs: N=3072 -> 32 CTAs (94% wave fill, one wave),
    // N=9216 -> 96 CTAs (94%), N=14336 -> 150 CTAs (88%); the 8-warp CTA
    // fills 24/34, 72/102 and 112/136 of the same waves. Numerics are
    // bit-identical: the split changes only which CTA computes which
    // 16-column unit, and every unit keeps its own K-order.
    FP4M_CASE(4, 8, 2, 6)
#undef FP4M_CASE
    return -3;
}

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
