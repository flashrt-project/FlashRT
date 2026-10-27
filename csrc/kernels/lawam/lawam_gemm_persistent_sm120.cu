// SPDX-License-Identifier: Apache-2.0
// Persistent FP16-accumulate GEMM for the LaWAM towers on RTX SM120.
//
//   D[M,N] = act(A[M,K] @ B[K,N] + bias)     A, B, D row-major FP16
//
// One CTA per SM walks a static, host-built list of (tile, k range) segments
// with one continuous cp.async pipeline (the next segment's stages are in
// flight while a tile's epilogue runs). The schedule runs full waves of output
// tiles data-parallel and spreads the k-iterations of the remaining tiles
// evenly over the CTAs (stream-K), so the 36 SMs of an RTX 5060 Ti stay busy
// on shapes whose tile count is not a multiple of the SM count.
//
// Split tiles meet in a workspace: every segment stores its FP16 partial sum,
// the last one to arrive (atomic counter per tile) adds the partials in k
// order (FP16, the mma's accumulator format) and runs the epilogue. Nothing
// spins on another CTA, so concurrent graph branches cannot deadlock, and the
// summation order is fixed by the schedule (reproducible results).
//
// The epilogue stages each warp's rows through the shared-memory stage the
// last k-tile consumed and writes 16-byte rows: bias, tanh / erf GELU in FP32
// (as lawam::bias_gelu on the stored GEMM output), or SwiGLU over a gate | up
// weight interleaved in blocks of 8 columns (out = silu(gate) * up).
//
// The weight (B) stages of the prologue are issued before the programmatic
// dependency wait: under PDL the kernel starts on the SMs the previous kernel
// frees and streams its first weight tiles while that kernel drains.
#include "lawam_kernels.cuh"

#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace flash_rt {
namespace lawam {
namespace {

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// 16-byte chunk swizzle of a shared-memory row of RC chunks (ldmatrix rows conflict-free)
template <int RC>
__device__ __forceinline__ int swz(int r, int c) {
    if constexpr (RC >= 8) return c ^ (r & 7);
    else if constexpr (RC == 4) return c ^ ((r >> 1) & 3);
    else return c ^ ((r >> 2) & 1);
}
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

enum Epi { E_NONE = 0, E_BIAS = 1, E_GELU_TANH = 2, E_GELU_ERF = 3, E_SWIGLU = 4 };

// segment: x tile, y k0 | k1 << 16 (k-tiles), z split index | count << 8 (count 0: whole tile),
// w tile counter | first workspace slot << 12
struct Params {
    const __half* A;
    const __half* B;
    __half* D;
    const __half* bias;
    int M, N, K, ldd;
    const int4* segs;
    const int* cta_seg;
    uint2* ws;
    int* counters;
};

template <int BM_, int BN_, int BK_, int WM_, int WN_, int S_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_, S = S_;
    static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, NT = WARPS_M * WARPS_N * 32;
    static constexpr int MT = WM / 16, NT8 = WN / 8;
    static constexpr int RCA = BK / 8, RCB = BN / 8;
    static constexpr int A_STAGE = BM * BK, B_STAGE = BK * BN;   // halfs
    static constexpr int SMEM = S * (A_STAGE + B_STAGE) * 2;
    static constexpr int E2 = MT * NT8;                          // uint2 partial words per thread
    static_assert(NT8 % 2 == 0 && WN % 16 == 0, "warp tile");
};

template <class C, int EPI>
__global__ void __launch_bounds__(C::NT, 1) gemm_kernel(Params p) {
    constexpr int BM = C::BM, BN = C::BN, BK = C::BK, WM = C::WM, WN = C::WN, S = C::S;
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ int s_last;
    __half* sA = reinterpret_cast<__half*>(smem);
    __half* sB = sA + S * C::A_STAGE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / C::WARPS_N, wn = warp % C::WARPS_N;
    const int M = p.M, N = p.N, K = p.K;
    const int mtiles = (M + BM - 1) / BM;
    const int s_beg = p.cta_seg[blockIdx.x], s_end = p.cta_seg[blockIdx.x + 1];
    if (s_beg == s_end) return;
    const unsigned sA0 = smem_u32(sA), sB0 = smem_u32(sB);
    int n_it = 0;
    for (int s = s_beg; s < s_end; ++s) {
        const int y = __ldg(&p.segs[s].y);
        n_it += (y >> 16) - (y & 0xffff);
    }

    // ── loader: the next (segment, k-tile) to load, advanced incrementally ──
    constexpr int CA = BM * BK / 8, CB = BK * BN / 8;
    constexpr int QA = (CA + C::NT - 1) / C::NT, QB = (CB + C::NT - 1) / C::NT;
    int l_seg = s_beg;
    int4 lg = __ldg(p.segs + l_seg);
    int l_k = lg.y & 0xffff;
    const __half* a_src[QA];
    bool a_ok[QA];
    const __half* b_src = nullptr;
    auto l_retile = [&]() {
        const int m0 = (lg.x % mtiles) * BM, n0 = (lg.x / mtiles) * BN;
#pragma unroll
        for (int q = 0; q < QA; ++q) {
            const int i = tid + q * C::NT, r = i / C::RCA, cc = i % C::RCA;
            const int gr = m0 + r;
            a_ok[q] = (i < CA) && gr < M;
            a_src[q] = p.A + static_cast<size_t>(gr < M ? gr : 0) * K + cc * 8;
        }
        b_src = p.B + n0;
    };
    auto load_a = [&](int slot) {
        const int k0 = l_k * BK;
#pragma unroll
        for (int q = 0; q < QA; ++q) {
            const int i = tid + q * C::NT, r = i / C::RCA, cc = i % C::RCA;
            if (CA % C::NT == 0 || i < CA)
                cp16(sA0 + (slot * C::A_STAGE + r * BK + swz<C::RCA>(r, cc) * 8) * 2, a_src[q] + k0, a_ok[q]);
        }
    };
    auto load_b = [&](int slot) {
        const int k0 = l_k * BK;
#pragma unroll
        for (int q = 0; q < QB; ++q) {
            const int i = tid + q * C::NT, r = i / C::RCB, cc = i % C::RCB;
            if (CB % C::NT == 0 || i < CB)
                cp16(sB0 + (slot * C::B_STAGE + r * BN + swz<C::RCB>(r, cc) * 8) * 2,
                     b_src + static_cast<size_t>(k0 + r) * N + cc * 8, true);
        }
    };
    auto advance = [&]() {
        if (++l_k == (lg.y >> 16) && l_seg + 1 < s_end) {
            lg = __ldg(p.segs + (++l_seg));
            l_k = lg.y & 0xffff;
            l_retile();
        }
    };
    l_retile();

    // ── prologue: weight stages first (independent of the previous kernel), then the activations ──
    {
        const int sv_seg = l_seg, sv_k = l_k;
        const int4 sv_lg = lg;
#pragma unroll
        for (int s = 0; s < S - 1; ++s)
            if (s < n_it) {
                load_b(s);
                advance();
            }
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
        l_seg = sv_seg;
        l_k = sv_k;
        lg = sv_lg;
        l_retile();
#pragma unroll
        for (int s = 0; s < S - 1; ++s) {
            if (s < n_it) {
                load_a(s);
                advance();
            }
            cp_commit();
        }
    }

    // ── fragments ──
    constexpr int nkk = BK / 16;
    unsigned a[2][C::MT][4], b[2][C::NT8 / 2][4];
    auto ldfrag = [&](int buf, int slot, int kk) {
        const unsigned aBase = sA0 + slot * C::A_STAGE * 2, bBase = sB0 + slot * C::B_STAGE * 2;
#pragma unroll
        for (int i = 0; i < C::MT; ++i) {
            const int r = wm * WM + i * 16 + (lane & 15);
            const int cc = kk * 2 + (lane >> 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a[buf][i][0]), "=r"(a[buf][i][1]), "=r"(a[buf][i][2]), "=r"(a[buf][i][3])
                         : "r"(aBase + (r * BK + swz<C::RCA>(r, cc) * 8) * 2));
        }
#pragma unroll
        for (int j = 0; j < C::NT8 / 2; ++j) {
            const int r = kk * 16 + (lane & 15);
            const int cc = (wn * WN) / 8 + j * 2 + (lane >> 4);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b[buf][j][0]), "=r"(b[buf][j][1]), "=r"(b[buf][j][2]), "=r"(b[buf][j][3])
                         : "r"(bBase + (r * BN + swz<C::RCB>(r, cc) * 8) * 2));
        }
    };
    unsigned acc[C::MT][C::NT8][2];

    cp_wait<S - 2>();
    __syncthreads();
    ldfrag(0, 0, 0);
    int slot = 0;
    int c_seg = s_beg;
    int4 cg = __ldg(p.segs + c_seg);
    int ck = cg.y & 0xffff;
    for (int it = 0; it < n_it; ++it) {
        if (ck == (cg.y & 0xffff)) {
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int j = 0; j < C::NT8; ++j) acc[i][j][0] = acc[i][j][1] = 0u;
        }
#pragma unroll
        for (int kk = 0; kk < nkk; ++kk) {
            if (kk + 1 < nkk) ldfrag((kk + 1) & 1, slot, kk + 1);
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int j = 0; j < C::NT8; ++j)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, "
                        "{%0,%1};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1])
                        : "r"(a[kk & 1][i][0]), "r"(a[kk & 1][i][1]), "r"(a[kk & 1][i][2]), "r"(a[kk & 1][i][3]),
                          "r"(b[kk & 1][j / 2][(j & 1) * 2]), "r"(b[kk & 1][j / 2][(j & 1) * 2 + 1]));
            if (kk == 0) {
                // refill the stage the previous k-tile consumed
                if (it + S - 1 < n_it) {
                    const int ls = slot == 0 ? S - 1 : slot - 1;
                    load_a(ls);
                    load_b(ls);
                    advance();
                }
                cp_commit();
            }
        }
        // the next k-tile's stage must be resident before its first fragments
        cp_wait<S - 2>();
        __syncthreads();
        slot = slot + 1 == S ? 0 : slot + 1;
        ldfrag(0, slot, 0);
        if (++ck != (cg.y >> 16)) continue;

        // ── segment end ──
        const int tile = cg.x;
        const int nsplit = (cg.z >> 8) & 255;
        if (nsplit) {
            const int q = cg.z & 255, cnt = cg.w & 4095, base = cg.w >> 12;
            uint2* w = p.ws + static_cast<size_t>(base + q) * C::E2 * C::NT + tid;
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int j = 0; j < C::NT8; ++j)
                    __stcg(w + (i * C::NT8 + j) * C::NT, make_uint2(acc[i][j][0], acc[i][j][1]));
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                const int last = atomicAdd(p.counters + cnt, 1) == nsplit - 1;
                if (last) atomicExch(p.counters + cnt, 0);
                s_last = last;
            }
            __syncthreads();
            if (!s_last) {
                if (c_seg + 1 < s_end) {
                    cg = __ldg(p.segs + (++c_seg));
                    ck = cg.y & 0xffff;
                }
                continue;
            }
            __threadfence();
            // the partials in k order, FP16 sums
#pragma unroll 1
            for (int s = 0; s < nsplit; ++s) {
                const uint2* r = p.ws + static_cast<size_t>(base + s) * C::E2 * C::NT + tid;
#pragma unroll
                for (int i = 0; i < C::MT; ++i)
#pragma unroll
                    for (int j = 0; j < C::NT8; ++j) {
                        const uint2 v = __ldcg(r + (i * C::NT8 + j) * C::NT);
                        if (s == 0) {
                            acc[i][j][0] = v.x;
                            acc[i][j][1] = v.y;
                        } else {
                            __half2 x0 = __hadd2(*reinterpret_cast<__half2*>(&acc[i][j][0]),
                                                 *reinterpret_cast<const __half2*>(&v.x));
                            __half2 x1 = __hadd2(*reinterpret_cast<__half2*>(&acc[i][j][1]),
                                                 *reinterpret_cast<const __half2*>(&v.y));
                            acc[i][j][0] = *reinterpret_cast<unsigned*>(&x0);
                            acc[i][j][1] = *reinterpret_cast<unsigned*>(&x1);
                        }
                    }
            }
        }
        {
            // epilogue staged through the stage the last k-tile consumed (free until the next refill)
            constexpr int RS = WN * 2 + 16;   // padded row stride (bytes)
            constexpr int REG = 8 * RS * C::WARPS_M * C::WARPS_N;
            static_assert(REG <= C::A_STAGE * 2 || REG <= C::B_STAGE * 2, "epilogue staging");
            const int fslot = slot == 0 ? S - 1 : slot - 1;
            unsigned char* ebase = REG <= C::A_STAGE * 2
                                       ? smem + static_cast<size_t>(fslot) * C::A_STAGE * 2
                                       : reinterpret_cast<unsigned char*>(sB) + static_cast<size_t>(fslot) * C::B_STAGE * 2;
            unsigned char* wr = ebase + warp * 8 * RS;
            const int m0 = (tile % mtiles) * BM, n0 = (tile / mtiles) * BN;
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int j = 0; j < C::NT8; ++j)
                        *reinterpret_cast<unsigned*>(wr + (lane >> 2) * RS + (j * 8 + (lane & 3) * 2) * 2) =
                            acc[i][j][h];
                    __syncwarp();
                    const int rbase = m0 + wm * WM + i * 16 + 8 * h;
                    if constexpr (EPI == E_SWIGLU) {
                        constexpr int CPR = WN / 16;   // output 16-byte chunks per row
#pragma unroll 1
                        for (int ch = lane; ch < 8 * CPR; ch += 32) {
                            const int r = ch / CPR, cc = ch % CPR;
                            if (rbase + r >= M) continue;
                            const uint4 g = *reinterpret_cast<const uint4*>(wr + r * RS + cc * 32);
                            const uint4 u = *reinterpret_cast<const uint4*>(wr + r * RS + cc * 32 + 16);
                            const __half2* gh = reinterpret_cast<const __half2*>(&g);
                            const __half2* uh = reinterpret_cast<const __half2*>(&u);
                            uint4 o;
                            __half2* oh = reinterpret_cast<__half2*>(&o);
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const float2 gf = __half22float2(gh[e]), uf = __half22float2(uh[e]);
                                oh[e] = __floats2half2_rn(silu(gf.x) * uf.x, silu(gf.y) * uf.y);
                            }
                            *reinterpret_cast<uint4*>(p.D + static_cast<size_t>(rbase + r) * p.ldd + (n0 + wn * WN) / 2 +
                                                      cc * 8) = o;
                        }
                    } else {
                        constexpr int CPR = WN / 8;
#pragma unroll 1
                        for (int ch = lane; ch < 8 * CPR; ch += 32) {
                            const int r = ch / CPR, cc = ch % CPR;
                            if (rbase + r >= M) continue;
                            uint4 o = *reinterpret_cast<const uint4*>(wr + r * RS + cc * 16);
                            const int col = n0 + wn * WN + cc * 8;
                            if constexpr (EPI != E_NONE) {
                                const uint4 bv = __ldg(reinterpret_cast<const uint4*>(p.bias + col));
                                __half2* oh = reinterpret_cast<__half2*>(&o);
                                const __half2* bh = reinterpret_cast<const __half2*>(&bv);
#pragma unroll
                                for (int e = 0; e < 4; ++e) {
                                    float2 f = __half22float2(oh[e]);
                                    const float2 bb = __half22float2(bh[e]);
                                    f.x += bb.x;
                                    f.y += bb.y;
                                    if constexpr (EPI == E_GELU_TANH) {
                                        f.x = gelu_tanh(f.x);
                                        f.y = gelu_tanh(f.y);
                                    }
                                    if constexpr (EPI == E_GELU_ERF) {
                                        f.x = gelu_erf(f.x);
                                        f.y = gelu_erf(f.y);
                                    }
                                    oh[e] = __float22half2_rn(f);
                                }
                            }
                            *reinterpret_cast<uint4*>(p.D + static_cast<size_t>(rbase + r) * p.ldd + col) = o;
                        }
                    }
                    __syncwarp();
                }
            __syncthreads();   // the staging stage is refilled by the next k-tile
        }
        if (c_seg + 1 < s_end) {
            cg = __ldg(p.segs + (++c_seg));
            ck = cg.y & 0xffff;
        }
    }
    cp_wait<0>();
}

// ── host side ──

// Tile configurations (CTA M x N x K, warp tile, stages).
using Cfg0 = Cfg<224, 128, 32, 112, 32, 4>;   // M <= 224 in one row of tiles (LLM, Se ~ 219)
using Cfg1 = Cfg<96, 128, 32, 48, 32, 4>;     // M = 261 in 3 rows (DINOv3)
using Cfg2 = Cfg<128, 256, 32, 64, 64, 4>;
using Cfg3 = Cfg<64, 256, 32, 32, 64, 4>;
using Cfg4 = Cfg<128, 128, 64, 64, 32, 3>;
using Cfg5 = Cfg<256, 128, 32, 64, 64, 4>;
constexpr int kNumCfgs = 6;

template <class C>
void cfg_dims(int& bm, int& bn, int& bk) {
    bm = C::BM;
    bn = C::BN;
    bk = C::BK;
}
bool dims_of(int cfg, int& bm, int& bn, int& bk, int& e2nt) {
    switch (cfg) {
        case 0: cfg_dims<Cfg0>(bm, bn, bk); e2nt = Cfg0::E2 * Cfg0::NT; return true;
        case 1: cfg_dims<Cfg1>(bm, bn, bk); e2nt = Cfg1::E2 * Cfg1::NT; return true;
        case 2: cfg_dims<Cfg2>(bm, bn, bk); e2nt = Cfg2::E2 * Cfg2::NT; return true;
        case 3: cfg_dims<Cfg3>(bm, bn, bk); e2nt = Cfg3::E2 * Cfg3::NT; return true;
        case 4: cfg_dims<Cfg4>(bm, bn, bk); e2nt = Cfg4::E2 * Cfg4::NT; return true;
        case 5: cfg_dims<Cfg5>(bm, bn, bk); e2nt = Cfg5::E2 * Cfg5::NT; return true;
        default: return false;
    }
}

// Static schedule: data-parallel waves for all but the last partial wave and
// one full wave (sched 0), whose k-iterations spread evenly over the CTAs;
// sched 1 all data-parallel, sched 2 all stream-K.
struct Schedule {
    std::vector<int4> segs;
    std::vector<int> cta;
    int slots = 0, counters = 0;
};
Schedule make_schedule(int tiles, int KT, int G, int sched) {
    int dp_tiles = 0;
    if (sched == 1 || tiles % G == 0) dp_tiles = tiles;
    else if (sched == 0) dp_tiles = std::max(0, (tiles / G - 1) * G);
    const int sk_tiles = tiles - dp_tiles;
    const long long I = static_cast<long long>(sk_tiles) * KT;
    // stream-K segments of every CTA, and the split count / first slot of every split tile
    std::vector<std::vector<std::array<int, 3>>> sk(G);   // tile, k0, k1
    std::vector<int> nsplit(sk_tiles, 0);
    for (int c = 0; c < G && sk_tiles; ++c) {
        long long b = I * c / G;
        const long long e = I * (c + 1) / G;
        while (b < e) {
            const int t = static_cast<int>(b / KT), k0 = static_cast<int>(b % KT);
            const int k1 = static_cast<int>(std::min<long long>(KT, k0 + (e - b)));
            sk[c].push_back({t, k0, k1});
            ++nsplit[t];
            b += k1 - k0;
        }
    }
    Schedule s;
    std::vector<int> base(sk_tiles, 0), counter(sk_tiles, 0), seen(sk_tiles, 0);
    for (int t = 0; t < sk_tiles; ++t)
        if (nsplit[t] > 1) {
            base[t] = s.slots;
            s.slots += nsplit[t];
            counter[t] = s.counters++;
        }
    auto seg_of = [&](int t, int k0, int k1) {
        int4 g;
        g.x = dp_tiles + t;
        g.y = k0 | (k1 << 16);
        if (nsplit[t] > 1) {
            g.z = seen[t]++ | (nsplit[t] << 8);
            g.w = counter[t] | (base[t] << 12);
        } else {
            g.z = 0;
            g.w = 0;
        }
        return g;
    };
    // k-order split index: segments of a tile are met in CTA order = k order
    std::vector<std::vector<int4>> per(G);
    for (int c = 0; c < G; ++c)
        for (auto& g : sk[c]) per[c].push_back(seg_of(g[0], g[1], g[2]));
    s.cta.push_back(0);
    for (int c = 0; c < G; ++c) {
        // split segments that end a tile's k range late come last; data-parallel tiles in between
        std::vector<int4> list;
        for (auto& g : per[c])
            if ((g.y & 0xffff) != 0) list.push_back(g);
        for (int t = c; t < dp_tiles; t += G) list.push_back(make_int4(t, KT << 16, 0, 0));
        for (auto& g : per[c])
            if ((g.y & 0xffff) == 0) list.push_back(g);
        s.segs.insert(s.segs.end(), list.begin(), list.end());
        s.cta.push_back(static_cast<int>(s.segs.size()));
    }
    return s;
}

struct DevPlan {
    int4* segs = nullptr;
    int* cta = nullptr;
    int grid = 0;
    int slots = 0, counters = 0;
};
struct StreamWs {
    uint2* ws = nullptr;
    size_t ws_words = 0;
    int* counters = nullptr;
};
constexpr int kMaxCounters = 256;
constexpr size_t kMaxPartialWords = 8192;   // E2 * NT of the largest configuration

struct Ctx {
    std::mutex mu;
    std::map<std::tuple<int, int, int, int, int>, DevPlan> plans;   // M, N, K, cfg, sched
    std::map<cudaStream_t, StreamWs> ws;
    int num_sms = 0;
};
Ctx& ctx() {
    static Ctx c;
    return c;
}

template <class C, int EPI>
int launch_cfg(const Params& p, int grid, cudaStream_t stream) {
    auto f = gemm_kernel<C, EPI>;
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess)
            return static_cast<int>(cudaErrorInvalidValue);
        attr = true;
    }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(grid);
    lc.blockDim = dim3(C::NT);
    lc.dynamicSmemBytes = C::SMEM;
    lc.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
    return static_cast<int>(cudaLaunchKernelEx(&lc, f, p));
}

template <int EPI>
int launch_epi(int cfg, const Params& p, int grid, cudaStream_t stream) {
    switch (cfg) {
        case 0: return launch_cfg<Cfg0, EPI>(p, grid, stream);
        case 1: return launch_cfg<Cfg1, EPI>(p, grid, stream);
        case 2: return launch_cfg<Cfg2, EPI>(p, grid, stream);
        case 3: return launch_cfg<Cfg3, EPI>(p, grid, stream);
        case 4: return launch_cfg<Cfg4, EPI>(p, grid, stream);
        default: return launch_cfg<Cfg5, EPI>(p, grid, stream);
    }
}

}  // namespace

int gemm_f16acc_persistent(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K,
                           int ldd, int act, int cfg, int sched, cudaStream_t stream) {
    int bm, bn, bk, e2nt;
    if (!dims_of(cfg, bm, bn, bk, e2nt) || sched < 0 || sched > 2 || act < 0 || act > 3 || M < 1 || N % bn != 0 ||
        K % bk != 0 || K / bk > 0xffff || (act > 0 && act < 3 && bias == nullptr))
        return static_cast<int>(cudaErrorInvalidValue);
    const int out_cols = act == 3 ? N / 2 : N;
    if (ldd < out_cols || ldd % 8 != 0) return static_cast<int>(cudaErrorInvalidValue);
    Ctx& c = ctx();
    DevPlan plan;
    StreamWs sw;
    {
        std::lock_guard<std::mutex> lock(c.mu);
        if (c.num_sms == 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&c.num_sms, cudaDevAttrMultiProcessorCount, dev);
        }
        const auto key = std::make_tuple(M, N, K, cfg, sched);
        auto it = c.plans.find(key);
        auto ws_it = c.ws.find(stream);
        if (it == c.plans.end() || ws_it == c.ws.end()) {
            // first use of a shape / stream: allocate (must not run inside a graph capture)
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap);
            if (cap != cudaStreamCaptureStatusNone) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
        }
        if (it == c.plans.end()) {
            const int tiles = ((M + bm - 1) / bm) * (N / bn);
            const int G = sched == 1 ? std::min(c.num_sms, tiles) : c.num_sms;
            Schedule s = make_schedule(tiles, K / bk, G, sched);
            if (s.counters > kMaxCounters) return static_cast<int>(cudaErrorInvalidValue);
            DevPlan d;
            d.grid = G;
            d.slots = s.slots;
            d.counters = s.counters;
            if (cudaMalloc(&d.segs, s.segs.size() * sizeof(int4)) != cudaSuccess ||
                cudaMalloc(&d.cta, s.cta.size() * sizeof(int)) != cudaSuccess)
                return static_cast<int>(cudaErrorMemoryAllocation);
            cudaMemcpy(d.segs, s.segs.data(), s.segs.size() * sizeof(int4), cudaMemcpyHostToDevice);
            cudaMemcpy(d.cta, s.cta.data(), s.cta.size() * sizeof(int), cudaMemcpyHostToDevice);
            it = c.plans.emplace(key, d).first;
        }
        plan = it->second;
        StreamWs& w = c.ws[stream];
        if (w.ws == nullptr) {
            // one fixed workspace per stream (captured graphs keep its address): at most two split
            // segments per CTA, a partial of the largest tile each
            w.ws_words = static_cast<size_t>(2 * c.num_sms) * kMaxPartialWords;
            if (cudaMalloc(&w.ws, w.ws_words * sizeof(uint2)) != cudaSuccess ||
                cudaMalloc(&w.counters, kMaxCounters * sizeof(int)) != cudaSuccess ||
                cudaMemset(w.counters, 0, kMaxCounters * sizeof(int)) != cudaSuccess)
                return static_cast<int>(cudaErrorMemoryAllocation);
        }
        if (static_cast<size_t>(plan.slots) * e2nt > w.ws_words) return static_cast<int>(cudaErrorInvalidValue);
        sw = w;
    }
    Params p{A, B, D, bias, M, N, K, ldd, plan.segs, plan.cta, sw.ws, sw.counters};
    switch (act) {
        case 0: return bias ? launch_epi<E_BIAS>(cfg, p, plan.grid, stream) : launch_epi<E_NONE>(cfg, p, plan.grid, stream);
        case 1: return launch_epi<E_GELU_TANH>(cfg, p, plan.grid, stream);
        case 2: return launch_epi<E_GELU_ERF>(cfg, p, plan.grid, stream);
        default: return launch_epi<E_SWIGLU>(cfg, p, plan.grid, stream);
    }
}

}  // namespace lawam
}  // namespace flash_rt
