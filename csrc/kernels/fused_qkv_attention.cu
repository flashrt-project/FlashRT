// SPDX-License-Identifier: Apache-2.0
// Fused-QKV non-causal prefill attention. See the header.
//
// One CTA per (64-row query tile, head, batch): four warps own 16 query rows
// each; the CTA stages 64-key chunks of K (row-major, head_dim padded to a
// multiple of 16 with zeros) and V (transposed) in shared memory, applying
// bias and RoPE on the way in, and every warp runs an online softmax over the
// chunks on mma.sync m16n8k16.
#include "fused_qkv_attention.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace flash_rt {
namespace prefill_attention {
namespace {

constexpr int kChunk = 64;   // keys per shared-memory chunk

template <typename T>
struct Op;

template <>
struct Op<__half> {
    static __device__ __forceinline__ float2 f2(uint32_t u) {
        return __half22float2(*reinterpret_cast<const __half2*>(&u));
    }
    static __device__ __forceinline__ uint32_t pack(float a, float b) {
        const __half2 v = __floats2half2_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
    static __device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

template <>
struct Op<__nv_bfloat16> {
    static __device__ __forceinline__ float2 f2(uint32_t u) {
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
    }
    static __device__ __forceinline__ uint32_t pack(float a, float b) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
    static __device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

template <typename T>
__device__ __forceinline__ uint32_t ld2(const T* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

// VW consecutive elements starting at c of the first half of a head row and
// their rotate-half partners (c + hd/2): raw loads, issued before use.
template <typename T, int VW>
struct Raw {
    using V = typename std::conditional<VW == 8, uint4, uint2>::type;
    V lo, hi, c, s;
};

template <typename T, int VW>
__device__ __forceinline__ void unpack(const typename Raw<T, VW>::V& r, float* f) {
    const uint32_t* u = reinterpret_cast<const uint32_t*>(&r);
#pragma unroll
    for (int j = 0; j < VW / 2; ++j) {
        const float2 x = Op<T>::f2(u[j]);
        f[2 * j] = x.x;
        f[2 * j + 1] = x.y;
    }
}

template <typename T, int VW>
__device__ __forceinline__ typename Raw<T, VW>::V pack_vec(const float* f) {
    typename Raw<T, VW>::V r;
    uint32_t* u = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
    for (int j = 0; j < VW / 2; ++j) u[j] = Op<T>::pack(f[2 * j], f[2 * j + 1]);
    return r;
}

// Load the raw unit (row, c): x[c..c+VW), x[c+H2..), cos / sin[c..) (rope).
template <typename T, int HD, int VW>
__device__ __forceinline__ void load_unit(Raw<T, VW>& r, const T* x, const T* cos_row, const T* sin_row, int c,
                                          bool valid) {
    using V = typename Raw<T, VW>::V;
    constexpr int H2 = HD / 2;
    if (!valid) {
        r.lo = r.hi = r.c = r.s = V{};
        return;
    }
    r.lo = *reinterpret_cast<const V*>(x + c);
    r.hi = *reinterpret_cast<const V*>(x + c + H2);
    if (cos_row != nullptr) {
        r.c = *reinterpret_cast<const V*>(cos_row + c);
        r.s = *reinterpret_cast<const V*>(sin_row + c);
    }
}

// Finish a unit: + bias, RoPE (when rope), into lo[VW] / hi[VW] floats.
template <typename T, int HD, int VW>
__device__ __forceinline__ void finish_unit(const Raw<T, VW>& r, const T* bias, int c, bool rope, bool valid,
                                            float* lo, float* hi) {
    using V = typename Raw<T, VW>::V;
    constexpr int H2 = HD / 2;
    unpack<T, VW>(r.lo, lo);
    unpack<T, VW>(r.hi, hi);
    if (!valid) return;
    if (bias != nullptr) {
        float bl[VW], bh[VW];
        unpack<T, VW>(*reinterpret_cast<const V*>(bias + c), bl);
        unpack<T, VW>(*reinterpret_cast<const V*>(bias + c + H2), bh);
#pragma unroll
        for (int j = 0; j < VW; ++j) {
            lo[j] += bl[j];
            hi[j] += bh[j];
        }
    }
    if (rope) {
        float cs[VW], sn[VW];
        unpack<T, VW>(r.c, cs);
        unpack<T, VW>(r.s, sn);
#pragma unroll
        for (int j = 0; j < VW; ++j) {
            const float a = lo[j], b = hi[j];
            lo[j] = a * cs[j] - b * sn[j];
            hi[j] = b * cs[j] + a * sn[j];
        }
    }
}

template <typename T, int HD, int NW>
__global__ void __launch_bounds__(NW * 32)
fused_qkv_attention_kernel(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, int ld,
                           const T* __restrict__ q_bias, const T* __restrict__ k_bias,
                           const T* __restrict__ v_bias, const T* __restrict__ cos, const T* __restrict__ sin,
                           T* __restrict__ out, int out_ld, int S, float scale_log2) {
    using O = Op<T>;
    constexpr int HDP = (HD + 15) / 16 * 16;
    constexpr int KS = HDP / 16;   // k-steps of Q K^T
    constexpr int NT = HD / 8;     // n-tiles of P V
    constexpr int H2 = HD / 2;
    constexpr int VW = H2 % 8 == 0 ? 8 : 4;   // elements per vector load
    constexpr int UK = H2 / VW;               // units per row
    constexpr int UNITS = kChunk * UK;        // units per 64-row chunk
    constexpr int kTileQ = NW * 16;           // query rows per CTA
    constexpr int UQ = kTileQ * UK;           // units of the Q tile
    constexpr int UPT = (UNITS + NW * 32 - 1) / (NW * 32);
    constexpr int UPTQ = (UQ + NW * 32 - 1) / (NW * 32);
    constexpr int KST = HDP + 8;
    using V = typename Raw<T, VW>::V;
    constexpr int KROWS = kChunk > kTileQ ? kChunk : kTileQ;
    __shared__ __align__(16) T ks[KROWS][KST];    // K chunk (also the Q tile before the loop)
    __shared__ __align__(16) T vt[HD][kChunk + 8];

    const int h = blockIdx.y, b = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const size_t base = static_cast<size_t>(b) * S;
    const int col0 = h * HD;
    const bool rope = cos != nullptr;
    const T* qbh = q_bias ? q_bias + col0 : nullptr;
    const T* kbh = k_bias ? k_bias + col0 : nullptr;
    const T* vbh = v_bias ? v_bias + col0 : nullptr;

    if constexpr (HDP > HD) {
        for (int i = threadIdx.x; i < KROWS * (HDP - HD); i += blockDim.x)
            ks[i / (HDP - HD)][HD + i % (HDP - HD)] = T(0.f);
    }

    // ── Q tile through shared memory: bias, RoPE, softmax scale * log2 e ──
    const int q0 = blockIdx.x * kTileQ;
    {
        Raw<T, VW> rq[UPTQ];
#pragma unroll
        for (int u = 0; u < UPTQ; ++u) {
            const int i = u * NW * 32 + threadIdx.x;
            const int r = i / UK, c = (i % UK) * VW;
            const bool valid = i < UQ && q0 + r < S;
            const size_t row = base + q0 + r;
            load_unit<T, HD, VW>(rq[u], q + row * ld + col0, rope ? cos + row * HD : nullptr,
                                 rope ? sin + row * HD : nullptr, c, valid);
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < UPTQ; ++u) {
            const int i = u * NW * 32 + threadIdx.x;
            if (i >= UQ) continue;
            const int r = i / UK, c = (i % UK) * VW;
            float lo[VW], hi[VW];
            finish_unit<T, HD, VW>(rq[u], qbh, c, rope, q0 + r < S, lo, hi);
#pragma unroll
            for (int j = 0; j < VW; ++j) {
                lo[j] *= scale_log2;
                hi[j] *= scale_log2;
            }
            *reinterpret_cast<V*>(&ks[r][c]) = pack_vec<T, VW>(lo);
            *reinterpret_cast<V*>(&ks[r][c + H2]) = pack_vec<T, VW>(hi);
        }
        __syncthreads();
    }
    uint32_t qa[KS][4];
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
        qa[kk][0] = ld2(&ks[w * 16 + g][kk * 16 + t * 2]);
        qa[kk][1] = ld2(&ks[w * 16 + g + 8][kk * 16 + t * 2]);
        qa[kk][2] = ld2(&ks[w * 16 + g][kk * 16 + 8 + t * 2]);
        qa[kk][3] = ld2(&ks[w * 16 + g + 8][kk * 16 + 8 + t * 2]);
    }

    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;

    // ── K / V chunks, the next chunk's loads in flight during the current one ──
    Raw<T, VW> rk[UPT], rv[UPT];
    auto load_chunk = [&](int c0) {
#pragma unroll
        for (int u = 0; u < UPT; ++u) {
            const int i = u * NW * 32 + threadIdx.x;
            const int r = i / UK, c = (i % UK) * VW;
            const bool valid = i < UNITS && c0 + r < S;
            const size_t row = base + c0 + r;
            load_unit<T, HD, VW>(rk[u], k + row * ld + col0, rope ? cos + row * HD : nullptr,
                                 rope ? sin + row * HD : nullptr, c, valid);
            load_unit<T, HD, VW>(rv[u], v + row * ld + col0, nullptr, nullptr, c, valid);
        }
    };
    load_chunk(0);
    for (int c0 = 0; c0 < S; c0 += kChunk) {
        __syncthreads();   // previous chunk (or the Q tile) fully consumed
#pragma unroll
        for (int u = 0; u < UPT; ++u) {
            const int i = u * NW * 32 + threadIdx.x;
            if (i >= UNITS) continue;
            const int r = i / UK, c = (i % UK) * VW;
            const bool valid = c0 + r < S;
            float lo[VW], hi[VW];
            finish_unit<T, HD, VW>(rk[u], kbh, c, rope, valid, lo, hi);
            *reinterpret_cast<V*>(&ks[r][c]) = pack_vec<T, VW>(lo);
            *reinterpret_cast<V*>(&ks[r][c + H2]) = pack_vec<T, VW>(hi);
            finish_unit<T, HD, VW>(rv[u], vbh, c, false, valid, lo, hi);
#pragma unroll
            for (int j = 0; j < VW; ++j) {
                vt[c + j][r] = T(lo[j]);
                vt[c + H2 + j][r] = T(hi[j]);
            }
        }
        __syncthreads();
        if (c0 + kChunk < S) load_chunk(c0 + kChunk);

        float s[kChunk / 8][4];
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < KS; ++kk)
                O::mma(s[n], qa[kk], ld2(&ks[n * 8 + g][kk * 16 + t * 2]), ld2(&ks[n * 8 + g][kk * 16 + 8 + t * 2]));
        }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (c0 + n * 8 + t * 2 + (j & 1) >= S) s[n][j] = -INFINITY;
                mx[j >> 1] = fmaxf(mx[j >> 1], s[n][j]);
            }
        }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            const float mn = fmaxf(m[r], mx[r]);
            corr[r] = exp2f(m[r] - mn);
            m[r] = mn;
            l[r] *= corr[r];
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            o[n][0] *= corr[0]; o[n][1] *= corr[0];
            o[n][2] *= corr[1]; o[n][3] *= corr[1];
        }
        uint32_t pa[kChunk / 16][4];
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
            const float p0 = exp2f(s[n][0] - m[0]), p1 = exp2f(s[n][1] - m[0]);
            const float p2 = exp2f(s[n][2] - m[1]), p3 = exp2f(s[n][3] - m[1]);
            l[0] += p0 + p1;
            l[1] += p2 + p3;
            pa[n >> 1][(n & 1) * 2 + 0] = O::pack(p0, p1);
            pa[n >> 1][(n & 1) * 2 + 1] = O::pack(p2, p3);
        }
#pragma unroll
        for (int kk = 0; kk < kChunk / 16; ++kk) {
#pragma unroll
            for (int n = 0; n < NT; ++n)
                O::mma(o[n], pa[kk], ld2(&vt[n * 8 + g][kk * 16 + t * 2]), ld2(&vt[n * 8 + g][kk * 16 + 8 + t * 2]));
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = blockIdx.x * kTileQ + w * 16 + g + r * 8;
        if (row >= S) continue;
        const float inv = 1.f / l[r];
        T* dst = out + (base + row) * out_ld + col0;
#pragma unroll
        for (int n = 0; n < NT; ++n)
            *reinterpret_cast<uint32_t*>(dst + n * 8 + t * 2) = O::pack(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
    }
}

template <typename T, int HD>
int launch(const void* q, const void* k, const void* v, int ld, const void* qb, const void* kb, const void* vb,
           const void* cos, const void* sin, void* out, int out_ld, int batches, int S, int heads, float sl2,
           cudaStream_t stream) {
    // 128-row query tiles halve the K / V staging per row on long sequences;
    // short ones keep 64-row tiles so the grid still covers the GPU.
    if (S >= 512) {
        const dim3 grid((S + 127) / 128, heads, batches);
        fused_qkv_attention_kernel<T, HD, 8><<<grid, 256, 0, stream>>>(
            static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v), ld,
            static_cast<const T*>(qb), static_cast<const T*>(kb), static_cast<const T*>(vb),
            static_cast<const T*>(cos), static_cast<const T*>(sin), static_cast<T*>(out), out_ld, S, sl2);
    } else {
        const dim3 grid((S + 63) / 64, heads, batches);
        fused_qkv_attention_kernel<T, HD, 4><<<grid, 128, 0, stream>>>(
            static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v), ld,
            static_cast<const T*>(qb), static_cast<const T*>(kb), static_cast<const T*>(vb),
            static_cast<const T*>(cos), static_cast<const T*>(sin), static_cast<T*>(out), out_ld, S, sl2);
    }
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

int fused_qkv_attention(int dtype, const void* q, const void* k, const void* v, int ld, const void* q_bias,
                        const void* k_bias, const void* v_bias, const void* cos, const void* sin, void* out,
                        int out_ld, int batches, int S, int heads, int hd, float scale, cudaStream_t stream) {
    if (!q || !k || !v || !out || batches < 1 || S < 1 || heads < 1 || ld % 2 != 0 || out_ld % 2 != 0 ||
        (cos == nullptr) != (sin == nullptr) || (dtype != 0 && dtype != 1))
        return static_cast<int>(cudaErrorInvalidValue);
    const float sl2 = scale * 1.4426950408889634f;
#define FRT_FQA(T_, HD_)                                                                                       \
    return launch<T_, HD_>(q, k, v, ld, q_bias, k_bias, v_bias, cos, sin, out, out_ld, batches, S, heads, sl2, \
                           stream)
    if (hd == 64) {
        if (dtype == 0) FRT_FQA(__half, 64);
        FRT_FQA(__nv_bfloat16, 64);
    }
    if (hd == 72) {
        if (dtype == 0) FRT_FQA(__half, 72);
        FRT_FQA(__nv_bfloat16, 72);
    }
#undef FRT_FQA
    return static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace prefill_attention
}  // namespace flash_rt
