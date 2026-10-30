// SPDX-License-Identifier: Apache-2.0
// MolmoAct2 action-expert persistent layer kernel (sm_120a). See the header.
#include "molmoact2_action_mk_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <math.h>

namespace flash_rt {
namespace molmoact2 {
namespace {

constexpr int D = 768;
constexpr int NH = 8;
constexpr int HD = 96;
constexpr int HID = 3072;
constexpr int QKVN = 3 * D;
constexpr int MR = 32;               // padded rows (two 16-row m-atoms)
constexpr int CW = 12;               // compute warps
constexpr int CT = CW * 32;          // compute threads
constexpr int THREADS = CT;
constexpr int A8_STR = D + 16;       // FP8 operand row stride (bytes)
constexpr int KV_STR = HD + 8;       // bf16 attention tile row stride
constexpr int RAW_STR = 3 * HD + 8;  // staged q|k|v head slice row stride
constexpr int XSPLITS = 4;           // cross-attention KV splits per head
constexpr int XCHUNK = 176;          // keys per split (11 x 16)
constexpr int XTILES = XCHUNK / 16;
constexpr int XGROUPS = CW / 2;      // key-tile groups per m-atom
constexpr unsigned PF_CHUNK = 16384;    // bytes per L2 bulk prefetch

// shared-memory carve (bytes); phases are separated by grid barriers
constexpr int SM_W8 = MR * A8_STR;                 // after the FP8 operand
constexpr int SM_XQRAW = 0;                        // cross query, raw projection
constexpr int SM_ATT_Q = MR * KV_STR * 2;          // cross query, normalized
constexpr int XP_STR = XCHUNK + 8;                 // probability tile row stride
constexpr int SM_XP = SM_ATT_Q + MR * KV_STR * 2;  // bf16 probabilities [32][XP_STR]
constexpr int SM_ATT_K = SM_W8;                    // K/V chunk, after the FP8 operand
constexpr int SM_ATT_V = SM_ATT_K + XCHUNK * KV_STR * 2;
constexpr int SM_ATT_END = SM_ATT_V + XCHUNK * KV_STR * 2;
constexpr int SM_FFN_END = SM_W8 + CW * (D / 128) * 2 * 512;   // half K of up+gate per warp
constexpr int SM_TOTAL = SM_ATT_END > SM_FFN_END ? SM_ATT_END : SM_FFN_END;
static_assert(SM_XP + MR * XP_STR * 2 <= SM_W8, "query/probability tiles overlay the FP8 operand");
static_assert(2 * XGROUPS * MR * 4 <= MR * HD * 2, "row max/sum overlay the raw query tile");
static_assert(3 * MR * KV_STR * 2 + MR * RAW_STR * 2 + 3 * HD * 2 + 2 * MR * HD * 2 <= SM_TOTAL,
              "self-attention staging");
static_assert(SM_TOTAL <= 99 * 1024, "shared memory budget");

__device__ __forceinline__ float b2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2b(float v) {
    __nv_bfloat16 r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(*reinterpret_cast<unsigned short*>(&r)) : "f"(v));
    return r;
}
__device__ __forceinline__ float rb(float v) { return b2f(f2b(v)); }

// Cross-CTA activations are read L1-bypassing (.cg); ordinary loads so the
// compiler can overlap independent requests.
__device__ __forceinline__ uint32_t ldcg_u32(const void* p) {
    return __ldcg(reinterpret_cast<const unsigned int*>(p));
}
__device__ __forceinline__ float ldcg_f32(const float* p) { return __ldcg(p); }
__device__ __forceinline__ uint4 ldcg_u128(const void* p) {
    return __ldcg(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ uint4 ldg_w128(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ldg_w64(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* bar) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(a) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred p;\nW_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra W_%=;\n}\n" :: "r"(a), "r"(parity) : "memory");
}
// One bulk global->shared copy completing on `bar` (bytes % 16 == 0).
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes,
                                         uint64_t* bar) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(d), "l"(src), "r"(bytes), "r"(b) : "memory");
}
__device__ __forceinline__ uint32_t lds_u32(const void* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_nvf4(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sfa,
                                         uint32_t sfb) {
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z),
          "r"(sfb), "h"(z), "h"(z));
}

// B fragments of two adjacent n8 tiles (columns c0..c0+15) of a row-major
// bf16 V tile, keys k0..k0+15, via one transposing ldmatrix: {b0, b1} of
// tile c0 and {b0, b1} of tile c0 + 8.
__device__ __forceinline__ void ldsm_v_x4(const __nv_bfloat16* V, int k0, int c0, uint32_t (&b)[4]) {
    const int lane = threadIdx.x & 31, m = lane >> 3;
    const __nv_bfloat16* row = V + (k0 + (m & 1) * 8 + (lane & 7)) * KV_STR + c0 + (m >> 1) * 8;
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(a));
}

// ── NVFP4 activation numerics (identical to the skinny family) ─────────────
__device__ __forceinline__ uint8_t act_scale_code(float v) {
    if (!(v > 0.f)) return 0;
    if (v >= 448.f) return 0x7E;
    const __nv_fp8_e4m3 q(v);
    return q.__x;
}
__device__ __forceinline__ float ue4m3_value(uint8_t b) {
    __nv_fp8_e4m3 q;
    q.__x = b;
    return static_cast<float>(q);
}
__device__ __forceinline__ uint8_t e2m1x2(float lo, float hi) {
    uint16_t r;
    asm("{\n"
        ".reg .b8 b0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %1, %2;\n"
        "mov.b16 %0, {b0, b0};\n"
        "}"
        : "=h"(r)
        : "f"(hi), "f"(lo));
    return static_cast<uint8_t>(r & 0xFF);
}
__device__ __forceinline__ void quant_block16(const float* v, uint8_t* out8, uint8_t* sf) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const uint8_t s = act_scale_code(amax * (1.0f / 6.0f));
    const float d = ue4m3_value(s);
    const float inv = d > 0.f ? 1.0f / d : 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) out8[i] = e2m1x2(v[2 * i] * inv, v[2 * i + 1] * inv);
    *sf = s;
}

// The four lanes of a quad hold four values each of one 16-element block of
// an attention output row (already scaled by 1 / global scale); quantize it
// to NVFP4 like quant_block16 and store the lane's two bytes and the scale.
__device__ __forceinline__ void quad_quant_store(float v0, float v1, float v2, float v3,
                                                 uint8_t* row_fp4, uint8_t* row_sf, int blk,
                                                 int b0, int b1, bool store, bool write_sf) {
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const uint8_t s = act_scale_code(amax * (1.0f / 6.0f));
    const float d = ue4m3_value(s);
    const float inv = d > 0.f ? 1.0f / d : 0.f;
    if (!store) return;
    row_fp4[blk * 8 + b0] = e2m1x2(v0 * inv, v1 * inv);
    row_fp4[blk * 8 + b1] = e2m1x2(v2 * inv, v3 * inv);
    if (write_sf) row_sf[blk] = s;
}

// Compute CTAs are 0 .. s_nc-1; the remaining CTAs only stream L2 prefetches.
__shared__ int s_nc;

// ── barriers ──────────────────────────────────────────────────────────────
__device__ __forceinline__ void csync() { asm volatile("bar.sync 1, %0;" :: "n"(CT) : "memory"); }

// Grid barrier on one monotonic 64-bit arrival counter: each compute CTA
// adds 1 (release) and thread 0 spins until all s_nc arrivals of this
// barrier are visible. `target` is this barrier's completion count.
__device__ __forceinline__ void grid_sync(unsigned long long* ctr, unsigned long long& target) {
    csync();
    if (threadIdx.x == 0) {
        target += static_cast<unsigned long long>(s_nc);
        __threadfence();
        asm volatile("red.release.gpu.global.add.u64 [%0], 1;" :: "l"(ctr) : "memory");
        unsigned long long v;
        do {
            asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(v) : "l"(ctr) : "memory");
        } while (static_cast<long long>(v - target) < 0);
        __threadfence();
    }
    csync();
}

__device__ __forceinline__ void prefetch_layer(const ActionMkLayer* L, int lane, int idx, int cnt) {
    if (L == nullptr) return;
    for (int r = 0; r < L->pf_count; ++r) {
        const uint8_t* base = static_cast<const uint8_t*>(L->pf_ptr[r]);
        const unsigned long long bytes = L->pf_bytes[r];
        const unsigned long long n = (bytes + PF_CHUNK - 1) / PF_CHUNK;
        for (unsigned long long c = static_cast<unsigned long long>(idx) * 32 + lane; c < n;
             c += static_cast<unsigned long long>(cnt) * 32) {
            const unsigned long long off = c * PF_CHUNK;
            const unsigned len =
                static_cast<unsigned>(bytes - off < PF_CHUNK ? bytes - off : PF_CHUNK) & ~15u;
            if (len)
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                             :: "l"(base + off), "r"(len) : "memory");
        }
    }
}

// ── modulated RMSNorm -> static FP8 into shared memory ─────────────────────
// Elementwise sequence of mod_rms_norm_rows768_quant_fp8_exact (norm, bf16
// rounding, (1 + scale) modulation, shift, static-scale FP8). Lane l owns
// elements [8l, 8l+8) of each 256-wide third of a row (16-byte accesses);
// each warp normalizes up to three rows at once. With WAIT the caller's
// staged cp.async groups are drained before the closing barrier.
template <bool WAIT = true>
__device__ void mod_norm_fp8_smem(const __nv_bfloat16* x, const __nv_bfloat16* scale,
                                  const __nv_bfloat16* shift, float qs, int rows, float eps,
                                  uint8_t* sA) {
    constexpr int RPW = (MR + CW - 1) / CW;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float inv_s = 1.0f / qs;
    uint4 sb[3], hb[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        sb[k] = __ldg(reinterpret_cast<const uint4*>(scale + 256 * k + 8 * lane));
        hb[k] = __ldg(reinterpret_cast<const uint4*>(shift + 256 * k + 8 * lane));
    }
    uint4 xv[RPW][3];
#pragma unroll
    for (int t = 0; t < RPW; ++t) {
        const int r = warp + CW * t;
#pragma unroll
        for (int k = 0; k < 3; ++k)
            xv[t][k] = r < rows ? ldcg_u128(x + static_cast<size_t>(r) * D + 256 * k + 8 * lane)
                                : make_uint4(0, 0, 0, 0);
    }
    float ss[RPW];
#pragma unroll
    for (int t = 0; t < RPW; ++t) {
        float a = 0.f;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(&xv[t][k]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[e]));
                a += f.x * f.x + f.y * f.y;
            }
        }
        ss[t] = a;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int t = 0; t < RPW; ++t) ss[t] += __shfl_xor_sync(0xffffffffu, ss[t], o);
    float rms[RPW];
#pragma unroll
    for (int t = 0; t < RPW; ++t) rms[t] = rsqrtf(ss[t] / 768.0f + eps);
    // bf16x2 forms of the same roundings: the products/sums of two bf16
    // values are exact in fp32, so one bf16x2 op equals op-then-round.
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const uint32_t* sw = reinterpret_cast<const uint32_t*>(&sb[k]);
        const uint32_t* hw = reinterpret_cast<const uint32_t*>(&hb[k]);
        __nv_bfloat162 a2[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 sv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sw[e]));
            a2[e] = __floats2bfloat162_rn(1.0f + sv.x, 1.0f + sv.y);
        }
#pragma unroll
        for (int t = 0; t < RPW; ++t) {
            const int r = warp + CW * t;
            if (r >= rows) continue;
            const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[t][k]);
            uint32_t q[2];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xw[e]));
                const __nv_bfloat162 n2 = __floats2bfloat162_rn(xf.x * rms[t], xf.y * rms[t]);
                const float2 m = __bfloat1622float2(
                    __hadd2(__hmul2(n2, a2[e]), *reinterpret_cast<const __nv_bfloat162*>(&hw[e])));
                uint16_t h;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h) : "f"(m.y * inv_s), "f"(m.x * inv_s));
                if (e & 1) q[e >> 1] |= static_cast<uint32_t>(h) << 16;
                else q[e >> 1] = h;
            }
            *reinterpret_cast<uint2*>(sA + r * A8_STR + 256 * k + 8 * lane) = make_uint2(q[0], q[1]);
        }
    }
    for (int i = threadIdx.x; i < (MR - rows) * (D / 16); i += CT) {
        const int r = rows + i / (D / 16);
        *reinterpret_cast<uint4*>(sA + r * A8_STR + (i % (D / 16)) * 16) = make_uint4(0, 0, 0, 0);
    }
    if constexpr (WAIT) cp_async_wait_all();   // weight chunks staged by the caller
    csync();
}

// Fragment-major FP8 weight chunk loads for one n8 tile, kc in [kc0, kc0+NKC).
template <int NKC>
__device__ __forceinline__ void fp8_load(const uint8_t* W, int tile, int kc0, uint4 (&w)[NKC]) {
    constexpr int KCS = D / 64;
    const int lane = threadIdx.x & 31;
    const uint8_t* wp = W + ((static_cast<size_t>(tile) * KCS + kc0) * 32 + lane) * 16;
#pragma unroll
    for (int kc = 0; kc < NKC; ++kc) w[kc] = ldg_w128(wp + kc * 512);
}

// Same chunks staged into shared memory with cp.async (no registers held
// across the norm that runs while they land).
template <int NKC>
__device__ __forceinline__ void fp8_stage(const uint8_t* W, int tile, int kc0, uint8_t* dst) {
    constexpr int KCS = D / 64;
    const int lane = threadIdx.x & 31;
    const uint8_t* wp = W + ((static_cast<size_t>(tile) * KCS + kc0) * 32 + lane) * 16;
#pragma unroll
    for (int kc = 0; kc < NKC; ++kc) cp_async16(dst + kc * 512 + lane * 16, wp + kc * 512, true);
}

template <int NKC>
__device__ __forceinline__ void fp8_mma_smem(const uint8_t* sA, int kc0, const uint8_t* sW,
                                             float acc[2][4]) {
    const int lane = threadIdx.x & 31, t0 = lane & 3, t1 = lane >> 2;
#pragma unroll
    for (int kc = 0; kc < NKC; ++kc) {
        const uint4 w = *reinterpret_cast<const uint4*>(sW + kc * 512 + lane * 16);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const uint32_t b0 = ks ? w.z : w.x;
            const uint32_t b1 = ks ? w.w : w.y;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const uint8_t* ra =
                    sA + (mt * 16 + t1) * A8_STR + (kc0 + kc) * 64 + ks * 32 + t0 * 4;
                mma_fp8(acc[mt], lds_u32(ra), lds_u32(ra + 8 * A8_STR), lds_u32(ra + 16),
                        lds_u32(ra + 8 * A8_STR + 16), b0, b1);
            }
        }
    }
}

// Up and gate together: each A fragment feeds both weight streams.
template <int NKC, bool SMEM>
__device__ __forceinline__ void fp8_mma2(const uint8_t* sA, int kc0, const uint8_t* sWu,
                                         const uint8_t* sWg, const uint4* wu, const uint4* wg,
                                         float au[2][4], float ag[2][4]) {
    const int lane = threadIdx.x & 31, t0 = lane & 3, t1 = lane >> 2;
#pragma unroll
    for (int kc = 0; kc < NKC; ++kc) {
        const uint4 u = SMEM ? *reinterpret_cast<const uint4*>(sWu + kc * 512 + lane * 16) : wu[kc];
        const uint4 g = SMEM ? *reinterpret_cast<const uint4*>(sWg + kc * 512 + lane * 16) : wg[kc];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const uint8_t* ra =
                    sA + (mt * 16 + t1) * A8_STR + (kc0 + kc) * 64 + ks * 32 + t0 * 4;
                const uint32_t a0 = lds_u32(ra), a1 = lds_u32(ra + 8 * A8_STR);
                const uint32_t a2 = lds_u32(ra + 16), a3 = lds_u32(ra + 8 * A8_STR + 16);
                mma_fp8(au[mt], a0, a1, a2, a3, ks ? u.z : u.x, ks ? u.w : u.y);
                mma_fp8(ag[mt], a0, a1, a2, a3, ks ? g.z : g.x, ks ? g.w : g.y);
            }
        }
    }
}

template <int NKC>
__device__ __forceinline__ void fp8_mma(const uint8_t* sA, int kc0, const uint4 (&w)[NKC],
                                        float acc[2][4]) {
    const int lane = threadIdx.x & 31, t0 = lane & 3, t1 = lane >> 2;
#pragma unroll
    for (int kc = 0; kc < NKC; ++kc)
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const uint32_t b0 = ks ? w[kc].z : w[kc].x;
            const uint32_t b1 = ks ? w[kc].w : w[kc].y;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const uint8_t* ra =
                    sA + (mt * 16 + t1) * A8_STR + (kc0 + kc) * 64 + ks * 32 + t0 * 4;
                mma_fp8(acc[mt], lds_u32(ra), lds_u32(ra + 8 * A8_STR), lds_u32(ra + 16),
                        lds_u32(ra + 8 * A8_STR + 16), b0, b1);
            }
        }
}

// FP8 GEMM phase (K = 768) preceded by the modulated norm: the CTA owns n8
// tiles blockIdx.x + i*grid; KS warps split K and reduce through smem.
// out[rows, N] = bf16(acc * alpha).
template <int KS>
__device__ void norm_fp8_gemm_phase(const __nv_bfloat16* x, const __nv_bfloat16* mod, float qs,
                                    float eps, const uint8_t* W, int N, float alpha,
                                    __nv_bfloat16* out, int rows, uint8_t* smem) {
    constexpr int NKC = (D / 64) / KS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t0 = lane & 3, t1 = lane >> 2;
    const int ntiles = N / 8;
    const int nlocal = (ntiles - static_cast<int>(blockIdx.x) + s_nc - 1) / s_nc;
    const int li = warp / KS, kp = warp % KS;
    const bool active = li < nlocal;
    const int tile = blockIdx.x + li * s_nc;
    uint8_t* sW = smem + SM_W8 + warp * NKC * 512;
    if (active) fp8_stage<NKC>(W, tile, kp * NKC, sW);
    mod_norm_fp8_smem(x, mod, mod + D, qs, rows, eps, smem);   // waits for cp.async
    float acc[2][4] = {};
    if (active) fp8_mma_smem<NKC>(smem, kp * NKC, sW, acc);
    if constexpr (KS > 1) {
        float* red = reinterpret_cast<float*>(smem + SM_W8 + CW * NKC * 512);
        if (active && kp != 0) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    red[(((li * KS + kp) * 2 + mt) * 4 + i) * 32 + lane] = acc[mt][i];
        }
        csync();
        if (active && kp == 0) {
#pragma unroll
            for (int q = 1; q < KS; ++q)
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        acc[mt][i] += red[(((li * KS + q) * 2 + mt) * 4 + i) * 32 + lane];
        }
    }
    if (active && kp == 0) {
        const int col = tile * 8 + 2 * t0;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = mt * 16 + t1 + 8 * h;
                if (row < rows)
                    *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(row) * N + col) =
                        pack_bf2(acc[mt][2 * h] * alpha, acc[mt][2 * h + 1] * alpha);
            }
    }
}

// ── skinny NVFP4 phase: x[:, n] += gate * (bias + alpha * A @ W^T) ─────────
// A is the NVFP4 operand emitted by the producing phase. Each CTA owns n8
// tiles blockIdx.x + i*grid; KS warps split K for one tile and reduce
// through shared memory.
template <int K, int KS>
__device__ void skinny_phase(const uint8_t* A, const uint8_t* A_sf, float alpha,
                             const uint8_t* W, const uint8_t* SFB, const __nv_bfloat16* bias,
                             const __nv_bfloat16* gate, __nv_bfloat16* x, int rows,
                             uint8_t* smem) {
    constexpr int KCH = K / 128;
    constexpr int CPW = KCH / KS;
    static_assert(KCH % KS == 0, "K chunks must split evenly");
    constexpr int AB = K / 2;
    constexpr int ASTR = AB + 16;
    constexpr int SB = K / 16;
    constexpr int SSTR = SB + 16;
    uint8_t* sA = smem;
    uint8_t* sS = smem + MR * ASTR;
    float* red = reinterpret_cast<float*>(smem + ((MR * ASTR + MR * SSTR + 127) & ~127));
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, t0 = lane & 3, t1 = lane >> 2;
    const int ntiles = D / 8;
    const int nlocal = (ntiles - static_cast<int>(blockIdx.x) + s_nc - 1) / s_nc;
    if (nlocal <= 0) return;   // uniform per CTA
    const int li = warp / KS, kh = warp % KS;
    const bool active = li < nlocal;
    const int tile = blockIdx.x + li * s_nc;
    uint4 w[CPW];
    uint2 sb[CPW];
    // epilogue operands (bias, gate, the residual rows of this tile) issued
    // with the weights so they are not dependent round trips at the end
    const int col = tile * 8 + 2 * t0;
    uint32_t bg2 = 0u, gg2 = 0u, xo[2][2] = {{0u, 0u}, {0u, 0u}};
    if (active && kh == 0) {
        bg2 = __ldg(reinterpret_cast<const unsigned int*>(bias + col));
        gg2 = __ldg(reinterpret_cast<const unsigned int*>(gate + col));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = mt * 16 + t1 + 8 * h;
                if (row < rows) xo[mt][h] = ldcg_u32(x + static_cast<size_t>(row) * D + col);
            }
    }
    if (active) {
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const size_t tg = static_cast<size_t>(tile) * KCH + kh * CPW + c;
            w[c] = ldg_w128(W + (tg * 32 + lane) * 16);
            sb[c] = ldg_w64(SFB + (tg * 8 + t1) * 8);
        }
    }
    for (int c = tid; c < MR * (AB / 16); c += CT) {
        const int row = c / (AB / 16), off = (c % (AB / 16)) * 16;
        const bool v = row < rows;
        cp_async16(sA + row * ASTR + off, A + static_cast<size_t>(v ? row : 0) * AB + off, v);
    }
    for (int c = tid; c < MR * (SB / 16); c += CT) {
        const int row = c / (SB / 16), off = (c % (SB / 16)) * 16;
        const bool v = row < rows;
        cp_async16(sS + row * SSTR + off, A_sf + static_cast<size_t>(v ? row : 0) * SB + off, v);
    }
    cp_async_wait_all();
    csync();

    float acc[2][4] = {};
    if (active) {
        const int sau = (lane & 1) * 8 + (lane >> 2);
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const int cg = kh * CPW + c;
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const uint8_t* ra = sA + (mt * 16 + t1) * ASTR + cg * 64 + s * 32 + t0 * 4;
                    const uint32_t sfa = lds_u32(sS + (mt * 16 + sau) * SSTR + cg * 8 + s * 4);
                    mma_nvf4(acc[mt], lds_u32(ra), lds_u32(ra + 8 * ASTR), lds_u32(ra + 16),
                             lds_u32(ra + 8 * ASTR + 16), s == 0 ? w[c].x : w[c].z,
                             s == 0 ? w[c].y : w[c].w, sfa, s == 0 ? sb[c].x : sb[c].y);
                }
        }
        if (kh != 0) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    red[(((li * KS + kh) * 2 + mt) * 4 + i) * 32 + lane] = acc[mt][i];
        }
    }
    csync();
    if (active && kh == 0) {
#pragma unroll
        for (int q = 1; q < KS; ++q)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    acc[mt][i] += red[(((li * KS + q) * 2 + mt) * 4 + i) * 32 + lane];
        const float2 bf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bg2));
        const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg2));
        const float b0 = bf.x, b1 = bf.y, g0 = gf.x, g1 = gf.y;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = mt * 16 + t1 + 8 * h;
                if (row >= rows) continue;
                float o[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float a = rb(rb(acc[mt][2 * h + e] * alpha) + (e ? b1 : b0));
                    o[e] = rb(a * (e ? g1 : g0));
                }
                __nv_bfloat16* xp = x + static_cast<size_t>(row) * D + col;
                const __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(&xo[mt][h]);
                *reinterpret_cast<uint32_t*>(xp) = pack_bf2(b2f(xv.x) + o[0], b2f(xv.y) + o[1]);
            }
    }
}

// ── self attention for one head (qkv split + bias + q/k norm + RoPE) ───────
// The head's q|k|v slice, its bias and the RoPE rows are staged in shared
// memory; warp (m-atom, 16-column group) computes the scores for its m-atom
// and the 16 output columns it owns.
__device__ void self_attn_head(const ActionMkParams& p, const ActionMkLayer& L, int h, int mq,
                               float ags, uint8_t* smem) {
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + MR * KV_STR;
    __nv_bfloat16* sV = sK + MR * KV_STR;
    __nv_bfloat16* sR = sV + MR * KV_STR;     // [rows][q | k | v] raw projection
    __nv_bfloat16* sB = sR + MR * RAW_STR;    // [q | k | v] bias of this head
    __nv_bfloat16* sC = sB + 3 * HD;          // [rows][96] RoPE cos
    __nv_bfloat16* sS = sC + MR * HD;         // [rows][96] RoPE sin
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, t0 = lane & 3, t1 = lane >> 2;
    const int rows = p.rows;
    for (int c = tid; c < rows * 36; c += CT) {
        const int row = c / 36, part = (c % 36) / 12, off = (c % 12) * 8;
        cp_async16(sR + row * RAW_STR + part * HD + off,
                   p.qkv + static_cast<size_t>(row) * QKVN + part * D + h * HD + off, true);
    }
    for (int c = tid; c < 36 + 2 * rows * 12; c += CT) {
        if (c < 36) {
            const int part = c / 12, off = (c % 12) * 8;
            cp_async16(sB + part * HD + off, L.qkv_b + part * D + h * HD + off, true);
        } else {
            const int cc = c - 36, which = cc / (rows * 12), r = (cc / 12) % rows, off = (cc % 12) * 8;
            cp_async16((which ? sS : sC) + r * HD + off,
                       (which ? p.rope_sin : p.rope_cos) + r * HD + off, true);
        }
    }
    cp_async_wait_all();
    csync();
    // Q/K: elementwise sequence of qkv_split_bias_norm_rope_kernel (bias, no
    // weight, RMSNorm over 96, RoPE)
    // this CTA owns query m-atom mq: its 16 Q rows and all K rows
    const int q0 = mq * 16, nq = max(0, min(16, rows - q0));
    // half-warp per row: lane l16 owns d = l16, l16 + 16, l16 + 32 (< 48)
    const int hw = lane >> 4, l16 = lane & 15;
    for (int t0r = warp * 2; t0r < nq + rows; t0r += CW * 2) {
        const int t = t0r + hw;
        const bool valid = t < nq + rows;
        const bool is_q = t < nq;
        const int row = valid ? (is_q ? q0 + t : t - nq) : 0;
        const __nv_bfloat16* src = sR + row * RAW_STR + (is_q ? 0 : HD);
        const __nv_bfloat16* bias = sB + (is_q ? 0 : HD);
        __nv_bfloat16* dst = (is_q ? sQ : sK) + row * KV_STR;
        float va[3], vb[3];
#pragma unroll
        for (int e = 0; e < 3; ++e) {
            const int d = l16 + 16 * e;
            va[e] = rb(b2f(src[d]) + b2f(bias[d]));
            vb[e] = rb(b2f(src[d + HD / 2]) + b2f(bias[d + HD / 2]));
        }
        float ssq = 0.f;
#pragma unroll
        for (int e = 0; e < 3; ++e) ssq += va[e] * va[e] + vb[e] * vb[e];
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
        if (!valid) continue;
        const float inv = rsqrtf(ssq / static_cast<float>(HD) + p.eps);
#pragma unroll
        for (int e = 0; e < 3; ++e) {
            const int d = l16 + 16 * e;
            const float nlo = rb(va[e] * inv);
            const float nhi = rb(vb[e] * inv);
            const float c = b2f(sC[row * HD + d]);
            const float si = b2f(sS[row * HD + d]);
            const float lo = rb(nlo * c);
            const float lo2 = rb(nhi * si);
            const float hi = rb(nhi * c);
            const float hi2 = rb(nlo * si);
            dst[d] = f2b(rb(lo - lo2));
            dst[d + HD / 2] = f2b(rb(hi + hi2));
        }
    }
    for (int i = tid; i < rows * HD; i += CT) {
        const int row = i / HD, d = i % HD;
        sV[row * KV_STR + d] = f2b(b2f(sR[row * RAW_STR + 2 * HD + d]) + b2f(sB[2 * HD + d]));
    }
    for (int i = tid; i < (MR - rows) * HD; i += CT) {
        const int row = rows + i / HD, d = i % HD;
        sQ[row * KV_STR + d] = f2b(0.f);
        sK[row * KV_STR + d] = f2b(0.f);
        sV[row * KV_STR + d] = f2b(0.f);
    }
    csync();
    if (warp >= HD / 16) return;
    const int mt = mq, dg = warp;   // dg: 16-column output group (0..5)
    const float sl2 = (1.0f / sqrtf(static_cast<float>(HD))) * 1.4426950408889634f;
    float s[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[n][i] = 0.f;
#pragma unroll
    for (int kk = 0; kk < HD / 16; ++kk) {
        const __nv_bfloat16* qa = sQ + (mt * 16 + t1) * KV_STR + kk * 16 + 2 * t0;
        const uint32_t a0 = lds_u32(qa), a1 = lds_u32(qa + 8 * KV_STR);
        const uint32_t a2 = lds_u32(qa + 8), a3 = lds_u32(qa + 8 * KV_STR + 8);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const __nv_bfloat16* kb = sK + (n * 8 + t1) * KV_STR + kk * 16 + 2 * t0;
            mma_bf16(s[n], a0, a1, a2, a3, lds_u32(kb), lds_u32(kb + 8));
        }
    }
    float m[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int key = n * 8 + 2 * t0 + (i & 1);
            if (key >= rows) s[n][i] = -INFINITY;
            m[i >> 1] = fmaxf(m[i >> 1], s[n][i]);
        }
    float l[2] = {0.f, 0.f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 1));
        m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 2));
    }
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float ms = m[i >> 1] == -INFINITY ? 0.f : m[i >> 1] * sl2;
            s[n][i] = exp2f(s[n][i] * sl2 - ms);
            l[i >> 1] += s[n][i];
        }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
    float o[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[n][i] = 0.f;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const uint32_t a0 = pack_bf2(s[2 * kt][0], s[2 * kt][1]);
        const uint32_t a1 = pack_bf2(s[2 * kt][2], s[2 * kt][3]);
        const uint32_t a2 = pack_bf2(s[2 * kt + 1][0], s[2 * kt + 1][1]);
        const uint32_t a3 = pack_bf2(s[2 * kt + 1][2], s[2 * kt + 1][3]);
        uint32_t bv[4];
        ldsm_v_x4(sV, kt * 16, dg * 16, bv);
        mma_bf16(o[0], a0, a1, a2, a3, bv[0], bv[1]);
        mma_bf16(o[1], a0, a1, a2, a3, bv[2], bv[3]);
    }
    float inv[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) inv[r] = (l[r] == 0.f || l[r] != l[r]) ? 1.f : 1.f / l[r];
    // bf16 output, emitted as the out-projection's NVFP4 operand: the warp's
    // 16 columns are exactly one block of each of its rows
    const float inv_gs = 1.0f / ags;
    const int blk = h * (HD / 16) + dg;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = mt * 16 + t1 + 8 * r;
        const bool ok = row < rows;
        const int rr = ok ? row : 0;
        quad_quant_store(rb(o[0][2 * r] * inv[r]) * inv_gs, rb(o[0][2 * r + 1] * inv[r]) * inv_gs,
                         rb(o[1][2 * r] * inv[r]) * inv_gs, rb(o[1][2 * r + 1] * inv[r]) * inv_gs,
                         p.attn_fp4 + static_cast<size_t>(rr) * (D / 2),
                         p.attn_sf + static_cast<size_t>(rr) * (D / 16), blk, t0, 4 + t0, ok,
                         t0 == 0);
    }
}

// ── cross attention: one (head, KV split) unit, last CTA per head combines ─
__device__ void cross_attn_unit(const ActionMkParams& p, const ActionMkLayer& L,
                                const ActionMkScales& sc, const __nv_bfloat16* mod, int h, int sp,
                                uint8_t* smem, uint64_t* kv_bar, uint32_t& kv_phase) {
    const float ags = sc.co_ags;
    __nv_bfloat16* sQraw = reinterpret_cast<__nv_bfloat16*>(smem + SM_XQRAW);
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem + SM_ATT_Q);
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem + SM_ATT_K);
    __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(smem + SM_ATT_V);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, t0 = lane & 3, t1 = lane >> 2;
    const int rows = p.rows;
    const int k0 = sp * XCHUNK;
    const int nk = min(XCHUNK, p.ctx_len - k0);
    // K/V chunk: two bulk copies of the head-major, row-padded context; they
    // land while the norm and the query projection run. Rows past nk are the
    // zero padding rows of the context buffers and are masked below.
    if (tid == 0) {
        constexpr uint32_t bytes = XCHUNK * KV_STR * 2;
        const size_t off = (static_cast<size_t>(h) * p.ctx_rows + k0) * KV_STR;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(kv_bar));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(b), "r"(2 * bytes) : "memory");
        bulk_g2s(sK, L.ctx_k + off, bytes, kv_bar);
        bulk_g2s(sV, L.ctx_v + off, bytes, kv_bar);
    }
    // cross query for this head: norm -> FP8, warp w computes n8 tile w
    mod_norm_fp8_smem<false>(p.x, mod, mod + D, sc.cq_qs, rows, p.eps, smem);
    {
        uint4 w[D / 64];
        fp8_load<D / 64>(L.cq_w, h * (HD / 8) + warp, 0, w);
        float acc[2][4] = {};
        fp8_mma<D / 64>(smem, 0, w, acc);
        csync();   // FP8 operand dead: the raw query tile overlays it
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
                *reinterpret_cast<uint32_t*>(sQraw + (mt * 16 + t1 + 8 * hh) * HD + warp * 8 + 2 * t0) =
                    pack_bf2(acc[mt][2 * hh] * sc.cq_alpha, acc[mt][2 * hh + 1] * sc.cq_alpha);
        csync();
    }
    // query bias + RMSNorm over 96 (bias_rms_norm_rows96 sequence); half-warp
    // per row, lane l16 owns the bf16 pairs l16, l16 + 16, l16 + 32
    const int hw = lane >> 4, l16 = lane & 15;
    const unsigned int* b2 = reinterpret_cast<const unsigned int*>(L.cq_b + h * HD);
    float2 bias2[3];
#pragma unroll
    for (int e = 0; e < 3; ++e) {
        const uint32_t bits = __ldg(b2 + l16 + 16 * e);
        bias2[e] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
    }
    for (int r0 = warp * 2; r0 < MR; r0 += CW * 2) {
        const int r = r0 + hw;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(sQ + r * KV_STR);
        const bool ok = r < rows;
        const uint32_t* xr = reinterpret_cast<const uint32_t*>(sQraw + (ok ? r : 0) * HD);
        float2 v[3];
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 3; ++e) {
            const uint32_t bits = xr[l16 + 16 * e];
            const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
            v[e].x = rb(xv.x + bias2[e].x);
            v[e].y = rb(xv.y + bias2[e].y);
            ss += v[e].x * v[e].x + v[e].y * v[e].y;
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float rms = rsqrtf(ss / 96.0f + p.eps);
#pragma unroll
        for (int e = 0; e < 3; ++e)
            o2[l16 + 16 * e] = ok ? __halves2bfloat162(f2b(v[e].x * rms), f2b(v[e].y * rms))
                                  : __floats2bfloat162_rn(0.f, 0.f);
    }
    mbar_wait(kv_bar, kv_phase);   // K/V chunk
    kv_phase ^= 1u;
    csync();

    // Two-pass softmax over the chunk: warps (m-atom, key group) compute
    // scores and the chunk row max, then write bf16 probabilities to shared
    // memory; warps (m-atom, 16-column group) then run P @ V over all keys,
    // so no per-warp output partials need combining.
    constexpr int TPG = (XTILES + XGROUPS - 1) / XGROUPS;   // key tiles per warp
    const int mt = warp & 1, g = warp >> 1;
    const float sl2 = (1.0f / sqrtf(static_cast<float>(HD))) * 1.4426950408889634f;
    __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(smem + SM_XP);
    float* sM = reinterpret_cast<float*>(smem + SM_XQRAW);   // [group][32 rows]
    float* sL = sM + XGROUPS * MR;
    float s[TPG][2][4];
    float m[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int jj = 0; jj < TPG; ++jj) {
        const int tile = g + XGROUPS * jj;
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
#pragma unroll
            for (int i = 0; i < 4; ++i) s[jj][nn][i] = 0.f;
        if (tile < XTILES && tile * 16 < nk) {
#pragma unroll
            for (int kk = 0; kk < HD / 16; ++kk) {
                const __nv_bfloat16* q = sQ + (mt * 16 + t1) * KV_STR + kk * 16 + 2 * t0;
                const uint32_t a0 = lds_u32(q), a1 = lds_u32(q + 8 * KV_STR);
                const uint32_t a2 = lds_u32(q + 8), a3 = lds_u32(q + 8 * KV_STR + 8);
#pragma unroll
                for (int nn = 0; nn < 2; ++nn) {
                    const __nv_bfloat16* kb = sK + (tile * 16 + nn * 8 + t1) * KV_STR + kk * 16 + 2 * t0;
                    mma_bf16(s[jj][nn], a0, a1, a2, a3, lds_u32(kb), lds_u32(kb + 8));
                }
            }
        }
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int key = tile * 16 + nn * 8 + 2 * t0 + (i & 1);
                if (tile >= XTILES || key >= nk) s[jj][nn][i] = -INFINITY;
                m[i >> 1] = fmaxf(m[i >> 1], s[jj][nn][i]);
            }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 1));
        m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 2));
    }
    if (t0 == 0) {
        sM[g * MR + mt * 16 + t1] = m[0];
        sM[g * MR + mt * 16 + t1 + 8] = m[1];
    }
    csync();
    float ms[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float M = -INFINITY;
#pragma unroll
        for (int gg = 0; gg < XGROUPS; ++gg) M = fmaxf(M, sM[gg * MR + mt * 16 + t1 + 8 * r]);
        ms[r] = M == -INFINITY ? 0.f : M * sl2;
    }
    float l[2] = {0.f, 0.f};
#pragma unroll
    for (int jj = 0; jj < TPG; ++jj) {
        const int tile = g + XGROUPS * jj;
        if (tile >= XTILES) continue;
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            float pv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                pv[i] = exp2f(s[jj][nn][i] * sl2 - ms[i >> 1]);
                l[i >> 1] += pv[i];
            }
            __nv_bfloat16* pp = sP + (mt * 16 + t1) * XP_STR + tile * 16 + nn * 8 + 2 * t0;
            *reinterpret_cast<uint32_t*>(pp) = pack_bf2(pv[0], pv[1]);
            *reinterpret_cast<uint32_t*>(pp + 8 * XP_STR) = pack_bf2(pv[2], pv[3]);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
    if (t0 == 0) {
        sL[g * MR + mt * 16 + t1] = l[0];
        sL[g * MR + mt * 16 + t1 + 8] = l[1];
    }
    csync();
    const int dg = g;   // 16-column output group
    float o[2][4] = {};
#pragma unroll
    for (int tile = 0; tile < XTILES; ++tile) {
        if (tile * 16 >= nk) break;
        const __nv_bfloat16* pa = sP + (mt * 16 + t1) * XP_STR + tile * 16 + 2 * t0;
        const uint32_t a0 = lds_u32(pa), a1 = lds_u32(pa + 8 * XP_STR);
        const uint32_t a2 = lds_u32(pa + 8), a3 = lds_u32(pa + 8 * XP_STR + 8);
        uint32_t bv[4];
        ldsm_v_x4(sV, tile * 16, dg * 16, bv);
        mma_bf16(o[0], a0, a1, a2, a3, bv[0], bv[1]);
        mma_bf16(o[1], a0, a1, a2, a3, bv[2], bv[3]);
    }
    float* part = p.xo_part + static_cast<size_t>(h * XSPLITS + sp) * MR * HD;
    float* plse = p.xo_lse + static_cast<size_t>(h * XSPLITS + sp) * MR;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = mt * 16 + t1 + 8 * r;
        float L = 0.f;
#pragma unroll
        for (int gg = 0; gg < XGROUPS; ++gg) L += sL[gg * MR + row];
        const float inv = 1.0f / L;
        if (row < rows) {
#pragma unroll
            for (int n = 0; n < 2; ++n)
                *reinterpret_cast<float2*>(part + row * HD + (2 * dg + n) * 8 + 2 * t0) =
                    make_float2(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
            if (dg == 0 && t0 == 0) plse[row] = ms[r] + log2f(L);
        }
    }
    __threadfence();
    csync();
    // The head's split CTAs meet on a monotonic arrival counter, then each
    // combines its own quarter of the rows (deterministic, no tail CTA).
    if (tid == 0) {
        const unsigned old = atomicAdd(&p.counters[h], 1u);
        const unsigned target = (old / XSPLITS + 1) * XSPLITS;
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p.counters + h) : "memory");
        } while (static_cast<int>(v - target) < 0);
        __threadfence();
    }
    csync();
    const int nsp = (p.ctx_len + XCHUNK - 1) / XCHUNK;
    constexpr int RPS = MR / XSPLITS;
    // quad q of this CTA's combine handles row sp*RPS + q/6, columns
    // 16*(q%6) + 4*lane_in_quad .. +3, and emits the NVFP4 operand block
    if (tid < RPS * (HD / 16) * 4) {
        const int q = tid >> 2, sub = tid & 3;
        const int r = sp * RPS + q / (HD / 16), blk = q % (HD / 16);
        const bool ok = r < rows;
        const int rr = ok ? r : 0;
        float lse[XSPLITS];
        float4 o[XSPLITS];
        float M = -INFINITY;
#pragma unroll
        for (int k = 0; k < XSPLITS; ++k) {
            const bool on = k < nsp;
            lse[k] = on ? ldcg_f32(p.xo_lse + static_cast<size_t>(h * XSPLITS + k) * MR + rr) : -INFINITY;
            o[k] = on ? __ldcg(reinterpret_cast<const float4*>(
                            p.xo_part + (static_cast<size_t>(h * XSPLITS + k) * MR + rr) * HD +
                            blk * 16 + sub * 4))
                      : make_float4(0.f, 0.f, 0.f, 0.f);
            M = fmaxf(M, lse[k]);
        }
        float W = 0.f;
        float4 O = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < XSPLITS; ++k) {
            if (k >= nsp) continue;
            const float wk = exp2f(lse[k] - M);
            W += wk;
            O.x += wk * o[k].x; O.y += wk * o[k].y; O.z += wk * o[k].z; O.w += wk * o[k].w;
        }
        const float inv_gs = 1.0f / ags;
        quad_quant_store(rb(O.x / W) * inv_gs, rb(O.y / W) * inv_gs, rb(O.z / W) * inv_gs,
                         rb(O.w / W) * inv_gs, p.attn_fp4 + static_cast<size_t>(rr) * (D / 2),
                         p.attn_sf + static_cast<size_t>(rr) * (D / 16), h * (HD / 16) + blk,
                         2 * sub, 2 * sub + 1, ok, sub == 0);
    }
}

// ── FFN up/gate (FP8) + SwiGLU -> NVFP4 down operand ───────────────────────
// The CTA owns 16-column groups blockIdx.x + i*grid; warp (i, j) computes
// n8 tile j of group i for up and gate over the full K, the SwiGLU values
// meet in shared memory and one thread per (group, row) quantizes the
// 16-element NVFP4 block.
__device__ void swiglu_phase(const ActionMkParams& p, const ActionMkLayer& L,
                             const ActionMkScales& sc, const __nv_bfloat16* mod,
                             uint8_t* smem) {
    constexpr int NKC = D / 64;
    constexpr int GROUPS = HID / 16;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, t0 = lane & 3, t1 = lane >> 2;
    const int rows = p.rows;
    const int nlocal = (GROUPS - static_cast<int>(blockIdx.x) + s_nc - 1) / s_nc;
    const int gi = warp >> 1, j = warp & 1;
    const bool active = gi < nlocal;
    const int tile = 2 * (blockIdx.x + gi * s_nc) + j;
    constexpr int HK = NKC / 2;
    uint8_t* sWu = smem + SM_W8 + warp * 2 * HK * 512;
    uint8_t* sWg = sWu + HK * 512;
    if (active) {
        fp8_stage<HK>(L.up_w, tile, 0, sWu);
        fp8_stage<HK>(L.gate_w, tile, 0, sWg);
    }
    mod_norm_fp8_smem(p.x, mod, mod + D, sc.ffn_qs, rows, p.eps, smem);
    float au[2][4] = {}, ag[2][4] = {};
    float2 gbf = make_float2(0.f, 0.f), ubf = make_float2(0.f, 0.f);
    const int col = tile * 8 + 2 * t0;
    if (active) {
        uint4 wu[HK], wg[HK];
        fp8_load<HK>(L.up_w, tile, HK, wu);
        fp8_load<HK>(L.gate_w, tile, HK, wg);
        const uint32_t gb2 = __ldg(reinterpret_cast<const unsigned int*>(L.gate_b + col));
        const uint32_t ub2 = __ldg(reinterpret_cast<const unsigned int*>(L.up_b + col));
        gbf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gb2));
        ubf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ub2));
        fp8_mma2<HK, true>(smem, 0, sWu, sWg, nullptr, nullptr, au, ag);
        fp8_mma2<HK, false>(smem, HK, nullptr, nullptr, wu, wg, au, ag);
    }
    csync();   // operand and staged weights are dead: the SwiGLU tile reuses them
    float* sact = reinterpret_cast<float*>(smem);   // [group][row][16]
    if (active) {
        const float inv_gs = 1.0f / sc.dn_ags;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float gv = rb(ag[mt][i] * sc.gate_alpha);
                float uv = rb(au[mt][i] * sc.up_alpha);
                gv = rb(gv + ((i & 1) ? gbf.y : gbf.x));
                uv = rb(uv + ((i & 1) ? ubf.y : ubf.x));
                const float silu = rb(gv / (1.0f + expf(-gv)));
                const int row = mt * 16 + t1 + 8 * (i >> 1);
                sact[(gi * MR + row) * 16 + j * 8 + 2 * t0 + (i & 1)] = rb(silu * uv) * inv_gs;
            }
    }
    csync();
    for (int t = tid; t < nlocal * rows; t += CT) {
        const int g = t / rows, row = t % rows;
        const int grp = blockIdx.x + g * s_nc;
        uint8_t* dst = p.act_fp4 + static_cast<size_t>(row) * (HID / 2) + grp * 8;
        uint8_t q[8];
        quant_block16(sact + (g * MR + row) * 16, q, p.act_sf + static_cast<size_t>(row) * (HID / 16) + grp);
        *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(q);
    }
}

__global__ void __launch_bounds__(THREADS, 1) action_mk_kernel(ActionMkParams p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1200)
    // block-scaled FP4 MMA and bulk L2 prefetch need sm_120a
#else
    extern __shared__ __align__(128) uint8_t smem[];
    if (threadIdx.x == 0) s_nc = static_cast<int>(gridDim.x) - p.prefetch;
    __syncthreads();
    if (static_cast<int>(blockIdx.x) >= s_nc) {
        // L2 producer CTA: stream layer li+1 once compute CTA 0 starts layer li
        if (threadIdx.x >= 32) return;
        const int j = blockIdx.x - s_nc;
        unsigned* prog = p.counters + 8 + j;
        for (int li = 0; li < p.num_layers; ++li) {
            unsigned v;
            do {
                asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(prog) : "memory");
            } while (static_cast<int>(v) < li + 1);
            prefetch_layer(li + 1 < p.num_layers ? &p.layers[li + 1] : p.next_first,
                           threadIdx.x, j, p.prefetch);
        }
        __syncwarp();
        if (threadIdx.x == 0) *prog = 0u;   // last reader of this launch
        return;
    }
    __shared__ __align__(8) uint64_t kv_bar;
    if (threadIdx.x == 0) mbar_init(&kv_bar);
    csync();
    uint32_t kv_phase = 0;
    const int rows = p.rows;
    // Counter value at entry lies in [k*s_nc, k*s_nc + s_nc - 1]: no CTA can
    // pass this launch's first barrier before every CTA has read it.
    unsigned long long* ctr = reinterpret_cast<unsigned long long*>(p.bar);
    unsigned long long gen = 0;
    if (threadIdx.x == 0) {
        unsigned long long v;
        asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(v) : "l"(ctr) : "memory");
        gen = v - v % static_cast<unsigned long long>(s_nc);
    }
    for (int li = 0; li < p.num_layers; ++li) {
        const ActionMkLayer& L = p.layers[li];
        const ActionMkScales sc = p.scales[li];
        const __nv_bfloat16* ms = p.mod_self + static_cast<size_t>(li) * 3 * D;
        const __nv_bfloat16* mc = p.mod_cross + static_cast<size_t>(li) * 3 * D;
        const __nv_bfloat16* mf = p.mod_ffn + static_cast<size_t>(li) * 3 * D;
        if (blockIdx.x == 0 && threadIdx.x < p.prefetch)
            asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(p.counters + 8 + threadIdx.x), "r"(li + 1) : "memory");

        // A: norm -> FP8, QKV GEMM
        norm_fp8_gemm_phase<1>(p.x, ms, sc.qkv_qs, p.eps, L.qkv_w, QKVN, sc.qkv_alpha, p.qkv,
                               rows, smem);
        grid_sync(ctr, gen);
        // B: self attention
        for (int u = blockIdx.x; u < NH * 2; u += s_nc) {
            self_attn_head(p, L, u >> 1, u & 1, sc.so_ags, smem);
            csync();
        }
        grid_sync(ctr, gen);
        // C: self out-projection + gate/residual
        skinny_phase<D, 3>(p.attn_fp4, p.attn_sf, sc.so_alpha, L.so_w, L.so_sf,
                                 L.so_b, ms + 2 * D, p.x, rows, smem);
        grid_sync(ctr, gen);
        // D+E: cross query projection + cross attention
        for (int u = blockIdx.x; u < NH * XSPLITS; u += s_nc) {
            cross_attn_unit(p, L, sc, mc, u / XSPLITS, u % XSPLITS, smem, &kv_bar, kv_phase);
            csync();
        }
        grid_sync(ctr, gen);
        // F: cross out-projection + gate/residual
        skinny_phase<D, 3>(p.attn_fp4, p.attn_sf, sc.co_alpha, L.co_w, L.co_sf,
                                 L.co_b, mc + 2 * D, p.x, rows, smem);
        grid_sync(ctr, gen);
        // G: norm -> FP8, up/gate + SwiGLU -> NVFP4
        swiglu_phase(p, L, sc, mf, smem);
        grid_sync(ctr, gen);
        // H: down projection + gate/residual
        skinny_phase<HID, 4>(p.act_fp4, p.act_sf, sc.dn_alpha, L.dn_w,
                                    L.dn_sf, L.dn_b, mf + 2 * D, p.x, rows, smem);
        grid_sync(ctr, gen);
    }
#endif
}

// [N, K] e4m3 -> [N/8][K/64][lane][16 B]: lane (t0, t1) of tile (nt, kc) holds
// W[nt*8+t1][kc*64 + q*16 + t0*4 + j] for q = 0..3, j = 0..3. With kn != 0
// the source is the transposed [K, N] layout (W[n][k] = src[k*N + n]).
__global__ void pack_fp8_kernel(const uint8_t* __restrict__ w, uint8_t* __restrict__ out, int N,
                                int K, int kn) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int kcs = K / 64;
    const size_t total = static_cast<size_t>(N / 8) * kcs * 32;
    if (g >= total) return;
    const int lane = static_cast<int>(g & 31);
    const size_t tile = g >> 5;
    const int kc = static_cast<int>(tile % kcs);
    const int nt = static_cast<int>(tile / kcs);
    const int t0 = lane & 3, t1 = lane >> 2;
    const int n = nt * 8 + t1;
    uint8_t b[16];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = kc * 64 + q * 16 + t0 * 4 + j;
            b[q * 4 + j] = kn ? w[static_cast<size_t>(k) * N + n] : w[static_cast<size_t>(n) * K + k];
        }
    *reinterpret_cast<uint4*>(out + g * 16) = *reinterpret_cast<const uint4*>(b);
}

}  // namespace

int action_mk_smem_bytes() { return SM_TOTAL; }

int action_mk_layers(const ActionMkParams& params, int grid, cudaStream_t stream) {
    // every phase's work split assumes at least 32 compute CTAs (one per
    // cross-attention (head, split) unit, <= 12 warps of tiles per CTA)
    if (params.rows < 1 || params.rows > MR || params.prefetch < 0 ||
        grid - params.prefetch < 32 || grid > 64)
        return 1;
    if (params.ctx_len < 1 || params.ctx_len > XSPLITS * XCHUNK ||
        params.ctx_rows < XSPLITS * XCHUNK)
        return 2;
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(action_mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 SM_TOTAL) != cudaSuccess)
            return 3;
        attr = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = SM_TOTAL;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeCooperative;
    at[0].val.cooperative = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, action_mk_kernel, params);
    return e == cudaSuccess ? 0 : static_cast<int>(e);
}

int action_mk_pack_fp8(const void* w, void* out, int N, int K, int kn, cudaStream_t stream) {
    if (N % 8 != 0 || K % 64 != 0) return 1;
    const size_t total = static_cast<size_t>(N / 8) * (K / 64) * 32;
    pack_fp8_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t*>(w), static_cast<uint8_t*>(out), N, K, kn);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace molmoact2
}  // namespace flash_rt
