// SPDX-License-Identifier: Apache-2.0
// The whole LaWAM flow-matching loop (all denoising steps of the DiT, the
// action encoder / decoder, the Euler updates and the unnormalization) as one
// persistent cooperative kernel on sm_120.
//
// The per-kernel chain (lawam_dit_sm120.cu + the skinny GEMMs) is bound by
// its ~140 dependent launches per step, not by DRAM or the tensor cores: every
// GEMM of the DiT works on 24 action rows. Here one CTA per SM walks a
// host-built op table (descriptor of the next op prefetched into shared
// memory); a grid barrier (one counter, release / acquire) separates dependent
// ops, and the step-invariant weights stream into a shared-memory ring
// (cp.async.bulk, mbarrier-tracked, L2 evict-first): the warp that consumed a
// unit refills its slot with the unit kSlots ahead, so the stream runs ahead of
// the eight MMA warps across barriers, attention and norms.
//
// GEMMs (out[m][n] = sum_k A[m][k] W[n][k], m < 24): each CTA owns a
// contiguous range of 16-column tiles (the ffn down projection: a range of a
// column group, over one of 4 k groups), the weight tile is the m16 operand
// and the 24 rows are three n8 operands of mma.sync m16n8k16. NVFP4 and INT8
// weight-only sites are packed at load time into the fragment order of this
// kernel: within each 32-wide k group lane quad t holds the 8 consecutive k
// [8t, 8t + 8), mma step s pairing (8t + 4s, +1) with the instruction's
// (2t, 2t + 1) and (8t + 4s + 2, +3) with (2t + 8, 2t + 9), so one 16-byte
// shared load feeds both steps of a group for the activations and the
// weights. E2M1 x UE4M3 and INT8 are exact in FP16; the MMAs accumulate in
// FP16 over one 256 (NVFP4) / 128 (INT8)-wide k unit, with exact power-of-two
// prescales keeping the sums in range, and in FP32 across units. The FP16 A
// operand of the CTA's k range is resident in shared memory: cp.async copies
// arriving on one mbarrier per 256-column chunk, so the MMAs start on the
// first chunk.
// Epilogues apply the weight scale, bias, residual, activation, Q pre-scaling
// or the Euler update with the rounding points of the per-kernel chain, and
// write the per-row (sum, sum of squares) of a residual update for the next
// LayerNorm.
//
// Block-input LayerNorms run once per op, distributed over the CTAs (into the
// FP16 operand of the following GEMM); after the K-split down projection the
// LayerNorm phase also sums its partials into the residual (two-pass
// statistics over the rebuilt rows).
//
// Attention: one CTA per (head, 16-row query tile); its warps take 16-key
// chunks round robin (online softmax, BF16 mma.sync, FP32 state) and merge
// through shared memory in warp order. K is read row major, V as V^T (the
// self-attention V^T is written by the QKV epilogue, the cross-attention V^T
// is transposed from the conditioning K / V in the kernel prologue).
#include "lawam_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace flash_rt {
namespace lawam {
namespace {

constexpr int kCW = 8;                   // warps (2 per SM sub-partition, 255 registers); each refills the ring slots it frees
constexpr int kConsumers = kCW * 32;
constexpr int kThreads = kConsumers;
constexpr int kRows = 24;                // action rows (three n8 operands)
constexpr int kChunk = 256;              // k per packed W4 unit / unit chunk
constexpr int kKMax = 1024;              // resident A columns per CTA
constexpr int kALd = kKMax + 32;         // staged row stride (16-bit elements): 2112 B, conflict-free LDS.128
constexpr int kUnit = 2304;              // ring slot / packed unit stride (bytes)
constexpr int kSlots = 20;
constexpr int kABytes = kRows * kALd * 2;         // 50688: A operand, then split partials / attention merge
constexpr int kMaxSites = 72;
constexpr int kOpVecs = sizeof(DitOp) / 16;   // 16-byte pieces of an op descriptor
constexpr int kRowPartOff = 8192;   // floats into the A region: per-tile row statistics (after the split partials)
constexpr int kSmem = kSlots * kUnit + kABytes + 256 + kSlots * 8 + (kKMax / kChunk) * 8 + kMaxSites * 16 +
                      2 * static_cast<int>(sizeof(DitOp));
static_assert(kSmem + 1024 <= 101376, "dynamic shared memory budget (sm_120: 99 KB per block)");

enum OpKind { OP_GEMM = 0, OP_ATTN = 1, OP_AE = 2, OP_KVT = 3, OP_UNNORM = 4, OP_LN = 5 };
enum EpMode { EP_RESID_INIT = 0, EP_RESID = 1, EP_GELU = 2, EP_RELU = 3, EP_QKV = 4, EP_Q = 5, EP_EULER = 6, EP_PARTIAL = 7 };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile(
        "{\n .reg .pred p;\n"
        "WAIT_%=:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra WAIT_%=;\n"
        "}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar, uint64_t policy) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], "
        "%4;\n" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;\n" ::"n"(kConsumers) : "memory"); }

// FP16 accumulation (full tensor rate on sm_120; used per 128 / 256-wide unit, FP32 across units)
__device__ __forceinline__ void mma_f16acc(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void add_h2acc(float* acc, const uint32_t* h) {
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&h[0]));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
    acc[0] += lo.x; acc[1] += lo.y; acc[2] += hi.x; acc[3] += hi.y;
}

// Two signed bytes (byte, byte + 1 of word ^ 0x80808080) -> FP16 pair of
// value / 128 (exact: 8 + u / 128 - 9, the 2^-7 compensated in the epilogue)
__device__ __forceinline__ uint32_t i8x2_f16x2(uint32_t biased, uint32_t sel) {
    uint32_t h;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(h) : "r"(biased), "r"(0x48484848u), "r"(sel));
    const __half2 v = __hsub2(*reinterpret_cast<const __half2*>(&h), __float2half2_rn(9.0f));
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float2 bf2f(uint32_t v) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
}

__device__ __forceinline__ uint32_t f2bf(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }

// BF16 pair -> FP16 pair (the W4 sites run FP16 operands; exact in the DiT's range)
__device__ __forceinline__ uint32_t bf2h(uint32_t v) {
    const __half2 h = __float22half2_rn(bf2f(v));
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t e2m1x2_scaled(uint32_t byte, uint32_t scale_h2) {
    uint32_t h2;
    asm("{\n .reg .b8 b;\n cvt.u8.u32 b, %1;\n cvt.rn.f16x2.e2m1x2 %0, b;\n}\n" : "=r"(h2) : "r"(byte));
    const __half2 v = __hmul2(*reinterpret_cast<const __half2*>(&h2), *reinterpret_cast<const __half2*>(&scale_h2));
    return *reinterpret_cast<const uint32_t*>(&v);
}

// UE4M3 byte -> FP16 pair (both halves)
__device__ __forceinline__ uint32_t ue4m3_h2(uint32_t byte) {
    uint32_t h2;
    const uint32_t pair = (byte & 0xffu) * 0x101u;
    asm("{\n .reg .b16 s;\n cvt.u16.u32 s, %1;\n cvt.rn.f16x2.e4m3x2 %0, s;\n}\n" : "=r"(h2) : "r"(pair));
    return h2;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float tanh_gelu(float g) {
    return __fdividef(g, 1.0f + __expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
}

__device__ __forceinline__ uint4 ldg16(const void* p) { return *reinterpret_cast<const uint4*>(p); }

// L2 loads for data other CTAs wrote in this launch (L1 is not coherent across SMs)
__device__ __forceinline__ uint4 ldcg16(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }
// Shared memory: ring | A operand (partials / attention merge) | stats | ring / A-chunk mbarriers | sites | op buffers.
constexpr int kOffA = kSlots * kUnit;
constexpr int kOffStats = kOffA + kABytes;
constexpr int kOffBars = kOffStats + 256;
constexpr int kAChunks = kKMax / kChunk;   // A operand chunks, each landing on its own mbarrier
constexpr int kOffSites = kOffBars + kSlots * 8 + kAChunks * 8;
constexpr int kOffOps = kOffSites + kMaxSites * 16;
extern __shared__ __align__(128) unsigned char g_dyn[];

// Fixed-offset views of the dynamic shared memory (no pointer registers held).
struct Smem {
    __device__ __forceinline__ unsigned char* ring() const { return g_dyn; }
    __device__ __forceinline__ __nv_bfloat16* abuf() const { return reinterpret_cast<__nv_bfloat16*>(g_dyn + kOffA); }
    __device__ __forceinline__ float* stats() const { return reinterpret_cast<float*>(g_dyn + kOffStats); }
    __device__ __forceinline__ uint64_t* full() const { return reinterpret_cast<uint64_t*>(g_dyn + kOffBars); }
    __device__ __forceinline__ uint64_t* achunk() const { return full() + kSlots; }
    __device__ __forceinline__ int4* sites() const { return reinterpret_cast<int4*>(g_dyn + kOffSites); }
    __device__ __forceinline__ DitOp* opbuf() const { return reinterpret_cast<DitOp*>(g_dyn + kOffOps); }
};

// The CTA's weight stream: its per-step units in consumption order, replayed every step. The warp
// that consumed ring unit r refills the slot with unit r + kSlots (lane 0, after the warp's reads).
struct Ring {
    const unsigned char* base;
    unsigned per, end;
    uint64_t policy;
    unsigned issued[kSlots];   // fill index (unit / kSlots) last issued into each slot
};
__shared__ Ring s_ring;

// L2 prefetch of the CTA's weight stream kPf units ahead of the ring: the DRAM reads of a
// GEMM phase start during the preceding norm / attention / barrier phases
constexpr unsigned kPf = 24;

__device__ __forceinline__ void prefetch_unit_l2(unsigned u) {
    if (u >= s_ring.end) return;
    const unsigned char* p = s_ring.base + static_cast<size_t>(u % s_ring.per) * kUnit;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(kUnit) : "memory");
}

__device__ __noinline__ void fill_unit(const Smem& sm, unsigned u) {
    const unsigned slot = u % kSlots;
    prefetch_unit_l2(u + kPf);
    mbar_expect_tx(&sm.full()[slot], kUnit);
    bulk_g2s(sm.ring() + slot * kUnit, s_ring.base + static_cast<size_t>(u % s_ring.per) * kUnit, kUnit,
             &sm.full()[slot], s_ring.policy);
    *reinterpret_cast<volatile unsigned*>(&s_ring.issued[slot]) = u / kSlots;
}

__device__ __forceinline__ void release_unit(const Smem& sm, unsigned r) {
    // every lane's reads of the slot are ordered before the async-proxy refill
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncwarp();
    if ((threadIdx.x & 31) == 0 && r + kSlots < s_ring.end) fill_unit(sm, r + kSlots);
}

// Wait for ring unit r. Its fill must have been issued (the slot's previous unit consumed) before
// the phase parity identifies it: a warp may reach unit r while that unit is still in flight.
__device__ __forceinline__ void wait_unit(const Smem& sm, unsigned r) {
    const unsigned slot = r % kSlots, k = r / kSlots;
    while (*reinterpret_cast<volatile unsigned*>(&s_ring.issued[slot]) < k) {
    }
    mbar_wait(&sm.full()[slot], k & 1);
}

// Tile range of a GEMM site for this CTA: first 16-column tile, tile count,
// first packed unit, first 256-wide k chunk.
struct Site {
    int tile0, tiles, unit0, kc0;
};

__device__ __forceinline__ Site site_s(const Smem& sm, int site) {
    const int4 e = sm.sites()[site];
    return Site{e.x, e.y, e.z, e.w};
}

__device__ __forceinline__ int units_per_chunk(int wtype) { return wtype == 0 ? 1 : 2; }

// ─────────────────────────────────────────────────────────────────────────
// grid barrier (all CTAs co-resident: cooperative launch)
// ─────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void grid_sync(unsigned* counter, unsigned target) {
    consumer_sync();
    if (threadIdx.x == 0) {
        // release (cumulative over the CTA's writes ordered by the bar.sync) / acquire
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(counter) : "memory");
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(counter) : "memory");
        } while (v < target);
    }
    consumer_sync();
}

// ─────────────────────────────────────────────────────────────────────────
// GEMM phase
// ─────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

// A = FP16 buffer [24, K] columns [col0, col0 + kc) into the resident operand (cp.async; the
// weight ring owns the bulk-copy engine): every consumer thread's copies of a 256-column chunk
// arrive on the chunk's mbarrier, so the MMAs of a chunk start as soon as it lands (wait_chunk).
__device__ __forceinline__ void stage_buf(const DitOp& op, const Smem& sm, int col0, int kc) {
    const int K = op.i[DIT_I_K];
    const __half* src = reinterpret_cast<const __half*>(op.p[DIT_P_ABUF]);
    constexpr int segs = kChunk / 8, total = kRows * segs;
    const int nch = kc / kChunk;
#pragma unroll 1
    for (int c = 0; c < nch; ++c) {
#pragma unroll 1
        for (int seg = threadIdx.x; seg < total; seg += kConsumers) {
            const int r = seg / segs, o = c * kChunk + (seg % segs) * 8;
            cp_async16(sm.abuf() + r * kALd + o, src + r * K + col0 + o);
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(&sm.achunk()[c]))
                     : "memory");
    }
}

__device__ __forceinline__ void wait_chunk(const Smem& sm, int c, unsigned parity) { mbar_wait(&sm.achunk()[c], parity); }

// Epilogue operands of one lane (issued before the MMA loop): weight scale,
// bias and residual / flow state of its 12 outputs.
struct EpiPre {
    float alpha[2];
    uint32_t bias[2];          // BF16 bits (FP32 bits for the Euler bias)
    uint32_t r[2][3][2];       // residual BF16 bits / flow state FP32 bits
};

// volatile loads: issued where written (before the MMA loop), not sunk to their use
__device__ __forceinline__ uint32_t vld_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t vld_u16(const void* p) {
    unsigned short v;
    asm volatile("ld.global.cg.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void epi_prefetch(const DitOp& op, int tile, EpiPre& pre) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ep = op.i[DIT_I_EP], N = op.i[DIT_I_N];
    const bool w8 = op.i[DIT_I_WTYPE] != 0;
    const float* ws = reinterpret_cast<const float*>(op.p[DIT_P_WS]);
    const float* gs = reinterpret_cast<const float*>(op.p[DIT_P_GS]);
    const unsigned short* b16 = reinterpret_cast<const unsigned short*>(op.p[DIT_P_BIAS]);
    const unsigned* b32 = reinterpret_cast<const unsigned*>(op.p[DIT_P_BIAS32]);
    const unsigned short* h16 = reinterpret_cast<const unsigned short*>(op.p[DIT_P_OUT0]);
    const unsigned* x32 = reinterpret_cast<const unsigned*>(op.p[DIT_P_OUT0]);
    const float pre_scale = w8 ? 128.0f : 4096.0f;   // the exact prescales of the FP16-accumulating MMAs
#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
        const int n = tile * 16 + g + 8 * hi;
        pre.alpha[hi] = __uint_as_float(vld_u32(w8 ? static_cast<const void*>(ws + n) : static_cast<const void*>(gs))) *
                        pre_scale;
        pre.bias[hi] = ep == EP_EULER ? vld_u32(b32 + n) : vld_u16(b16 + n);
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int m = 8 * j + 2 * t + e;
                pre.r[hi][j][e] = ep == EP_RESID ? vld_u16(h16 + m * N + n) : ep == EP_EULER ? vld_u32(x32 + m * N + n) : 0u;
            }
    }
}

__device__ __forceinline__ float bf_bits(uint32_t v) { return __uint_as_float(v << 16); }

// Epilogue of one lane's 12 outputs (cols n = tile * 16 + g (+8), rows m = 8j + 2t (+1)).
#define DIT_EPI_LOOP(...)                                                          \
    _Pragma("unroll") for (int idx = 0; idx < 12; ++idx) {                         \
        const int hi = idx / 6, j = (idx / 2) % 3, e = idx & 1;                    \
        if (!((jmask >> j) & 1)) continue;                                         \
        const int n = tile * 16 + g + 8 * hi, m = 8 * j + 2 * t + e;               \
        const float v = acc[j][2 * hi + e] * pre.alpha[hi];                        \
        const uint32_t rb = pre.r[hi][j][e], bb = pre.bias[hi];                    \
        (void)rb; (void)bb; (void)m; (void)n;                                      \
        __VA_ARGS__;                                                               \
    }
__device__ __forceinline__ void gemm_epilogue(const DitOp& op, int tile, const float (&acc)[3][4], const EpiPre& pre,
                                              float (&rowacc)[6][2], int jmask = 7) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int ep = op.i[DIT_I_EP], N = op.i[DIT_I_N];
    const float qscale = op.f[DIT_F_QSCALE], dt = op.f[DIT_F_DT];
    __nv_bfloat16* o16 = reinterpret_cast<__nv_bfloat16*>(op.p[DIT_P_OUT0]);
    __half* h16 = reinterpret_cast<__half*>(op.p[DIT_P_OUT0]);
    switch (ep) {
        case EP_RESID_INIT:
            DIT_EPI_LOOP({
                const float hs = bfr(bfr(v) + bf_bits(bb));
                o16[m * N + n] = __float2bfloat16(hs);
                rowacc[j * 2 + e][0] += hs;
                rowacc[j * 2 + e][1] += hs * hs;
            })
            break;
        case EP_RESID:
            DIT_EPI_LOOP({
                const float hs = bfr(bf_bits(rb) + bfr(bfr(v) + bf_bits(bb)));
                o16[m * N + n] = __float2bfloat16(hs);
                rowacc[j * 2 + e][0] += hs;
                rowacc[j * 2 + e][1] += hs * hs;
            })
            break;
        // GELU / ReLU outputs feed a GEMM only: stored as FP16 of the BF16-rounded value
        case EP_GELU: DIT_EPI_LOOP(h16[m * N + n] = __float2half_rn(bfr(tanh_gelu(bfr(bfr(v) + bf_bits(bb)))))) break;
        case EP_RELU: DIT_EPI_LOOP(h16[m * N + n] = __float2half_rn(bfr(fmaxf(bfr(bfr(v) + bf_bits(bb)), 0.f)))) break;
        case EP_QKV: {
            __nv_bfloat16* k16 = reinterpret_cast<__nv_bfloat16*>(op.p[DIT_P_OUT1]);
            __nv_bfloat16* vt16 = reinterpret_cast<__nv_bfloat16*>(op.p[DIT_P_OUT2]);
            DIT_EPI_LOOP({
                const int sec = n >> 10, c = n & 1023;
                const float y = v + bf_bits(bb);
                if (sec == 0)
                    o16[m * 1024 + c] = __float2bfloat16(y * qscale);
                else if (sec == 1)
                    k16[m * 1024 + c] = __float2bfloat16(y);
                else
                    vt16[c * 32 + m] = __float2bfloat16(y);
            })
        } break;
        case EP_Q: DIT_EPI_LOOP(o16[m * 1024 + n] = __float2bfloat16((v + bf_bits(bb)) * qscale)) break;
        default: {   // EP_EULER: x += dt * (sum + bias)
            float* x = reinterpret_cast<float*>(op.p[DIT_P_OUT0]);
            DIT_EPI_LOOP(x[m * N + n] = __uint_as_float(rb) + dt * (v + __uint_as_float(bb)))
        } break;
    }
}
#undef DIT_EPI_LOOP

// Row statistics of a residual update for the next LayerNorm: the lane's
// (sum, sum of squares) over its two columns, reduced over the tile's 16
// columns, summed over the CTA's tiles in tile order and written as this
// column owner's partial [owner][24][2] (the consumer sums the owners in order).
__device__ __forceinline__ void rowstat_tile(const float (&rowacc)[6][2], int tt, const Smem& sm) {
    const int lane = threadIdx.x & 31, t = lane & 3;
    float r[6][2];
#pragma unroll
    for (int i = 0; i < 6; ++i)
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            float v = rowacc[i][k];
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            r[i][k] = v;
        }
    if (lane < 4) {
        float* rp = reinterpret_cast<float*>(sm.abuf()) + kRowPartOff + tt * kRows * 2;
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int m = 8 * j + 2 * t + e;
                rp[2 * m] = r[j * 2 + e][0];
                rp[2 * m + 1] = r[j * 2 + e][1];
            }
    }
}

__device__ __forceinline__ void rowstat_cta(const DitOp& op, int T, int owner, const Smem& sm) {
    consumer_sync();
    if (threadIdx.x < kRows * 2) {
        const float* rp = reinterpret_cast<const float*>(sm.abuf()) + kRowPartOff;
        float v = 0.f;
        for (int tt = 0; tt < T; ++tt) v += rp[tt * kRows * 2 + threadIdx.x];
        reinterpret_cast<float*>(op.p[DIT_P_STATS])[owner * kRows * 2 + threadIdx.x] = v;
    }
}

// MMA over the 8 k32 groups of one NVFP4 unit (tile x 256 k); a points at
// the unit's FP16 A columns. FP16 accumulation over the unit (the block scales
// carry an exact 2^-12 so the sums stay in range; alpha restores it), added to
// the FP32 accumulator at the end.
__device__ __forceinline__ void mma_w4(const unsigned char* unit, const __nv_bfloat16* a, float (&acc)[3][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint4 sc_lo = *reinterpret_cast<const uint4*>(unit + 2048 + g * 16);
    const uint4 sc_hi = *reinterpret_cast<const uint4*>(unit + 2048 + (g + 8) * 16);
    const uint32_t slo[4] = {sc_lo.x, sc_lo.y, sc_lo.z, sc_lo.w};
    const uint32_t shi[4] = {sc_hi.x, sc_hi.y, sc_hi.z, sc_hi.w};
    const __half2 pre = __float2half2_rn(1.0f / 4096.0f);
    uint32_t hacc[3][2] = {{0u, 0u}, {0u, 0u}, {0u, 0u}};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint4 wq = *reinterpret_cast<const uint4*>(unit + q * 512 + lane * 16);
        const uint32_t wlo = q == 0 ? slo[0] : q == 1 ? slo[1] : q == 2 ? slo[2] : slo[3];
        const uint32_t whi = q == 0 ? shi[0] : q == 1 ? shi[1] : q == 2 ? shi[2] : shi[3];
#pragma unroll
        for (int hg = 0; hg < 2; ++hg) {
            const int G = 2 * q + hg;
            const int sh8 = 8 * (2 * hg + (t >> 1));    // block 2G + t / 2 = 4q + 2hg + t / 2
            uint32_t s0 = ue4m3_h2(wlo >> sh8), s1 = ue4m3_h2(whi >> sh8);
            const __half2 a0 = __hmul2(*reinterpret_cast<const __half2*>(&s0), pre);
            const __half2 a1 = __hmul2(*reinterpret_cast<const __half2*>(&s1), pre);
            s0 = *reinterpret_cast<const uint32_t*>(&a0);
            s1 = *reinterpret_cast<const uint32_t*>(&a1);
            uint4 bv[3];
#pragma unroll
            for (int j = 0; j < 3; ++j)
                bv[j] = *reinterpret_cast<const uint4*>(a + (8 * j + g) * kALd + 32 * G + 8 * t);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const uint32_t word = hg ? (s ? wq.w : wq.z) : (s ? wq.y : wq.x);
                uint32_t af[4];
                af[0] = e2m1x2_scaled(word & 0xffu, s0);
                af[1] = e2m1x2_scaled((word >> 8) & 0xffu, s1);
                af[2] = e2m1x2_scaled((word >> 16) & 0xffu, s0);
                af[3] = e2m1x2_scaled(word >> 24, s1);
#pragma unroll
                for (int j = 0; j < 3; ++j) mma_f16acc(hacc[j], af, s ? bv[j].z : bv[j].x, s ? bv[j].w : bv[j].y);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) add_h2acc(acc[j], hacc[j]);
}

// MMA over the 4 k32 groups of one INT8 unit (tile x 128 k): FP16 operands
// (weights / 128, exact; alpha restores the 128) and FP16 accumulation over the unit.
__device__ __forceinline__ void mma_w8(const unsigned char* unit, const __nv_bfloat16* a, float (&acc)[3][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t hacc[3][2] = {{0u, 0u}, {0u, 0u}, {0u, 0u}};
#pragma unroll
    for (int G = 0; G < 4; ++G) {
        const uint4 wv = *reinterpret_cast<const uint4*>(unit + G * 512 + lane * 16);
        uint4 bv[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) bv[j] = *reinterpret_cast<const uint4*>(a + (8 * j + g) * kALd + 32 * G + 8 * t);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            // a0 (bytes 0,1), a1 (2,3) of word w0; a2, a3 of w1
            const uint32_t b0 = (s ? wv.z : wv.x) ^ 0x80808080u, b1 = (s ? wv.w : wv.y) ^ 0x80808080u;
            uint32_t af[4];
            af[0] = i8x2_f16x2(b0, 0x4140u);
            af[1] = i8x2_f16x2(b0, 0x4342u);
            af[2] = i8x2_f16x2(b1, 0x4140u);
            af[3] = i8x2_f16x2(b1, 0x4342u);
#pragma unroll
            for (int j = 0; j < 3; ++j) mma_f16acc(hacc[j], af, s ? bv[j].z : bv[j].x, s ? bv[j].w : bv[j].y);
        }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) add_h2acc(acc[j], hacc[j]);
}

// Units (tile, k chunk) of a CTA's GEMM, u = tile * nk + chunk, are split
// over the kCW warps in contiguous runs (the first U % kCW warps take one more);
// the packed stream interleaves the runs round by round, so run position r of
// warp w is stream unit r * kCW + w. A warp accumulates over consecutive
// chunks of a tile and holds at most two tile segments (runs <= nk + 1).
__device__ __forceinline__ void warp_run(int U, int w, int& start, int& cnt) {
    const int q = U / kCW, a = U % kCW;
    cnt = q + (w < a ? 1 : 0);
    start = w * q + min(w, a);
}

// The warp whose run holds unit u (u < U).
__device__ __forceinline__ int warp_of(int U, int u) {
    const int q = U / kCW, a = U % kCW, big = a * (q + 1);
    return u < big ? u / (q + 1) : a + (u - big) / q;
}

__device__ void gemm_phase(const DitOp& op, const Smem& sm, unsigned& ucount, unsigned& astage) {
    const Site st = site_s(sm, op.i[DIT_I_SITE]);
    if (st.tiles == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wtype = op.i[DIT_I_WTYPE];
    const int upc = units_per_chunk(wtype);
    const int ksplit = op.i[DIT_I_KSPLIT];
    const int nk = op.i[DIT_I_K] / kChunk / ksplit;
    const int T = st.tiles, U = T * nk;

    stage_buf(op, sm, st.kc0 * kChunk, nk * kChunk);
    const unsigned apar = astage;   // phase bit of each chunk barrier
    astage ^= (1u << nk) - 1u;
    // epilogue tiles: tile `warp`; a tile kCW (T = kCW + 1) is split by row block over the
    // last three warps (the shortest MMA runs). Operands land during the MMAs (A chunk by chunk).
    EpiPre pre, pre2;
    const bool extra = T > kCW && warp >= kCW - 3;
    if (warp < T) epi_prefetch(op, st.tile0 + warp, pre);
    if (extra) epi_prefetch(op, st.tile0 + kCW, pre2);

    int start, cnt;
    warp_run(U, warp, start, cnt);
    const int q = U / kCW;
    float acc[3][4], seg0[3][4];
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = seg0[j][e] = 0.f;
    bool split_run = false;
    for (int r = 0; r < cnt; ++r) {
        const int u = start + r, c = u % nk;
        if (r > 0 && c == 0) {   // the run crosses into the next tile
            split_run = true;
#pragma unroll
            for (int j = 0; j < 3; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    seg0[j][e] = acc[j][e];
                    acc[j][e] = 0.f;
                }
        }
        const unsigned si = static_cast<unsigned>(r < q ? r * kCW + warp : q * kCW + warp);
        const __nv_bfloat16* a = sm.abuf() + c * kChunk;
        wait_chunk(sm, c, (apar >> c) & 1u);
        if (wtype == 0) {
            const unsigned ri = ucount + si;
            wait_unit(sm, ri);
            mma_w4(sm.ring() + (ri % kSlots) * kUnit, a, acc);
            release_unit(sm, ri);
        } else {
#pragma unroll 1
            for (int half = 0; half < 2; ++half) {
                const unsigned ri = ucount + 2 * si + half;
                wait_unit(sm, ri);
                mma_w8(sm.ring() + (ri % kSlots) * kUnit, a + half * 128, acc);
                release_unit(sm, ri);
            }
        }
    }
    ucount += U * upc;

    // run segments -> shared (aliases A): slot 2w = first tile of warp w's run, 2w + 1 = second
    float* red = reinterpret_cast<float*>(sm.abuf());
    consumer_sync();
    if (cnt > 0) {
        float* d = red + ((2 * warp) * 32 + lane) * 12;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float* v = split_run ? seg0[j] : acc[j];
            *reinterpret_cast<float4*>(d + 4 * j) = make_float4(v[0], v[1], v[2], v[3]);
        }
        if (split_run) {
            d += 32 * 12;
#pragma unroll
            for (int j = 0; j < 3; ++j)
                *reinterpret_cast<float4*>(d + 4 * j) = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        }
    }
    consumer_sync();
    // tile tt: segments of the runs that cover its chunks [tt * nk, tt * nk + nk), in warp order
    // (a run that starts inside the tile holds it as its first segment, else as its second)
    auto tile_sum = [&](int tt, float (&sum)[3][4]) {
#pragma unroll
        for (int j = 0; j < 3; ++j) sum[j][0] = sum[j][1] = sum[j][2] = sum[j][3] = 0.f;
        const int u0 = tt * nk;
        const int wl = warp_of(U, u0 + nk - 1);
        for (int w = warp_of(U, u0); w <= wl; ++w) {
            int s0, n0;
            warp_run(U, w, s0, n0);
            const float* sp = red + ((2 * w + (s0 >= u0 ? 0 : 1)) * 32 + lane) * 12;
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const float4 v = *reinterpret_cast<const float4*>(sp + 4 * j);
                sum[j][0] += v.x; sum[j][1] += v.y; sum[j][2] += v.z; sum[j][3] += v.w;
            }
        }
    };
    const int ep = op.i[DIT_I_EP];
    if (ep == EP_PARTIAL) {
        // K split over ksplit CTAs of a column group (CTA c: k group c % ksplit):
        // FP32 partials [k group][24][N]; the next LayerNorm phase sums them in
        // k-group order into the residual.
        const int N = op.i[DIT_I_N], kg = blockIdx.x % ksplit;
        const int g = lane >> 2, t = lane & 3;
        float* part = reinterpret_cast<float*>(op.p[DIT_P_PART]);
#pragma unroll 1
        for (int tt = warp; tt < T; tt += kCW) {
            float sum[3][4];
            tile_sum(tt, sum);
            const int tile = st.tile0 + tt;
#pragma unroll
            for (int hi = 0; hi < 2; ++hi)
#pragma unroll
                for (int j = 0; j < 3; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                        part[(kg * kRows + 8 * j + 2 * t + e) * N + tile * 16 + g + 8 * hi] = sum[j][2 * hi + e];
        }
        return;
    }
    const bool rstat = ep == EP_RESID || ep == EP_RESID_INIT;
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        const int tt = pass == 0 ? warp : kCW;
        if (pass == 0 ? warp >= T : !extra) continue;
        const int jmask = pass == 0 ? 7 : 1 << (warp - (kCW - 3));
        float sum[3][4];
        tile_sum(tt, sum);
        float rowacc[6][2] = {};
        gemm_epilogue(op, st.tile0 + tt, sum, pass == 0 ? pre : pre2, rowacc, jmask);
        if (rstat) rowstat_tile(rowacc, tt, sm);
    }
    if (rstat) rowstat_cta(op, T, blockIdx.x, sm);
}

// ─────────────────────────────────────────────────────────────────────────
// attention phase: CTA = (head, 16-row query tile), warps over 32-key chunks
// ─────────────────────────────────────────────────────────────────────────
struct AttnMerge {
    float o[kCW][16][64];
    float m[kCW][16];
    float l[kCW][16];
};
static_assert(sizeof(AttnMerge) <= kABytes, "attention merge aliases the A operand");
static_assert(kCW >= 3, "the extra epilogue tile is split over three warps");

// One CTA per (head, 16-row query tile); warps take 16-key chunks round robin
// (online softmax, BF16 mma.sync, FP32 state) and merge in warp order. Within
// a chunk the n8 tile n, column c of S is key kb + 4 (c / 2) + (c % 2) + 2n,
// so the P fragment pairs (2t, 2t+1) / (2t+8, 2t+9) are the keys kb + 4t + {0,1}
// / {2,3}: one 8-byte V^T load per lane and dim tile.
__device__ __noinline__ void attn_phase(const DitOp& op, const Smem& sm) {
    const int pair = blockIdx.x;
    if (pair >= 32) return;
    const int h = pair >> 1, q0 = (pair & 1) * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int len = op.i[DIT_I_KVLEN], kld = op.i[DIT_I_KLD], vtld = op.i[DIT_I_VTLD];
    const __nv_bfloat16* Q = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_Q]);
    const __nv_bfloat16* Kp = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_K]) + h * 64;
    const __nv_bfloat16* Vt = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_VT]) + static_cast<size_t>(h) * 64 * vtld;
    AttnMerge& mg = *reinterpret_cast<AttnMerge*>(sm.abuf());

    // Q fragments (pre-scaled BF16), rows q0 + g / q0 + g + 8 (< 24)
    uint32_t qa[4][4];
    {
        uint4 qr[2][2];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int row = q0 + g + 8 * rr;
#pragma unroll
            for (int G = 0; G < 2; ++G)
                qr[rr][G] = row < kRows ? ldcg16(Q + row * 1024 + h * 64 + 32 * G + 8 * t) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int G = ks >> 1, s = ks & 1;
            qa[ks][0] = s ? qr[0][G].z : qr[0][G].x;
            qa[ks][1] = s ? qr[1][G].z : qr[1][G].x;
            qa[ks][2] = s ? qr[0][G].w : qr[0][G].y;
            qa[ks][3] = s ? qr[1][G].w : qr[1][G].y;
        }
    }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;

    const int nchunks = (len + 15) / 16;
    for (int c = warp; c < nchunks; c += kCW) {
        const int kb = c * 16;
        uint4 kr[2][2];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int key = min(kb + 4 * (g >> 1) + (g & 1) + 2 * n, len - 1);
#pragma unroll
            for (int G = 0; G < 2; ++G) kr[n][G] = ldcg16(Kp + static_cast<size_t>(key) * kld + 32 * G + 8 * t);
        }
        uint2 vr[8];
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
            vr[nd] = __ldcg(reinterpret_cast<const uint2*>(Vt + static_cast<size_t>(nd * 8 + g) * vtld + kb + 4 * t));
        float s[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int G = ks >> 1, ss = ks & 1;
                mma_bf16(s[n], qa[ks], ss ? kr[n][G].z : kr[n][G].x, ss ? kr[n][G].w : kr[n][G].y);
            }
        }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (kb + 4 * t + (j & 1) + 2 * n >= len) s[n][j] = -INFINITY;
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
        for (int n = 0; n < 8; ++n) {
            o[n][0] *= corr[0]; o[n][1] *= corr[0];
            o[n][2] *= corr[1]; o[n][3] *= corr[1];
        }
        uint32_t pa[4];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const float p0 = exp2f(s[n][0] - m[0]), p1 = exp2f(s[n][1] - m[0]);
            const float p2 = exp2f(s[n][2] - m[1]), p3 = exp2f(s[n][3] - m[1]);
            l[0] += p0 + p1;
            l[1] += p2 + p3;
            pa[n * 2 + 0] = f2bf(p0, p1);
            pa[n * 2 + 1] = f2bf(p2, p3);
        }
#pragma unroll
        for (int nd = 0; nd < 8; ++nd) mma_bf16(o[nd], pa, vr[nd].x, vr[nd].y);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        *reinterpret_cast<float2*>(&mg.o[warp][g][n * 8 + t * 2]) = make_float2(o[n][0], o[n][1]);
        *reinterpret_cast<float2*>(&mg.o[warp][g + 8][n * 8 + t * 2]) = make_float2(o[n][2], o[n][3]);
    }
    if (t == 0) {
        mg.m[warp][g] = m[0]; mg.l[warp][g] = l[0];
        mg.m[warp][g + 8] = m[1]; mg.l[warp][g + 8] = l[1];
    }
    consumer_sync();
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(op.p[DIT_P_OUT0]);
    for (int i = threadIdx.x; i < 16 * 32; i += kConsumers) {
        const int r = i >> 5, c = (i & 31) * 2;
        if (q0 + r >= kRows) continue;
        float M = -INFINITY;
        for (int k = 0; k < kCW; ++k) M = fmaxf(M, mg.m[k][r]);
        float L = 0.f, a = 0.f, b = 0.f;
        for (int k = 0; k < kCW; ++k) {
            const float f = mg.l[k][r] > 0.f ? exp2f(mg.m[k][r] - M) : 0.f;
            L += f * mg.l[k][r];
            a += f * mg.o[k][r][c];
            b += f * mg.o[k][r][c + 1];
        }
        const float inv = 1.f / L;
        // the o projection's operand: FP16 of the BF16-rounded output
        const __half2 o2 = __floats2half2_rn(bfr(a * inv), bfr(b * inv));
        *reinterpret_cast<__half2*>(out + (q0 + r) * 1024 + h * 64 + c) = o2;
    }
    consumer_sync();   // merge buffers alias the A operand of the next op
}

// ─────────────────────────────────────────────────────────────────────────
// block-input LayerNorm, once per op: CTA c normalizes a contiguous range of
// the 3072 eight-column segments of the BF16 [24, 1024] residual into the FP16
// GEMM operand xa [24, 1024] (statistics: the column owners' partials)
// ─────────────────────────────────────────────────────────────────────────
// With a pending K-split residual update (i[KSPLIT] = k groups): the CTA
// first rebuilds the full new residual rows of its range in shared memory,
// h' = bf16(h + bf16(bf16(alpha * sum_k part[k]) + bias)), two-pass statistics
// over them, and writes its own segments of h' into the other residual buffer
// (p[OUT2]; the residual stream alternates between two buffers).
__device__ void ln_resid_phase(const DitOp& op, const Smem& sm, int ncta) {
    constexpr int kSegs = kRows * 128;
    const int per = (kSegs + ncta - 1) / ncta;
    const int s0 = blockIdx.x * per, s1 = min(kSegs, s0 + per);
    if (s0 >= s1) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ks = op.i[DIT_I_KSPLIT];
    // old residual rows (read whole, also the segments other CTAs update in this phase) and the
    // buffer receiving the new residual: never the same, so the rebuilt rows do not race
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_OUT1]);
    __nv_bfloat16* h_new = reinterpret_cast<__nv_bfloat16*>(op.p[DIT_P_OUT2]);
    const float* part = reinterpret_cast<const float*>(op.p[DIT_P_PART]);
    // NVFP4 site: global scale, times the 2^12 prescale of the FP16-accumulating MMA
    const float alpha = *reinterpret_cast<const float*>(op.p[DIT_P_GS]) * 4096.0f;
    const __nv_bfloat16* rbias = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_BIAS]);
    const __nv_bfloat16* scale = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_LNSC]);
    const __nv_bfloat16* shift = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_LNSH]);
    const __nv_bfloat16* pos = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_POS]);
    const int r0 = s0 >> 7, nrows = ((s1 - 1) >> 7) - r0 + 1;   // <= 2
    float* rows = reinterpret_cast<float*>(sm.abuf());             // [nrows][1024] new residual
    // the step-invariant norm operands of this thread's output segment, in flight from the start
    const int seg = s0 + threadIdx.x;
    const bool mine = seg < s1;
    const int row = seg >> 7, col = (seg & 127) * 8;
    uint4 sc = make_uint4(0, 0, 0, 0), sh = sc, pv = sc;
    if (mine) {
        if (scale != nullptr) {
            sc = ldg16(scale + col);
            sh = ldg16(shift + col);
        }
        if (pos != nullptr) pv = ldg16(pos + row * 1024 + col);
    }
    // new residual of the whole rows: thread -> 4 consecutive columns of each row (all loads issued first)
    static_assert(kConsumers == 256, "one 4-column item per thread and row");
    constexpr int kMaxSplit = 4;   // k groups of a K-split site (pipeline_rtx.DIT_KSPLIT_FC2)
    float4 pa[2][kMaxSplit];
    uint2 hb[2], bb;
    const int c = threadIdx.x * 4;
    bb = __ldg(reinterpret_cast<const uint2*>(rbias + c));
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (rr >= nrows) break;
        const int r = r0 + rr;
#pragma unroll
        for (int k = 0; k < kMaxSplit; ++k)
            if (k < ks) pa[rr][k] = __ldcg(reinterpret_cast<const float4*>(part + (k * kRows + r) * 1024 + c));
        hb[rr] = __ldcg(reinterpret_cast<const uint2*>(h + r * 1024 + c));
    }
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (rr >= nrows) break;
        float4 acc = pa[rr][0];
#pragma unroll
        for (int k = 1; k < kMaxSplit; ++k) {
            if (k >= ks) break;
            acc.x += pa[rr][k].x; acc.y += pa[rr][k].y; acc.z += pa[rr][k].z; acc.w += pa[rr][k].w;
        }
        const float2 h0 = bf2f(hb[rr].x), h1 = bf2f(hb[rr].y), b0 = bf2f(bb.x), b1 = bf2f(bb.y);
        float* d = rows + rr * 1024 + c;
        d[0] = bfr(h0.x + bfr(bfr(acc.x * alpha) + b0.x));
        d[1] = bfr(h0.y + bfr(bfr(acc.y * alpha) + b0.y));
        d[2] = bfr(h1.x + bfr(bfr(acc.z * alpha) + b1.x));
        d[3] = bfr(h1.y + bfr(bfr(acc.w * alpha) + b1.y));
    }
    consumer_sync();
    // two-pass statistics, one warp per row
    if (warp < nrows) {
        const float* rp = rows + warp * 1024;
        float sum = 0.f;
        for (int c = lane; c < 1024; c += 32) sum += rp[c];
        const float mean = warp_sum(sum) * (1.0f / 1024.0f);
        float d = 0.f;
        for (int c = lane; c < 1024; c += 32) d += (rp[c] - mean) * (rp[c] - mean);
        const float inv = rsqrtf(warp_sum(d) * (1.0f / 1024.0f) + op.f[DIT_F_EPS]);
        if (lane == 0) {
            sm.stats()[2 * warp] = mean;
            sm.stats()[2 * warp + 1] = inv;
        }
    }
    consumer_sync();
    if (!mine) return;
    const int rr = row - r0;
    const float mean = sm.stats()[2 * rr], inv = sm.stats()[2 * rr + 1];
    const float* rp = rows + rr * 1024 + col;
    const uint32_t scw[4] = {sc.x, sc.y, sc.z, sc.w}, shw[4] = {sh.x, sh.y, sh.z, sh.w};
    const uint32_t psw[4] = {pv.x, pv.y, pv.z, pv.w};
    uint32_t o[4], hn[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float v0 = rp[2 * e], v1 = rp[2 * e + 1];
        hn[e] = f2bf(v0, v1);
        float n0 = (v0 - mean) * inv, n1 = (v1 - mean) * inv;
        if (scale != nullptr) {
            const float2 a = bf2f(scw[e]), b = bf2f(shw[e]);
            n0 = n0 * (1.0f + a.x) + b.x;
            n1 = n1 * (1.0f + a.y) + b.y;
        }
        if (pos != nullptr) {
            const float2 p = bf2f(psw[e]);
            n0 = bfr(n0) + p.x;
            n1 = bfr(n1) + p.y;
        }
        o[e] = bf2h(f2bf(n0, n1));
    }
    *reinterpret_cast<uint4*>(h_new + row * 1024 + col) = make_uint4(hn[0], hn[1], hn[2], hn[3]);
    *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(op.p[DIT_P_OUT0]) + row * 1024 + col) =
        make_uint4(o[0], o[1], o[2], o[3]);
}

__device__ void ln_phase(const DitOp& op, const Smem& sm, int ncta) {
    if (op.i[DIT_I_KSPLIT] > 1) {
        ln_resid_phase(op, sm, ncta);
        return;
    }
    constexpr int kSegs = kRows * 128;
    const int per = (kSegs + ncta - 1) / ncta;
    const int s0 = blockIdx.x * per, s1 = min(kSegs, s0 + per);
    if (s0 >= s1) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_ABUF]);
    const __nv_bfloat16* scale = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_LNSC]);
    const __nv_bfloat16* shift = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_LNSH]);
    const __nv_bfloat16* pos = reinterpret_cast<const __nv_bfloat16*>(op.p[DIT_P_POS]);
    const float* rstats = reinterpret_cast<const float*>(op.p[DIT_P_STATS]);
    const int owners = op.i[DIT_I_OWNERS];
    const int seg = s0 + threadIdx.x;
    const bool mine = seg < s1;
    const int row = mine ? seg >> 7 : 0, col = mine ? (seg & 127) * 8 : 0;
    uint4 hv = make_uint4(0, 0, 0, 0), sc = hv, sh = hv, pv = hv;
    if (mine) {
        hv = ldcg16(h + row * 1024 + col);
        if (scale != nullptr) {
            sc = ldg16(scale + col);
            sh = ldg16(shift + col);
        }
        if (pos != nullptr) pv = ldg16(pos + row * 1024 + col);
    }
    // statistics of the (at most kCW) rows of this range, one warp per row
    const int r0 = s0 >> 7, nrows = ((s1 - 1) >> 7) - r0 + 1;
    if (warp < nrows) {
        const int r = r0 + warp;
        float2 a = __ldcg(reinterpret_cast<const float2*>(rstats) + min(lane, owners - 1) * kRows + r);
        float2 b = __ldcg(reinterpret_cast<const float2*>(rstats) + min(lane + 32, owners - 1) * kRows + r);
        const float m0 = lane < owners ? 1.f : 0.f, m1 = lane + 32 < owners ? 1.f : 0.f;
        const float mean = warp_sum(a.x * m0 + b.x * m1) * (1.0f / 1024.0f);
        const float var = fmaxf(warp_sum(a.y * m0 + b.y * m1) * (1.0f / 1024.0f) - mean * mean, 0.f);
        if (lane == 0) {
            sm.stats()[2 * warp] = mean;
            sm.stats()[2 * warp + 1] = rsqrtf(var + op.f[DIT_F_EPS]);
        }
    }
    consumer_sync();
    if (!mine) return;
    const float mean = sm.stats()[2 * (row - r0)], inv = sm.stats()[2 * (row - r0) + 1];
    const uint32_t w[4] = {hv.x, hv.y, hv.z, hv.w}, scw[4] = {sc.x, sc.y, sc.z, sc.w};
    const uint32_t shw[4] = {sh.x, sh.y, sh.z, sh.w}, psw[4] = {pv.x, pv.y, pv.z, pv.w};
    uint32_t o[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 v = bf2f(w[e]);
        float n0 = (v.x - mean) * inv, n1 = (v.y - mean) * inv;
        if (scale != nullptr) {
            const float2 a = bf2f(scw[e]), b = bf2f(shw[e]);
            n0 = n0 * (1.0f + a.x) + b.x;
            n1 = n1 * (1.0f + a.y) + b.y;
        }
        if (pos != nullptr) {
            // the modulated norm is rounded to BF16 before the positional add
            const float2 p = bf2f(psw[e]);
            n0 = bfr(n0) + p.x;
            n1 = bfr(n1) + p.y;
        }
        o[e] = bf2h(f2bf(n0, n1));
    }
    *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(op.p[DIT_P_OUT0]) + row * 1024 + col) =
        make_uint4(o[0], o[1], o[2], o[3]);
}

// ─────────────────────────────────────────────────────────────────────────
// small ops
// ─────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

template <int N>
__device__ __forceinline__ float ae_dot(const float* xr, const float* w1, int j, int n, float acc) {
    if (N > 0) {
#pragma unroll
        for (int k = 0; k < N; ++k) acc += xr[k] * w1[k * 1024 + j];
    } else {
        for (int k = 0; k < n; ++k) acc += xr[k] * w1[k * 1024 + j];
    }
    return acc;
}

__device__ void ae_phase(const DitOp& op, const Smem& sm, int ncta) {
    const float* x = reinterpret_cast<const float*>(op.p[0]);
    const float* w1 = reinterpret_cast<const float*>(op.p[1]);
    const float* b1 = reinterpret_cast<const float*>(op.p[2]);
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(op.p[3]);
    const int rows = op.i[DIT_I_N], n = op.i[DIT_I_K], hdim = 1024;
    // x (other CTAs' Euler update, L2) staged once per CTA (aliases the A operand, free between ops)
    float* xs = reinterpret_cast<float*>(sm.abuf());
    for (int i = threadIdx.x; i < rows * n; i += kConsumers) xs[i] = __ldcg(x + i);
    consumer_sync();
    for (int i = blockIdx.x * kConsumers + threadIdx.x; i < rows * hdim; i += ncta * kConsumers) {
        const int r = i / hdim, j = i % hdim;
        const float acc = n == 32 ? ae_dot<32>(xs + r * n, w1, j, n, b1[j]) : ae_dot<0>(xs + r * n, w1, j, n, b1[j]);
        reinterpret_cast<__half*>(out)[i] = __float2half_rn(bfr(silu(acc)));   // FP16 operand of the BF16-rounded value
    }
}

// V^T[blk][d][s] = V[s][blk * 2048 + 1024 + d] for the cross-attention K / V
// (zero padded keys untouched): every warp transposes 32 keys x 64 dims tiles
// through its own shared-memory tile with 16-byte coalesced loads (8 lanes per
// key row) and stores (4 lanes per 32-key dim row).
__device__ void kvt_phase(const DitOp& op, const Smem& sm, int ncta) {
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(op.p[0]);
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(op.p[1]);
    const int S = op.i[DIT_I_KVLEN], spad = op.i[DIT_I_VTLD], ld = op.i[DIT_I_KLD];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int kLd = 64 + 8;                          // [32 keys][64 dims + pad]
    __nv_bfloat16* tile = sm.abuf() + warp * 32 * kLd;
    const int key_tiles = (S + 31) / 32;
    const int total = 4 * 16 * key_tiles;                // blocks x 64-dim tiles x 32-key tiles
    for (int tI = blockIdx.x * kCW + warp; tI < total; tI += ncta * kCW) {
        const int blk = tI / (16 * key_tiles), rem = tI % (16 * key_tiles);
        const int dt = rem / key_tiles, kt = rem % key_tiles;
        // load: lane -> (key = 4 i + lane / 8, dims (lane % 8) * 8 .. +8)
        uint4 v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int key = kt * 32 + 4 * i + (lane >> 3);
            v[i] = key < S ? ldcg16(src + static_cast<size_t>(key) * ld + blk * 2048 + 1024 + dt * 64 + (lane & 7) * 8)
                           : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
            *reinterpret_cast<uint4*>(tile + (4 * i + (lane >> 3)) * kLd + (lane & 7) * 8) = v[i];
        __syncwarp();
        // store: lane -> (dim = 8 i + lane / 4, keys (lane % 4) * 8 .. +8)
        const int nk = min(32, S - kt * 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = 8 * i + (lane >> 2), k0 = (lane & 3) * 8;
            __align__(16) __nv_bfloat16 e[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) e[j] = tile[(k0 + j) * kLd + d];
            __nv_bfloat16* p = dst + (static_cast<size_t>(blk) * 1024 + dt * 64 + d) * spad + kt * 32 + k0;
            if (k0 + 8 <= nk) {
                *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(e);
            } else {
                for (int j = 0; j < 8 && k0 + j < nk; ++j) p[j] = e[j];
            }
        }
        __syncwarp();
    }
}

__global__ void __launch_bounds__(kThreads, 1)
dit_flow_kernel(const DitOp* __restrict__ ops, int n_ops, const int* __restrict__ site_tab, int n_sites,
                const int* __restrict__ cta_units, int n_steps, const unsigned char* __restrict__ stream,
                long long cta_stride, unsigned* __restrict__ counter) {
    const Smem sm;
    const int ncta = gridDim.x;
    for (int i = threadIdx.x; i < n_sites; i += kThreads)
        sm.sites()[i] = *reinterpret_cast<const int4*>(site_tab + (static_cast<size_t>(i) * ncta + blockIdx.x) * 4);
    if (threadIdx.x < kOpVecs)
        reinterpret_cast<uint4*>(sm.opbuf())[threadIdx.x] = reinterpret_cast<const uint4*>(ops)[threadIdx.x];

    if (threadIdx.x == 0) {
        for (int i = 0; i < kSlots; ++i) mbar_init(&sm.full()[i], 1);
        for (int i = 0; i < kAChunks; ++i) mbar_init(&sm.achunk()[i], kConsumers);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        uint64_t policy;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
        s_ring.base = stream + cta_stride * blockIdx.x;
        s_ring.per = static_cast<unsigned>(cta_units[blockIdx.x]);
        s_ring.end = s_ring.per * static_cast<unsigned>(n_steps);
        s_ring.policy = policy;
        for (unsigned u = kSlots; u < kPf; ++u) prefetch_unit_l2(u);
        for (unsigned u = 0; u < kSlots; ++u) {
            s_ring.issued[u] = 0;
            if (u < s_ring.end) fill_unit(sm, u);
        }
    }
    __syncthreads();

    unsigned ucount = 0, bars = 0, astage = 0;
    for (int oi = 0; oi < n_ops; ++oi) {
        const DitOp& op = sm.opbuf()[oi & 1];
        // the next descriptor lands in the other buffer during this op
        if (oi + 1 < n_ops && threadIdx.x < kOpVecs)
            cp_async16(reinterpret_cast<uint4*>(&sm.opbuf()[(oi + 1) & 1]) + threadIdx.x,
                       reinterpret_cast<const uint4*>(ops + oi + 1) + threadIdx.x);
        switch (op.i[DIT_I_KIND]) {
            case OP_GEMM: gemm_phase(op, sm, ucount, astage); break;
            case OP_ATTN: attn_phase(op, sm); break;
            case OP_AE: ae_phase(op, sm, ncta); break;
            case OP_KVT: kvt_phase(op, sm, ncta); break;
            case OP_LN: ln_phase(op, sm, ncta); break;
            default:
                if (blockIdx.x == 0) {
                    const float* x = reinterpret_cast<const float*>(op.p[0]);
                    const float* lo = reinterpret_cast<const float*>(op.p[1]);
                    const float* hi = reinterpret_cast<const float*>(op.p[2]);
                    float* out = reinterpret_cast<float*>(op.p[3]);
                    const int rows = op.i[DIT_I_N], ld = op.i[DIT_I_K], dim = op.i[DIT_I_KVLEN];
                    for (int i = threadIdx.x; i < rows * dim; i += kConsumers) {
                        const int r = i / dim, c = i % dim;
                        out[i] = (__ldcg(x + r * ld + c) + 1.0f) * 0.5f * (hi[c] - lo[c]) + lo[c];
                    }
                }
                break;
        }
        if (threadIdx.x < kOpVecs) cp_async_wait_all();
        const bool bar = op.i[DIT_I_BAR] != 0;
        if (bar)
            grid_sync(counter, static_cast<unsigned>(ncta) * ++bars);
        else
            consumer_sync();
    }
}

}  // namespace

int dit_flow_smem_bytes() { return kSmem; }

int dit_flow_persistent(const DitOp* ops, int n_ops, const int* site_tab, int n_sites, const int* cta_units,
                        int n_steps, const uint8_t* stream, long long cta_stride, unsigned* counter, int num_ctas,
                        cudaStream_t st, bool reset_counter) {
    if (ops == nullptr || site_tab == nullptr || stream == nullptr || counter == nullptr || cta_units == nullptr || n_ops < 1 || n_steps < 1 ||
        num_ctas < 32 || num_ctas > 64 || num_ctas % 4 != 0 || cta_stride <= 0 ||
        cta_stride % 16 != 0 || n_sites < 1 || n_sites > kMaxSites)
        return static_cast<int>(cudaErrorInvalidValue);
    static bool opted = false;
    if (!opted) {
        const cudaError_t e =
            cudaFuncSetAttribute(dit_flow_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        if (e != cudaSuccess) return static_cast<int>(e);
        opted = true;
    }
    int device = 0, blocks_per_sm = 0;
    cudaDeviceProp prop;
    cudaError_t residency = cudaGetDevice(&device);
    if (residency != cudaSuccess) return static_cast<int>(residency);
    residency = cudaGetDeviceProperties(&prop, device);
    if (residency != cudaSuccess) return static_cast<int>(residency);
    residency = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, dit_flow_kernel, kThreads, kSmem);
    if (residency != cudaSuccess) return static_cast<int>(residency);
    if (num_ctas > blocks_per_sm * prop.multiProcessorCount)
        return static_cast<int>(cudaErrorCooperativeLaunchTooLarge);
    // the grid barrier counter starts at 0 (the caller may clear it earlier in the graph instead)
    cudaError_t e = reset_counter ? cudaMemsetAsync(counter, 0, sizeof(unsigned), st) : cudaSuccess;
    if (e != cudaSuccess) return static_cast<int>(e);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(num_ctas);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return static_cast<int>(
        cudaLaunchKernelEx(&cfg, dit_flow_kernel, ops, n_ops, site_tab, n_sites, cta_units, n_steps, stream, cta_stride,
                           counter));
}

}  // namespace lawam
}  // namespace flash_rt
