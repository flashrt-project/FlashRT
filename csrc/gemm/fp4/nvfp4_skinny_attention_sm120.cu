// SPDX-License-Identifier: Apache-2.0
// Small-row attention over skinny-GEMM partials. See the header.
//
// One CTA per (head, 16-row query tile). The CTA stages the head's whole K
// (row-major) and V (transposed) in shared memory as BF16 -- before the grid
// dependency wait for static external K / V -- then the four warps split the
// keys, each running an online softmax over 32-key chunks on mma.sync
// m16n8k16 (BF16 operands, FP32 accumulation), and the per-warp (max, sum, O)
// states merge through shared memory.
#include "nvfp4_skinny_attention_sm120.cuh"
#include "nvfp4_skinny_common.cuh"

#include <algorithm>
#include <cstdint>

namespace flash_rt {
namespace nvfp4_skinny {
namespace {

constexpr int kWarps = 4;
constexpr int kTileQ = 16;
constexpr int kChunk = 32;   // keys per warp iteration

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Dynamic shared memory: K [kv_pad][HD + 8] then V^T [HD][kv_pad + 8] (BF16),
// later reused for the merge buffers.
template <int HD>
__host__ __device__ inline size_t stage_bytes(int kv_pad) {
    return (static_cast<size_t>(kv_pad) * (HD + 8) + static_cast<size_t>(HD) * (kv_pad + 8)) * 2;
}

template <int HD>
struct Merge {
    float o[kWarps][kTileQ][HD + 4];
    float m[kWarps][kTileQ];
    float l[kWarps][kTileQ];
};

// The attention body. Inputs are read for head h at column h * HD of a
// [rows, n_proj] partials buffer whose Q / K / V blocks are `width` apart (the
// bias is indexed the same way); results are written for head h_out of a
// heads * HD wide output. The standalone kernel passes h = h_out and
// width = heads * HD; the fused projection kernel below passes its own
// shared-memory copy of one head (h = 0, width = HD).
template <int HD, bool PDL>
__device__ __forceinline__ void attention_core(
        unsigned char* dyn, const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
        const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias, int n_proj,
        const __nv_bfloat16* __restrict__ k_ext, const __nv_bfloat16* __restrict__ v_ext,
        int kv_ld, int kv_seq, bool kv_static, __nv_bfloat16* __restrict__ out, int rows,
        int heads, float scale_log2, uint8_t* __restrict__ out4, uint8_t* __restrict__ out_sf,
        int h, int width, int h_out, int q0) {
    constexpr int KS = HD / 16;   // k-steps of Q K^T
    constexpr int NT = HD / 8;    // n-tiles of P V
    const int kv_pad = (kv_seq + kChunk - 1) / kChunk * kChunk;
    const int kst = HD + 8, vst = kv_pad + 8;   // row strides (elements)
    __nv_bfloat16* ks_ = reinterpret_cast<__nv_bfloat16*>(dyn);
    __nv_bfloat16* vt_ = ks_ + static_cast<size_t>(kv_pad) * kst;
    Merge<HD>& mg = *reinterpret_cast<Merge<HD>*>(dyn);

    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;   // fragment row group / thread in group
    const bool self = k_ext == nullptr;
    const int width_out = heads * HD;
    const size_t elems = static_cast<size_t>(rows) * n_proj;
    // this warp's key range, chunk-aligned so every chunk read stays inside kv_pad
    const int per = ((kv_seq + kWarps - 1) / kWarps + kChunk - 1) / kChunk * kChunk;
    const int kb = w * per, ke = min(kv_seq, kb + per);
    float alpha = 0.f;

    auto put = [&](int key, int c, const float* kv, const float* vv, int n) {
#pragma unroll
        for (int j = 0; j < n; j += 2)
            *reinterpret_cast<uint32_t*>(ks_ + static_cast<size_t>(key) * kst + c + j) = pack_bf16(kv[j], kv[j + 1]);
#pragma unroll
        for (int j = 0; j < n; ++j) vt_[static_cast<size_t>(c + j) * vst + key] = __float2bfloat16(vv[j]);
    };
    // Stage all keys of head h (rows past kv_seq zeroed up to kv_pad). Loads go
    // out in batches of kBatch units per thread so each batch costs one memory
    // round trip (per split) instead of one per unit.
    constexpr int kBatch = 8;
    auto stage = [&]() {
        if (self) {
            constexpr int U = HD / 4;   // float4 units per key
            const int total = kv_pad * U;
            for (int b0 = 0; b0 < total; b0 += kBatch * blockDim.x) {
                float4 ka[kBatch], va[kBatch];
                size_t ko[kBatch];
                bool ok[kBatch];
#pragma unroll
                for (int u = 0; u < kBatch; ++u) {
                    const int i = b0 + u * blockDim.x + threadIdx.x;
                    const int key = i / U, c = (i % U) * 4;
                    ok[u] = i < total && key < kv_seq;
                    ko[u] = static_cast<size_t>(key) * n_proj + width + h * HD + c;
                    ka[u] = va[u] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
                for (int sp = 0; sp < splits; ++sp) {
#pragma unroll
                    for (int u = 0; u < kBatch; ++u) {
                        if (!ok[u]) continue;
                        const float4 x = *reinterpret_cast<const float4*>(partials + sp * elems + ko[u]);
                        const float4 y = *reinterpret_cast<const float4*>(partials + sp * elems + ko[u] + width);
                        ka[u].x += x.x; ka[u].y += x.y; ka[u].z += x.z; ka[u].w += x.w;
                        va[u].x += y.x; va[u].y += y.y; va[u].z += y.z; va[u].w += y.w;
                    }
                }
#pragma unroll
                for (int u = 0; u < kBatch; ++u) {
                    const int i = b0 + u * blockDim.x + threadIdx.x;
                    if (i >= total) continue;
                    const int key = i / U, c = (i % U) * 4;
                    float kv[4] = {0.f, 0.f, 0.f, 0.f}, vv[4] = {0.f, 0.f, 0.f, 0.f};
                    if (ok[u]) {
                        const int kc = width + h * HD + c;
                        const float2 kb0 = bf2(*reinterpret_cast<const uint32_t*>(bias + kc));
                        const float2 kb1 = bf2(*reinterpret_cast<const uint32_t*>(bias + kc + 2));
                        const float2 vb0 = bf2(*reinterpret_cast<const uint32_t*>(bias + kc + width));
                        const float2 vb1 = bf2(*reinterpret_cast<const uint32_t*>(bias + kc + width + 2));
                        kv[0] = ka[u].x * alpha + kb0.x; kv[1] = ka[u].y * alpha + kb0.y;
                        kv[2] = ka[u].z * alpha + kb1.x; kv[3] = ka[u].w * alpha + kb1.y;
                        vv[0] = va[u].x * alpha + vb0.x; vv[1] = va[u].y * alpha + vb0.y;
                        vv[2] = va[u].z * alpha + vb1.x; vv[3] = va[u].w * alpha + vb1.y;
                    }
                    put(key, c, kv, vv, 4);
                }
            }
        } else {
            constexpr int U = HD / 8;   // 16-byte units per key
            const int total = kv_pad * U;
            for (int b0 = 0; b0 < total; b0 += kBatch * blockDim.x) {
                uint4 kr[kBatch], vr[kBatch];
#pragma unroll
                for (int u = 0; u < kBatch; ++u) {
                    const int i = b0 + u * blockDim.x + threadIdx.x;
                    const int key = i / U, c = (i % U) * 8;
                    kr[u] = vr[u] = make_uint4(0u, 0u, 0u, 0u);
                    if (i < total && key < kv_seq) {
                        const size_t off = static_cast<size_t>(key) * kv_ld + h_out * HD + c;
                        kr[u] = *reinterpret_cast<const uint4*>(k_ext + off);
                        vr[u] = *reinterpret_cast<const uint4*>(v_ext + off);
                    }
                }
#pragma unroll
                for (int u = 0; u < kBatch; ++u) {
                    const int i = b0 + u * blockDim.x + threadIdx.x;
                    if (i >= total) continue;
                    const int key = i / U, c = (i % U) * 8;
                    *reinterpret_cast<uint4*>(ks_ + static_cast<size_t>(key) * kst + c) = kr[u];
                    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vr[u]);
#pragma unroll
                    for (int j = 0; j < 8; ++j) vt_[static_cast<size_t>(c + j) * vst + key] = vb[j];
                }
            }
        }
    };

    const bool pre = !self && kv_static;
    if (pre) stage();
    if constexpr (PDL) pdl_prologue();
    alpha = (*a_gs) * (*w_gs);
    if (!pre) stage();

    // Q fragments (pre-scaled by softmax scale * log2 e), rows q0 + g and q0 + g + 8;
    // all 4 * KS pairs of a split are loaded together.
    float2 qf[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
#pragma unroll
        for (int j = 0; j < 4; ++j) qf[ks][j] = make_float2(0.f, 0.f);
    for (int sp = 0; sp < splits; ++sp) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {   // j = half * 2 + rr: cols +8*half, rows +8*rr
                const int row = q0 + g + (j & 1) * 8;
                const int col = h * HD + ks * 16 + (j >> 1) * 8 + t * 2;
                if (row < rows) {
                    const float2 v = *reinterpret_cast<const float2*>(partials + sp * elems +
                                                                       static_cast<size_t>(row) * n_proj + col);
                    qf[ks][j].x += v.x;
                    qf[ks][j].y += v.y;
                }
            }
        }
    }
    uint32_t qa[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = q0 + g + (j & 1) * 8;
            const int col = h * HD + ks * 16 + (j >> 1) * 8 + t * 2;
            float2 v = make_float2(0.f, 0.f);
            if (row < rows) {
                const float2 bv = bf2(*reinterpret_cast<const uint32_t*>(bias + col));
                v = make_float2(qf[ks][j].x * alpha + bv.x, qf[ks][j].y * alpha + bv.y);
            }
            qa[ks][j] = pack_bf16(v.x * scale_log2, v.y * scale_log2);
        }
    }

    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;

    __syncthreads();
    for (int base = kb; base < ke; base += kChunk) {
        float s[kChunk / 8][4];
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
                const __nv_bfloat16* kr = ks_ + static_cast<size_t>(base + n * 8 + g) * kst + ks * 16 + t * 2;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + 8);
                mma_bf16(s[n], qa[ks], b0, b1);
            }
        }
        // mask keys past the warp range, row max over the quad
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = base + n * 8 + t * 2 + (j & 1);
                if (key >= ke) s[n][j] = -INFINITY;
                mx[j >> 1] = fmaxf(mx[j >> 1], s[n][j]);
            }
        }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            const float mn = fmaxf(m[r], mx[r]);
            corr[r] = exp2f(m[r] - mn);   // m = -inf on the first chunk -> 0
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
            // C fragments of key tiles 2k / 2k+1 form the A fragment of P-step k
            pa[n >> 1][(n & 1) * 2 + 0] = pack_bf16(p0, p1);
            pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(p2, p3);
        }
#pragma unroll
        for (int kk = 0; kk < kChunk / 16; ++kk) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const __nv_bfloat16* vr = vt_ + static_cast<size_t>(n * 8 + g) * vst + base + kk * 16 + t * 2;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(vr);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(vr + 8);
                mma_bf16(o[n], pa[kk], b0, b1);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }

    __syncthreads();   // staged K / V -> merge buffers
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        *reinterpret_cast<float2*>(&mg.o[w][g][n * 8 + t * 2]) = make_float2(o[n][0], o[n][1]);
        *reinterpret_cast<float2*>(&mg.o[w][g + 8][n * 8 + t * 2]) = make_float2(o[n][2], o[n][3]);
    }
    if (t == 0) {
        mg.m[w][g] = m[0]; mg.l[w][g] = l[0];
        mg.m[w][g + 8] = m[1]; mg.l[w][g + 8] = l[1];
    }
    __syncthreads();
    if (out4 != nullptr) {
        // NVFP4 output for the next skinny GEMM: the BF16-rounded values that
        // gemm_bf16_act would read, quantized with the same per-16 block code
        // (unit activation global scale), row-major e2m1 + UE4M3.
        for (int u = threadIdx.x; u < kTileQ * (HD / 16); u += blockDim.x) {
            const int r = u / (HD / 16), blk = u % (HD / 16);
            if (q0 + r >= rows) continue;
            float M = -INFINITY;
#pragma unroll
            for (int k = 0; k < kWarps; ++k) M = fmaxf(M, mg.m[k][r]);
            float f[kWarps];
            float L = 0.f;
#pragma unroll
            for (int k = 0; k < kWarps; ++k) {
                f[k] = mg.l[k][r] > 0.f ? exp2f(mg.m[k][r] - M) : 0.f;
                L += f[k] * mg.l[k][r];
            }
            const float inv = 1.f / L;
            float v[16];
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                const int c = blk * 16 + j;
                float a = 0.f, b = 0.f;
#pragma unroll
                for (int k = 0; k < kWarps; ++k) {
                    a += f[k] * mg.o[k][r][c];
                    b += f[k] * mg.o[k][r][c + 1];
                }
                const float2 ab = bf2(pack_bf16(a * inv, b * inv));
                v[j] = ab.x;
                v[j + 1] = ab.y;
            }
            const size_t row = static_cast<size_t>(q0 + r);
            quant_block16(v, out4 + row * (width_out / 2) + (h_out * HD + blk * 16) / 2,
                          out_sf + row * (width_out / 16) + (h_out * HD) / 16 + blk);
        }
        return;
    }
    for (int i = threadIdx.x; i < kTileQ * (HD / 2); i += blockDim.x) {
        const int r = i / (HD / 2), c = (i % (HD / 2)) * 2;
        if (q0 + r >= rows) continue;
        float M = -INFINITY;
#pragma unroll
        for (int k = 0; k < kWarps; ++k) M = fmaxf(M, mg.m[k][r]);
        float L = 0.f, a = 0.f, b = 0.f;
#pragma unroll
        for (int k = 0; k < kWarps; ++k) {
            const float f = mg.l[k][r] > 0.f ? exp2f(mg.m[k][r] - M) : 0.f;
            L += f * mg.l[k][r];
            a += f * mg.o[k][r][c];
            b += f * mg.o[k][r][c + 1];
        }
        const float inv = 1.f / L;
        *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(q0 + r) * width_out + h_out * HD + c) =
            pack_bf16(a * inv, b * inv);
    }
}

template <int HD, bool PDL>
__global__ void __launch_bounds__(kWarps * 32)
partials_attention_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                          const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias, int n_proj,
                          const __nv_bfloat16* __restrict__ k_ext, const __nv_bfloat16* __restrict__ v_ext,
                          int kv_ld, int kv_seq, bool kv_static, __nv_bfloat16* __restrict__ out, int rows,
                          int heads, float scale_log2, uint8_t* __restrict__ out4,
                          uint8_t* __restrict__ out_sf) {
    extern __shared__ __align__(16) unsigned char dyn[];
    attention_core<HD, PDL>(dyn, partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq,
                            kv_static, out, rows, heads, scale_log2, out4, out_sf, blockIdx.x, heads * HD,
                            blockIdx.x, blockIdx.y * kTileQ);
}

// ── Self-attention fused with its head's Q/K/V projection ──────────────────
// One CTA per (head, 16-row query tile): the CTA runs the skinny W4A4 GEMM for
// its head's 3 x HD projection columns over the whole K (same fragment-major
// weights, same per-k64 MMA order as the single-split gemm_kernel, so the fp32
// sums equal that GEMM's partials), keeps them in shared memory and runs the
// attention body on them. The partials round trip through global memory and
// one launch per block go away; every value matches gemm -> partials_attention.
constexpr int kFusedK = 768;                  // DiT-B hidden
#ifndef FLASHRT_FUSED_PF
#define FLASHRT_FUSED_PF 4
#endif
constexpr int kFusedPF = FLASHRT_FUSED_PF;    // weight chunks in flight

template <int HD>
struct FusedSmem {
    static constexpr int ROWS = 48;           // 3 x 16-row MMA tiles (M <= 48)
    static constexpr int AB = kFusedK / 2, ASTR = AB + 16;
    static constexpr int SB = kFusedK / 16, SSTR = SB + 4;
    static constexpr int NP = 3 * HD;         // q | k | v of one head
    static constexpr size_t attn = (sizeof(Merge<HD>) > (size_t(64) * (HD + 8) + size_t(HD) * (64 + 8)) * 2)
                                       ? sizeof(Merge<HD>) : (size_t(64) * (HD + 8) + size_t(HD) * (64 + 8)) * 2;
    static constexpr size_t pb_off = (attn + 15) / 16 * 16;
    static constexpr size_t bias_off = pb_off + size_t(ROWS) * NP * 4;
    static constexpr size_t a_off = bias_off + size_t(NP) * 2;
    static constexpr size_t s_off = a_off + size_t(ROWS) * ASTR;
    static constexpr size_t bytes = s_off + size_t(ROWS) * SSTR;
};

template <int HD>
__global__ void __launch_bounds__(kWarps * 32)
qkv_self_attention_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ SFA,
                          const uint8_t* __restrict__ W, const uint8_t* __restrict__ SFB,
                          const float* __restrict__ a_gs, const float* __restrict__ w_gs,
                          const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
                          int rows, int heads, float scale_log2, uint8_t* __restrict__ out4,
                          uint8_t* __restrict__ out_sf) {
    using L = FusedSmem<HD>;
    constexpr int NTL = L::NP / 8;            // local n8 tiles (3 * HD / 8)
    constexpr int NTW = NTL / kWarps;         // per warp
    constexpr int KCH = kFusedK / 128;
    static_assert(NTL % kWarps == 0, "projection tiles must split across the warps");
    extern __shared__ __align__(16) unsigned char dyn[];
    float* pb = reinterpret_cast<float*>(dyn + L::pb_off);
    __nv_bfloat16* sbias = reinterpret_cast<__nv_bfloat16*>(dyn + L::bias_off);
    uint8_t* sA = dyn + L::a_off;
    uint8_t* sS = dyn + L::s_off;
    const int h = blockIdx.x, q0 = blockIdx.y * kTileQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, t0 = lane & 3, t1 = lane >> 2;
    const int width = heads * HD;
    const int kchunks = kFusedK / 128;
    // local tile -> global n8 tile of the packed [3 * width, K] projection
    auto gtile = [&](int lt) {
        const int part = lt / (HD / 8), i = lt % (HD / 8);
        return (part * width + h * HD) / 8 + i;
    };
    // the head's bias slice (constant) and the first weight chunk before the wait
    for (int i = tid; i < L::NP; i += blockDim.x) {
        const int part = i / HD, c = i % HD;
        sbias[i] = bias[part * width + h * HD + c];
    }
    uint4 wv[kFusedPF][NTW];
    uint2 sv[kFusedPF][NTW];
#pragma unroll
    for (int c = 0; c < kFusedPF - 1; ++c)
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const size_t tile = static_cast<size_t>(gtile(warp * NTW + t)) * kchunks + c;
            wv[c][t] = ldg_stream(W + (tile * 32 + lane) * 16);
            sv[c][t] = ldg_stream8(SFB + (tile * 8 + t1) * 8);
        }
    pdl_prologue();
    for (int c = tid; c < L::ROWS * (L::AB / 16); c += blockDim.x) {
        const int row = c / (L::AB / 16);
        const int off = (c % (L::AB / 16)) * 16;
        const bool v = row < rows;
        cp_async_16(sA + row * L::ASTR + off, A + static_cast<size_t>(v ? row : 0) * (kFusedK / 2) + off, v);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    for (int c = tid; c < L::ROWS * (L::SB / 4); c += blockDim.x) {
        const int row = c / (L::SB / 4);
        const int off = (c % (L::SB / 4)) * 4;
        uint32_t sv4 = 0;
        if (row < rows) sv4 = *reinterpret_cast<const uint32_t*>(SFA + static_cast<size_t>(row) * L::SB + off);
        *reinterpret_cast<uint32_t*>(sS + row * L::SSTR + off) = sv4;
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    float acc[3][NTW][4];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt)
#pragma unroll
        for (int t = 0; t < NTW; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][t][i] = 0.f;
    const int sau = (lane & 1) * 8 + (lane >> 2);
#pragma unroll
    for (int c = 0; c < KCH; ++c) {
        const int cur = c % kFusedPF, nc = c + kFusedPF - 1;
        if (nc < KCH) {              // chunk c + PF - 1 in flight during this one
#pragma unroll
            for (int t = 0; t < NTW; ++t) {
                const size_t tile = static_cast<size_t>(gtile(warp * NTW + t)) * kchunks + nc;
                wv[nc % kFusedPF][t] = ldg_stream(W + (tile * 32 + lane) * 16);
                sv[nc % kFusedPF][t] = ldg_stream8(SFB + (tile * 8 + t1) * 8);
            }
        }
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int mt = 0; mt < 3; ++mt) {
                const uint8_t* ra = sA + (mt * 16 + t1) * L::ASTR + c * 64 + s * 32 + t0 * 4;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ra);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ra + 8 * L::ASTR);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ra + 16);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ra + 8 * L::ASTR + 16);
                const uint32_t sfa = *reinterpret_cast<const uint32_t*>(sS + (mt * 16 + sau) * L::SSTR + c * 8 + s * 4);
#pragma unroll
                for (int t = 0; t < NTW; ++t) {
                    const uint32_t b0 = s == 0 ? wv[cur][t].x : wv[cur][t].z;
                    const uint32_t b1 = s == 0 ? wv[cur][t].y : wv[cur][t].w;
                    const uint32_t sfb = s == 0 ? sv[cur][t].x : sv[cur][t].y;
                    mma_nvf4(acc[mt][t], a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 3; ++mt)
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const int col = (warp * NTW + t) * 8 + 2 * t0;
            const int row0 = mt * 16 + t1;
            *reinterpret_cast<float2*>(pb + row0 * L::NP + col) = make_float2(acc[mt][t][0], acc[mt][t][1]);
            *reinterpret_cast<float2*>(pb + (row0 + 8) * L::NP + col) = make_float2(acc[mt][t][2], acc[mt][t][3]);
        }
    __syncthreads();
    attention_core<HD, false>(dyn, pb, 1, a_gs, w_gs, sbias, L::NP, nullptr, nullptr, 0, rows, false, out, rows,
                              heads, scale_log2, out4, out_sf, 0, HD, h, q0);
}

template <int HD, bool PDL>
int launch_one(const float* partials, int splits, const float* a_gs, const float* w_gs, const __nv_bfloat16* bias,
               int n_proj, const __nv_bfloat16* k_ext, const __nv_bfloat16* v_ext, int kv_ld, int kv_seq,
               bool kv_static, __nv_bfloat16* out, int rows, int heads, float scale_log2, uint8_t* out4,
               uint8_t* out_sf, cudaStream_t stream) {
    const int kv_pad = (kv_seq + kChunk - 1) / kChunk * kChunk;
    const size_t smem = std::max(stage_bytes<HD>(kv_pad), sizeof(Merge<HD>));
    const void* fn = (const void*)partials_attention_kernel<HD, PDL>;
    static int opted = 0;   // bytes the function attribute currently allows
    if (smem > 48 * 1024 && static_cast<int>(smem) > opted) {
        const cudaError_t e =
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (e != cudaSuccess) return static_cast<int>(e);
        opted = static_cast<int>(smem);
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(heads, (rows + kTileQ - 1) / kTileQ);
    cfg.blockDim = dim3(kWarps * 32);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = PDL ? 1 : 0;
    return static_cast<int>(cudaLaunchKernelEx(&cfg, partials_attention_kernel<HD, PDL>, partials, splits, a_gs,
                                               w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq, kv_static, out, rows,
                                               heads, scale_log2, out4, out_sf));
}

template <int HD>
int launch(bool pdl, const float* partials, int splits, const float* a_gs, const float* w_gs,
           const __nv_bfloat16* bias, int n_proj, const __nv_bfloat16* k_ext, const __nv_bfloat16* v_ext, int kv_ld,
           int kv_seq, bool kv_static, __nv_bfloat16* out, int rows, int heads, float scale_log2,
           uint8_t* out4, uint8_t* out_sf, cudaStream_t stream) {
    return pdl ? launch_one<HD, true>(partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq,
                                      kv_static, out, rows, heads, scale_log2, out4, out_sf, stream)
               : launch_one<HD, false>(partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq,
                                       kv_static, out, rows, heads, scale_log2, out4, out_sf, stream);
}

// Cross attention fused with its Q projection: one CTA per (head, 16-row q
// tile) projects only its own rows and head (MT = 1). The projection keeps the
// K-split structure of the standalone GEMM config (SPL splits of K / SPL, each
// summed from zero in the same chunk order) so the attention body sees the same
// per-split partials, summed in the same order. K/V come from the external
// conditioning cache.
template <int HD, int SPL>
__global__ void __launch_bounds__(kWarps * 32)
q_cross_attention_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ SFA,
                         const uint8_t* __restrict__ W, const uint8_t* __restrict__ SFB,
                         const float* __restrict__ a_gs, const float* __restrict__ w_gs,
                         const __nv_bfloat16* __restrict__ bias, const __nv_bfloat16* __restrict__ k_ext,
                         const __nv_bfloat16* __restrict__ v_ext, int kv_ld, int kv_seq,
                         __nv_bfloat16* __restrict__ out, int rows, int heads, float scale_log2,
                         uint8_t* __restrict__ out4, uint8_t* __restrict__ out_sf, int pb_off) {
    constexpr int AB = kFusedK / 2, ASTR = AB + 16;
    constexpr int SB = kFusedK / 16, SSTR = SB + 4;
    constexpr int NTW = HD / 8 / kWarps;      // n8 tiles per warp
    constexpr int KCH = kFusedK / 128;
    constexpr int CPS = KCH / SPL;            // 128-chunks per split
    static_assert(KCH % SPL == 0 && (HD / 8) % kWarps == 0, "bad split / tile shape");
    extern __shared__ __align__(16) unsigned char dyn[];
    float* pb = reinterpret_cast<float*>(dyn + pb_off);                      // [SPL][rows][HD]
    __nv_bfloat16* sbias = reinterpret_cast<__nv_bfloat16*>(pb + SPL * rows * HD);
    uint8_t* sA = reinterpret_cast<uint8_t*>(sbias + HD);
    uint8_t* sS = sA + 16 * ASTR;
    const int h = blockIdx.x, q0 = blockIdx.y * kTileQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, t0 = lane & 3, t1 = lane >> 2;
    for (int i = tid; i < HD; i += blockDim.x) sbias[i] = bias[h * HD + i];
    // the whole K of this warp's 2 tiles goes out before the dependency wait
    uint4 wv[KCH][NTW];
    uint2 sv[KCH][NTW];
#pragma unroll
    for (int c = 0; c < KCH; ++c)
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const size_t tile = static_cast<size_t>(h * HD / 8 + warp * NTW + t) * KCH + c;
            wv[c][t] = ldg_stream(W + (tile * 32 + lane) * 16);
            sv[c][t] = ldg_stream8(SFB + (tile * 8 + t1) * 8);
        }
    pdl_prologue();
    for (int c = tid; c < 16 * (AB / 16); c += blockDim.x) {
        const int row = c / (AB / 16);
        const int off = (c % (AB / 16)) * 16;
        const bool v = q0 + row < rows;
        cp_async_16(sA + row * ASTR + off, A + static_cast<size_t>(v ? q0 + row : 0) * AB + off, v);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    for (int c = tid; c < 16 * (SB / 4); c += blockDim.x) {
        const int row = c / (SB / 4);
        const int off = (c % (SB / 4)) * 4;
        uint32_t sv4 = 0;
        if (q0 + row < rows)
            sv4 = *reinterpret_cast<const uint32_t*>(SFA + static_cast<size_t>(q0 + row) * SB + off);
        *reinterpret_cast<uint32_t*>(sS + row * SSTR + off) = sv4;
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    float acc[SPL][NTW][4];
#pragma unroll
    for (int sp = 0; sp < SPL; ++sp)
#pragma unroll
        for (int t = 0; t < NTW; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[sp][t][i] = 0.f;
    const int sau = (lane & 1) * 8 + (lane >> 2);
#pragma unroll
    for (int c = 0; c < KCH; ++c) {
        const int cur = c, sp = c / CPS;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const uint8_t* ra = sA + t1 * ASTR + c * 64 + s * 32 + t0 * 4;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ra);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ra + 8 * ASTR);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ra + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ra + 8 * ASTR + 16);
            const uint32_t sfa = *reinterpret_cast<const uint32_t*>(sS + sau * SSTR + c * 8 + s * 4);
#pragma unroll
            for (int t = 0; t < NTW; ++t) {
                const uint32_t b0 = s == 0 ? wv[cur][t].x : wv[cur][t].z;
                const uint32_t b1 = s == 0 ? wv[cur][t].y : wv[cur][t].w;
                const uint32_t sfb = s == 0 ? sv[cur][t].x : sv[cur][t].y;
                mma_nvf4(acc[sp][t], a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }
    }
#pragma unroll
    for (int sp = 0; sp < SPL; ++sp)
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const int col = (warp * NTW + t) * 8 + 2 * t0;
            const int row0 = q0 + t1;
            float* base = pb + static_cast<size_t>(sp) * rows * HD;
            if (row0 < rows)
                *reinterpret_cast<float2*>(base + row0 * HD + col) = make_float2(acc[sp][t][0], acc[sp][t][1]);
            if (row0 + 8 < rows)
                *reinterpret_cast<float2*>(base + (row0 + 8) * HD + col) = make_float2(acc[sp][t][2], acc[sp][t][3]);
        }
    __syncthreads();
    attention_core<HD, false>(dyn, pb, SPL, a_gs, w_gs, sbias, HD, k_ext, v_ext, kv_ld, kv_seq, true, out, rows,
                              heads, scale_log2, out4, out_sf, 0, HD, h, q0);
}

}  // namespace

int q_cross_attention(const void* A, const void* SFA, const void* W, const void* SFB, const float* a_gs,
                      const float* w_gs, const __nv_bfloat16* bias, const __nv_bfloat16* k_ext,
                      const __nv_bfloat16* v_ext, int kv_ld, int kv_seq, __nv_bfloat16* out, int rows, int heads,
                      int hd, int K, int splits, float scale, bool pdl, cudaStream_t stream, uint8_t* out4,
                      uint8_t* out_sf) {
    if (hd != 64 || K != kFusedK || splits != 3 || rows < 1 || rows > 48 || heads < 1 || kv_seq < 1 ||
        !a_gs || !w_gs || !bias || !k_ext || !v_ext || !(out || out4) || (out4 && !out_sf))
        return static_cast<int>(cudaErrorInvalidValue);
    const int kv_pad = (kv_seq + kChunk - 1) / kChunk * kChunk;
    const size_t attn = std::max(stage_bytes<64>(kv_pad), sizeof(Merge<64>));
    const size_t pb_off = (attn + 15) / 16 * 16;
    const size_t smem = pb_off + size_t(3) * rows * 64 * 4 + 64 * 2 + 16 * (kFusedK / 2 + 16) +
                        16 * (kFusedK / 16 + 4);
    const void* fn = (const void*)q_cross_attention_kernel<64, 3>;
    static int opted = 48 * 1024;
    if (static_cast<int>(smem) > opted) {
        const cudaError_t e =
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (e != cudaSuccess) return static_cast<int>(e);
        opted = static_cast<int>(smem);
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(heads, (rows + kTileQ - 1) / kTileQ);
    cfg.blockDim = dim3(kWarps * 32);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    return static_cast<int>(cudaLaunchKernelEx(&cfg, q_cross_attention_kernel<64, 3>,
                                               static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(SFA),
                                               static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(SFB),
                                               a_gs, w_gs, bias, k_ext, v_ext, kv_ld, kv_seq, out, rows, heads,
                                               scale * 1.4426950408889634f, out4, out_sf,
                                               static_cast<int>(pb_off)));
}

int qkv_self_attention(const void* A, const void* SFA, const void* W, const void* SFB, const float* a_gs,
                       const float* w_gs, const __nv_bfloat16* bias, __nv_bfloat16* out, int rows, int heads,
                       int hd, int K, float scale, bool pdl, cudaStream_t stream, uint8_t* out4, uint8_t* out_sf) {
    if (hd != 64 || K != kFusedK || rows < 1 || rows > 48 || heads < 1 || !a_gs || !w_gs || !bias ||
        !(out || out4) || (out4 && !out_sf))
        return static_cast<int>(cudaErrorInvalidValue);
    using L = FusedSmem<64>;
    static bool opted = false;
    if (!opted) {
        const cudaError_t e = cudaFuncSetAttribute((const void*)qkv_self_attention_kernel<64>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(L::bytes));
        if (e != cudaSuccess) return static_cast<int>(e);
        opted = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(heads, (rows + kTileQ - 1) / kTileQ);
    cfg.blockDim = dim3(kWarps * 32);
    cfg.dynamicSmemBytes = L::bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    return static_cast<int>(cudaLaunchKernelEx(&cfg, qkv_self_attention_kernel<64>, static_cast<const uint8_t*>(A),
                                               static_cast<const uint8_t*>(SFA), static_cast<const uint8_t*>(W),
                                               static_cast<const uint8_t*>(SFB), a_gs, w_gs, bias, out, rows, heads,
                                               scale * 1.4426950408889634f, out4, out_sf));
}

int partials_attention(const float* partials, int splits, const float* a_gs, const float* w_gs,
                       const __nv_bfloat16* bias, int n_proj, const __nv_bfloat16* k_ext,
                       const __nv_bfloat16* v_ext, int kv_ld, int kv_seq, bool kv_static, __nv_bfloat16* out,
                       int rows, int heads, int hd, float scale, bool pdl, cudaStream_t stream,
                       uint8_t* out4, uint8_t* out_sf) {
    const bool self = k_ext == nullptr;
    if (out4 && (!out_sf || (heads * hd) % 16 != 0)) return static_cast<int>(cudaErrorInvalidValue);
    if (!partials || !a_gs || !w_gs || !bias || !(out || out4) || rows < 1 || heads < 1 || splits < 1 ||
        (self ? (v_ext != nullptr || kv_seq != rows || n_proj < 3 * heads * hd)
              : (v_ext == nullptr || kv_seq < 1 || kv_ld % 8 != 0 || n_proj < heads * hd)) ||
        n_proj % 4 != 0 || kv_seq > 1024)
        return static_cast<int>(cudaErrorInvalidValue);
    const float sl2 = scale * 1.4426950408889634f;
    if (hd == 48)
        return launch<48>(pdl, partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq, kv_static,
                          out, rows, heads, sl2, out4, out_sf, stream);
    if (hd == 64)
        return launch<64>(pdl, partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq, kv_static,
                          out, rows, heads, sl2, out4, out_sf, stream);
    return static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace nvfp4_skinny
}  // namespace flash_rt
