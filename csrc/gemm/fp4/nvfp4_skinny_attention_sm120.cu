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

template <int HD, bool PDL>
__global__ void __launch_bounds__(kWarps * 32)
partials_attention_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                          const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias, int n_proj,
                          const __nv_bfloat16* __restrict__ k_ext, const __nv_bfloat16* __restrict__ v_ext,
                          int kv_ld, int kv_seq, bool kv_static, __nv_bfloat16* __restrict__ out, int rows,
                          int heads, float scale_log2) {
    constexpr int KS = HD / 16;   // k-steps of Q K^T
    constexpr int NT = HD / 8;    // n-tiles of P V
    extern __shared__ __align__(16) unsigned char dyn[];
    const int kv_pad = (kv_seq + kChunk - 1) / kChunk * kChunk;
    const int kst = HD + 8, vst = kv_pad + 8;   // row strides (elements)
    __nv_bfloat16* ks_ = reinterpret_cast<__nv_bfloat16*>(dyn);
    __nv_bfloat16* vt_ = ks_ + static_cast<size_t>(kv_pad) * kst;
    Merge<HD>& mg = *reinterpret_cast<Merge<HD>*>(dyn);

    const int h = blockIdx.x, q0 = blockIdx.y * kTileQ;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;   // fragment row group / thread in group
    const bool self = k_ext == nullptr;
    const int width = heads * HD;
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
                        const size_t off = static_cast<size_t>(key) * kv_ld + h * HD + c;
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
        *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(q0 + r) * width + h * HD + c) =
            pack_bf16(a * inv, b * inv);
    }
}

template <int HD, bool PDL>
int launch_one(const float* partials, int splits, const float* a_gs, const float* w_gs, const __nv_bfloat16* bias,
               int n_proj, const __nv_bfloat16* k_ext, const __nv_bfloat16* v_ext, int kv_ld, int kv_seq,
               bool kv_static, __nv_bfloat16* out, int rows, int heads, float scale_log2, cudaStream_t stream) {
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
                                               heads, scale_log2));
}

template <int HD>
int launch(bool pdl, const float* partials, int splits, const float* a_gs, const float* w_gs,
           const __nv_bfloat16* bias, int n_proj, const __nv_bfloat16* k_ext, const __nv_bfloat16* v_ext, int kv_ld,
           int kv_seq, bool kv_static, __nv_bfloat16* out, int rows, int heads, float scale_log2,
           cudaStream_t stream) {
    return pdl ? launch_one<HD, true>(partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq,
                                      kv_static, out, rows, heads, scale_log2, stream)
               : launch_one<HD, false>(partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq,
                                       kv_static, out, rows, heads, scale_log2, stream);
}

}  // namespace

int partials_attention(const float* partials, int splits, const float* a_gs, const float* w_gs,
                       const __nv_bfloat16* bias, int n_proj, const __nv_bfloat16* k_ext,
                       const __nv_bfloat16* v_ext, int kv_ld, int kv_seq, bool kv_static, __nv_bfloat16* out,
                       int rows, int heads, int hd, float scale, bool pdl, cudaStream_t stream) {
    const bool self = k_ext == nullptr;
    if (!partials || !a_gs || !w_gs || !bias || !out || rows < 1 || heads < 1 || splits < 1 ||
        (self ? (v_ext != nullptr || kv_seq != rows || n_proj < 3 * heads * hd)
              : (v_ext == nullptr || kv_seq < 1 || kv_ld % 8 != 0 || n_proj < heads * hd)) ||
        n_proj % 4 != 0 || kv_seq > 1024)
        return static_cast<int>(cudaErrorInvalidValue);
    const float sl2 = scale * 1.4426950408889634f;
    if (hd == 48)
        return launch<48>(pdl, partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq, kv_static,
                          out, rows, heads, sl2, stream);
    if (hd == 64)
        return launch<64>(pdl, partials, splits, a_gs, w_gs, bias, n_proj, k_ext, v_ext, kv_ld, kv_seq, kv_static,
                          out, rows, heads, sl2, stream);
    return static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace nvfp4_skinny
}  // namespace flash_rt
