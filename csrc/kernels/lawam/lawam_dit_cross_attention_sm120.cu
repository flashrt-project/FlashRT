// SPDX-License-Identifier: Apache-2.0
// Cross-attention of the LaWAM flow DiT action rows over a long static K / V
// (the 512 image tokens) on sm_120.
//
// The skinny partials attention (gemm/fp4/nvfp4_skinny_attention_sm120.cu)
// stages a head's whole K and V^T in shared memory, which caps it near 300
// keys at head dim 64. Here the keys are split over gridDim.z CTAs per
// (head, 16-row query tile): each CTA stages its key slice before the grid
// dependency wait (K / V are computed once per call, before the flow; both
// row-major, the P V operand comes through ldmatrix.trans), runs the same
// four-warp online softmax on mma.sync m16n8k16 (BF16 operands, FP32
// accumulation) with Q read from the q-projection's K-split partials (+ bias,
// pre-scaled by softmax scale * log2 e), and writes its (max, sum, O) state;
// the last CTA of the tile to finish merges the slices in slice order and
// writes the BF16 output (deterministic).
#include "lawam_kernels.cuh"

#include "gemm/fp4/nvfp4_skinny_common.cuh"

#include <algorithm>

namespace flash_rt {
namespace lawam {
namespace {

using nvfp4_skinny::bf2;
using nvfp4_skinny::pdl_prologue;

constexpr int kWarps = 4;
constexpr int kTileQ = 16;
constexpr int kChunk = 32;   // keys per warp iteration
constexpr int kHd = 64;
constexpr int kKs = kHd / 16;
constexpr int kNt = kHd / 8;

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

struct Merge {
    float o[kWarps][kTileQ][kHd + 4];
    float m[kWarps][kTileQ];
    float l[kWarps][kTileQ];
};

// per-slice state in global scratch: [tile][slice] { m[16], l[16], o[16][64] }
constexpr int kStateFloats = kTileQ * 2 + kTileQ * kHd;

__host__ __device__ inline size_t stage_bytes(int kv_pad) {
    return static_cast<size_t>(kv_pad) * (kHd + 8) * 2 * 2;
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t* r, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

template <bool PDL>
__global__ void __launch_bounds__(kWarps * 32)
cross_attention_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                       const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias,
                       const __nv_bfloat16* __restrict__ k_ext, const __nv_bfloat16* __restrict__ v_ext, int kv_ld,
                       int kv_seq, int slice, __nv_bfloat16* __restrict__ out, int rows, int heads,
                       float scale_log2, float* __restrict__ state, unsigned* __restrict__ counters) {
    extern __shared__ __align__(16) unsigned char dyn[];
    __shared__ int last;
    const int kv_pad = slice;                    // a multiple of kChunk
    const int kst = kHd + 8;                     // K / V row stride (elements)
    __nv_bfloat16* ks_ = reinterpret_cast<__nv_bfloat16*>(dyn);
    __nv_bfloat16* vs_ = ks_ + static_cast<size_t>(kv_pad) * kst;
    Merge& mg = *reinterpret_cast<Merge*>(dyn);

    const int h = blockIdx.x, qt = blockIdx.y, z = blockIdx.z, nz = gridDim.z;
    const int q0 = qt * kTileQ;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int width = heads * kHd;
    const int k0 = z * slice;                         // first key of this CTA
    const int nk = max(0, min(kv_seq - k0, slice));   // keys of this CTA
    const size_t elems = static_cast<size_t>(rows) * width;
    const int per = ((nk + kWarps - 1) / kWarps + kChunk - 1) / kChunk * kChunk;
    const int kb = w * per, ke = min(nk, kb + per);

    // stage this CTA's keys (rows past nk zeroed) before the dependency wait
    {
        constexpr int U = kHd / 8;   // 16-byte units per key
        constexpr int kBatch = 8;
        const int total = kv_pad * U;
        for (int b0 = 0; b0 < total; b0 += kBatch * blockDim.x) {
            uint4 kr[kBatch], vr[kBatch];
#pragma unroll
            for (int u = 0; u < kBatch; ++u) {
                const int i = b0 + u * blockDim.x + threadIdx.x;
                const int key = i / U, c = (i % U) * 8;
                kr[u] = vr[u] = make_uint4(0u, 0u, 0u, 0u);
                if (i < total && key < nk) {
                    const size_t off = static_cast<size_t>(k0 + key) * kv_ld + h * kHd + c;
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
                *reinterpret_cast<uint4*>(vs_ + static_cast<size_t>(key) * kst + c) = vr[u];
            }
        }
    }
    if constexpr (PDL) pdl_prologue();
    const float alpha = (*a_gs) * (*w_gs);

    // Q fragments, rows q0 + g and q0 + g + 8
    float2 qf[kKs][4];
#pragma unroll
    for (int ks = 0; ks < kKs; ++ks)
#pragma unroll
        for (int j = 0; j < 4; ++j) qf[ks][j] = make_float2(0.f, 0.f);
    for (int sp = 0; sp < splits; ++sp) {
#pragma unroll
        for (int ks = 0; ks < kKs; ++ks) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = q0 + g + (j & 1) * 8;
                const int col = h * kHd + ks * 16 + (j >> 1) * 8 + t * 2;
                if (row < rows) {
                    const float2 v = *reinterpret_cast<const float2*>(partials + sp * elems +
                                                                       static_cast<size_t>(row) * width + col);
                    qf[ks][j].x += v.x;
                    qf[ks][j].y += v.y;
                }
            }
        }
    }
    uint32_t qa[kKs][4];
#pragma unroll
    for (int ks = 0; ks < kKs; ++ks) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = q0 + g + (j & 1) * 8;
            const int col = h * kHd + ks * 16 + (j >> 1) * 8 + t * 2;
            float2 v = make_float2(0.f, 0.f);
            if (row < rows) {
                const float2 bv = bf2(*reinterpret_cast<const uint32_t*>(bias + col));
                v = make_float2(qf[ks][j].x * alpha + bv.x, qf[ks][j].y * alpha + bv.y);
            }
            qa[ks][j] = pack_bf16(v.x * scale_log2, v.y * scale_log2);
        }
    }

    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[kNt][4];
#pragma unroll
    for (int n = 0; n < kNt; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;

    __syncthreads();
    for (int base = kb; base < ke; base += kChunk) {
        float s[kChunk / 8][4];
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int ks = 0; ks < kKs; ++ks) {
                const __nv_bfloat16* kr = ks_ + static_cast<size_t>(base + n * 8 + g) * kst + ks * 16 + t * 2;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + 8);
                mma_bf16(s[n], qa[ks], b0, b1);
            }
        }
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
            corr[r] = exp2f(m[r] - mn);
            m[r] = mn;
            l[r] *= corr[r];
        }
#pragma unroll
        for (int n = 0; n < kNt; ++n) {
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
            pa[n >> 1][(n & 1) * 2 + 0] = pack_bf16(p0, p1);
            pa[n >> 1][(n & 1) * 2 + 1] = pack_bf16(p2, p3);
        }
#pragma unroll
        for (int kk = 0; kk < kChunk / 16; ++kk) {
            // 8x8 tiles (keys x dims), transposed: keys +0 / +8 of dim tiles n, n + 1
            const int key = base + kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int n = 0; n < kNt; n += 2) {
                uint32_t b[4];
                ldsm_x4_trans(b, vs_ + static_cast<size_t>(key) * kst + n * 8 + (lane >> 4) * 8);
                mma_bf16(o[n], pa[kk], b[0], b[1]);
                mma_bf16(o[n + 1], pa[kk], b[2], b[3]);
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
    for (int n = 0; n < kNt; ++n) {
        *reinterpret_cast<float2*>(&mg.o[w][g][n * 8 + t * 2]) = make_float2(o[n][0], o[n][1]);
        *reinterpret_cast<float2*>(&mg.o[w][g + 8][n * 8 + t * 2]) = make_float2(o[n][2], o[n][3]);
    }
    if (t == 0) {
        mg.m[w][g] = m[0]; mg.l[w][g] = l[0];
        mg.m[w][g + 8] = m[1]; mg.l[w][g + 8] = l[1];
    }
    __syncthreads();

    // this CTA's (max, sum, unnormalized O) per row -> its slice state
    float* st = state + (static_cast<size_t>(h * gridDim.y + qt) * nz + z) * kStateFloats;
    for (int r = threadIdx.x; r < kTileQ; r += blockDim.x) {
        float M = -INFINITY;
#pragma unroll
        for (int k = 0; k < kWarps; ++k) M = fmaxf(M, mg.m[k][r]);
        float L = 0.f;
#pragma unroll
        for (int k = 0; k < kWarps; ++k) L += mg.l[k][r] > 0.f ? exp2f(mg.m[k][r] - M) * mg.l[k][r] : 0.f;
        st[r] = M;
        st[kTileQ + r] = L;
    }
    for (int i = threadIdx.x; i < kTileQ * kHd; i += blockDim.x) {
        const int r = i / kHd, c = i % kHd;
        float M = -INFINITY;
#pragma unroll
        for (int k = 0; k < kWarps; ++k) M = fmaxf(M, mg.m[k][r]);
        float a = 0.f;
#pragma unroll
        for (int k = 0; k < kWarps; ++k) a += mg.l[k][r] > 0.f ? exp2f(mg.m[k][r] - M) * mg.o[k][r][c] : 0.f;
        st[2 * kTileQ + i] = a;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* cnt = counters + h * gridDim.y + qt;
        last = atomicAdd(cnt, 1u) == static_cast<unsigned>(nz - 1);
        if (last) *cnt = 0u;   // every slice has arrived: ready for the next launch
    }
    __syncthreads();
    if (!last) return;
    __threadfence();

    // merge the slices in slice order
    const float* s0 = state + static_cast<size_t>(h * gridDim.y + qt) * nz * kStateFloats;
    for (int i = threadIdx.x; i < kTileQ * (kHd / 2); i += blockDim.x) {
        const int r = i / (kHd / 2), c = (i % (kHd / 2)) * 2;
        if (q0 + r >= rows) continue;
        float M = -INFINITY;
        for (int k = 0; k < nz; ++k) M = fmaxf(M, __ldcg(s0 + k * kStateFloats + r));
        float L = 0.f, a = 0.f, b = 0.f;
        for (int k = 0; k < nz; ++k) {
            const float* sk = s0 + k * kStateFloats;
            const float lk = __ldcg(sk + kTileQ + r);
            const float f = lk > 0.f ? exp2f(__ldcg(sk + r) - M) : 0.f;
            L += f * lk;
            a += f * __ldcg(sk + 2 * kTileQ + r * kHd + c);
            b += f * __ldcg(sk + 2 * kTileQ + r * kHd + c + 1);
        }
        const float inv = 1.f / L;
        *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(q0 + r) * width + h * kHd + c) =
            pack_bf16(a * inv, b * inv);
    }
}

}  // namespace

size_t dit_cross_attention_state_floats(int rows, int heads, int kv_slices) {
    return static_cast<size_t>(heads) * ((rows + kTileQ - 1) / kTileQ) * kv_slices * kStateFloats;
}

int dit_cross_attention(const float* partials, int splits, const float* a_gs, const float* w_gs,
                        const __nv_bfloat16* bias, const __nv_bfloat16* k_ext, const __nv_bfloat16* v_ext, int kv_ld,
                        int kv_seq, int kv_slices, __nv_bfloat16* out, int rows, int heads, int hd, float scale,
                        float* state, unsigned* counters, bool pdl, cudaStream_t stream) {
    if (!partials || !a_gs || !w_gs || !bias || !k_ext || !v_ext || !out || !state || !counters || rows < 1 ||
        heads < 1 || splits < 1 || hd != kHd || kv_seq < 1 || kv_ld % 8 != 0 || kv_slices < 1 || kv_slices > 16)
        return static_cast<int>(cudaErrorInvalidValue);
    const int per = (kv_seq + kv_slices - 1) / kv_slices;
    const int slice = (per + kChunk - 1) / kChunk * kChunk;
    const size_t smem = std::max(stage_bytes(slice), sizeof(Merge));
    if (smem > 99 * 1024) return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)cross_attention_kernel<true> : (const void*)cross_attention_kernel<false>;
    if (smem > 48 * 1024) {
        const cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   static_cast<int>(smem));
        if (e != cudaSuccess) return static_cast<int>(e);
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(heads, (rows + kTileQ - 1) / kTileQ, kv_slices);
    cfg.blockDim = dim3(kWarps * 32);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const float sl2 = scale * 1.4426950408889634f;
    if (pdl)
        return static_cast<int>(cudaLaunchKernelEx(&cfg, cross_attention_kernel<true>, partials, splits, a_gs, w_gs,
                                                   bias, k_ext, v_ext, kv_ld, kv_seq, slice, out, rows, heads, sl2,
                                                   state, counters));
    return static_cast<int>(cudaLaunchKernelEx(&cfg, cross_attention_kernel<false>, partials, splits, a_gs, w_gs,
                                               bias, k_ext, v_ext, kv_ld, kv_seq, slice, out, rows, heads, sl2,
                                               state, counters));
}

}  // namespace lawam
}  // namespace flash_rt
