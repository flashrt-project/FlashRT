// SPDX-License-Identifier: Apache-2.0
// Skinny FP8 GEMM family for small-row action decoders on sm_120a. See the header.
#include "pi05_decoder_skinny_fp8_sm120.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace flash_rt {
namespace pi05_dec_skinny {
namespace {

__device__ __forceinline__ void mma_e4m3(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void pdl_prologue() {
    asm volatile("griddepcontrol.launch_dependents;\n" ::);
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

__device__ __forceinline__ uint8_t to_e4m3_bits(float v) {
    const __nv_fp8_e4m3 q(fminf(fmaxf(v, -448.0f), 448.0f));
    return *reinterpret_cast<const uint8_t*>(&q);
}

__device__ __forceinline__ float tanh_gelu(float g) {
    return g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
}

// One CTA: BN weight rows x KC k for one 16-row tile (blockIdx.z picks the tile).
template <int BN, int KC, int NW, bool ABF16, bool PDL>
__global__ void __launch_bounds__(NW * 32)
gemm_kernel(const void* __restrict__ A_, const float* __restrict__ a_scale,
            const uint8_t* __restrict__ W, float* __restrict__ partials, int M, int N, int K) {
    constexpr int THREADS = NW * 32;
    constexpr int NT = BN / 8;
    constexpr int NTW = NT / NW;
    constexpr int KG = KC / 64;
    constexpr int ASTRIDE = KC + 16;
    static_assert(NT % NW == 0, "n8 tiles must split across warps");
    __shared__ __align__(16) uint8_t sA[16 * ASTRIDE];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, h = lane >> 2, l = lane & 3;
    const int n0 = blockIdx.x * BN;
    const int kb = blockIdx.y * KC;
    const int m0 = blockIdx.z * 16;

    uint4 w[KG][NTW];
#pragma unroll
    for (int g = 0; g < KG; ++g)
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const int n = n0 + (warp * NTW + t) * 8 + h;
            w[g][t] = ldg_stream(W + static_cast<size_t>(n) * K + kb + g * 64 + l * 16);
        }

    if constexpr (PDL) pdl_prologue();

    constexpr int ACH = 16 * (KC / 16);
    if constexpr (!ABF16) {
        const uint8_t* A = static_cast<const uint8_t*>(A_);
        for (int c = tid; c < ACH; c += THREADS) {
            const int row = c / (KC / 16);
            const int ko = (c % (KC / 16)) * 16;
            const bool v = m0 + row < M;
            cp_async_16(sA + row * ASTRIDE + ko, A + static_cast<size_t>(v ? m0 + row : 0) * K + kb + ko, v);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
    } else {
        const __nv_bfloat16* A = static_cast<const __nv_bfloat16*>(A_);
        const float inv = 1.0f / (*a_scale);
        for (int c = tid; c < ACH; c += THREADS) {
            const int row = c / (KC / 16);
            const int ko = (c % (KC / 16)) * 16;
            uint8_t q[16];
            if (m0 + row < M) {
                const __nv_bfloat16* src = A + static_cast<size_t>(m0 + row) * K + kb + ko;
                const uint4 p0 = *reinterpret_cast<const uint4*>(src);
                const uint4 p1 = *reinterpret_cast<const uint4*>(src + 8);
                const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&p0);
                const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&p1);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f0 = __bfloat1622float2(b0[i]);
                    const float2 f1 = __bfloat1622float2(b1[i]);
                    q[2 * i] = to_e4m3_bits(f0.x * inv);
                    q[2 * i + 1] = to_e4m3_bits(f0.y * inv);
                    q[8 + 2 * i] = to_e4m3_bits(f1.x * inv);
                    q[8 + 2 * i + 1] = to_e4m3_bits(f1.y * inv);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) q[i] = 0;
            }
            *reinterpret_cast<uint4*>(sA + row * ASTRIDE + ko) = *reinterpret_cast<const uint4*>(q);
        }
    }
    __syncthreads();

    float acc[NTW][4];
#pragma unroll
    for (int t = 0; t < NTW; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;

    // mma logical k (4l+i | 16+4l+i) <-> physical kb + g*64 + 16l + 8j + (i | 4+i)
#pragma unroll
    for (int g = 0; g < KG; ++g) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint8_t* r0 = sA + h * ASTRIDE + g * 64 + l * 16 + j * 8;
            const uint8_t* r1 = r0 + 8 * ASTRIDE;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(r0);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(r1);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(r0 + 4);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(r1 + 4);
#pragma unroll
            for (int t = 0; t < NTW; ++t) {
                const uint32_t b0 = j == 0 ? w[g][t].x : w[g][t].z;
                const uint32_t b1 = j == 0 ? w[g][t].y : w[g][t].w;
                mma_e4m3(acc[t], a0, a1, a2, a3, b0, b1);
            }
        }
    }

    float* base = partials + static_cast<size_t>(blockIdx.y) * M * N;
#pragma unroll
    for (int t = 0; t < NTW; ++t) {
        const int col = n0 + (warp * NTW + t) * 8 + 2 * l;
        const int row0 = m0 + h;
        if (row0 < M)
            *reinterpret_cast<float2*>(base + static_cast<size_t>(row0) * N + col) =
                make_float2(acc[t][0], acc[t][1]);
        if (row0 + 8 < M)
            *reinterpret_cast<float2*>(base + static_cast<size_t>(row0 + 8) * N + col) =
                make_float2(acc[t][2], acc[t][3]);
    }
}

// Sum of the split partials for one element pair, scaled and rounded to BF16
// (the unfused path consumed the GEMM's BF16 output).
__device__ __forceinline__ float2 sum_pair(const float* __restrict__ partials, int splits, size_t elements,
                                           size_t pair, float alpha) {
    float2 s0 = make_float2(0.f, 0.f), s1 = make_float2(0.f, 0.f);
    int s = 0;
    for (; s + 1 < splits; s += 2) {
        const float2 p0 = *reinterpret_cast<const float2*>(partials + s * elements + 2 * pair);
        const float2 p1 = *reinterpret_cast<const float2*>(partials + (s + 1) * elements + 2 * pair);
        s0.x += p0.x; s0.y += p0.y; s1.x += p1.x; s1.y += p1.y;
    }
    if (s < splits) {
        const float2 p0 = *reinterpret_cast<const float2*>(partials + s * elements + 2 * pair);
        s0.x += p0.x; s0.y += p0.y;
    }
    return __bfloat1622float2(__floats2bfloat162_rn((s0.x + s1.x) * alpha, (s0.y + s1.y) * alpha));
}

template <bool PDL>
__global__ void sum_rope_kernel(const float* __restrict__ partials, int splits,
                                const float* __restrict__ a_scale, const float* __restrict__ w_scale,
                                const __nv_bfloat16* __restrict__ rope, __nv_bfloat16* __restrict__ Q,
                                __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V,
                                const int* __restrict__ devpos, int rows, int q_dim, int k_dim,
                                int v_dim, int head_dim, int sample_rows, long long kv_sample_stride) {
    if constexpr (PDL) pdl_prologue();
    const int qkv_dim = q_dim + k_dim + v_dim;
    const int pairs = qkv_dim >> 1;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * pairs) return;
    const int row = index / pairs;
    const int col = (index % pairs) * 2;
    const size_t elements = static_cast<size_t>(rows) * qkv_dim;
    const float alpha = (*a_scale) * (*w_scale);
    const float2 x = sum_pair(partials, splits, elements, static_cast<size_t>(index), alpha);
    const int pos = devpos ? devpos[0] : 0;
    const int sample = row / sample_rows;
    const int r = row - sample * sample_rows;   // row within the sample (RoPE position)
    const long long kv_row = static_cast<long long>(sample) * kv_sample_stride +
                             static_cast<long long>(pos + r);
    if (col < q_dim + k_dim) {
        const int d = (col < q_dim ? col : col - q_dim) % head_dim;
        const int rope_base = r * head_dim + d;
        const float c = __bfloat162float(rope[rope_base]);
        const float s = __bfloat162float(rope[rope_base + 1]);
        const __nv_bfloat162 packed = __floats2bfloat162_rn(x.x * c - x.y * s, x.y * c + x.x * s);
        if (col < q_dim) {
            *reinterpret_cast<__nv_bfloat162*>(Q + static_cast<size_t>(row) * q_dim + col) = packed;
        } else {
            *reinterpret_cast<__nv_bfloat162*>(K + kv_row * k_dim + (col - q_dim)) = packed;
        }
    } else {
        *reinterpret_cast<__nv_bfloat162*>(V + kv_row * v_dim + (col - q_dim - k_dim)) =
            __floats2bfloat162_rn(x.x, x.y);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One 256-thread block per row; dim must be a multiple of 512 (pairs per thread = dim / 512).
template <int DIM, bool OUT_FP8, bool PDL>
__global__ void __launch_bounds__(256)
residual_ada_norm_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_scale,
                         const float* __restrict__ w_scale, __nv_bfloat16* __restrict__ residual,
                         const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ weight,
                         const __nv_bfloat16* __restrict__ style, __nv_fp8_e4m3* __restrict__ out_fp8,
                         __nv_bfloat16* __restrict__ out_bf16, const float* __restrict__ out_scale,
                         __nv_bfloat16* __restrict__ gate_out, int rows, float eps) {
    constexpr int PPT = DIM / 512;
    __shared__ float warp_partial[8];
    const int row = blockIdx.x;
    const size_t base = static_cast<size_t>(row) * DIM;
    const size_t elements = static_cast<size_t>(rows) * DIM;
    // Style and norm weight are constants of the graph: fetch them before
    // waiting on the producer so the wait overlaps their latency.
    const __nv_bfloat16* style_row = style + static_cast<size_t>(row) * 3 * DIM;
    const __nv_bfloat162* sc2 = reinterpret_cast<const __nv_bfloat162*>(style_row);
    const __nv_bfloat162* sh2 = reinterpret_cast<const __nv_bfloat162*>(style_row + DIM);
    const __nv_bfloat162* gt2 = reinterpret_cast<const __nv_bfloat162*>(style_row + 2 * DIM);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162 wv2[PPT], sv2[PPT], hv2[PPT], gv2[PPT];
#pragma unroll
    for (int p = 0; p < PPT; ++p) {
        const int i = threadIdx.x + p * 256;
        wv2[p] = w2[i]; sv2[p] = sc2[i]; hv2[p] = sh2[i]; gv2[p] = gt2[i];
    }
    if constexpr (PDL) pdl_prologue();
    const float alpha = (*a_scale) * (*w_scale);
    __nv_bfloat162* res2 = reinterpret_cast<__nv_bfloat162*>(residual + base);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(gate + base);
    __nv_bfloat162 kept[PPT];
    float sq = 0.f;
#pragma unroll
    for (int p = 0; p < PPT; ++p) {
        const int i = threadIdx.x + p * 256;
        const float2 x = sum_pair(partials, splits, elements, base / 2 + i, alpha);
        const float2 rv = __bfloat1622float2(res2[i]);
        const float2 gv = __bfloat1622float2(g2[i]);
        const float r0 = rv.x + x.x * gv.x;
        const float r1 = rv.y + x.y * gv.y;
        kept[p] = __floats2bfloat162_rn(r0, r1);
        res2[i] = kept[p];
        if constexpr (OUT_FP8) {
            sq += r0 * r0 + r1 * r1;                     // as gate_residual_ada_norm_fp8
        } else {
            const float2 q = __bfloat1622float2(kept[p]);  // as gate_mul_residual + ada_rms_norm_style
            sq += q.x * q.x + q.y * q.y;
        }
    }
    sq = warp_sum(sq);
    if ((threadIdx.x & 31) == 0) warp_partial[threadIdx.x >> 5] = sq;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += warp_partial[w];
    const float rms = rsqrtf(total / DIM + eps);
    const float inv_scale = OUT_FP8 ? 1.0f / (*out_scale) : 1.0f;
#pragma unroll
    for (int p = 0; p < PPT; ++p) {
        const int i = threadIdx.x + p * 256;
        const float2 rv = __bfloat1622float2(kept[p]);
        const float2 wv = __bfloat1622float2(wv2[p]);
        const float2 sv = __bfloat1622float2(sv2[p]);
        const float2 hv = __bfloat1622float2(hv2[p]);
        const float n0 = rv.x * rms * wv.x;
        const float n1 = rv.y * rms * wv.y;
        const float v0 = n0 * (1.0f + sv.x) + hv.x;
        const float v1 = n1 * (1.0f + sv.y) + hv.y;
        if constexpr (OUT_FP8) {
            out_fp8[base + 2 * i] = __nv_fp8_e4m3(fminf(fmaxf(v0 * inv_scale, -448.0f), 448.0f));
            out_fp8[base + 2 * i + 1] = __nv_fp8_e4m3(fminf(fmaxf(v1 * inv_scale, -448.0f), 448.0f));
        } else {
            reinterpret_cast<__nv_bfloat162*>(out_bf16 + base)[i] = __floats2bfloat162_rn(v0, v1);
        }
        reinterpret_cast<__nv_bfloat162*>(gate_out + base)[i] = gv2[p];
    }
}

template <bool PDL>
__global__ void residual_gate_mul_kernel(const float* __restrict__ partials, int splits,
                                         const float* __restrict__ a_scale, const float* __restrict__ w_scale,
                                         __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ gate,
                                         size_t elements) {
    if constexpr (PDL) pdl_prologue();
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (2 * i >= elements) return;
    const float alpha = (*a_scale) * (*w_scale);
    const float2 x = sum_pair(partials, splits, elements, i, alpha);
    __nv_bfloat162* res2 = reinterpret_cast<__nv_bfloat162*>(residual);
    const float2 rv = __bfloat1622float2(res2[i]);
    const float2 gv = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(gate)[i]);
    res2[i] = __floats2bfloat162_rn(rv.x + x.x * gv.x, rv.y + x.y * gv.y);
}

template <bool PDL>
__global__ void gate_gelu_fp8_kernel(const float* __restrict__ partials, int splits,
                                     const float* __restrict__ a_scale, const float* __restrict__ w_scale,
                                     __nv_fp8_e4m3* __restrict__ out, int rows, int half,
                                     const float* __restrict__ out_scale) {
    if constexpr (PDL) pdl_prologue();
    const int index = blockIdx.x * blockDim.x + threadIdx.x;   // one pair of columns
    const int half_pairs = half >> 1;
    if (index >= rows * half_pairs) return;
    const int row = index / half_pairs;
    const int pair = index % half_pairs;
    const size_t elements = static_cast<size_t>(rows) * 2 * half;
    const float alpha = (*a_scale) * (*w_scale);
    const size_t row_pairs = static_cast<size_t>(row) * half;   // pairs per row = 2*half/2
    const float2 g = sum_pair(partials, splits, elements, row_pairs + pair, alpha);
    const float2 u = sum_pair(partials, splits, elements, row_pairs + half_pairs + pair, alpha);
    const float inv_scale = 1.0f / (*out_scale);
    const float v0 = tanh_gelu(g.x) * u.x;
    const float v1 = tanh_gelu(g.y) * u.y;
    out[static_cast<size_t>(row) * half + 2 * pair] = __nv_fp8_e4m3(fminf(fmaxf(v0 * inv_scale, -448.0f), 448.0f));
    out[static_cast<size_t>(row) * half + 2 * pair + 1] = __nv_fp8_e4m3(fminf(fmaxf(v1 * inv_scale, -448.0f), 448.0f));
}


// ── Decoder cross-attention (split-KV pair, both launches in the PDL chain) ──
namespace attn_impl {

constexpr int HD = 256;
constexpr int KCH = 64;                 // keys per CTA
constexpr int THREADS = 128;            // 4 warps
constexpr int KSTR = HD + 8;            // bf16 elements per K/V/Q smem row
constexpr int PSTR = KCH + 8;           // bf16 elements per P smem row
constexpr int PART = 32 + 16 * HD;      // floats per split partial: m[16], l[16], O[16][256]
constexpr int MAX_SPLITS = 32;
constexpr size_t SMEM = (2 * KCH + 16) * KSTR * 2 + 16 * PSTR * 2 + 16 * KCH * 4 + 32 * 4;

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, const void* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}

__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

// One CTA per (key chunk, head, sample): S = Q K^T over its keys, row max and
// exp-sum, O = P V, written as an FP32 partial for the combine kernel.
template <bool PDL>
__global__ void __launch_bounds__(THREADS)
attn_split_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                  const __nv_bfloat16* __restrict__ V, int rows, int heads, int q_row_stride,
                  int kv_len, const int* __restrict__ seqused, long long kv_sample_stride,
                  float scale, float* __restrict__ scratch, int splits) {
    extern __shared__ __align__(16) uint8_t smem_raw[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sV = sK + KCH * KSTR;
    __nv_bfloat16* sQ = sV + KCH * KSTR;
    __nv_bfloat16* sP = sQ + 16 * KSTR;
    float* sS = reinterpret_cast<float*>(sP + 16 * PSTR);
    float* sM = sS + 16 * KCH;
    float* sL = sM + 16;

    const int split = blockIdx.x, head = blockIdx.y, sample = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    if constexpr (PDL) pdl_prologue();

    const int valid = seqused ? seqused[0] : kv_len;
    const int k0 = split * KCH;
    const __nv_bfloat16* Kb = K + static_cast<size_t>(sample) * kv_sample_stride * HD;
    const __nv_bfloat16* Vb = V + static_cast<size_t>(sample) * kv_sample_stride * HD;
    const __nv_bfloat16* Qb = Q + static_cast<size_t>(sample) * rows * q_row_stride + head * HD;
    for (int c = tid; c < KCH * (HD / 8); c += THREADS) {
        const int r = c / (HD / 8);
        const int d = (c % (HD / 8)) * 8;
        const bool ok = (k0 + r) < valid && (k0 + r) < kv_len;
        const size_t src = static_cast<size_t>(ok ? k0 + r : 0) * HD + d;
        cp_async_16(sK + r * KSTR + d, Kb + src, ok);
    }
    for (int c = tid; c < 16 * (HD / 8); c += THREADS) {
        const int r = c / (HD / 8);
        const int d = (c % (HD / 8)) * 8;
        const bool ok = r < rows;
        cp_async_16(sQ + r * KSTR + d, Qb + static_cast<size_t>(ok ? r : 0) * q_row_stride + d, ok);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    for (int c = tid; c < KCH * (HD / 8); c += THREADS) {
        const int r = c / (HD / 8);
        const int d = (c % (HD / 8)) * 8;
        const bool ok = (k0 + r) < valid && (k0 + r) < kv_len;
        const size_t src = static_cast<size_t>(ok ? k0 + r : 0) * HD + d;
        cp_async_16(sV + r * KSTR + d, Vb + src, ok);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);   // K and Q landed; V still in flight
    __syncthreads();

    // S = Q K^T for this warp's 16 keys (two n8 tiles).
    float s[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[j][e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < HD / 16; ++ks) {
        const int d0 = ks * 16 + 2 * t;
        const uint32_t a0 = ld32(sQ + g * KSTR + d0);
        const uint32_t a1 = ld32(sQ + (g + 8) * KSTR + d0);
        const uint32_t a2 = ld32(sQ + g * KSTR + d0 + 8);
        const uint32_t a3 = ld32(sQ + (g + 8) * KSTR + d0 + 8);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int key = (warp * 2 + j) * 8 + g;
            const uint32_t b0 = ld32(sK + key * KSTR + d0);
            const uint32_t b1 = ld32(sK + key * KSTR + d0 + 8);
            mma_bf16(s[j], a0, a1, a2, a3, b0, b1);
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = g + (e >= 2 ? 8 : 0);
            const int key = (warp * 2 + j) * 8 + 2 * t + (e & 1);
            sS[row * KCH + key] = (k0 + key) < valid ? s[j][e] * scale : -INFINITY;
        }
    __syncthreads();

    // Row statistics and P = exp(S - m): 8 threads per row, 8 keys each.
    {
        const int row = tid >> 3, part = tid & 7;
        float m = -INFINITY;
#pragma unroll
        for (int k = 0; k < 8; ++k) m = fmaxf(m, sS[row * KCH + part * 8 + k]);
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float l = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float v = sS[row * KCH + part * 8 + k];
            const float pval = (m == -INFINITY) ? 0.f : __expf(v - m);
            l += pval;
            sP[row * PSTR + part * 8 + k] = __float2bfloat16(pval);
        }
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (part == 0) { sM[row] = m; sL[row] = l; }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);   // V landed
    __syncthreads();

    // O = P V for this warp's 64 output dims (eight n8 tiles).
    float o[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KCH / 16; ++ks) {
        const int kk = ks * 16 + 2 * t;
        const uint32_t a0 = ld32(sP + g * PSTR + kk);
        const uint32_t a1 = ld32(sP + (g + 8) * PSTR + kk);
        const uint32_t a2 = ld32(sP + g * PSTR + kk + 8);
        const uint32_t a3 = ld32(sP + (g + 8) * PSTR + kk + 8);
        const __nv_bfloat16* vrow = sV + (ks * 16 + (lane & 15)) * KSTR + warp * 64;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t b0, b1;
            ldmatrix_x2_trans(b0, b1, vrow + j * 8);
            mma_bf16(o[j], a0, a1, a2, a3, b0, b1);
        }
    }

    float* part = scratch + (static_cast<size_t>(sample * heads + head) * splits + split) * PART;
    if (tid < 16) { part[tid] = sM[tid]; part[16 + tid] = sL[tid]; }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * t;
        *reinterpret_cast<float2*>(part + 32 + g * HD + d) = make_float2(o[j][0], o[j][1]);
        *reinterpret_cast<float2*>(part + 32 + (g + 8) * HD + d) = make_float2(o[j][2], o[j][3]);
    }
}

// One CTA per (row, head, sample); thread owns two output dims. All split
// loads are independent, so they overlap.
template <bool PDL>
__global__ void __launch_bounds__(THREADS)
attn_combine_kernel(const float* __restrict__ scratch, __nv_bfloat16* __restrict__ O, int rows,
                    int heads, int q_row_stride, int splits) {
    __shared__ float sW[MAX_SPLITS];
    __shared__ float sInv;
    const int row = blockIdx.x, head = blockIdx.y, sample = blockIdx.z;
    const int tid = threadIdx.x;
    if constexpr (PDL) pdl_prologue();
    const float* base = scratch + static_cast<size_t>(sample * heads + head) * splits * PART;
    if (tid < 32) {
        const float ms = tid < splits ? base[tid * PART + row] : -INFINITY;
        const float ls = tid < splits ? base[tid * PART + 16 + row] : 0.f;
        float M = ms;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
        const float w = (ms == -INFINITY) ? 0.f : __expf(ms - M);
        float L = w * ls;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
        sW[tid] = w;
        if (tid == 0) sInv = 1.0f / L;
    }
    __syncthreads();
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll 8
    for (int sp = 0; sp < splits; ++sp) {
        const float2 ov = *reinterpret_cast<const float2*>(base + sp * PART + 32 + row * HD + 2 * tid);
        const float w = sW[sp];
        acc0 += w * ov.x;
        acc1 += w * ov.y;
    }
    *reinterpret_cast<__nv_bfloat162*>(
        O + (static_cast<size_t>(sample) * rows + row) * q_row_stride + head * HD + 2 * tid) =
        __floats2bfloat162_rn(acc0 * sInv, acc1 * sInv);
}

}  // namespace attn_impl


// ── Action projections around the layer stack ────────────────────────────
// in: x = bf16(bf16(noise @ W_in) + b_in) into the residual stream, then the
// first layer's adaptive RMS norm to FP8 (as ada_rms_norm_style_fp8).
template <bool PDL>
__global__ void __launch_bounds__(256)
action_in_norm_kernel(const __nv_bfloat16* __restrict__ noise, const __nv_bfloat16* __restrict__ w_in,
                      const __nv_bfloat16* __restrict__ b_in, __nv_bfloat16* __restrict__ x,
                      const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ style,
                      __nv_fp8_e4m3* __restrict__ out, __nv_bfloat16* __restrict__ gate_out,
                      const float* __restrict__ out_scale, float eps) {
    constexpr int DIM = 1024, KDIM = 32;
    __shared__ float warp_partial[8];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int n0 = tid * 4;
    const __nv_bfloat16* style_row = style + static_cast<size_t>(row) * 3 * DIM;
    // graph constants first (overlap the producer wait)
    float2 wq[KDIM][2];
#pragma unroll
    for (int k = 0; k < KDIM; ++k) {
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(w_in + k * DIM + n0);
        wq[k][0] = __bfloat1622float2(wp[0]);
        wq[k][1] = __bfloat1622float2(wp[1]);
    }
    const float2 bv0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(b_in + n0)[0]);
    const float2 bv1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(b_in + n0)[1]);
    const float2 wv0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(weight + n0)[0]);
    const float2 wv1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(weight + n0)[1]);
    const __nv_bfloat162 sc0 = reinterpret_cast<const __nv_bfloat162*>(style_row + n0)[0];
    const __nv_bfloat162 sc1 = reinterpret_cast<const __nv_bfloat162*>(style_row + n0)[1];
    const __nv_bfloat162 sh0 = reinterpret_cast<const __nv_bfloat162*>(style_row + DIM + n0)[0];
    const __nv_bfloat162 sh1 = reinterpret_cast<const __nv_bfloat162*>(style_row + DIM + n0)[1];
    const __nv_bfloat162 gt0 = reinterpret_cast<const __nv_bfloat162*>(style_row + 2 * DIM + n0)[0];
    const __nv_bfloat162 gt1 = reinterpret_cast<const __nv_bfloat162*>(style_row + 2 * DIM + n0)[1];
    if constexpr (PDL) pdl_prologue();
    float nv[KDIM];
#pragma unroll
    for (int k = 0; k < KDIM; k += 2) {
        const float2 v = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(noise + static_cast<size_t>(row) * KDIM + k));
        nv[k] = v.x; nv[k + 1] = v.y;
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < KDIM; ++k) {
        acc[0] += nv[k] * wq[k][0].x; acc[1] += nv[k] * wq[k][0].y;
        acc[2] += nv[k] * wq[k][1].x; acc[3] += nv[k] * wq[k][1].y;
    }
    // GEMM output rounded to BF16, then bias added and rounded again
    const float2 g0 = __bfloat1622float2(__floats2bfloat162_rn(acc[0], acc[1]));
    const float2 g1 = __bfloat1622float2(__floats2bfloat162_rn(acc[2], acc[3]));
    const __nv_bfloat162 x0 = __floats2bfloat162_rn(g0.x + bv0.x, g0.y + bv0.y);
    const __nv_bfloat162 x1 = __floats2bfloat162_rn(g1.x + bv1.x, g1.y + bv1.y);
    __nv_bfloat162* xrow = reinterpret_cast<__nv_bfloat162*>(x + static_cast<size_t>(row) * DIM + n0);
    xrow[0] = x0; xrow[1] = x1;
    const float2 xf0 = __bfloat1622float2(x0), xf1 = __bfloat1622float2(x1);
    float sq = xf0.x * xf0.x + xf0.y * xf0.y + xf1.x * xf1.x + xf1.y * xf1.y;
    sq = warp_sum(sq);
    if ((tid & 31) == 0) warp_partial[tid >> 5] = sq;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += warp_partial[w];
    const float rms = rsqrtf(total / DIM + eps);
    const float inv_scale = 1.0f / (*out_scale);
    const float2 s0 = __bfloat1622float2(sc0), s1 = __bfloat1622float2(sc1);
    const float2 h0 = __bfloat1622float2(sh0), h1 = __bfloat1622float2(sh1);
    const float vals[4] = {
        (xf0.x * rms * wv0.x * (1.0f + s0.x) + h0.x) * inv_scale,
        (xf0.y * rms * wv0.y * (1.0f + s0.y) + h0.y) * inv_scale,
        (xf1.x * rms * wv1.x * (1.0f + s1.x) + h1.x) * inv_scale,
        (xf1.y * rms * wv1.y * (1.0f + s1.y) + h1.y) * inv_scale};
#pragma unroll
    for (int j = 0; j < 4; ++j)
        out[static_cast<size_t>(row) * DIM + n0 + j] = __nv_fp8_e4m3(fminf(fmaxf(vals[j], -448.0f), 448.0f));
    __nv_bfloat162* grow = reinterpret_cast<__nv_bfloat162*>(gate_out + static_cast<size_t>(row) * DIM + n0);
    grow[0] = gt0; grow[1] = gt1;
}

// out: a = bf16(bf16(x_normed @ W_out) + b_out) (W_out, b_out carry the
// -1/num_steps factor), optionally traced, then noise += a in BF16.
template <bool PDL>
__global__ void __launch_bounds__(256)
action_out_residual_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w_out,
                           const __nv_bfloat16* __restrict__ b_out, __nv_bfloat16* __restrict__ action,
                           __nv_bfloat16* __restrict__ noise, __nv_bfloat16* __restrict__ trace_x,
                           __nv_bfloat16* __restrict__ trace_delta) {
    constexpr int KDIM = 1024, NDIM = 32, SLICES = 8, KS = KDIM / SLICES;
    __shared__ float part[SLICES][NDIM];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int n = tid & 31, slice = tid >> 5;
    const float bias = __bfloat162float(b_out[n]);
    if constexpr (PDL) pdl_prologue();
    const __nv_bfloat16* xr = x + static_cast<size_t>(row) * KDIM + slice * KS;
    const __nv_bfloat16* wr = w_out + static_cast<size_t>(slice) * KS * NDIM + n;
    float acc = 0.f;
#pragma unroll 8
    for (int k = 0; k < KS; ++k) acc += __bfloat162float(xr[k]) * __bfloat162float(wr[k * NDIM]);
    part[slice][n] = acc;
    __syncthreads();
    if (slice == 0) {
        float total = 0.f;
#pragma unroll
        for (int s2 = 0; s2 < SLICES; ++s2) total += part[s2][n];
        const float g = __bfloat162float(__float2bfloat16(total));
        const __nv_bfloat16 a = __float2bfloat16(g + bias);
        const size_t idx = static_cast<size_t>(row) * NDIM + n;
        const __nv_bfloat16 old = noise[idx];
        action[idx] = a;
        if (trace_x) { trace_x[idx] = old; trace_delta[idx] = a; }
        noise[idx] = __float2bfloat16(__bfloat162float(old) + __bfloat162float(a));
    }
}

template <typename... Args>
int launch_ex(const void* fn, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    void* pargs[] = {(void*)&args...};
    return static_cast<int>(cudaLaunchKernelExC(&cfg, fn, pargs));
}

struct Cfg { int bn, kc, nw; };
constexpr Cfg kCfgs[] = {
    {128, 256, 4}, {64, 256, 4}, {128, 128, 4}, {64, 128, 4},
    {128, 512, 4}, {64, 512, 4}, {128, 256, 8}, {256, 256, 8},
};
constexpr int kNumCfgs = sizeof(kCfgs) / sizeof(kCfgs[0]);

template <int BN, int KC, int NW, bool ABF16>
int launch_gemm(const void* A, const float* a_scale, const void* W, float* partials, int M, int N, int K,
                bool pdl, cudaStream_t stream) {
    if (M < 1 || N % BN != 0 || K % KC != 0) return static_cast<int>(cudaErrorInvalidValue);
    const dim3 grid(N / BN, K / KC, (M + 15) / 16);
    const uint8_t* w = static_cast<const uint8_t*>(W);
    if (pdl)
        return launch_ex((const void*)gemm_kernel<BN, KC, NW, ABF16, true>, grid, dim3(NW * 32), stream, true,
                         A, a_scale, w, partials, M, N, K);
    return launch_ex((const void*)gemm_kernel<BN, KC, NW, ABF16, false>, grid, dim3(NW * 32), stream, false,
                     A, a_scale, w, partials, M, N, K);
}

template <bool ABF16>
int dispatch_gemm(const void* A, const float* a_scale, const void* W, float* partials, int M, int N, int K,
                  int cfg, bool pdl, cudaStream_t stream) {
    switch (cfg) {
        case 0: return launch_gemm<128, 256, 4, ABF16>(A, a_scale, W, partials, M, N, K, pdl, stream);
        case 1: return launch_gemm<64, 256, 4, ABF16>(A, a_scale, W, partials, M, N, K, pdl, stream);
        case 2: return launch_gemm<128, 128, 4, ABF16>(A, a_scale, W, partials, M, N, K, pdl, stream);
        case 3: return launch_gemm<64, 128, 4, ABF16>(A, a_scale, W, partials, M, N, K, pdl, stream);
        case 4: return launch_gemm<128, 512, 4, ABF16>(A, a_scale, W, partials, M, N, K, pdl, stream);
        case 5: return launch_gemm<64, 512, 4, ABF16>(A, a_scale, W, partials, M, N, K, pdl, stream);
        case 6: return launch_gemm<128, 256, 8, ABF16>(A, a_scale, W, partials, M, N, K, pdl, stream);
        case 7: return launch_gemm<256, 256, 8, ABF16>(A, a_scale, W, partials, M, N, K, pdl, stream);
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
}

}  // namespace

int config_count() { return kNumCfgs; }

int config_k_chunk(int cfg) { return (cfg >= 0 && cfg < kNumCfgs) ? kCfgs[cfg].kc : 0; }

bool config_supports(int cfg, int N, int K) {
    return cfg >= 0 && cfg < kNumCfgs && N % kCfgs[cfg].bn == 0 && K % kCfgs[cfg].kc == 0;
}

int gemm(const void* A, const void* W, float* partials, int M, int N, int K, int cfg, bool pdl,
         cudaStream_t stream) {
    return dispatch_gemm<false>(A, nullptr, W, partials, M, N, K, cfg, pdl, stream);
}

int gemm_bf16_act(const __nv_bfloat16* A, const float* a_scale, const void* W, float* partials, int M, int N,
                  int K, int cfg, bool pdl, cudaStream_t stream) {
    return dispatch_gemm<true>(A, a_scale, W, partials, M, N, K, cfg, pdl, stream);
}

int sum_rope(const float* partials, int splits, const float* a_scale, const float* w_scale,
             const __nv_bfloat16* rope, __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
             const int* devpos, int rows, int q_dim, int k_dim, int v_dim, int head_dim, int sample_rows,
             long long kv_sample_stride, bool pdl, cudaStream_t stream) {
    if (sample_rows <= 0) sample_rows = rows;
    const int total = rows * ((q_dim + k_dim + v_dim) >> 1);
    const dim3 grid((total + 255) / 256);
    if (pdl)
        return launch_ex((const void*)sum_rope_kernel<true>, grid, dim3(256), stream, true, partials, splits,
                         a_scale, w_scale, rope, Q, K, V, devpos, rows, q_dim, k_dim, v_dim, head_dim,
                         sample_rows, kv_sample_stride);
    return launch_ex((const void*)sum_rope_kernel<false>, grid, dim3(256), stream, false, partials, splits,
                     a_scale, w_scale, rope, Q, K, V, devpos, rows, q_dim, k_dim, v_dim, head_dim,
                     sample_rows, kv_sample_stride);
}

int residual_ada_norm(const float* partials, int splits, const float* a_scale, const float* w_scale,
                      __nv_bfloat16* residual, const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                      const __nv_bfloat16* style, __nv_fp8_e4m3* out_fp8, __nv_bfloat16* out_bf16,
                      const float* out_scale, __nv_bfloat16* gate_out, int rows, int dim, float eps, bool pdl,
                      cudaStream_t stream) {
    if (dim != 1024) return static_cast<int>(cudaErrorInvalidValue);
    const dim3 grid(rows);
    const bool fp8 = out_fp8 != nullptr;
#define LAUNCH(F8, P)                                                                                 \
    return launch_ex((const void*)residual_ada_norm_kernel<1024, F8, P>, grid, dim3(256), stream, P,  \
                     partials, splits, a_scale, w_scale, residual, gate, weight, style, out_fp8,       \
                     out_bf16, out_scale, gate_out, rows, eps)
    if (fp8 && pdl) LAUNCH(true, true);
    if (fp8) LAUNCH(true, false);
    if (pdl) LAUNCH(false, true);
    LAUNCH(false, false);
#undef LAUNCH
}

int residual_gate_mul(const float* partials, int splits, const float* a_scale, const float* w_scale,
                      __nv_bfloat16* residual, const __nv_bfloat16* gate, int rows, int dim, bool pdl,
                      cudaStream_t stream) {
    const size_t elements = static_cast<size_t>(rows) * dim;
    const dim3 grid(static_cast<unsigned>((elements / 2 + 255) / 256));
    if (pdl)
        return launch_ex((const void*)residual_gate_mul_kernel<true>, grid, dim3(256), stream, true, partials,
                         splits, a_scale, w_scale, residual, gate, elements);
    return launch_ex((const void*)residual_gate_mul_kernel<false>, grid, dim3(256), stream, false, partials,
                     splits, a_scale, w_scale, residual, gate, elements);
}

int gate_gelu_fp8(const float* partials, int splits, const float* a_scale, const float* w_scale,
                  __nv_fp8_e4m3* out, int rows, int half, const float* out_scale, bool pdl,
                  cudaStream_t stream) {
    const int total = rows * (half >> 1);
    const dim3 grid((total + 255) / 256);
    if (pdl)
        return launch_ex((const void*)gate_gelu_fp8_kernel<true>, grid, dim3(256), stream, true, partials, splits,
                         a_scale, w_scale, out, rows, half, out_scale);
    return launch_ex((const void*)gate_gelu_fp8_kernel<false>, grid, dim3(256), stream, false, partials, splits,
                     a_scale, w_scale, out, rows, half, out_scale);
}


int attn_splits(int kv_len) { return (kv_len + attn_impl::KCH - 1) / attn_impl::KCH; }

size_t attn_scratch_floats(int splits, int samples, int heads) {
    return static_cast<size_t>(splits) * samples * heads * attn_impl::PART;
}

int attn(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O,
         int rows_per_sample, int samples, int heads, int q_row_stride, int kv_len, const int* seqused,
         long long kv_sample_stride, float scale, float* scratch, int* counters, bool pdl,
         cudaStream_t stream) {
    using namespace attn_impl;
    (void)counters;
    const int splits = attn_splits(kv_len);
    if (rows_per_sample < 1 || rows_per_sample > 16 || kv_len < 1 || heads < 1 || samples < 1 ||
        splits > MAX_SPLITS)
        return static_cast<int>(cudaErrorInvalidValue);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute((const void*)attn_split_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
        cudaFuncSetAttribute((const void*)attn_split_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
        attr_set = true;
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(splits, heads, samples);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = SMEM;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    const void* fn = pdl ? (const void*)attn_split_kernel<true> : (const void*)attn_split_kernel<false>;
    void* args[] = {(void*)&Q, (void*)&K, (void*)&V, (void*)&rows_per_sample, (void*)&heads,
                    (void*)&q_row_stride, (void*)&kv_len, (void*)&seqused, (void*)&kv_sample_stride,
                    (void*)&scale, (void*)&scratch, (void*)&splits};
    const int rc = static_cast<int>(cudaLaunchKernelExC(&cfg, fn, args));
    if (rc != 0) return rc;

    cudaLaunchConfig_t cfg2 = {};
    cfg2.gridDim = dim3(rows_per_sample, heads, samples);
    cfg2.blockDim = dim3(THREADS);
    cfg2.dynamicSmemBytes = 0;
    cfg2.stream = stream;
    cfg2.attrs = attr;
    cfg2.numAttrs = pdl ? 1 : 0;
    const void* fn2 = pdl ? (const void*)attn_combine_kernel<true> : (const void*)attn_combine_kernel<false>;
    void* args2[] = {(void*)&scratch, (void*)&O, (void*)&rows_per_sample, (void*)&heads, (void*)&q_row_stride,
                     (void*)&splits};
    return static_cast<int>(cudaLaunchKernelExC(&cfg2, fn2, args2));
}


int action_in_norm(const __nv_bfloat16* noise, const __nv_bfloat16* w_in, const __nv_bfloat16* b_in,
                   __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* style,
                   __nv_fp8_e4m3* out, __nv_bfloat16* gate_out, const float* out_scale, int rows,
                   float eps, bool pdl, cudaStream_t stream) {
    const dim3 grid(rows);
    if (pdl)
        return launch_ex((const void*)action_in_norm_kernel<true>, grid, dim3(256), stream, true, noise, w_in,
                         b_in, x, weight, style, out, gate_out, out_scale, eps);
    return launch_ex((const void*)action_in_norm_kernel<false>, grid, dim3(256), stream, false, noise, w_in,
                     b_in, x, weight, style, out, gate_out, out_scale, eps);
}

int action_out_residual(const __nv_bfloat16* x, const __nv_bfloat16* w_out, const __nv_bfloat16* b_out,
                        __nv_bfloat16* action, __nv_bfloat16* noise, __nv_bfloat16* trace_x,
                        __nv_bfloat16* trace_delta, int rows, bool pdl, cudaStream_t stream) {
    const dim3 grid(rows);
    if (pdl)
        return launch_ex((const void*)action_out_residual_kernel<true>, grid, dim3(256), stream, true, x, w_out,
                         b_out, action, noise, trace_x, trace_delta);
    return launch_ex((const void*)action_out_residual_kernel<false>, grid, dim3(256), stream, false, x, w_out,
                     b_out, action, noise, trace_x, trace_delta);
}

}  // namespace pi05_dec_skinny
}  // namespace flash_rt
