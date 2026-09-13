// SPDX-License-Identifier: Apache-2.0
// Skinny FP8 GEMM family for small-row action decoders on sm_120a. See the header.
#include "decoder_skinny_fp8_sm120.cuh"

#include <cstddef>
#include <cstdint>

namespace flash_rt {
namespace dec_skinny {
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
    if constexpr (PDL) pdl_prologue();
    __shared__ float warp_partial[8];
    const int row = blockIdx.x;
    const size_t base = static_cast<size_t>(row) * DIM;
    const size_t elements = static_cast<size_t>(rows) * DIM;
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
    const __nv_bfloat16* style_row = style + static_cast<size_t>(row) * 3 * DIM;
    const __nv_bfloat162* sc2 = reinterpret_cast<const __nv_bfloat162*>(style_row);
    const __nv_bfloat162* sh2 = reinterpret_cast<const __nv_bfloat162*>(style_row + DIM);
    const __nv_bfloat162* gt2 = reinterpret_cast<const __nv_bfloat162*>(style_row + 2 * DIM);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    const float inv_scale = OUT_FP8 ? 1.0f / (*out_scale) : 1.0f;
#pragma unroll
    for (int p = 0; p < PPT; ++p) {
        const int i = threadIdx.x + p * 256;
        const float2 rv = __bfloat1622float2(kept[p]);
        const float2 wv = __bfloat1622float2(w2[i]);
        const float2 sv = __bfloat1622float2(sc2[i]);
        const float2 hv = __bfloat1622float2(sh2[i]);
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
        reinterpret_cast<__nv_bfloat162*>(gate_out + base)[i] = gt2[i];
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

}  // namespace dec_skinny
}  // namespace flash_rt
