// SPDX-License-Identifier: Apache-2.0
// LaWAM row / elementwise / head kernels. See the header.
#include "lawam_kernels.cuh"

#include <curand_kernel.h>

#include <algorithm>

namespace flash_rt {
namespace lawam {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    return warp_sum(lane < nw ? red[lane] : 0.f);
}

__device__ __forceinline__ void load8(const __half* p, float* v) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 f = __half22float2(h[k]);
        v[2 * k] = f.x;
        v[2 * k + 1] = f.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float* v) {
    uint4 u;
    __half2* h = reinterpret_cast<__half2*>(&u);
#pragma unroll
    for (int k = 0; k < 4; ++k) h[k] = __floats2half2_rn(v[2 * k], v[2 * k + 1]);
    return u;
}

__device__ __forceinline__ uint4 pack8_bf16(const float* v) {
    uint4 u;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int k = 0; k < 4; ++k) h[k] = __floats2bfloat162_rn(v[2 * k], v[2 * k + 1]);
    return u;
}

// 8 values -> 8 e4m3 bytes (round to nearest, saturate to +-448).
__device__ __forceinline__ uint2 e4m3x8(const float* v, float inv) {
    uint32_t w[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const __nv_fp8x2_storage_t lo =
            __nv_cvt_float2_to_fp8x2(make_float2(v[4 * k] * inv, v[4 * k + 1] * inv), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi =
            __nv_cvt_float2_to_fp8x2(make_float2(v[4 * k + 2] * inv, v[4 * k + 3] * inv), __NV_SATFINITE, __NV_E4M3);
        w[k] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    return make_uint2(w[0], w[1]);
}

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// ── camera preprocessing ──

constexpr int kImg = 256, kPatch = 16, kGrid = kImg / kPatch;   // 16x16 patches per view

__global__ void preprocess_kernel(const uint8_t* __restrict__ frames, int num_views, __half* __restrict__ qwen,
                                  __half* __restrict__ dino) {
    // one thread per (view, patch, channel, patch row): 16 pixels
    const int total = num_views * kGrid * kGrid * 3 * kPatch;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int py = i % kPatch;
    const int c = (i / kPatch) % 3;
    const int p = (i / (kPatch * 3)) % (kGrid * kGrid);   // merge-order patch index
    const int view = i / (kPatch * 3 * kGrid * kGrid);
    // merge order: ((block_row * 8 + block_col) * 2 + sub_row) * 2 + sub_col
    const int sub_c = p & 1, sub_r = (p >> 1) & 1, blk = p >> 2;
    const int prow = (blk / (kGrid / 2)) * 2 + sub_r, pcol = (blk % (kGrid / 2)) * 2 + sub_c;
    const uint8_t* src = frames + (static_cast<size_t>(view) * kImg * kImg +
                                   static_cast<size_t>(prow * kPatch + py) * kImg + pcol * kPatch) * 3 + c;
    float q[kPatch];
#pragma unroll
    for (int x = 0; x < kPatch; ++x) q[x] = static_cast<float>(src[3 * x]) * (1.0f / 255.0f);
    __half* qo = qwen + (static_cast<size_t>(view) * kGrid * kGrid + p) * 768 + c * 256 + py * kPatch;
    float v[8];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int x = 0; x < 8; ++x) v[x] = (q[8 * h + x] - 0.5f) / 0.5f;
        *reinterpret_cast<uint4*>(qo + 8 * h) = pack8(v);
    }
    if (view == 0) {
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float stdv[3] = {0.229f, 0.224f, 0.225f};
        __half* d = dino + static_cast<size_t>(prow * kGrid + pcol) * 768 + c * 256 + py * kPatch;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int x = 0; x < 8; ++x) v[x] = (q[8 * h + x] - mean[c]) / stdv[c];
            *reinterpret_cast<uint4*>(d + 8 * h) = pack8(v);
        }
    }
}

// ── row norm family ──

__global__ void __launch_bounds__(1024)
gated_residual_norm_kernel(__half* __restrict__ x, const __half* __restrict__ delta,
                           const __half* __restrict__ delta_bias, const __half* __restrict__ gate,
                           const __half* __restrict__ delta2, int mode, float eps, const __half* __restrict__ weight,
                           const __half* __restrict__ bias, const __half* __restrict__ scale,
                           const __half* __restrict__ shift, __nv_fp8_e4m3* __restrict__ out_fp8,
                           const float* __restrict__ out_scale, __half* __restrict__ out_f16,
                           __nv_bfloat16* __restrict__ out_bf16, int dim) {
    __shared__ float red[32];
    const size_t off = static_cast<size_t>(blockIdx.x) * dim + threadIdx.x * 8;
    const int c0 = threadIdx.x * 8;
    float v[8], t[8];
    load8(x + off, v);
    if (delta != nullptr || delta_bias != nullptr || delta2 != nullptr) {
        if (delta != nullptr || delta_bias != nullptr) {
            float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if (delta != nullptr) {
                load8(delta + off, t);
#pragma unroll
                for (int k = 0; k < 8; ++k) d[k] += t[k];
            }
            if (delta_bias != nullptr) {
                load8(delta_bias + c0, t);
#pragma unroll
                for (int k = 0; k < 8; ++k) d[k] += t[k];
            }
            if (gate != nullptr) {
                load8(gate + c0, t);
#pragma unroll
                for (int k = 0; k < 8; ++k) d[k] *= t[k];
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] += d[k];
        }
        if (delta2 != nullptr) {
            load8(delta2 + off, t);
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] += t[k];
        }
        *reinterpret_cast<uint4*>(x + off) = pack8(v);
        load8(x + off, v);   // the norm reads the stored (FP16-rounded) residual row
    }
    if (out_fp8 == nullptr && out_f16 == nullptr && out_bf16 == nullptr) return;
    if (mode != 2) {
        if (mode == 0) {
            const float mean = block_sum(v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7], red) / dim;
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] -= mean;
        }
        float ss = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
        const float inv = rsqrtf(block_sum(ss, red) / dim + eps);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] *= inv;
    }
    if (weight != nullptr) {
        load8(weight + c0, t);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] *= t[k];
    }
    if (bias != nullptr) {
        load8(bias + c0, t);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] += t[k];
    }
    if (scale != nullptr) {
        float h[8];
        load8(scale + c0, t);
        load8(shift + c0, h);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = v[k] * (1.0f + t[k]) + h[k];
    }
    if (out_f16 != nullptr) *reinterpret_cast<uint4*>(out_f16 + off) = pack8(v);
    if (out_bf16 != nullptr) *reinterpret_cast<uint4*>(out_bf16 + off) = pack8_bf16(v);
    if (out_fp8 != nullptr)
        *reinterpret_cast<uint2*>(out_fp8 + off) = e4m3x8(v, 1.0f / fmaxf(*out_scale, 1e-12f));
}

__global__ void __launch_bounds__(256)
bias_gelu_kernel(const __half* __restrict__ x, int ld, const __half* __restrict__ bias, bool tanh_approx,
                     __nv_fp8_e4m3* __restrict__ out_fp8, const float* __restrict__ scale,
                     __half* __restrict__ out_f16, int rows, int n) {
    const int per_row = n / 8;
    const int total = rows * per_row;
    const float inv = out_fp8 != nullptr ? 1.0f / fmaxf(*scale, 1e-12f) : 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int r = i / per_row, c = (i - r * per_row) * 8;
        float v[8];
        load8(x + static_cast<size_t>(r) * ld + c, v);
        if (bias != nullptr) {
            float b[8];
            load8(bias + c, b);
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] += b[k];
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = tanh_approx ? gelu_tanh(v[k]) : gelu_erf(v[k]);
        const size_t o = static_cast<size_t>(r) * n + c;
        if (out_f16 != nullptr) *reinterpret_cast<uint4*>(out_f16 + o) = pack8(v);
        if (out_fp8 != nullptr) *reinterpret_cast<uint2*>(out_fp8 + o) = e4m3x8(v, inv);
    }
}

__global__ void __launch_bounds__(256)
silu_mul_kernel(const __half* __restrict__ gate, const __half* __restrict__ up, int ld, __half* __restrict__ out,
                int rows, int n) {
    const int per_row = n / 8;
    const int total = rows * per_row;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int r = i / per_row, c = (i - r * per_row) * 8;
        float g[8], u[8];
        load8(gate + static_cast<size_t>(r) * ld + c, g);
        load8(up + static_cast<size_t>(r) * ld + c, u);
#pragma unroll
        for (int k = 0; k < 8; ++k) g[k] = silu(g[k]) * u[k];
        *reinterpret_cast<uint4*>(out + static_cast<size_t>(r) * n + c) = pack8(g);
    }
}

__global__ void scatter_rows_kernel(const __half* __restrict__ src, const __half* __restrict__ bias,
                                    const int* __restrict__ idx, __half* __restrict__ dst, int n, int dim) {
    const int per_row = dim / 8;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * per_row) return;
    const int r = i / per_row, c = (i - r * per_row) * 8;
    float v[8];
    load8(src + static_cast<size_t>(r) * dim + c, v);
    if (bias != nullptr) {
        float b[8];
        load8(bias + c, b);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] += b[k];
    }
    *reinterpret_cast<uint4*>(dst + static_cast<size_t>(idx[r]) * dim + c) = pack8(v);
}

// ── VLM -> latent action QFormer ──

// K / V projections of the normalized act-query rows. One warp per act row,
// kQfOutsPerCta outputs per CTA. Lane l owns columns l + 32 k, and the row
// statistics and dot products are summed in the order of a 256-thread CTA per
// row (thread t: columns t + 256 j, then the two-level warp / CTA tree).
constexpr int kQfVdim = 2048, kQfOutsPerCta = 2;

__global__ void __launch_bounds__(256)
qformer_kv_kernel(const __half* __restrict__ vlm, const int* __restrict__ idx, int nq, int code, QFormerWeights w,
                  float* __restrict__ kv) {
    constexpr int V = kQfVdim, PER = V / 32;
    const int lane = threadIdx.x & 31, r = threadIdx.x >> 5;
    if (r >= nq) return;
    const __half* src = vlm + static_cast<size_t>(idx[r]) * V;
    float x[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) x[k] = __half2float(src[lane + 32 * k]);
    // reduction of the 256-thread form: thread (warp g, lane l) sums columns l + 32 g + 256 j
    auto tree = [&](const float* part) {
        float red = 0.f;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const float t = warp_sum(part[g]);
            if (lane == g) red = t;
        }
        return warp_sum(red);
    };
    float part[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < PER / 8; ++j) s += x[g + 8 * j];
        part[g] = s;
    }
    const float mean = tree(part) / V;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < PER / 8; ++j) {
            const float d = x[g + 8 * j] - mean;
            ss += d * d;
        }
        part[g] = ss;
    }
    const float inv = rsqrtf(tree(part) / V + 1e-5f);
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int c = lane + 32 * k;
        x[k] = (x[k] - mean) * inv * w.norm_kv_w[c] + w.norm_kv_b[c];
    }
    for (int i = 0; i < kQfOutsPerCta; ++i) {
        const int o = blockIdx.x * kQfOutsPerCta + i;
        if (o >= 2 * code) break;
        const float* wr = (o < code ? w.wk + static_cast<size_t>(o) * V : w.wv + static_cast<size_t>(o - code) * V);
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < PER; ++k) acc += wr[lane + 32 * k] * x[k];
        acc = warp_sum(acc);
        if (lane == 0) kv[r * 2 * code + o] = acc + w.in_bias[code + o];
    }
}

__device__ float small_layer_norm(const float* x, int i, int n, const float* g, const float* b, float* red) {
    // all threads of the CTA participate; thread i < n owns x[i]
    const float v = i < n ? x[i] : 0.f;
    const float mean = block_sum(v, red) / n;
    const float d = i < n ? v - mean : 0.f;
    const float inv = rsqrtf(block_sum(d * d, red) / n + 1e-5f);
    return i < n ? d * inv * g[i] + b[i] : 0.f;
}

// Query path, attention over the act rows, FFN, final norm and the LAM
// conditioning projection: one CTA, one thread per output of each projection
// (transposed weights: the threads of a warp read consecutive addresses).
__global__ void __launch_bounds__(1024)
qformer_head_kernel(const float* __restrict__ kv, int nq, int code, int heads, int ctx, QFormerWeights w,
                    float* __restrict__ latent, __half* __restrict__ cond_silu) {
    __shared__ float red[32];
    __shared__ float qn[64], q[64], o[64], h[64], f[256], p[8 * 64];
    const int t = threadIdx.x;
    const int hd = code / heads;
    // q = Wq · LN(query) + bq, pre-scaled
    float v = small_layer_norm(w.query, t, code, w.norm_q_w, w.norm_q_b, red);
    if (t < code) qn[t] = v;
    __syncthreads();
    if (t < code) {
        float acc = w.in_bias[t];
        for (int c = 0; c < code; ++c) acc += w.wq_t[c * code + t] * qn[c];
        q[t] = acc * rsqrtf(static_cast<float>(hd));
    }
    __syncthreads();
    // scores [heads, nq] -> softmax over the act rows
    if (t < heads * nq) {
        const int hh = t / nq, k = t % nq;
        float s = 0.f;
        for (int d = 0; d < hd; ++d) s += q[hh * hd + d] * kv[k * 2 * code + hh * hd + d];
        p[t] = s;
    }
    __syncthreads();
    if (t < heads) {
        float m = -1e30f;
        for (int k = 0; k < nq; ++k) m = fmaxf(m, p[t * nq + k]);
        float z = 0.f;
        for (int k = 0; k < nq; ++k) {
            const float e = expf(p[t * nq + k] - m);
            p[t * nq + k] = e;
            z += e;
        }
        for (int k = 0; k < nq; ++k) p[t * nq + k] /= z;
    }
    __syncthreads();
    if (t < code) {
        const int hh = t / hd;
        float acc = 0.f;
        for (int k = 0; k < nq; ++k) acc += p[hh * nq + k] * kv[k * 2 * code + code + t];
        o[t] = acc;
    }
    __syncthreads();
    if (t < code) {
        float acc = w.bo[t];
        for (int c = 0; c < code; ++c) acc += w.wo_t[c * code + t] * o[c];
        h[t] = w.query[t] + acc;   // queries + attention
    }
    __syncthreads();
    v = small_layer_norm(h, t, code, w.ffn_ln_w, w.ffn_ln_b, red);
    if (t < code) qn[t] = v;
    __syncthreads();
    const int hid = 4 * code;
    if (t < hid) {
        float acc = w.b1[t];
        for (int c = 0; c < code; ++c) acc += w.w1_t[c * hid + t] * qn[c];
        f[t] = gelu_erf(acc);
    }
    __syncthreads();
    if (t < code) {
        float acc = w.b2[t];
        for (int c = 0; c < hid; ++c) acc += w.w2_t[c * code + t] * f[c];
        h[t] += acc;
    }
    __syncthreads();
    v = small_layer_norm(h, t, code, w.final_w, w.final_b, red);
    if (t < code) {
        qn[t] = v;
        latent[t] = v;
    }
    __syncthreads();
    for (int j = t; j < ctx; j += blockDim.x) {
        float acc = w.act_b[j];
        for (int c = 0; c < code; ++c) acc += w.act_w_t[c * ctx + j] * qn[c];
        cond_silu[j] = __float2half(silu(acc));
    }
}

__global__ void action_encode_kernel(const float* __restrict__ x, const float* __restrict__ w1,
                                     const float* __restrict__ b1, __nv_bfloat16* __restrict__ out, int rows, int n,
                                     int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * h) return;
    const int r = i / h, j = i % h;
    float acc = b1[j];
    for (int k = 0; k < n; ++k) acc += x[r * n + k] * w1[k * h + j];
    out[i] = __float2bfloat16(silu(acc));
}

__global__ void gaussian_noise_kernel(float* __restrict__ x, unsigned long long* __restrict__ seed, int count) {
    const unsigned long long s = *seed;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        curandStatePhilox4_32_10_t st;
        curand_init(s, static_cast<unsigned long long>(i), 0ull, &st);
        x[i] = curand_normal(&st);
    }
    __syncthreads();
    if (threadIdx.x == 0) *seed = s + 1;
}

__global__ void unnormalize_kernel(const float* __restrict__ x, int ld, const float* __restrict__ lo,
                                   const float* __restrict__ hi, float* __restrict__ out, int rows, int out_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * out_dim) return;
    const int r = i / out_dim, c = i % out_dim;
    out[i] = (x[r * ld + c] + 1.0f) * 0.5f * (hi[c] - lo[c]) + lo[c];
}

// One warp per output row; every lane streams 16 INT8 weights per 512-wide
// K chunk with one 16-byte load, x staged in shared memory as FP32.
__global__ void __launch_bounds__(256)
gemv_i8w_kernel(const __half* __restrict__ x, const int8_t* __restrict__ w, const float* __restrict__ w_scale,
                const __half* __restrict__ bias, __half* __restrict__ out, int N, int K) {
    extern __shared__ float xs[];
    for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = __half2float(x[k]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const int8_t* wr = w + static_cast<size_t>(row) * K;
    float acc = 0.f;
    for (int k0 = lane * 16; k0 < K; k0 += 512) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(wr + k0));
        const int8_t* b = reinterpret_cast<const int8_t*>(&u);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc += static_cast<float>(b[i]) * xs[k0 + i];
    }
    acc = warp_sum(acc);
    if (lane == 0) out[row] = __float2half(acc * w_scale[row] + (bias != nullptr ? __half2float(bias[row]) : 0.f));
}

// NVFP4-weight GEMV, K = 1024: lane l always owns k in [32 l, 32 l + 32) (one
// 16-byte load of E2M1 pairs + its two UE4M3 block scales per row), keeps that
// slice of x in registers and walks rows with four loads in flight; E2M1 pairs
// convert in hardware (cvt.rn.f16x2.e2m1x2), products accumulate in FP32.
constexpr int kNvfp4GemvK = 1024;
constexpr int kNvfp4GemvRows = 4;

__device__ __forceinline__ float2 e2m1x2_to_float2(uint32_t byte) {
    uint32_t h2;
    asm("{\n .reg .b8 b;\n cvt.u8.u32 b, %1;\n cvt.rn.f16x2.e2m1x2 %0, b;\n}\n" : "=r"(h2) : "r"(byte));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

__global__ void __launch_bounds__(256)
gemv_nvfp4w_kernel(const __half* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sf,
                   const float* __restrict__ gs, const __half* __restrict__ bias, __half* __restrict__ out, int N) {
    constexpr int K = kNvfp4GemvK;
    const int lane = threadIdx.x & 31;
    float xr[32];
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(x + lane * 32 + i));
        const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h[j]);
            xr[i + 2 * j] = f.x;
            xr[i + 2 * j + 1] = f.y;
        }
    }
    const float g = *gs;
    const int warps = gridDim.x * (blockDim.x >> 5);
    for (int r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * kNvfp4GemvRows; r0 < N;
         r0 += warps * kNvfp4GemvRows) {
        uint4 u[kNvfp4GemvRows];
        uint32_t s2[kNvfp4GemvRows];
#pragma unroll
        for (int r = 0; r < kNvfp4GemvRows; ++r) {
            const int row = min(r0 + r, N - 1);
            u[r] = __ldg(reinterpret_cast<const uint4*>(w + static_cast<size_t>(row) * (K / 2) + lane * 16));
            s2[r] = __ldg(reinterpret_cast<const uint16_t*>(sf + static_cast<size_t>(row) * (K / 16) + lane * 2));
        }
#pragma unroll
        for (int r = 0; r < kNvfp4GemvRows; ++r) {
            const uint32_t wv[4] = {u[r].x, u[r].y, u[r].z, u[r].w};
            float acc = 0.f;
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                float a = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float2 v = e2m1x2_to_float2((wv[blk * 2 + i / 4] >> (8 * (i % 4))) & 0xffu);
                    a = fmaf(v.x, xr[blk * 16 + 2 * i], a);
                    a = fmaf(v.y, xr[blk * 16 + 2 * i + 1], a);
                }
                const uint8_t sb = static_cast<uint8_t>(s2[r] >> (8 * blk));
                acc = fmaf(a, static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&sb)), acc);
            }
            acc = warp_sum(acc);
            const int row = r0 + r;
            if (lane == 0 && row < N)
                out[row] = __float2half(acc * g + (bias != nullptr ? __half2float(bias[row]) : 0.f));
        }
    }
}

// Q / K / V split of a fused qkv GEMM output with bias, K / V scale and
// rotate-half RoPE on Q, K (head dim 64): the arithmetic of the FlashRT
// fp8_prefill qkv_split (without its QK norm), each thread owning 4
// consecutive dims of the first half and their partners (8-byte accesses).
template <bool BF16>
__device__ __forceinline__ void store4(void* base, size_t i, const float* v) {
    if constexpr (BF16) {
        const __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(base) + i) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
    } else {
        const __half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        *reinterpret_cast<uint2*>(reinterpret_cast<__half*>(base) + i) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
    }
}

__device__ __forceinline__ void load4(const __half* p, float* v) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

template <bool BF16>
__global__ void __launch_bounds__(256)
qkv_rope_split_kernel(const __half* __restrict__ qkv, int ld, const __half* __restrict__ q_bias,
                      const __half* __restrict__ k_bias, const __half* __restrict__ v_bias, float k_mul, float v_mul,
                      const __half* __restrict__ cos, const __half* __restrict__ sin, void* q_out, void* k_out,
                      void* v_out, int n_q, int n_k) {
    constexpr int HD = 64, H2 = 32;
    const int s = blockIdx.x;
    const int head = blockIdx.y * 32 + (threadIdx.x >> 3);
    const int d0 = (threadIdx.x & 7) * 4;
    if (head >= n_q + 2 * n_k) return;
    const __half* bias;
    void* dst;
    float mul;
    bool rope;
    int hh, orow;
    if (head < n_q) {
        hh = head; bias = q_bias; dst = q_out; mul = 1.f; rope = true; orow = n_q;
    } else if (head < n_q + n_k) {
        hh = head - n_q; bias = k_bias; dst = k_out; mul = k_mul; rope = true; orow = n_k;
    } else {
        hh = head - n_q - n_k; bias = v_bias; dst = v_out; mul = v_mul; rope = false; orow = n_k;
    }
    rope = rope && cos != nullptr;
    const __half* p = qkv + static_cast<size_t>(s) * ld + head * HD;
    float lo[4], hi[4];
    load4(p + d0, lo);
    load4(p + d0 + H2, hi);
    if (bias != nullptr) {
        float bl[4], bh[4];
        load4(bias + hh * HD + d0, bl);
        load4(bias + hh * HD + d0 + H2, bh);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            lo[e] += bl[e];
            hi[e] += bh[e];
        }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        lo[e] *= mul;
        hi[e] *= mul;
    }
    if (rope) {
        float c[4], sn[4];
        load4(cos + static_cast<size_t>(s) * HD + d0, c);
        load4(sin + static_cast<size_t>(s) * HD + d0, sn);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float a = lo[e], b = hi[e];
            lo[e] = fmaf(a, c[e], -__fmul_rn(b, sn[e]));   // the contraction of the FlashRT kernel
            hi[e] = fmaf(b, c[e], __fmul_rn(a, sn[e]));
        }
    }
    const size_t o = (static_cast<size_t>(s) * orow + hh) * HD + d0;
    store4<BF16>(dst, o, lo);
    store4<BF16>(dst, o + H2, hi);
}

inline int last_error() { return static_cast<int>(cudaGetLastError()); }

}  // namespace

int preprocess_u8(const uint8_t* frames, int num_views, __half* qwen, __half* dino, cudaStream_t stream) {
    if (num_views < 1 || frames == nullptr || qwen == nullptr || dino == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    const int total = num_views * kGrid * kGrid * 3 * kPatch;
    preprocess_kernel<<<(total + 255) / 256, 256, 0, stream>>>(frames, num_views, qwen, dino);
    return last_error();
}

int gated_residual_norm(__half* x, const __half* delta, const __half* delta_bias, const __half* gate,
                        const __half* delta2, int mode, float eps, const __half* weight, const __half* bias,
                        const __half* scale, const __half* shift, __nv_fp8_e4m3* out_fp8, const float* out_scale,
                        __half* out_f16, __nv_bfloat16* out_bf16, int rows, int dim, cudaStream_t stream) {
    if (rows < 1 || dim % 8 != 0 || dim > 8192 || mode < 0 || mode > 2 || ((scale == nullptr) != (shift == nullptr)) ||
        (out_fp8 != nullptr && out_scale == nullptr))
        return static_cast<int>(cudaErrorInvalidValue);
    gated_residual_norm_kernel<<<rows, dim / 8, 0, stream>>>(x, delta, delta_bias, gate, delta2, mode, eps, weight,
                                                             bias, scale, shift, out_fp8, out_scale, out_f16,
                                                             out_bf16, dim);
    return last_error();
}

int bias_gelu(const __half* x, int ld, const __half* bias, bool tanh_approx, __nv_fp8_e4m3* out_fp8,
              const float* scale, __half* out_f16, int rows, int n, cudaStream_t stream) {
    if (rows < 1 || n % 8 != 0 || ld % 8 != 0 || (out_fp8 != nullptr && scale == nullptr))
        return static_cast<int>(cudaErrorInvalidValue);
    const int total = rows * (n / 8);
    const int blocks = (total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096;
    bias_gelu_kernel<<<blocks, 256, 0, stream>>>(x, ld, bias, tanh_approx, out_fp8, scale, out_f16, rows, n);
    return last_error();
}

int silu_mul(const __half* gate, const __half* up, int ld, __half* out, int rows, int n, cudaStream_t stream) {
    if (rows < 1 || n % 8 != 0 || ld % 8 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const int total = rows * (n / 8);
    const int blocks = (total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096;
    silu_mul_kernel<<<blocks, 256, 0, stream>>>(gate, up, ld, out, rows, n);
    return last_error();
}

int scatter_rows(const __half* src, const __half* bias, const int* idx, __half* dst, int n, int dim,
                 cudaStream_t stream) {
    if (n < 1 || dim % 8 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const int total = n * (dim / 8);
    scatter_rows_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, bias, idx, dst, n, dim);
    return last_error();
}

int qformer_latent(const __half* vlm, const int* idx, int nq, int vdim, int code, int heads, int ctx,
                   const QFormerWeights& w, float* scratch, float* latent, __half* cond_silu, cudaStream_t stream) {
    if (nq < 1 || nq > 8 || code > 64 || code % heads != 0 || 4 * code > 256 || vdim != kQfVdim)
        return static_cast<int>(cudaErrorInvalidValue);
    qformer_kv_kernel<<<(2 * code + kQfOutsPerCta - 1) / kQfOutsPerCta, 256, 0, stream>>>(vlm, idx, nq, code, w,
                                                                                        scratch);
    qformer_head_kernel<<<1, 1024, 0, stream>>>(scratch, nq, code, heads, ctx, w, latent, cond_silu);
    return last_error();
}

int action_encode(const float* x, const float* w1, const float* b1, __nv_bfloat16* out, int rows, int n, int h,
                  cudaStream_t stream) {
    if (rows < 1 || n < 1 || h < 1) return static_cast<int>(cudaErrorInvalidValue);
    action_encode_kernel<<<(rows * h + 255) / 256, 256, 0, stream>>>(x, w1, b1, out, rows, n, h);
    return last_error();
}

int gaussian_noise(float* x, unsigned long long* seed, int count, cudaStream_t stream) {
    if (count < 1) return static_cast<int>(cudaErrorInvalidValue);
    gaussian_noise_kernel<<<1, 1024, 0, stream>>>(x, seed, count);
    return last_error();
}

int gemv_i8w(const __half* x, const int8_t* w, const float* w_scale, const __half* bias, __half* out, int N, int K,
             cudaStream_t stream) {
    if (N < 1 || K % 512 != 0 || K > 12288) return static_cast<int>(cudaErrorInvalidValue);
    gemv_i8w_kernel<<<(N + 7) / 8, 256, K * sizeof(float), stream>>>(x, w, w_scale, bias, out, N, K);
    return last_error();
}

int gemv_nvfp4w(const __half* x, const uint8_t* w, const uint8_t* sf, const float* gs, const __half* bias,
                __half* out, int N, int K, cudaStream_t stream, int max_ctas) {
    if (N < 1 || K != kNvfp4GemvK || x == nullptr || w == nullptr || sf == nullptr || gs == nullptr ||
        out == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int rows_per_cta = 8 * kNvfp4GemvRows;
    int blocks = std::min((N + rows_per_cta - 1) / rows_per_cta, sms * 8);
    if (max_ctas > 0) blocks = std::min(blocks, max_ctas);
    // the shared-memory carveout of the GEMMs this GEMV runs next to: an SM configured for
    // less shared memory cannot take their CTAs until it drains
    static const cudaError_t carve = cudaFuncSetAttribute(
        gemv_nvfp4w_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    if (carve != cudaSuccess) return static_cast<int>(carve);
    gemv_nvfp4w_kernel<<<blocks, 256, 0, stream>>>(x, w, sf, gs, bias, out, N);
    return last_error();
}

// Rotate-half RoPE (head dim 64) in place on the first n_qk heads (Q | K) of a fused [S, ld] QKV
// buffer whose bias the GEMM epilogue added: attention reads Q / K / V from it through row strides.
// Four threads per head (8 columns of each half, 16-byte accesses), 256 / (4 n_qk) rows per block.
__global__ void __launch_bounds__(256)
qk_rope_inplace_kernel(__half* __restrict__ qkv, int ld, const __half* __restrict__ cos,
                       const __half* __restrict__ sin, int S, int n_qk) {
    constexpr int HD = 64, H2 = 32;
    const int per_row = 4 * n_qk, rows = blockDim.x / per_row;
    const int s = blockIdx.x * rows + threadIdx.x / per_row;
    const int t = threadIdx.x % per_row, head = t >> 2, d0 = (t & 3) * 8;
    if (s >= S || threadIdx.x >= rows * per_row) return;
    __half* p = qkv + static_cast<size_t>(s) * ld + head * HD;
    const uint4 ul = *reinterpret_cast<const uint4*>(p + d0), uh = *reinterpret_cast<const uint4*>(p + d0 + H2);
    const uint4 uc = __ldg(reinterpret_cast<const uint4*>(cos + static_cast<size_t>(s) * HD + d0));
    const uint4 us = __ldg(reinterpret_cast<const uint4*>(sin + static_cast<size_t>(s) * HD + d0));
    const __half2* hl = reinterpret_cast<const __half2*>(&ul);
    const __half2* hh = reinterpret_cast<const __half2*>(&uh);
    const __half2* hc = reinterpret_cast<const __half2*>(&uc);
    const __half2* hs = reinterpret_cast<const __half2*>(&us);
    uint4 ol, oh;
    __half2* pl = reinterpret_cast<__half2*>(&ol);
    __half2* ph = reinterpret_cast<__half2*>(&oh);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 a = __half22float2(hl[e]), b = __half22float2(hh[e]);
        const float2 c = __half22float2(hc[e]), sn = __half22float2(hs[e]);
        pl[e] = __floats2half2_rn(fmaf(a.x, c.x, -__fmul_rn(b.x, sn.x)), fmaf(a.y, c.y, -__fmul_rn(b.y, sn.y)));
        ph[e] = __floats2half2_rn(fmaf(b.x, c.x, __fmul_rn(a.x, sn.x)), fmaf(b.y, c.y, __fmul_rn(a.y, sn.y)));
    }
    *reinterpret_cast<uint4*>(p + d0) = ol;
    *reinterpret_cast<uint4*>(p + d0 + H2) = oh;
}

int qk_rope_inplace(__half* qkv, int ld, const __half* cos, const __half* sin, int S, int n_qk, int hd,
                    cudaStream_t stream) {
    if (qkv == nullptr || cos == nullptr || sin == nullptr || S < 1 || hd != 64 || ld % 4 != 0 || n_qk < 1)
        return static_cast<int>(cudaErrorInvalidValue);
    if (4 * n_qk > 256) return static_cast<int>(cudaErrorInvalidValue);
    const int rows = 256 / (4 * n_qk);
    qk_rope_inplace_kernel<<<(S + rows - 1) / rows, 256, 0, stream>>>(qkv, ld, cos, sin, S, n_qk);
    return last_error();
}

int qkv_rope_split(const __half* qkv, int ld, const __half* q_bias, const __half* k_bias, const __half* v_bias,
                   float k_mul, float v_mul, const __half* cos, const __half* sin, void* q_out, void* k_out,
                   void* v_out, bool out_bf16, int S, int n_q, int n_k, int hd, cudaStream_t stream) {
    if (qkv == nullptr || q_out == nullptr || k_out == nullptr || v_out == nullptr || S < 1 || hd != 64 ||
        ld % 4 != 0 || (cos == nullptr) != (sin == nullptr))
        return static_cast<int>(cudaErrorInvalidValue);
    const dim3 grid(S, (n_q + 2 * n_k + 31) / 32);
    if (out_bf16)
        qkv_rope_split_kernel<true><<<grid, 256, 0, stream>>>(qkv, ld, q_bias, k_bias, v_bias, k_mul, v_mul, cos,
                                                              sin, q_out, k_out, v_out, n_q, n_k);
    else
        qkv_rope_split_kernel<false><<<grid, 256, 0, stream>>>(qkv, ld, q_bias, k_bias, v_bias, k_mul, v_mul, cos,
                                                               sin, q_out, k_out, v_out, n_q, n_k);
    return last_error();
}

int unnormalize_actions(const float* x, int ld, const float* lo, const float* hi, float* actions, int rows,
                        int out_dim, cudaStream_t stream) {
    if (rows < 1 || out_dim < 1 || out_dim > ld) return static_cast<int>(cudaErrorInvalidValue);
    unnormalize_kernel<<<(rows * out_dim + 255) / 256, 256, 0, stream>>>(x, ld, lo, hi, actions, rows, out_dim);
    return last_error();
}

}  // namespace lawam
}  // namespace flash_rt
