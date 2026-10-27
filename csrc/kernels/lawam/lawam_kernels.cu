// SPDX-License-Identifier: Apache-2.0
// LaWAM row / elementwise / head kernels. See the header.
#include "lawam_kernels.cuh"

#include <curand_kernel.h>

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

__global__ void cast_kernel(const __half* __restrict__ in, __nv_bfloat16* __restrict__ out, int n8) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    float v[8];
    load8(in + static_cast<size_t>(i) * 8, v);
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(i) * 8) = pack8_bf16(v);
}

// ── VLM -> latent action QFormer ──

// K / V projections of the normalized act-query rows: one CTA per row.
__global__ void __launch_bounds__(256)
qformer_kv_kernel(const __half* __restrict__ vlm, const int* __restrict__ idx, int vdim, int code,
                  QFormerWeights w, float* __restrict__ kv) {
    extern __shared__ float row[];
    __shared__ float red[32];
    const __half* src = vlm + static_cast<size_t>(idx[blockIdx.x]) * vdim;
    float s = 0.f;
    for (int c = threadIdx.x; c < vdim; c += blockDim.x) {
        const float v = __half2float(src[c]);
        row[c] = v;
        s += v;
    }
    const float mean = block_sum(s, red) / vdim;
    float ss = 0.f;
    for (int c = threadIdx.x; c < vdim; c += blockDim.x) {
        const float d = row[c] - mean;
        ss += d * d;
    }
    const float inv = rsqrtf(block_sum(ss, red) / vdim + 1e-5f);
    for (int c = threadIdx.x; c < vdim; c += blockDim.x)
        row[c] = (row[c] - mean) * inv * w.norm_kv_w[c] + w.norm_kv_b[c];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int o = warp; o < 2 * code; o += nw) {
        const float* wr = (o < code ? w.wk + static_cast<size_t>(o) * vdim : w.wv + static_cast<size_t>(o - code) * vdim);
        float acc = 0.f;
        for (int c = lane; c < vdim; c += 32) acc += wr[c] * row[c];
        acc = warp_sum(acc);
        if (lane == 0) kv[blockIdx.x * 2 * code + o] = acc + w.in_bias[code + o];
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
// conditioning projection: one CTA.
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
        for (int c = 0; c < code; ++c) acc += w.wq[t * code + c] * qn[c];
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
        for (int c = 0; c < code; ++c) acc += w.wo[t * code + c] * o[c];
        h[t] = w.query[t] + acc;   // queries + attention
    }
    __syncthreads();
    v = small_layer_norm(h, t, code, w.ffn_ln_w, w.ffn_ln_b, red);
    if (t < code) qn[t] = v;
    __syncthreads();
    const int hid = 4 * code;
    if (t < hid) {
        float acc = w.b1[t];
        for (int c = 0; c < code; ++c) acc += w.w1[t * code + c] * qn[c];
        f[t] = gelu_erf(acc);
    }
    __syncthreads();
    if (t < code) {
        float acc = w.b2[t];
        for (int c = 0; c < hid; ++c) acc += w.w2[t * hid + c] * f[c];
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
        for (int c = 0; c < code; ++c) acc += w.act_w[j * code + c] * qn[c];
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

int cast_f16_bf16(const __half* in, __nv_bfloat16* out, int n, cudaStream_t stream) {
    if (n < 8 || n % 8 != 0) return static_cast<int>(cudaErrorInvalidValue);
    cast_kernel<<<(n / 8 + 255) / 256, 256, 0, stream>>>(in, out, n / 8);
    return last_error();
}

int qformer_latent(const __half* vlm, const int* idx, int nq, int vdim, int code, int heads, int ctx,
                   const QFormerWeights& w, float* scratch, float* latent, __half* cond_silu, cudaStream_t stream) {
    if (nq < 1 || nq > 8 || code > 64 || code % heads != 0 || 4 * code > 256 || vdim % 8 != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    qformer_kv_kernel<<<nq, 256, vdim * sizeof(float), stream>>>(vlm, idx, vdim, code, w, scratch);
    qformer_head_kernel<<<1, 256, 0, stream>>>(scratch, nq, code, heads, ctx, w, latent, cond_silu);
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

int unnormalize_actions(const float* x, int ld, const float* lo, const float* hi, float* actions, int rows,
                        int out_dim, cudaStream_t stream) {
    if (rows < 1 || out_dim < 1 || out_dim > ld) return static_cast<int>(cudaErrorInvalidValue);
    unnormalize_kernel<<<(rows * out_dim + 255) / 256, 256, 0, stream>>>(x, ld, lo, hi, actions, rows, out_dim);
    return last_error();
}

}  // namespace lawam
}  // namespace flash_rt
