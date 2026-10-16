// SPDX-License-Identifier: Apache-2.0
// Fused FP8 prefill row kernels on an FP16 residual stream. See the header.
#include "fp8_prefill_fp16.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace flash_rt {
namespace fp8_prefill {
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

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void __launch_bounds__(1024)
residual_norm_quant_kernel(__half* __restrict__ x, const __half* __restrict__ delta,
                           const __half* __restrict__ delta_bias, const __half* __restrict__ delta2,
                           __half* __restrict__ x_copy, const __half* __restrict__ weight,
                           const __half* __restrict__ bias, int mode, float eps, __nv_fp8_e4m3* __restrict__ out_fp8,
                           const float* __restrict__ scale, __half* __restrict__ out_f16, int dim) {
    __shared__ float red[32];
    const size_t off = static_cast<size_t>(blockIdx.x) * dim + threadIdx.x * 8;
    const int c0 = threadIdx.x * 8;
    float v[8];
    load8(x + off, v);
    if (delta != nullptr || delta_bias != nullptr || delta2 != nullptr) {
        float t[8];
        if (delta != nullptr) {
            load8(delta + off, t);
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] += t[k];
        }
        if (delta_bias != nullptr) {
            load8(delta_bias + c0, t);
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] += t[k];
        }
        if (delta2 != nullptr) {
            load8(delta2 + off, t);
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] += t[k];
        }
        const uint4 u = pack8(v);
        *reinterpret_cast<uint4*>(x + off) = u;
        // the norm reads the stored (FP16-rounded) residual row
        load8(x + off, v);
    }
    if (x_copy != nullptr) *reinterpret_cast<uint4*>(x_copy + off) = *reinterpret_cast<const uint4*>(x + off);
    if (out_fp8 == nullptr && out_f16 == nullptr) return;

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
    float t[8];
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
    if (out_f16 != nullptr) *reinterpret_cast<uint4*>(out_f16 + off) = pack8(v);
    if (out_fp8 != nullptr)
        *reinterpret_cast<uint2*>(out_fp8 + off) = e4m3x8(v, 1.0f / fmaxf(*scale, 1e-12f));
}

__global__ void __launch_bounds__(256)
bias_gelu_quant_kernel(const __half* __restrict__ x, int ld, const __half* __restrict__ bias,
                       __nv_fp8_e4m3* __restrict__ out, const float* __restrict__ scale, int rows, int n) {
    const int per_row = n / 8;
    const int total = rows * per_row;
    const float inv = 1.0f / fmaxf(*scale, 1e-12f);
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
        for (int k = 0; k < 8; ++k) v[k] = gelu_tanh(v[k]);
        *reinterpret_cast<uint2*>(out + static_cast<size_t>(r) * n + c) = e4m3x8(v, inv);
    }
}

template <bool BF16>
__device__ __forceinline__ void store_val(void* base, size_t i, float v) {
    if constexpr (BF16)
        reinterpret_cast<__nv_bfloat16*>(base)[i] = __float2bfloat16(v);
    else
        reinterpret_cast<__half*>(base)[i] = __float2half(v);
}

template <bool BF16>
__device__ __forceinline__ void store_pair(void* base, size_t i, float a, float b) {
    if constexpr (BF16)
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(base) + i) = __floats2bfloat162_rn(a, b);
    else
        *reinterpret_cast<__half2*>(reinterpret_cast<__half*>(base) + i) = __floats2half2_rn(a, b);
}

// One warp per (row, head); lane owns E = hd / 64 consecutive elements of the
// first half and their rotate-half partners.
template <int HD, bool BF16>
__global__ void __launch_bounds__(128)
qkv_split_kernel(const __half* q_in, const __half* k_in, const __half* v_in,
                 int ld, const __half* __restrict__ q_bias, const __half* __restrict__ k_bias,
                 const __half* __restrict__ v_bias, float k_mul, float v_mul, const __half* __restrict__ q_norm,
                 const __half* __restrict__ k_norm, float eps, const __half* __restrict__ cos,
                 const __half* __restrict__ sin, void* q_out, void* k_out, void* v_out, int n_q, int n_k) {
    constexpr int E = HD / 64, H2 = HD / 2;
    const int s = blockIdx.x, head = blockIdx.y * blockDim.y + threadIdx.y, lane = threadIdx.x;
    if (head >= n_q + 2 * n_k) return;
    const __half* src;
    const __half* bias;
    const __half* nw;
    void* dst;
    float mul;
    bool rope;
    int hh;
    size_t orow;
    if (head < n_q) {
        hh = head; src = q_in; bias = q_bias; nw = q_norm; dst = q_out; mul = 1.f; rope = true; orow = n_q;
    } else if (head < n_q + n_k) {
        hh = head - n_q; src = k_in; bias = k_bias; nw = k_norm; dst = k_out; mul = k_mul; rope = true; orow = n_k;
    } else {
        hh = head - n_q - n_k; src = v_in; bias = v_bias; nw = nullptr; dst = v_out; mul = v_mul; rope = false;
        orow = n_k;
    }
    rope = rope && cos != nullptr;
    const __half* p = src + static_cast<size_t>(s) * ld + hh * HD;
    const int i0 = lane * E;
    float lo[E], hi[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        lo[e] = __half2float(p[i0 + e]);
        hi[e] = __half2float(p[i0 + e + H2]);
    }
    if (bias != nullptr) {
        const __half* b = bias + hh * HD;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            lo[e] += __half2float(b[i0 + e]);
            hi[e] += __half2float(b[i0 + e + H2]);
        }
    }
#pragma unroll
    for (int e = 0; e < E; ++e) {
        lo[e] *= mul;
        hi[e] *= mul;
    }
    if (nw != nullptr) {
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) ss += lo[e] * lo[e] + hi[e] * hi[e];
        const float r = rsqrtf(warp_sum(ss) / HD + eps);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            lo[e] *= r * __half2float(nw[i0 + e]);
            hi[e] *= r * __half2float(nw[i0 + e + H2]);
        }
    }
    if (rope) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float c = __half2float(cos[static_cast<size_t>(s) * HD + i0 + e]);
            const float sn = __half2float(sin[static_cast<size_t>(s) * HD + i0 + e]);
            const float a = lo[e], b = hi[e];
            lo[e] = a * c - b * sn;
            hi[e] = b * c + a * sn;
        }
    }
    const size_t o = (static_cast<size_t>(s) * orow + hh) * HD + i0;
    if constexpr (E == 2) {
        store_pair<BF16>(dst, o, lo[0], lo[1]);
        store_pair<BF16>(dst, o + H2, hi[0], hi[1]);
    } else {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            store_val<BF16>(dst, o + e, lo[e]);
            store_val<BF16>(dst, o + e + H2, hi[e]);
        }
    }
}

template <int HD, bool BF16>
void launch_qkv(const __half* q_in, const __half* k_in, const __half* v_in, int ld, const __half* q_bias,
                const __half* k_bias, const __half* v_bias, float k_mul, float v_mul, const __half* q_norm,
                const __half* k_norm, float eps, const __half* cos, const __half* sin, void* q_out, void* k_out,
                void* v_out, int S, int n_q, int n_k, cudaStream_t stream) {
    const dim3 block(32, 4);
    const dim3 grid(S, (n_q + 2 * n_k + 3) / 4);
    qkv_split_kernel<HD, BF16><<<grid, block, 0, stream>>>(q_in, k_in, v_in, ld, q_bias, k_bias, v_bias, k_mul,
                                                           v_mul, q_norm, k_norm, eps, cos, sin, q_out, k_out,
                                                           v_out, n_q, n_k);
}

}  // namespace

int residual_norm_quant(__half* x, const __half* delta, const __half* delta_bias, const __half* delta2,
                        __half* x_copy, const __half* weight, const __half* bias, int mode, float eps,
                        __nv_fp8_e4m3* out_fp8, const float* scale, __half* out_f16, int rows, int dim,
                        cudaStream_t stream) {
    if (x == nullptr || rows <= 0 || dim % 8 != 0 || dim > 8192 || (mode != 0 && mode != 1) ||
        (out_fp8 != nullptr && scale == nullptr))
        return 1;
    residual_norm_quant_kernel<<<rows, dim / 8, 0, stream>>>(x, delta, delta_bias, delta2, x_copy, weight, bias,
                                                             mode, eps, out_fp8, scale, out_f16, dim);
    return static_cast<int>(cudaGetLastError());
}

int bias_gelu_quant(const __half* x, int ld, const __half* bias, __nv_fp8_e4m3* out, const float* scale, int rows,
                    int n, cudaStream_t stream) {
    if (x == nullptr || out == nullptr || scale == nullptr || rows <= 0 || n % 8 != 0 || ld % 8 != 0) return 1;
    const int total = rows * (n / 8);
    const int blocks = (total + 255) / 256;
    bias_gelu_quant_kernel<<<blocks, 256, 0, stream>>>(x, ld, bias, out, scale, rows, n);
    return static_cast<int>(cudaGetLastError());
}

int qkv_split(const __half* q_in, const __half* k_in, const __half* v_in, int ld, const __half* q_bias,
              const __half* k_bias, const __half* v_bias, float k_mul, float v_mul, const __half* q_norm,
              const __half* k_norm, float eps, const __half* cos, const __half* sin, void* q_out, void* k_out,
              void* v_out, bool out_bf16, int S, int n_q, int n_k, int hd, cudaStream_t stream) {
    if (!q_in || !k_in || !v_in || !q_out || !k_out || !v_out || S <= 0 || (cos == nullptr) != (sin == nullptr))
        return 1;
#define FRT_QKV(HD_, BF_)                                                                                    \
    launch_qkv<HD_, BF_>(q_in, k_in, v_in, ld, q_bias, k_bias, v_bias, k_mul, v_mul, q_norm, k_norm, eps, cos, \
                         sin, q_out, k_out, v_out, S, n_q, n_k, stream)
    if (hd == 64) {
        if (out_bf16) FRT_QKV(64, true); else FRT_QKV(64, false);
    } else if (hd == 128) {
        if (out_bf16) FRT_QKV(128, true); else FRT_QKV(128, false);
    } else {
        return 1;
    }
#undef FRT_QKV
    return static_cast<int>(cudaGetLastError());
}

}  // namespace fp8_prefill
}  // namespace flash_rt
