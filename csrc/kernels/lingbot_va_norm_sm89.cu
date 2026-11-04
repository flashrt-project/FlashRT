// SPDX-License-Identifier: Apache-2.0
// Opt-in LingBot-VA normalization/quantization profiles; legacy norm.cu is unchanged.
#include "norm.cuh"
#include "common.cuh"
#include <cuda_fp8.h>
#include <stdexcept>

__global__ void bias_rms_norm_bf16_q8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int8_t* __restrict__ out_q8,
    int dim, float eps, float inv_scale) {
    using T2 = typename packed2<__nv_bfloat16>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int8_t* q8 = out_q8 + row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], bv = b2[i];
        __nv_bfloat16 xb0_b = __float2bfloat16(
            __bfloat162float(xv.x) + __bfloat162float(bv.x));
        __nv_bfloat16 xb1_b = __float2bfloat16(
            __bfloat162float(xv.y) + __bfloat162float(bv.y));
        float xb0 = __bfloat162float(xb0_b);
        float xb1 = __bfloat162float(xb1_b);
        local_sum += xb0 * xb0 + xb1 * xb1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], bv = b2[i], wv = w2[i];
        __nv_bfloat16 xb0_b = __float2bfloat16(
            __bfloat162float(xv.x) + __bfloat162float(bv.x));
        __nv_bfloat16 xb1_b = __float2bfloat16(
            __bfloat162float(xv.y) + __bfloat162float(bv.y));
        float v0 = __bfloat162float(xb0_b) * rms * __bfloat162float(wv.x);
        float v1 = __bfloat162float(xb1_b) * rms * __bfloat162float(wv.y);
        out2[i] = __halves2bfloat162(__float2bfloat16(v0),
                                     __float2bfloat16(v1));
        int q0 = __float2int_rn(v0 * inv_scale);
        int q1 = __float2int_rn(v1 * inv_scale);
        q0 = max(-127, min(127, q0));
        q1 = max(-127, min(127, q1));
        q8[2 * i] = (int8_t)q0;
        q8[2 * i + 1] = (int8_t)q1;
    }
}

template<int NP>
__global__ void bias_rms_norm_bf16_q8_ilp_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int8_t* __restrict__ out_q8,
    int dim, float eps, float inv_scale) {
    using T2 = typename packed2<__nv_bfloat16>::type;
    const int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int8_t* q8 = out_q8 + row * dim;
    extern __shared__ float shared[];
    float xb[NP][2];
    T2 wv[NP];
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        const int i = threadIdx.x + k * 256;
        const T2 xv = x2[i], bv = b2[i];
        wv[k] = w2[i];
        xb[k][0] = __bfloat162float(__float2bfloat16(
            __bfloat162float(xv.x) + __bfloat162float(bv.x)));
        xb[k][1] = __bfloat162float(__float2bfloat16(
            __bfloat162float(xv.y) + __bfloat162float(bv.y)));
    }
    float local_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < NP; ++k) local_sum += xb[k][0] * xb[k][0] + xb[k][1] * xb[k][1];
    const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        const int i = threadIdx.x + k * 256;
        const float v0 = xb[k][0] * rms * __bfloat162float(wv[k].x);
        const float v1 = xb[k][1] * rms * __bfloat162float(wv[k].y);
        out2[i] = __halves2bfloat162(__float2bfloat16(v0), __float2bfloat16(v1));
        int q0 = __float2int_rn(v0 * inv_scale);
        int q1 = __float2int_rn(v1 * inv_scale);
        q0 = max(-127, min(127, q0));
        q1 = max(-127, min(127, q1));
        q8[2 * i] = (int8_t)q0;
        q8[2 * i + 1] = (int8_t)q1;
    }
}

void bias_rms_norm_bf16_q8(const __nv_bfloat16* x,
                           const __nv_bfloat16* bias,
                           const __nv_bfloat16* weight,
                           __nv_bfloat16* out,
                           int8_t* out_q8,
                           int seq_len, int dim, float eps, float scale,
                           cudaStream_t stream) {
    if (dim == 3072) {
        bias_rms_norm_bf16_q8_ilp_kernel<6><<<seq_len, 256, 256 * sizeof(float), stream>>>(
            x, bias, weight, out, out_q8, dim, eps, 1.0f / scale);
        return;
    }
    bias_rms_norm_bf16_q8_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, bias, weight, out, out_q8, dim, eps, 1.0f / scale);
}

__global__ void quantize_i8_f16_kernel(const __nv_bfloat16* __restrict__ k,
                                       const __nv_bfloat16* __restrict__ v,
                                       int8_t* __restrict__ k8,
                                       __half* __restrict__ v16,
                                       long n, float inv_scale) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float kv = __bfloat162float(k[i]);
    int q = __float2int_rn(kv * inv_scale);
    k8[i] = (int8_t)max(-127, min(127, q));
    v16[i] = __float2half_rn(__bfloat162float(v[i]));
}

void quantize_i8_f16(const void* k, const void* v, void* k8, void* v16,
                     long n, float scale, cudaStream_t stream) {
    long blocks = (n + 255) / 256;
    quantize_i8_f16_kernel<<<(unsigned)blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<int8_t*>(k8), reinterpret_cast<__half*>(v16),
        n, 1.0f / scale);
}

template<typename T>
__global__ void lingbot_layer_norm_kernel(const T* __restrict__ x,
                                  const T* __restrict__ weight,
                                  const T* __restrict__ bias,
                                  T* __restrict__ out,
                                  float* __restrict__ out_scale,
                                  int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        local_sum += to_f32(val.x) + to_f32(val.y);
    }
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    if (out_scale != nullptr) {
        // Same values as the loop below, written straight out as FP8 with
        // 128-wide per-token scales -- the `fp8_per_token_block128_quant_bf16`
        // launch the caller used to run over this kernel's output, folded in.
        // Exact: a block scale is a **max**, and `fmaxf` is associative and
        // commutative, so unlike the sum reductions in this file its order is
        // free. The per-element arithmetic and the `s = max(amax/448, 1e-12)`,
        // `q = clamp(v * (1/s), +-448)` sequence are copied from that kernel.
        const int warp = (int)(threadIdx.x >> 5);
        const int lane = (int)(threadIdx.x & 31u);
        const int nblk = dim >> 7;
        const int nwarp = (int)(blockDim.x >> 5);
        for (int blk = warp; blk < nblk; blk += nwarp) {
            float vals[4];
            float amax = 0.0f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int i = (blk << 6) + (lane << 1) + j;   // packed pair index
                T2 xv = x2[i], wv = w2[i], bv = b2[i];
                float v0 = (to_f32(xv.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
                float v1 = (to_f32(xv.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
                // The standalone path rounds to T here and the quantiser reads it
                // back, so the quantised input is the rounded value, not v.
                const float b0 = to_f32(from_f32<T>(v0));
                const float b1 = to_f32(from_f32<T>(v1));
                vals[2 * j] = b0;
                vals[2 * j + 1] = b1;
                amax = fmaxf(amax, fmaxf(fabsf(b0), fabsf(b1)));
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float sc = fmaxf(amax / 448.0f, 1.0e-12f);
            const float inv_sc = 1.0f / sc;
            if (lane == 0) out_scale[row * nblk + blk] = sc;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = (blk << 7) + (lane << 2) + j;
                const float q = fminf(fmaxf(vals[j] * inv_sc, -448.0f), 448.0f);
                reinterpret_cast<__nv_fp8_e4m3*>(out)[row * dim + d] = __nv_fp8_e4m3(q);
            }
        }
        return;
    }

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i], bv = b2[i];
        float v0 = (to_f32(xv.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
        float v1 = (to_f32(xv.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

__global__ void sstp_layer_norm_fp8_block128_kernel(
    __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ xbias, const float* __restrict__ gate_ss_row,
    const float* __restrict__ gate_tp, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias, __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ out_scale, int dim, float eps) {
    using T2 = __nv_bfloat162;
    const int row = blockIdx.x;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const long i = (long)row * dim + d;
        float xb = __bfloat162float(x[i]);
        if (xbias != nullptr)
            xb = __bfloat162float(
                __float2bfloat16_rn(__fadd_rn(xb, __bfloat162float(xbias[d]))));
        const float v = __fmul_rn(xb, __fadd_rn(gate_ss_row[d], gate_tp[i]));
        residual[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(residual[i]), v));
    }
    __syncthreads();

    const T2* x2 = reinterpret_cast<const T2*>(residual + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    const int dim2 = dim >> 1;
    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        local_sum += to_f32(val.x) + to_f32(val.y);
    }
    float mean = block_reduce_sum(local_sum, shared) / dim;
    __syncthreads();
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int nblk = dim >> 7;
    const int nwarp = (int)(blockDim.x >> 5);
    for (int blk = warp; blk < nblk; blk += nwarp) {
        float vals[4];
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = (blk << 6) + (lane << 1) + j;
            T2 xv = x2[i], wv = w2[i], bv = b2[i];
            float v0 = (to_f32(xv.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
            float v1 = (to_f32(xv.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
            const float r0 = to_f32(from_f32<__nv_bfloat16>(v0));
            const float r1 = to_f32(from_f32<__nv_bfloat16>(v1));
            vals[2 * j] = r0;
            vals[2 * j + 1] = r1;
            amax = fmaxf(amax, fmaxf(fabsf(r0), fabsf(r1)));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        const float sc = fmaxf(amax / 448.0f, 1.0e-12f);
        const float inv_sc = 1.0f / sc;
        if (lane == 0) out_scale[row * nblk + blk] = sc;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = (blk << 7) + (lane << 2) + j;
            const float q = fminf(fmaxf(vals[j] * inv_sc, -448.0f), 448.0f);
            out[row * dim + d] = __nv_fp8_e4m3(q);
        }
    }
}

void gated_residual_sstp_layer_norm_fp8_block128(
    __nv_bfloat16* residual, const __nv_bfloat16* x, const __nv_bfloat16* xbias,
    const float* gate_ss_row, const float* gate_tp, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_fp8_e4m3* out, float* out_scale,
    int seq_len, int dim, float eps, cudaStream_t stream) {
    sstp_layer_norm_fp8_block128_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, xbias, gate_ss_row, gate_tp, weight, bias, out, out_scale, dim, eps);
}

__global__ void layer_norm_fp8_block128_3072_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias, __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ out_scale, int dim, float eps) {
    // dim stays a runtime value: the statistics divide by it, and a constant
    // divisor would compile to a reciprocal multiply instead of the division
    // layer_norm_kernel performs.
    using T2 = __nv_bfloat162;
    constexpr int NP = 6;
    const int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    extern __shared__ float shared[];
    float xv[NP][2];
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        const T2 v = x2[threadIdx.x + k * 256];
        xv[k][0] = to_f32(v.x);
        xv[k][1] = to_f32(v.y);
    }
    float local_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < NP; ++k) local_sum += xv[k][0] + xv[k][1];
    const float mean = block_reduce_sum(local_sum, shared) / dim;
    // every lane must have read shared[0] before the next reduction overwrites
    // it (layer_norm_kernel relies on its reload loop for that spacing)
    __syncthreads();
    float local_var = 0.0f;
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        const float d0 = xv[k][0] - mean, d1 = xv[k][1] - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    const float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    constexpr int nblk = 24;
#pragma unroll
    for (int r = 0; r < nblk / 8; ++r) {
        const int blk = warp + r * 8;
        float vals[4];
        float amax = 0.0f;
        T2 xs[2], ws[2], bs[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = (blk << 6) + (lane << 1) + j;
            xs[j] = x2[i]; ws[j] = w2[i]; bs[j] = b2[i];
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            float v0 = (to_f32(xs[j].x) - mean) * inv_std * to_f32(ws[j].x) + to_f32(bs[j].x);
            float v1 = (to_f32(xs[j].y) - mean) * inv_std * to_f32(ws[j].y) + to_f32(bs[j].y);
            const float r0 = to_f32(from_f32<__nv_bfloat16>(v0));
            const float r1 = to_f32(from_f32<__nv_bfloat16>(v1));
            vals[2 * j] = r0;
            vals[2 * j + 1] = r1;
            amax = fmaxf(amax, fmaxf(fabsf(r0), fabsf(r1)));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        const float sc = fmaxf(amax / 448.0f, 1.0e-12f);
        const float inv_sc = 1.0f / sc;
        if (lane == 0) out_scale[row * nblk + blk] = sc;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = (blk << 7) + (lane << 2) + j;
            const float q = fminf(fmaxf(vals[j] * inv_sc, -448.0f), 448.0f);
            out[row * dim + d] = __nv_fp8_e4m3(q);
        }
    }
}

void layer_norm_fp8_block128(const __nv_bfloat16* x,
                              const __nv_bfloat16* weight,
                              const __nv_bfloat16* bias,
                              __nv_fp8_e4m3* out, float* out_scale,
                              int seq_len, int dim, float eps,
                              cudaStream_t stream) {
    if (dim == 3072) {
        layer_norm_fp8_block128_3072_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
            x, weight, bias, out, out_scale, dim, eps);
        return;
    }
    lingbot_layer_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, weight, bias, reinterpret_cast<__nv_bfloat16*>(out), out_scale, dim, eps);
}
