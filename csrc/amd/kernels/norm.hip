// ================================================================
// FlashRT AMD — Normalization kernels (dtype-generic, CDNA wave64)
// RMSNorm, LayerNorm, AdaRMSNorm+Style, bias+residual+LayerNorm,
// vision-token average pooling, and the fused Norm → FP8 set
// (rms_norm_fp8, ada_rms_norm_style_fp8, residual_add_rms_norm_fp8).
// Mirrors csrc/kernels/norm.cu host-wrapper signatures exactly so
// the pipeline layer stays portable text across platforms.
//
// FP8 output dtype: OCP e4m3 via <hip/hip_fp8.h> __hip_fp8_e4m3
// (saturating RNE conversion, same as __nv_fp8_e4m3; values are
// pre-clamped to ±448 exactly as in norm.cu). Same byte format as
// the HIP_R_8F_E4M3 the hipBLASLt GEMM side consumes.
// ================================================================

#include "common_hip.h"
#include <hip/hip_fp8.h>

// 16-byte packet (8 bf16) and 8-byte FP8 packet for the wide (CDNA
// retune) kernel variants — measured on the ew_tune probe: wide
// row-per-block b256 wins ~1.4x on encoder-seq shapes.
struct alignas(16) nrm_bf16x8 { __hip_bfloat162 p[4]; };
struct alignas(8) nrm_fp8x8 { unsigned char b[8]; };

// ── RMSNorm ──
template<typename T>
__global__ void rms_norm_kernel(const T* __restrict__ x,
                                const T* __restrict__ weight,
                                T* __restrict__ out,
                                int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + (size_t)row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + (size_t)row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

template __global__ void rms_norm_kernel<__half>(const __half*, const __half*, __half*, int, float);
template __global__ void rms_norm_kernel<__hip_bfloat16>(const __hip_bfloat16*, const __hip_bfloat16*, __hip_bfloat16*, int, float);

void rms_norm(const __hip_bfloat16* x, const __hip_bfloat16* weight,
              __hip_bfloat16* out, int seq_len, int dim, float eps,
              hipStream_t stream) {
    rms_norm_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(x, weight, out, dim, eps);
}

void rms_norm_fp16(const __half* x, const __half* weight,
                   __half* out, int seq_len, int dim, float eps,
                   hipStream_t stream) {
    rms_norm_kernel<__half><<<seq_len, 256, 4 * sizeof(float), stream>>>(x, weight, out, dim, eps);
}

void rms_norm_inplace(const __hip_bfloat16* weight,
                      __hip_bfloat16* x, int seq_len, int dim, float eps,
                      hipStream_t stream) {
    rms_norm_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(x, weight, x, dim, eps);
}

// ── LayerNorm ──
template<typename T>
__global__ void layer_norm_kernel(const T* __restrict__ x,
                                  const T* __restrict__ weight,
                                  const T* __restrict__ bias,
                                  T* __restrict__ out,
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

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i], bv = b2[i];
        float v0 = (to_f32(xv.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
        float v1 = (to_f32(xv.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

template __global__ void layer_norm_kernel<__half>(const __half*, const __half*, const __half*, __half*, int, float);
template __global__ void layer_norm_kernel<__hip_bfloat16>(const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, __hip_bfloat16*, int, float);

// Wide row-per-block variant: dwordx4 chunked accesses, same 2-pass
// mean/var structure. Per-thread accumulation order changes (chunked)
// -> last-ULP differences only; gated by the cos parity suite.
__global__ void layer_norm_wide_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ bias,
    __hip_bfloat16* __restrict__ out,
    int dim, float eps) {
    const int row = blockIdx.x;
    const int chunks = dim >> 3;
    const nrm_bf16x8* x8 = reinterpret_cast<const nrm_bf16x8*>(x + (size_t)row * dim);
    const nrm_bf16x8* w8 = reinterpret_cast<const nrm_bf16x8*>(weight);
    const nrm_bf16x8* b8 = reinterpret_cast<const nrm_bf16x8*>(bias);
    nrm_bf16x8* out8 = reinterpret_cast<nrm_bf16x8*>(out + (size_t)row * dim);

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        nrm_bf16x8 v = x8[c];
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            local_sum += to_f32(v.p[k].x) + to_f32(v.p[k].y);
    }
    const float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        nrm_bf16x8 v = x8[c];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float d0 = to_f32(v.p[k].x) - mean;
            const float d1 = to_f32(v.p[k].y) - mean;
            local_var += d0 * d0 + d1 * d1;
        }
    }
    const float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        nrm_bf16x8 v = x8[c];
        nrm_bf16x8 wv = w8[c];
        nrm_bf16x8 bv = b8[c];
        nrm_bf16x8 o;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float n0 = (to_f32(v.p[k].x) - mean) * inv_std * to_f32(wv.p[k].x) + to_f32(bv.p[k].x);
            const float n1 = (to_f32(v.p[k].y) - mean) * inv_std * to_f32(wv.p[k].y) + to_f32(bv.p[k].y);
            o.p[k] = make_packed2<__hip_bfloat16>(
                from_f32<__hip_bfloat16>(n0), from_f32<__hip_bfloat16>(n1));
        }
        out8[c] = o;
    }
}

void layer_norm(const __hip_bfloat16* x, const __hip_bfloat16* weight,
                const __hip_bfloat16* bias, __hip_bfloat16* out,
                int seq_len, int dim, float eps, hipStream_t stream) {
    if (dim % 8 == 0) {
        layer_norm_wide_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
            x, weight, bias, out, dim, eps);
        return;
    }
    layer_norm_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(x, weight, bias, out, dim, eps);
}

// ── AdaRMSNorm + Style ──
// style row layout: (seq, 3*dim) = [scale | shift | gate] per row.
// out = rmsnorm(x)*weight * (1 + scale) + shift; gate_out = gate (copy).
template<typename T>
__global__ void ada_rms_norm_style_kernel(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ style, T* __restrict__ out, T* __restrict__ gate_out,
    int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    T2* gate2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(xv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(
            from_f32<T>(n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)),
            from_f32<T>(n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)));
        gate2[i] = gv;
    }
}

template __global__ void ada_rms_norm_style_kernel<__half>(const __half*, const __half*, const __half*, __half*, __half*, int, float);
template __global__ void ada_rms_norm_style_kernel<__hip_bfloat16>(const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, __hip_bfloat16*, __hip_bfloat16*, int, float);

void ada_rms_norm_style(const __hip_bfloat16* x, const __hip_bfloat16* weight,
                        const __hip_bfloat16* style,
                        __hip_bfloat16* out, __hip_bfloat16* gate_out,
                        int seq_len, int dim, float eps, hipStream_t stream) {
    ada_rms_norm_style_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(
        x, weight, style, out, gate_out, dim, eps);
}

// ── RMSNorm → FP8 ──
template<typename T>
__global__ void rms_norm_fp8_kernel(const T* __restrict__ x,
                                     const T* __restrict__ weight,
                                     __hip_fp8_e4m3* __restrict__ out,
                                     int dim, float eps,
                                     const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __hip_fp8_e4m3* out_row = out + row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        float v0 = to_f32(xv.x) * rms * to_f32(wv.x) * inv_scale;
        float v1 = to_f32(xv.y) * rms * to_f32(wv.y) * inv_scale;
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

template __global__ void rms_norm_fp8_kernel<__half>(const __half*, const __half*, __hip_fp8_e4m3*, int, float, const float*);
template __global__ void rms_norm_fp8_kernel<__hip_bfloat16>(const __hip_bfloat16*, const __hip_bfloat16*, __hip_fp8_e4m3*, int, float, const float*);

void rms_norm_fp8(const __hip_bfloat16* x, const __hip_bfloat16* weight,
                   __hip_fp8_e4m3* out, int seq_len, int dim, float eps,
                   const float* d_scale, hipStream_t stream) {
    rms_norm_fp8_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(x, weight, out, dim, eps, d_scale);
}

// ── AdaRMSNorm + Style → FP8 ──
template<typename T>
__global__ void ada_rms_norm_style_fp8_kernel(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ style, __hip_fp8_e4m3* __restrict__ out, T* __restrict__ gate_out,
    int dim, float eps, const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __hip_fp8_e4m3* out_row = out + row * dim;
    T2* gate2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(xv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(val0, -448.0f), 448.0f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(val1, -448.0f), 448.0f));
        gate2[i] = gv;
    }
}

template __global__ void ada_rms_norm_style_fp8_kernel<__half>(const __half*, const __half*, const __half*, __hip_fp8_e4m3*, __half*, int, float, const float*);
template __global__ void ada_rms_norm_style_fp8_kernel<__hip_bfloat16>(const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, __hip_fp8_e4m3*, __hip_bfloat16*, int, float, const float*);

void ada_rms_norm_style_fp8(const __hip_bfloat16* x, const __hip_bfloat16* weight,
                             const __hip_bfloat16* style,
                             __hip_fp8_e4m3* out, __hip_bfloat16* gate_out,
                             int seq_len, int dim, float eps,
                             const float* d_scale, hipStream_t stream) {
    ada_rms_norm_style_fp8_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(
        x, weight, style, out, gate_out, dim, eps, d_scale);
}

// ── Residual Add + RMSNorm → FP8 ──
// Pass 1 writes the low-precision (rounded) residual back to global and
// accumulates the sum of squares from the UNROUNDED float sums; pass 2
// re-reads the rounded residual for the normalize — matching norm.cu.
template<typename T>
__global__ void residual_add_rms_norm_fp8_kernel(
    T* __restrict__ residual, const T* __restrict__ x,
    const T* __restrict__ weight, __hip_fp8_e4m3* __restrict__ out,
    int dim, float eps, const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __hip_fp8_e4m3* out_row = out + row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        float v0 = to_f32(rv.x) * rms * to_f32(wv.x) * inv_scale;
        float v1 = to_f32(rv.y) * rms * to_f32(wv.y) * inv_scale;
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

template __global__ void residual_add_rms_norm_fp8_kernel<__half>(__half*, const __half*, const __half*, __hip_fp8_e4m3*, int, float, const float*);
template __global__ void residual_add_rms_norm_fp8_kernel<__hip_bfloat16>(__hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, __hip_fp8_e4m3*, int, float, const float*);

// Wide row-per-block variant (see layer_norm_wide_kernel notes).
// Keeps the rounded residual in registers for pass 2 — bit-identical
// to the re-read (the register value IS the stored rounded value)
// as long as each chunk stays with one thread (it does).
__global__ void residual_add_rms_norm_fp8_wide_kernel(
    __hip_bfloat16* __restrict__ residual, const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ weight, __hip_fp8_e4m3* __restrict__ out,
    int dim, float eps, const float* __restrict__ d_scale) {
    const int row = blockIdx.x;
    const int chunks = dim >> 3;
    nrm_bf16x8* res8 = reinterpret_cast<nrm_bf16x8*>(residual + (size_t)row * dim);
    const nrm_bf16x8* x8 = reinterpret_cast<const nrm_bf16x8*>(x + (size_t)row * dim);
    const nrm_bf16x8* w8 = reinterpret_cast<const nrm_bf16x8*>(weight);
    nrm_fp8x8* out8 = reinterpret_cast<nrm_fp8x8*>(
        reinterpret_cast<unsigned char*>(out) + (size_t)row * dim);

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        nrm_bf16x8 rv = res8[c];
        nrm_bf16x8 xv = x8[c];
        nrm_bf16x8 rounded;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float r0 = to_f32(rv.p[k].x) + to_f32(xv.p[k].x);
            const float r1 = to_f32(rv.p[k].y) + to_f32(xv.p[k].y);
            rounded.p[k] = make_packed2<__hip_bfloat16>(
                from_f32<__hip_bfloat16>(r0), from_f32<__hip_bfloat16>(r1));
            local_sum += r0 * r0 + r1 * r1;
        }
        res8[c] = rounded;
    }
    const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    const float inv_scale = 1.0f / (*d_scale);

    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        nrm_bf16x8 rv = res8[c];
        nrm_bf16x8 wv = w8[c];
        nrm_fp8x8 o;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float v0 = to_f32(rv.p[k].x) * rms * to_f32(wv.p[k].x) * inv_scale;
            const float v1 = to_f32(rv.p[k].y) * rms * to_f32(wv.p[k].y) * inv_scale;
            o.b[2*k]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f)).__x;
            o.b[2*k+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f)).__x;
        }
        out8[c] = o;
    }
}

void residual_add_rms_norm_fp8(__hip_bfloat16* residual, const __hip_bfloat16* x,
                                const __hip_bfloat16* weight, __hip_fp8_e4m3* out,
                                int seq_len, int dim, float eps,
                                const float* d_scale, hipStream_t stream) {
    if (dim % 8 == 0) {
        residual_add_rms_norm_fp8_wide_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
            residual, x, weight, out, dim, eps, d_scale);
        return;
    }
    residual_add_rms_norm_fp8_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps, d_scale);
}

// ── Fused bias + residual add + LayerNorm ──
// residual = lowp(residual + x + bias_pre) written back to global, then
// LayerNorm over the UPDATED residual into `out`. Passes 2 and 3 re-read
// residual from global (now rounded to bf16) — this strictly matches
// running bias_residual then layer_norm sequentially, bit for bit.
// LDS: block_reduce_sum partials, one float per wave (4 for 256 threads).
template<typename T>
__global__ void bias_residual_layer_norm_kernel(
        T* __restrict__ residual,
        const T* __restrict__ x,
        const T* __restrict__ bias_pre,
        const T* __restrict__ ln_weight,
        const T* __restrict__ ln_bias,
        T* __restrict__ out,
        int dim, float eps) {
    extern __shared__ float partial[];

    int row = blockIdx.x;
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual + (size_t)row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + (size_t)row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias_pre);
    const T2* w2 = reinterpret_cast<const T2*>(ln_weight);
    const T2* lnb2 = reinterpret_cast<const T2*>(ln_bias);
    T2* out2 = reinterpret_cast<T2*>(out + (size_t)row * dim);
    int dim2 = dim >> 1;

    // Pass 1: residual = lowp(residual + x + bias_pre) (write to global),
    // accumulate sum for mean.
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], bv = b2[i];
        T r0 = from_f32<T>(to_f32(rv.x) + to_f32(xv.x) + to_f32(bv.x));
        T r1 = from_f32<T>(to_f32(rv.y) + to_f32(xv.y) + to_f32(bv.y));
        res2[i] = make_packed2<T>(r0, r1);
        local_sum += to_f32(r0) + to_f32(r1);
    }
    float mean = block_reduce_sum(local_sum, partial) / dim;

    // Pass 2: re-read residual from global (now rounded), accumulate var.
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, partial) / dim + eps);

    // Pass 3: re-read residual, normalize, apply weight/bias, write out.
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i];
        T2 wv = w2[i], lbv = lnb2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        float n0 = (v0 - mean) * inv_std * to_f32(wv.x) + to_f32(lbv.x);
        float n1 = (v1 - mean) * inv_std * to_f32(wv.y) + to_f32(lbv.y);
        out2[i] = make_packed2<T>(from_f32<T>(n0), from_f32<T>(n1));
    }
}

template __global__ void bias_residual_layer_norm_kernel<__half>(__half*, const __half*, const __half*, const __half*, const __half*, __half*, int, float);
template __global__ void bias_residual_layer_norm_kernel<__hip_bfloat16>(__hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, __hip_bfloat16*, int, float);

void bias_residual_layer_norm_bf16(
        __hip_bfloat16* residual, const __hip_bfloat16* x,
        const __hip_bfloat16* bias_pre,
        const __hip_bfloat16* ln_weight, const __hip_bfloat16* ln_bias,
        __hip_bfloat16* out, int seq_len, int dim, float eps,
        hipStream_t stream) {
    int smem = 4 * sizeof(float);
    bias_residual_layer_norm_kernel<__hip_bfloat16><<<seq_len, 256, smem, stream>>>(
        residual, x, bias_pre, ln_weight, ln_bias, out, dim, eps);
}

// ── Vision Token Spatial Average Pooling (BF16) ──
//
// Reduces vision-tower output from (nv * spv, dim) to
// (nv * spv / (f*f), dim) by f×f average pooling in the (H, W) token
// grid. Token layout: each view's spv tokens form an (H, W) grid with
// spv = H * W; each f×f block of tokens averages into one output token.
//
// Launch: gridDim = (nv * out_spv, 1), blockDim = (256, 1)
//         where out_spv = spv / (f*f). No LDS (pure BW-limited kernel).
__global__ void avg_pool_vision_tokens_kernel(
        const __hip_bfloat16* __restrict__ x,   // (nv * spv, dim) BF16
        __hip_bfloat16* __restrict__ out,       // (nv * out_spv, dim) BF16
        int nv, int H, int W, int dim, int f) {
    // output token index in [0, nv * (H/f) * (W/f))
    int out_tok = blockIdx.x;
    int H_out = H / f;
    int W_out = W / f;
    int spv_out = H_out * W_out;

    int v   = out_tok / spv_out;          // view index
    int rc  = out_tok % spv_out;          // spatial index within view
    int r_out = rc / W_out;
    int c_out = rc % W_out;

    float inv_f2 = 1.0f / float(f * f);

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.f;
        for (int dr = 0; dr < f; dr++) {
            for (int dc = 0; dc < f; dc++) {
                int r_in  = r_out * f + dr;
                int c_in  = c_out * f + dc;
                int in_tok = v * H * W + r_in * W + c_in;
                sum += __bfloat162float(x[in_tok * dim + d]);
            }
        }
        out[out_tok * dim + d] = __float2bfloat16(sum * inv_f2);
    }
}

// pool_factor: 1 = no-op, 2 = 2x2 (spv 256→64), 4 = 4x4 (spv 256→16)
// H = W = sqrt(spv) must be divisible by pool_factor.
void avg_pool_vision_tokens(
        const __hip_bfloat16* x, __hip_bfloat16* out,
        int nv, int H, int W, int dim, int pool_factor,
        hipStream_t stream) {
    int H_out = H / pool_factor;
    int W_out = W / pool_factor;
    int out_tokens = nv * H_out * W_out;
    avg_pool_vision_tokens_kernel<<<out_tokens, 256, 0, stream>>>(
        x, out, nv, H, W, dim, pool_factor);
}
