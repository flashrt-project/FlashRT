// ================================================================
// FlashRT AMD — Cross-layer fusion kernels (dtype-generic, CDNA wave64)
// Fused gate*residual + AdaRMSNorm -> FP8 (decoder optimization).
// Mirrors csrc/kernels/fusion.cu host-wrapper signatures exactly so
// the pipeline layer stays portable text across platforms.
//
// FP8 output dtype: OCP e4m3 via <hip/hip_fp8.h> __hip_fp8_e4m3
// (saturating RNE conversion, same as __nv_fp8_e4m3; values are
// pre-clamped to ±448 exactly as in fusion.cu). Only the FP8 path is
// ported; gate_residual_ada_norm_fp16 stays CUDA-only.
// ================================================================

#include "common_hip.h"
#include <hip/hip_fp8.h>

// ── Fused Gate*Residual + AdaRMSNorm + Style -> FP8 ──
// Pass 1: residual += x * gate; sum_sq += residual^2
// Reduce: rms = rsqrt(sum_sq / dim + eps)
// Pass 2: out_fp8 = clamp((norm(residual) * (1+scale) + shift) * inv_scale)
template<typename T>
__global__ void gate_residual_ada_norm_fp8_kernel(
    T* __restrict__ residual,
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const T* __restrict__ weight,
    const T* __restrict__ style,
    __hip_fp8_e4m3* __restrict__ out,
    T* __restrict__ gate_out,
    int dim, float eps,
    const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    __hip_fp8_e4m3* out_row = out + row * dim;
    T2* gate_out2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];

    // Pass 1: residual += x * gate, compute sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    // Pass 2: AdaRMSNorm with style -> FP8
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(val0, -448.0f), 448.0f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(val1, -448.0f), 448.0f));
        gate_out2[i] = gv;
    }
}

template __global__ void gate_residual_ada_norm_fp8_kernel<__half>(
    __half*, const __half*, const __half*, const __half*, const __half*,
    __hip_fp8_e4m3*, __half*, int, float, const float*);
template __global__ void gate_residual_ada_norm_fp8_kernel<__hip_bfloat16>(
    __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*,
    __hip_fp8_e4m3*, __hip_bfloat16*, int, float, const float*);

// ── ksum variant: consume split-K FP32 GEMM partials in place of x ──
// x[row][i] = bf16_round( (Σ_s partial[s][row][i]) * dsa[0]*dsb[0] );
// ascending-split sum (deterministic), rounded to bf16 BEFORE use so
// the dataflow matches the unfused GEMM(bf16 out) -> this-kernel
// chain. Everything after x is identical to gate_residual_ada_norm_fp8.
template<typename T>
__global__ void gate_residual_ada_norm_fp8_ksum_kernel(
    T* __restrict__ residual,
    const float* __restrict__ partial,   // (splits, seq, dim) f32
    int splits,
    const float* __restrict__ dsa, const float* __restrict__ dsb,
    const T* __restrict__ gate,
    const T* __restrict__ weight,
    const T* __restrict__ style,
    __hip_fp8_e4m3* __restrict__ out,
    T* __restrict__ gate_out,
    int seq_len, int dim, float eps,
    const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    __hip_fp8_e4m3* out_row = out + row * dim;
    T2* gate_out2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;
    const size_t plane = (size_t)seq_len * dim;
    const float* p_row = partial + (size_t)row * dim;

    extern __shared__ float shared[];

    float sab = (*dsa) * (*dsb);
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        float x0 = 0.0f, x1 = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float* p = p_row + (size_t)s * plane + 2 * i;
            x0 += p[0];
            x1 += p[1];
        }
        T xb0 = from_f32<T>(x0 * sab);
        T xb1 = from_f32<T>(x1 * sab);
        T2 rv = res2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xb0) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xb1) * to_f32(gv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(val0, -448.0f), 448.0f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(val1, -448.0f), 448.0f));
        gate_out2[i] = gv;
    }
}

template __global__ void gate_residual_ada_norm_fp8_ksum_kernel<__hip_bfloat16>(
    __hip_bfloat16*, const float*, int, const float*, const float*,
    const __hip_bfloat16*, const __hip_bfloat16*, const __hip_bfloat16*,
    __hip_fp8_e4m3*, __hip_bfloat16*, int, int, float, const float*);

void gate_residual_ada_norm_fp8_ksum(
    __hip_bfloat16* residual, const float* partial, int splits,
    const float* d_scale_a, const float* d_scale_b,
    const __hip_bfloat16* gate, const __hip_bfloat16* weight,
    const __hip_bfloat16* style,
    __hip_fp8_e4m3* out, __hip_bfloat16* gate_out,
    int seq_len, int dim, float eps,
    const float* d_scale, hipStream_t stream) {
    gate_residual_ada_norm_fp8_ksum_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(
        residual, partial, splits, d_scale_a, d_scale_b,
        gate, weight, style, out, gate_out, seq_len, dim, eps, d_scale);
}

// Wide row-per-block variant (CDNA retune, ew_tune probe winner):
// dwordx4 chunked accesses + packed FP8 stores; the rounded residual
// stays in registers for pass 2 (bit-identical to the re-read — each
// chunk is owned by one thread in both passes). Reduction order
// changes (chunked) -> cos-gated.
struct alignas(16) fus_bf16x8 { __hip_bfloat162 p[4]; };
struct alignas(8) fus_fp8x8 { unsigned char b[8]; };

__global__ void gate_residual_ada_norm_fp8_wide_kernel(
    __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ gate,
    const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ style,
    __hip_fp8_e4m3* __restrict__ out,
    __hip_bfloat16* __restrict__ gate_out,
    int dim, float eps,
    const float* __restrict__ d_scale) {
    const int row = blockIdx.x;
    const int chunks = dim >> 3;
    fus_bf16x8* res8 = reinterpret_cast<fus_bf16x8*>(residual + (size_t)row * dim);
    const fus_bf16x8* x8 = reinterpret_cast<const fus_bf16x8*>(x + (size_t)row * dim);
    const fus_bf16x8* g8 = reinterpret_cast<const fus_bf16x8*>(gate + (size_t)row * dim);
    const fus_bf16x8* w8 = reinterpret_cast<const fus_bf16x8*>(weight);
    const __hip_bfloat16* style_row = style + (size_t)row * 3 * dim;
    const fus_bf16x8* sc8 = reinterpret_cast<const fus_bf16x8*>(style_row);
    const fus_bf16x8* sh8 = reinterpret_cast<const fus_bf16x8*>(style_row + dim);
    const fus_bf16x8* gt8 = reinterpret_cast<const fus_bf16x8*>(style_row + 2 * dim);
    fus_fp8x8* out8 = reinterpret_cast<fus_fp8x8*>(
        reinterpret_cast<unsigned char*>(out) + (size_t)row * dim);
    fus_bf16x8* gate_out8 = reinterpret_cast<fus_bf16x8*>(gate_out + (size_t)row * dim);

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        fus_bf16x8 rv = res8[c];
        fus_bf16x8 xv = x8[c];
        fus_bf16x8 gv = g8[c];
        fus_bf16x8 rounded;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float r0 = to_f32(rv.p[k].x) + to_f32(xv.p[k].x) * to_f32(gv.p[k].x);
            const float r1 = to_f32(rv.p[k].y) + to_f32(xv.p[k].y) * to_f32(gv.p[k].y);
            rounded.p[k] = make_packed2<__hip_bfloat16>(
                from_f32<__hip_bfloat16>(r0), from_f32<__hip_bfloat16>(r1));
            local_sum += r0 * r0 + r1 * r1;
        }
        res8[c] = rounded;
    }
    const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    const float inv_scale = 1.0f / (*d_scale);

    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        fus_bf16x8 rv = res8[c];
        fus_bf16x8 wv = w8[c];
        fus_bf16x8 sv = sc8[c];
        fus_bf16x8 hv = sh8[c];
        fus_bf16x8 gv = gt8[c];
        fus_fp8x8 o;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float n0 = to_f32(rv.p[k].x) * rms * to_f32(wv.p[k].x);
            const float n1 = to_f32(rv.p[k].y) * rms * to_f32(wv.p[k].y);
            const float v0 = (n0 * (1.0f + to_f32(sv.p[k].x)) + to_f32(hv.p[k].x)) * inv_scale;
            const float v1 = (n1 * (1.0f + to_f32(sv.p[k].y)) + to_f32(hv.p[k].y)) * inv_scale;
            o.b[2*k]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f)).__x;
            o.b[2*k+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f)).__x;
        }
        out8[c] = o;
        gate_out8[c] = gv;
    }
}

void gate_residual_ada_norm_fp8(__hip_bfloat16* residual, const __hip_bfloat16* x,
                                 const __hip_bfloat16* gate, const __hip_bfloat16* weight,
                                 const __hip_bfloat16* style,
                                 __hip_fp8_e4m3* out, __hip_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 const float* d_scale, hipStream_t stream) {
    if (dim % 8 == 0) {
        gate_residual_ada_norm_fp8_wide_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
            residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
        return;
    }
    gate_residual_ada_norm_fp8_kernel<__hip_bfloat16><<<seq_len, 256, 4 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
}
