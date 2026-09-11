// ================================================================
// FlashRT AMD — Activation kernels (dtype-generic, CDNA wave64)
// GeLU, GeGLU (split and merged), fused bias+GeLU (strict),
// merged GeGLU → FP8 (gate_geglu_merged_fp8).
// Mirrors csrc/kernels/activation.cu host-wrapper signatures exactly
// so the pipeline layer stays portable text across platforms.
//
// FP8 output dtype: OCP e4m3 via <hip/hip_fp8.h> __hip_fp8_e4m3
// (saturating RNE conversion, same as __nv_fp8_e4m3; values are
// pre-clamped to ±448 exactly as in activation.cu). Only the generic
// (BF16-bound) gate_geglu_merged_fp8 path is ported; the vectorized
// FP16 encoder variant stays CUDA-only.
// ================================================================

#include "common_hip.h"
#include <hip/hip_fp8.h>

// ── Gate GELU Multiply ──
// GELU(x) approx: x * sigmoid(1.5957691216 * x * (1 + 0.044715 * x^2))
// (tanh/sigmoid approx GELU, not SiLU — matches the CUDA kernel.)
template<typename T>
__global__ void gate_geglu_kernel(const T* __restrict__ gate,
                                  const T* __restrict__ up,
                                  T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = to_f32(gate[idx]);
        float u = to_f32(up[idx]);
        float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
        out[idx] = from_f32<T>(gelu * u);
    }
}

template __global__ void gate_geglu_kernel<__half>(const __half*, const __half*, __half*, int);
template __global__ void gate_geglu_kernel<__hip_bfloat16>(const __hip_bfloat16*, const __hip_bfloat16*, __hip_bfloat16*, int);

void gate_geglu(const __hip_bfloat16* gate, const __hip_bfloat16* up,
                __hip_bfloat16* out, int n, hipStream_t stream) {
    gate_geglu_kernel<__hip_bfloat16><<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
}

// ── GELU in-place (tanh approx) ──
template<typename T>
__global__ void gelu_kernel(T* __restrict__ x, int n) {
    using T2 = typename packed2<T>::type;
    T2* x2 = reinterpret_cast<T2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 val = x2[idx];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        float t0 = tanhf(0.7978845608f * (v0 + 0.044715f * v0 * v0 * v0));
        float t1 = tanhf(0.7978845608f * (v1 + 0.044715f * v1 * v1 * v1));
        x2[idx] = make_packed2<T>(
            from_f32<T>(v0 * 0.5f * (1.0f + t0)),
            from_f32<T>(v1 * 0.5f * (1.0f + t1)));
    }
}

template __global__ void gelu_kernel<__half>(__half*, int);
template __global__ void gelu_kernel<__hip_bfloat16>(__hip_bfloat16*, int);

void gelu_inplace(__hip_bfloat16* x, int n, hipStream_t stream) {
    int n2 = n >> 1;
    gelu_kernel<__hip_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(x, n);
}

// ── Fused (bias add + GELU(tanh)) in-place, strict rounding ──
// The bias-added value is rounded to the storage dtype BEFORE the GELU
// (bit-matching an unfused add_bias → gelu_inplace pair). In-place on x;
// bias broadcast over rows.
template<typename T>
__global__ void bias_gelu_strict_kernel(T* __restrict__ x,
                                        const T* __restrict__ bias,
                                        int M, int N) {
    using T2 = typename packed2<T>::type;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total2 = (M * N) >> 1;        // pair index
    if (idx >= total2) return;
    int n2 = N >> 1;                  // pairs per row
    int row = idx / n2;
    int col_pair = idx - row * n2;
    T2* x2 = reinterpret_cast<T2*>(x);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    T2 v = x2[idx];
    T2 b = b2[col_pair];
    T v0_lp = from_f32<T>(to_f32(v.x) + to_f32(b.x));
    T v1_lp = from_f32<T>(to_f32(v.y) + to_f32(b.y));
    float v0 = to_f32(v0_lp);
    float v1 = to_f32(v1_lp);
    float t0 = tanhf(0.7978845608f * (v0 + 0.044715f * v0 * v0 * v0));
    float t1 = tanhf(0.7978845608f * (v1 + 0.044715f * v1 * v1 * v1));
    x2[idx] = make_packed2<T>(
        from_f32<T>(v0 * 0.5f * (1.0f + t0)),
        from_f32<T>(v1 * 0.5f * (1.0f + t1)));
    (void)row;
}

template __global__ void bias_gelu_strict_kernel<__half>(__half*, const __half*, int, int);
template __global__ void bias_gelu_strict_kernel<__hip_bfloat16>(__hip_bfloat16*, const __hip_bfloat16*, int, int);

void bias_gelu_inplace_bf16_strict(__hip_bfloat16* x,
                                   const __hip_bfloat16* bias,
                                   int M, int N, hipStream_t stream) {
    int total2 = (M * N) >> 1;
    bias_gelu_strict_kernel<__hip_bfloat16><<<(total2 + 255) / 256, 256, 0, stream>>>(
        x, bias, M, N);
}

// ── Gate GELU Mul Merged ──
// Input: (seq, 2*half_dim), gate = [:, :half_dim], up = [:, half_dim:]
template<typename T>
__global__ void gate_geglu_merged_kernel(const T* __restrict__ merged,
                                         T* __restrict__ out,
                                         int seq, int half_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * half_dim;
    if (idx < total) {
        int row = idx / half_dim;
        int col = idx % half_dim;
        int full_dim = half_dim * 2;
        float g = to_f32(merged[row * full_dim + col]);
        float u = to_f32(merged[row * full_dim + half_dim + col]);
        float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
        out[idx] = from_f32<T>(gelu * u);
    }
}

template __global__ void gate_geglu_merged_kernel<__half>(const __half*, __half*, int, int);
template __global__ void gate_geglu_merged_kernel<__hip_bfloat16>(const __hip_bfloat16*, __hip_bfloat16*, int, int);

void gate_geglu_merged(const __hip_bfloat16* merged, __hip_bfloat16* out,
                       int seq, int half_dim, hipStream_t stream) {
    int total = seq * half_dim;
    int blocks = (total + 255) / 256;
    gate_geglu_merged_kernel<__hip_bfloat16><<<blocks, 256, 0, stream>>>(merged, out, seq, half_dim);
}

// ── Gate GELU Mul Merged → FP8 ──
// Merged layout: merged[s, 0..H-1] = gate, merged[s, H..2H-1] = up
template<typename T>
__global__ void gate_geglu_merged_fp8_kernel(const T* __restrict__ merged,
                                                 __hip_fp8_e4m3* __restrict__ out,
                                                 int seq, int half_dim,
                                                 const float* __restrict__ d_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * half_dim;
    if (idx < total) {
        int row = idx / half_dim;
        int col = idx % half_dim;
        int full_dim = half_dim * 2;
        float g = to_f32(merged[row * full_dim + col]);
        float u = to_f32(merged[row * full_dim + half_dim + col]);
        float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
        float val = gelu * u;
        float inv_scale = 1.0f / (*d_scale);
        val = fminf(fmaxf(val * inv_scale, -448.0f), 448.0f);
        out[idx] = __hip_fp8_e4m3(val);
    }
}

template __global__ void gate_geglu_merged_fp8_kernel<__half>(const __half*, __hip_fp8_e4m3*, int, int, const float*);
template __global__ void gate_geglu_merged_fp8_kernel<__hip_bfloat16>(const __hip_bfloat16*, __hip_fp8_e4m3*, int, int, const float*);

void gate_geglu_merged_fp8(const __hip_bfloat16* merged, __hip_fp8_e4m3* out,
                               int seq, int half_dim,
                               const float* d_scale, hipStream_t stream) {
    int total = seq * half_dim;
    int blocks = (total + 255) / 256;
    gate_geglu_merged_fp8_kernel<__hip_bfloat16><<<blocks, 256, 0, stream>>>(
        merged, out, seq, half_dim, d_scale);
}
