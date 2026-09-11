// ================================================================
// FlashRT AMD — LayerNorm-style AdaLN kernels, BF16 + FP8 out
// (CDNA wave64). Additive port of:
//   layer_norm_no_affine_bf16, ada_layer_norm_bf16
//     (csrc/kernels/dit_bf16.cu)
//   ada_layer_norm_fp8 (csrc/quantize/ada_layer_norm_fp8.cu)
//   bias_gelu_quantize_fp8_static_bf16
//     (csrc/quantize/bias_gelu_quantize_fp8.cu)
//
// NOTE: distinct from the RMS-style ada_rms_norm_style in norm.hip —
// this family is LayerNorm math (mean/variance, no weight), with
// out = LN(x) * (1 + scale) + shift where scale/shift are (dim,)
// vectors broadcast over rows.
//
// ada_layer_norm_fp8 rounds the modulated value through bf16 in
// registers BEFORE the fp8 conversion, matching the CUDA kernel's
// bit-exact equivalence with the 2-launch (ada_layer_norm_bf16 →
// quantize_fp8_static) chain.
//
// Reductions use the tree's wave64 block_reduce_sum (common_hip.h);
// summation order differs from the CUDA warp32 originals — last-ULP
// differences only, gated by the cos parity suite.
//
// FP8 output dtype: OCP e4m3 via __hip_fp8_e4m3 (never fnuz);
// values pre-clamped to ±448 exactly as in the CUDA originals.
// ================================================================

#include "common_hip.h"
#include <hip/hip_fp8.h>

namespace {  // internal kernels

constexpr float kFp8Max = 448.0f;

// ── LayerNorm (no affine) — BF16, row per block ──
__global__ void aln_layer_norm_no_affine_bf16_kernel(
        const __hip_bfloat16* __restrict__ x,
        __hip_bfloat16* __restrict__ out,
        int dim, float eps) {
    int row = blockIdx.x;
    const __hip_bfloat162* x2 = reinterpret_cast<const __hip_bfloat162*>(x + (size_t)row * dim);
    __hip_bfloat162* out2 = reinterpret_cast<__hip_bfloat162*>(out + (size_t)row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 val = x2[i];
        local_sum += __bfloat162float(val.x) + __bfloat162float(val.y);
    }
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 v = x2[i];
        float d0 = __bfloat162float(v.x) - mean, d1 = __bfloat162float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 xv = x2[i];
        float v0 = (__bfloat162float(xv.x) - mean) * inv_std;
        float v1 = (__bfloat162float(xv.y) - mean) * inv_std;
        out2[i] = __halves2bfloat162(__float2bfloat16(v0), __float2bfloat16(v1));
    }
}

// ── AdaLayerNorm — BF16: out = LN(x, no_affine) * (1 + scale) + shift ──
// scale/shift: (dim,) broadcast over rows.
__global__ void aln_ada_layer_norm_bf16_kernel(
        const __hip_bfloat16* __restrict__ x,
        const __hip_bfloat16* __restrict__ scale,
        const __hip_bfloat16* __restrict__ shift,
        __hip_bfloat16* __restrict__ out,
        int dim, float eps) {
    int row = blockIdx.x;
    const __hip_bfloat162* x2 = reinterpret_cast<const __hip_bfloat162*>(x + (size_t)row * dim);
    const __hip_bfloat162* sc2 = reinterpret_cast<const __hip_bfloat162*>(scale);
    const __hip_bfloat162* sh2 = reinterpret_cast<const __hip_bfloat162*>(shift);
    __hip_bfloat162* out2 = reinterpret_cast<__hip_bfloat162*>(out + (size_t)row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 val = x2[i];
        local_sum += __bfloat162float(val.x) + __bfloat162float(val.y);
    }
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 v = x2[i];
        float d0 = __bfloat162float(v.x) - mean, d1 = __bfloat162float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 xv = x2[i], sv = sc2[i], hv = sh2[i];
        float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
        float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
        float v0 = n0 * (1.0f + __bfloat162float(sv.x)) + __bfloat162float(hv.x);
        float v1 = n1 * (1.0f + __bfloat162float(sv.y)) + __bfloat162float(hv.y);
        out2[i] = __halves2bfloat162(__float2bfloat16(v0), __float2bfloat16(v1));
    }
}

// ── AdaLayerNorm → static FP8 — BF16 input ──
// Register-level round-through bf16 BEFORE fp8 conversion (see header).
__global__ void aln_ada_layer_norm_fp8_kernel(
        const __hip_bfloat16* __restrict__ x,
        const __hip_bfloat16* __restrict__ scale,
        const __hip_bfloat16* __restrict__ shift,
        __hip_fp8_e4m3* __restrict__ out,
        const float* __restrict__ act_scale_ptr,
        int dim, float eps) {
    const int row = blockIdx.x;
    const __hip_bfloat162* x2 =
        reinterpret_cast<const __hip_bfloat162*>(x + (size_t)row * dim);
    const __hip_bfloat162* sc2 = reinterpret_cast<const __hip_bfloat162*>(scale);
    const __hip_bfloat162* sh2 = reinterpret_cast<const __hip_bfloat162*>(shift);
    __hip_fp8_e4m3* out_row = out + (size_t)row * dim;
    const int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 v = x2[i];
        local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
    }
    const float mean = block_reduce_sum(local_sum, shared) / static_cast<float>(dim);

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 v = x2[i];
        const float d0 = __bfloat162float(v.x) - mean;
        const float d1 = __bfloat162float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    const float inv_std =
        rsqrtf(block_reduce_sum(local_var, shared) / static_cast<float>(dim) + eps);

    const float inv_a = 1.0f / *act_scale_ptr;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __hip_bfloat162 xv = x2[i], sv = sc2[i], hv = sh2[i];
        const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
        const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
        const float v0_f32 = n0 * (1.0f + __bfloat162float(sv.x))
                              + __bfloat162float(hv.x);
        const float v1_f32 = n1 * (1.0f + __bfloat162float(sv.y))
                              + __bfloat162float(hv.y);
        // Round through bf16 in registers (no memory write).
        const float v0 = __bfloat162float(__float2bfloat16(v0_f32));
        const float v1 = __bfloat162float(__float2bfloat16(v1_f32));
        float q0 = fminf(fmaxf(v0 * inv_a, -kFp8Max), kFp8Max);
        float q1 = fminf(fmaxf(v1 * inv_a, -kFp8Max), kFp8Max);
        out_row[2 * i]     = __hip_fp8_e4m3(q0);
        out_row[2 * i + 1] = __hip_fp8_e4m3(q1);
    }
}

// ── Fused bias + GELU(tanh) + static FP8 quantize — BF16 input ──
__device__ __forceinline__ float aln_gelu_tanh(float x) {
    // 0.7978845608f = sqrt(2/pi) — matches gelu_kernel in activation.cu
    return 0.5f * x * (1.0f + tanhf(
        0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void aln_bias_gelu_quantize_fp8_static_bf16_kernel(
        const __hip_bfloat16* __restrict__ in,
        const __hip_bfloat16* __restrict__ bias,   // may be nullptr
        __hip_fp8_e4m3* __restrict__ out,
        const float* __restrict__ act_scale_ptr,
        long long total,                           // M * N
        int N,
        int has_bias) {
    const long long idx =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float v = __bfloat162float(in[idx]);
    if (has_bias) {
        int n = (int)(idx % N);
        v += __bfloat162float(bias[n]);
    }
    float g = aln_gelu_tanh(v);
    float inv_s = 1.0f / *act_scale_ptr;
    float q = g * inv_s;
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    out[idx] = __hip_fp8_e4m3(q);
}

}  // namespace

// ── Host launchers (CUDA-signature mirrors) ──

void layer_norm_no_affine_bf16(const __hip_bfloat16* x, __hip_bfloat16* out,
                               int seq_len, int dim, float eps,
                               hipStream_t stream) {
    aln_layer_norm_no_affine_bf16_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
        x, out, dim, eps);
}

void ada_layer_norm_bf16(const __hip_bfloat16* x,
                         const __hip_bfloat16* scale, const __hip_bfloat16* shift,
                         __hip_bfloat16* out, int seq_len, int dim, float eps,
                         hipStream_t stream) {
    aln_ada_layer_norm_bf16_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
        x, scale, shift, out, dim, eps);
}

void ada_layer_norm_fp8(const __hip_bfloat16* x,
                        const __hip_bfloat16* scale, const __hip_bfloat16* shift,
                        __hip_fp8_e4m3* out, const float* act_scale,
                        int seq_len, int dim, float eps,
                        hipStream_t stream) {
    if (seq_len <= 0 || dim <= 0) return;
    // dim must be even for bf162 vectorization (always true for our shapes).
    aln_ada_layer_norm_fp8_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
        x, scale, shift, out, act_scale, dim, eps);
}

void bias_gelu_quantize_fp8_static_bf16(const __hip_bfloat16* in,
                                        const __hip_bfloat16* bias,  // may be nullptr
                                        __hip_fp8_e4m3* out,
                                        const float* act_scale,
                                        long long M, int N,
                                        hipStream_t stream) {
    const long long total = M * (long long)N;
    const int block_sz = 256;
    const unsigned grid = (unsigned)((total + block_sz - 1) / block_sz);
    const int has_bias = (bias != nullptr) ? 1 : 0;
    aln_bias_gelu_quantize_fp8_static_bf16_kernel<<<grid, block_sz, 0, stream>>>(
        in, bias, out, act_scale, total, N, has_bias);
}
