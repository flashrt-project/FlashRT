// ================================================================
// FlashRT AMD — FP16 norm / RoPE / norm→FP8 kernels (CDNA wave64).
// Additive port of the CUDA FP16 backbone surface:
//   layer_norm_fp16 (+_vec), rms_norm_fp16_vec,
//   rope_rotate_half_fp16 (+_vec),
//   rms_norm_fp8_fp16, residual_add_rms_norm_fp8_fp16,
//   layer_norm_fp8_static_fp16_vec
// Mirrors csrc/kernels/norm.cu, csrc/kernels/rope_qwen3.cu and
// csrc/kernels/vec_fp16_backbone.cu host-wrapper signatures and
// per-element math exactly (fp32 intermediates, clamp ±448, and the
// fp16 round-through before quantize in layer_norm_fp8_static_*).
//
// Reductions use the tree's wave64 block_reduce_sum (common_hip.h);
// per-thread accumulation order therefore differs from the CUDA
// warp32 originals — last-ULP differences only, gated by the cos
// parity suite (same policy as layer_norm_wide_kernel in norm.hip).
//
// The *_vec entries keep the CUDA precondition contract (return -1
// when dim%8 / alignment does not hold, 0 on success) but dispatch
// to the same-math row-per-block kernels; wide-load retunes come
// later against profiles. rms_norm_fp16_vec binds to the existing
// rms_norm_fp16 (kernels/norm.hip) — same math, not duplicated.
//
// FP8 output dtype: OCP e4m3 via __hip_fp8_e4m3 (never fnuz).
// ================================================================

#include "common_hip.h"
#include <hip/hip_fp8.h>
#include <cstdint>

// Existing launcher from kernels/norm.hip (host-side call only).
void rms_norm_fp16(const __half* x, const __half* weight,
                   __half* out, int seq_len, int dim, float eps,
                   hipStream_t stream);

namespace {  // internal kernels

// ── LayerNorm (gamma/beta), FP16, row per block, packed __half2 ──
__global__ void nrm16_layer_norm_fp16_kernel(const __half* __restrict__ x,
                                             const __half* __restrict__ weight,
                                             const __half* __restrict__ bias,
                                             __half* __restrict__ out,
                                             int dim, float eps) {
    int row = blockIdx.x;
    const __half2* x2 = reinterpret_cast<const __half2*>(x + (size_t)row * dim);
    __half2* out2 = reinterpret_cast<__half2*>(out + (size_t)row * dim);
    const __half2* w2 = reinterpret_cast<const __half2*>(weight);
    const __half2* b2 = reinterpret_cast<const __half2*>(bias);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        local_sum += __half2float(val.x) + __half2float(val.y);
    }
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        float d0 = __half2float(val.x) - mean, d1 = __half2float(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 xv = x2[i], wv = w2[i], bv = b2[i];
        float v0 = (__half2float(xv.x) - mean) * inv_std * __half2float(wv.x) + __half2float(bv.x);
        float v1 = (__half2float(xv.y) - mean) * inv_std * __half2float(wv.y) + __half2float(bv.y);
        out2[i] = __halves2half2(__float2half(v0), __float2half(v1));
    }
}

// ── LayerNorm (gamma/beta) → static FP8, FP16 input.
// Norm output is rounded through fp16 BEFORE the fp8 quantize, keeping
// this identical to the two-step (fp16 LayerNorm, then static quantize)
// chain — exactly like layer_norm_fp8_static_fp16_vec_kernel in
// vec_fp16_backbone.cu (which also guards the scale with 1e-12).
__global__ void nrm16_layer_norm_fp8_static_fp16_kernel(
        const __half* __restrict__ x, const __half* __restrict__ weight,
        const __half* __restrict__ bias, __hip_fp8_e4m3* __restrict__ out,
        const float* __restrict__ descale_ptr, int dim, float eps) {
    int row = blockIdx.x;
    const __half2* x2 = reinterpret_cast<const __half2*>(x + (size_t)row * dim);
    const __half2* w2 = reinterpret_cast<const __half2*>(weight);
    const __half2* b2 = reinterpret_cast<const __half2*>(bias);
    __hip_fp8_e4m3* out_row = out + (size_t)row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        local_sum += __half2float(val.x) + __half2float(val.y);
    }
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        float d0 = __half2float(val.x) - mean, d1 = __half2float(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);
    const float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 xv = x2[i], wv = w2[i], bv = b2[i];
        // fp16 round-through of the norm output (see header comment).
        const float v0 = __half2float(__float2half(
            (__half2float(xv.x) - mean) * inv_std * __half2float(wv.x) + __half2float(bv.x)));
        const float v1 = __half2float(__float2half(
            (__half2float(xv.y) - mean) * inv_std * __half2float(wv.y) + __half2float(bv.y)));
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(v0 * inv_scale, -448.f), 448.f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(v1 * inv_scale, -448.f), 448.f));
    }
}

// ── Split-half RoPE (rotate_half, Qwen3/LLaMA format), in-place.
// x: [S, NH*HD]; cos/sin tables: [S, HD] (first HD/2 entries used per
// position, broadcast over heads). Each thread owns one (d, d+HD/2) pair.
__global__ void nrm16_rope_rotate_half_fp16_kernel(
        __half* __restrict__ x,
        const __half* __restrict__ cos_t,
        const __half* __restrict__ sin_t,
        int S, int NH, int HD) {
    int half_hd = HD / 2;
    int total_pairs = S * NH * half_hd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int d = idx % half_hd;
    int remainder = idx / half_hd;
    int h = remainder % NH;
    int s = remainder / NH;

    int rope_idx = s * HD + d;
    float c = __half2float(cos_t[rope_idx]);
    float si = __half2float(sin_t[rope_idx]);

    int base = s * NH * HD + h * HD;
    float x_lo = __half2float(x[base + d]);
    float x_hi = __half2float(x[base + d + half_hd]);

    x[base + d]           = __float2half(x_lo * c - x_hi * si);
    x[base + d + half_hd] = __float2half(x_hi * c + x_lo * si);
}

// ── RMSNorm → FP8, FP16 input (static device scale) ──
__global__ void nrm16_rms_norm_fp8_fp16_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ weight,
        __hip_fp8_e4m3* __restrict__ out,
        int dim, float eps, const float* __restrict__ d_scale) {
    int row = blockIdx.x;
    const __half2* x2 = reinterpret_cast<const __half2*>(x + (size_t)row * dim);
    const __half2* w2 = reinterpret_cast<const __half2*>(weight);
    __hip_fp8_e4m3* out_row = out + (size_t)row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        float v0 = __half2float(val.x), v1 = __half2float(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 xv = x2[i], wv = w2[i];
        float v0 = __half2float(xv.x) * rms * __half2float(wv.x) * inv_scale;
        float v1 = __half2float(xv.y) * rms * __half2float(wv.y) * inv_scale;
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

// ── Residual add + RMSNorm → FP8, FP16 (pass 1 writes the rounded
// residual back; sum-of-squares from the UNROUNDED float sums; pass 2
// re-reads the rounded residual — matching norm.cu exactly).
__global__ void nrm16_residual_add_rms_norm_fp8_fp16_kernel(
        __half* __restrict__ residual, const __half* __restrict__ x,
        const __half* __restrict__ weight, __hip_fp8_e4m3* __restrict__ out,
        int dim, float eps, const float* __restrict__ d_scale) {
    int row = blockIdx.x;
    __half2* res2 = reinterpret_cast<__half2*>(residual + (size_t)row * dim);
    const __half2* x2 = reinterpret_cast<const __half2*>(x + (size_t)row * dim);
    const __half2* w2 = reinterpret_cast<const __half2*>(weight);
    __hip_fp8_e4m3* out_row = out + (size_t)row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 rv = res2[i], xv = x2[i];
        float r0 = __half2float(rv.x) + __half2float(xv.x);
        float r1 = __half2float(rv.y) + __half2float(xv.y);
        res2[i] = __halves2half2(__float2half(r0), __float2half(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 rv = res2[i], wv = w2[i];
        float v0 = __half2float(rv.x) * rms * __half2float(wv.x) * inv_scale;
        float v1 = __half2float(rv.y) * rms * __half2float(wv.y) * inv_scale;
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

}  // namespace

// ── Host launchers (CUDA-signature mirrors) ──

void layer_norm_fp16(const __half* x, const __half* weight,
                     const __half* bias, __half* out,
                     int seq_len, int dim, float eps, hipStream_t stream) {
    nrm16_layer_norm_fp16_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
        x, weight, bias, out, dim, eps);
}

void rope_rotate_half_fp16(__half* x, const __half* cos_table,
                           const __half* sin_table,
                           int S, int NH, int HD, hipStream_t stream) {
    int total_pairs = S * NH * (HD / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    nrm16_rope_rotate_half_fp16_kernel<<<blocks, threads, 0, stream>>>(
        x, cos_table, sin_table, S, NH, HD);
}

void rms_norm_fp8_fp16(const __half* x, const __half* weight,
                       __hip_fp8_e4m3* out, int seq_len, int dim, float eps,
                       const float* d_scale, hipStream_t stream) {
    nrm16_rms_norm_fp8_fp16_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
        x, weight, out, dim, eps, d_scale);
}

void residual_add_rms_norm_fp8_fp16(__half* residual, const __half* x,
                                    const __half* weight, __hip_fp8_e4m3* out,
                                    int seq_len, int dim, float eps,
                                    const float* d_scale, hipStream_t stream) {
    nrm16_residual_add_rms_norm_fp8_fp16_kernel<<<seq_len, 256, 4 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps, d_scale);
}

// ── *_vec entries: CUDA precondition contract, same-math dispatch ──

int rms_norm_fp16_vec(const __half* x, const __half* w, __half* out,
                      int rows, int dim, float eps, hipStream_t stream) {
    if (dim % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(w) & 15) ||
        (reinterpret_cast<uintptr_t>(out) & 15)) return -1;
    rms_norm_fp16(x, w, out, rows, dim, eps, stream);
    const hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : -static_cast<int>(e);
}

int layer_norm_fp16_vec(const __half* x, const __half* w, const __half* b,
                        __half* out, int rows, int dim, float eps,
                        hipStream_t stream) {
    if (dim % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(w) & 15) ||
        (reinterpret_cast<uintptr_t>(b) & 15) ||
        (reinterpret_cast<uintptr_t>(out) & 15)) return -1;
    layer_norm_fp16(x, w, b, out, rows, dim, eps, stream);
    const hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : -static_cast<int>(e);
}

int layer_norm_fp8_static_fp16_vec(const __half* x, const __half* w,
                                   const __half* b, __hip_fp8_e4m3* out,
                                   const float* d_scale, int rows, int dim,
                                   float eps, hipStream_t stream) {
    if (dim % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(w) & 15) ||
        (reinterpret_cast<uintptr_t>(b) & 15) ||
        (reinterpret_cast<uintptr_t>(out) & 7)) return -1;
    nrm16_layer_norm_fp8_static_fp16_kernel<<<rows, 256, 4 * sizeof(float), stream>>>(
        x, w, b, out, d_scale, dim, eps);
    const hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : -static_cast<int>(e);
}

int rope_rotate_half_fp16_vec(__half* x, const __half* cos_t,
                              const __half* sin_t, int S, int NH, int HD,
                              hipStream_t stream) {
    const int half_hd = HD / 2;
    if (half_hd % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(cos_t) & 15) ||
        (reinterpret_cast<uintptr_t>(sin_t) & 15)) return -1;
    rope_rotate_half_fp16(x, cos_t, sin_t, S, NH, HD, stream);
    const hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : -static_cast<int>(e);
}
