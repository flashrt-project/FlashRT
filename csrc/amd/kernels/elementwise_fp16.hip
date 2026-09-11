// ================================================================
// FlashRT AMD — FP16 elementwise / activation / quantize kernels
// (CDNA wave64). Additive port of the CUDA FP16 backbone surface:
//   add_bias_fp16, gelu_inplace_fp16, silu_inplace_fp16, mul_fp16,
//   residual_add_fp16 (+_vec), gpu_strided_copy_fp16,
//   gpu_repeat_interleave_heads (+_vec), gpu_fill_neginf_fp16,
//   relu_inplace_bf16, cast_bf16_to_fp16 / cast_fp16_to_bf16,
//   concat2_bf16, quantize_fp8_static_fp16 (+_vec),
//   silu_mul_split_fp8_fp16
// Mirrors csrc/kernels/{decoder_fused.cu, activation.cu,
// elementwise.cu, dit_bf16.cu, quantize.cu, vec_fp16_backbone.cu}
// host-wrapper signatures and per-element math exactly (fp32
// intermediates, same clamps/constants).
//
// The *_vec entries keep the CUDA precondition contract (return -1
// when dim%8 / n%16 / 16-byte alignment does not hold, 0 on success)
// but dispatch to the same-math scalar kernels: element-independent
// ops (residual add, quantize, repeat-interleave) are bit-identical
// either way; wide-load retunes come later against profiles.
//
// FP8 output dtype: OCP e4m3 via <hip/hip_fp8.h> __hip_fp8_e4m3
// (saturating RNE conversion, same as __nv_fp8_e4m3; values are
// pre-clamped to ±448 exactly as in the CUDA originals). Never fnuz.
//
// Existing kernel files are untouched (additive-only discipline).
// ================================================================

#include "common_hip.h"
#include <hip/hip_fp8.h>
#include <algorithm>
#include <cstdint>

namespace {  // internal kernels — avoid any symbol overlap with the
             // template instantiations in activation.hip/elementwise.hip

// ── Simple bias add: x[i] += b[i % D] (pi05 bias_k) ──
__global__ void ew16_add_bias_fp16_kernel(__half* x, const __half* b, int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S * D) x[i] = __float2half(__half2float(x[i]) + __half2float(b[i % D]));
}

// ── GELU in-place (tanh approx), packed __half2 ──
__global__ void ew16_gelu_fp16_kernel(__half* __restrict__ x, int n) {
    __half2* x2 = reinterpret_cast<__half2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        __half2 val = x2[idx];
        float v0 = __half2float(val.x), v1 = __half2float(val.y);
        float t0 = tanhf(0.7978845608f * (v0 + 0.044715f * v0 * v0 * v0));
        float t1 = tanhf(0.7978845608f * (v1 + 0.044715f * v1 * v1 * v1));
        x2[idx] = __halves2half2(
            __float2half(v0 * 0.5f * (1.0f + t0)),
            __float2half(v1 * 0.5f * (1.0f + t1)));
    }
}

// ── SiLU in-place, packed __half2 ──
__global__ void ew16_silu_fp16_kernel(__half* __restrict__ x, int n) {
    __half2* x2 = reinterpret_cast<__half2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        __half2 val = x2[idx];
        float v0 = __half2float(val.x), v1 = __half2float(val.y);
        float s0 = v0 / (1.0f + expf(-v0));
        float s1 = v1 / (1.0f + expf(-v1));
        x2[idx] = __halves2half2(__float2half(s0), __float2half(s1));
    }
}

// ── ReLU in-place, packed __hip_bfloat162 ──
__global__ void ew16_relu_bf16_kernel(__hip_bfloat16* __restrict__ x, int n) {
    __hip_bfloat162* x2 = reinterpret_cast<__hip_bfloat162*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        __hip_bfloat162 val = x2[idx];
        float v0 = fmaxf(__bfloat162float(val.x), 0.0f);
        float v1 = fmaxf(__bfloat162float(val.y), 0.0f);
        x2[idx] = __halves2bfloat162(__float2bfloat16(v0), __float2bfloat16(v1));
    }
}

// ── Elementwise multiply, 8 halves/thread (float4 loads) + scalar tail.
// Same math as mul_fp16_kernel in activation.cu: half-precision __hmul2
// on the vector path, fp32 multiply on the tail.
__global__ void ew16_mul_fp16_kernel(const __half* __restrict__ a,
                                     const __half* __restrict__ b,
                                     __half* __restrict__ out, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 > n) {
        for (int k = 0; k < 8 && i + k < n; ++k) {
            float av = __half2float(a[i + k]);
            float bv = __half2float(b[i + k]);
            out[i + k] = __float2half(av * bv);
        }
        return;
    }
    const float4* a4 = reinterpret_cast<const float4*>(a + i);
    const float4* b4 = reinterpret_cast<const float4*>(b + i);
    float4 av = *a4, bv = *b4;
    __half2 ah[4], bh[4], oh[4];
    ah[0] = *reinterpret_cast<__half2*>(&av.x);
    ah[1] = *reinterpret_cast<__half2*>(&av.y);
    ah[2] = *reinterpret_cast<__half2*>(&av.z);
    ah[3] = *reinterpret_cast<__half2*>(&av.w);
    bh[0] = *reinterpret_cast<__half2*>(&bv.x);
    bh[1] = *reinterpret_cast<__half2*>(&bv.y);
    bh[2] = *reinterpret_cast<__half2*>(&bv.z);
    bh[3] = *reinterpret_cast<__half2*>(&bv.w);
    #pragma unroll
    for (int k = 0; k < 4; ++k) oh[k] = __hmul2(ah[k], bh[k]);
    float4 ov;
    ov.x = *reinterpret_cast<float*>(&oh[0]);
    ov.y = *reinterpret_cast<float*>(&oh[1]);
    ov.z = *reinterpret_cast<float*>(&oh[2]);
    ov.w = *reinterpret_cast<float*>(&oh[3]);
    *reinterpret_cast<float4*>(out + i) = ov;
}

// ── Residual add: residual += x, packed __half2, fp32 add ──
__global__ void ew16_res_add_fp16_kernel(__half* __restrict__ residual,
                                         const __half* __restrict__ x, int n) {
    __half2* res2 = reinterpret_cast<__half2*>(residual);
    const __half2* x2 = reinterpret_cast<const __half2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        __half2 rv = res2[idx], xv = x2[idx];
        res2[idx] = __halves2half2(
            __float2half(__half2float(rv.x) + __half2float(xv.x)),
            __float2half(__half2float(rv.y) + __half2float(xv.y)));
    }
}

// ── Fill FP16 buffer with large negative (softmax masking) ──
__global__ void ew16_fill_neginf_fp16_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = __float2half(-1e30f);
}

// ── Strided copy: src[rows, src_stride] cols [col_offset, col_offset+dst_cols)
//    → dst[rows, dst_cols] ──
__global__ void ew16_strided_copy_fp16_kernel(const __half* src, __half* dst,
                                              int rows, int dst_cols,
                                              int src_stride, int col_offset) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    for (int cc = c; cc < dst_cols; cc += blockDim.x) {
        dst[r * dst_cols + cc] = src[r * src_stride + col_offset + cc];
    }
}

// ── GQA repeat interleave: src [S, NH_src*HD] → dst [S, NH_src*repeat*HD] ──
__global__ void ew16_repeat_interleave_heads_kernel(
        const __half* src, __half* dst,
        int S, int NH_src, int HD, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int NH_dst = NH_src * repeat;
    int total = S * NH_dst * HD;
    if (idx >= total) return;

    int d = idx % HD;
    int remainder = idx / HD;
    int h_dst = remainder % NH_dst;
    int s = remainder / NH_dst;

    int h_src = h_dst / repeat;
    dst[s * NH_dst * HD + h_dst * HD + d] = src[s * NH_src * HD + h_src * HD + d];
}

// ── Cast fp16 ↔ bf16 (DiT input/output boundary) ──
__global__ void ew16_cast_fp16_to_bf16_kernel(const __half* in, __hip_bfloat16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__half2float(in[i]));
}

__global__ void ew16_cast_bf16_to_fp16_kernel(const __hip_bfloat16* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__bfloat162float(in[i]));
}

// ── Row-wise concat: out[r,:] = [a[r,:cols_a] | b[r,:cols_b]] ──
__global__ void ew16_concat2_bf16_kernel(
        const __hip_bfloat16* __restrict__ a,
        const __hip_bfloat16* __restrict__ b,
        __hip_bfloat16* __restrict__ out,
        int rows, int cols_a, int cols_b) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    if (r >= rows) return;
    const int cols = cols_a + cols_b;
    for (int cc = c; cc < cols; cc += blockDim.x) {
        if (cc < cols_a) {
            out[r * cols + cc] = a[r * cols_a + cc];
        } else {
            const int bc = cc - cols_a;
            out[r * cols + cc] = b[r * cols_b + bc];
        }
    }
}

// ── Static FP8 quantize, FP16 input, 4 elem/thread (pi05 quant_fp8_static_k).
// n is assumed to be a multiple of 4 (production shapes are); inv_scale
// guards against a zero device scale exactly like quantize.cu.
__global__ void ew16_quantize_fp8_fp16_kernel(const __half* in,
                                              __hip_fp8_e4m3* out,
                                              const float* descale_ptr, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    const __half2* in2 = reinterpret_cast<const __half2*>(in);
    __half2 vA = in2[i/2], vB = in2[i/2+1];
    float fv[4] = {__half2float(vA.x), __half2float(vA.y),
                   __half2float(vB.x), __half2float(vB.y)};
    unsigned char pk[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        pk[j] = __hip_fp8_e4m3(fminf(fmaxf(fv[j] * inv_scale, -448.f), 448.f)).__x;
    }
    uint32_t w;
    __builtin_memcpy(&w, pk, 4);
    *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(out) + i) = w;
}

// ── Split SiLU × Up → FP8 (separate gate/up buffers, pi05 silu_mul_split_fp8_k).
// Note: no 1e-12 guard on the scale here — matches activation.cu exactly.
__global__ void ew16_silu_mul_split_fp8_fp16_kernel(
        const __half* __restrict__ gate,
        const __half* __restrict__ up,
        __hip_fp8_e4m3* __restrict__ out,
        int n, const float* __restrict__ d_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float inv_scale = 1.0f / (*d_scale);
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    float val = silu_g * u * inv_scale;
    out[idx] = __hip_fp8_e4m3(fminf(fmaxf(val, -448.0f), 448.0f));
}

}  // namespace

// ── Host launchers (CUDA-signature mirrors) ──

void add_bias_fp16(__half* x, const __half* b, int S, int D, hipStream_t stream) {
    ew16_add_bias_fp16_kernel<<<(S*D + 255)/256, 256, 0, stream>>>(x, b, S, D);
}

void gelu_inplace_fp16(__half* x, int n, hipStream_t stream) {
    int n2 = n >> 1;
    ew16_gelu_fp16_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(x, n);
}

void silu_inplace_fp16(__half* x, int n, hipStream_t stream) {
    int n2 = n >> 1;
    ew16_silu_fp16_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(x, n);
}

void relu_inplace_bf16(__hip_bfloat16* x, int n, hipStream_t stream) {
    int n2 = n >> 1;
    ew16_relu_bf16_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(x, n);
}

void mul_fp16(const __half* a, const __half* b, __half* out, int n, hipStream_t stream) {
    int per_block = 256 * 8;
    int blocks = (n + per_block - 1) / per_block;
    ew16_mul_fp16_kernel<<<blocks, 256, 0, stream>>>(a, b, out, n);
}

void residual_add_fp16(__half* residual, const __half* x, int n,
                       hipStream_t stream) {
    int n2 = n >> 1;
    ew16_res_add_fp16_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, n);
}

void gpu_fill_neginf_fp16(__half* dst, int n, hipStream_t stream) {
    ew16_fill_neginf_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, n);
}

void gpu_strided_copy_fp16(const __half* src, __half* dst,
                           int rows, int dst_cols, int src_stride, int col_offset,
                           hipStream_t stream) {
    ew16_strided_copy_fp16_kernel<<<rows, std::min(256, dst_cols), 0, stream>>>(
        src, dst, rows, dst_cols, src_stride, col_offset);
}

void gpu_repeat_interleave_heads(const __half* src, __half* dst,
                                 int S, int NH_src, int HD, int repeat,
                                 hipStream_t stream) {
    int NH_dst = NH_src * repeat;
    int total = S * NH_dst * HD;
    ew16_repeat_interleave_heads_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        src, dst, S, NH_src, HD, repeat);
}

void cast_fp16_to_bf16(const __half* in, __hip_bfloat16* out, int n,
                       hipStream_t stream) {
    ew16_cast_fp16_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(in, out, n);
}

void cast_bf16_to_fp16(const __hip_bfloat16* in, __half* out, int n,
                       hipStream_t stream) {
    ew16_cast_bf16_to_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(in, out, n);
}

void concat2_bf16(const __hip_bfloat16* a, const __hip_bfloat16* b,
                  __hip_bfloat16* out,
                  int rows, int cols_a, int cols_b,
                  hipStream_t stream) {
    const int cols = cols_a + cols_b;
    ew16_concat2_bf16_kernel<<<rows, std::min(256, cols), 0, stream>>>(
        a, b, out, rows, cols_a, cols_b);
}

void quantize_fp8_static_fp16(const __half* input, __hip_fp8_e4m3* output,
                              const float* d_scale, int n, hipStream_t stream) {
    // 4 elem/thread, matching production quant_fp8_static_k
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    ew16_quantize_fp8_fp16_kernel<<<blocks, threads, 0, stream>>>(input, output, d_scale, n);
}

void silu_mul_split_fp8_fp16(const __half* gate, const __half* up,
                             __hip_fp8_e4m3* out, int n,
                             const float* d_scale, hipStream_t stream) {
    ew16_silu_mul_split_fp8_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        gate, up, out, n, d_scale);
}

// ── *_vec entries: CUDA precondition contract, same-math dispatch ──
// (element-independent → bit-identical to the CUDA vec kernels)

int residual_add_fp16_vec(__half* residual, const __half* x, int n,
                          hipStream_t stream) {
    if (n % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(residual) & 15) ||
        (reinterpret_cast<uintptr_t>(x) & 15)) return -1;
    residual_add_fp16(residual, x, n, stream);
    const hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : -static_cast<int>(e);
}

int quantize_fp8_static_fp16_vec(const __half* in, __hip_fp8_e4m3* out,
                                 const float* descale_ptr, int n,
                                 hipStream_t stream) {
    if (n % 16 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(in) & 15) ||
        (reinterpret_cast<uintptr_t>(out) & 15)) return -1;
    quantize_fp8_static_fp16(in, out, descale_ptr, n, stream);
    const hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : -static_cast<int>(e);
}

int gpu_repeat_interleave_heads_vec(const __half* src, __half* dst,
                                    int S, int NH_src, int HD, int repeat,
                                    hipStream_t stream) {
    if (HD % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(src) & 15) ||
        (reinterpret_cast<uintptr_t>(dst) & 15)) return -1;
    gpu_repeat_interleave_heads(src, dst, S, NH_src, HD, repeat, stream);
    const hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : -static_cast<int>(e);
}
