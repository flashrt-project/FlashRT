// FlashAttention-4 (AOT) for the SigLIP vision attention (Thor SM110).
//
// The vendored fa4_aot/ artifacts are the CuTe-DSL AOT export of the FA4
// SM100-compatible forward compiled for sm_110a at head_dim 80 (the
// padded-head layout this adapter's vision path already uses). The kernel
// takes (batch, seq, heads, head_dim) f16 tensors with arbitrary leading
// strides; softmax scale is a runtime argument.
//
// The ggml flash-attention node's padded Q/K/V/dst all share one linear
// layout (d + h*hd + s*hd*H + b*hd*H*S), which is exactly FA4's
// (B, S, H, D) with strides (S*H*hd, H*hd, hd) — so the f32 Q input and
// f32 output only need dense elementwise converts, and the f16 K/V pass
// straight through.

#include "fr_kernels.h"

#include "fa4_aot/fa4_siglip_fwd.h"
#include "fa4_aot/fa4_prefill_fwd.h"

#include <cuda_fp16.h>

namespace ggml_cuda_flashrt {

namespace {

__global__ void kernel_f32_to_f16_dense(const float * __restrict__ src,
                                        __half * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void kernel_f16_to_f32_dense(const __half * __restrict__ src,
                                        float * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

// convert dropping the head padding: one block per (b, s) token; src rows
// are H heads of D elements, dst rows are H packed slices of D2 (< D)
__global__ void kernel_f16_to_f32_depad(const __half * __restrict__ src,
                                        float * __restrict__ dst,
                                        int H, int D, int D2) {
    const int64_t row = blockIdx.x;
    const __half * s = src + row * (int64_t) H * D;
    float * d = dst + row * (int64_t) H * D2;
    for (int t = threadIdx.x; t < H * D2; t += blockDim.x) {
        const int h = t / D2;
        const int e = t - h * D2;
        d[t] = __half2float(s[(int64_t) h * D + e]);
    }
}

fa4_siglip_fwd_Kernel_Module_t g_fa4_module;
bool g_fa4_loaded = false;

fa4_prefill_fwd_Kernel_Module_t g_fa4p_module;
bool g_fa4p_loaded = false;

} // namespace

// Loads the AOT module once; must not run during CUDA graph capture (the
// caller checks and falls back to the unfused path on the first capture).
int fa4_vit_ensure_loaded(cudaStream_t stream) {
    if (g_fa4_loaded) {
        return 0;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        return -1;
    }
    fa4_siglip_fwd_Kernel_Module_Load(&g_fa4_module);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        return -static_cast<int>(e);
    }
    g_fa4_loaded = true;
    fa4_prefill_fwd_Kernel_Module_Load(&g_fa4p_module);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        return -static_cast<int>(e);
    }
    g_fa4p_loaded = true;
    return 0;
}

// Full (non-causal) self-attention for the pi0.5 prefill: hd-256 GQA FA4.
// q_f32/dst_f32 dense (1, S, H, D); k16/v16 are the first S contiguous
// [D]-rows of the (possibly padded) f16 KV buffers, one KV head.
// The padded tail rows are simply outside the dynamic shape, which is
// exactly the graph's row-uniform pad mask.
int fa4_prefill_attention(const float * q_f32, const void * k16, const void * v16,
                          float * dst_f32, void * q16_ws, void * o16_ws,
                          int S, int H, int D, float scale,
                          cudaStream_t stream) {
    if (!g_fa4p_loaded) {
        return -1;
    }
    const int64_t n = (int64_t) S * H * D;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;

    kernel_f32_to_f16_dense<<<(unsigned) blocks, threads, 0, stream>>>(
        q_f32, (__half *) q16_ws, n);

    auto fill_q = [&](void * data, auto * t, int heads) {
        t->data = data;
        t->dynamic_shapes[0] = 1;
        t->dynamic_shapes[1] = S;
        t->dynamic_shapes[2] = heads;
        t->dynamic_shapes[3] = D;
        t->dynamic_strides[0] = (int64_t) S * heads * D;
        t->dynamic_strides[1] = (int64_t) heads * D;
        t->dynamic_strides[2] = D;
    };
    fa4_prefill_fwd_Tensor_mQ_t tq; fill_q(q16_ws, &tq, H);
    fa4_prefill_fwd_Tensor_mK_t tk; fill_q(const_cast<void *>(k16), &tk, 1);
    fa4_prefill_fwd_Tensor_mV_t tv; fill_q(const_cast<void *>(v16), &tv, 1);
    fa4_prefill_fwd_Tensor_mO_t to; fill_q(o16_ws, &to, H);

    const int32_t rc = cute_dsl_fa4_prefill_fwd_wrapper(
        &g_fa4p_module, &tq, &tk, &tv, &to, scale, stream);
    if (rc != 0) {
        return -1000 - rc;
    }

    kernel_f16_to_f32_dense<<<(unsigned) blocks, threads, 0, stream>>>(
        (const __half *) o16_ws, dst_f32, n);

    const cudaError_t e2 = cudaGetLastError();
    return (e2 == cudaSuccess) ? 0 : -static_cast<int>(e2);
}

// q_f32: dense (B,S,H,D); k16/v16: dense f16 same layout. When d_out == D
// dst_f32 is the dense padded layout; when d_out < D the head padding is
// dropped and dst_f32 is the packed [(H*d_out), S, B] contiguous tensor.
// q16_ws / o16_ws: workspaces of B*S*H*D halves.
int fa4_vit_attention(const float * q_f32, const void * k16, const void * v16,
                      float * dst_f32, int d_out, void * q16_ws, void * o16_ws,
                      int B, int S, int H, int D, float scale,
                      cudaStream_t stream) {
    if (!g_fa4_loaded) {
        return -1;
    }
    const int64_t n = (int64_t) B * S * H * D;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;

    kernel_f32_to_f16_dense<<<(unsigned) blocks, threads, 0, stream>>>(
        q_f32, (__half *) q16_ws, n);

    auto fill = [&](void * data, auto * t) {
        t->data = data;
        t->dynamic_shapes[0] = B;
        t->dynamic_shapes[1] = S;
        t->dynamic_shapes[2] = H;
        t->dynamic_shapes[3] = D;
        t->dynamic_strides[0] = (int64_t) S * H * D;
        t->dynamic_strides[1] = (int64_t) H * D;
        t->dynamic_strides[2] = D;
    };
    fa4_siglip_fwd_Tensor_mQ_t tq; fill(q16_ws, &tq);
    fa4_siglip_fwd_Tensor_mK_t tk; fill(const_cast<void *>(k16), &tk);
    fa4_siglip_fwd_Tensor_mV_t tv; fill(const_cast<void *>(v16), &tv);
    fa4_siglip_fwd_Tensor_mO_t to; fill(o16_ws, &to);

    const int32_t rc = cute_dsl_fa4_siglip_fwd_wrapper(
        &g_fa4_module, &tq, &tk, &tv, &to, scale, stream);
    if (rc != 0) {
        return -1000 - rc;
    }

    if (d_out == D) {
        kernel_f16_to_f32_dense<<<(unsigned) blocks, threads, 0, stream>>>(
            (const __half *) o16_ws, dst_f32, n);
    } else {
        kernel_f16_to_f32_depad<<<(unsigned) ((int64_t) B * S), threads, 0, stream>>>(
            (const __half *) o16_ws, dst_f32, H, D, d_out);
    }

    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

} // namespace ggml_cuda_flashrt
