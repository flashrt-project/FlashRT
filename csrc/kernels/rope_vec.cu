// ============================================================================
//  FlashRT — vectorized QKV split + RoPE + KV-cache append (FP16).
//
//  Bit-exact drop-in for qkv_split_rope_kvcache_fp16 (rope.cu) on shapes
//  where every region is 16-byte aligned: the per-pair rotation arithmetic
//  is identical, only the memory access width changes. The scalar kernel
//  issues one 2-byte load per thread, which starves memory-level
//  parallelism at decoder sizes; this version moves 8 halves (one int4)
//  per thread.
//
//  Additive: the scalar kernel remains the fallback for unaligned shapes.
// ============================================================================
#include <cuda_fp16.h>

#include "rope_vec.cuh"

namespace {

// Rotate 4 interleaved (x0, x1) pairs held in one int4 with the matching
// 4 (cos, sin) pairs. Identical arithmetic to the scalar kernel.
__device__ __forceinline__ int4 rope_rotate4(int4 x_raw, int4 cs_raw) {
    const __half2* x = reinterpret_cast<const __half2*>(&x_raw);
    const __half2* cs = reinterpret_cast<const __half2*>(&cs_raw);
    int4 out_raw;
    __half2* out = reinterpret_cast<__half2*>(&out_raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = __half2float(x[i].x);
        const float x1 = __half2float(x[i].y);
        const float c = __half2float(cs[i].x);
        const float s = __half2float(cs[i].y);
        out[i] = __halves2half2(__float2half(x0 * c - x1 * s),
                                __float2half(x1 * c + x0 * s));
    }
    return out_raw;
}

__global__ void qkv_split_rope_kvcache_fp16_vec_kernel(
    const int4* __restrict__ qkv, const int4* __restrict__ rope,
    int4* __restrict__ Q, int4* __restrict__ Kc, int4* __restrict__ Vc,
    int S, int Q_dim8, int K_dim8, int HD8, int qkv_stride8,
    long kc_offset8, int kc_stride8) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = S * qkv_stride8;
    if (idx >= total) return;
    const int s = idx / qkv_stride8;
    const int c8 = idx - s * qkv_stride8;

    if (c8 < Q_dim8) {
        const int hc8 = c8 & (HD8 - 1);       // HD8 is a power of two
        const int4 x = qkv[s * qkv_stride8 + c8];
        const int4 cs = rope[s * HD8 + hc8];
        Q[s * Q_dim8 + c8] = rope_rotate4(x, cs);
    } else if (c8 < Q_dim8 + K_dim8) {
        const int k8 = c8 - Q_dim8;
        const int4 x = qkv[s * qkv_stride8 + c8];
        const int4 cs = rope[s * HD8 + k8];
        Kc[kc_offset8 + s * kc_stride8 + k8] = rope_rotate4(x, cs);
    } else {
        const int v8 = c8 - Q_dim8 - K_dim8;
        Vc[kc_offset8 + s * kc_stride8 + v8] = qkv[s * qkv_stride8 + c8];
    }
}

}  // namespace

int qkv_split_rope_kvcache_fp16_vec(
    const __half* qkv, const __half* rope,
    __half* Q, __half* Kc, __half* Vc,
    int S, int Q_dim, int K_dim, int HD, int qkv_stride,
    long kc_offset, int kc_stride,
    cudaStream_t stream) {
    // 16-byte alignment of every region and power-of-two head dim.
    if ((Q_dim | K_dim | HD | qkv_stride | kc_stride) & 7) return -1;
    if (kc_offset & 7) return -1;
    if (HD & (HD - 1)) return -1;
    if (qkv_stride < Q_dim + 2 * K_dim) return -1;

    const int total = S * (qkv_stride >> 3);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    qkv_split_rope_kvcache_fp16_vec_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(qkv),
        reinterpret_cast<const int4*>(rope),
        reinterpret_cast<int4*>(Q),
        reinterpret_cast<int4*>(Kc),
        reinterpret_cast<int4*>(Vc),
        S, Q_dim >> 3, K_dim >> 3, HD >> 3, qkv_stride >> 3,
        kc_offset >> 3, kc_stride >> 3);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}
