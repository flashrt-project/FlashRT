// Vectorized QKV split + RoPE + KV-cache append (FP16). Bit-exact with
// qkv_split_rope_kvcache_fp16 on 16-byte-aligned shapes; returns nonzero
// (without launching) when the shape does not qualify so the caller can
// fall back to the scalar kernel.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

int qkv_split_rope_kvcache_fp16_vec(
    const __half* qkv, const __half* rope,
    __half* Q, __half* Kc, __half* Vc,
    int S, int Q_dim, int K_dim, int HD, int qkv_stride,
    long kc_offset, int kc_stride,
    cudaStream_t stream);
