// ================================================================
// FlashRT — seqused attention with the mask folded into softmax.
// Drop-in for attention_qkv_fp16_seqused with one fewer kernel;
// requires S_kv_max <= 1024 (softmax per-warp register budget).
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

void attention_qkv_fp16_seqused_v2(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int S, int S_kv_max, int NH, int HD,
    const int* seqused_k,
    float attn_scale,
    cudaStream_t stream);
