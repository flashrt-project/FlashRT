// FlashRT — Hy-VLA prefill segment-mask attention declaration.
#pragma once
#include <cuda_runtime.h>

extern "C" void hyvla_prefill_attn_bf16(
    const void* q, const void* k, const void* v, void* o,
    const void* mask, const void* active,
    int S, int H, float scale,
    int q_stride_h, int k_stride_h, int v_stride_h, cudaStream_t stream);
