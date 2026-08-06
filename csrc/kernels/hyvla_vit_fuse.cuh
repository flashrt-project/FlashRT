// FlashRT — Hy-VLA Orin ViT fusion kernel declarations.
#pragma once
#include <cuda_runtime.h>

extern "C" void hyvla_vit_add_layer_norm_bf16(
    void* residual, const void* x_add,
    const void* ln_weight, const void* ln_bias,
    void* out, int rows, int dim, float eps, cudaStream_t stream);
