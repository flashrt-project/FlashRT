// FlashRT — Hy-VLA denoise Euler update (fp32 x += dt * bf16 v).
#pragma once
#include <cuda_runtime.h>

extern "C" void hyvla_euler_update_bf16_fp32(
    void* x, const void* v, void* x_bf16, float dt, long n, cudaStream_t stream);
