// LingBot-VLA model-specific kernel helpers (additive, Thor sm_110a / SM100-class).
// Shared by lingbot_norm_fp8.cu / lingbot_rope_qkv.cu / lingbot_silu_mul_fp8.cu.
// These kernels are compiled INTO flash_rt_kernels (gated by ENABLE_LINGBOT); the
// pybind entries live in csrc/bindings.cpp under #ifdef ENABLE_LINGBOT, all
// lingbot_-prefixed. No separate .so (mirrors the qwen36 model kernels in this dir).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

static inline cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}
