// FlashRT — Hy-VLA single-CTA dynamic per-tensor FP8 quant declaration.
#pragma once
#include <cuda_runtime.h>

// Dynamic per-tensor FP8 (e4m3) quant of a small bf16 tensor in ONE launch.
// Matches quantize_fp8_device numerics exactly but collapses its
// memset+absmax+compute_scale+quantize (4 nodes) into a single deterministic
// single-block kernel — for the denoise expert tower where M<=64 keeps the
// whole tensor comfortably inside one CTA. Graph-safe (no atomics, 1 block).
extern "C" void hyvla_quant_fp8_dyn_bf16(
    const void* x, void* out, float* scale, int n, cudaStream_t stream);
