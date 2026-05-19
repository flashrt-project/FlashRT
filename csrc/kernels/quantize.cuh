// ================================================================
// FlashRT — Quantization kernel declarations
// FP8 dynamic/static quantize, NVFP4 block-scaled (SM120+)
// FP8 functions support: __half (FP16), __nv_bfloat16 (BF16) input
// ================================================================
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signatures, backward compatible) ──

// FP8 quantize with host sync (NOT CUDA Graph compatible)
float quantize_fp8(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                   float* d_scale, int n, cudaStream_t stream = 0);

// FP8 quantize with pre-computed static scale (CUDA Graph compatible)
void quantize_fp8_static(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                         const float* d_scale, int n, cudaStream_t stream = 0);

// FP8 quantize device-only: scale computed on device (CUDA Graph compatible)
void quantize_fp8_device(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                         float* d_scale, int n, cudaStream_t stream = 0);

// ── FP16 variants ──

float quantize_fp8_fp16(const __half* input, __nv_fp8_e4m3* output,
                        float* d_scale, int n, cudaStream_t stream = 0);

void quantize_fp8_static_fp16(const __half* input, __nv_fp8_e4m3* output,
                               const float* d_scale, int n, cudaStream_t stream = 0);

void quantize_fp8_device_fp16(const __half* input, __nv_fp8_e4m3* output,
                               float* d_scale, int n, cudaStream_t stream = 0);

// ── INT8 (BF16 activations, device-only dynamic scale) ──

void quantize_int8_device(const __nv_bfloat16* input, int8_t* output,
                          float* d_scale, int n, cudaStream_t stream = 0);

// Static INT8: uses pre-calibrated d_scale, no amax reduction (1 kernel vs 3).
void quantize_int8_static(const __nv_bfloat16* input, int8_t* output,
                           const float* d_scale, int n, cudaStream_t stream = 0);

void quantize_int8_rowwise(const __nv_bfloat16* input, int8_t* output,
                           float* d_scales, int rows, int cols,
                           cudaStream_t stream = 0);

// Static per-row INT8 quantize: uses pre-calibrated per-row scales,
// skips the per-row amax reduction → single-pass over data.
void quantize_int8_rowwise_static(const __nv_bfloat16* input, int8_t* output,
                                   const float* d_scales, int rows, int cols,
                                   cudaStream_t stream = 0);

void dequant_int32_to_bf16(const int32_t* input, __nv_bfloat16* output,
                           const float* d_act_scale, const float* d_weight_scale,
                           int n, cudaStream_t stream = 0);

// ── L2 weight prefetch ──

void prefetch_l2(const void* data, size_t num_bytes, cudaStream_t stream = 0);

// ── NVFP4 (BF16-only, SM120+) ──

#ifdef ENABLE_NVFP4
void quantize_bf16_to_nvfp4(const __nv_bfloat16* input, uint8_t* fp4_data,
                              uint8_t* scale_factors, int rows, int cols,
                              cudaStream_t stream = 0);

void quantize_bf16_to_nvfp4_swizzled(const __nv_bfloat16* input, uint8_t* fp4_data,
                                       uint8_t* scale_factors, int rows, int cols,
                                       cudaStream_t stream = 0);

// Fused: rms_norm(x, weight) -> nvfp4 packed + swizzled SF (Qwen3.5 (1+w)
// convention; weight is the precomputed (1+w) tensor).
void rms_norm_to_nvfp4_swizzled_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* rms_weight,
    uint8_t* packed, uint8_t* sf_swz,
    int rows, int cols, float eps,
    cudaStream_t stream = 0);

// Fused: h_post = h_in + attn_proj; rms_norm(h_post, weight) -> nvfp4
// packed + swizzled SF. The h_post bf16 buffer is also written to
// global memory because the post-MLP residual addition needs it.
//
// Replaces the (torch.add + rms_norm + quantize_bf16_to_nvfp4_swizzled)
// 3-launch sequence at every per-layer post-attn / post-MLP transition.
// Bit-equivalent to the unfused sequence under the same bf16 rounding
// model (residual sum -> bf16 round -> ssq + amax over bf16 values).
void residual_add_rms_norm_to_nvfp4_swizzled_bf16(
    const __nv_bfloat16* h_in,
    const __nv_bfloat16* attn_proj,
    __nv_bfloat16* h_post,
    const __nv_bfloat16* rms_weight,
    uint8_t* packed, uint8_t* sf_swz,
    int rows, int cols, float eps,
    cudaStream_t stream = 0);

void quantize_bf16_to_mxfp8(const __nv_bfloat16* input, __nv_fp8_e4m3* fp8_data,
                              uint8_t* scale_factors, int rows, int cols,
                              cudaStream_t stream = 0);

int get_mxfp8_sf_size(int rows, int cols);

void quantize_bf16_to_mxfp4_cutlass(const __nv_bfloat16* input, uint8_t* fp4_data,
                                      uint8_t* scale_factors, int N, int K,
                                      cudaStream_t stream = 0);

int get_mxfp4_sf_size(int N, int K);
#endif
