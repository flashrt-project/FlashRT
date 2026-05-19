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

// FP8 dequantize with pre-computed static scale (CUDA Graph compatible)
void dequantize_fp8_static_bf16(const __nv_fp8_e4m3* input, __nv_bfloat16* output,
                                const float* d_scale, int n, cudaStream_t stream = 0);

void dequantize_fp8_static_bf16_6(
    const __nv_fp8_e4m3* in0, const __nv_fp8_e4m3* in1,
    const __nv_fp8_e4m3* in2, const __nv_fp8_e4m3* in3,
    const __nv_fp8_e4m3* in4, const __nv_fp8_e4m3* in5,
    __nv_bfloat16* out0, __nv_bfloat16* out1,
    __nv_bfloat16* out2, __nv_bfloat16* out3,
    __nv_bfloat16* out4, __nv_bfloat16* out5,
    const float* s0, const float* s1, const float* s2,
    const float* s3, const float* s4, const float* s5,
    int n, cudaStream_t stream = 0);

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

// Specialized swizzled NVFP4 quantizer for Motus Wan FFN intermediate
// activations (cols=14336). Byte-equivalent to the generic swizzled path.
int quantize_bf16_to_nvfp4_swizzled_k14336(
    const __nv_bfloat16* input,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows,
    int cols,
    cudaStream_t stream = 0);

// Same layout as quantize_bf16_to_nvfp4_swizzled, but caps each 16-channel
// group's virtual amax by clip_amax[group]. Values above the cap saturate
// naturally in FP4 while the rest of the group keeps a finer scale.
void quantize_bf16_to_nvfp4_swizzled_clipped(
    const __nv_bfloat16* input,
    const float* clip_amax,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

// Uses clip_amax[group] as a fixed per-16-channel group amax for every row.
// This is an experimental static activation-scale path for W4A4 calibration.
void quantize_bf16_to_nvfp4_swizzled_static_groups(
    const __nv_bfloat16* input,
    const float* group_amax,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

// Experimental diagnostic: choose each 16-channel group's scale from the
// second-largest absolute value instead of the largest, so a single outlier
// saturates instead of destroying the remaining 15 values' resolution.
void quantize_bf16_to_nvfp4_swizzled_secondmax(
    const __nv_bfloat16* input,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    float scale_mult,
    cudaStream_t stream = 0);

// Experimental diagnostic: choose each 16-channel group's UE4M3 scale by
// minimizing reconstruction error over a small fixed set of candidate scales.
void quantize_bf16_to_nvfp4_swizzled_mse(
    const __nv_bfloat16* input,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

// Fused: input[:, k] * inv_s[k] -> nvfp4 packed + swizzled SF.
// Used by SmoothQuant/AWQ NVFP4 paths so replay does not need a separate
// broadcast-multiply kernel before activation quantization.
void awq_quant_bf16_to_nvfp4_swizzled(
    const __nv_bfloat16* input,
    const __nv_bfloat16* inv_s,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

// Fused: gelu(input + bias) -> nvfp4 packed + swizzled SF.
// Optional AWQ version additionally multiplies by inv_s[col] before
// choosing block scales and packing. Used by Motus full FFN FP4 path.
void bias_gelu_quant_bf16_to_nvfp4_swizzled(
    const __nv_bfloat16* input,
    const __nv_bfloat16* bias,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

// Diagnostic Motus FP4 rescue helpers:
//   gather_bf16_cols: out[row, j] = input[row, indices[j]]
//   add_side_bias_gelu_gather_zero_quant: gelu(main + side + bias), gather
//   selected GELU values to side_out, zero those selected columns in the NVFP4
//   quantized residual, and emit swizzled scale factors.
void gather_bf16_cols(
    const __nv_bfloat16* input,
    const int* indices,
    __nv_bfloat16* output,
    int rows, int cols, int n_idx,
    cudaStream_t stream = 0);

void add_side_bias_gelu_gather_zero_quant_bf16_to_nvfp4_swizzled(
    const __nv_bfloat16* main,
    const __nv_bfloat16* side,
    const __nv_bfloat16* bias,
    const int* zero_gather_indices,
    __nv_bfloat16* side_out,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols, int n_idx,
    cudaStream_t stream = 0);

void awq_bias_gelu_quant_bf16_to_nvfp4_swizzled(
    const __nv_bfloat16* input,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* inv_s,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

// Fused: gelu(input + bias) -> BF16 in shared memory -> NVFP4 packed +
// swizzled SF. Unlike bias_gelu_quant_bf16_to_nvfp4_swizzled, GELU is
// evaluated once and rounded through BF16 to match the old
// bias_gelu_inplace_bf16 + quantize path more closely.
void bias_gelu_quant_cached_bf16_to_nvfp4_swizzled(
    const __nv_bfloat16* input,
    const __nv_bfloat16* bias,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

void awq_bias_gelu_quant_cached_bf16_to_nvfp4_swizzled(
    const __nv_bfloat16* input,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* inv_s,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

// Fused: statically-scaled FP8 activation -> NVFP4 packed + swizzled SF.
// Used by Motus FFN down-only experiments where cuBLASLt produces
// GELU+bias output directly in FP8 and the NVFP4 down GEMM consumes it.
void fp8_static_to_nvfp4_swizzled(
    const __nv_fp8_e4m3* input,
    const float* scale,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

void awq_fp8_static_to_nvfp4_swizzled(
    const __nv_fp8_e4m3* input,
    const float* scale,
    const __nv_bfloat16* inv_s,
    uint8_t* fp4_data,
    uint8_t* scale_factors,
    int rows, int cols,
    cudaStream_t stream = 0);

// Fused: rms_norm(x, weight) -> nvfp4 packed + swizzled SF (Qwen3.5 (1+w)
// convention; weight is the precomputed (1+w) tensor).
void rms_norm_to_nvfp4_swizzled_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* rms_weight,
    uint8_t* packed, uint8_t* sf_swz,
    int rows, int cols, float eps,
    cudaStream_t stream = 0);

// Fused: affine LayerNorm(x, weight, bias) -> nvfp4 packed + swizzled SF.
// Used by Motus cross-attn norm3 -> Q NVFP4 projection so the BF16
// normalized activation does not round-trip through global memory.
void layer_norm_to_nvfp4_swizzled_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    uint8_t* packed,
    uint8_t* sf_swz,
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

// ---- Public INT8 quantization helper declarations ----
void quantize_int8_device(const __nv_bfloat16* input, int8_t* output,
                          float* d_scale, int n, cudaStream_t stream = 0);
void quantize_int8_static(const __nv_bfloat16* input, int8_t* output,
                           const float* d_scale, int n, cudaStream_t stream = 0);
void quantize_int8_rowwise(const __nv_bfloat16* input, int8_t* output,
                           float* d_scales, int rows, int cols,
                           cudaStream_t stream = 0);
void quantize_int8_rowwise_static(const __nv_bfloat16* input, int8_t* output,
                                   const float* d_scales, int rows, int cols,
                                   cudaStream_t stream = 0);
void dequant_int32_to_bf16(const int32_t* input, __nv_bfloat16* output,
                           const float* d_act_scale, const float* d_weight_scale,
                           int n, cudaStream_t stream = 0);
