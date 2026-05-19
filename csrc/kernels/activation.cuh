// ================================================================
// FlashRT — Activation kernel declarations
// GeLU, SiLU, Gate*Act*Mul (BF16/FP16 and fused FP8 variants)
// Supports: __half (FP16), __nv_bfloat16 (BF16)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signatures, backward compatible) ──

void gate_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, int n, cudaStream_t stream = 0);

void gelu_inplace(__nv_bfloat16* x, int n, cudaStream_t stream = 0);

// Fused bias-add + GELU on the SigLIP FFN-up output. Two flavours,
// differing only in how the bias-add result is handed to GELU:
//
// * `bias_gelu_bf16` (fp32 intermediate): more numerically accurate
//   than the original kernel pair, but downstream INT8 calibration is
//   fitted against the original bf16-rounded intermediate. The drift
//   accumulates over 27 SigLIP layers to ~0.94-0.99 action cosine.
//   Microbench 3.19× over the pair; pipeline ~0.7 ms saving. NOT
//   enabled by default.
//
// * `bias_gelu_bf16_strict` (bf16 round-trip in middle): bit-identical
//   to add_bias_bf16 + gelu_inplace pair. Microbench 2.85× over the
//   pair; pipeline ~1.4 ms saving (6/6 frames bit-equal vs baseline
//   action). **Use this for lossless pipelines.**
void bias_gelu_bf16(__nv_bfloat16* x, const __nv_bfloat16* bias,
                    int seq_len, int dim, cudaStream_t stream = 0);
void bias_gelu_fp16(__half* x, const __half* bias,
                    int seq_len, int dim, cudaStream_t stream = 0);

void bias_gelu_bf16_strict(__nv_bfloat16* x, const __nv_bfloat16* bias,
                            int seq_len, int dim, cudaStream_t stream = 0);
void bias_gelu_fp16_strict(__half* x, const __half* bias,
                            int seq_len, int dim, cudaStream_t stream = 0);

void gate_silu_mul_merged(const __nv_bfloat16* merged, __nv_bfloat16* out,
                           int seq, int half_dim, cudaStream_t stream = 0);

void gate_silu_mul_merged_fp8(const __nv_bfloat16* merged, __nv_fp8_e4m3* out,
                               int seq, int half_dim,
                               const float* d_scale, cudaStream_t stream = 0);

// ── FP16 variants ──

void gate_silu_mul_fp16(const __half* gate, const __half* up,
                        __half* out, int n, cudaStream_t stream = 0);

void gelu_inplace_fp16(__half* x, int n, cudaStream_t stream = 0);

void gate_silu_mul_merged_fp16(const __half* merged, __half* out,
                                int seq, int half_dim, cudaStream_t stream = 0);

void gate_silu_mul_merged_fp8_fp16(const __half* merged, __nv_fp8_e4m3* out,
                                    int seq, int half_dim,
                                    const float* d_scale, cudaStream_t stream = 0);

// Split SiLU: separate gate and up buffers → FP8 output
// Matches pi05 silu_mul_split_fp8_k (split gate+up GEMMs for L2 optimization)
void silu_mul_split_fp8_fp16(const __half* gate, const __half* up,
                              __nv_fp8_e4m3* out, int n,
                              const float* d_scale, cudaStream_t stream = 0);
