// ================================================================
// FlashRT — Cross-layer fusion kernel declarations
// Fused gate*residual + AdaRMSNorm -> FP8
// Supports: __half (FP16), __nv_bfloat16 (BF16)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signature, backward compatible) ──

void gate_residual_ada_norm_fp8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                                 const __nv_bfloat16* style,
                                 __nv_fp8_e4m3* out, __nv_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 const float* d_scale, cudaStream_t stream = 0);

// ── FP16 variant ──

void gate_residual_ada_norm_fp8_fp16(__half* residual, const __half* x,
                                      const __half* gate, const __half* weight,
                                      const __half* style,
                                      __nv_fp8_e4m3* out, __half* gate_out,
                                      int seq_len, int dim, float eps,
                                      const float* d_scale, cudaStream_t stream = 0);

void gate_residual_ada_norm_fp16(__half* residual, const __half* x,
                                  const __half* gate, const __half* weight,
                                  const __half* style,
                                  __half* out, __half* gate_out,
                                  int seq_len, int dim, float eps,
                                  cudaStream_t stream = 0);

void gate_residual_ada_norm_int8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                                 const __nv_bfloat16* style,
                                 int8_t* out, __nv_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 float* d_scales, cudaStream_t stream = 0);
