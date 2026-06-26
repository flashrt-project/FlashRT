// ================================================================
// FlashRT — MelBandRoformer custom fused kernels
//
// Fused kernels for MelBandRoformer inference acceleration:
//   1. qkv_split_rope        — QKV split + interleaved RoPE
//   2. gated_attn_quant      — sigmoid gate * attn + reshape + FP8 quant
//   3. fp8_dequant           — FP8 → BF16 dequantize
//   4. resadd_rmsnorm_fp8    — residual add + RMSNorm → FP8 (keeps residual)
//   5. fused_add_rmsnorm_bf16 — residual add + RMSNorm (BF16 in/out)
//
// Kernels are templated on input dtype T (__nv_bfloat16 / __half).
// Currently instantiated for __nv_bfloat16 (BF16).
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_rt { namespace mbr {

void qkv_split_rope(const __nv_bfloat16* qkv, const float* cosT, const float* sinT,
                    __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
                    int B, int S, int H, int D, cudaStream_t st);

void gated_attn_quant(const __nv_bfloat16* o, const __nv_bfloat16* gates,
                      __nv_fp8_e4m3* out_fp8,
                      int B, int H, int S, int D, float scale, cudaStream_t st);

void fp8_dequant_bf16(const __nv_fp8_e4m3* inp, float scale,
                      __nv_bfloat16* out, int n, cudaStream_t st);

void resadd_rmsnorm_fp8_keepres(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                const __nv_bfloat16* gamma,
                                __nv_bfloat16* sum_out, __nv_fp8_e4m3* norm_fp8,
                                int M, int dim, float scale, cudaStream_t st);

void fused_add_rmsnorm_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                            const __nv_bfloat16* gamma, __nv_bfloat16* out,
                            int M, int dim, cudaStream_t st);

}}  // namespace flash_rt::mbr
