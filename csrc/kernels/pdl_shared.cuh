// FlashRT — opt-in PDL-aware shared elementwise/norm entries.
//
// These are additive, distinctly named operations for the X-VLA RTX pipeline.
// The existing add_bias_bf16/residual_add/layer_norm entries in the shared
// library are unchanged and remain the default for every other caller.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

void add_bias_bf16_pdl(__nv_bfloat16* x, const __nv_bfloat16* b,
                       int S, int D, cudaStream_t stream);
void residual_add_pdl(__nv_bfloat16* residual, const __nv_bfloat16* x,
                      int n, cudaStream_t stream);
void layer_norm_pdl(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    const __nv_bfloat16* bias, __nv_bfloat16* out,
                    int seq_len, int dim, float eps, cudaStream_t stream);
