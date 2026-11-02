// SPDX-License-Identifier: Apache-2.0
//
// Non-causal BF16 attention for the MolmoAct2 vision tower on SM120a.
// head_dim must be 72 and num_heads 16; q/k/v/out are contiguous
// (batch, seqlen, 16, 72) bf16. Returns non-zero for an unsupported shape.
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace attention {

// qkv_row_stride is the element stride between consecutive sequence rows
// for q, k and v (they share it; e.g. 3*head_dim*heads for a packed qkv
// buffer). out_row_stride is the output row stride.
int molmo_vit_attn_bf16(const void* q, const void* k, const void* v, void* out,
                        int batch, int seqlen, int num_heads, int head_dim,
                        int qkv_row_stride, int out_row_stride,
                        float softmax_scale, cudaStream_t stream);

// Same attention, emitting only a static-scale FP8 E4M3 output
// (out = e4m3(clamp(bf16(o) / *out_scale))), byte-identical to
// quantize_fp8_static applied to the bf16 output.
int molmo_vit_attn_bf16_fp8out(const void* q, const void* k, const void* v,
                               void* out_fp8, const float* out_scale,
                               int batch, int seqlen, int num_heads,
                               int head_dim, int qkv_row_stride,
                               int out_row_stride, float softmax_scale,
                               cudaStream_t stream);

// Same attention with the per-KV-tile P*V product in fp16 inputs and an
// fp16 accumulator (twice the f32-accumulate MMA rate on SM120 GeForce),
// folded into the fp32 running output. Not bit-identical to molmo_vit_attn_bf16.
int molmo_vit_attn_f16pv(const void* q, const void* k, const void* v, void* out,
                         int batch, int seqlen, int num_heads, int head_dim,
                         int qkv_row_stride, int out_row_stride,
                         float softmax_scale, cudaStream_t stream);

// Persistent-CTA schedule: `blocks` CTAs walk all (batch, head, q-tile)
// work items in launch order. blocks <= 0 selects the per-item grid.
int molmo_vit_attn_bf16_persist(const void* q, const void* k, const void* v,
                                void* out, int batch, int seqlen,
                                int num_heads, int head_dim,
                                int qkv_row_stride, int out_row_stride,
                                float softmax_scale, int blocks,
                                cudaStream_t stream);

}  // namespace attention
}  // namespace flash_rt
