// SPDX-License-Identifier: Apache-2.0
#pragma once
// Non-causal prefill attention that reads Q / K / V straight from the rows of
// a projection GEMM output (fused [rows, ld] QKV, or three [rows, ld]
// buffers), so no head-split / bias / RoPE pass runs before it:
//
//   for batch b, head h, row s of the batch (rows b*S .. b*S + S - 1):
//     x = src[b*S + s, h*hd .. h*hd + hd) + bias[h*hd ..]        (bias nullable)
//     Q, K: rotate-half RoPE with [S, hd] tables (cos / sin nullable; the first
//           half of each row is used: out_i = x_i c_i - x_{i+hd/2} s_i,
//           out_{i+hd/2} = x_{i+hd/2} c_i + x_i s_i)
//   out[b*S + s, h*hd + d] = softmax(Q K^T * scale) V          ([rows, out_ld])
//
// FP16 or BF16 operands (mma.sync m16n8k16, FP32 softmax / accumulation),
// head_dim 64 or 72 (padded to a multiple of 16 inside the kernel only).
#include <cuda_runtime.h>

namespace flash_rt {
namespace prefill_attention {

// dtype: 0 = FP16, 1 = BF16. All pointers / strides in elements of dtype.
int fused_qkv_attention(int dtype, const void* q, const void* k, const void* v, int ld, const void* q_bias,
                        const void* k_bias, const void* v_bias, const void* cos, const void* sin, void* out,
                        int out_ld, int batches, int S, int heads, int hd, float scale, cudaStream_t stream);

}  // namespace prefill_attention
}  // namespace flash_rt
