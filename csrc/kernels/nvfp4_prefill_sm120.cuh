// SPDX-License-Identifier: Apache-2.0
#pragma once
// Row kernels that produce and route the NVFP4 operands of prefill-shaped
// transformer layers (vision towers, VLM prefixes) on sm_120a, for the
// CUTLASS block-scaled GEMMs: activations e2m1 [M, K/2] row-major with UE4M3
// per-16 scales in the CUTLASS Sm1xx swizzled layout (rows padded to 128;
// padding rows must stay zero, so allocate the scale buffer zeroed).
//
// Block scales round to nearest (sf = ue4m3_rn(amax16(x / gs) / 6)), the
// same convention as the skinny NVFP4 family; gs is a device scalar.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace flash_rt {
namespace nvfp4_prefill {

// Bytes of a swizzled scale buffer for rows x cols (cols % 16 == 0).
size_t sf_bytes(int rows, int cols);

// Row normalization straight to NVFP4: mode 0 = LayerNorm (weight / bias
// optional), mode 1 = RMSNorm (weight optional, y = x * rsqrt(mean x^2 + eps) * w).
// With residual (nullable) the row is first updated in place, x += residual
// (BF16), and the norm reads the updated row. With residual_bias (nullable,
// BF16 [dim]) the residual term is bf16(residual + residual_bias) — the
// projection bias of the producing GEMM folded into the update. dim % 128 == 0,
// dim <= 4096; one CTA per row.
int norm_quant(__nv_bfloat16* x, const __nv_bfloat16* residual, const __nv_bfloat16* weight,
               const __nv_bfloat16* bias, int mode, float eps, uint8_t* out, uint8_t* out_sf, const float* gs,
               int rows, int dim, cudaStream_t stream, const __nv_bfloat16* residual_bias = nullptr);

// Same contract and output bytes as norm_quant, one warp per row (no CTA
// barriers) for dim in {1024, 1152, 2048, 2560, 4096}; other dims fall back.
int norm_quant_warp(__nv_bfloat16* x, const __nv_bfloat16* residual, const __nv_bfloat16* weight,
                    const __nv_bfloat16* bias, int mode, float eps, uint8_t* out, uint8_t* out_sf,
                    const float* gs, int rows, int dim, cudaStream_t stream,
                    const __nv_bfloat16* residual_bias = nullptr,
                    int mw_wpr = 0);   // 3 (dim 1152) / 4 (dim 2560): warps per row

// NVFP4(bf16(bf16(gelu_tanh(a)) * b)) for BF16 [rows, cols] a, b: the
// gelu_inplace + mul + quant sequence in one pass. cols % 128 == 0.
int gelu_mul_quant(const __nv_bfloat16* a, const __nv_bfloat16* b, uint8_t* out, uint8_t* out_sf,
                   const float* gs, int rows, int cols, cudaStream_t stream);

// BF16 [rows, cols] (row stride ld elements) -> NVFP4. cols % 128 == 0.
int quant(const __nv_bfloat16* x, int ld, uint8_t* out, uint8_t* out_sf, const float* gs, int rows, int cols,
          cudaStream_t stream);

// In-place rotate-half RoPE on the Q and K heads of a fused [S, ld] BF16
// projection (Q heads at column q_off, K heads at k_off, head_dim hd), with an
// optional per-head RMSNorm before the rotation (q_norm / k_norm weights,
// nullable). cos / sin: fp32 [S, hd] rotate-half tables (first half used):
// out_i = x_i cos_i - x_{i+h} sin_i, out_{i+h} = x_{i+h} cos_i + x_i sin_i,
// h = hd / 2, i < h.
int qk_norm_rope(__nv_bfloat16* qkv, int ld, int q_off, int k_off, int n_q, int n_k, int hd,
                 const float* cos, const float* sin, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                 float eps, int S, cudaStream_t stream);

// h[idx[i], :] += src[i, :] for i < n (BF16, dim % 8 == 0).
int scatter_add_rows(__nv_bfloat16* h, const __nv_bfloat16* src, const int64_t* idx, int n, int dim,
                     cudaStream_t stream);

}  // namespace nvfp4_prefill
}  // namespace flash_rt
