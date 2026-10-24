// ================================================================
// FlashRT — BF16 RMSNorm / q-k RMSNorm + RoPE / SiLU-gate kernels that
// reproduce the HuggingFace (PyTorch eager, CUDA) result bit for bit.
//
// The HF formulation rounds to bf16 at fixed points and reduces the fp32
// sum of squares in PyTorch's reduction order; both are mirrored here, so
// the output is not merely close to the reference but equal to it. The
// launch configuration of torch's reduction depends on how many rows are
// reduced, and the kernels follow it (see rms_norm_hf_bf16.cu).
//
// `torch_reduce` selects PyTorch's reduction tree for the mean of squares:
// 0 for PyTorch <= 2.9, 1 for PyTorch >= 2.10 (the release that reversed the
// intra-warp shuffle order and vectorized 128-wide rows).
//
// Every entry takes `pdl`: when true the kernel is launched with
// programmatic stream serialization and waits on griddepcontrol before it
// reads any input. Requires SM90+; pass false on SM8x.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt::kernels {

// out[r] = w * bf16(x[r] * rsqrt(mean(x[r]^2) + eps)), HF Qwen3RMSNorm.
// dim must be a multiple of 4 and larger than 128 (torch's vectorized
// reduction path); rows of x / out are contiguous.
void rms_norm_hf_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                      __nv_bfloat16* out, int rows, int dim, float eps,
                      int torch_reduce, bool pdl, cudaStream_t stream);

// residual = bf16(residual + delta) in place, then out = RMSNorm(residual)
// with the rounding of rms_norm_hf_bf16 -- HF's decoder-layer residual
// followed by the next norm.
void residual_add_rms_norm_hf_bf16(__nv_bfloat16* residual,
                                   const __nv_bfloat16* delta,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* out, int rows, int dim,
                                   float eps, int torch_reduce, bool pdl,
                                   cudaStream_t stream);

// Per-head RMSNorm over q and k followed by rotate-half RoPE, in place,
// head_dim 128. Rows of q / k start every *_row_stride elements and hold
// *_heads heads. cos_half / sin_half are [seq, 64] bf16: the first half of
// HF's [seq, 128] tables, whose two halves are equal.
void qk_rms_norm_rope_hf_bf16(__nv_bfloat16* q, long long q_row_stride,
                              int q_heads,
                              __nv_bfloat16* k, long long k_row_stride,
                              int k_heads,
                              const __nv_bfloat16* q_weight,
                              const __nv_bfloat16* k_weight,
                              const __nv_bfloat16* cos_half,
                              const __nv_bfloat16* sin_half,
                              int seq, int head_dim, float eps,
                              int torch_reduce, bool pdl, cudaStream_t stream);

// out = bf16(bf16(silu(gate)) * up), HF's act_fn(gate_proj(x)) * up_proj(x).
// n must be a multiple of 8; pointers 16-byte aligned. out may alias gate.
void silu_mul_hf_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                      __nv_bfloat16* out, long long n, bool pdl,
                      cudaStream_t stream);

}  // namespace flash_rt::kernels
