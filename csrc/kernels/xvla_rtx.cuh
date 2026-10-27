// FlashRT — X-VLA (soft-prompted flow-matching VLA) RTX helpers.
//
// X-VLA's flow-matching loop needs a handful of bf16 elementwise steps that no
// existing entry provides. They exist so the steady-state path carries no
// torch/ATen op: the constant positional table is folded into the sequence
// with a plain add, the Euler interpolant lands directly in the action-token
// buffer alongside its timestep and proprio columns, and the emitted chunk is
// sigmoid-finalised and widened in one launch.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace flash_rt {
namespace xvla {

// out[i] = a[i] + b[i]
void add2_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
               __nv_bfloat16* out, int n, cudaStream_t stream);

// Per-denoising-step action-token assembly. Writes, for every row r < rows:
//   act_tok[r,     d]        = x1[r,d] * t + prev[r,d] * (1 - t),  d < dim
//   act_tok[r,     g0/g1]    = 0                       (gripper columns)
//   act_tok[r, toff + k]     = time[k]                 k < time_dim
// `act_tok` is row-strided by `stride` (dim + proprio + time columns).
void flow_step_bf16(const __nv_bfloat16* x1, const __nv_bfloat16* prev,
                    __nv_bfloat16* act_tok, const __nv_bfloat16* time, float t,
                    int rows, int dim, int stride, int g0, int g1, int toff,
                    int time_dim, cudaStream_t stream);

// Broadcasts the raw fp32 proprio reading into the action-token buffer:
//   act_tok[r, poff + d] = (d < n_state && d not a gripper column) ? state[d] : 0
void proprio_bf16(const float* state, __nv_bfloat16* act_tok, int rows,
                  int stride, int poff, int dim_proprio, int n_state, int g0,
                  int g1, cudaStream_t stream);

// out[r, d] = sigmoid(in[r, d]) for the two gripper columns, else fp32(in).
void finalize_bf16(const __nv_bfloat16* in, float* out, int rows, int dim,
                   int stride, int g0, int g1, cudaStream_t stream);

// LayerNorm(x) -> NVFP4 packed + cutlass-swizzled UE4M3 block scales, in one
// launch and one read of x. `x` is left untouched.
// LayerNorm emitting the normalised bf16 row only. Same arithmetic and same
// reduction order as the fused entry; used where a bare bf16 row is required.
void ln_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
             const __nv_bfloat16* w, const __nv_bfloat16* b, int rows, int cols,
             float eps, cudaStream_t stream);

void ln_quant_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                   const __nv_bfloat16* b, uint8_t* packed, uint8_t* sf_swz,
                   int rows, int cols, float eps, cudaStream_t stream);

// Same, with `residual += h` folded into the staging pass; the running sum is
// written back to `residual` (bf16, rounded exactly as `residual_add` rounds)
// and normalized from the rounded value, matching the unfused pair.
void add_ln_quant_bf16(__nv_bfloat16* residual, const __nv_bfloat16* h,
                       const __nv_bfloat16* w, const __nv_bfloat16* b,
                       uint8_t* packed, uint8_t* sf_swz, int rows, int cols,
                       float eps, cudaStream_t stream);

// Fold + LayerNorm + NVFP4 quantize that also writes the normalised bf16 row
// (the BART encoder is post-norm, so its stream carries the normalised value).
// Fold-free LayerNorm -> NVFP4 at 256 threads (layer_norm_kernel's order).
void ln_quant_bf16_nt256(const __nv_bfloat16* x, const __nv_bfloat16* w,
                         const __nv_bfloat16* b, uint8_t* packed,
                         uint8_t* sf_swz, int rows, int cols, float eps,
                         cudaStream_t stream);

void add_ln_quant_norm_bf16(__nv_bfloat16* residual, const __nv_bfloat16* h,
                            __nv_bfloat16* norm_out, const __nv_bfloat16* w,
                            const __nv_bfloat16* b, uint8_t* packed,
                            uint8_t* sf_swz, int rows, int cols, float eps,
                            cudaStream_t stream);

// out[i] = bf16(in[i]), fp16 input — the DaViT/encoder dtype boundary.
void cast_f16_to_bf16(const __half* in, __nv_bfloat16* out, int n,
                      cudaStream_t stream);

// out[i] = in[i] (fp16)
void copy_f16(const __half* in, __half* out, int n, cudaStream_t stream);

// out[i] = in[i] (bf16)
void copy_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int n,
               cudaStream_t stream);

// Action decoder: one warp per (row, output); w is (N,K), bias (N), x (M,K).
void action_decoder_gemv_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                              const __nv_bfloat16* bias, __nv_bfloat16* out,
                              int M, int N, int K, cudaStream_t stream);

}  // namespace xvla
}  // namespace flash_rt
