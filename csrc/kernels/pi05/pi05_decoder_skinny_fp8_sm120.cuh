// SPDX-License-Identifier: Apache-2.0
#pragma once

// Pi0.5 skinny FP8 GEMM family for its small-row action decoder on sm_120a.
//
// The action expert of a flow-matching VLA runs its denoising steps over a
// handful of rows (the action chunk) while every step streams the full
// decoder weight set from HBM. cuBLASLt tiles that shape for reuse it does
// not have; these kernels stream each weight row once into mma fragments,
// split K across CTAs and hand the FP32 partial sums to a fused consumer
// (RoPE + cache write, gated residual + adaptive RMS norm + FP8 quantize,
// GeGLU + FP8 quantize). Every launch may use programmatic dependent launch
// so that a GEMM prefetches its weights while the previous kernel drains.
//
// Layouts: A e4m3 [M, K] row-major; W e4m3 [N, K] row-major (the "nk" FP8
// layout); partials fp32 [K / k_chunk][M][N]. Per-tensor scales are device
// pointers, applied by the consumers as alpha = (*a_scale) * (*w_scale).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace pi05_dec_skinny {

// Tile configurations (BN x KC x warps). Returns the number of configs.
int config_count();
// K chunk of a configuration (partials have K / k_chunk splits); 0 if invalid.
int config_k_chunk(int cfg);
// Whether (N, K) is divisible for the configuration.
bool config_supports(int cfg, int N, int K);

// partials[s] = A(M, Ks) @ W(N, Ks)^T. Rows beyond 16 are covered by a grid
// dimension (weights re-read through L2). Returns cudaError_t as int.
int gemm(const void* A, const void* W, float* partials, int M, int N, int K,
         int cfg, bool pdl, cudaStream_t stream);

// Same with a BF16 activation quantized on load: a = e4m3(clamp(x / *a_scale)).
int gemm_bf16_act(const __nv_bfloat16* A, const float* a_scale, const void* W,
                  float* partials, int M, int N, int K, int cfg, bool pdl,
                  cudaStream_t stream);

// QKV consumer: x = bf16(alpha * sum partials); Q gets RoPE, K gets RoPE and
// is written at cache row (pos + row), V is copied to (pos + row); pos comes
// from *devpos (may be null -> 0). Pair layout matches qkv_split_rope. Rows
// are grouped into samples of sample_rows (0 -> all rows one sample); sample
// b writes its cache rows at row b * kv_sample_stride + pos + r (row units).
int sum_rope(const float* partials, int splits, const float* a_scale,
             const float* w_scale, const __nv_bfloat16* rope,
             __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
             const int* devpos, int rows, int q_dim, int k_dim, int v_dim,
             int head_dim, int sample_rows, long long kv_sample_stride,
             bool pdl, cudaStream_t stream);

// Residual consumer: x = bf16(alpha * sum partials); residual += x * gate;
// then the adaptive RMS norm of the updated row with style (scale | shift |
// gate) and the norm weight; out is e4m3(clamp(normed / *out_scale)) when
// out_fp8 is given, else BF16 into out_bf16; gate_out receives the style gate.
int residual_ada_norm(const float* partials, int splits, const float* a_scale,
                      const float* w_scale, __nv_bfloat16* residual,
                      const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                      const __nv_bfloat16* style, __nv_fp8_e4m3* out_fp8,
                      __nv_bfloat16* out_bf16, const float* out_scale,
                      __nv_bfloat16* gate_out, int rows, int dim, float eps,
                      bool pdl, cudaStream_t stream);

// Residual-only consumer for the last layer: residual += bf16(alpha * sum) * gate.
int residual_gate_mul(const float* partials, int splits, const float* a_scale,
                      const float* w_scale, __nv_bfloat16* residual,
                      const __nv_bfloat16* gate, int rows, int dim, bool pdl,
                      cudaStream_t stream);

// GeGLU consumer over merged [gate | up] partials: out = e4m3(clamp(
// gelu_tanh(bf16(g)) * bf16(u) / *out_scale)).
int gate_gelu_fp8(const float* partials, int splits, const float* a_scale,
                  const float* w_scale, __nv_fp8_e4m3* out, int rows, int half,
                  const float* out_scale, bool pdl, cudaStream_t stream);

// Decoder cross-attention: rows_per_sample query rows (<= 16) times `heads`
// query heads over one shared K/V head of width 256, keys [0, valid) with
// valid = *seqused if given else kv_len. Two launches, both able to join the
// PDL chain: split-KV partials (one CTA per 64 keys, head, sample) and the
// combine (one CTA per row, head, sample). scratch holds
// splits * samples * heads * (32 + 16 * 256) floats; counters is unused.
// Q/O rows have q_row_stride elements (heads * 256 for the packed layout);
// sample b's K/V start kv_sample_stride rows after sample 0's.
int attn_splits(int kv_len);
size_t attn_scratch_floats(int splits, int samples, int heads);
int attn(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
         __nv_bfloat16* O, int rows_per_sample, int samples, int heads,
         int q_row_stride, int kv_len, const int* seqused,
         long long kv_sample_stride, float scale, float* scratch,
         int* counters, bool pdl, cudaStream_t stream);

// Action input projection (K = 32, N = 1024) with bias, written to the
// residual stream x, followed by the first layer's adaptive RMS norm to FP8
// (weight, style, out_scale as in residual_ada_norm). One CTA per row.
int action_in_norm(const __nv_bfloat16* noise, const __nv_bfloat16* w_in,
                   const __nv_bfloat16* b_in, __nv_bfloat16* x,
                   const __nv_bfloat16* weight, const __nv_bfloat16* style,
                   __nv_fp8_e4m3* out, __nv_bfloat16* gate_out,
                   const float* out_scale, int rows, float eps, bool pdl,
                   cudaStream_t stream);

// Action output projection (K = 1024, N = 32) with bias into `action`, the
// optional trace copies (pre-update noise, increment), then the in-place
// BF16 update noise += action. One CTA per row.
int action_out_residual(const __nv_bfloat16* x, const __nv_bfloat16* w_out,
                        const __nv_bfloat16* b_out, __nv_bfloat16* action,
                        __nv_bfloat16* noise, __nv_bfloat16* trace_x,
                        __nv_bfloat16* trace_delta, int rows, bool pdl,
                        cudaStream_t stream);

#ifdef ENABLE_PI05_SDE
// Stochastic step: noise += a + sigma * eps (device-side sigma; bit-identical
// to action_out_residual when *sigma == 0). eps rows match the noise rows.
int action_out_residual_sde(const __nv_bfloat16* x, const __nv_bfloat16* w_out, const __nv_bfloat16* b_out,
                            __nv_bfloat16* action, __nv_bfloat16* noise, __nv_bfloat16* trace_x,
                            __nv_bfloat16* trace_delta, const __nv_bfloat16* eps, const float* sigma, int rows,
                            bool pdl, cudaStream_t stream);
#endif

}  // namespace pi05_dec_skinny
}  // namespace flash_rt
