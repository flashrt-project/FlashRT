// SPDX-License-Identifier: Apache-2.0
#pragma once
// LaWAM (latent world action model) kernels FlashRT has no equivalent for.
// Every entry takes raw device pointers and a stream, allocates nothing and
// never synchronizes, so the whole policy step records into one CUDA graph.
// Entries return cudaError_t as int (0 on success).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flash_rt {
namespace lawam {

// Camera frames (uint8 HWC, 256x256x3) into the two vision-tower operands:
//   qwen  FP16 [views * 256, 768]  Qwen3-VL patch rows in 2x2 spatial-merge
//         order (block row, block col, sub row, sub col), features (c, py, px)
//         of (x / 255 - 0.5) / 0.5. The two temporal-patch copies of a still
//         image are folded into the patch-embedding weight by the caller.
//   dino  FP16 [256, 768]  DINOv3 patch rows of view 0 in raster order,
//         features (c, py, px) of ImageNet-normalized x / 255.
// frames is [num_views, 256, 256, 3] contiguous.
int preprocess_u8(const uint8_t* frames, int num_views, __half* qwen, __half* dino, cudaStream_t stream);

// Gated residual update and row norm, one CTA per row (dim / 8 threads):
//   x += (delta + delta_bias) * gate + delta2     (each term nullable; gate a
//                                                  [dim] vector, nullable = 1)
//   y  = norm(x)        mode 0 LayerNorm, 1 RMSNorm, 2 identity
//   y  = y * weight + bias                         (nullable)
//   y  = y * (1 + scale) + shift                   (AdaLN modulation, nullable)
//   out_fp8 = e4m3(y / *out_scale), out_f16 = y, out_bf16 = y   (each nullable)
// x and the outputs use row stride dim; dim % 8 == 0, dim <= 8192.
int gated_residual_norm(__half* x, const __half* delta, const __half* delta_bias, const __half* gate,
                        const __half* delta2, int mode, float eps, const __half* weight, const __half* bias,
                        const __half* scale, const __half* shift, __nv_fp8_e4m3* out_fp8, const float* out_scale,
                        __half* out_f16, __nv_bfloat16* out_bf16, int rows, int dim, cudaStream_t stream);

// y = gelu(x + bias) (erf, or the tanh approximation) for x [rows, n] (row
// stride ld) into out_fp8 = e4m3(y / *scale) and / or out_f16 [rows, n] (nullable).
int bias_gelu(const __half* x, int ld, const __half* bias, bool tanh_approx, __nv_fp8_e4m3* out_fp8,
              const float* scale, __half* out_f16, int rows, int n, cudaStream_t stream);

// out = silu(gate) * up, FP16 (gate / up row stride ld, out [rows, n]).
int silu_mul(const __half* gate, const __half* up, int ld, __half* out, int rows, int n, cudaStream_t stream);

// dst[idx[i], :] = src[i, :] + bias for i < n (bias nullable), FP16 rows of dim.
int scatter_rows(const __half* src, const __half* bias, const int* idx, __half* dst, int n, int dim,
                 cudaStream_t stream);

// out[i] = in[i] for FP16 -> BF16 (n % 8 == 0).
int cast_f16_bf16(const __half* in, __nv_bfloat16* out, int n, cudaStream_t stream);

// VLM -> latent action head (VLMToLAMQFormer, one query, one layer) and the
// LAM decoder conditioning: reads the act-query rows idx[0..nq) of the FP16
// VLM states [*, vdim], writes the latent (FP32 [code]) and
// silu(action_in_proj(latent)) (FP16 [ctx]) consumed by the AdaLN modulation.
// FP32 weights in nn.Module layout; scratch FP32 of at least nq * 2 * code.
struct QFormerWeights {
    const float* query;       // [code]
    const float* norm_q_w; const float* norm_q_b;     // [code]
    const float* norm_kv_w; const float* norm_kv_b;   // [vdim]
    const float* wq; const float* wk; const float* wv; // [code, code], [code, vdim] x2
    const float* in_bias;     // [3 * code] (q, k, v)
    const float* wo; const float* bo;                 // [code, code], [code]
    const float* ffn_ln_w; const float* ffn_ln_b;     // [code]
    const float* w1; const float* b1;                 // [4 code, code], [4 code]
    const float* w2; const float* b2;                 // [code, 4 code], [code]
    const float* final_w; const float* final_b;       // [code]
    const float* act_w; const float* act_b;           // [ctx, code], [ctx]
};
int qformer_latent(const __half* vlm, const int* idx, int nq, int vdim, int code, int heads, int ctx,
                   const QFormerWeights& w, float* scratch, float* latent, __half* cond_silu, cudaStream_t stream);

// x_t (FP32 [rows, n]) -> silu(x_t @ W1 + b1) BF16 [rows, h]; W1 FP32 [n, h].
int action_encode(const float* x, const float* w1, const float* b1, __nv_bfloat16* out, int rows, int n, int h,
                  cudaStream_t stream);

// Gaussian noise for the flow sample: x[rows, n] ~ N(0, 1) from a counter
// based generator keyed by (*seed, element); *seed is advanced by one per
// call, on device, so a replayed graph draws fresh noise every step.
int gaussian_noise(float* x, unsigned long long* seed, int count, cudaStream_t stream);

// actions[r, c] = (x[r, c] + 1) / 2 * (hi[c] - lo[c]) + lo[c] for c < out_dim
// of the FP32 flow sample x [rows, ld].
int unnormalize_actions(const float* x, int ld, const float* lo, const float* hi, float* actions, int rows,
                        int out_dim, cudaStream_t stream);

// D[M, N] = A[M, K] @ B[K, N] (+ bias[N]), FP16 operands and FP16 accumulation
// (cuBLASLt CUBLAS_COMPUTE_16F; row-major, B as [K, N]); gelu applies the
// tanh-approximated GELU after the bias (requires bias). Plans are cached per
// shape on first use: call once before graph capture.
int gemm_f16acc(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, bool gelu,
                cudaStream_t stream);
// Plans created after enable=true time the heuristic's non-split-K candidates
// and keep the fastest (default: the heuristic's first choice).
void gemm_f16acc_set_autotune(bool enable);

// D[M, N] (FP32) = A[M, K] @ W[N, K]^T, BF16 operands (W an nn.Linear weight
// as stored), FP32 accumulation. D is a single-split partial for the DiT
// consumers below (splits = 1, alpha = 1).
int gemm_bf16_f32out(const __nv_bfloat16* A, const __nv_bfloat16* W, float* D, int M, int N, int K,
                     cudaStream_t stream);

// Skinny weight-only INT8 GEMM for the DiT (M <= 32): partials[s][m][n] =
// w_scale[n] * sum_{k in split s} A[m][k] * W[n][k], A BF16 [M, K], W INT8
// [N, K] with per-output-channel scales, partials FP32 [K / KC][M][N] (the
// skinny NVFP4 layout; consumers use alpha = 1). Weights are prefetched
// before the programmatic-dependent-launch wait.
int w8a16_skinny_config_count();
int w8a16_skinny_config_k_chunk(int cfg);
int w8a16_skinny_gemm(const __nv_bfloat16* A, const int8_t* W, const float* w_scale, float* partials, int M, int N,
                      int K, int cfg, bool pdl, cudaStream_t stream);

// ── Skinny NVFP4 consumers for the flow DiT (sm_120a) ──
// Same partial layout, numerics and output formats as
// gemm/fp4/nvfp4_skinny_consumers_sm120.cuh, plus the per-row additive
// positional embedding LaWAM adds to every block input and the residual
// initialization of the first block.

// residual = (accumulate ? residual : 0) + bf16(bf16(alpha * sum partials) + bias),
// y = LayerNorm(residual) * (1 + scale) + shift + pos[row]   (scale/shift/pos nullable)
// into NVFP4 (out, out_sf, global *out_gs) or BF16 out_bf16.
int dit_sum_bias_residual_norm(const float* partials, int splits, const float* a_gs, const float* w_gs,
                               const __nv_bfloat16* bias, __nv_bfloat16* residual, bool accumulate,
                               const __nv_bfloat16* scale, const __nv_bfloat16* shift, const __nv_bfloat16* pos,
                               uint8_t* out, uint8_t* out_sf, const float* out_gs, __nv_bfloat16* out_bf16,
                               int rows, int dim, float eps, bool pdl, cudaStream_t stream);

// LayerNorm(x) * (1 + scale) + shift + pos[row] of a BF16 residual into NVFP4 / BF16.
int dit_layer_norm(const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
                   const __nv_bfloat16* pos, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                   __nv_bfloat16* out_bf16, int rows, int dim, float eps, bool pdl, cudaStream_t stream);

// y = act(bf16(bf16(alpha * sum partials) + bias)), act 0 identity, 1 ReLU,
// 2 tanh-GELU, into NVFP4 (out, out_sf, *out_gs) or BF16 out_bf16 [rows, n].
int dit_sum_bias_act(const float* partials, int splits, const float* a_gs, const float* w_gs,
                     const __nv_bfloat16* bias, int act, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                     __nv_bfloat16* out_bf16, int rows, int n, bool pdl, cudaStream_t stream);

// out = bf16(bf16(alpha * sum partials) + bias) (BF16 [rows, n]): a projection
// consumed by a regular attention kernel.
int dit_sum_bias_bf16(const float* partials, int splits, const float* a_gs, const float* w_gs,
                      const __nv_bfloat16* bias, __nv_bfloat16* out, int rows, int n, bool pdl, cudaStream_t stream);

// Euler step of the flow: x += dt * (alpha * sum partials + bias) (FP32 x [rows, n]).
int dit_sum_bias_euler(const float* partials, int splits, const float* a_gs, const float* w_gs,
                       const float* bias, float* x, float dt, int rows, int n, bool pdl, cudaStream_t stream);

}  // namespace lawam
}  // namespace flash_rt
