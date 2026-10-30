// SPDX-License-Identifier: Apache-2.0
// MolmoAct2 inference helper kernels. See molmoact2_kernels.cu.
#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace molmoact2 {

// out[n] = a[n] * b[n], bf16.
int mul_bf16_out(const void* a, const void* b, void* out, int n,
                 cudaStream_t stream, void* of8 = nullptr,
                 const void* fscale = nullptr);

// HF-exact RMSNorm: norm -> bf16, optional weight multiply -> bf16.
int rms_norm_bf16(const void* x, const void* w, void* out, int rows,
                  int dim, float eps, cudaStream_t stream);

// res[r,d] = bf16(res[r,d] + bf16(x[r,d] * gate[d])).
int gate_residual_bf16(void* res, const void* x, const void* gate, int rows,
                       int dim, cudaStream_t stream);

// Fused producers: bias / static-FP8 quantization variants.
int bias_rms_norm_bf16(const void* x, const void* bias, const void* w,
                        void* out, int rows, int heads, int hd, float eps,
                        cudaStream_t stream);
int skinny_qkv_split_bias(const void* partials, int splits, const void* ags,
                         const void* wgs, const void* bias, void* Q, void* K,
                         void* V, int rows, int qd, int kd, int vd,
                         cudaStream_t stream);
int skinny_gate_residual(const void* partials, int splits, const void* ags,
                         const void* wgs, const void* bias, void* res,
                         const void* gate, int rows, int dim,
                         cudaStream_t stream);
int skinny_bias_rms_norm(const void* partials, int splits, const void* ags,
                         const void* wgs, const void* bias, void* out,
                         int rows, int heads, int hd, float eps,
                         cudaStream_t stream);
int skinny_silu_mul(const void* pg, int splits_g, const void* ags_g,
                    const void* wgs_g, const void* pu, int splits_u,
                    const void* ags_u, const void* wgs_u, const void* bg,
                    const void* bu, void* out, int rows, int dim,
                    cudaStream_t stream);
// rms_norm_bf16 (dim 768, unweighted) then rms_norm_rows96_exact over the
// 8 heads of each row, in one launch (bit-identical). hm_rows > 0 writes the
// head-major layout [rows / hm_rows][8 heads][hm_rows][hm_stride] instead of
// [rows, 768].
int rms_norm768_heads96(const void* x, void* out, int rows, float eps,
                        float eps_head, cudaStream_t stream, int hm_rows = 0,
                        int hm_stride = 0);
// rms_norm_bf16 (dim 768, unweighted) with that head-major output.
// Byte-identical warp-per-row forms of rms_norm768_heads96 (heads != 0)
// and rms_norm768_headmajor (heads == 0).
int rms_norm768_ctx_warp(const void* x, void* out, int rows, float eps,
                         float eps_head, int heads, int hm_rows,
                         int hm_stride, cudaStream_t stream);
int rms_norm768_headmajor(const void* x, void* out, int rows, float eps,
                          int hm_rows, int hm_stride, cudaStream_t stream);

// Q, K or V may be null to skip that bf16 output (the FP8 copies, when
// requested, are still written).
// FP8-only vectorized form for head_dim 128 (no bias, RoPE, q/k norm
// weights). Qf8 / Vf8 may be null. Not bit-identical to the generic kernel:
// the per-head sum of squares is reduced in a different order.
int qkv_norm_rope_fp8_hd128(const void* qkv, const void* qw, const void* kw,
                            const void* cos_t, const void* sin_t, void* Qf8,
                            void* Kf8, void* Vf8, const void* qscale,
                            const void* kscale, const void* vscale, int rows,
                            int nq, int nk, int vd, float eps,
                            cudaStream_t stream);

int qkv_split_bias_norm_rope(const void* qkv, const void* bias,
                              const void* qw, const void* kw,
                              const void* cos_t, const void* sin_t, void* Q,
                              void* K, void* V, int rows, int qd, int kd,
                              int vd, int nq, int nk, int hd, float eps,
                              int has_bias, int has_rope, void* Kf8,
                              void* Vf8, const void* kscale,
                              const void* vscale, void* Qf8,
                              const void* qscale, cudaStream_t stream);
int layer_norm_quant_fp8(const void* x, const void* w, const void* b,
                         void* out, const void* scale, int rows, int dim,
                         float eps, cudaStream_t stream);
// dst[i] = 1.0f / src[i], compiled with the same flags as the static-scale
// FP8 producers so the reciprocal matches their in-kernel `1.0f / *scale`.
int reciprocal_f32(const void* src, void* dst, int n, cudaStream_t stream);

int bias_gelu_quant_fp8_vec(const void* x, const void* b, void* out,
                            const void* scale, int n, int dim,
                            cudaStream_t stream);
int bias_gelu_quant_fp8(const void* x, const void* b, void* out,
                        const void* scale, int rows, int dim,
                        cudaStream_t stream);
int qkv_split_vec(const void* qkv, void* Q, void* K, void* V, int rows,
                  int qd, int kd, int vd, cudaStream_t stream);
int qkv_split_bias(const void* qkv, const void* bias, void* Q, void* K,
                   void* V, int seq, int qd, int kd, int vd,
                   cudaStream_t stream);
int silu_mul_bias(const void* g, const void* u, const void* bg,
                  const void* bu, void* out, int rows, int dim,
                  cudaStream_t stream);
int rms_norm_rows96_exact(const void* x, void* out, int rows, float eps,
                          cudaStream_t stream);
// Same replica with the per-head bias add folded into the load.
int bias_rms_norm_rows96_exact(const void* x, const void* bias, void* out,
                               int rows, int heads, float eps,
                               cudaStream_t stream);
// Bit-exact warp-per-row replica of mod_rms_norm_quant_fp8 for dim == 768.
// skinny_gate_residual + mod_rms_norm_rows768_quant_fp8_exact (dim 768) in
// one launch; residual stream and FP8 output are bit-identical.
int skinny_gate_residual_mod_rms768_quant_fp8(
    const void* partials, int splits, const void* ags, const void* wgs,
    const void* bias, void* res, const void* gate, const void* scale,
    const void* shift, void* out, const void* qscale, int rows, float eps,
    cudaStream_t stream);

int mod_rms_norm_rows768_quant_fp8_exact(
    const void* x, const void* scale, const void* shift, void* out,
    const void* qscale, int rows, float eps, cudaStream_t stream);
// Bit-exact warp-per-row replica of mod_rms_norm_bf16 for dim == 768.
int mod_rms_norm_rows768_exact(const void* x, const void* scale,
                               const void* shift, void* out, int rows,
                               float eps, cudaStream_t stream);
int rms_norm_rows_bf16(const void* x, const void* w, void* out, int rows,
                       int dim, float eps, cudaStream_t stream);
int gate_residual_bias(void* res, const void* x, const void* gate,
                       const void* bias, int rows, int dim,
                       cudaStream_t stream);
int mod_rms_norm_quant_fp8(const void* x, const void* w, const void* scale,
                           const void* shift, void* out, const void* qscale,
                           int rows, int dim, float eps,
                           cudaStream_t stream);
int silu_mul_quant_fp8(const void* g, const void* u, const void* bg,
                       const void* bu, void* out, const void* qscale,
                       int rows, int dim, cudaStream_t stream);

// bf16 <-> fp16 casts.
int cast_bf16_to_fp16(const void* src, void* dst, int n, cudaStream_t stream);
int cast_fp16_to_bf16(const void* src, void* dst, int n, cudaStream_t stream);

// out[n] = silu(g[n]) * u[n], bf16.
int silu_mul_bf16(const void* g, const void* u, void* out, int n,
                  cudaStream_t stream);

// out[r,d] = bf16(acc_f32[r,d] + bias_bf16[d]) -- fused-epilogue equivalent.
// out = bf16(bf16(acc + bias) + add): bias_round_bf16 then res_add, one pass.
int bias_round_add_bf16(const void* acc, const void* bias, const void* add,
                        void* out, int rows, int dim, cudaStream_t stream);
int bias_round_bf16(const void* acc, const void* bias, void* out, int rows,
                    int dim, cudaStream_t stream);

// x[rows, dim] += bias[dim], bf16.
int add_bias_bf16(void* x, const void* bias, int rows, int dim,
                  cudaStream_t stream);

// out[r,d] = clamp(traj,-1,1) unnormalized from q01/q99 (mask passthrough).
int action_denorm_bf16(const void* traj, void* out, const void* q01,
                       const void* q99, const void* mask, int rows,
                       int stride, int act_dim, cudaStream_t stream);

// Per-head RMSNorm + rotate-half RoPE, in place over x[S, NH*HD].
// w may be nullptr when the norm has no affine weight.
int qk_norm_rope_bf16(void* x, const void* w, const void* cos_t,
                      const void* sin_t, int S, int NH, int HD, float eps,
                      int has_weight, cudaStream_t stream);

// out[P,D] = mean over valid rows feats[idx[P,I], D].
int pool_mean_bf16(const void* feats, const void* idx, void* out,
                   int P, int I, int D, cudaStream_t stream,
                   void* of8 = nullptr, const void* fscale = nullptr);

// out[P,NH,HD] = softmax(q[P,NH,HD] @ k[idx[P,I],NH,HD]^T) @ v[idx[P,I],NH,HD].
int pool_attn_bf16(const void* q, const void* k, const void* v,
                   const void* idx, void* out, int P, int I, int NH, int HD,
                   float scale, cudaStream_t stream, void* of8 = nullptr,
                   const void* fscale = nullptr);
int concat2_quant_fp8(const void* a, const void* b, void* out, int rows,
                      int cols_a, int cols_b, cudaStream_t stream,
                      void* of8 = nullptr, const void* fscale = nullptr);

// out[rows,dim] = rmsnorm(x,w)[rows,dim] * (1 + scale) + shift, with
// scale/shift broadcast over rows. w may be nullptr.
int mod_rms_norm_bf16(const void* x, const void* w, const void* scale,
                      const void* shift, void* out, int rows, int dim,
                      float eps, cudaStream_t stream);

// q01/q99 state normalization -> discrete bins -> token ids.
// mask/zero_mask are optional int8[d].
int state_discretize_f32(const void* state, const void* q01, const void* q99,
                         const void* mask, const void* zero_mask, void* ids,
                         int D, int num_tokens, int base_id,
                         cudaStream_t stream);

// Zero x[r, valid:dim] for every row.
int mask_dims_bf16(void* x, int rows, int dim, int valid,
                   cudaStream_t stream);

// uint8 HWC image -> bf16 [n_patch_y*n_patch_w, patch*patch*C] rows.
// Bilinear resize (align_corners=False) to (OH,OW), uint8 round, then
// normalize to [-1,1]. One image per launch; caller offsets pointers.
// out_stride: element stride between patch rows of out (0: patch*patch*C).
int resize_patchify_bf16(const void* img, void* out, int H, int W, int C,
                         int OH, int OW, int patch, cudaStream_t stream,
                         int out_stride = 0);

// Fused action-expert SwiGLU producer: up/gate FP8 GEMMs + biases +
// silu(gate)*up in one launch. A: [M,K] e4m3, Wu/Wg: [N,K] e4m3 row-major,
// out: bf16 [M,N]; alpha_* fold the activation and weight scales. M <= 32,
// K a multiple of 128, N a multiple of 32.
int action_swiglu_fp8(const void* A, const void* Wu, const void* Wg,
                      const void* bu, const void* bg, void* out,
                      int M, int K, int N, float alpha_u, float alpha_g,
                      cudaStream_t stream);

}  // namespace molmoact2
}  // namespace flash_rt
