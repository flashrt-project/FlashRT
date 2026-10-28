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

// VLM -> latent action head (VLMToLAMQFormer, one query, one layer) and the
// LAM decoder conditioning: reads the act-query rows idx[0..nq) of the FP16
// VLM states [*, vdim], writes the latent (FP32 [code]) and
// silu(action_in_proj(latent)) (FP16 [ctx]) consumed by the AdaLN modulation.
// FP32 weights; wk / wv in nn.Module layout [out, in], the head projections
// transposed (_t: [in, out]). vdim 2048; scratch FP32 of at least nq * 2 * code.
struct QFormerWeights {
    const float* query;       // [code]
    const float* norm_q_w; const float* norm_q_b;     // [code]
    const float* norm_kv_w; const float* norm_kv_b;   // [vdim]
    const float* wq_t; const float* wk; const float* wv; // [code, code], [code, vdim] x2
    const float* in_bias;     // [3 * code] (q, k, v)
    const float* wo_t; const float* bo;               // [code, code], [code]
    const float* ffn_ln_w; const float* ffn_ln_b;     // [code]
    const float* w1_t; const float* b1;               // [code, 4 code], [4 code]
    const float* w2_t; const float* b2;               // [4 code, code], [code]
    const float* final_w; const float* final_b;       // [code]
    const float* act_w_t; const float* act_b;         // [code, ctx], [ctx]
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

// out[n] = w_scale[n] * sum_k w[n, k] * x[k] + bias[n]: one-row GEMV with INT8
// weights [N, K] (per-output-channel scales), FP16 input / output, FP32
// accumulation. K % 512 == 0, K <= 12288.
// out[n] = gs * sum_k e2m1(w[n, k]) * sf[n, k / 16] * x[k] + bias[n]: one-row GEMV
// with NVFP4 weights (w [N, K / 2] bytes, element 2j in the low nibble; sf
// [N, K / 16] UE4M3 block scales; gs one FP32 global scale), FP16 input /
// output, FP32 accumulation. K == 1024. max_ctas > 0 caps the grid (a GEMV
// running next to other kernels leaves them thread slots); rows are computed
// the same way for any grid.
int gemv_nvfp4w(const __half* x, const uint8_t* w, const uint8_t* sf, const float* gs, const __half* bias,
                __half* out, int N, int K, cudaStream_t stream, int max_ctas = 0);

int gemv_i8w(const __half* x, const int8_t* w, const float* w_scale, const __half* bias, __half* out, int N, int K,
             cudaStream_t stream);

// D[M, N] = A[M, K] @ B[K, N] (+ bias[N]), FP16 operands and FP16 accumulation
// (cuBLASLt CUBLAS_COMPUTE_16F; row-major, B as [K, N]); gelu applies the
// tanh-approximated GELU after the bias (requires bias). Plans are cached per
// shape on first use: call once before graph capture.
int gemm_f16acc(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, bool gelu,
                cudaStream_t stream);
// Plans created after enable=true time the heuristic's non-split-K candidates
// and keep the fastest (default: the heuristic's first choice).
void gemm_f16acc_set_autotune(bool enable);
int gemm_f16acc_bf16out(const __half* A, const __half* B, __nv_bfloat16* D, const __nv_bfloat16* bias, int M, int N,
                        int K, int tile, cudaStream_t stream);

// D[M, N] (FP32) = A[M, K] @ W[N, K]^T, BF16 operands (W an nn.Linear weight
// as stored), FP32 accumulation. D is a single-split partial for the DiT
// consumers below (splits = 1, alpha = 1).
int gemm_bf16_f32out(const __nv_bfloat16* A, const __nv_bfloat16* W, float* D, int M, int N, int K,
                     cudaStream_t stream);

// Skinny weight-only NVFP4 GEMM (lawam_w4a16_skinny_sm120.cu): FP32 K-split
// partials [K / kc][M][N] of A (BF16 [M, K], M <= 32) @ (E2M1 W [N, K / 2] *
// UE4M3 sf [N, K / 16])^T, without the per-tensor scale (the consumers' alpha).
int w4a16_skinny_config_count();
int w4a16_skinny_config_k_chunk(int cfg);
int w4a16_skinny_gemm(const __nv_bfloat16* A, const uint8_t* W, const uint8_t* sf, float* partials, int M, int N,
                      int K, int cfg, bool pdl, cudaStream_t stream);


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

// D[M,N] = act(fp16(A @ B) + bias) with FP16 accumulation (CUTLASS SM80 tiles,
// lawam_gemm_f16acc_act.cu); A [M,K], B [K,N], D row-major FP16; act 0 identity (bias
// optional), 1 tanh-GELU, 2 erf-GELU, computed in FP32 as bias_gelu does. split
// fixes the serial split-K count (each K slice accumulates its own FP16 sum;
// gemm_f16acc_split gives the one gemm_f16acc uses, i.e. its exact numerics),
// split <= 0 times it too. The tile is timed on the first call per shape
// (outside graph capture); _cfg forces a tile (0..7) and split (1..8).
int gemm_f16acc_act(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, int act,
                    int split, cudaStream_t stream);
// SwiGLU as one dual GEMM (FP16 accumulation, serial split-K split; split
// <= 0 times it with the tile):
// Dg = A @ Bg, Du = A @ Bu (FP16 [M, N] slices / sums, leading dim ldd; Bg, Bu
// [K, N] with leading dim ldb) and D [M, N] = silu(Dg) * Du in FP32, as
// gemm_f16acc on [Bg | Bu] followed by silu_mul. The tile is timed per shape
// on the first call (outside graph capture); _cfg forces tile 0..3.
int gemm_f16acc_swiglu(const __half* A, const __half* Bg, const __half* Bu, int ldb, __half* Dg, __half* Du, int ldd,
                       __half* D, int M, int N, int K, int split, cudaStream_t stream);
int gemm_f16acc_swiglu_cfg(const __half* A, const __half* Bg, const __half* Bu, int ldb, __half* Dg, __half* Du,
                           int ldd, __half* D, int M, int N, int K, int tile, int split, cudaStream_t stream);
// Serial split-K count of the cuBLASLt plan gemm_f16acc runs for this shape (-1: none).
int gemm_f16acc_split(int M, int N, int K, bool bias, bool gelu);
int gemm_f16acc_act_cfg(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, int act,
                        int tile, int split, cudaStream_t stream);

// Persistent FP16-accumulate GEMM (lawam_gemm_persistent_sm120.cu): one CTA
// per SM over a static data-parallel + stream-K schedule of output tiles,
// D [M, out] (leading dim ldd) = act(A[M,K] @ B[K,N] + bias): act 0 identity
// (bias optional), 1 tanh-GELU, 2 erf-GELU, 3 SwiGLU over B columns
// interleaved in blocks of 8 (gate block, up block; out = N / 2 columns).
// cfg: tile 0 224x128, 1 96x128, 2 128x256, 3 64x256, 4 128x128 (k 64),
// 5 256x128; sched 0 hybrid, 1 data-parallel, 2 stream-K. The first call of
// a shape / stream allocates its schedule and workspace (outside a capture).
int gemm_f16acc_persistent(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K,
                           int ldd, int act, int cfg, int sched, cudaStream_t stream);

// Cross-attention of the DiT action rows over a long static K / V
// (lawam_dit_cross_attention_sm120.cu): Q = alpha * sum of the q-projection's
// K-split partials [splits][rows][heads * hd] + bias, K / V [kv_seq] rows of
// leading dim kv_ld (BF16), keys split over kv_slices CTAs per (head, 16-row
// tile) and merged by the last one. state: dit_cross_attention_state_floats
// FP32, counters: heads * ceil(rows / 16) zeroed words (left zeroed). hd 64.
size_t dit_cross_attention_state_floats(int rows, int heads, int kv_slices);
int dit_cross_attention(const float* partials, int splits, const float* a_gs, const float* w_gs,
                        const __nv_bfloat16* bias, const __nv_bfloat16* k_ext, const __nv_bfloat16* v_ext, int kv_ld,
                        int kv_seq, int kv_slices, __nv_bfloat16* out, int rows, int heads, int hd, float scale,
                        float* state, unsigned* counters, bool pdl, cudaStream_t stream);

// Q / K / V of a fused [S, ld] qkv GEMM output (Q | K | V heads of 64):
// + bias, K / V scale, rotate-half RoPE (cos / sin [S, 64]) on Q and K, into
// [S, heads, 64] FP16 / BF16 (the FlashRT qkv_split arithmetic without QK norm).
int qk_rope_inplace(__half* qkv, int ld, const __half* cos, const __half* sin, int S, int n_qk, int hd,
                    cudaStream_t stream);
int qkv_rope_split(const __half* qkv, int ld, const __half* q_bias, const __half* k_bias, const __half* v_bias,
                   float k_mul, float v_mul, const __half* cos, const __half* sin, void* q_out, void* k_out,
                   void* v_out, bool out_bf16, int S, int n_q, int n_k, int hd, cudaStream_t stream);

// The flow loop as one persistent cooperative kernel
// (lawam_dit_persistent_sm120.cu). ops: device table of DitOp built by the
// frontend (kind, GEMM site / mode / scalars, raw pointers); site_tab: int32
// [sites][num_ctas][4] (first 16-column tile, tile count, first packed unit,
// first 256-wide k chunk) of every GEMM site; stream: each CTA's weight units
// of one flow step packed in consumption order (unit stride 2304 bytes, CTA
// stride cta_stride; cta_units[c] units, replayed for every one of n_steps
// steps). counter: one word (zeroed by the call). num_ctas: one CTA per SM.
struct DitOp {
    int32_t i[16];
    float f[8];
    uint64_t p[16];
};
static_assert(sizeof(DitOp) == 224, "DitOp layout is mirrored by the frontend");
enum DitOpField {
    DIT_I_KIND = 0, DIT_I_BAR = 1, DIT_I_SITE = 2, DIT_I_WTYPE = 3, DIT_I_N = 4, DIT_I_K = 5, DIT_I_EP = 6,
    DIT_I_KVLEN = 7, DIT_I_KLD = 8, DIT_I_VTLD = 9, DIT_I_KSPLIT = 10, DIT_I_OWNERS = 11,
    DIT_F_EPS = 0, DIT_F_DT = 1, DIT_F_QSCALE = 2,
    DIT_P_ABUF = 0, DIT_P_LNSC = 1, DIT_P_LNSH = 2, DIT_P_POS = 3, DIT_P_BIAS = 4, DIT_P_BIAS32 = 5, DIT_P_GS = 6,
    DIT_P_WS = 7, DIT_P_OUT0 = 8, DIT_P_OUT1 = 9, DIT_P_OUT2 = 10, DIT_P_Q = 11, DIT_P_K = 12, DIT_P_VT = 13,
    DIT_P_PART = 14, DIT_P_STATS = 15,
};
int dit_flow_smem_bytes();
int dit_flow_persistent(const DitOp* ops, int n_ops, const int* site_tab, int n_sites, const int* cta_units,
                        int n_steps, const uint8_t* stream, long long cta_stride, unsigned* counter, int num_ctas,
                        cudaStream_t st, bool reset_counter = true);

}  // namespace lawam
}  // namespace flash_rt
