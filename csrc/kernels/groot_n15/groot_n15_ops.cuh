// ================================================================
// FlashRT — GR00T N1.5 BF16 norms and elementwise ops.
//
// The reference runs under bf16 autocast. These ops reproduce the dtype each
// value has there and where it is rounded:
//
//   * LayerNorm computes in fp32 and its output stays fp32 until the next
//     Linear casts it to bf16 -- so a LayerNorm feeding a Linear is written
//     here as bf16 directly (one rounding of the fp32 result).
//   * The vision tower, the language model and the DiT keep a bf16 residual
//     stream (a residual add rounds to bf16). The VL self-attention keeps an
//     fp32 one: vlln's fp32 output is added to bf16 sub-layer outputs.
//   * AdaLayerNorm is norm(x) * (1 + scale) + shift with norm(x) in fp32 and
//     (1 + scale), shift in bf16 -- two fp32 roundings after the norm. The
//     caller passes the bf16 tables (1 + scale) and shift.
//   * The embodiment-specific MLPs are torch.bmm followed by a bf16 bias add
//     (two roundings), then ReLU or swish = x * sigmoid(x) with sigmoid
//     rounded to bf16 before the product.
//   * GELU (tanh) follows torch's expression order in fp32.
//
// All entries take raw device pointers and a stream and allocate nothing, so
// they can be captured in a CUDA graph.
// ================================================================
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flash_rt::groot_n15 {

// r = res_in (+ delta, rounded to bf16), stored to res_out when given;
// out = bf16(LayerNorm(r)) with gamma/beta when given, none when null.
void layer_norm_bf16(const __nv_bfloat16* res_in, const __nv_bfloat16* delta,
                     __nv_bfloat16* res_out, const __nv_bfloat16* gamma,
                     const __nv_bfloat16* beta, __nv_bfloat16* out,
                     int rows, int dim, float eps, cudaStream_t stream);

// out32 = LayerNorm(x) in fp32 (bf16 input, affine).
void layer_norm_to_f32(const __nv_bfloat16* x, const __nv_bfloat16* gamma,
                       const __nv_bfloat16* beta, float* out32,
                       int rows, int dim, float eps, cudaStream_t stream);

// res32 += delta (fp32 add) when delta is given; out = bf16(LayerNorm(res32)).
void residual_layer_norm_f32(float* res32, const __nv_bfloat16* delta,
                             const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
                             __nv_bfloat16* out, int rows, int dim, float eps,
                             cudaStream_t stream);

// out = bf16(res32 + delta).
void residual_add_f32_to_bf16(const float* res32, const __nv_bfloat16* delta,
                              __nv_bfloat16* out, long long n, cudaStream_t stream);

// r = res_in (+ delta, rounded to bf16), stored to res_out when given;
// out = bf16((LayerNorm(r) * scale1) + shift) with LayerNorm unscaled and
// scale1 = bf16(1 + scale), shift bf16 vectors of length dim.
void ada_layer_norm_bf16(const __nv_bfloat16* res_in, const __nv_bfloat16* delta,
                         __nv_bfloat16* res_out, const __nv_bfloat16* scale1,
                         const __nv_bfloat16* shift, __nv_bfloat16* out,
                         int rows, int dim, float eps, cudaStream_t stream);

// t = x (+ bias[col], rounded to bf16); out = bf16(t + table[(row % period), col]).
void add_rows_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                   const __nv_bfloat16* table, __nv_bfloat16* out,
                   int rows, int dim, int period, cudaStream_t stream);

enum BiasAct : int { kBiasOnly = 0, kBiasRelu = 1, kBiasSwish = 2 };

// t = bf16(x + bias[col]); out = act(t).
void bias_act_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                   __nv_bfloat16* out, int rows, int dim, int act,
                   cudaStream_t stream);

// out[r, :dim] = bf16(x[r] + bias); out[r, dim:dim + tail_dim] = tail[r].
void concat_bias_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                      const __nv_bfloat16* tail, __nv_bfloat16* out,
                      int rows, int dim, int tail_dim, cudaStream_t stream);

// In-place GELU, tanh approximation, torch's fp32 expression.
void gelu_tanh_bf16(__nv_bfloat16* x, long long n, cudaStream_t stream);

// Per-row E4M3 quantization: scales[m] = amax(|x[m, :]|) / 448 and
// out[m, k] = e4m3(x[m, k] / scales[m]) (saturating).
void quant_rows_fp8(const __nv_bfloat16* x, uint8_t* out, float* scales,
                    int M, int K, cudaStream_t stream);

// In place: d[m, n] = bf16(d[m, n] * row_scale[m] * col_scale[n] (+ bias[n])),
// the dequantization of an FP8 GEMM run with unit scales. bias may be null.
void rescale_rows_bf16(__nv_bfloat16* d, const float* row_scale, const float* col_scale,
                       const __nv_bfloat16* bias, int M, int N, cudaStream_t stream);

// act32 = noise32; act_bf16 = bf16(noise32).
void actions_init(const float* noise32, float* act32, __nv_bfloat16* act_bf16,
                  long long n, cudaStream_t stream);

// act32 += bf16(dt * pred); act_bf16 = bf16(act32).
void euler_step(const __nv_bfloat16* pred, float* act32, __nv_bfloat16* act_bf16,
                float dt, long long n, cudaStream_t stream);

}  // namespace flash_rt::groot_n15
