// FlashRT — X-VLA / Florence-2 DaViT vision tower kernels (SM120).
//
// DaViT alternates a spatial block (12x12 window attention) with a channel
// block (grouped channel attention) inside each of four conv-embed stages.
// The graph is kept in NHWC throughout: the reference's NCHW-to-NHWC permutes
// around every LayerNorm are pure layout, and NHWC makes the per-token
// dimension contiguous for the norm, GEMM and attention steps.
//
// Tokens are batched over the camera views: the two real views share one pass
// through each stage, so every GEMM's M is doubled and the tower's ~460 small
// kernels run once instead of twice. Kernels whose indexing is per-image
// (im2col, depthwise conv, window partition, channel attention) take a `batch`
// count and derive the image index from the token index.
//
// Every entry is templated on the token dtype and instantiated for bf16 and
// fp16. The tower is 72 sequential GEMMs, so its accumulated rounding error is
// what limits end-to-end action fidelity; fp16's wider mantissa cuts it ~5x.
//
// Two details matter for numerical equivalence with `Florence2VisionBackbone`:
//   * window padding is zero-filled and NOT masked, so padded tokens take part
//     in the softmax with their (bias-only) keys and values;
//   * the channel-attention scale is `N ** -0.5` where N is the number of
//     spatial tokens, not the channel-group width.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace flash_rt {
namespace xvla {

// LayerNorm with an fp16 input and a bf16 output — the DaViT/encoder dtype
// boundary, fused so no separate narrowing kernel is needed.
void vision_ln_f16_bf16(const __half* x, const __half* w, const __half* b,
                        __nv_bfloat16* out, int rows, int dim, float eps,
                        cudaStream_t stream);

// Bilinear resize of an f32 CHW batch into NHWC, matching
// `F.interpolate(mode="bilinear", align_corners=False)`:
//   src = max(0, (dst + 0.5) * in/out - 0.5)
void vision_resize_bilinear_f32(void* dst, const float* src, int n, int cin,
                                int hin, int win_, int hout, int wout,
                                bool half, cudaStream_t stream);

// im2col for a NHWC image: col[oh*OW + ow, ci*k*k + kh*k + kw] <- x[h, w, ci].
// Zero past the image border. `col` is [out_h*out_w, cin*k*k].
void vision_im2col(void* col, const void* x, int H, int W, int cin, int k,
                   int stride, int pad, int out_h, int out_w, int batch, bool half,
                   cudaStream_t stream);

// x[..., co] += bias[co], over `rows` rows of a contiguous [rows, cols] buffer.
void vision_add_bias(void* x, const void* bias, int rows, int cols, bool half,
                     cudaStream_t stream);

// Depthwise 3x3 conv in NHWC with a fused residual: out = dwconv(x) + b + x.
void vision_dwconv3x3_res(void* out, const void* x, const void* w,
                          const void* bias, int H, int W, int C, int batch,
                          bool half, cudaStream_t stream);


// Window attention over a fused QKV buffer [nw*ws*ws, 3*C].
//   out[n, h*32 + d] = sum_j softmax_j(q.k^T * hd^-0.5)[j] * v[j]
// One block per (window, head); head_dim is 32 for every DaViT stage.
void vision_window_attn(void* out, const void* qkv, int nw, int ws, int C,
                        int heads, bool half, cudaStream_t stream);

// Scatter the image-order QKV buffer [batch*H*W, 3*C] into the padded window
// (batch*nwin, ws*ws, heads, 64) q/k/v operands; slots past the image edge get
// the QKV bias (the reference pads before the Linear) and the head dims above
// 32 stay zero. DaViT's head_dim is 32 and the vendored FA2 has no hd=32
// instantiation, but padding the head to 64 with zeros is exact: the extra q.k
// terms are 0 and the softmax denominator is unchanged, so only the op count
// moves.
void vision_window_qkv_pad(void* q, void* k, void* v, const void* qkv,
                           const void* bias, int H, int W, int ws, int C, int heads,
                           int nwin, int batch, bool half, cudaStream_t stream);

// One-time fill of a window block's padded slots with its QKV bias.
void vision_window_qkv_pad_fill(void* q, void* k, void* v, const void* bias,
                                int nw, int ws, int C, int heads, bool half,
                                cudaStream_t stream);

// Inverse: gather the real window slots back into image order, keeping the
// first 32 head dims and emitting [batch*H*W, C].
void vision_window_out_unpad_to_nvfp4(void* packed, void* sf_swz,
                                      const void* o, int H, int W, int ws,
                                      int C, int heads, int batch,
                                      bool half, cudaStream_t stream);

// Grouped channel attention over a fused QKV buffer [N, 3*C], C = groups*hg.
//   attn[g,i,j] = softmax_j(q[.,g,i] . k[.,g,j] * N^-0.5)
//   out[n, g*hg + i] = sum_j attn[g,i,j] * v[n, g, j]
void vision_channel_attn(void* out, const void* qkv, float* attn_ws, int N,
                         int C, int groups, int batch, bool half,
                         cudaStream_t stream);

// Florence2MultiModalProjector: add the learned 2-D position map and the
// temporal vector, then emit [spatial_avg_pool, patch tokens] as 1 + rows.
//   out[0, :]      = mean_r(feat[r] + pos[r] + tv)
//   out[1 + r, :]  = feat[r] + pos[r] + tv
void vision_projector_pack(void* out, const void* feat, const void* pos,
                           const void* temporal, int rows, int dim, bool half,
                           cudaStream_t stream);

// erf GELU straight into the packed NVFP4 operand fc2 reads. The DaViT FFN
// uses the erf form; `gelu_inplace_fp16` and `bias_gelu_*_strict` are the tanh
// approximation and would diverge.
void vision_gelu_erf_to_nvfp4(const void* x, void* packed, void* sf_swz,
                               int rows, int cols, cudaStream_t stream);

}  // namespace xvla
}  // namespace flash_rt
