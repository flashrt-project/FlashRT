// FlashRT — Hy-VLA Orin ViT fusion kernel declarations.
#pragma once
#include <cuda_runtime.h>

extern "C" void hyvla_vit_add_layer_norm_bf16(
    void* residual, const void* x_add,
    const void* ln_weight, const void* ln_bias,
    void* out, int rows, int dim, float eps, cudaStream_t stream);

// Fused (residual += x_add) + LayerNorm(residual + time_pe); pe is added for
// the norm only and not stored. x_add may be nullptr to skip the residual add.
extern "C" void hyvla_vit_res_add_ln_time_bf16(
    void* residual, const void* x_add, const void* pe,
    const void* ln_weight, const void* ln_bias, void* out,
    int rows, int dim, int n, int kf, float eps, cudaStream_t stream);

// ViT tail: out (num_cam,n,d) = last history frame of (x + x_add); x_add may be
// nullptr; x/x_add are (num_cam*K, n, d).
extern "C" void hyvla_vit_tail_slice_bf16(
    const void* x, const void* x_add, void* out,
    int num_cam, int K, int n, int d, cudaStream_t stream);

// ViT patch-embed bias add: y(B,C,H,W) += bias(C,).
extern "C" void hyvla_vit_patch_bias_bf16(
    void* y, const void* bias, int B, int C, int H, int W, cudaStream_t stream);

// ViT patch-embed + pos-embed: out(B,n,d) = xbuf(B,d,n)^T + pe(n,d), contiguous.
extern "C" void hyvla_vit_pos_add_bf16(
    const void* xbuf, const void* pe, void* out,
    int B, int n, int d, cudaStream_t stream);

// Merger NormalizedDwPooler 2x2 gating helpers.
extern "C" void hyvla_merger_pool_bf16(
    const void* x, void* new_x, void* fused,
    int B, int H, int W, int C, cudaStream_t stream);
extern "C" void hyvla_merger_gate_bf16(
    const void* score, const void* new_x, void* out,
    int B, int h, int w, int C, cudaStream_t stream);

// Scatter image-dependent prefix tokens into a preallocated prefix buffer.
extern "C" void hyvla_prefix_scatter_bf16(
    const void* merged, void* buf, const void* dest, int nt, int C,
    cudaStream_t stream);

// Fused (residual += x_add) + LayerNorm + block-128 FP8 quant. dim must be a
// multiple of 128; bit-identical to hyvla_vit_add_layer_norm_bf16 followed by
// fp8_per_token_block128_quant for the same inputs.
extern "C" void hyvla_vit_add_layer_norm_to_fp8_block128_bf16(
    void* residual, const void* x_add,
    const void* ln_weight, const void* ln_bias,
    void* out_fp8, float* scale, int rows, int dim, float eps,
    cudaStream_t stream);

// Gather the ViT spatial FA2 output o (bk,H,N,Ds) into the proj activation
// (rows=bk*N, K=H*Dh) and quantize to NVFP4 (packed u8 + swizzled UE4M3 SFA).
extern "C" void hyvla_vit_proj_gather_nvfp4_swizzled_bf16(
    const void* o, void* out_fp4, void* out_sfa, int bk, int H, int N,
    int Ds, int Dh, cudaStream_t stream);

// Aspect-preserving bilinear resize + zero centre-pad (Pi0 style), bf16.
extern "C" void hyvla_resize_pad_bilinear_bf16(
    const void* in, void* out, int B, int Cc, int H, int W,
    int RH, int RW, int OH, int OW, int pad_top, int pad_left,
    float pad_value, cudaStream_t stream);

// Same with a fused affine on the output: out = resize(...)*scale + offset.
extern "C" void hyvla_resize_pad_bilinear_scale_bf16(
    const void* in, void* out, int B, int Cc, int H, int W,
    int RH, int RW, int OH, int OW, int pad_top, int pad_left,
    float pad_value, float scale, float offset, cudaStream_t stream);

// Fused (residual += x_add) + LayerNorm + NVFP4 swizzled quant (per-16 UE4M3
// group scales). out_fp4 is (rows, dim/2) u8; out_sfa is the Sm1xx swizzled
// scale layout (nvfp4_sf_swizzled_bytes(rows, dim) bytes). dim % 16 == 0.
extern "C" void hyvla_vit_add_layer_norm_to_nvfp4_swizzled_bf16(
    void* residual, const void* x_add,
    const void* ln_weight, const void* ln_bias,
    void* out_fp4, void* out_sfa, int rows, int dim, float eps,
    cudaStream_t stream);

// Causal-in-time softmax over K frames folded onto V (VyVLA spacetime /
// memory-encoder mix). q/k/v are (b*kf, H, N, D) bf16 with the given strides
// for the (b*kf, H, N, D) view (D contiguous); out is contiguous in that
// shape. kf in [1,8], D <= 128. Unsupported shapes return without writing.
extern "C" void hyvla_vit_temporal_mix_bf16(
    const void* q, const void* k, const void* v, void* out,
    int b, int kf, int H, int N, int D,
    long s_bk, long s_h, long s_n, float scale, cudaStream_t stream);

// Gather the FA2 denoise query buffer qb (2,S,H,D) from q (1,H,S,D); writes
// only the non-dummy rows (qb must be zeroed once at allocation).
extern "C" void hyvla_fa2_denoise_prepare_q_bf16(
    const void* q, void* qb, int S, int H, int D, cudaStream_t stream);

// Assemble att (S,H*D) from the FA2 denoise ob (2,S,H,D) and quantize it to
// block-128 FP8 (per-token, per-128-K-block scales) in one pass.
extern "C" void hyvla_fa2_denoise_gather_o_fp8_block128_bf16(
    const void* ob, void* a8, void* ascale, int S, int H, int D,
    cudaStream_t stream);

// Assemble att (S,H*D) from the FA2 denoise ob (2,S,H,D) and quantize it to
// NVFP4 (per-16 UE4M3 scales, swizzled) in one pass.
extern "C" void hyvla_fa2_denoise_gather_o_nvfp4_bf16(
    const void* ob, void* packed, void* sf_swz, int S, int H, int D,
    cudaStream_t stream);
