// pi0.5 Thor (SM110) SigLIP vision stage on FlashRT kernels: patch embedding,
// transformer layers (FP8 attention with FA4, NVFP4 FFN), post-LayerNorm and
// projection to the language model width.
//
// Framework-free: raw device pointers in, raw device pointers out. The
// TensorRT plugins and the parity test both call this.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

class GemmRunner;

namespace flashrt_trt {
namespace pi05 {

struct SiglipDims {
    int D = 1152;         // hidden size
    int H_pad = 4320;     // FFN width, zero-padded to a multiple of 32
    int NH = 16;          // attention heads
    int HD = 72;          // head dim
    int spv = 256;        // tokens per view (224 / 14 squared)
    int De = 2048;        // projection width
    int up_variant = 2;   // NVFP4 GEMM variant for the FFN up projection
};

struct SiglipLayerWeights {
    const void* ln_attn_w = nullptr;    // fp16 [D]
    const void* ln_attn_b = nullptr;    // fp16 [D]
    const void* qkv_w = nullptr;        // e4m3 [D, 3D]
    const void* qkv_b = nullptr;        // fp16 [3D]
    float qkv_alpha = 1.0f;             // weight descale (host)
    const void* o_w = nullptr;          // e4m3 [D, D]
    const void* o_b = nullptr;          // fp16 [D]
    float o_alpha = 1.0f;
    const void* ln_ffn_w = nullptr;     // fp16 [D]
    const void* ln_ffn_b = nullptr;     // fp16 [D]
    const void* awq_inv_s = nullptr;    // fp16 [D]
    const void* up_packed = nullptr;    // NVFP4 [H_pad, D/2]
    const void* up_sfb = nullptr;
    const void* up_b = nullptr;         // fp16 [H_pad]
    const void* down_packed = nullptr;  // NVFP4 [D, H_pad/2]
    const void* down_sfb = nullptr;
    const void* down_b = nullptr;       // fp16 [D]
};

struct SiglipEmbedWeights {
    const void* lut = nullptr;       // fp16 [256], uint8 pixel -> [-1, 1] (uint8 images only)
    const void* pe_w = nullptr;      // fp16 [588, D], HWC im2col order
    const void* pe_b = nullptr;      // fp16 [D]
    const void* pos_emb = nullptr;   // fp16 [spv, D]
    const void* postln_w = nullptr;  // fp16 [D]
    const void* postln_b = nullptr;  // fp16 [D]
    const void* proj_w = nullptr;    // fp16 [D, De]
    const void* proj_b = nullptr;    // fp16 [De]
};

struct SiglipScratch {
    void* x_fp8 = nullptr;       // [S, D]
    void* qkv = nullptr;         // fp16 [S, 3D], Q/K/V interleaved per row
    void* attn = nullptr;        // fp16 [S, D]
    void* ln_packed = nullptr;   // NVFP4 activations into the up projection
    void* ln_sfa = nullptr;
    void* hid_packed = nullptr;  // NVFP4 activations into the down projection
    void* hid_sfa = nullptr;
    void* patches = nullptr;     // fp16 [S, 588]
    void* norm = nullptr;        // fp16 [S, D]
    void* x = nullptr;           // fp16 [S, D], residual stream for the whole-stage caller
};

// Bytes of one contiguous scratch block for max_s tokens.
int64_t siglip_scratch_bytes(const SiglipDims& dims, int max_s);

// Carve the scratch block into aligned sub-buffers for max_s tokens.
void siglip_bind_scratch(const SiglipDims& dims, int max_s, void* base, SiglipScratch* out);

// Load the FA4 modules and the static FP8 unit scale. Must not run while a
// CUDA graph is being captured. Idempotent. Returns 0 on success.
int siglip_load_kernels();

// cuBLASLt runner for the FP8/FP16 GEMMs; one per plugin instance.
GemmRunner* siglip_gemm_create();
void siglip_gemm_destroy(GemmRunner* gemm);

// images: [nv, 224, 224, 3], uint8 pixels mapped through w.lut, or fp16
// already in [-1, 1] (uint8_images = false). x: fp16 [nv*spv, D] out.
int siglip_patch_embed(const SiglipDims& dims, int nv, const SiglipEmbedWeights& w,
                       const SiglipScratch& s, GemmRunner* gemm, const void* images,
                       bool uint8_images, void* x, cudaStream_t stream);

// x: fp16 [nv*spv, D], updated in place. Returns 0 on success.
int siglip_layer_forward(const SiglipDims& dims, int nv, const SiglipLayerWeights& w,
                         const SiglipScratch& s, GemmRunner* gemm, void* x, cudaStream_t stream);

// x: fp16 [S, D] in; tokens: fp16 [S, De] out.
int siglip_post_project(const SiglipDims& dims, int S, const SiglipEmbedWeights& w,
                        const SiglipScratch& s, GemmRunner* gemm, const void* x, void* tokens,
                        cudaStream_t stream);

}  // namespace pi05
}  // namespace flashrt_trt
