// pi0.5 Thor (SM110) encoder layer on FlashRT kernels.
//
// Framework-free: raw device pointers in, raw device pointers out. The
// TensorRT plugin and the parity test both call this.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flashrt_trt {
namespace pi05 {

struct EncoderLayerDims {
    int Se = 0;              // prefix tokens (dynamic)
    int D = 2048;            // hidden size
    int H = 16384;           // FFN width
    int NH = 8;              // query heads
    int HD = 256;            // head dim (one KV head)
    int attn_o_variant = 1;  // NVFP4 GEMM variant for the attention O projection
    int down_variant = 8;    // NVFP4 GEMM variant for the FFN down projection
    bool last = false;       // last layer: QKV + KV write only
};

struct EncoderLayerWeights {
    const void* qkv_w = nullptr;           // e4m3 [2560, D]
    float qkv_alpha = 1.0f;                // weight descale (host)
    const float* qkv_act_scale = nullptr;  // device, 1 value
    const void* rope = nullptr;            // fp16 [Se, HD] (cos/sin halves)
    const void* o_packed = nullptr;        // NVFP4 [D, D/2]
    const void* o_sfb = nullptr;
    const void* awq_inv_s_gu = nullptr;    // fp16 [D]
    const void* gu_il_packed = nullptr;    // NVFP4 [2H, D/2], gate/up interleaved
    const void* gu_il_sfb = nullptr;
    const void* down_packed = nullptr;     // NVFP4 [D, H/2]
    const void* down_sfb = nullptr;
};

struct EncoderLayerScratch {
    void* x_fp8 = nullptr;     // [Se, D]
    void* qkv = nullptr;       // fp16 [Se, 2560]
    void* q = nullptr;         // fp16 [Se, NH*HD]
    void* attn = nullptr;      // fp16 [Se, NH*HD]
    void* at_packed = nullptr; // NVFP4 activations for O
    void* at_sfa = nullptr;
    void* gu_packed = nullptr; // NVFP4 activations for gate/up
    void* gu_sfa = nullptr;
    void* dummy = nullptr;     // [Se, H] never written, needs a real pointer
    void* dn_packed = nullptr; // NVFP4 activations for down
    void* dn_sfa = nullptr;
};

// Bytes of one contiguous scratch block for max_se tokens.
int64_t encoder_layer_scratch_bytes(const EncoderLayerDims& dims, int max_se);

// Carve the scratch block into aligned sub-buffers for max_se tokens.
void encoder_layer_bind_scratch(const EncoderLayerDims& dims, int max_se,
                                void* base, EncoderLayerScratch* out);

// Load the ahead-of-time FA4 module. Must not run while a CUDA graph is
// being captured. Idempotent. Returns 0 on success.
int encoder_layer_load_kernels();

// Enable programmatic dependent launch in the FlashRT GEMM kernels.
void encoder_layer_set_pdl(bool on);

// x: fp16 [Se, D], updated in place (residual stream).
// k_out, v_out: fp16 [Se, HD], this layer's KV cache rows.
// Returns 0 on success.
int encoder_layer_forward(const EncoderLayerDims& dims,
                          const EncoderLayerWeights& w,
                          const EncoderLayerScratch& s,
                          void* x, void* k_out, void* v_out,
                          cudaStream_t stream);

}  // namespace pi05
}  // namespace flashrt_trt
