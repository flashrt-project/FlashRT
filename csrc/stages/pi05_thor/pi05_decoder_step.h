// pi0.5 action-expert denoise step on Thor (SM110), FlashRT FP4 profile.
//
// One call runs the action input projection, all decoder layers, the final
// AdaRMSNorm and the flow-matching action update. Per layer:
//   (layer 0) AdaRMSNorm -> NVFP4, NVFP4 QKV GEMM, QKV split + RoPE + suffix
//   K/V write into the shared cache, cuBLAS attention over prefix + suffix,
//   NVFP4 quantize, NVFP4 O GEMM, gated residual + AdaRMSNorm -> NVFP4,
//   interleaved GeGLU GEMM writing the Down input, NVFP4 Down GEMM, gated
//   residual + next layer's AdaRMSNorm -> NVFP4 (plain gated residual after
//   the last layer).
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flashrt_trt {
namespace pi05 {

struct DecoderDims {
    int S = 10;         // action tokens
    int D = 1024;       // hidden
    int H = 4096;       // FFN width
    int NH = 8;
    int HD = 256;
    int L = 18;
    int total_keys = 0; // prefix + S (dynamic)
    int v_qkv = 28, v_o = 28, v_gu = 10, v_down = 28;
    float dt = -0.1f;
};

// Device pointers. Per-layer blobs are concatenated over layers; every layer
// block has the same size. Style tables are this step's slices.
struct DecoderWeights {
    const void* ain_w = nullptr, *ain_b = nullptr;
    const void* aow = nullptr, *aob = nullptr;
    const void* rope = nullptr;       // fp16 [S, 256]
    const void* sa = nullptr;         // fp16 [L * S * 3D] attention styles
    const void* sf = nullptr;         // fp16 [L * S * 3D] FFN styles
    const void* fs = nullptr;         // fp16 [S * 3D] final style
    const void* qw_fp4 = nullptr, *qw_sfb = nullptr;
    const void* ow_fp4 = nullptr, *ow_sfb = nullptr;
    const void* gwil_fp4 = nullptr, *gwil_sfb = nullptr;
    const void* dw_fp4 = nullptr, *dw_sfb = nullptr;
};

struct DecoderScratch {
    void* x = nullptr, *xn = nullptr, *gate = nullptr, *fg = nullptr;
    void* qkv = nullptr, *attn = nullptr, *logits = nullptr, *action_f32 = nullptr;
    void* xn_packed = nullptr, *xn_sfa = nullptr;
    void* ctx_packed = nullptr, *ctx_sfa = nullptr;
    void* hid_packed = nullptr, *hid_sfa = nullptr;
    void* gu_dummy = nullptr;
};

int64_t decoder_scratch_bytes(const DecoderDims& d, int max_total_keys);
void decoder_bind_scratch(const DecoderDims& d, int max_total_keys, void* base,
                          DecoderScratch* out);

// noise: fp16 [S, 32], updated in place.
// kv_k / kv_v: fp16 [L, total_keys, HD]; prefix rows are read, suffix rows
// [total_keys - S, total_keys) are written.
int decoder_step_forward(const DecoderDims& d, const DecoderWeights& w,
                         const DecoderScratch& s, cublasHandle_t cublas,
                         void* noise, void* kv_k, void* kv_v, cudaStream_t stream);

}  // namespace pi05
}  // namespace flashrt_trt
