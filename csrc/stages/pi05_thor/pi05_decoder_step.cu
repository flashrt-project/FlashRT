#include "pi05_decoder_step.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "fused_fp4/norm_silu_fp4_sfa.cuh"
#include "fused_fp4/pi05_rowops_v2.cuh"
#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#include "kernels/attention_cublas.cuh"
#include "kernels/decoder_fused.cuh"
#include "kernels/rope_vec.cuh"
#include "quantize/reshape_scales_sfa.cuh"

namespace flashrt_trt {
namespace pi05 {
namespace {

constexpr int kQkvWidth = 2560;
constexpr int kActionDim = 32;
constexpr int64_t kAlign = 64;

int64_t align_up(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
int64_t sfa_bytes(int64_t rows, int64_t K) {
    return flash_rt::fp4::sfa_size_bytes(static_cast<int>(rows), static_cast<int>(K), false);
}
int64_t block_bytes(int64_t total, int layers) { return total / layers; }

}  // namespace

int64_t decoder_scratch_bytes(const DecoderDims& d, int max_total_keys) {
    const int64_t S = d.S, A = static_cast<int64_t>(d.NH) * d.HD;
    int64_t t = 0;
    t += 4 * align_up(S * d.D * 2);                         // x, xn, gate, fg
    t += align_up(S * kQkvWidth * 2);                       // qkv
    t += align_up(S * A * 2);                               // attention out
    t += align_up(S * d.NH * static_cast<int64_t>(max_total_keys) * 2);  // logits
    t += align_up(S * kActionDim * 4);                      // action delta fp32
    t += align_up(S * (d.D / 2)) + align_up(sfa_bytes(S, d.D));
    t += align_up(S * (A / 2)) + align_up(sfa_bytes(S, A));
    t += align_up(S * (d.H / 2)) + align_up(sfa_bytes(S, d.H));
    t += align_up(S * 2 * d.H);                             // GeGLU D placeholder
    return t;
}

void decoder_bind_scratch(const DecoderDims& d, int max_total_keys, void* base,
                          DecoderScratch* o) {
    const int64_t S = d.S, A = static_cast<int64_t>(d.NH) * d.HD;
    char* p = static_cast<char*>(base);
    auto take = [&p](int64_t bytes) { void* r = p; p += align_up(bytes); return r; };
    o->x = take(S * d.D * 2); o->xn = take(S * d.D * 2);
    o->gate = take(S * d.D * 2); o->fg = take(S * d.D * 2);
    o->qkv = take(S * kQkvWidth * 2);
    o->attn = take(S * A * 2);
    o->logits = take(S * d.NH * static_cast<int64_t>(max_total_keys) * 2);
    o->action_f32 = take(S * kActionDim * 4);
    o->xn_packed = take(S * (d.D / 2)); o->xn_sfa = take(sfa_bytes(S, d.D));
    o->ctx_packed = take(S * (A / 2)); o->ctx_sfa = take(sfa_bytes(S, A));
    o->hid_packed = take(S * (d.H / 2)); o->hid_sfa = take(sfa_bytes(S, d.H));
    o->gu_dummy = take(S * 2 * d.H);
}

int decoder_step_forward(const DecoderDims& d, const DecoderWeights& w, const DecoderScratch& s,
                         cublasHandle_t cublas, void* noise, void* kv_k, void* kv_v,
                         cudaStream_t stream) {
    using flash_rt::fp4::cutlass_fp4_gemm_variant;
    using flash_rt::fused_fp4::pi05_adarms_fp4_sfa_native_fp16;
    using flash_rt::fused_fp4::pi05_gate_res_adarms_fp4_sfa_native_fp16;
    using h16 = __half;

    const int S = d.S, D = d.D, H = d.H, NH = d.NH, HD = d.HD, L = d.L;
    const int A = NH * HD;
    const int enc_seq = d.total_keys - S;
    const int64_t style = static_cast<int64_t>(S) * 3 * D * 2;  // bytes per (layer) style block
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    if (enc_seq <= 0) return 1;
    if (cublasSetStream(cublas, stream) != CUBLAS_STATUS_SUCCESS) return 2;

    // NVFP4 scale buffers: layout padding must read as zero.
    if (cudaMemsetAsync(s.xn_sfa, 0, sfa_bytes(S, D), stream) != cudaSuccess ||
        cudaMemsetAsync(s.ctx_sfa, 0, sfa_bytes(S, A), stream) != cudaSuccess ||
        cudaMemsetAsync(s.hid_sfa, 0, sfa_bytes(S, H), stream) != cudaSuccess) {
        return 3;
    }

    const auto* ain_w = static_cast<const h16*>(w.ain_w);
    auto* x = static_cast<h16*>(s.x);
    auto* gate = static_cast<h16*>(s.gate);
    auto* fg = static_cast<h16*>(s.fg);
    auto* xn_p = static_cast<uint8_t*>(s.xn_packed);
    auto* xn_s = static_cast<uint8_t*>(s.xn_sfa);
    const char* sa = static_cast<const char*>(w.sa);
    const char* sf = static_cast<const char*>(w.sf);

    // Per-layer block sizes of the concatenated weight blobs.
    const int64_t qw_p = static_cast<int64_t>(kQkvWidth) * (D / 2);
    const int64_t ow_p = static_cast<int64_t>(D) * (A / 2);
    const int64_t gw_p = static_cast<int64_t>(2 * H) * (D / 2);
    const int64_t dw_p = static_cast<int64_t>(D) * (H / 2);
    const int64_t qw_s = flash_rt::fp4::sfa_size_bytes(kQkvWidth, D, true);
    const int64_t ow_s = flash_rt::fp4::sfa_size_bytes(D, A, true);
    const int64_t gw_s = flash_rt::fp4::sfa_size_bytes(2 * H, D, true);
    const int64_t dw_s = flash_rt::fp4::sfa_size_bytes(D, H, true);
    auto at = [](const void* base, int64_t block, int l) {
        return static_cast<const void*>(static_cast<const char*>(base) + block * l);
    };

    gmm_fp16(cublas, static_cast<const h16*>(noise), ain_w, x, S, D, kActionDim, 0.0f, stream);
    add_bias_fp16(x, static_cast<const h16*>(w.ain_b), S, D, stream);

    int rc = 0;
    for (int l = 0; l < L; ++l) {
        const auto* sa_l = reinterpret_cast<const h16*>(sa + style * l);
        const auto* sf_l = reinterpret_cast<const h16*>(sf + style * l);
        if (l == 0) {
            pi05_adarms_fp4_sfa_native_fp16(x, sa_l, xn_p, xn_s, gate, S, D, stream);
        }
        if ((rc = cutlass_fp4_gemm_variant(d.v_qkv, xn_p, xn_s, at(w.qw_fp4, qw_p, l),
                                           at(w.qw_sfb, qw_s, l), s.qkv, S, kQkvWidth, D, 1.0f,
                                           0.0f, stream)) != 0) {
            return 100 + l;
        }
        const long kv_off = static_cast<long>(l) * d.total_keys * HD + static_cast<long>(enc_seq) * HD;
        if ((rc = qkv_split_rope_kvcache_fp16_vec(
                 static_cast<const h16*>(s.qkv), static_cast<const h16*>(w.rope),
                 static_cast<h16*>(s.attn), static_cast<h16*>(kv_k), static_cast<h16*>(kv_v), S, A, HD,
                 HD, kQkvWidth, kv_off, HD, stream)) != 0) {
            return 200 + l;
        }
        const int64_t layer_kv = static_cast<int64_t>(l) * d.total_keys * HD * 2;
        attention_qkv_fp16(cublas, static_cast<const h16*>(s.attn),
                           static_cast<const h16*>(static_cast<void*>(static_cast<char*>(kv_k) + layer_kv)),
                           static_cast<const h16*>(static_cast<void*>(static_cast<char*>(kv_v) + layer_kv)),
                           static_cast<h16*>(s.logits), static_cast<h16*>(s.attn), S, d.total_keys, NH, HD,
                           scale, stream);
        if ((rc = flash_rt::fused_fp4::rowops_quantize_fp4_sfa_v2(
                 static_cast<const h16*>(s.attn), s.ctx_packed, s.ctx_sfa, S, A, stream)) != 0) {
            return 300 + l;
        }
        if ((rc = cutlass_fp4_gemm_variant(d.v_o, s.ctx_packed, s.ctx_sfa, at(w.ow_fp4, ow_p, l),
                                           at(w.ow_sfb, ow_s, l), fg, S, D, A, 1.0f, 0.0f, stream)) != 0) {
            return 400 + l;
        }
        pi05_gate_res_adarms_fp4_sfa_native_fp16(fg, gate, x, sf_l, xn_p, xn_s, gate, S, D, stream);
        if ((rc = flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_v10(
                 xn_p, xn_s, at(w.gwil_fp4, gw_p, l), at(w.gwil_sfb, gw_s, l), s.gu_dummy,
                 s.hid_packed, s.hid_sfa, S, 2 * H, D, stream)) != 0) {
            return 500 + l;
        }
        if ((rc = cutlass_fp4_gemm_variant(d.v_down, s.hid_packed, s.hid_sfa, at(w.dw_fp4, dw_p, l),
                                           at(w.dw_sfb, dw_s, l), fg, S, D, H, 1.0f, 0.0f, stream)) != 0) {
            return 600 + l;
        }
        if (l < L - 1) {
            const auto* sa_next = reinterpret_cast<const h16*>(sa + style * (l + 1));
            pi05_gate_res_adarms_fp4_sfa_native_fp16(fg, gate, x, sa_next, xn_p, xn_s, gate, S, D, stream);
        } else {
            gate_res_fp16(fg, gate, x, S * D, stream);
        }
    }
    adarms_fp16(x, static_cast<const h16*>(w.fs), static_cast<h16*>(s.xn), gate, S, D, stream);
    gmm_fp16_out_fp32(cublas, static_cast<const h16*>(s.xn), static_cast<const h16*>(w.aow),
                      static_cast<float*>(s.action_f32), S, kActionDim, D, stream);
    action_update_from_fp32(static_cast<const float*>(s.action_f32), static_cast<const h16*>(w.aob),
                            static_cast<h16*>(noise), S, kActionDim, d.dt, true, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : 900;
}

}  // namespace pi05
}  // namespace flashrt_trt
