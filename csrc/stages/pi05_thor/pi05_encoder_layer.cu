#include "pi05_encoder_layer.h"

#include "fa4_attention.h"

#include <cuda_fp16.h>

#include <cmath>

// FlashRT kernel entry points (vendored csrc).
#include "fused_fp4/pdl.cuh"
#include "fused_fp4/pi05_rowops_v2.cuh"
#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#include "kernels/rope_vec.cuh"
#include "quantize/reshape_scales_sfa.cuh"

// Defined in csrc/gemm/cutlass_sm100.cu without a header.
extern "C" int cutlass_fp8_sq(void* A, void* B, void* D, int M, int N, int K,
                              float alpha, float beta, cudaStream_t stream);

namespace flashrt_trt {
namespace pi05 {
namespace {

constexpr int kQkvWidth = 2560;
constexpr int64_t kAlign = 64;

int64_t align_up(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// Block-scale buffer for an NVFP4 activation operand [rows, K].
int64_t sfa_bytes(int64_t rows, int64_t K) {
    return flash_rt::fp4::sfa_size_bytes(static_cast<int>(rows), static_cast<int>(K), false);
}

}  // namespace

int64_t encoder_layer_scratch_bytes(const EncoderLayerDims& d, int max_se) {
    const int64_t S = max_se;
    const int64_t Q = static_cast<int64_t>(d.NH) * d.HD;
    int64_t total = 0;
    total += align_up(S * d.D);                    // x_fp8
    total += align_up(S * kQkvWidth * 2);          // qkv fp16
    total += align_up(S * Q * 2);                  // q fp16
    total += align_up(S * Q * 2);                  // attention out fp16
    total += align_up(S * (d.D / 2));              // O activation codes
    total += align_up(sfa_bytes(S, d.D));
    total += align_up(S * (d.D / 2));              // gate/up activation codes
    total += align_up(sfa_bytes(S, d.D));
    total += align_up(S * d.H);                    // dummy
    total += align_up(S * (d.H / 2));              // down activation codes
    total += align_up(sfa_bytes(S, d.H));
    return total;
}

void encoder_layer_bind_scratch(const EncoderLayerDims& d, int max_se,
                                void* base, EncoderLayerScratch* out) {
    const int64_t S = max_se;
    const int64_t Q = static_cast<int64_t>(d.NH) * d.HD;
    char* p = static_cast<char*>(base);
    auto take = [&p](int64_t bytes) {
        void* r = p;
        p += align_up(bytes);
        return r;
    };
    out->x_fp8 = take(S * d.D);
    out->qkv = take(S * kQkvWidth * 2);
    out->q = take(S * Q * 2);
    out->attn = take(S * Q * 2);
    out->at_packed = take(S * (d.D / 2));
    out->at_sfa = take(sfa_bytes(S, d.D));
    out->gu_packed = take(S * (d.D / 2));
    out->gu_sfa = take(sfa_bytes(S, d.D));
    out->dummy = take(S * d.H);
    out->dn_packed = take(S * (d.H / 2));
    out->dn_sfa = take(sfa_bytes(S, d.H));
}

int encoder_layer_load_kernels() { return fa4_load(); }

void encoder_layer_set_pdl(bool on) { flash_rt::fp4::pdl_flag() = on; }

int encoder_layer_forward(const EncoderLayerDims& d,
                          const EncoderLayerWeights& w,
                          const EncoderLayerScratch& s,
                          void* x, void* k_out, void* v_out,
                          cudaStream_t stream) {
    const int Se = d.Se;
    const int D = d.D;
    const int H = d.H;
    const int Q = d.NH * d.HD;
    auto* xh = static_cast<__half*>(x);
    int rc = 0;

    // 1-2. RMSNorm -> e4m3 with the calibrated static scale, FP8 QKV GEMM.
    if ((rc = flash_rt::fused_fp4::rowops_rms_fp8_v2(
             xh, s.x_fp8, Se, D, w.qkv_act_scale, stream)) != 0) {
        return 100 + rc;
    }
    if ((rc = cutlass_fp8_sq(s.x_fp8, const_cast<void*>(w.qkv_w), s.qkv, Se,
                             kQkvWidth, D, w.qkv_alpha, 0.0f, stream)) != 0) {
        return 200 + rc;
    }

    // 3-4. Split QKV, RoPE, write this layer's KV rows.
    if ((rc = qkv_split_rope_kvcache_fp16_vec(
             static_cast<const __half*>(s.qkv),
             static_cast<const __half*>(w.rope),
             static_cast<__half*>(s.q), static_cast<__half*>(k_out),
             static_cast<__half*>(v_out), Se, Q, d.HD, d.HD, kQkvWidth,
             0L, d.HD, stream)) != 0) {
        return 300 + rc;
    }
    if (d.last) {
        return 0;
    }

    // 5. FlashAttention-4 prefill, GQA with one KV head.
    if ((rc = fa4_hd256_gqa(s.q, k_out, v_out, s.attn, Se, Se, d.NH, fa4_default_scale(d.HD),
                            stream)) != 0) {
        return 500 + rc;
    }

    // Block-scale padding entries are never written by the quantizers and
    // must read as zero; the scratch may come from a shared workspace.
    if (cudaMemsetAsync(s.at_sfa, 0, sfa_bytes(Se, D), stream) != cudaSuccess ||
        cudaMemsetAsync(s.gu_sfa, 0, sfa_bytes(Se, D), stream) != cudaSuccess ||
        cudaMemsetAsync(s.dn_sfa, 0, sfa_bytes(Se, H), stream) != cudaSuccess) {
        return 550;
    }

    // 6. Quantize attention output, O projection accumulated into x.
    if ((rc =flash_rt::fused_fp4::rowops_quantize_fp4_sfa_v2(
             static_cast<const __half*>(s.attn), s.at_packed, s.at_sfa, Se, D,
             stream)) != 0) {
        return 600 + rc;
    }
    if ((rc = flash_rt::fp4::cutlass_fp4_gemm_variant(
             d.attn_o_variant, s.at_packed, s.at_sfa, w.o_packed, w.o_sfb, x,
             Se, D, D, 1.0f, 1.0f, stream)) != 0) {
        return 700 + rc;
    }

    // 7-9. RMSNorm x AWQ -> NVFP4, fused GeGLU into down's input, down
    // projection accumulated into x.
    if ((rc = flash_rt::fused_fp4::rowops_rms_mul_fp4_sfa_v2(
             xh, static_cast<const __half*>(w.awq_inv_s_gu), s.gu_packed,
             s.gu_sfa, Se, D, stream)) != 0) {
        return 800 + rc;
    }
    if ((rc = flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_nod(
             s.gu_packed, s.gu_sfa, w.gu_il_packed, w.gu_il_sfb, s.dummy,
             s.dn_packed, s.dn_sfa, Se, 2 * H, D, stream)) != 0) {
        return 900 + rc;
    }
    if ((rc = flash_rt::fp4::cutlass_fp4_gemm_variant(
             d.down_variant, s.dn_packed, s.dn_sfa, w.down_packed, w.down_sfb,
             x, Se, D, H, 1.0f, 1.0f, stream)) != 0) {
        return 1000 + rc;
    }
    return 0;
}

}  // namespace pi05
}  // namespace flashrt_trt
