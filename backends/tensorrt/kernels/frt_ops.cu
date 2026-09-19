// FlashRT edge kernels: the operator layer declared in frt_ops.h.
//
// Each call chains the FlashRT kernels in the order the production pipeline
// runs them, so an operator here and the corresponding slice of a pi0.5 stage
// produce the same bits.
#include "frt_ops.h"

#include <cuda_fp16.h>

#include <cmath>

#include "fused_fp4/pdl.cuh"
#include "fused_fp4/pi05_rowops_v2.cuh"
#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_siglip_ffn_variants_sm100.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "quantize/reshape_scales_sfa.cuh"

#include "stages/pi05_thor/fa4_attention.h"

namespace {

constexpr int64_t kAlign = 64;

int64_t align_up(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

int64_t sfa_bytes(int64_t rows, int64_t K) {
    return flash_rt::fp4::sfa_size_bytes(static_cast<int>(rows), static_cast<int>(K), false);
}

// Activation quantization shared by the linear and the MLP operator.
int32_t quantize_activation(const void* x, const void* gamma, const void* beta,
                            const void* awq_inv_s, void* packed, void* sfa,
                            int32_t M, int32_t K, int32_t norm_mode, float eps,
                            cudaStream_t stream) {
    const __half* xh = static_cast<const __half*>(x);
    switch (norm_mode) {
    case FRT_NORM_NONE:
        return flash_rt::fused_fp4::rowops_quantize_fp4_sfa_v2(xh, packed, sfa, M, K, stream);
    case FRT_NORM_RMS:
        return flash_rt::fused_fp4::rowops_rms_mul_fp4_sfa_v2(
            xh, static_cast<const __half*>(awq_inv_s), packed, sfa, M, K, stream);
    case FRT_NORM_LAYERNORM:
        return flash_rt::fused_fp4::rowops_layer_norm_mul_fp4_sfa_v2(
            xh, static_cast<const __half*>(gamma), static_cast<const __half*>(beta),
            static_cast<const __half*>(awq_inv_s), packed, sfa, M, K, eps, stream);
    default:
        return -1;
    }
}


// The SigLIP FFN kernels name only the first four of the seven tiles they
// build, so the last three are spelled out here; the shapes are the ones
// cutlass_fp4_gemm_siglip_ffn_variants_sm100.cu instantiates.
const char* kGateNames[] = {
    "up   128x256x256 (base)", "up   128x128x256", "up   128x128x128", "up   128x64x256",
    "up   256x128x128 cluster2x1x1 (2-SM UMMA)", "up   256x256x128 cluster2x1x1 (2-SM UMMA)",
    "up   256x128x256 cluster2x1x1 (2-SM UMMA)"};
const char* kDownNames[] = {
    "down 128x128x256 (base)", "down 128x64x256", "down 128x128x128", "down 128x256x256",
    "down 256x128x256 cluster2x1x1 (2-SM UMMA)", "down 256x256x256 cluster2x1x1 (2-SM UMMA)",
    "down 256x64x256 cluster2x1x1 (2-SM UMMA)"};
constexpr int32_t kSiglipVariants = 7;

}  // namespace

extern "C" {

size_t frt_nvfp4_linear_workspace(int32_t M, int32_t K) {
    return static_cast<size_t>(align_up(static_cast<int64_t>(M) * (K / 2)) +
                               align_up(sfa_bytes(M, K)));
}

int32_t frt_nvfp4_linear(const void* x, const void* norm_gamma, const void* norm_beta,
                         const void* awq_inv_s, const void* w_packed, const void* w_sfb,
                         const void* bias, const void* residual, void* out,
                         int32_t M, int32_t N, int32_t K, int32_t norm_mode,
                         int32_t epilogue, int32_t variant, float eps,
                         void* workspace, cudaStream_t stream) {
    char* ws = static_cast<char*>(workspace);
    void* packed = ws;
    void* sfa = ws + align_up(static_cast<int64_t>(M) * (K / 2));

    int rc = quantize_activation(x, norm_gamma, norm_beta, awq_inv_s, packed, sfa, M, K,
                                 norm_mode, eps, stream);
    if (rc != 0) {
        return 100 + rc;
    }

    if (epilogue == FRT_EPI_BIAS_RES) {
        rc = flash_rt::fp4::cutlass_fp4_gemm_bias_res_fp16_v(variant, packed, sfa, w_packed,
                                                             w_sfb, bias, residual, out,
                                                             M, N, K, stream);
    } else {
        const float beta = (epilogue == FRT_EPI_ACCUM) ? 1.0f : 0.0f;
        rc = flash_rt::fp4::cutlass_fp4_gemm_variant(variant, packed, sfa, w_packed, w_sfb,
                                                     out, M, N, K, 1.0f, beta, stream);
    }
    return rc == 0 ? 0 : 200 + rc;
}

size_t frt_nvfp4_mlp_workspace(int32_t M, int32_t D, int32_t H, int32_t gate_mode) {
    const int64_t m = M;
    int64_t total = align_up(m * (D / 2)) + align_up(sfa_bytes(m, D));  // gate/up input
    total += align_up(m * (H / 2)) + align_up(sfa_bytes(m, H));         // down input
    if (gate_mode == FRT_GATE_GEGLU_IL) {
        total += align_up(m * H);  // the interleaved kernel's unwritten D operand
    }
    return static_cast<size_t>(total);
}

int32_t frt_nvfp4_mlp(const void* x, const void* norm_gamma, const void* norm_beta,
                      const void* awq_inv_s, const void* gate_up_packed,
                      const void* gate_up_sfb, const void* gate_up_bias,
                      const void* down_packed, const void* down_sfb, const void* down_bias,
                      const void* residual, void* out, int32_t M, int32_t D, int32_t H,
                      int32_t norm_mode, int32_t gate_mode, int32_t gate_variant,
                      int32_t down_variant, int32_t epilogue, float eps,
                      void* workspace, cudaStream_t stream) {
    const int64_t m = M;
    char* ws = static_cast<char*>(workspace);
    void* gu_packed = ws;
    ws += align_up(m * (D / 2));
    void* gu_sfa = ws;
    ws += align_up(sfa_bytes(m, D));
    void* dn_packed = ws;
    ws += align_up(m * (H / 2));
    void* dn_sfa = ws;
    ws += align_up(sfa_bytes(m, H));
    void* dummy = ws;  // only the interleaved kernel needs it

    int rc = quantize_activation(x, norm_gamma, norm_beta, awq_inv_s, gu_packed, gu_sfa, M, D,
                                 norm_mode, eps, stream);
    if (rc != 0) {
        return 100 + rc;
    }

    if (gate_mode == FRT_GATE_GEGLU_IL) {
        rc = flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_nod(gu_packed, gu_sfa, gate_up_packed,
                                                             gate_up_sfb, dummy, dn_packed,
                                                             dn_sfa, M, 2 * H, D, stream);
    } else {
        rc = flash_rt::fp4::cutlass_fp4_gemm_bias_gelu_fp4out_v(gate_variant, gu_packed, gu_sfa,
                                                                gate_up_packed, gate_up_sfb,
                                                                gate_up_bias, dn_packed, dn_sfa,
                                                                M, H, D, stream);
    }
    if (rc != 0) {
        return 300 + rc;
    }

    if (epilogue == FRT_EPI_BIAS_RES) {
        rc = flash_rt::fp4::cutlass_fp4_gemm_bias_res_fp16_v(down_variant, dn_packed, dn_sfa,
                                                             down_packed, down_sfb, down_bias,
                                                             residual, out, M, D, H, stream);
    } else {
        const float beta = (epilogue == FRT_EPI_ACCUM) ? 1.0f : 0.0f;
        rc = flash_rt::fp4::cutlass_fp4_gemm_variant(down_variant, dn_packed, dn_sfa, down_packed,
                                                     down_sfb, out, M, D, H, 1.0f, beta, stream);
    }
    return rc == 0 ? 0 : 400 + rc;
}

int32_t frt_fa4_load(void) { return flashrt_trt::fa4_load(); }

float frt_fa4_default_scale(int32_t head_dim) { return flashrt_trt::fa4_default_scale(head_dim); }

int32_t frt_fa4_gqa_hd256(const void* q, const void* k, const void* v, void* o, int32_t sq,
                          int32_t sk, int32_t nh, float scale, cudaStream_t stream) {
    return flashrt_trt::fa4_hd256_gqa(const_cast<void*>(q), const_cast<void*>(k),
                                      const_cast<void*>(v), o, sq, sk, nh, scale, stream);
}

int32_t frt_fa4_mha_hd72(const void* q, int64_t q_row_stride, const void* k, int64_t k_row_stride,
                         const void* v, int64_t v_row_stride, void* o, int32_t b, int32_t s,
                         int32_t nh, float scale, cudaStream_t stream) {
    return flashrt_trt::fa4_hd72_mha(const_cast<void*>(q), q_row_stride, const_cast<void*>(k),
                                     k_row_stride, const_cast<void*>(v), v_row_stride, o, b, s,
                                     nh, scale, stream);
}

size_t frt_nvfp4_weight_sfb_bytes(int32_t N, int32_t K) {
    return static_cast<size_t>(flash_rt::fp4::sfa_size_bytes(N, K, true));
}

int32_t frt_nvfp4_pack_weight(const void* w_fp16, void* packed, void* sfb, int32_t N, int32_t K,
                              cudaStream_t stream) {
    return flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(w_fp16, packed, sfb, N, K, true, stream);
}

int32_t frt_nvfp4_num_variants(int32_t kind) {
    switch (kind) {
    case FRT_VARIANT_GEMM: return flash_rt::fp4::cutlass_fp4_gemm_num_variants();
    case FRT_VARIANT_GATE_BIAS_GELU:
    case FRT_VARIANT_DOWN_BIAS_RES: return kSiglipVariants;
    default: return 0;
    }
}

const char* frt_nvfp4_variant_name(int32_t kind, int32_t idx) {
    if (idx < 0 || idx >= frt_nvfp4_num_variants(kind)) {
        return "<invalid>";
    }
    switch (kind) {
    case FRT_VARIANT_GEMM: return flash_rt::fp4::cutlass_fp4_gemm_variant_name(idx);
    case FRT_VARIANT_GATE_BIAS_GELU: return kGateNames[idx];
    default: return kDownNames[idx];
    }
}

void frt_set_pdl(int32_t on) { flash_rt::fp4::pdl_flag() = (on != 0); }

}  // extern "C"
