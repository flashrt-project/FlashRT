#include "pi05_siglip.h"

#include "fa4_attention.h"

#include <cuda_fp16.h>

#include <cmath>
#include <exception>
#include <mutex>

// FlashRT kernel entry points (vendored csrc).
#include "fused_fp4/pi05_rowops_v2.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_siglip_ffn_variants_sm100.cuh"
#include "gemm/gemm_runner.h"
#include "kernels/decoder_fused.cuh"
#include "kernels/norm.cuh"
#include "kernels/patch_embed.cuh"
#include "kernels/quantize.cuh"
#include "quantize/reshape_scales_sfa.cuh"

namespace flashrt_trt {
namespace pi05 {
namespace {

using h16 = __half;

constexpr int64_t kAlign = 64;
constexpr int kPatchDim = 588;  // 14 * 14 * 3
constexpr float kLnEps = 1e-5f;
constexpr float kPostLnEps = 1e-6f;

int64_t align_up(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

int64_t sfa_bytes(int64_t rows, int64_t K) {
    return flash_rt::fp4::sfa_size_bytes(static_cast<int>(rows), static_cast<int>(K), false);
}

std::mutex g_mu;
bool g_loaded = false;
float* g_unit_scale = nullptr;  // device 1.0f: attention output is cast to e4m3 unscaled

}  // namespace

int64_t siglip_scratch_bytes(const SiglipDims& d, int max_s) {
    const int64_t S = max_s;
    int64_t t = 0;
    t += align_up(S * d.D);
    t += align_up(S * 3 * d.D * 2);
    t += align_up(S * d.D * 2);
    t += align_up(S * (d.D / 2));
    t += align_up(sfa_bytes(S, d.D));
    t += align_up(S * (d.H_pad / 2));
    t += align_up(sfa_bytes(S, d.H_pad));
    t += align_up(S * kPatchDim * 2);
    t += align_up(S * d.D * 2);
    t += align_up(S * d.D * 2);
    return t;
}

void siglip_bind_scratch(const SiglipDims& d, int max_s, void* base, SiglipScratch* out) {
    const int64_t S = max_s;
    char* p = static_cast<char*>(base);
    auto take = [&p](int64_t n) {
        void* r = p;
        p += align_up(n);
        return r;
    };
    out->x_fp8 = take(S * d.D);
    out->qkv = take(S * 3 * d.D * 2);
    out->attn = take(S * d.D * 2);
    out->ln_packed = take(S * (d.D / 2));
    out->ln_sfa = take(sfa_bytes(S, d.D));
    out->hid_packed = take(S * (d.H_pad / 2));
    out->hid_sfa = take(sfa_bytes(S, d.H_pad));
    out->patches = take(S * kPatchDim * 2);
    out->norm = take(S * d.D * 2);
    out->x = take(S * d.D * 2);
}

int siglip_load_kernels() {
    if (int rc = fa4_load()) {
        return rc;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_loaded) {
        return 0;
    }
    void* p = nullptr;
    const float one = 1.0f;
    if (cudaMalloc(&p, sizeof(float)) != cudaSuccess ||
        cudaMemcpy(p, &one, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }
    g_unit_scale = static_cast<float*>(p);
    g_loaded = true;
    return 0;
}

GemmRunner* siglip_gemm_create() {
    try {
        return new GemmRunner();
    } catch (const std::exception&) {
        return nullptr;
    }
}

void siglip_gemm_destroy(GemmRunner* gemm) { delete gemm; }

int siglip_patch_embed(const SiglipDims& d, int nv, const SiglipEmbedWeights& w,
                       const SiglipScratch& s, GemmRunner* gemm, const void* images,
                       bool uint8_images, void* x, cudaStream_t stream) {
    const int S = nv * d.spv;
    try {
        if (uint8_images) {
            patch_im2col_uint8(static_cast<const uint8_t*>(images), static_cast<const half*>(w.lut),
                               static_cast<half*>(s.patches), nv, stream);
        } else {
            patch_im2col(static_cast<const half*>(images), static_cast<half*>(s.patches), nv, stream);
        }
        gemm->fp16_nn(s.patches, const_cast<void*>(w.pe_w), x, S, d.D, kPatchDim, stream);
        patch_embed_bias_pos(static_cast<half*>(x), static_cast<const half*>(w.pe_b),
                             static_cast<const half*>(w.pos_emb), S, d.D, d.spv, stream);
    } catch (const std::exception&) {
        return 10;
    }
    return 0;
}

int siglip_layer_forward(const SiglipDims& d, int nv, const SiglipLayerWeights& w,
                         const SiglipScratch& s, GemmRunner* gemm, void* x, cudaStream_t stream) {
    if (!g_loaded) {
        return 1;
    }
    const int S = nv * d.spv;
    const int D = d.D;
    const auto* xh = static_cast<const h16*>(x);
    int rc = 0;
    try {
        // 1-2. LayerNorm -> e4m3, FP8 QKV GEMM with bias.
        if ((rc = flash_rt::fused_fp4::rowops_layer_norm_fp8_v2(
                 xh, static_cast<const h16*>(w.ln_attn_w), static_cast<const h16*>(w.ln_attn_b),
                 s.x_fp8, S, D, kLnEps, stream)) != 0) {
            return 100 + rc;
        }
        gemm->fp8_nn_bias(s.x_fp8, const_cast<void*>(w.qkv_w), s.qkv, const_cast<void*>(w.qkv_b), S,
                          3 * D, D, w.qkv_alpha, stream);

        // 3. FlashAttention-4 over each view; Q, K, V are views of the QKV rows.
        const int64_t row = 3 * static_cast<int64_t>(D);
        char* qkv = static_cast<char*>(s.qkv);
        if ((rc = fa4_hd72_mha(qkv, row, qkv + D * 2, row, qkv + 2 * D * 2, row, s.attn, nv, d.spv,
                               d.NH, fa4_default_scale(d.HD), stream)) != 0) {
            return 300 + rc;
        }

        // 4. Attention output to e4m3 (unit scale), O projection with bias into x.
        quantize_fp8_static_fp16(static_cast<const h16*>(s.attn),
                                 static_cast<__nv_fp8_e4m3*>(s.x_fp8), g_unit_scale, S * D, stream);
        gemm->fp8_nn_bias_res(s.x_fp8, const_cast<void*>(w.o_w), x, const_cast<void*>(w.o_b), S, D,
                              D, w.o_alpha, stream);

        // Block-scale padding entries are never written by the quantizers and
        // must read as zero; the scratch may come from a shared workspace.
        if (cudaMemsetAsync(s.ln_sfa, 0, sfa_bytes(S, D), stream) != cudaSuccess ||
            cudaMemsetAsync(s.hid_sfa, 0, sfa_bytes(S, d.H_pad), stream) != cudaSuccess) {
            return 450;
        }

        // 5-7. LayerNorm x AWQ -> NVFP4, up (bias + GELU, NVFP4 out),
        // down (bias + residual into x).
        if ((rc = flash_rt::fused_fp4::rowops_layer_norm_mul_fp4_sfa_v2(
                 xh, static_cast<const h16*>(w.ln_ffn_w), static_cast<const h16*>(w.ln_ffn_b),
                 static_cast<const h16*>(w.awq_inv_s), s.ln_packed, s.ln_sfa, S, D, kLnEps,
                 stream)) != 0) {
            return 500 + rc;
        }
        if ((rc = flash_rt::fp4::cutlass_fp4_gemm_bias_gelu_fp4out_v(
                 d.up_variant, s.ln_packed, s.ln_sfa, w.up_packed, w.up_sfb, w.up_b, s.hid_packed,
                 s.hid_sfa, S, d.H_pad, D, stream)) != 0) {
            return 600 + rc;
        }
        if ((rc = flash_rt::fp4::cutlass_fp4_gemm_bias_res_fp16(
                 s.hid_packed, s.hid_sfa, w.down_packed, w.down_sfb, w.down_b, x, x, S, D, d.H_pad,
                 stream)) != 0) {
            return 700 + rc;
        }
    } catch (const std::exception&) {
        return 900;
    }
    return 0;
}

int siglip_post_project(const SiglipDims& d, int S, const SiglipEmbedWeights& w,
                        const SiglipScratch& s, GemmRunner* gemm, const void* x, void* tokens,
                        cudaStream_t stream) {
    try {
        layer_norm_fp16(static_cast<const h16*>(x), static_cast<const h16*>(w.postln_w),
                        static_cast<const h16*>(w.postln_b), static_cast<h16*>(s.norm), S, d.D,
                        kPostLnEps, stream);
        gemm->fp16_nn(s.norm, const_cast<void*>(w.proj_w), tokens, S, d.De, d.D, stream);
        add_bias_fp16(static_cast<h16*>(tokens), static_cast<const h16*>(w.proj_b), S, d.De, stream);
    } catch (const std::exception&) {
        return 20;
    }
    return 0;
}

}  // namespace pi05
}  // namespace flashrt_trt

