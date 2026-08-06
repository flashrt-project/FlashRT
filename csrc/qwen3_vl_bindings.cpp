// ============================================================================
//  FlashRT — pybind module for Qwen3-VL kernels.
//
//  Built as a SEPARATE .so (flash_rt_qwen3_vl_kernels) from
//  flash_rt_kernels.so, gated by the FLASHRT_BUILD_QWEN3_VL CMake option, so
//  the shared production kernel binary is never rebuilt for this model. Same
//  pattern as flash_rt_fa2 / fmha_fp16_strided.
//
//  Python-side usage:
//
//      import flash_rt.flash_rt_kernels        as fvk     # unchanged
//      import flash_rt.flash_rt_qwen3_vl_kernels as vlk   # additive
//      vlk.rope_neox_qk_bf16(...)
//
//  Kernels here are general (rotate_half RoPE, etc.); they live in this
//  module only to keep the shared binary stable, and may be promoted to
//  flash_rt_kernels if another model needs them.
// ============================================================================

#include <pybind11/pybind11.h>

#include <cstdint>

#include <cuda_bf16.h>
#ifdef ENABLE_QWEN3_VL_FP8_ACT
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>

// SM89 Qwen3-VL FP8 kernels (block-128 GEMM/GEMV + fused act/norm quant +
// fused QK norm-rope-kvwrite). Bound here so the SM89 path imports them from
// this dedicated module, just like the SM120 ViT helpers below, instead of
// bloating the central flash_rt_kernels bindings.
#if defined(ENABLE_SM89_BLOCK_FP8_GEMM) || defined(ENABLE_QWEN3_VL_BF16_CUBLASLT)
#include "kernels/bf16_matmul_bf16.cuh"
#endif

#ifdef ENABLE_QWEN3_VL_BF16_GEMV_M1
#include "kernels/qwen3_vl_bf16_gemv_m1.cuh"
#include "kernels/qwen3_vl_w8_gemv_m1.cuh"
#include "kernels/qwen3_vl_w4_gemv_m1.cuh"
#endif

// INT8/INT4 decode GEMVs + INT8 KV cache: Orin only (see CMakeLists.txt).
#ifdef ENABLE_QWEN3_VL_INT_DECODE
#include "kernels/qwen3_vl_int8_gemv_m1.cuh"
#include "kernels/qwen3_vl_int4_gemv_m1.cuh"
#include "kernels/qwen3_int8_kv.cuh"
#endif

#ifdef ENABLE_SM89_BLOCK_FP8_GEMM
#include "gemm/fp8_block128_gemm_mma_sm89.cuh"
#include "gemm/fp8_gemv_m1_sm89.cuh"
#include "quantize/fp8_per_token_block_quant.cuh"
#endif

// Fused QK norm-rope-kvwrite. Compiled on every arch that builds
// qwen3_qkv_post_proc.cu: SM89 for the FP8 path, SM110 (Thor) for BF16 prefill.
#ifdef ENABLE_QWEN3_VL_QKV_POSTPROC
#include "kernels/qwen3_qkv_post_proc.cuh"
#endif

namespace py = pybind11;

namespace flash_rt {
namespace kernels {
void rope_neox_qk_bf16(
    const __nv_bfloat16* q_in, const __nv_bfloat16* k_in,
    const __nv_bfloat16* cos_tab, const __nv_bfloat16* sin_tab,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out,
    int rows, int q_heads, int k_heads, int head_dim, cudaStream_t stream);

#ifdef ENABLE_QWEN3_VL_BF16_GEMV_M1
void qwen3_vl_bf16_gemv_m1(
    const __nv_bfloat16* x, const __nv_bfloat16* W, __nv_bfloat16* out,
    int N, int K, cudaStream_t stream);
#endif

#ifdef ENABLE_QWEN3_VL_FP8_ACT
void layer_norm_to_fp8_block128_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, float eps, cudaStream_t stream);

void gelu_tanh_to_fp8_block128_bf16(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, cudaStream_t stream);

void gelu_tanh_bias_to_fp8_block128_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias, __nv_fp8_e4m3* out,
    float* scale, int rows, int dim, cudaStream_t stream);
#endif

void residual_add_bias_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x,
    const __nv_bfloat16* bias, int rows, int dim, cudaStream_t stream);

void qkv_split_bias_bf16(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias, __nv_bfloat16* q,
    __nv_bfloat16* k, __nv_bfloat16* v, int rows, int hq, int hk, int hv,
    cudaStream_t stream);
}  // namespace kernels
}  // namespace flash_rt

static cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

template <typename T>
static T* as_ptr(uintptr_t p) {
    return reinterpret_cast<T*>(p);
}

#if defined(ENABLE_SM89_BLOCK_FP8_GEMM) || defined(ENABLE_QWEN3_VL_QKV_POSTPROC)
static void* to_ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }
#endif

PYBIND11_MODULE(flash_rt_qwen3_vl_kernels, m) {
    m.doc() = "FlashRT Qwen3-VL kernels (separate module; additive).";

    m.def(
        "rope_neox_qk_bf16",
        [](uintptr_t q_in, uintptr_t k_in, uintptr_t cos, uintptr_t sin,
           uintptr_t q_out, uintptr_t k_out, int rows, int q_heads,
           int k_heads, int head_dim, uintptr_t stream) {
            flash_rt::kernels::rope_neox_qk_bf16(
                as_ptr<const __nv_bfloat16>(q_in),
                as_ptr<const __nv_bfloat16>(k_in),
                as_ptr<const __nv_bfloat16>(cos),
                as_ptr<const __nv_bfloat16>(sin),
                as_ptr<__nv_bfloat16>(q_out), as_ptr<__nv_bfloat16>(k_out),
                rows, q_heads, k_heads, head_dim, to_stream(stream));
        },
        py::arg("q_in"), py::arg("k_in"), py::arg("cos"), py::arg("sin"),
        py::arg("q_out"), py::arg("k_out"), py::arg("rows"),
        py::arg("q_heads"), py::arg("k_heads"), py::arg("head_dim"),
        py::arg("stream") = 0);

#ifdef ENABLE_QWEN3_VL_FP8_ACT
    m.def(
        "layer_norm_to_fp8_block128_bf16",
        [](uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t out,
           uintptr_t scale, int rows, int dim, float eps, uintptr_t stream) {
            flash_rt::kernels::layer_norm_to_fp8_block128_bf16(
                as_ptr<const __nv_bfloat16>(x),
                as_ptr<const __nv_bfloat16>(gamma),
                as_ptr<const __nv_bfloat16>(beta),
                as_ptr<__nv_fp8_e4m3>(out), as_ptr<float>(scale),
                rows, dim, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("out"),
        py::arg("scale"), py::arg("rows"), py::arg("dim"), py::arg("eps"),
        py::arg("stream") = 0);

    m.def(
        "gelu_tanh_to_fp8_block128_bf16",
        [](uintptr_t x, uintptr_t out, uintptr_t scale, int rows, int dim,
           uintptr_t stream) {
            flash_rt::kernels::gelu_tanh_to_fp8_block128_bf16(
                as_ptr<const __nv_bfloat16>(x), as_ptr<__nv_fp8_e4m3>(out),
                as_ptr<float>(scale), rows, dim, to_stream(stream));
        },
        py::arg("x"), py::arg("out"), py::arg("scale"), py::arg("rows"),
        py::arg("dim"), py::arg("stream") = 0);

    m.def(
        "gelu_tanh_bias_to_fp8_block128_bf16",
        [](uintptr_t x, uintptr_t bias, uintptr_t out, uintptr_t scale,
           int rows, int dim, uintptr_t stream) {
            flash_rt::kernels::gelu_tanh_bias_to_fp8_block128_bf16(
                as_ptr<const __nv_bfloat16>(x),
                as_ptr<const __nv_bfloat16>(bias), as_ptr<__nv_fp8_e4m3>(out),
                as_ptr<float>(scale), rows, dim, to_stream(stream));
        },
        py::arg("x"), py::arg("bias"), py::arg("out"), py::arg("scale"),
        py::arg("rows"), py::arg("dim"), py::arg("stream") = 0);
#endif

    m.def(
        "residual_add_bias_bf16",
        [](uintptr_t residual, uintptr_t x, uintptr_t bias, int rows, int dim,
           uintptr_t stream) {
            flash_rt::kernels::residual_add_bias_bf16(
                as_ptr<__nv_bfloat16>(residual),
                as_ptr<const __nv_bfloat16>(x),
                as_ptr<const __nv_bfloat16>(bias), rows, dim,
                to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("bias"), py::arg("rows"),
        py::arg("dim"), py::arg("stream") = 0);

    m.def(
        "qkv_split_bias_bf16",
        [](uintptr_t qkv, uintptr_t bias, uintptr_t q, uintptr_t k,
           uintptr_t v, int rows, int hq, int hk, int hv, uintptr_t stream) {
            flash_rt::kernels::qkv_split_bias_bf16(
                as_ptr<const __nv_bfloat16>(qkv),
                as_ptr<const __nv_bfloat16>(bias), as_ptr<__nv_bfloat16>(q),
                as_ptr<__nv_bfloat16>(k), as_ptr<__nv_bfloat16>(v),
                rows, hq, hk, hv, to_stream(stream));
        },
        py::arg("qkv"), py::arg("bias"), py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("rows"), py::arg("hq"), py::arg("hk"),
        py::arg("hv"), py::arg("stream") = 0);

#ifdef ENABLE_SM89_BLOCK_FP8_GEMM
    // ---- SM89 Qwen3-VL FP8 kernels (additive; Ada has no TMA so these are
    // hand-written, see docs/qwen3_vl_fp8_sm89.md) ----
    m.def("rms_norm_to_fp8_block128_bf16",
        [](uintptr_t input, uintptr_t weight, uintptr_t output_fp8,
           uintptr_t output_scale, int M, int K, float eps,
           uintptr_t stream) {
            flash_rt::quantize::rms_norm_to_fp8_block128_bf16(
                to_ptr(input), to_ptr(weight), to_ptr(output_fp8),
                reinterpret_cast<float*>(output_scale),
                M, K, eps, to_stream(stream));
        },
        py::arg("input"), py::arg("weight"), py::arg("output_fp8"),
        py::arg("output_scale"), py::arg("M"), py::arg("K"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("residual_add_rms_norm_to_fp8_block128_bf16",
        [](uintptr_t residual, uintptr_t x, uintptr_t residual_out,
           uintptr_t weight, uintptr_t output_fp8, uintptr_t output_scale,
           int M, int K, float eps, uintptr_t stream) {
            flash_rt::quantize::residual_add_rms_norm_to_fp8_block128_bf16(
                to_ptr(residual), to_ptr(x), to_ptr(residual_out),
                to_ptr(weight), to_ptr(output_fp8),
                reinterpret_cast<float*>(output_scale),
                M, K, eps, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("residual_out"),
        py::arg("weight"), py::arg("output_fp8"), py::arg("output_scale"),
        py::arg("M"), py::arg("K"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    m.def("silu_mul_to_fp8_block128_bf16",
        [](uintptr_t gate, uintptr_t up, uintptr_t output_fp8,
           uintptr_t output_scale, int M, int K, uintptr_t stream) {
            flash_rt::quantize::silu_mul_to_fp8_block128_bf16(
                to_ptr(gate), to_ptr(up), to_ptr(output_fp8),
                reinterpret_cast<float*>(output_scale),
                M, K, to_stream(stream));
        },
        py::arg("gate"), py::arg("up"), py::arg("output_fp8"),
        py::arg("output_scale"), py::arg("M"), py::arg("K"),
        py::arg("stream") = 0);

    m.def("silu_mul_merged_to_fp8_block128_bf16",
        [](uintptr_t gate_up, uintptr_t output_fp8,
           uintptr_t output_scale, int M, int K, uintptr_t stream) {
            flash_rt::quantize::silu_mul_merged_to_fp8_block128_bf16(
                to_ptr(gate_up), to_ptr(output_fp8),
                reinterpret_cast<float*>(output_scale),
                M, K, to_stream(stream));
        },
        py::arg("gate_up"), py::arg("output_fp8"),
        py::arg("output_scale"), py::arg("M"), py::arg("K"),
        py::arg("stream") = 0);

    // BF16-output norm variants: skip the FP8 quant pass, emit BF16 activations
    // for the bf16in GEMV path (qkv / gate_up in decode).
    m.def("rms_norm_bf16_out",
        [](uintptr_t input, uintptr_t weight, uintptr_t output,
           int M, int K, float eps, uintptr_t stream) {
            flash_rt::quantize::rms_norm_bf16_out(
                to_ptr(input), to_ptr(weight), to_ptr(output),
                M, K, eps, to_stream(stream));
        },
        py::arg("input"), py::arg("weight"), py::arg("output"),
        py::arg("M"), py::arg("K"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    m.def("residual_add_rms_norm_bf16_out",
        [](uintptr_t residual, uintptr_t x, uintptr_t residual_out,
           uintptr_t weight, uintptr_t output,
           int M, int K, float eps, uintptr_t stream) {
            flash_rt::quantize::residual_add_rms_norm_bf16_out(
                to_ptr(residual), to_ptr(x), to_ptr(residual_out),
                to_ptr(weight), to_ptr(output),
                M, K, eps, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("residual_out"),
        py::arg("weight"), py::arg("output"),
        py::arg("M"), py::arg("K"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    m.def("fp8_block128_gemm_blockscaled_sm89_bf16out",
        [](uintptr_t A, uintptr_t B, uintptr_t D,
           int M, int N, int K,
           uintptr_t act_scale, uintptr_t w_scale,
           uintptr_t stream) {
            int rc = flash_rt::gemm::block128_sm89::
                fp8_block128_gemm_blockscaled_sm89_bf16out(
                    to_ptr(A), to_ptr(B), to_ptr(D),
                    M, N, K,
                    reinterpret_cast<const float*>(act_scale),
                    reinterpret_cast<const float*>(w_scale),
                    to_stream(stream));
            if (rc != 0)
                throw std::runtime_error(
                    "fp8_block128_gemm_blockscaled_sm89_bf16out launch failed");
        },
        py::arg("A"), py::arg("B"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("act_block_scale"), py::arg("w_block_scale"),
        py::arg("stream") = 0);

    // GeGLU silu-fold (production entry, env-gated by the caller): fuses the
    // gate_up GEMM + silu_mul + per-token block-128 FP8 quant into one launch.
    // This is a NEW binding (no legacy alias replaced). Argument shapes:
    //   A         : [M, K]        FP8 e4m3 row-major   (per-token quantized act)
    //   B         : [2*N, K]      FP8 e4m3 row-major   (gate_up_w: gate rows
    //                                                  [0,N), up rows [N,2N))
    //   act_scale : [M, K/128]    fp32 row-major
    //   w_scale   : [2*N/128, K/128] fp32 row-major    (gate_up_s; up row =
    //                                                  gate row + N/128)
    //   output    : [M, N]        FP8 e4m3 row-major
    //   out_scale : [M, N/128]    fp32 row-major
    // N and K must be multiples of 128. Beats the baseline (gate_up GEMM +
    // silu_mul) only in the small-M (launch-bound) regime; the frontend
    // dispatcher gates on M before calling this. Additive: the baseline path is
    // unchanged when this is not selected. See fp8_bs_geglu_silu_fold_kernel in
    // fp8_bs_gemm_device.cuh.
    m.def("fp8_bs_geglu_silu_fold_sm89_fp8out",
        [](uintptr_t A, uintptr_t B,
           int M, int N, int K,
           uintptr_t act_scale, uintptr_t w_scale,
           uintptr_t output, uintptr_t out_scale,
           uintptr_t stream) {
            int rc = flash_rt::gemm::block128_sm89::
                fp8_bs_geglu_silu_fold_sm89_32x128_w4_s1(
                    to_ptr(A), to_ptr(B), M, N, K,
                    reinterpret_cast<const float*>(act_scale),
                    reinterpret_cast<const float*>(w_scale),
                    to_ptr(output), reinterpret_cast<float*>(out_scale),
                    to_stream(stream));
            if (rc != 0)
                throw std::runtime_error(
                    "fp8_bs_geglu_silu_fold_sm89_fp8out launch failed");
        },
        py::arg("A"), py::arg("B"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("act_block_scale"), py::arg("w_block_scale"),
        py::arg("output"), py::arg("out_scale"),
        py::arg("stream") = 0);

    // Bench-only tile-variant bindings for prefill GEMM tuning. Not used by
    // the frontend; exposed only for explicit Qwen3-VL dev builds so the
    // production pybind surface stays runtime-only.
#ifdef ENABLE_QWEN3_VL_DEV_KERNELS
#define BIND_GEMM_TILE(NAME)                                                        m.def("bench_" #NAME,                                                               [](uintptr_t A, uintptr_t B, uintptr_t D, int M, int N, int K,                     uintptr_t act_scale, uintptr_t w_scale, uintptr_t stream) {                      return flash_rt::gemm::block128_sm89::NAME(                                         to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,                                      reinterpret_cast<const float*>(act_scale),                                      reinterpret_cast<const float*>(w_scale), to_stream(stream));            },                                                                              py::arg("A"), py::arg("B"), py::arg("D"),                                      py::arg("M"), py::arg("N"), py::arg("K"),                                      py::arg("act_block_scale"), py::arg("w_block_scale"),                          py::arg("stream") = 0)
    BIND_GEMM_TILE(fp8_block128_gemm_bs_sm89_16x64x128_w4);
    BIND_GEMM_TILE(fp8_block128_gemm_bs_sm89_32x64x128_w4);
    BIND_GEMM_TILE(fp8_block128_gemm_bs_sm89_64x64x128_w4);
    BIND_GEMM_TILE(fp8_block128_gemm_bs_sm89_64x64x128_w4_s1);
    BIND_GEMM_TILE(fp8_block128_gemm_bs_sm89_128x128x128_w8_s1);
    BIND_GEMM_TILE(fp8_block128_gemm_bs_sm89_32x128x128_w4);
    BIND_GEMM_TILE(fp8_block128_gemm_bs_sm89_64x128x128_w8);
    BIND_GEMM_TILE(fp8_block128_gemm_bs_sm89_128x128x128_w8);
#undef BIND_GEMM_TILE

    // Residual-fold GEMM bench bindings: D = bf16(acc) + resid. Same tile set
    // as the residual-fold kernels (fp8_block128_gemm_*_resid). Exposed only for
    // dev builds so the production pybind surface stays runtime-only.
#define BIND_GEMM_TILE_RESID(NAME)                                                   m.def("bench_" #NAME,                                                          [](uintptr_t A, uintptr_t B, uintptr_t D, int M, int N, int K,                   uintptr_t act_scale, uintptr_t w_scale, uintptr_t resid, uintptr_t stream) { return flash_rt::gemm::block128_sm89::NAME(                                      to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,                                     reinterpret_cast<const float*>(act_scale),                                       reinterpret_cast<const float*>(w_scale), to_ptr(resid), to_stream(stream));         },                                                                             py::arg("A"), py::arg("B"), py::arg("D"),                                      py::arg("M"), py::arg("N"), py::arg("K"),                                       py::arg("act_block_scale"), py::arg("w_block_scale"),                          py::arg("resid"), py::arg("stream") = 0)
    BIND_GEMM_TILE_RESID(fp8_block128_gemm_bs_sm89_32x64x128_w4_resid);
    BIND_GEMM_TILE_RESID(fp8_block128_gemm_bs_sm89_64x64x128_w4_resid);
    BIND_GEMM_TILE_RESID(fp8_block128_gemm_bs_sm89_64x64x128_w4_s1_resid);
    BIND_GEMM_TILE_RESID(fp8_block128_gemm_bs_sm89_128x128x128_w8_s1_resid);
#undef BIND_GEMM_TILE_RESID

    // GeGLU silu-fold bench bindings: fuse gate+up GEMM + silu(gate)*up +
    // per-token block-128 FP8 quant into one launch (no [M,2N] BF16 transient).
    // B = gate_up_w [2*N, K], w_scale = gate_up_s [2*N/128, K/128]; outputs
    // FP8 [M,N] + scale [M,N/128]. Dev-builds only.
#define BIND_GEGLU_TILE(NAME)                                                  m.def("bench_" #NAME,                                                        [](uintptr_t A, uintptr_t B, int M, int N, int K,                          uintptr_t act_scale, uintptr_t w_scale, uintptr_t output,                  uintptr_t out_scale, uintptr_t stream) {                                  return flash_rt::gemm::block128_sm89::NAME(                                 to_ptr(A), to_ptr(B), M, N, K,                                           reinterpret_cast<const float*>(act_scale),                                    reinterpret_cast<const float*>(w_scale), to_ptr(output),                       reinterpret_cast<float*>(out_scale), to_stream(stream));                },                                                                          py::arg("A"), py::arg("B"), py::arg("M"), py::arg("N"), py::arg("K"),        py::arg("act_block_scale"), py::arg("w_block_scale"), py::arg("output"),  py::arg("out_scale"), py::arg("stream") = 0)
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_sm89_32x128_w4_s2);
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_sm89_16x128_w4_s2);
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_sm89_64x128_w4_s2);
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_sm89_128x128_w8_s1);
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_sm89_32x128_w4_s1);
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_sm89_16x128_w4_s1);
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_apersist_sm89_32x128_w4_s1);
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_apersist_sm89_16x128_w4_s1);
    BIND_GEGLU_TILE(fp8_bs_geglu_silu_fold_apersist_sm89_32x128_w4_s2);
#undef BIND_GEGLU_TILE
#endif

#define BIND_BLOCK128_GEMV_M1(NAME)                                           \
    m.def("ht_" #NAME,                                                       \
        [](uintptr_t A, uintptr_t B, uintptr_t D,                            \
           int M, int N, int K, uintptr_t act_scale, uintptr_t w_scale,       \
           float alpha, uintptr_t stream) {                                  \
            return flash_rt::gemm::gemv_m1_sm89::NAME(                            \
                to_ptr(A), to_ptr(B), to_ptr(D),                             \
                M, N, K, reinterpret_cast<const float*>(act_scale),          \
                reinterpret_cast<const float*>(w_scale), alpha,              \
                to_stream(stream));                                          \
        },                                                                   \
        py::arg("A"), py::arg("B"), py::arg("D"),                          \
        py::arg("M"), py::arg("N"), py::arg("K"),                          \
        py::arg("act_scale"), py::arg("w_scale"), py::arg("alpha"),        \
        py::arg("stream") = 0)

    BIND_BLOCK128_GEMV_M1(gemv_fp8_block128_m1_w4);
    BIND_BLOCK128_GEMV_M1(gemv_fp8_block128_m1_w8);
    BIND_BLOCK128_GEMV_M1(gemv_fp8_block128_m1_w16);

    // BF16-input GEMV: A is BF16, B is FP8 with block-128 weight scale.
    // Eliminates standalone FP8 activation quantization before O-proj.
#define BIND_BLOCK128_GEMV_M1_BF16IN(NAME)                                     \
    m.def("ht_" #NAME,                                                         \
        [](uintptr_t A, uintptr_t B, uintptr_t D,                              \
           int M, int N, int K, uintptr_t w_scale, uintptr_t stream) {         \
            return flash_rt::gemm::gemv_m1_sm89::NAME(                          \
                to_ptr(A), to_ptr(B), to_ptr(D),                               \
                M, N, K, reinterpret_cast<const float*>(w_scale),              \
                to_stream(stream));                                            \
        },                                                                     \
        py::arg("A"), py::arg("B"), py::arg("D"),                              \
        py::arg("M"), py::arg("N"), py::arg("K"),                              \
        py::arg("w_scale"), py::arg("stream") = 0)

    BIND_BLOCK128_GEMV_M1_BF16IN(gemv_fp8_block128_m1_bf16in_w8);
    BIND_BLOCK128_GEMV_M1_BF16IN(gemv_fp8_block128_m1_bf16in_w16);

#undef BIND_BLOCK128_GEMV_M1_BF16IN

#endif  // ENABLE_SM89_BLOCK_FP8_GEMM

// ── Fused QK norm-rope-kvwrite ──
// Shared by the SM89 FP8 path and the SM110 (Thor) BF16 prefill, which builds
// qwen3_qkv_post_proc.cu without the SM89 block-FP8 GEMM. The batched variant
// is the one the Thor prefill uses.
#ifdef ENABLE_QWEN3_VL_QKV_POSTPROC
    m.def("qwen3_qk_norm_rope_kvwrite_bf16",
        [](uintptr_t q_pre, uintptr_t k_pre, uintptr_t v_pre,
           uintptr_t q_norm_w, uintptr_t k_norm_w,
           uintptr_t cos, uintptr_t sin,
           uintptr_t q_buf_dst,
           uintptr_t k_cache_dst, uintptr_t v_cache_dst,
           int n_q_heads, int n_kv_heads, float eps,
           uintptr_t stream) -> int {
            return flash_rt::kernels::qwen3_qk_norm_rope_kvwrite_bf16(
                to_ptr(q_pre), to_ptr(k_pre), to_ptr(v_pre),
                to_ptr(q_norm_w), to_ptr(k_norm_w),
                to_ptr(cos), to_ptr(sin),
                to_ptr(q_buf_dst),
                to_ptr(k_cache_dst), to_ptr(v_cache_dst),
                n_q_heads, n_kv_heads, eps, to_stream(stream));
        },
        py::arg("q_pre"), py::arg("k_pre"), py::arg("v_pre"),
        py::arg("q_norm_w"), py::arg("k_norm_w"),
        py::arg("cos"), py::arg("sin"),
        py::arg("q_buf_dst"),
        py::arg("k_cache_dst"), py::arg("v_cache_dst"),
        py::arg("n_q_heads"), py::arg("n_kv_heads"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("qwen3_qk_norm_rope_kvwrite_batched_bf16",
        [](uintptr_t q_pre, uintptr_t k_pre, uintptr_t v_pre,
           uintptr_t q_norm_w, uintptr_t k_norm_w,
           uintptr_t cos, uintptr_t sin,
           uintptr_t q_buf_dst,
           uintptr_t k_cache_dst, uintptr_t v_cache_dst,
           int seq_len,
           int q_pre_row_elems, int k_pre_row_elems, int v_pre_row_elems,
           int q_dst_row_elems, int kv_dst_row_elems,
           int n_q_heads, int n_kv_heads, float eps,
           uintptr_t stream) -> int {
            return flash_rt::kernels::qwen3_qk_norm_rope_kvwrite_batched_bf16(
                to_ptr(q_pre), to_ptr(k_pre), to_ptr(v_pre),
                to_ptr(q_norm_w), to_ptr(k_norm_w),
                to_ptr(cos), to_ptr(sin),
                to_ptr(q_buf_dst),
                to_ptr(k_cache_dst), to_ptr(v_cache_dst),
                seq_len,
                q_pre_row_elems, k_pre_row_elems, v_pre_row_elems,
                q_dst_row_elems, kv_dst_row_elems,
                n_q_heads, n_kv_heads, eps, to_stream(stream));
        },
        py::arg("q_pre"), py::arg("k_pre"), py::arg("v_pre"),
        py::arg("q_norm_w"), py::arg("k_norm_w"),
        py::arg("cos"), py::arg("sin"),
        py::arg("q_buf_dst"),
        py::arg("k_cache_dst"), py::arg("v_cache_dst"),
        py::arg("seq_len"),
        py::arg("q_pre_row_elems"), py::arg("k_pre_row_elems"),
        py::arg("v_pre_row_elems"),
        py::arg("q_dst_row_elems"), py::arg("kv_dst_row_elems"),
        py::arg("n_q_heads"), py::arg("n_kv_heads"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);
#endif  // ENABLE_QWEN3_VL_QKV_POSTPROC

#ifdef ENABLE_QWEN3_VL_BF16_CUBLASLT
    // BF16 cuBLASLt matmul for Qwen3-VL BF16 linears on SM87/SM89/SM110. SM120
    // uses w16a16_gemm_sm120_bf16 from flash_rt_kernels instead.
    m.def("bf16_matmul_cublaslt_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t out,
           int M, int N, int K, uintptr_t stream) {
            flash_rt::kernels::bf16_matmul_cublaslt_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W),
                reinterpret_cast<__nv_bfloat16*>(out),
                M, N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("out"),
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

#ifdef ENABLE_QWEN3_VL_BF16_GEMV_M1
    m.def("qwen3_vl_bf16_gemv_m1",
        [](uintptr_t x, uintptr_t W, uintptr_t out,
           int N, int K, uintptr_t stream) {
            flash_rt::kernels::qwen3_vl_bf16_gemv_m1(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W),
                reinterpret_cast<__nv_bfloat16*>(out),
                N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("out"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // ── Weight-only quantized M=1 decode GEMVs ──
    // Decode at M=1 is bound by the weight HBM read, so shrinking the weights
    // buys throughput directly. All of these take packed weights + bf16 per-16
    // block scales and dequantize in-kernel to bf16, so no FP8/FP4 tensor core
    // is required. Prefill keeps using the bf16 weights.

    // FP8 e4m3 (scale = amax/448): 1.125 B/elem vs bf16 2.0. Relies on the
    // hardware e4m3 conversion, so it only stays bandwidth-bound on sm_89+.
    m.def("qwen3_vl_w8_gemv_m1",
        [](uintptr_t x, uintptr_t Wp, uintptr_t Ws, uintptr_t out,
           int N, int K, uintptr_t stream) {
            flash_rt::kernels::qwen3_vl_w8_gemv_m1(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const uint8_t*>(Wp),
                reinterpret_cast<const __nv_bfloat16*>(Ws),
                reinterpret_cast<__nv_bfloat16*>(out),
                N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("Wp"), py::arg("Ws"), py::arg("out"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // NVFP4 e2m1 (scale = amax/6), two nibbles per byte: 0.625 B/elem, ~3.2x
    // less weight traffic than bf16. Dequant is a __byte_perm LUT into e4m3, so
    // like the W8 sibling it wants sm_89+.
    m.def("qwen3_vl_w4_gemv_m1",
        [](uintptr_t x, uintptr_t Wp, uintptr_t Ws, uintptr_t out,
           int N, int K, uintptr_t stream) {
            flash_rt::kernels::qwen3_vl_w4_gemv_m1(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const uint8_t*>(Wp),
                reinterpret_cast<const __nv_bfloat16*>(Ws),
                reinterpret_cast<__nv_bfloat16*>(out),
                N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("Wp"), py::arg("Ws"), py::arg("out"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);
#endif  // ENABLE_QWEN3_VL_BF16_GEMV_M1
#endif  // ENABLE_QWEN3_VL_BF16_CUBLASLT

#ifdef ENABLE_QWEN3_VL_INT_DECODE
    // INT8 symmetric (scale = amax/127): 1.125 B/elem vs bf16 2.0. Dequant is
    // a hardware I2F, which is why this — not the e4m3 sibling — is the W8
    // choice on Ampere (sm_87), where FP8 conversion is emulated in software.
    m.def("qwen3_vl_int8_gemv_m1",
        [](uintptr_t x, uintptr_t Wp, uintptr_t Ws, uintptr_t out,
           int N, int K, uintptr_t stream) {
            flash_rt::kernels::qwen3_vl_int8_gemv_m1(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const uint8_t*>(Wp),
                reinterpret_cast<const __nv_bfloat16*>(Ws),
                reinterpret_cast<__nv_bfloat16*>(out),
                N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("Wp"), py::arg("Ws"), py::arg("out"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // INT4 symmetric (scale = amax/7), two nibbles per byte: 0.625 B/elem.
    // Unpack is shift + sign-extend + I2F, also Ampere-friendly. Coarser than
    // INT8 (15 levels) — validate per model.
    m.def("qwen3_vl_int4_gemv_m1",
        [](uintptr_t x, uintptr_t Wp, uintptr_t Ws, uintptr_t out,
           int N, int K, uintptr_t stream) {
            flash_rt::kernels::qwen3_vl_int4_gemv_m1(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const uint8_t*>(Wp),
                reinterpret_cast<const __nv_bfloat16*>(Ws),
                reinterpret_cast<__nv_bfloat16*>(out),
                N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("Wp"), py::arg("Ws"), py::arg("out"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // ── INT8 KV cache: row quantize + q=1 flash-decoding attention ──
    // KV rows are mirrored into int8 with one bf16 scale per (position,
    // kv-head) 128-element row, halving the KV bytes each decode step reads.
    // Prefill still runs FA2 against the bf16 cache.

    // Quantize n_rows contiguous 128-element bf16 rows. Layout-agnostic, so it
    // serves both the per-step pass (n_rows = kv_heads at one layer/position)
    // and the post-prefill bulk pass over the whole cache prefix.
    m.def("qwen3_kv_rows_quant_int8",
        [](uintptr_t src, uintptr_t dst, uintptr_t scales, int n_rows,
           uintptr_t stream) {
            flash_rt::kernels::qwen3_kv_rows_quant_int8(
                reinterpret_cast<const __nv_bfloat16*>(src),
                reinterpret_cast<int8_t*>(dst),
                reinterpret_cast<__nv_bfloat16*>(scales),
                n_rows, to_stream(stream));
        },
        py::arg("src"), py::arg("dst"), py::arg("scales"),
        py::arg("n_rows"), py::arg("stream") = 0);

    // Partial pass: one block per (kv-head, 128-position chunk). Specialized
    // for GQA 16Q/8KV with head_dim 128.
    m.def("qwen3_attn_decode_int8kv_partial",
        [](uintptr_t q, uintptr_t k8, uintptr_t v8, uintptr_t ks,
           uintptr_t vs, uintptr_t part_o, uintptr_t part_m, uintptr_t part_l,
           int kv_len, int n_chunks, float softmax_scale, uintptr_t stream) {
            flash_rt::kernels::qwen3_attn_decode_int8kv_partial(
                reinterpret_cast<const __nv_bfloat16*>(q),
                reinterpret_cast<const int8_t*>(k8),
                reinterpret_cast<const int8_t*>(v8),
                reinterpret_cast<const __nv_bfloat16*>(ks),
                reinterpret_cast<const __nv_bfloat16*>(vs),
                reinterpret_cast<float*>(part_o),
                reinterpret_cast<float*>(part_m),
                reinterpret_cast<float*>(part_l),
                kv_len, n_chunks, softmax_scale, to_stream(stream));
        },
        py::arg("q"), py::arg("k8"), py::arg("v8"), py::arg("ks"),
        py::arg("vs"), py::arg("part_o"), py::arg("part_m"), py::arg("part_l"),
        py::arg("kv_len"), py::arg("n_chunks"), py::arg("softmax_scale"),
        py::arg("stream") = 0);

    // Combine pass: rescale and merge the chunk partials into bf16 O.
    m.def("qwen3_attn_decode_int8kv_combine",
        [](uintptr_t part_o, uintptr_t part_m, uintptr_t part_l,
           uintptr_t out, int n_chunks, uintptr_t stream) {
            flash_rt::kernels::qwen3_attn_decode_int8kv_combine(
                reinterpret_cast<const float*>(part_o),
                reinterpret_cast<const float*>(part_m),
                reinterpret_cast<const float*>(part_l),
                reinterpret_cast<__nv_bfloat16*>(out),
                n_chunks, to_stream(stream));
        },
        py::arg("part_o"), py::arg("part_m"), py::arg("part_l"),
        py::arg("out"), py::arg("n_chunks"), py::arg("stream") = 0);
#endif  // ENABLE_QWEN3_VL_INT_DECODE
}
