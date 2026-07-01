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
#endif

#ifdef ENABLE_SM89_BLOCK_FP8_GEMM
#include "gemm/fp8_block128_gemm_mma_sm89.cuh"
#include "gemm/fp8_gemv_m1_sm89.cuh"
#include "quantize/fp8_per_token_block_quant.cuh"
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

#ifdef ENABLE_SM89_BLOCK_FP8_GEMM
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
#endif  // ENABLE_SM89_BLOCK_FP8_GEMM

#ifdef ENABLE_QWEN3_VL_BF16_CUBLASLT
    // BF16 cuBLASLt matmul for Qwen3-VL BF16 linears on SM87/SM89. SM120
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
#endif
#endif
}
