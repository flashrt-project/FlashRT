// ================================================================
// FlashRT — pybind11 bindings
// Exposes GemmRunner + all CUDA kernels to Python
// ================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include "context.h"
#include "gemm/gemm_runner.h"
#include "gemm/fp8_block128_gemm.cuh"
#ifdef ENABLE_CUTLASS_SM120_BLOCK_FP8
#include "gemm/cutlass_sm120_block128_fp8_gemm.cuh"
#endif
#ifdef ENABLE_CUTLASS_SM120_NVFP4_W4A16
#include "gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cuh"
#include "quantize/nvfp4_sf_reshape_sm120.cuh"
#endif
#include "kernels/kernels.h"
#include "kernels/causal_conv1d_qwen36.cuh"
#include "kernels/gated_deltanet_qwen36.cuh"
#include "kernels/rms_norm_gated_silu_qwen36.cuh"
#include "kernels/silu_mul_qwen36.cuh"
#include "kernels/bf16_matvec_qwen36.cuh"
#include "kernels/bf16_matmul_qwen36.cuh"
#include "kernels/silu_mul_to_nvfp4_swizzled.cuh"
#include "kernels/qwen3_qkv_post_proc.cuh"
#include "kernels/fp4_w4a4_matvec_sm120.cuh"
#include "kernels/fp4_w4a4_mma_sm120.cuh"
#include "quantize/fp8_block128_dequant.cuh"
#include "quantize/fp8_block128_to_nvfp4_swizzled.cuh"
#include "quantize/bf16_weight_to_nvfp4_swizzled.cuh"
#include "quantize/fp8_per_token_block_quant.cuh"
#include "attention/fmha_dispatch.h"

namespace py = pybind11;

static void* to_ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }
template<typename T> static T* typed_ptr(uintptr_t addr) { return reinterpret_cast<T*>(addr); }
static cudaStream_t to_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

// TurboQuant unpack + combine (csrc/quantize/tq_dequant_kv.cu)
extern "C" void tq_unpack_packed_mixed_launch(
    const void* k_idx_packed, const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse, const void* cb_v,
    void* y_k_bf16, void* qjl_fp32, void* y_v_bf16,
    int M, int b_k_mse, int b_v,
    cudaStream_t stream);
extern "C" void tq_unpack_packed_bf16_launch(
    const void* k_idx_packed, const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse, const void* cb_v,
    void* y_k, void* qjl_bf, void* y_v,
    int M, int b_k_mse, int b_v,
    cudaStream_t stream);
extern "C" void tq_write_kv_packed_launch(
    const void* k_in, const void* v_in,
    int s_start, int S,
    const void* rotation, const void* jl,
    const void* cb_k_mse, const void* cb_v,
    void* k_idx_packed_layer, void* k_qjl_packed_layer,
    void* k_norm_layer, void* k_rnorm_layer,
    void* v_idx_packed_layer, void* v_norm_layer,
    int b_k_mse, int b_v,
    cudaStream_t stream);
extern "C" void tq_write_k1_unit_norm_launch(
    const void* k_in, const void* v_in,
    void* k_unit_out, void* v_unit_out,
    void* norm_k_out, void* norm_v_out,
    int M, int b_k_mse, int b_v, cudaStream_t stream);
extern "C" void tq_write_k2_argmin_pack_launch(
    const void* y_k, const void* y_v,
    const void* cb_k_mse, const void* cb_v,
    void* k_idx_packed_layer, void* v_idx_packed_layer,
    void* dq_in,
    int s_start, int num_kv, int M,
    int b_k_mse, int b_v, cudaStream_t stream);
extern "C" void tq_write_k3_residual_rnorm_launch(
    const void* k_unit, const void* dq_k,
    void* residual, void* rnorm_k,
    int M, cudaStream_t stream);
extern "C" void tq_write_k4_qjl_norms_launch(
    const void* Sr,
    const void* norm_k, const void* rnorm_k, const void* norm_v,
    void* k_qjl_packed_layer,
    void* k_norm_layer, void* k_rnorm_layer, void* v_norm_layer,
    int s_start, int num_kv, int M, cudaStream_t stream);
extern "C" void tq_unpack_packed_fp32_launch(
    const void* k_idx_packed, const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse, const void* cb_v,
    void* y_k, void* qjl_f, void* y_v,
    int M, int b_k_mse, int b_v,
    cudaStream_t stream);
extern "C" void tq_combine_kv_bf16_launch(
    const void* k_mse, const void* k_qjl, const void* v_unit,
    const void* k_norm, const void* k_rnorm, const void* v_norm,
    void* k_out, void* v_out,
    int M, float coef,
    cudaStream_t stream);
extern "C" void tq_combine_kv_fp32_in_launch(
    const void* k_mse, const void* k_qjl, const void* v_unit,
    const void* k_norm, const void* k_rnorm, const void* v_norm,
    void* k_out, void* v_out,
    int M, float coef,
    cudaStream_t stream);
extern "C" void tq_bf16_fp32_gemm_launch(
    const void* a_bf16, const void* b_bf16,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream);
extern "C" void tq_fp32_gemm_tf32_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream);
extern "C" void wmma_probe_launch(
    const void* a_bf16, const void* b_bf16, void* c_fp32,
    int M, cudaStream_t stream);
extern "C" void tq_cutlass_bf16_gemm_launch(
    const void* a_bf16, const void* b_bf16, void* d_bf16,
    int M, int N, int K, cudaStream_t stream);
extern "C" void tq_cutlass_v_combine_launch(
    const void* a_bf16, const void* b_bf16, const void* norm_v_fp32,
    void* d_bf16, int M, int N, int K, cudaStream_t stream);
extern "C" void tq_cutlass_k_combine_launch(
    const void* a_bf16, const void* b_bf16,
    const void* sr_fp32,
    const void* norm_k_fp32, const void* coef_rnorm_fp32,
    void* d_bf16, int M, int N, int K, cudaStream_t stream);
extern "C" void tq_dequant_kv_fused_launch(
    const void* k_idx_packed, const void* k_qjl_packed,
    const void* k_norm, const void* k_rnorm,
    const void* v_idx_packed, const void* v_norm,
    const void* rotation, const void* jl,
    const void* cb_k_mse, const void* cb_v,
    void* k_out, void* v_out,
    int M, float coef,
    int b_k_mse, int b_v,
    cudaStream_t stream);

#ifdef ENABLE_NVFP4
extern "C" int run_w4a8_gemm(void*, void*, void*, void*, void*, int, int, int, cudaStream_t);
extern "C" float launch_w4a8_gemm(void*, void*, void*, void*, void*, void*, int, int, int, float, float, int, int);
#endif

// ENABLE_FA2 moved to a separate pybind module (flash_rt_fa2.so —
// csrc/fa2_bindings.cpp). This keeps the main flash_rt_kernels.so
// small and its build fast by isolating FA2's heavy CUTLASS 3.x
// template codegen.

#ifdef ENABLE_SM100_CUTLASS
extern "C" int cutlass_fp8_sq(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_t1(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_wide(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_plain(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_gelu(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_sq_f32out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_wide_f32out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_sq_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_wide_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_t1_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
#endif

PYBIND11_MODULE(flash_rt_kernels, m) {
    m.doc() = "FlashRT C++/CUDA inference kernels";

    // ── FvkContext: per-instance cuBLAS handle ──
    py::class_<FvkContext>(m, "FvkContext")
        .def(py::init<>())
        .def_property_readonly("handle_ptr", [](const FvkContext& ctx) {
            return reinterpret_cast<uintptr_t>(ctx.cublas_handle);
        });

    // ── GemmRunner ──
    py::class_<GemmRunner>(m, "GemmRunner")
        .def(py::init<>())
        .def("bf16_gemm", [](GemmRunner& self,
                              uintptr_t A, uintptr_t B, uintptr_t D,
                              int M, int N, int K,
                              float alpha, float beta,
                              int warmup, int iters) {
            return self.bf16_gemm(to_ptr(A), to_ptr(B), to_ptr(D),
                                  M, N, K, alpha, beta, warmup, iters);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
           py::arg("warmup") = 3, py::arg("iters") = 100)
        .def("fp8_gemm", [](GemmRunner& self,
                             uintptr_t A, uintptr_t B, uintptr_t D,
                             int M, int N, int K,
                             float scale_a, float scale_b,
                             int warmup, int iters) {
            return self.fp8_gemm(to_ptr(A), to_ptr(B), to_ptr(D),
                                 M, N, K, scale_a, scale_b, warmup, iters);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("scale_a") = 1.0f, py::arg("scale_b") = 1.0f,
           py::arg("warmup") = 3, py::arg("iters") = 100)
#ifdef ENABLE_NVFP4
        .def("fp4_gemm", [](GemmRunner& self,
                             uintptr_t A, uintptr_t SFA,
                             uintptr_t B, uintptr_t SFB,
                             uintptr_t D,
                             int M, int N, int K,
                             int warmup, int iters) {
            return self.fp4_gemm(to_ptr(A), to_ptr(SFA),
                                 to_ptr(B), to_ptr(SFB),
                                 to_ptr(D), M, N, K, warmup, iters);
        }, py::arg("A"), py::arg("SFA"),
           py::arg("B"), py::arg("SFB"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("warmup") = 3, py::arg("iters") = 100)
#endif
        // Inference methods (stream-based, CUDA Graph compatible)
        .def("fp16_nn", [](GemmRunner& self,
                            uintptr_t A, uintptr_t B, uintptr_t D,
                            int M, int N, int K, uintptr_t stream) {
            self.fp16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn", [](GemmRunner& self,
                            uintptr_t A, uintptr_t B, uintptr_t D,
                            int M, int N, int K, uintptr_t stream) {
            self.bf16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_res", [](GemmRunner& self,
                                uintptr_t A, uintptr_t B, uintptr_t D,
                                int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_res(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias", [](GemmRunner& self,
                                 uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                 int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                               M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias_gelu", [](GemmRunner& self,
                                      uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                      int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias_gelu(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                    M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias_res", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                     int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias_res(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                   M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("fp8_run_dev", [](GemmRunner& self,
                                uintptr_t A, uintptr_t B, uintptr_t D,
                                int M, int N, int K,
                                uintptr_t d_scale_a, uintptr_t d_scale_b,
                                uintptr_t stream) {
            self.fp8_run_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                              reinterpret_cast<float*>(d_scale_a),
                              reinterpret_cast<float*>(d_scale_b), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("stream") = 0)
        .def("fp8_nn_dev", [](GemmRunner& self,
                               uintptr_t A, uintptr_t B, uintptr_t D,
                               int M, int N, int K,
                               uintptr_t d_scale_a, uintptr_t d_scale_b,
                               uintptr_t stream) {
            self.fp8_nn_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                             reinterpret_cast<float*>(d_scale_a),
                             reinterpret_cast<float*>(d_scale_b), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("stream") = 0)
        // FP8 with device descale → FP16 (GemmRunner handle, matching pi05)
        .def("fp8_descale_fp16", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K,
                                     uintptr_t act_descale, uintptr_t w_descale, uintptr_t stream) {
            self.fp8_descale_fp16(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                                   reinterpret_cast<float*>(act_descale),
                                   reinterpret_cast<float*>(w_descale), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0)
        // FP8 GEMM with epilogues (matches pi05 cublaslt_fp8.cuh)
        .def("fp8_nn_bias", [](GemmRunner& self,
                                uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                              M, N, K, alpha, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha") = 1.0f, py::arg("stream") = 0)
        .def("fp8_nn_bias_res", [](GemmRunner& self,
                                    uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                    int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_bias_res(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                  M, N, K, alpha, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha") = 1.0f, py::arg("stream") = 0)
        .def("fp8_nn_gelu_bias", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                     int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_gelu_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                   M, N, K, alpha, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha") = 1.0f, py::arg("stream") = 0)
        // Autotune
        .def("autotune_bf16_nn", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K, int num_algos) {
            self.autotune_bf16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("num_algos") = 16)
        .def("autotune_fp8_nn_dev", [](GemmRunner& self,
                                        uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K,
                                        uintptr_t d_scale_a, uintptr_t d_scale_b,
                                        int num_algos) {
            self.autotune_fp8_nn_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                                      reinterpret_cast<float*>(d_scale_a),
                                      reinterpret_cast<float*>(d_scale_b), num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("num_algos") = 16)
#ifdef ENABLE_NVFP4
        .def("fp4_nn_dev", [](GemmRunner& self,
                               uintptr_t A_fp4, uintptr_t SFA,
                               uintptr_t B_fp4, uintptr_t SFB,
                               uintptr_t D,
                               int M, int N, int K, uintptr_t stream) {
            self.fp4_nn_dev(to_ptr(A_fp4), to_ptr(SFA),
                             to_ptr(B_fp4), to_ptr(SFB),
                             to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A_fp4"), py::arg("SFA"),
           py::arg("B_fp4"), py::arg("SFB"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("autotune_fp4_nn_dev", [](GemmRunner& self,
                                        uintptr_t A_fp4, uintptr_t SFA,
                                        uintptr_t B_fp4, uintptr_t SFB,
                                        uintptr_t D,
                                        int M, int N, int K, int num_algos) {
            self.autotune_fp4_nn_dev(to_ptr(A_fp4), to_ptr(SFA),
                                      to_ptr(B_fp4), to_ptr(SFB),
                                      to_ptr(D), M, N, K, num_algos);
        }, py::arg("A_fp4"), py::arg("SFA"),
           py::arg("B_fp4"), py::arg("SFB"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("num_algos") = 16)
#endif
    ;

    // ── Kernel functions ──
    // Norm
    m.def("rms_norm", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                          int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                 typed_ptr<__nv_bfloat16>(out), seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("rms_norm_inplace", [](uintptr_t weight, uintptr_t x,
                                  int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm_inplace(typed_ptr<__nv_bfloat16>(weight),
                         typed_ptr<__nv_bfloat16>(x), seq_len, dim, eps, to_stream(stream));
    }, py::arg("weight"), py::arg("x"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("layer_norm", [](uintptr_t x, uintptr_t weight, uintptr_t bias,
                            uintptr_t out, int seq_len, int dim, float eps, uintptr_t stream) {
        layer_norm(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                   typed_ptr<__nv_bfloat16>(bias), typed_ptr<__nv_bfloat16>(out),
                   seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("ada_rms_norm_style", [](uintptr_t x, uintptr_t weight, uintptr_t style,
                                    uintptr_t out, uintptr_t gate_out,
                                    int seq_len, int dim, float eps, uintptr_t stream) {
        ada_rms_norm_style(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                           typed_ptr<__nv_bfloat16>(style),
                           typed_ptr<__nv_bfloat16>(out), typed_ptr<__nv_bfloat16>(gate_out),
                           seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("style"),
       py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // Fused Norm → FP8
    m.def("rms_norm_fp8", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                              int seq_len, int dim, float eps,
                              uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                     typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, eps,
                     reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    m.def("ada_rms_norm_style_fp8", [](uintptr_t x, uintptr_t weight, uintptr_t style,
                                        uintptr_t out, uintptr_t gate_out,
                                        int seq_len, int dim, float eps,
                                        uintptr_t d_scale, uintptr_t stream) {
        ada_rms_norm_style_fp8(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                               typed_ptr<__nv_bfloat16>(style),
                               typed_ptr<__nv_fp8_e4m3>(out), typed_ptr<__nv_bfloat16>(gate_out),
                               seq_len, dim, eps,
                               reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("style"),
       py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    m.def("residual_add_rms_norm_fp8", [](uintptr_t residual, uintptr_t x,
                                           uintptr_t weight, uintptr_t out,
                                           int seq_len, int dim, float eps,
                                           uintptr_t d_scale, uintptr_t stream) {
        residual_add_rms_norm_fp8(typed_ptr<__nv_bfloat16>(residual),
                                   typed_ptr<__nv_bfloat16>(x),
                                   typed_ptr<__nv_bfloat16>(weight),
                                   typed_ptr<__nv_fp8_e4m3>(out),
                                   seq_len, dim, eps,
                                   reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    m.def("residual_add_rms_norm", [](uintptr_t residual, uintptr_t x,
                                       uintptr_t weight, uintptr_t out,
                                       int seq_len, int dim, float eps, uintptr_t stream) {
        residual_add_rms_norm(typed_ptr<__nv_bfloat16>(residual),
                               typed_ptr<__nv_bfloat16>(x),
                               typed_ptr<__nv_bfloat16>(weight),
                               typed_ptr<__nv_bfloat16>(out),
                               seq_len, dim, eps, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // Activation — GEGLU (tanh-approx GELU(gate) * up), not SiLU.
    m.def("gate_geglu", [](uintptr_t gate, uintptr_t up, uintptr_t out, int n, uintptr_t stream) {
        gate_silu_mul(typed_ptr<__nv_bfloat16>(gate), typed_ptr<__nv_bfloat16>(up),
                      typed_ptr<__nv_bfloat16>(out), n, to_stream(stream));
    }, py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    m.def("gate_geglu_fp16", [](uintptr_t gate, uintptr_t up, uintptr_t out, int n, uintptr_t stream) {
        gate_silu_mul_fp16(typed_ptr<__half>(gate), typed_ptr<__half>(up),
                           typed_ptr<__half>(out), n, to_stream(stream));
    }, py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    m.def("gelu_inplace", [](uintptr_t x, int n, uintptr_t stream) {
        gelu_inplace(typed_ptr<__nv_bfloat16>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("gate_geglu_merged", [](uintptr_t merged, uintptr_t out,
                                   int seq, int half_dim, uintptr_t stream) {
        gate_silu_mul_merged(typed_ptr<__nv_bfloat16>(merged),
                              typed_ptr<__nv_bfloat16>(out), seq, half_dim, to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"), py::arg("stream") = 0);

    m.def("gate_geglu_merged_fp8", [](uintptr_t merged, uintptr_t out,
                                       int seq, int half_dim,
                                       uintptr_t d_scale, uintptr_t stream) {
        gate_silu_mul_merged_fp8(typed_ptr<__nv_bfloat16>(merged),
                                  typed_ptr<__nv_fp8_e4m3>(out), seq, half_dim,
                                  reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"),
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // RoPE
    m.def("rope_apply", [](uintptr_t rope_weights, uintptr_t Q, uintptr_t K,
                            int seq_len, int num_heads, int head_dim, uintptr_t stream) {
        rope_apply(typed_ptr<__nv_bfloat16>(rope_weights),
                   typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                   seq_len, num_heads, head_dim, to_stream(stream));
    }, py::arg("rope_weights"), py::arg("Q"), py::arg("K"),
       py::arg("seq_len"), py::arg("num_heads"), py::arg("head_dim"), py::arg("stream") = 0);

    m.def("qkv_split", [](uintptr_t qkv, uintptr_t Q, uintptr_t K, uintptr_t V,
                           int seq, int q_dim, int k_dim, int v_dim, uintptr_t stream) {
        qkv_split(typed_ptr<__nv_bfloat16>(qkv),
                   typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                   typed_ptr<__nv_bfloat16>(V), seq, q_dim, k_dim, v_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"), py::arg("stream") = 0);

    m.def("qkv_split_fp16", [](uintptr_t qkv, uintptr_t Q, uintptr_t K, uintptr_t V,
                                int seq, int q_dim, int k_dim, int v_dim, uintptr_t stream) {
        qkv_split_fp16(typed_ptr<__half>(qkv),
                        typed_ptr<__half>(Q), typed_ptr<__half>(K),
                        typed_ptr<__half>(V), seq, q_dim, k_dim, v_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"), py::arg("stream") = 0);

    m.def("qkv_split_rope", [](uintptr_t qkv, uintptr_t rope_weights,
                                 uintptr_t Q, uintptr_t K, uintptr_t V,
                                 int seq, int q_dim, int k_dim, int v_dim,
                                 int head_dim, uintptr_t stream) {
        qkv_split_rope(typed_ptr<__nv_bfloat16>(qkv), typed_ptr<__nv_bfloat16>(rope_weights),
                        typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                        typed_ptr<__nv_bfloat16>(V),
                        seq, q_dim, k_dim, v_dim, head_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope_weights"),
       py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"),
       py::arg("head_dim"), py::arg("stream") = 0);

    // Elementwise
    m.def("gate_mul_residual", [](uintptr_t residual, uintptr_t x, uintptr_t gate, int n, uintptr_t stream) {
        gate_mul_residual(typed_ptr<__nv_bfloat16>(residual),
                          typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(gate), n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("n"), py::arg("stream") = 0);

    m.def("bias_residual", [](uintptr_t residual, uintptr_t x, uintptr_t bias,
                               int seq_len, int dim, uintptr_t stream) {
        bias_residual(typed_ptr<__nv_bfloat16>(residual),
                      typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(bias),
                      seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("residual_add", [](uintptr_t residual, uintptr_t x, int n, uintptr_t stream) {
        residual_add(typed_ptr<__nv_bfloat16>(residual),
                     typed_ptr<__nv_bfloat16>(x), n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("cfg_combine_into_residual",
          [](uintptr_t residual, uintptr_t v_cond, uintptr_t v_uncond,
             float beta, int n, uintptr_t stream) {
        cfg_combine_into_residual(typed_ptr<__nv_bfloat16>(residual),
                                  typed_ptr<__nv_bfloat16>(v_cond),
                                  typed_ptr<__nv_bfloat16>(v_uncond),
                                  beta, n, to_stream(stream));
    }, py::arg("residual"), py::arg("v_cond"), py::arg("v_uncond"),
       py::arg("beta"), py::arg("n"), py::arg("stream") = 0);

    // Fusion
    m.def("gate_residual_ada_norm_fp8", [](uintptr_t residual, uintptr_t x,
                                            uintptr_t gate, uintptr_t weight,
                                            uintptr_t style,
                                            uintptr_t out, uintptr_t gate_out,
                                            int seq_len, int dim, float eps,
                                            uintptr_t d_scale, uintptr_t stream) {
        gate_residual_ada_norm_fp8(typed_ptr<__nv_bfloat16>(residual),
                                    typed_ptr<__nv_bfloat16>(x),
                                    typed_ptr<__nv_bfloat16>(gate),
                                    typed_ptr<__nv_bfloat16>(weight),
                                    typed_ptr<__nv_bfloat16>(style),
                                    typed_ptr<__nv_fp8_e4m3>(out),
                                    typed_ptr<__nv_bfloat16>(gate_out),
                                    seq_len, dim, eps,
                                    reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("weight"),
       py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // Quantize
    m.def("quantize_fp8", [](uintptr_t input, uintptr_t output,
                              uintptr_t d_scale, int n, uintptr_t stream) {
        return quantize_fp8(typed_ptr<__nv_bfloat16>(input),
                            typed_ptr<__nv_fp8_e4m3>(output),
                            reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("quantize_fp8_static", [](uintptr_t input, uintptr_t output,
                                     uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_static(typed_ptr<__nv_bfloat16>(input),
                            typed_ptr<__nv_fp8_e4m3>(output),
                            reinterpret_cast<const float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("quantize_fp8_device", [](uintptr_t input, uintptr_t output,
                                     uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_device(typed_ptr<__nv_bfloat16>(input),
                            typed_ptr<__nv_fp8_e4m3>(output),
                            reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    // FP16 device-only FP8 quantize (GPU absmax + scale + quantize, CUDA Graph compatible)
    m.def("quantize_fp8_device_fp16", [](uintptr_t input, uintptr_t output,
                                          uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_device_fp16(reinterpret_cast<const __half*>(input),
                                  typed_ptr<__nv_fp8_e4m3>(output),
                                  reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

#ifdef ENABLE_NVFP4
    m.def("quantize_bf16_to_nvfp4", [](uintptr_t input, uintptr_t fp4_data,
                                         uintptr_t scale_factors, int rows, int cols,
                                         uintptr_t stream) {
        quantize_bf16_to_nvfp4(typed_ptr<__nv_bfloat16>(input),
                                reinterpret_cast<uint8_t*>(fp4_data),
                                reinterpret_cast<uint8_t*>(scale_factors),
                                rows, cols, to_stream(stream));
    }, py::arg("input"), py::arg("fp4_data"), py::arg("scale_factors"),
       py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("quantize_bf16_to_nvfp4_swizzled", [](uintptr_t input, uintptr_t fp4_data,
                                                  uintptr_t scale_factors, int rows, int cols,
                                                  uintptr_t stream) {
        quantize_bf16_to_nvfp4_swizzled(typed_ptr<__nv_bfloat16>(input),
                                         reinterpret_cast<uint8_t*>(fp4_data),
                                         reinterpret_cast<uint8_t*>(scale_factors),
                                         rows, cols, to_stream(stream));
    }, py::arg("input"), py::arg("fp4_data"), py::arg("scale_factors"),
       py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    // Fused: rms_norm(x, weight) -> nvfp4 packed + swizzled SF.
    // Replaces (rms_norm + quantize_bf16_to_nvfp4_swizzled) at every
    // pre-projection norm site on the NVFP4 path. weight = Qwen3.5
    // (1+w) precomputed tensor (same convention as fvk.rms_norm).
    m.def("rms_norm_to_nvfp4_swizzled_bf16",
        [](uintptr_t x, uintptr_t weight,
           uintptr_t packed, uintptr_t sf_swz,
           int rows, int cols, float eps, uintptr_t stream) {
            rms_norm_to_nvfp4_swizzled_bf16(
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(weight),
                reinterpret_cast<uint8_t*>(packed),
                reinterpret_cast<uint8_t*>(sf_swz),
                rows, cols, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("weight"),
        py::arg("packed"), py::arg("sf_swz"),
        py::arg("rows"), py::arg("cols"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // Fused: residual_add(h_in, attn_proj) -> h_post (bf16 written to
    // global) -> rms_norm(h_post, weight) -> nvfp4 packed + swizzled SF.
    // Replaces the (torch.add + rms_norm + quantize_bf16_to_nvfp4_swizzled)
    // 3-launch sequence at every per-layer post-attn / post-MLP transition
    // on the NVFP4 path. h_post is preserved in BF16 because the next
    // residual addition (post-MLP) needs it.
    m.def("residual_add_rms_norm_to_nvfp4_swizzled_bf16",
        [](uintptr_t h_in, uintptr_t attn_proj, uintptr_t h_post,
           uintptr_t weight,
           uintptr_t packed, uintptr_t sf_swz,
           int rows, int cols, float eps, uintptr_t stream) {
            residual_add_rms_norm_to_nvfp4_swizzled_bf16(
                typed_ptr<__nv_bfloat16>(h_in),
                typed_ptr<__nv_bfloat16>(attn_proj),
                typed_ptr<__nv_bfloat16>(h_post),
                typed_ptr<__nv_bfloat16>(weight),
                reinterpret_cast<uint8_t*>(packed),
                reinterpret_cast<uint8_t*>(sf_swz),
                rows, cols, eps, to_stream(stream));
        },
        py::arg("h_in"), py::arg("attn_proj"), py::arg("h_post"),
        py::arg("weight"),
        py::arg("packed"), py::arg("sf_swz"),
        py::arg("rows"), py::arg("cols"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);
#endif

    // Patch embedding
    m.def("patch_im2col", [](uintptr_t input, uintptr_t output, int nv, uintptr_t stream) {
        patch_im2col(reinterpret_cast<const half*>(input),
                     reinterpret_cast<half*>(output), nv, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("nv"), py::arg("stream") = 0);

    m.def("patch_embed_bias_pos", [](uintptr_t output, uintptr_t bias, uintptr_t pos_emb,
                                      int S, int D, int S_per_view, uintptr_t stream) {
        patch_embed_bias_pos(reinterpret_cast<half*>(output),
                             reinterpret_cast<const half*>(bias),
                             reinterpret_cast<const half*>(pos_emb),
                             S, D, S_per_view, to_stream(stream));
    }, py::arg("output"), py::arg("bias"), py::arg("pos_emb"),
       py::arg("S"), py::arg("D"), py::arg("S_per_view"), py::arg("stream") = 0);

    // ── FP16 variants (Thor SM110 path) ──
    // All use uintptr_t for pointers, same as BF16 versions.

    // Norm FP16
    m.def("rms_norm_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                               int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(weight),
                       reinterpret_cast<__half*>(out), seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("layer_norm_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t bias,
                                 uintptr_t out, int seq_len, int dim, float eps, uintptr_t stream) {
        layer_norm_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(weight),
                         reinterpret_cast<const __half*>(bias), reinterpret_cast<__half*>(out),
                         seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("layer_norm_fp8", [](uintptr_t x, uintptr_t out, uintptr_t gamma, uintptr_t beta,
                                int seq_len, int dim, float eps, uintptr_t stream) {
        layer_norm_fp8(reinterpret_cast<const __half*>(x), reinterpret_cast<__nv_fp8_e4m3*>(out),
                        reinterpret_cast<const __half*>(gamma), reinterpret_cast<const __half*>(beta),
                        seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("gamma"), py::arg("beta"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // QKV Split + RoPE + KV Cache (FP16, matches pi05 qkv_split_rope_kvcache_k)
    m.def("qkv_split_rope_kvcache_fp16", [](uintptr_t qkv, uintptr_t rope,
                                              uintptr_t Q, uintptr_t Kc, uintptr_t Vc,
                                              int S, int Q_dim, int K_dim, int HD, int qkv_stride,
                                              int kc_offset, int kc_stride, uintptr_t stream) {
        qkv_split_rope_kvcache_fp16(reinterpret_cast<const __half*>(qkv),
                                     reinterpret_cast<const __half*>(rope),
                                     reinterpret_cast<__half*>(Q),
                                     reinterpret_cast<__half*>(Kc),
                                     reinterpret_cast<__half*>(Vc),
                                     S, Q_dim, K_dim, HD, qkv_stride,
                                     kc_offset, kc_stride, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope"), py::arg("Q"), py::arg("Kc"), py::arg("Vc"),
       py::arg("S"), py::arg("Q_dim"), py::arg("K_dim"), py::arg("HD"), py::arg("qkv_stride"),
       py::arg("kc_offset"), py::arg("kc_stride"), py::arg("stream") = 0);

    // Elementwise FP16
    m.def("bias_residual_fp16", [](uintptr_t residual, uintptr_t x, uintptr_t bias,
                                    int seq_len, int dim, uintptr_t stream) {
        bias_residual_fp16(reinterpret_cast<__half*>(residual), reinterpret_cast<const __half*>(x),
                            reinterpret_cast<const __half*>(bias), seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("residual_add_fp16", [](uintptr_t residual, uintptr_t x, int n, uintptr_t stream) {
        residual_add_fp16(reinterpret_cast<__half*>(residual), reinterpret_cast<const __half*>(x),
                           n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("cfg_combine_into_residual_fp16",
          [](uintptr_t residual, uintptr_t v_cond, uintptr_t v_uncond,
             float beta, int n, uintptr_t stream) {
        cfg_combine_into_residual_fp16(reinterpret_cast<__half*>(residual),
                                       reinterpret_cast<const __half*>(v_cond),
                                       reinterpret_cast<const __half*>(v_uncond),
                                       beta, n, to_stream(stream));
    }, py::arg("residual"), py::arg("v_cond"), py::arg("v_uncond"),
       py::arg("beta"), py::arg("n"), py::arg("stream") = 0);

    // Activation FP16
    m.def("gelu_inplace_fp16", [](uintptr_t x, int n, uintptr_t stream) {
        gelu_inplace_fp16(reinterpret_cast<__half*>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("gate_geglu_merged_fp16", [](uintptr_t merged, uintptr_t out,
                                        int seq, int half_dim, uintptr_t stream) {
        gate_silu_mul_merged_fp16(reinterpret_cast<const __half*>(merged),
                                   reinterpret_cast<__half*>(out), seq, half_dim, to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"), py::arg("stream") = 0);

    // Merged GEGLU (tanh-approx GELU) → FP8 (FP16 input, matches pi05 FFN quant path)
    m.def("gate_geglu_merged_fp8_fp16", [](uintptr_t merged, uintptr_t out,
                                            int seq, int half_dim,
                                            uintptr_t d_scale, uintptr_t stream) {
        gate_silu_mul_merged_fp8_fp16(reinterpret_cast<const __half*>(merged),
                                       typed_ptr<__nv_fp8_e4m3>(out), seq, half_dim,
                                       reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"),
       py::arg("d_scale"), py::arg("stream") = 0);

    // Norm FP16 → FP8 (fused, with scale)
    m.def("rms_norm_fp8_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                                    int seq_len, int dim, float eps,
                                    uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(weight),
                           typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, eps,
                           reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // RMSNorm → FP8 without weight (FP16, verbatim production rms_norm_fp8_static_k)
    m.def("rms_norm_fp8_noweight_fp16", [](uintptr_t x, uintptr_t out,
                                            int seq_len, int dim,
                                            uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8_noweight_fp16(reinterpret_cast<const __half*>(x),
                                    typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim,
                                    reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"),
       py::arg("d_scale"), py::arg("stream") = 0);

    // Residual + RMSNorm → FP8 without weight (FP16, matches production res_rms_fp8_static_k)
    m.def("residual_add_rms_norm_fp8_noweight_fp16", [](uintptr_t residual, uintptr_t x,
                                                          uintptr_t out,
                                                          int seq_len, int dim,
                                                          uintptr_t d_scale, uintptr_t stream) {
        residual_add_rms_norm_fp8_noweight_fp16(reinterpret_cast<__half*>(residual),
                                                  reinterpret_cast<const __half*>(x),
                                                  typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim,
                                                  reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"),
       py::arg("d_scale"), py::arg("stream") = 0);

    // Residual + RMSNorm → FP8 (FP16)
    m.def("residual_add_rms_norm_fp8_fp16", [](uintptr_t residual, uintptr_t x,
                                                 uintptr_t weight, uintptr_t out,
                                                 int seq_len, int dim, float eps,
                                                 uintptr_t d_scale, uintptr_t stream) {
        residual_add_rms_norm_fp8_fp16(reinterpret_cast<__half*>(residual),
                                        reinterpret_cast<const __half*>(x),
                                        reinterpret_cast<const __half*>(weight),
                                        typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, eps,
                                        reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // Split SiLU → FP8 (FP16, separate gate/up)
    m.def("silu_mul_split_fp8_fp16", [](uintptr_t gate, uintptr_t up, uintptr_t out,
                                         int n, uintptr_t d_scale, uintptr_t stream) {
        silu_mul_split_fp8_fp16(reinterpret_cast<const __half*>(gate),
                                 reinterpret_cast<const __half*>(up),
                                 typed_ptr<__nv_fp8_e4m3>(out), n,
                                 reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("gate"), py::arg("up"), py::arg("out"),
       py::arg("n"), py::arg("d_scale"), py::arg("stream") = 0);

    // ── Production-exact kernels (no weight, no scale) ──

    // RMSNorm → FP8 (no weight, no d_scale). Matches pi05 fused_rms_fp8.
    m.def("plain_rms_fp8_fp16", [](uintptr_t x, uintptr_t out,
                                     int seq_len, int dim, uintptr_t stream) {
        plain_rms_fp8_fp16(reinterpret_cast<const __half*>(x),
                            typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
       py::arg("stream") = 0);

    // Residual + RMSNorm → FP8 (no weight, no d_scale). Matches pi05 res_rms_fp8_k.
    m.def("plain_res_rms_fp8_fp16", [](uintptr_t residual, uintptr_t x,
                                         uintptr_t out, int seq_len, int dim,
                                         uintptr_t stream) {
        plain_res_rms_fp8_fp16(reinterpret_cast<__half*>(residual),
                                reinterpret_cast<const __half*>(x),
                                typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    // Cast FP16 → FP8 (no scale). Matches pi05 cast_fp16_fp8_k.
    m.def("cast_fp16_fp8", [](uintptr_t input, uintptr_t output, int n, uintptr_t stream) {
        cast_fp16_fp8(reinterpret_cast<const __half*>(input),
                       typed_ptr<__nv_fp8_e4m3>(output), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("n"), py::arg("stream") = 0);

    // Quantize FP16→FP8
    m.def("quantize_fp8_static_fp16", [](uintptr_t input, uintptr_t output,
                                          uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_static_fp16(reinterpret_cast<const __half*>(input),
                                  typed_ptr<__nv_fp8_e4m3>(output),
                                  reinterpret_cast<const float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    // ── Decoder fused kernels (FP16, matching pi05 ae_forward_static) ──
    m.def("fused_adarms_fp8_static_fp16", [](uintptr_t x, uintptr_t style,
            uintptr_t out, uintptr_t gate_out, int S, int D, uintptr_t descale, uintptr_t stream) {
        fused_adarms_fp8_static_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(style),
            typed_ptr<__nv_fp8_e4m3>(out), reinterpret_cast<__half*>(gate_out),
            S, D, reinterpret_cast<const float*>(descale), to_stream(stream));
    }, py::arg("x"), py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("S"), py::arg("D"), py::arg("descale"), py::arg("stream") = 0);

    m.def("gate_res_adarms_fp8_static_fp16", [](uintptr_t gemm_out, uintptr_t prev_gate,
            uintptr_t residual, uintptr_t style, uintptr_t fp8_out, uintptr_t gate_out,
            int S, int D, uintptr_t descale, uintptr_t stream) {
        gate_res_adarms_fp8_static_fp16(reinterpret_cast<const __half*>(gemm_out),
            reinterpret_cast<const __half*>(prev_gate), reinterpret_cast<__half*>(residual),
            reinterpret_cast<const __half*>(style), typed_ptr<__nv_fp8_e4m3>(fp8_out),
            reinterpret_cast<__half*>(gate_out), S, D, reinterpret_cast<const float*>(descale), to_stream(stream));
    }, py::arg("gemm_out"), py::arg("prev_gate"), py::arg("residual"), py::arg("style"),
       py::arg("fp8_out"), py::arg("gate_out"), py::arg("S"), py::arg("D"), py::arg("descale"), py::arg("stream") = 0);

    m.def("geglu_fp8_static_fp16", [](uintptr_t merged, uintptr_t out, int S, int H,
            uintptr_t descale, uintptr_t stream) {
        geglu_fp8_static_fp16(reinterpret_cast<const __half*>(merged), typed_ptr<__nv_fp8_e4m3>(out),
            S, H, reinterpret_cast<const float*>(descale), to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("S"), py::arg("H"), py::arg("descale"), py::arg("stream") = 0);

    m.def("gate_res_fp16", [](uintptr_t gemm_out, uintptr_t gate, uintptr_t residual, int n, uintptr_t stream) {
        gate_res_fp16(reinterpret_cast<const __half*>(gemm_out), reinterpret_cast<const __half*>(gate),
            reinterpret_cast<__half*>(residual), n, to_stream(stream));
    }, py::arg("gemm_out"), py::arg("gate"), py::arg("residual"), py::arg("n"), py::arg("stream") = 0);

    m.def("adarms_fp16", [](uintptr_t x, uintptr_t style, uintptr_t out, uintptr_t gate_out,
            int S, int D, uintptr_t stream) {
        adarms_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(style),
            reinterpret_cast<__half*>(out), reinterpret_cast<__half*>(gate_out), S, D, to_stream(stream));
    }, py::arg("x"), py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("S"), py::arg("D"), py::arg("stream") = 0);

    // Simple bias add (pi05 bias_k)
    m.def("add_bias_fp16", [](uintptr_t x, uintptr_t b, int S, int D, uintptr_t stream) {
        add_bias_fp16(reinterpret_cast<__half*>(x), reinterpret_cast<const __half*>(b),
                       S, D, to_stream(stream));
    }, py::arg("x"), py::arg("b"), py::arg("S"), py::arg("D"), py::arg("stream") = 0);

    // cuBLAS NN GEMM: C = A @ B + beta * C (pi05 gmm)
    // Requires FvkContext for cuBLAS handle.
    m.def("gmm_fp16", [](FvkContext& ctx, uintptr_t A, uintptr_t B, uintptr_t C,
                           int M, int N, int K, float beta, uintptr_t stream) {
        gmm_fp16(ctx.cublas_handle,
                  reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B),
                  reinterpret_cast<__half*>(C), M, N, K, beta, to_stream(stream));
    }, py::arg("ctx"), py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("beta") = 0.0f, py::arg("stream") = 0);

    // FP8 GEMM with device descale → FP16 output (pi05 gmm_fp8_kn_descale)
    m.def("fp8_gemm_descale_fp16", [](uintptr_t A, uintptr_t B, uintptr_t C,
            int M, int N, int K, uintptr_t act_descale, uintptr_t w_descale, uintptr_t stream) {
        fp8_gemm_descale_fp16(to_ptr(A), to_ptr(B), to_ptr(C), M, N, K,
            reinterpret_cast<const float*>(act_descale), reinterpret_cast<const float*>(w_descale),
            to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0);

    // FP8 GEMM with device descale → FP32 output (for models with activations > FP16 range)
    m.def("fp8_gemm_descale_f32out", [](uintptr_t A, uintptr_t B, uintptr_t C,
            int M, int N, int K, uintptr_t act_descale, uintptr_t w_descale, uintptr_t stream) {
        fp8_gemm_descale_f32out(to_ptr(A), to_ptr(B), to_ptr(C), M, N, K,
            reinterpret_cast<const float*>(act_descale), reinterpret_cast<const float*>(w_descale),
            to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0);

    // FP8 GEMM with device descale → BF16 output (Pi0-FAST decode_step path)
    m.def("fp8_gemm_descale_bf16out", [](uintptr_t A, uintptr_t B, uintptr_t C,
            int M, int N, int K, uintptr_t act_descale, uintptr_t w_descale, uintptr_t stream) {
        fp8_gemm_descale_bf16out(to_ptr(A), to_ptr(B), to_ptr(C), M, N, K,
            reinterpret_cast<const float*>(act_descale), reinterpret_cast<const float*>(w_descale),
            to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0);

    // cuBLAS decomposed attention (GQA, matching pi05 engine)
    // Requires FvkContext for cuBLAS handle.
    m.def("attention_qkv_fp16", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S, int S_kv, int NH, int HD,
                                    float attn_scale, uintptr_t stream) {
        attention_qkv_fp16(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S, S_kv, NH, HD, attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    // Padded attention: supports odd S_kv (pads logits lda to even).
    // logits buffer must have room for S*NH * (S_kv + S_kv%2) elements.
    m.def("attention_qkv_fp16_padded", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S, int S_kv, int NH, int HD,
                                    float attn_scale, uintptr_t stream) {
        attention_qkv_fp16_padded(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S, S_kv, NH, HD, attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    // State-masked attention: single call with AR mask for Pi0 state token.
    m.def("attention_qkv_fp16_state_masked", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S, int S_kv, int NH, int HD,
                                    int state_nk, float attn_scale, uintptr_t stream) {
        attention_qkv_fp16_state_masked(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S, S_kv, NH, HD, state_nk, attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("state_nk"), py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    m.def("softmax_fp16", [](uintptr_t data, int rows, int cols, uintptr_t stream) {
        softmax_fp16(reinterpret_cast<__half*>(data), rows, cols, to_stream(stream));
    }, py::arg("data"), py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    // ── CUTLASS FP8 GEMMs (SM100/SM110, pi05-equivalent tile configs) ──
#ifdef ENABLE_SM100_CUTLASS
    m.def("cutlass_fp8_sq", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                 int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_sq(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_t1", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                 int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_t1(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_wide", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                   int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_wide(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_plain", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                    int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_plain(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_gelu", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                   int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_gelu(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    // FP32 output variants — for models with activations exceeding FP16 range
    m.def("cutlass_fp8_sq_f32out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                       int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_sq_f32out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_wide_f32out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                         int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_wide_f32out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    // BF16 output variants — for models trained in BF16 with large activations
    m.def("cutlass_fp8_sq_bf16out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_sq_bf16out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_wide_bf16out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                          int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_wide_bf16out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_t1_bf16out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_t1_bf16out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("has_cutlass_sm100", []() { return true; });
#else
    m.def("has_cutlass_sm100", []() { return false; });
#endif

    // BF16 noweight norm kernels (always available, not SM100-gated)
    m.def("rms_norm_fp8_noweight_bf16", [](uintptr_t x, uintptr_t out,
            int seq_len, int dim, uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8_noweight_bf16(reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_fp8_e4m3*>(out), seq_len, dim,
            reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
       py::arg("d_scale"), py::arg("stream") = 0);

    m.def("residual_add_rms_norm_fp8_noweight_bf16", [](uintptr_t residual, uintptr_t x,
            uintptr_t out, int seq_len, int dim, uintptr_t d_scale, uintptr_t stream) {
        residual_add_rms_norm_fp8_noweight_bf16(reinterpret_cast<__nv_bfloat16*>(residual),
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_fp8_e4m3*>(out),
            seq_len, dim, reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("d_scale"), py::arg("stream") = 0);

    // Hardware info
    m.def("get_sm_version", []() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        return prop.major * 10 + prop.minor;
    });

#ifdef ENABLE_NVFP4
    m.def("has_nvfp4", []() { return true; });
#else
    m.def("has_nvfp4", []() { return false; });
#endif

    // ── Attention dispatch ──
    m.def("load_fmha_library", [](const std::string& path) {
        return load_fmha_library(path.c_str());
    }, py::arg("path"));

    m.def("load_fmha_strided_library", [](const std::string& path) {
        return load_fmha_strided_library(path.c_str());
    }, py::arg("path"));

    m.def("has_cutlass_fmha", &has_cutlass_fmha);

    m.def("fmha_forward", [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
                              int seq_q, int seq_kv, int num_heads, int head_dim,
                              float scale, uintptr_t stream) {
        return fmha_forward(typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                            typed_ptr<__nv_bfloat16>(V), typed_ptr<__nv_bfloat16>(O),
                            seq_q, seq_kv, num_heads, head_dim, scale, to_stream(stream));
    }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
       py::arg("seq_q"), py::arg("seq_kv"), py::arg("num_heads"), py::arg("head_dim"),
       py::arg("scale") = 1.0f, py::arg("stream") = 0);

    m.def("fmha_strided_forward", [](uintptr_t qkv_buf, uintptr_t O,
                                      int seq, int num_heads, int head_dim,
                                      float scale, uintptr_t stream) {
        return fmha_strided_forward(typed_ptr<__nv_bfloat16>(qkv_buf),
                                     typed_ptr<__nv_bfloat16>(O),
                                     seq, num_heads, head_dim, scale, to_stream(stream));
    }, py::arg("qkv_buf"), py::arg("O"),
       py::arg("seq"), py::arg("num_heads"), py::arg("head_dim"),
       py::arg("scale") = 1.0f, py::arg("stream") = 0);

    // Full strided FMHA: Q/K/V separate pointers + batch + strides
    // Used by SigLIP multi-view: batch=NV, seq=256, stride=3*D
    m.def("fmha_strided_full", [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
                                   int batch, int seq_q, int seq_kv,
                                   int nheads_q, int nheads_kv, int head_dim,
                                   int stride_q, int stride_kv, uintptr_t stream) {
        return fmha_strided_full(to_ptr(Q), to_ptr(K), to_ptr(V), to_ptr(O),
                                  batch, seq_q, seq_kv, nheads_q, nheads_kv, head_dim,
                                  stride_q, stride_kv, to_stream(stream));
    }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
       py::arg("batch"), py::arg("seq_q"), py::arg("seq_kv"),
       py::arg("nheads_q"), py::arg("nheads_kv"), py::arg("head_dim"),
       py::arg("stride_q"), py::arg("stride_kv"), py::arg("stream") = 0);

    // ── DiT kernels (GROOT N1.6) ──

    // LayerNorm without affine parameters (elementwise_affine=False)
    m.def("layer_norm_no_affine_fp16", [](uintptr_t x, uintptr_t out,
                                           int seq_len, int dim, float eps, uintptr_t stream) {
        extern void layer_norm_no_affine_fp16(const __half*, __half*, int, int, float, cudaStream_t);
        layer_norm_no_affine_fp16(reinterpret_cast<const __half*>(x),
                                   reinterpret_cast<__half*>(out),
                                   seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-5f, py::arg("stream") = 0);

    // Fused AdaLayerNorm: LN(x, no_affine) * (1 + scale) + shift
    m.def("ada_layer_norm_fp16", [](uintptr_t x, uintptr_t scale, uintptr_t shift,
                                     uintptr_t out, int seq_len, int dim, float eps, uintptr_t stream) {
        extern void ada_layer_norm_fp16(const __half*, const __half*, const __half*,
                                         __half*, int, int, float, cudaStream_t);
        ada_layer_norm_fp16(reinterpret_cast<const __half*>(x),
                             reinterpret_cast<const __half*>(scale),
                             reinterpret_cast<const __half*>(shift),
                             reinterpret_cast<__half*>(out),
                             seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("scale"), py::arg("shift"),
       py::arg("out"), py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-5f, py::arg("stream") = 0);

    // ── GPU memory ops (CUDA Graph compatible, explicit stream) ──
    m.def("gpu_copy", [](uintptr_t dst, uintptr_t src, int nbytes, uintptr_t stream) {
        extern void gpu_copy_async(void*, const void*, size_t, cudaStream_t);
        gpu_copy_async(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
                        nbytes, to_stream(stream));
    }, py::arg("dst"), py::arg("src"), py::arg("nbytes"), py::arg("stream") = 0);

    m.def("gpu_fill_neginf_fp16", [](uintptr_t dst, int n, uintptr_t stream) {
        extern void gpu_fill_neginf_fp16(__half*, int, cudaStream_t);
        gpu_fill_neginf_fp16(reinterpret_cast<__half*>(dst), n, to_stream(stream));
    }, py::arg("dst"), py::arg("n"), py::arg("stream") = 0);

    m.def("gpu_strided_copy_fp16", [](uintptr_t src, uintptr_t dst,
                                       int rows, int dst_cols, int src_stride, int col_offset,
                                       uintptr_t stream) {
        extern void gpu_strided_copy_fp16(const __half*, __half*, int, int, int, int, cudaStream_t);
        gpu_strided_copy_fp16(reinterpret_cast<const __half*>(src), reinterpret_cast<__half*>(dst),
                               rows, dst_cols, src_stride, col_offset, to_stream(stream));
    }, py::arg("src"), py::arg("dst"),
       py::arg("rows"), py::arg("dst_cols"), py::arg("src_stride"), py::arg("col_offset"),
       py::arg("stream") = 0);

    m.def("gpu_cast_fp32_to_fp16", [](uintptr_t src, uintptr_t dst, int n, uintptr_t stream) {
        extern void gpu_cast_fp32_to_fp16(const float*, __half*, int, cudaStream_t);
        gpu_cast_fp32_to_fp16(reinterpret_cast<const float*>(src),
                               reinterpret_cast<__half*>(dst), n, to_stream(stream));
    }, py::arg("src"), py::arg("dst"), py::arg("n"), py::arg("stream") = 0);

    m.def("gpu_euler_step", [](uintptr_t actions, uintptr_t velocity,
                                int T, int action_dim, float dt, int vel_elem_offset,
                                uintptr_t stream) {
        extern void gpu_euler_step(float*, const __half*, int, int, float, int, cudaStream_t);
        gpu_euler_step(reinterpret_cast<float*>(actions),
                        reinterpret_cast<const __half*>(velocity),
                        T, action_dim, dt, vel_elem_offset, to_stream(stream));
    }, py::arg("actions"), py::arg("velocity"),
       py::arg("T"), py::arg("action_dim"), py::arg("dt"), py::arg("vel_elem_offset"),
       py::arg("stream") = 0);

    // SiLU in-place FP16 (for DiT action encoder)
    m.def("silu_inplace_fp16", [](uintptr_t x, int n, uintptr_t stream) {
        extern void silu_inplace_fp16(__half*, int, cudaStream_t);
        silu_inplace_fp16(reinterpret_cast<__half*>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    // Fused add + SiLU in-place: a = silu(a + b). Used by Pi0 action_time_mlp.
    m.def("fused_add_silu_fp16", [](uintptr_t a, uintptr_t b, int n, uintptr_t stream) {
        extern void fused_add_silu_fp16(__half*, const __half*, int, cudaStream_t);
        fused_add_silu_fp16(reinterpret_cast<__half*>(a),
                            reinterpret_cast<const __half*>(b), n, to_stream(stream));
    }, py::arg("a"), py::arg("b"), py::arg("n"), py::arg("stream") = 0);

    m.def("fused_add_silu_bf16", [](uintptr_t a, uintptr_t b, int n, uintptr_t stream) {
        extern void fused_add_silu_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, cudaStream_t);
        fused_add_silu_bf16(reinterpret_cast<__nv_bfloat16*>(a),
                            reinterpret_cast<const __nv_bfloat16*>(b), n, to_stream(stream));
    }, py::arg("a"), py::arg("b"), py::arg("n"), py::arg("stream") = 0);

    // ReLU in-place FP16 (for DiT action decoder)
    m.def("relu_inplace_fp16", [](uintptr_t x, int n, uintptr_t stream) {
        extern void relu_inplace_fp16(__half*, int, cudaStream_t);
        relu_inplace_fp16(reinterpret_cast<__half*>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    // GQA KV repeat interleave (for Qwen3 8→16 heads)
    m.def("gpu_repeat_interleave_heads", [](uintptr_t src, uintptr_t dst,
                                             int S, int NH_src, int HD, int repeat, uintptr_t stream) {
        extern void gpu_repeat_interleave_heads(const __half*, __half*, int, int, int, int, cudaStream_t);
        gpu_repeat_interleave_heads(reinterpret_cast<const __half*>(src),
                                     reinterpret_cast<__half*>(dst),
                                     S, NH_src, HD, repeat, to_stream(stream));
    }, py::arg("src"), py::arg("dst"),
       py::arg("S"), py::arg("NH_src"), py::arg("HD"), py::arg("repeat"),
       py::arg("stream") = 0);

    // Qwen3 RoPE (rotate_half style, in-place)
    m.def("rope_rotate_half_fp16", [](uintptr_t x, uintptr_t cos_table, uintptr_t sin_table,
                                       int S, int NH, int HD, uintptr_t stream) {
        extern void rope_rotate_half_fp16(__half*, const __half*, const __half*, int, int, int, cudaStream_t);
        rope_rotate_half_fp16(reinterpret_cast<__half*>(x),
                               reinterpret_cast<const __half*>(cos_table),
                               reinterpret_cast<const __half*>(sin_table),
                               S, NH, HD, to_stream(stream));
    }, py::arg("x"), py::arg("cos_table"), py::arg("sin_table"),
       py::arg("S"), py::arg("NH"), py::arg("HD"), py::arg("stream") = 0);

    // MHA batched cuBLAS attention (for DiT — per-head independent attention)
    m.def("attention_mha_fp16", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S_q, int S_kv, int NH, int HD,
                                    float attn_scale, uintptr_t stream) {
        extern void attention_mha_fp16(cublasHandle_t, const __half*, const __half*, const __half*,
                                        __half*, __half*, int, int, int, int, float, cudaStream_t);
        attention_mha_fp16(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S_q, S_kv, NH, HD, attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S_q"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    // Causal MHA — N1.7 Qwen3-VL LLM (is_causal=True per HF). Same layout
    // and cuBLAS path as attention_mha_fp16; differs only in the softmax
    // step (strict upper-triangular mask via softmax_causal_fp16).
    m.def("attention_mha_causal_fp16",
          [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
             uintptr_t logits, uintptr_t out,
             int S_q, int S_kv, int NH, int HD,
             float attn_scale, uintptr_t stream) {
        extern void attention_mha_causal_fp16(
            cublasHandle_t, const __half*, const __half*, const __half*,
            __half*, __half*, int, int, int, int, float, cudaStream_t);
        attention_mha_causal_fp16(ctx.cublas_handle,
                                   reinterpret_cast<const __half*>(Q),
                                   reinterpret_cast<const __half*>(K),
                                   reinterpret_cast<const __half*>(V),
                                   reinterpret_cast<__half*>(logits),
                                   reinterpret_cast<__half*>(out),
                                   S_q, S_kv, NH, HD, attn_scale,
                                   to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S_q"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    // FA2 bindings (fvk.attention_fa2_fwd_fp16/bf16) moved to a separate
    // pybind module flash_rt_fa2.so — see csrc/fa2_bindings.cpp. This
    // keeps flash_rt_kernels.so small and fast to rebuild.

    // ── DiT bf16 helpers (Phase 5a-2) ────────────────────────────────
    m.def("layer_norm_no_affine_bf16",
          [](uintptr_t x, uintptr_t out, int seq_len, int dim, float eps,
             uintptr_t stream) {
        extern void layer_norm_no_affine_bf16(
            const __nv_bfloat16*, __nv_bfloat16*, int, int, float, cudaStream_t);
        layer_norm_no_affine_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(out),
            seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-5f, py::arg("stream") = 0);

    m.def("ada_layer_norm_bf16",
          [](uintptr_t x, uintptr_t scale, uintptr_t shift, uintptr_t out,
             int seq_len, int dim, float eps, uintptr_t stream) {
        extern void ada_layer_norm_bf16(
            const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
            __nv_bfloat16*, int, int, float, cudaStream_t);
        ada_layer_norm_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(scale),
            reinterpret_cast<const __nv_bfloat16*>(shift),
            reinterpret_cast<__nv_bfloat16*>(out),
            seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("scale"), py::arg("shift"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-5f, py::arg("stream") = 0);

    m.def("add_bias_bf16",
          [](uintptr_t x, uintptr_t b, int S, int D, uintptr_t stream) {
        extern void add_bias_bf16(
            __nv_bfloat16*, const __nv_bfloat16*, int, int, cudaStream_t);
        add_bias_bf16(reinterpret_cast<__nv_bfloat16*>(x),
                       reinterpret_cast<const __nv_bfloat16*>(b),
                       S, D, to_stream(stream));
    }, py::arg("x"), py::arg("b"), py::arg("S"), py::arg("D"),
       py::arg("stream") = 0);

    m.def("cast_fp16_to_bf16",
          [](uintptr_t in, uintptr_t out, int n, uintptr_t stream) {
        extern void cast_fp16_to_bf16(
            const __half*, __nv_bfloat16*, int, cudaStream_t);
        cast_fp16_to_bf16(reinterpret_cast<const __half*>(in),
                           reinterpret_cast<__nv_bfloat16*>(out),
                           n, to_stream(stream));
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    m.def("cast_bf16_to_fp16",
          [](uintptr_t in, uintptr_t out, int n, uintptr_t stream) {
        extern void cast_bf16_to_fp16(
            const __nv_bfloat16*, __half*, int, cudaStream_t);
        cast_bf16_to_fp16(reinterpret_cast<const __nv_bfloat16*>(in),
                           reinterpret_cast<__half*>(out),
                           n, to_stream(stream));
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    // ── DiT bf16 attention path (Phase 5a-2) ─────────────────────────
    m.def("softmax_bf16", [](uintptr_t data, int rows, int cols,
                              uintptr_t stream) {
        extern void softmax_bf16(__nv_bfloat16*, int, int, cudaStream_t);
        softmax_bf16(reinterpret_cast<__nv_bfloat16*>(data),
                      rows, cols, to_stream(stream));
    }, py::arg("data"), py::arg("rows"), py::arg("cols"),
       py::arg("stream") = 0);

    m.def("gpu_fill_neginf_bf16", [](uintptr_t x, int n, uintptr_t stream) {
        extern void gpu_fill_neginf_bf16(__nv_bfloat16*, int, cudaStream_t);
        gpu_fill_neginf_bf16(reinterpret_cast<__nv_bfloat16*>(x),
                              n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("attention_mha_bf16",
          [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
             uintptr_t logits, uintptr_t out,
             int S_q, int S_kv, int NH, int HD,
             float attn_scale, int logits_kv_stride, uintptr_t stream) {
        extern void attention_mha_bf16(
            cublasHandle_t, const __nv_bfloat16*, const __nv_bfloat16*,
            const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
            int, int, int, int, float, int, cudaStream_t);
        attention_mha_bf16(ctx.cublas_handle,
                            reinterpret_cast<const __nv_bfloat16*>(Q),
                            reinterpret_cast<const __nv_bfloat16*>(K),
                            reinterpret_cast<const __nv_bfloat16*>(V),
                            reinterpret_cast<__nv_bfloat16*>(logits),
                            reinterpret_cast<__nv_bfloat16*>(out),
                            S_q, S_kv, NH, HD, attn_scale,
                            logits_kv_stride, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S_q"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("logits_kv_stride") = 0,
       py::arg("stream") = 0);

    // ------------------------------------------------------------------
    //  FP8 block-128 dequantization + GEMM (Phase 2.2 / Path D)
    //  Used by Qwen3.6-27B; see internal-docs/qwen36_fp8_block128_gemm_design.md
    //  All entries are additive — existing fp8_gemm_descale_* untouched.
    // ------------------------------------------------------------------
    m.def("fp8_block128_dequantize_to_bf16",
        [](uintptr_t in_fp8, uintptr_t scale, uintptr_t out_bf16,
           int N, int K, uintptr_t stream) {
            flash_rt::quantize::fp8_block128_dequantize_to_bf16(
                to_ptr(in_fp8),
                reinterpret_cast<const float*>(scale),
                to_ptr(out_bf16),
                N, K, to_stream(stream));
        },
        py::arg("in_fp8"), py::arg("scale"), py::arg("out_bf16"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // Single-shot FP8 block-128 -> NVFP4 (swizzled SF + per-tensor global).
    // Replaces the lossy two-step (dequantize_to_bf16 + bf16_to_nvfp4_swizzled)
    // for weight tensors — see csrc/quantize/fp8_block128_to_nvfp4_swizzled.cuh
    // for the precision rationale (no BF16 mantissa truncation, proper UE4M3
    // SF range via per-tensor global_scale).
    //
    // Caller pre-allocates: nvfp4_packed (N, K/2 u8), nvfp4_sf_swizzled
    // (nvfp4_sf_swizzled_bytes(N, K) u8, zeroed), scratch_global_amax (1 fp32),
    // out_global_scale (1 fp32). out_global_scale is to be passed as the GEMM
    // alpha (= act_global * w_global; for per-token activation quant
    // act_global = 1 so alpha = w_global = out_global_scale).
    m.def("fp8_block128_to_nvfp4_swizzled_bf16",
        [](uintptr_t w_fp8, uintptr_t w_block_scale_fp32,
           uintptr_t nvfp4_packed, uintptr_t nvfp4_sf_swizzled,
           uintptr_t scratch_global_amax, uintptr_t out_global_scale,
           int N, int K, uintptr_t stream) {
            flash_rt::quantize::fp8_block128_to_nvfp4_swizzled_bf16(
                to_ptr(w_fp8),
                reinterpret_cast<const float*>(w_block_scale_fp32),
                reinterpret_cast<uint8_t*>(nvfp4_packed),
                reinterpret_cast<uint8_t*>(nvfp4_sf_swizzled),
                reinterpret_cast<float*>(scratch_global_amax),
                reinterpret_cast<float*>(out_global_scale),
                N, K, to_stream(stream));
        },
        py::arg("w_fp8"), py::arg("w_block_scale_fp32"),
        py::arg("nvfp4_packed"), py::arg("nvfp4_sf_swizzled"),
        py::arg("scratch_global_amax"), py::arg("out_global_scale"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // Single-shot BF16 weight -> NVFP4 (swizzled SF + per-tensor global).
    // For weights that arrive as plain BF16 (e.g. the 30% of Qwen3.6
    // NVFP4 ckpt weights left unquantized: lin-attn in_proj_qkv / z /
    // out_proj). Same alpha = out_global_scale contract as the FP8
    // sibling.
    m.def("bf16_weight_to_nvfp4_swizzled",
        [](uintptr_t w_bf16,
           uintptr_t nvfp4_packed, uintptr_t nvfp4_sf_swizzled,
           uintptr_t scratch_global_amax, uintptr_t out_global_scale,
           int N, int K, uintptr_t stream) {
            flash_rt::quantize::bf16_weight_to_nvfp4_swizzled(
                typed_ptr<__nv_bfloat16>(w_bf16),
                reinterpret_cast<uint8_t*>(nvfp4_packed),
                reinterpret_cast<uint8_t*>(nvfp4_sf_swizzled),
                reinterpret_cast<float*>(scratch_global_amax),
                reinterpret_cast<float*>(out_global_scale),
                N, K, to_stream(stream));
        },
        py::arg("w_bf16"),
        py::arg("nvfp4_packed"), py::arg("nvfp4_sf_swizzled"),
        py::arg("scratch_global_amax"), py::arg("out_global_scale"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // ── TurboQuant unpack (Phase 3A B9 step S3) ───────────────────────
    // Packed B8 (4-bit K idx + 1-bit qjl + 4-bit V idx) → 3 BF16 outputs:
    // y_k, qjl_bf, y_v (each shape (M=S*4, 256)).  Caller follows up
    // with two cuBLAS bf16 GEMMs (rotation, jl) and the combine kernel.
    // Phase 3B-α 3.5b: mixed unpack (yk/yv bf16, qjl fp32) — fuses
    // the bf16→fp32 qjl cast into the unpack, saves ~192 MB BW/call
    // at 32K vs the bf16-then-cast pattern.
    m.def("tq_unpack_packed_mixed",
        [](uintptr_t k_idx_packed, uintptr_t k_qjl_packed,
           uintptr_t v_idx_packed,
           uintptr_t cb_k_mse, uintptr_t cb_v,
           uintptr_t y_k_bf16, uintptr_t qjl_fp32, uintptr_t y_v_bf16,
           int M, int b_k_mse, int b_v, uintptr_t stream) {
            tq_unpack_packed_mixed_launch(
                to_ptr(k_idx_packed), to_ptr(k_qjl_packed),
                to_ptr(v_idx_packed),
                to_ptr(cb_k_mse), to_ptr(cb_v),
                reinterpret_cast<void*>(y_k_bf16),
                reinterpret_cast<void*>(qjl_fp32),
                reinterpret_cast<void*>(y_v_bf16),
                M, b_k_mse, b_v, to_stream(stream));
        },
        py::arg("k_idx_packed"), py::arg("k_qjl_packed"),
        py::arg("v_idx_packed"),
        py::arg("cb_k_mse"), py::arg("cb_v"),
        py::arg("y_k_bf16"), py::arg("qjl_fp32"), py::arg("y_v_bf16"),
        py::arg("M"), py::arg("b_k_mse"), py::arg("b_v"),
        py::arg("stream") = 0);

    m.def("tq_unpack_packed_bf16",
        [](uintptr_t k_idx_packed, uintptr_t k_qjl_packed,
           uintptr_t v_idx_packed,
           uintptr_t cb_k_mse, uintptr_t cb_v,
           uintptr_t y_k, uintptr_t qjl_bf, uintptr_t y_v,
           int M, int b_k_mse, int b_v, uintptr_t stream) {
            tq_unpack_packed_bf16_launch(
                to_ptr(k_idx_packed), to_ptr(k_qjl_packed),
                to_ptr(v_idx_packed),
                to_ptr(cb_k_mse), to_ptr(cb_v),
                reinterpret_cast<void*>(y_k),
                reinterpret_cast<void*>(qjl_bf),
                reinterpret_cast<void*>(y_v),
                M, b_k_mse, b_v, to_stream(stream));
        },
        py::arg("k_idx_packed"), py::arg("k_qjl_packed"),
        py::arg("v_idx_packed"),
        py::arg("cb_k_mse"), py::arg("cb_v"),
        py::arg("y_k"), py::arg("qjl_bf"), py::arg("y_v"),
        py::arg("M"), py::arg("b_k_mse"), py::arg("b_v"),
        py::arg("stream") = 0);

    // ── TurboQuant combine ────────────────────────────────────────────
    // Element-wise: K = norm·(K_mse + coef·rnorm·K_qjl); V = v_norm·V_unit.
    m.def("tq_combine_kv_bf16",
        [](uintptr_t k_mse, uintptr_t k_qjl, uintptr_t v_unit,
           uintptr_t k_norm, uintptr_t k_rnorm, uintptr_t v_norm,
           uintptr_t k_out, uintptr_t v_out,
           int M, float coef, uintptr_t stream) {
            tq_combine_kv_bf16_launch(
                to_ptr(k_mse), to_ptr(k_qjl), to_ptr(v_unit),
                to_ptr(k_norm), to_ptr(k_rnorm), to_ptr(v_norm),
                reinterpret_cast<void*>(k_out),
                reinterpret_cast<void*>(v_out),
                M, coef, to_stream(stream));
        },
        py::arg("k_mse"), py::arg("k_qjl"), py::arg("v_unit"),
        py::arg("k_norm"), py::arg("k_rnorm"), py::arg("v_norm"),
        py::arg("k_out"), py::arg("v_out"),
        py::arg("M"), py::arg("coef"), py::arg("stream") = 0);

    m.def("tq_write_kv_packed",
        [](uintptr_t k_in, uintptr_t v_in,
           int s_start, int S,
           uintptr_t rotation, uintptr_t jl,
           uintptr_t cb_k_mse, uintptr_t cb_v,
           uintptr_t k_idx_packed_layer, uintptr_t k_qjl_packed_layer,
           uintptr_t k_norm_layer, uintptr_t k_rnorm_layer,
           uintptr_t v_idx_packed_layer, uintptr_t v_norm_layer,
           int b_k_mse, int b_v, uintptr_t stream) {
            tq_write_kv_packed_launch(
                to_ptr(k_in), to_ptr(v_in),
                s_start, S,
                to_ptr(rotation), to_ptr(jl),
                to_ptr(cb_k_mse), to_ptr(cb_v),
                reinterpret_cast<void*>(k_idx_packed_layer),
                reinterpret_cast<void*>(k_qjl_packed_layer),
                reinterpret_cast<void*>(k_norm_layer),
                reinterpret_cast<void*>(k_rnorm_layer),
                reinterpret_cast<void*>(v_idx_packed_layer),
                reinterpret_cast<void*>(v_norm_layer),
                b_k_mse, b_v, to_stream(stream));
        },
        py::arg("k_in"), py::arg("v_in"),
        py::arg("s_start"), py::arg("S"),
        py::arg("rotation"), py::arg("jl"),
        py::arg("cb_k_mse"), py::arg("cb_v"),
        py::arg("k_idx_packed_layer"), py::arg("k_qjl_packed_layer"),
        py::arg("k_norm_layer"), py::arg("k_rnorm_layer"),
        py::arg("v_idx_packed_layer"), py::arg("v_norm_layer"),
        py::arg("b_k_mse"), py::arg("b_v"),
        py::arg("stream") = 0);

    // ── B9-S10: capture-safe write path (K1-K4 + 3 cuBLAS GEMMs) ───
    m.def("tq_write_k1_unit_norm",
        [](uintptr_t k_in, uintptr_t v_in,
           uintptr_t k_unit_out, uintptr_t v_unit_out,
           uintptr_t norm_k_out, uintptr_t norm_v_out,
           int M, int b_k_mse, int b_v, uintptr_t stream) {
            tq_write_k1_unit_norm_launch(
                to_ptr(k_in), to_ptr(v_in),
                reinterpret_cast<void*>(k_unit_out),
                reinterpret_cast<void*>(v_unit_out),
                reinterpret_cast<void*>(norm_k_out),
                reinterpret_cast<void*>(norm_v_out),
                M, b_k_mse, b_v, to_stream(stream));
        },
        py::arg("k_in"), py::arg("v_in"),
        py::arg("k_unit_out"), py::arg("v_unit_out"),
        py::arg("norm_k_out"), py::arg("norm_v_out"),
        py::arg("M"), py::arg("b_k_mse"), py::arg("b_v"),
        py::arg("stream") = 0);

    m.def("tq_write_k2_argmin_pack",
        [](uintptr_t y_k, uintptr_t y_v,
           uintptr_t cb_k_mse, uintptr_t cb_v,
           uintptr_t k_idx_packed_layer, uintptr_t v_idx_packed_layer,
           uintptr_t dq_in,
           int s_start, int num_kv, int M,
           int b_k_mse, int b_v, uintptr_t stream) {
            tq_write_k2_argmin_pack_launch(
                to_ptr(y_k), to_ptr(y_v),
                to_ptr(cb_k_mse), to_ptr(cb_v),
                reinterpret_cast<void*>(k_idx_packed_layer),
                reinterpret_cast<void*>(v_idx_packed_layer),
                reinterpret_cast<void*>(dq_in),
                s_start, num_kv, M,
                b_k_mse, b_v, to_stream(stream));
        },
        py::arg("y_k"), py::arg("y_v"),
        py::arg("cb_k_mse"), py::arg("cb_v"),
        py::arg("k_idx_packed_layer"), py::arg("v_idx_packed_layer"),
        py::arg("dq_in"),
        py::arg("s_start"), py::arg("num_kv"), py::arg("M"),
        py::arg("b_k_mse"), py::arg("b_v"),
        py::arg("stream") = 0);

    m.def("tq_write_k3_residual_rnorm",
        [](uintptr_t k_unit, uintptr_t dq_k,
           uintptr_t residual, uintptr_t rnorm_k,
           int M, uintptr_t stream) {
            tq_write_k3_residual_rnorm_launch(
                to_ptr(k_unit), to_ptr(dq_k),
                reinterpret_cast<void*>(residual),
                reinterpret_cast<void*>(rnorm_k),
                M, to_stream(stream));
        },
        py::arg("k_unit"), py::arg("dq_k"),
        py::arg("residual"), py::arg("rnorm_k"),
        py::arg("M"), py::arg("stream") = 0);

    m.def("tq_write_k4_qjl_norms",
        [](uintptr_t Sr,
           uintptr_t norm_k, uintptr_t rnorm_k, uintptr_t norm_v,
           uintptr_t k_qjl_packed_layer,
           uintptr_t k_norm_layer, uintptr_t k_rnorm_layer,
           uintptr_t v_norm_layer,
           int s_start, int num_kv, int M, uintptr_t stream) {
            tq_write_k4_qjl_norms_launch(
                to_ptr(Sr),
                to_ptr(norm_k), to_ptr(rnorm_k), to_ptr(norm_v),
                reinterpret_cast<void*>(k_qjl_packed_layer),
                reinterpret_cast<void*>(k_norm_layer),
                reinterpret_cast<void*>(k_rnorm_layer),
                reinterpret_cast<void*>(v_norm_layer),
                s_start, num_kv, M, to_stream(stream));
        },
        py::arg("Sr"),
        py::arg("norm_k"), py::arg("rnorm_k"), py::arg("norm_v"),
        py::arg("k_qjl_packed_layer"),
        py::arg("k_norm_layer"), py::arg("k_rnorm_layer"),
        py::arg("v_norm_layer"),
        py::arg("s_start"), py::arg("num_kv"), py::arg("M"),
        py::arg("stream") = 0);

    m.def("tq_unpack_packed_fp32",
        [](uintptr_t k_idx_packed, uintptr_t k_qjl_packed,
           uintptr_t v_idx_packed,
           uintptr_t cb_k_mse, uintptr_t cb_v,
           uintptr_t y_k, uintptr_t qjl_f, uintptr_t y_v,
           int M, int b_k_mse, int b_v, uintptr_t stream) {
            tq_unpack_packed_fp32_launch(
                to_ptr(k_idx_packed), to_ptr(k_qjl_packed),
                to_ptr(v_idx_packed),
                to_ptr(cb_k_mse), to_ptr(cb_v),
                reinterpret_cast<void*>(y_k),
                reinterpret_cast<void*>(qjl_f),
                reinterpret_cast<void*>(y_v),
                M, b_k_mse, b_v, to_stream(stream));
        },
        py::arg("k_idx_packed"), py::arg("k_qjl_packed"),
        py::arg("v_idx_packed"),
        py::arg("cb_k_mse"), py::arg("cb_v"),
        py::arg("y_k"), py::arg("qjl_f"), py::arg("y_v"),
        py::arg("M"), py::arg("b_k_mse"), py::arg("b_v"),
        py::arg("stream") = 0);

    m.def("tq_fp32_gemm_tf32",
        [](uintptr_t a_fp32, uintptr_t b_fp32, uintptr_t c_fp32,
           int M, int N, int K, uintptr_t stream) {
            tq_fp32_gemm_tf32_launch(
                to_ptr(a_fp32), to_ptr(b_fp32),
                reinterpret_cast<void*>(c_fp32),
                M, N, K, to_stream(stream));
        },
        py::arg("a_fp32"), py::arg("b_fp32"), py::arg("c_fp32"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    m.def("tq_bf16_fp32_gemm",
        [](uintptr_t a_bf16, uintptr_t b_bf16, uintptr_t c_fp32,
           int M, int N, int K, uintptr_t stream) {
            tq_bf16_fp32_gemm_launch(
                to_ptr(a_bf16), to_ptr(b_bf16),
                reinterpret_cast<void*>(c_fp32),
                M, N, K, to_stream(stream));
        },
        py::arg("a_bf16"), py::arg("b_bf16"), py::arg("c_fp32"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    m.def("tq_combine_kv_fp32_in",
        [](uintptr_t k_mse, uintptr_t k_qjl, uintptr_t v_unit,
           uintptr_t k_norm, uintptr_t k_rnorm, uintptr_t v_norm,
           uintptr_t k_out, uintptr_t v_out,
           int M, float coef, uintptr_t stream) {
            tq_combine_kv_fp32_in_launch(
                to_ptr(k_mse), to_ptr(k_qjl), to_ptr(v_unit),
                to_ptr(k_norm), to_ptr(k_rnorm), to_ptr(v_norm),
                reinterpret_cast<void*>(k_out),
                reinterpret_cast<void*>(v_out),
                M, coef, to_stream(stream));
        },
        py::arg("k_mse"), py::arg("k_qjl"), py::arg("v_unit"),
        py::arg("k_norm"), py::arg("k_rnorm"), py::arg("v_norm"),
        py::arg("k_out"), py::arg("v_out"),
        py::arg("M"), py::arg("coef"), py::arg("stream") = 0);

    // α probe: CUTLASS bf16×bf16→bf16 GEMM at sm_120.
    m.def("tq_cutlass_bf16_gemm",
        [](uintptr_t a_bf16, uintptr_t b_bf16, uintptr_t d_bf16,
           int M, int N, int K, uintptr_t stream) {
            tq_cutlass_bf16_gemm_launch(
                to_ptr(a_bf16), to_ptr(b_bf16),
                reinterpret_cast<void*>(d_bf16),
                M, N, K, to_stream(stream));
        },
        py::arg("a_bf16"), py::arg("b_bf16"), py::arg("d_bf16"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    // α Phase 2: CUTLASS EVT V combine — D = norm_v[r] * (A @ B), bf16 out.
    m.def("tq_cutlass_v_combine",
        [](uintptr_t a_bf16, uintptr_t b_bf16, uintptr_t norm_v_fp32,
           uintptr_t d_bf16, int M, int N, int K, uintptr_t stream) {
            tq_cutlass_v_combine_launch(
                to_ptr(a_bf16), to_ptr(b_bf16), to_ptr(norm_v_fp32),
                reinterpret_cast<void*>(d_bf16),
                M, N, K, to_stream(stream));
        },
        py::arg("a_bf16"), py::arg("b_bf16"), py::arg("norm_v_fp32"),
        py::arg("d_bf16"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    // α Phase 2: CUTLASS EVT K combine — D = norm_k[r]*(A@B + coef_rnorm[r]*Sr).
    m.def("tq_cutlass_k_combine",
        [](uintptr_t a_bf16, uintptr_t b_bf16, uintptr_t sr_fp32,
           uintptr_t norm_k_fp32, uintptr_t coef_rnorm_fp32,
           uintptr_t d_bf16, int M, int N, int K, uintptr_t stream) {
            tq_cutlass_k_combine_launch(
                to_ptr(a_bf16), to_ptr(b_bf16), to_ptr(sr_fp32),
                to_ptr(norm_k_fp32), to_ptr(coef_rnorm_fp32),
                reinterpret_cast<void*>(d_bf16),
                M, N, K, to_stream(stream));
        },
        py::arg("a_bf16"), py::arg("b_bf16"), py::arg("sr_fp32"),
        py::arg("norm_k_fp32"), py::arg("coef_rnorm_fp32"),
        py::arg("d_bf16"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    // α probe: hand-rolled wmma bf16×bf16→fp32 GEMM kernel.
    m.def("wmma_probe",
        [](uintptr_t a_bf16, uintptr_t b_bf16, uintptr_t c_fp32,
           int M, uintptr_t stream) {
            wmma_probe_launch(
                to_ptr(a_bf16), to_ptr(b_bf16),
                reinterpret_cast<void*>(c_fp32),
                M, to_stream(stream));
        },
        py::arg("a_bf16"), py::arg("b_bf16"), py::arg("c_fp32"),
        py::arg("M"), py::arg("stream") = 0);

    // Phase 3B-α S3: single-launch fused dequant.
    // Replaces unpack + 2× cuBLAS GEMM + combine with one kernel.
    m.def("tq_dequant_kv_fused",
        [](uintptr_t k_idx_packed, uintptr_t k_qjl_packed,
           uintptr_t k_norm, uintptr_t k_rnorm,
           uintptr_t v_idx_packed, uintptr_t v_norm,
           uintptr_t rotation, uintptr_t jl,
           uintptr_t cb_k_mse, uintptr_t cb_v,
           uintptr_t k_out, uintptr_t v_out,
           int M, float coef, int b_k_mse, int b_v, uintptr_t stream) {
            tq_dequant_kv_fused_launch(
                to_ptr(k_idx_packed), to_ptr(k_qjl_packed),
                to_ptr(k_norm), to_ptr(k_rnorm),
                to_ptr(v_idx_packed), to_ptr(v_norm),
                to_ptr(rotation), to_ptr(jl),
                to_ptr(cb_k_mse), to_ptr(cb_v),
                reinterpret_cast<void*>(k_out),
                reinterpret_cast<void*>(v_out),
                M, coef, b_k_mse, b_v, to_stream(stream));
        },
        py::arg("k_idx_packed"), py::arg("k_qjl_packed"),
        py::arg("k_norm"), py::arg("k_rnorm"),
        py::arg("v_idx_packed"), py::arg("v_norm"),
        py::arg("rotation"), py::arg("jl"),
        py::arg("cb_k_mse"), py::arg("cb_v"),
        py::arg("k_out"), py::arg("v_out"),
        py::arg("M"), py::arg("coef"),
        py::arg("b_k_mse"), py::arg("b_v"),
        py::arg("stream") = 0);

    // Per-token x per-128K FP8 quant (replaces HF triton_fp8_act_quant).
    // Pre-allocated output_fp8 + output_scale buffers from caller.
    m.def("fp8_per_token_block128_quant_bf16",
        [](uintptr_t input, uintptr_t output_fp8, uintptr_t output_scale,
           int M, int K, uintptr_t stream) {
            flash_rt::quantize::fp8_per_token_block128_quant_bf16(
                to_ptr(input), to_ptr(output_fp8),
                reinterpret_cast<float*>(output_scale),
                M, K, to_stream(stream));
        },
        py::arg("input"), py::arg("output_fp8"),
        py::arg("output_scale"),
        py::arg("M"), py::arg("K"), py::arg("stream") = 0);

    m.def("fp8_per_token_block128_dequantize_to_bf16",
        [](uintptr_t in_fp8, uintptr_t scale, uintptr_t out_bf16,
           int M, int K, uintptr_t stream) {
            flash_rt::quantize::fp8_per_token_block128_dequantize_to_bf16(
                to_ptr(in_fp8),
                reinterpret_cast<const float*>(scale),
                to_ptr(out_bf16),
                M, K, to_stream(stream));
        },
        py::arg("in_fp8"), py::arg("scale"), py::arg("out_bf16"),
        py::arg("M"), py::arg("K"), py::arg("stream") = 0);

    m.def("fp8_block128_gemm_descale_bf16out",
        [](uintptr_t A, uintptr_t B, uintptr_t D,
           int M, int N, int K,
           uintptr_t act_scale, uintptr_t w_scale,
           uintptr_t scratch_A, uintptr_t scratch_B,
           uintptr_t stream) {
            flash_rt::gemm::fp8_block128_gemm_descale_bf16out(
                to_ptr(A), to_ptr(B), to_ptr(D),
                M, N, K,
                reinterpret_cast<const float*>(act_scale),
                reinterpret_cast<const float*>(w_scale),
                to_ptr(scratch_A), to_ptr(scratch_B),
                to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("act_block_scale"), py::arg("w_block_scale"),
        py::arg("scratch_A_bf16"), py::arg("scratch_B_bf16"),
        py::arg("stream") = 0);

    // Phase 3.2 — causal_conv1d for Qwen3.6 linear-attention. SiLU
    // fused into the epilogue. Two variants for prefill / decode.
    m.def("causal_conv1d_qwen36_bf16",
        [](uintptr_t x, uintptr_t w, uintptr_t bias, uintptr_t out,
           int B, int S, int conv_dim, int k, bool apply_silu,
           uintptr_t stream) {
            flash_rt::kernels::causal_conv1d_qwen36_bf16(
                to_ptr(x), to_ptr(w),
                bias ? to_ptr(bias) : nullptr,
                to_ptr(out),
                B, S, conv_dim, k, apply_silu, to_stream(stream));
        },
        py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("out"),
        py::arg("B"), py::arg("S"), py::arg("conv_dim"), py::arg("k"),
        py::arg("apply_silu") = true, py::arg("stream") = 0);

    m.def("causal_conv1d_qwen36_update_bf16",
        [](uintptr_t x_new, uintptr_t w, uintptr_t bias,
           uintptr_t out, uintptr_t state,
           int B, int conv_dim, int k, bool apply_silu,
           uintptr_t stream) {
            flash_rt::kernels::causal_conv1d_qwen36_update_bf16(
                to_ptr(x_new), to_ptr(w),
                bias ? to_ptr(bias) : nullptr,
                to_ptr(out), to_ptr(state),
                B, conv_dim, k, apply_silu, to_stream(stream));
        },
        py::arg("x_new"), py::arg("w"), py::arg("bias"),
        py::arg("out"), py::arg("state"),
        py::arg("B"), py::arg("conv_dim"), py::arg("k"),
        py::arg("apply_silu") = true, py::arg("stream") = 0);

    m.def("causal_conv1d_qwen36_update_inout_bf16",
        [](uintptr_t x_new, uintptr_t w, uintptr_t bias,
           uintptr_t out, uintptr_t state_in, uintptr_t state_out,
           int B, int conv_dim, int k, bool apply_silu,
           uintptr_t stream) {
            flash_rt::kernels::causal_conv1d_qwen36_update_inout_bf16(
                to_ptr(x_new), to_ptr(w),
                bias ? to_ptr(bias) : nullptr,
                to_ptr(out),
                to_ptr(state_in), to_ptr(state_out),
                B, conv_dim, k, apply_silu, to_stream(stream));
        },
        py::arg("x_new"), py::arg("w"), py::arg("bias"),
        py::arg("out"),
        py::arg("state_in"), py::arg("state_out"),
        py::arg("B"), py::arg("conv_dim"), py::arg("k"),
        py::arg("apply_silu") = true, py::arg("stream") = 0);

    // Phase 4.4 — stream-invariant bf16 matvec for Qwen3.6 (replaces F.linear
    // / cuBLASLt for the small in_proj_a/b and the lm_head bf16 GEMM whose
    // per-stream / per-graph algo selection breaks CUDA Graph correctness).
    m.def("bf16_matvec_qwen36_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t out,
           int N, int K, uintptr_t stream) {
            flash_rt::kernels::bf16_matvec_qwen36_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W),
                reinterpret_cast<__nv_bfloat16*>(out),
                N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("out"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // bf16 row-major matmul for small M (= Qwen3.6 NVFP4 verify path).
    // Sibling of bf16_matvec_qwen36_bf16: same warp-per-output kernel
    // but tiled across M output rows so W is read once instead of M
    // times. Replaces the K-loop matvec at lin-attn unquantized
    // projections (in_proj_qkv / in_proj_z / out_proj). Stream-invariant
    // and CUDA Graph compatible.
    m.def("bf16_matmul_qwen36_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t out,
           int M, int N, int K, uintptr_t stream) {
            flash_rt::kernels::bf16_matmul_qwen36_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W),
                reinterpret_cast<__nv_bfloat16*>(out),
                M, N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("out"),
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // Phase 4.3 — SiLU-gate elementwise multiply for Qwen3.6 SwiGLU MLP.
    // out[i] = silu(gate[i]) * up[i], bf16 in/out, fp32 internal.
    // Replaces the F.silu(gate) * up Python composite (2 allocs/call).
    m.def("silu_mul_qwen36_bf16",
        [](uintptr_t gate, uintptr_t up, uintptr_t out,
           int n, uintptr_t stream) {
            flash_rt::kernels::silu_mul_qwen36_bf16(
                reinterpret_cast<const __nv_bfloat16*>(gate),
                reinterpret_cast<const __nv_bfloat16*>(up),
                reinterpret_cast<__nv_bfloat16*>(out),
                n, to_stream(stream));
        },
        py::arg("gate"), py::arg("up"), py::arg("out"),
        py::arg("n"), py::arg("stream") = 0);

    // Fused RMSNorm + weight + silu(gate) for Qwen3.6 linear-attn output.
    m.def("rms_norm_gated_silu_qwen36_bf16",
        [](uintptr_t x, uintptr_t gate, uintptr_t weight, uintptr_t out,
           int M, int dim, float eps, uintptr_t stream) {
            flash_rt::kernels::rms_norm_gated_silu_qwen36_bf16(
                to_ptr(x), to_ptr(gate), to_ptr(weight), to_ptr(out),
                M, dim, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("gate"), py::arg("weight"), py::arg("out"),
        py::arg("M"), py::arg("dim"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    // Phase 3.3 — Gated DeltaNet recurrent (single-token decode).
    m.def("gated_deltanet_recurrent_qwen36_bf16",
        [](uintptr_t q, uintptr_t k, uintptr_t v,
           uintptr_t g, uintptr_t beta,
           uintptr_t state, uintptr_t out,
           int B, int num_v_heads, int head_k_dim, int head_v_dim,
           bool use_qk_l2norm, uintptr_t stream) {
            flash_rt::kernels::gated_deltanet_recurrent_qwen36_bf16(
                to_ptr(q), to_ptr(k), to_ptr(v),
                to_ptr(g), to_ptr(beta),
                to_ptr(state), to_ptr(out),
                B, num_v_heads, head_k_dim, head_v_dim,
                use_qk_l2norm, to_stream(stream));
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("g"), py::arg("beta"),
        py::arg("state"), py::arg("out"),
        py::arg("B"), py::arg("num_v_heads"),
        py::arg("head_k_dim"), py::arg("head_v_dim"),
        py::arg("use_qk_l2norm") = true, py::arg("stream") = 0);

    // In/out-state variant for K-iter chained per-step save (A2c-3).
    m.def("gated_deltanet_recurrent_inout_qwen36_bf16",
        [](uintptr_t q, uintptr_t k, uintptr_t v,
           uintptr_t g, uintptr_t beta,
           uintptr_t state_in, uintptr_t state_out, uintptr_t out,
           int B, int num_v_heads, int head_k_dim, int head_v_dim,
           bool use_qk_l2norm, uintptr_t stream) {
            flash_rt::kernels::gated_deltanet_recurrent_inout_qwen36_bf16(
                to_ptr(q), to_ptr(k), to_ptr(v),
                to_ptr(g), to_ptr(beta),
                to_ptr(state_in), to_ptr(state_out), to_ptr(out),
                B, num_v_heads, head_k_dim, head_v_dim,
                use_qk_l2norm, to_stream(stream));
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("g"), py::arg("beta"),
        py::arg("state_in"), py::arg("state_out"), py::arg("out"),
        py::arg("B"), py::arg("num_v_heads"),
        py::arg("head_k_dim"), py::arg("head_v_dim"),
        py::arg("use_qk_l2norm") = true, py::arg("stream") = 0);

#ifdef ENABLE_CUTLASS_SM120_BLOCK_FP8
    // Path B: native CUTLASS block-128 FP8 GEMM on SM120a — no
    // dequant intermediate, ~12-13x faster than Path D for Qwen3.6
    // shapes. Drop-in replacement (no scratch buffers needed).
    m.def("fp8_block128_gemm_cutlass_sm120_bf16out",
        [](uintptr_t A, uintptr_t B, uintptr_t D,
           int M, int N, int K,
           uintptr_t act_scale, uintptr_t w_scale,
           uintptr_t stream) {
            flash_rt::gemm::fp8_block128_gemm_cutlass_sm120_bf16out(
                to_ptr(A), to_ptr(B), to_ptr(D),
                M, N, K,
                reinterpret_cast<const float*>(act_scale),
                reinterpret_cast<const float*>(w_scale),
                to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("act_block_scale"), py::arg("w_block_scale"),
        py::arg("stream") = 0);
#endif

#ifdef ENABLE_CUTLASS_SM120_NVFP4_W4A16
    // NVFP4 W4A16 GEMM on SM120a (RTX 5090 / Blackwell GeForce).
    // Used by the Qwen3.6 NVFP4 main path (alternative to FP8 Path B).
    //
    // Inputs:
    //   A_packed (M, K/2) u8  — FP4 e2m1 act, two values per byte
    //   B_packed (N, K/2) u8  — FP4 e2m1 weight, row-major (HF natural);
    //                            CUTLASS reads as (K, N) ColumnMajor.
    //   D_bf16   (M, N)  bf16 — output, row-major
    //   SFA      (M, K/16) e4m3 — Sm1xx blockscaled atom layout
    //   SFB      (N, K/16) e4m3 — Sm1xx blockscaled atom layout
    //   alpha    f32          = act_global_scale * w_global_scale
    //
    // The activation quantizer (BF16->FP4 + FP8 SF) emits SFA in the
    // expected layout; the weight loader does the same one-time
    // transform on SFB at ckpt load.
    m.def("fp4_w4a16_gemm_sm120_bf16out",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           int M, int N, int K,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out(
                to_ptr(A_packed), to_ptr(B_packed), to_ptr(D),
                M, N, K,
                to_ptr(SFA), to_ptr(SFB),
                alpha,
                to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("alpha") = 1.0f,
        py::arg("stream") = 0);

    // Wide-N variant of fp4_w4a16_gemm_sm120_bf16out. TileShape
    // <128, 256, 128> instead of <128, 128, 256>. Profiled faster
    // for shapes with very large N (lm_head N=248320: 88% peak BW
    // vs 64% baseline; MLP gate/up N=17408: 66% vs 56%). Slower for
    // small/medium N — caller must dispatch by shape.
    m.def("fp4_w4a16_gemm_sm120_bf16out_widen",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           int M, int N, int K,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out_widen(
                to_ptr(A_packed), to_ptr(B_packed), to_ptr(D),
                M, N, K,
                to_ptr(SFA), to_ptr(SFB),
                alpha,
                to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("alpha") = 1.0f,
        py::arg("stream") = 0);

    // Reshape linear (rows, K/16) FP8E4M3 group-scale tensor into the
    // CUTLASS Sm1xx blockscaled tile-interleaved layout that the
    // GEMM kernel expects. Run once per weight tensor at ckpt load.
    // The activation path uses quantize_bf16_to_nvfp4_swizzled which
    // already emits the swizzled layout directly.
    m.def("nvfp4_sf_linear_to_swizzled",
        [](uintptr_t src_linear, uintptr_t dst_swz,
           int rows, int D, bool is_sfb, uintptr_t stream) {
            return flash_rt::fp4::nvfp4_sf_linear_to_swizzled(
                to_ptr(src_linear), to_ptr(dst_swz),
                rows, D, is_sfb, to_stream(stream));
        },
        py::arg("src_linear"), py::arg("dst_swz"),
        py::arg("rows"), py::arg("D"),
        py::arg("is_sfb") = false, py::arg("stream") = 0);

    m.def("nvfp4_sf_swizzled_bytes",
        &flash_rt::fp4::nvfp4_sf_swizzled_bytes,
        py::arg("rows"), py::arg("D"));

    // ── Qwen3-8B NVFP4 W4A4 M=1 matvec / MMA (decode hot path) ──
    // Custom SM120 kernels specialized for M=1 LLM decode where
    // CUTLASS NVFP4 GEMM tiles assume M ≥ 16 and run at ~30 % of HBM
    // BW. The matvec is the SIMT fallback / oracle; the MMA path
    // (full_n) is the production decode kernel.
    m.def("fp4_w4a4_matvec_sm120_bf16out",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           int N, int K,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) -> int {
            return flash_rt::gemm::fp4_w4a4_matvec_sm120_bf16out(
                to_ptr(A_packed), to_ptr(B_packed), to_ptr(D),
                N, K, to_ptr(SFA), to_ptr(SFB),
                alpha, to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"), py::arg("D"),
        py::arg("N"), py::arg("K"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("alpha") = 1.0f,
        py::arg("stream") = 0);

    m.def("fp4_w4a4_matvec_sm120_init",
        []() { flash_rt::gemm::fp4_w4a4_matvec_init_luts(); },
        "Idempotent UE4M3 LUT initialization for the matvec kernel.");

    m.def("fp4_w4a4_mma_sm120_single_tile_bf16out",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) -> int {
            return flash_rt::gemm::fp4_w4a4_mma_sm120_single_tile_bf16out(
                to_ptr(A_packed), to_ptr(B_packed), to_ptr(D),
                to_ptr(SFA), to_ptr(SFB),
                alpha, to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"), py::arg("D"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("alpha") = 1.0f,
        py::arg("stream") = 0);

    m.def("fp4_w4a4_mma_sm120_multi_k_bf16out",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           uintptr_t SFA, uintptr_t SFB,
           float alpha, int K,
           uintptr_t stream) -> int {
            return flash_rt::gemm::fp4_w4a4_mma_sm120_multi_k_bf16out(
                to_ptr(A_packed), to_ptr(B_packed), to_ptr(D),
                to_ptr(SFA), to_ptr(SFB),
                alpha, K, to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"), py::arg("D"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("alpha") = 1.0f, py::arg("K"),
        py::arg("stream") = 0);

    m.def("fp4_w4a4_mma_sm120_full_n_bf16out",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           int N, int K,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) -> int {
            return flash_rt::gemm::fp4_w4a4_mma_sm120_full_n_bf16out(
                to_ptr(A_packed), to_ptr(B_packed), to_ptr(D),
                N, K, to_ptr(SFA), to_ptr(SFB),
                alpha, to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"), py::arg("D"),
        py::arg("N"), py::arg("K"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("alpha") = 1.0f,
        py::arg("stream") = 0);

    // ── Fused qkv post-processing for Qwen3-8B ──
    // Replaces (q_norm + RoPE + Q_buf copy) with one launch and
    // (k_norm + RoPE + K_cache write + V_cache write) with another.
    // head_dim hardcoded at 128; S=1 decode hot path only.
    m.def("qwen3_q_norm_rope_qstage_bf16",
        [](uintptr_t q_pre, uintptr_t q_norm_w,
           uintptr_t cos, uintptr_t sin,
           uintptr_t q_buf_dst,
           int n_q_heads, float eps, uintptr_t stream) -> int {
            return flash_rt::kernels::qwen3_q_norm_rope_qstage_bf16(
                to_ptr(q_pre), to_ptr(q_norm_w),
                to_ptr(cos), to_ptr(sin),
                to_ptr(q_buf_dst),
                n_q_heads, eps, to_stream(stream));
        },
        py::arg("q_pre"), py::arg("q_norm_w"),
        py::arg("cos"), py::arg("sin"),
        py::arg("q_buf_dst"),
        py::arg("n_q_heads"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    m.def("qwen3_k_norm_rope_kvwrite_bf16",
        [](uintptr_t k_pre, uintptr_t v_pre, uintptr_t k_norm_w,
           uintptr_t cos, uintptr_t sin,
           uintptr_t k_cache_dst, uintptr_t v_cache_dst,
           int n_kv_heads, float eps, uintptr_t stream) -> int {
            return flash_rt::kernels::qwen3_k_norm_rope_kvwrite_bf16(
                to_ptr(k_pre), to_ptr(v_pre), to_ptr(k_norm_w),
                to_ptr(cos), to_ptr(sin),
                to_ptr(k_cache_dst), to_ptr(v_cache_dst),
                n_kv_heads, eps, to_stream(stream));
        },
        py::arg("k_pre"), py::arg("v_pre"), py::arg("k_norm_w"),
        py::arg("cos"), py::arg("sin"),
        py::arg("k_cache_dst"), py::arg("v_cache_dst"),
        py::arg("n_kv_heads"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    // ── Fused silu_mul + nvfp4 swizzled quantize ──
    m.def("silu_mul_to_nvfp4_swizzled_bf16",
        [](uintptr_t gate, uintptr_t up,
           uintptr_t packed, uintptr_t sf_swz,
           int rows, int cols, uintptr_t stream) -> int {
            return flash_rt::kernels::silu_mul_to_nvfp4_swizzled_bf16(
                to_ptr(gate), to_ptr(up),
                to_ptr(packed), to_ptr(sf_swz),
                rows, cols, to_stream(stream));
        },
        py::arg("gate"), py::arg("up"),
        py::arg("packed"), py::arg("sf_swz"),
        py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

#endif
}
