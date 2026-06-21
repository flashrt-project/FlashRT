// ================================================================
// FlashRT — pybind11 bindings
// Exposes GemmRunner + all CUDA kernels to Python
// ================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <stdexcept>
#include "context.h"
#include "gemm/gemm_runner.h"
#include "gemm/fp8_block128_gemm.cuh"
#ifdef ENABLE_CUTLASS_SM120_BLOCK_FP8
#include "gemm/cutlass_sm120_block128_fp8_gemm.cuh"
#include "gemm/fp8_smallM_handtuned_sm120.cuh"
#include "gemm/fp8_smallM_handtuned_splitk_sm120.cuh"
#include "gemm/fp8_smallM_handtuned_ldmatrix_sm120.cuh"
#endif
#ifdef ENABLE_DECODE_GEMV_M1
#include "gemm/fp8_gemv_m1_sm120.cuh"
#include "gemm/bf16_gemv_m1_sm120.cuh"
#endif
#ifdef ENABLE_CUTLASS_SM120_NVFP4_W4A16
#include "gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cuh"
#include "gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_bf16out_sm120.cuh"
#include "gemm/fp4/cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.cuh"
#include "gemm/fp4/cutlass_nvfp4_gemm_dn_streamk_bias_sm120.cuh"
#endif
#ifdef ENABLE_CUTLASS_SM100_NVFP4_W4A16
#include "gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm100.cuh"
#endif
#ifdef ENABLE_ACTION_FFN_MEGAKERNEL_V6T
#include "kernels/megakernel/action_ffn_megakernel_v6t_sm120.cuh"
#endif
#ifdef ENABLE_UND_FFN_MEGAKERNEL_V5T
#include "kernels/megakernel/und_ffn_megakernel_v5t_sm120.cuh"
#include "kernels/megakernel/und_ffn_megakernel_v5split_stage3_sm120.cuh"
#endif
#ifdef ENABLE_TINYFP8_KERNELS
#include "kernels/megakernel/tinyfp8_kernels_sm120.cuh"
#endif
#ifdef ENABLE_QWEN36_FLASHINFER_XQA
#include "kernels/qwen36_flashinfer_xqa.cuh"
#endif
#if defined(ENABLE_CUTLASS_SM120_NVFP4_W4A16) || defined(ENABLE_CUTLASS_SM100_NVFP4_W4A16)
#include "quantize/nvfp4_sf_reshape_sm120.cuh"
#endif
#ifdef ENABLE_FP8_CONV3D_V17
extern "C" int fp8_conv3d_v17_ndhwc_bf16out(
    const void* cache_x_fp8, const void* new_x_fp8,
    const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);
extern "C" int fp8_conv3d_v17_anyco_ndhwc_bf16out(
    const void* cache_x_fp8, const void* new_x_fp8,
    const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);
#endif
#ifdef ENABLE_FP8_CONV3D_V18
extern "C" int fp8_conv3d_v18_ncdhw_res_bf16out(
    const void* cache_x_fp8, const void* new_x_fp8,
    const void* w_fp8, void* y_bf16,
    const void* bias_bf16, const void* residual_bf16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);
#endif
#ifdef ENABLE_FP8_CONV2D_3X3_V1
extern "C" int fp8_conv2d_3x3_v1_nhwc_bf16out(
    const void* x_fp8, const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int N, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);
extern "C" int fp8_conv2d_3x3_v2_nhwc_bf16out(
    const void* x_fp8, const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int N, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);
extern "C" int fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out(
    const void* x_fp8, const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int B, int T, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);
#endif
#ifdef ENABLE_CUDNN_FP8_CONV2D_3X3
extern "C" int cudnn_fp8_conv2d_3x3_nhwc_bf16out(
    const void* x_fp8, const void* w_fp8, void* y_bf16,
    const void* bias_bf16,
    int N, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);
#endif
#ifdef ENABLE_MOTUS
extern "C" int motus_fp4_conv3d_v19sf_ndhwc_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*,
    int, int, int, int, int, int, int, float, cudaStream_t);
extern "C" int motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, void*, const void*,
    int, int, int, int, int, int, int, float, cudaStream_t);
extern "C" int motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*, const void*,
    int, int, int, int, int, int, int, float, cudaStream_t);
extern "C" int motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out_v2(
    const void*, const void*, const void*, const void*, const void*,
    const void*, const void*, void*, const void*, const void*,
    int, int, int, int, int, int, int, float, cudaStream_t);
extern "C" int motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out(
    const void*, const void*, const void*, const void*, const void*,
    const void*, void*, const void*, const void*,
    int, int, int, int, int, int, int, float, cudaStream_t);
extern "C" int motus_bf16_rms_silu_quant_nvfp4_to_ndhwc_v1(
    const void*, const void*, const void*, void*, void*,
    int, int, int, int, int, float, cudaStream_t);
#endif
#ifdef ENABLE_SM80_INT8_CUTLASS
extern "C" int cutlass_int8_silu_gated_bf16out(
    void const*, void const*, void const*, void const*, void const*, void*,
    int, int, int, cudaStream_t);
extern "C" int cutlass_int8_rowwise_bf16out(
    void const*, void const*, void const*, void const*, void*,
    int, int, int, cudaStream_t);
extern "C" int cutlass_int8_rowwise_bf16out_t64x128(
    void const*, void const*, void const*, void const*, void*,
    int, int, int, cudaStream_t);
#endif
#include "kernels/kernels.h"
#include "kernels/fusion.cuh"
#include "kernels/causal_conv1d_qwen36.cuh"
#include "kernels/linear_attention/gated_delta_wy_bf16.cuh"
#include "kernels/gated_deltanet_qwen36.cuh"
#include "kernels/qwen3_qkv_post_proc.cuh"
#include "kernels/silu_mul_to_nvfp4_swizzled.cuh"
#include "kernels/rms_norm_gated_silu_qwen36.cuh"
#include "kernels/silu_mul_qwen36.cuh"
#include "kernels/qwen36_misc.cuh"
#include "kernels/nexn2_misc.cuh"
#include "kernels/nexn2_moe_grouped.cuh"
#include "kernels/nexn2_bf16_gemv.cuh"
#include "kernels/nexn2_w4a16_gemv.cuh"
#include "kernels/bf16_matvec_qwen36.cuh"
#include "kernels/bf16_matmul_qwen36.cuh"
#include "kernels/bf16_matmul_qwen36_thor.cuh"
#include "kernels/fp4_w4a4_matvec_sm120.cuh"
#include "kernels/fp4_w4a4_mma_sm120.cuh"
#include "kernels/fp4_w4a4_mma_warpsplit_sm120.cuh"
#include "kernels/fp4_w4a4_mma_warpsplit_mrows_sm120.cuh"
#include "quantize/fp8_block128_dequant.cuh"
#include "quantize/fp8_block128_to_nvfp4_swizzled.cuh"
#include "quantize/bf16_weight_to_nvfp4_swizzled.cuh"
#include "quantize/fp8_per_token_block_quant.cuh"
#include "quantize/bias_gelu_quantize_fp8.cuh"
#include "quantize/awq_quant_fp8_static_bf16.cuh"
#include "quantize/rope_apply_bf16.cuh"
#include "quantize/ada_layer_norm_fp8.cuh"
#include "quantize/bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4.cuh"
#include "quantize/bf16_rms_silu_ncdhw.cuh"
#include "quantize/bf16_ndhwc_to_ncdhw_transpose.cuh"
#include "quantize/bf16_quant_fp8_ncdhw_to_ndhwc.cuh"
#include "quantize/qkv_split_norm_rope_bf16.cuh"
#include "attention/fmha_dispatch.h"
#ifdef ENABLE_MOTUS_SAGE2_RAW
#include "attention/sage2/sage2_attn_raw.cuh"
#endif

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
extern "C" void tq_fp32_gemm_fp32_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream);
extern "C" void tq_fp32_gemm_fp32_bt_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream);
extern "C" void tq_fp32_gemm_lt_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream);
extern "C" void tq_fp32_gemm_lt_bt_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream);
extern "C" void tq_fp32_gemm_lt_algo_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    int algo_idx,
    cudaStream_t stream);
extern "C" void tq_fp32_gemm_lt_bt_algo_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    int algo_idx,
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

void layer_norm_no_affine_fp8_static_bf16(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out, const float* d_scale,
    int seq_len, int dim, float eps, cudaStream_t stream);
void ada_layer_norm_bf16_per_token(
    const __nv_bfloat16* x, const __nv_bfloat16* scale,
    const __nv_bfloat16* shift, __nv_bfloat16* out,
    int seq_len, int dim, float eps, cudaStream_t stream);
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
// CUTLASS FP16 variants (encoder/SigLIP FP16 path)
extern "C" int cutlass_fp16_plain(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_sq(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_t1(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_wide(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_k64(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_2sm21(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_k64_gelu(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_sq_gelu(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_k64_mul_aux(void*, void*, void*, void*, int, int, int, cudaStream_t);
#ifdef FLASHRT_HAVE_SM100_ENCODER_MLP
extern "C" int encoder_mlp_fused_fp16(void*, void*, void*, void*,
                                       void*, void*, void*, void*,
                                       int, int, int, int, cudaStream_t);
#endif
extern "C" int flashrt_megakernel_single_fp16(void*, void*, void*,
                                               int, int, int,
                                               float, float, cudaStream_t);
extern "C" int flashrt_megakernel_geglu_fp16(void*, void*, void*,
                                              void*, void*,
                                              int, int, int,
                                              cudaStream_t);
// Fused encoder GeGLU + down-proj with residual.
// Args: (X, W_gate, W_up, W_down, hidden_scratch, x_inout, M, H, D, stream).
// Computes x_inout += GeGLU(X @ W_gate, X @ W_up) @ W_down, bundling the
// GeGLU megakernel and the down GEMM (beta=1) into one C entry.
extern "C" int flashrt_megakernel_geglu_g8_fp16(void*, void*, void*,
                                                 void*,
                                                 void*, void*,
                                                 int, int, int,
                                                 cudaStream_t);

// Bundle: rms_norm + GeGLU + down-proj + residual into one C entry.
extern "C" int flashrt_encoder_ffn_block_fp16(void*, void*, void*,
                                               void*, void*, void*,
                                               void*, void*,
                                               int, int, int, float,
                                               cudaStream_t);

// Bundle: rms_norm + QKV (k64) into one C entry.
extern "C" int flashrt_rms_qkv_fp16(void*, void*, void*, void*, void*,
                                     int, int, int, float, cudaStream_t);
#ifdef FLASHRT_HAVE_SM100_SWEEP
extern "C" int cutlass_fp16_sweep(int variant, void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp16_sweep_count();
#endif
extern "C" int cutlass_fp8_plain(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_gelu(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_sq_f32out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_wide_f32out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_sq_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_wide_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_t1_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
#endif

#ifdef ENABLE_LINGBOT
#include "kernels/lingbot_kernels.h"   // LingBot-VLA model kernel decls (lingbot_-prefixed, Thor sm_110a)
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
        .def("fp8_nt_dev", [](GemmRunner& self,
                               uintptr_t A, uintptr_t B, uintptr_t D,
                               int M, int N, int K,
                               uintptr_t d_scale_a, uintptr_t d_scale_b,
                               uintptr_t stream) {
            self.fp8_nt_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
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
        .def("fp8_nn_bias_bf16", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                     int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_bias_bf16(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
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
        .def("autotune_fp16_nn", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K, int num_algos) {
            self.autotune_fp16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, num_algos);
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
        .def("autotune_fp8_nt_dev", [](GemmRunner& self,
                                        uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K,
                                        uintptr_t d_scale_a, uintptr_t d_scale_b,
                                        int num_algos) {
            self.autotune_fp8_nt_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
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

    m.def("bias_rms_norm_bf16", [](uintptr_t x, uintptr_t bias, uintptr_t weight,
                                    uintptr_t out, int seq_len, int dim,
                                    float eps, uintptr_t stream) {
        bias_rms_norm_bf16(typed_ptr<__nv_bfloat16>(x),
                           typed_ptr<__nv_bfloat16>(bias),
                           typed_ptr<__nv_bfloat16>(weight),
                           typed_ptr<__nv_bfloat16>(out),
                           seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("bias"), py::arg("weight"), py::arg("out"),
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

    // G7.11 — fused (bias + GELU(tanh)) in-place on bf16 (M, N) tensor.
    m.def("bias_gelu_inplace_bf16", [](uintptr_t x, uintptr_t bias,
                                         int M, int N, uintptr_t stream) {
        bias_gelu_inplace_bf16(typed_ptr<__nv_bfloat16>(x),
                                typed_ptr<__nv_bfloat16>(bias),
                                M, N, to_stream(stream));
    }, py::arg("x"), py::arg("bias"),
       py::arg("M"), py::arg("N"), py::arg("stream") = 0);

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

    // QKV split + RoPE with a runtime device K/V-cache row offset (devpos).
    // K/V are cache BASE pointers; row written = devpos[0] + token index.
    m.def("qkv_split_rope_devpos", [](uintptr_t qkv, uintptr_t rope_weights,
                                       uintptr_t Q, uintptr_t K, uintptr_t V,
                                       uintptr_t devpos,
                                       int seq, int q_dim, int k_dim, int v_dim,
                                       int head_dim, uintptr_t stream) {
        qkv_split_rope_devpos(typed_ptr<__nv_bfloat16>(qkv),
                              typed_ptr<__nv_bfloat16>(rope_weights),
                              typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                              typed_ptr<__nv_bfloat16>(V),
                              typed_ptr<int>(devpos),
                              seq, q_dim, k_dim, v_dim, head_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope_weights"),
       py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("devpos"),
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

    // G6.7: residual += (x + bias) * gate. Replaces add_bias + gate_mul_residual chain.
    m.def("bias_gate_mul_residual_bf16",
          [](uintptr_t residual, uintptr_t x, uintptr_t bias, uintptr_t gate,
             int seq_len, int dim, uintptr_t stream) {
        bias_gate_mul_residual_bf16(typed_ptr<__nv_bfloat16>(residual),
                                     typed_ptr<__nv_bfloat16>(x),
                                     typed_ptr<__nv_bfloat16>(bias),
                                     typed_ptr<__nv_bfloat16>(gate),
                                     seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias"), py::arg("gate"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

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

    m.def("gate_residual_ada_norm_fp16", [](uintptr_t residual, uintptr_t x,
                                             uintptr_t gate, uintptr_t weight,
                                             uintptr_t style,
                                             uintptr_t out, uintptr_t gate_out,
                                             int seq_len, int dim, float eps,
                                             uintptr_t stream) {
        gate_residual_ada_norm_fp16(reinterpret_cast<__half*>(residual),
                                     reinterpret_cast<const __half*>(x),
                                     reinterpret_cast<const __half*>(gate),
                                     reinterpret_cast<const __half*>(weight),
                                     reinterpret_cast<const __half*>(style),
                                     reinterpret_cast<__half*>(out),
                                     reinterpret_cast<__half*>(gate_out),
                                     seq_len, dim, eps, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("weight"),
       py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("stream") = 0);

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

    m.def("fp8_accumulate_scale_max", [](uintptr_t src_scale,
                                         uintptr_t dst_scale,
                                         uintptr_t stream) {
        fp8_accumulate_scale_max(reinterpret_cast<const float*>(src_scale),
                                 reinterpret_cast<float*>(dst_scale),
                                 to_stream(stream));
    }, py::arg("src_scale"), py::arg("dst_scale"), py::arg("stream") = 0);

    // FP16 device-only FP8 quantize (GPU absmax + scale + quantize, CUDA Graph compatible)
    m.def("quantize_fp8_device_fp16", [](uintptr_t input, uintptr_t output,
                                          uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_device_fp16(reinterpret_cast<const __half*>(input),
                                  typed_ptr<__nv_fp8_e4m3>(output),
                                  reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

// Bindings below cover the BF16->NVFP4 quantize / norm-fused-quantize
// family. The kernels themselves live in csrc/kernels/quantize.cu and
// are compiled into flash_rt_kernels unconditionally for every Blackwell
// target (sm_110 Thor and sm_120 RTX). The original gate is sm_120-only
// because the symbol family was introduced alongside the Motus NVFP4
// path; Qwen3.6 on Thor reaches the same fused norm + quant kernels at
// every pre-projection site, so the SM100 W4A16 build also opts in here.
#if defined(ENABLE_NVFP4) || defined(ENABLE_CUTLASS_SM100_NVFP4_W4A16)
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

    m.def("quantize_bf16_to_nvfp4_swizzled_k14336",
        [](uintptr_t input, uintptr_t fp4_data, uintptr_t scale_factors,
           int rows, int cols, uintptr_t stream) {
            return quantize_bf16_to_nvfp4_swizzled_k14336(
                typed_ptr<__nv_bfloat16>(input),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, to_stream(stream));
        }, py::arg("input"), py::arg("fp4_data"),
        py::arg("scale_factors"), py::arg("rows"), py::arg("cols"),
        py::arg("stream") = 0);

    m.def("quantize_bf16_to_nvfp4_swizzled_clipped",
        [](uintptr_t input, uintptr_t clip_amax, uintptr_t fp4_data,
           uintptr_t scale_factors, int rows, int cols, uintptr_t stream) {
            quantize_bf16_to_nvfp4_swizzled_clipped(
                typed_ptr<__nv_bfloat16>(input),
                reinterpret_cast<const float*>(clip_amax),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, to_stream(stream));
        }, py::arg("input"), py::arg("clip_amax"), py::arg("fp4_data"),
        py::arg("scale_factors"), py::arg("rows"), py::arg("cols"),
        py::arg("stream") = 0);

    m.def("quantize_bf16_to_nvfp4_swizzled_static_groups",
        [](uintptr_t input, uintptr_t group_amax, uintptr_t fp4_data,
           uintptr_t scale_factors, int rows, int cols, uintptr_t stream) {
            quantize_bf16_to_nvfp4_swizzled_static_groups(
                typed_ptr<__nv_bfloat16>(input),
                reinterpret_cast<const float*>(group_amax),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, to_stream(stream));
        }, py::arg("input"), py::arg("group_amax"), py::arg("fp4_data"),
        py::arg("scale_factors"), py::arg("rows"), py::arg("cols"),
        py::arg("stream") = 0);

    m.def("quantize_bf16_to_nvfp4_swizzled_secondmax",
        [](uintptr_t input, uintptr_t fp4_data, uintptr_t scale_factors,
           int rows, int cols, float scale_mult, uintptr_t stream) {
            quantize_bf16_to_nvfp4_swizzled_secondmax(
                typed_ptr<__nv_bfloat16>(input),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, scale_mult, to_stream(stream));
        }, py::arg("input"), py::arg("fp4_data"),
        py::arg("scale_factors"), py::arg("rows"), py::arg("cols"),
        py::arg("scale_mult") = 1.0f, py::arg("stream") = 0);

    m.def("quantize_bf16_to_nvfp4_swizzled_mse",
        [](uintptr_t input, uintptr_t fp4_data, uintptr_t scale_factors,
           int rows, int cols, uintptr_t stream) {
            quantize_bf16_to_nvfp4_swizzled_mse(
                typed_ptr<__nv_bfloat16>(input),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, to_stream(stream));
        }, py::arg("input"), py::arg("fp4_data"),
        py::arg("scale_factors"), py::arg("rows"), py::arg("cols"),
        py::arg("stream") = 0);

    m.def("awq_quant_bf16_to_nvfp4_swizzled", [](uintptr_t input,
                                                  uintptr_t inv_s,
                                                  uintptr_t fp4_data,
                                                  uintptr_t scale_factors,
                                                  int rows, int cols,
                                                  uintptr_t stream) {
        awq_quant_bf16_to_nvfp4_swizzled(
            typed_ptr<__nv_bfloat16>(input),
            typed_ptr<__nv_bfloat16>(inv_s),
            reinterpret_cast<uint8_t*>(fp4_data),
            reinterpret_cast<uint8_t*>(scale_factors),
            rows, cols, to_stream(stream));
    }, py::arg("input"), py::arg("inv_s"), py::arg("fp4_data"),
       py::arg("scale_factors"), py::arg("rows"), py::arg("cols"),
       py::arg("stream") = 0);

    m.def("bias_gelu_quant_bf16_to_nvfp4_swizzled",
        [](uintptr_t input, uintptr_t bias,
           uintptr_t fp4_data, uintptr_t scale_factors,
           int rows, int cols, uintptr_t stream) {
            bias_gelu_quant_bf16_to_nvfp4_swizzled(
                typed_ptr<__nv_bfloat16>(input),
                typed_ptr<__nv_bfloat16>(bias),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, to_stream(stream));
        },
        py::arg("input"), py::arg("bias"),
        py::arg("fp4_data"), py::arg("scale_factors"),
        py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("gather_bf16_cols",
        [](uintptr_t input, uintptr_t indices, uintptr_t output,
           int rows, int cols, int n_idx, uintptr_t stream) {
            gather_bf16_cols(
                typed_ptr<__nv_bfloat16>(input),
                reinterpret_cast<const int*>(indices),
                typed_ptr<__nv_bfloat16>(output),
                rows, cols, n_idx, to_stream(stream));
        },
        py::arg("input"), py::arg("indices"), py::arg("output"),
        py::arg("rows"), py::arg("cols"), py::arg("n_idx"),
        py::arg("stream") = 0);

    m.def("add_side_bias_gelu_gather_zero_quant_bf16_to_nvfp4_swizzled",
        [](uintptr_t main, uintptr_t side, uintptr_t bias,
           uintptr_t zero_gather_indices, uintptr_t side_out,
           uintptr_t fp4_data, uintptr_t scale_factors,
           int rows, int cols, int n_idx, uintptr_t stream) {
            add_side_bias_gelu_gather_zero_quant_bf16_to_nvfp4_swizzled(
                typed_ptr<__nv_bfloat16>(main),
                typed_ptr<__nv_bfloat16>(side),
                typed_ptr<__nv_bfloat16>(bias),
                reinterpret_cast<const int*>(zero_gather_indices),
                typed_ptr<__nv_bfloat16>(side_out),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, n_idx, to_stream(stream));
        },
        py::arg("main"), py::arg("side"), py::arg("bias"),
        py::arg("zero_gather_indices"), py::arg("side_out"),
        py::arg("fp4_data"), py::arg("scale_factors"),
        py::arg("rows"), py::arg("cols"), py::arg("n_idx"),
        py::arg("stream") = 0);

    m.def("awq_bias_gelu_quant_bf16_to_nvfp4_swizzled",
        [](uintptr_t input, uintptr_t bias, uintptr_t inv_s,
           uintptr_t fp4_data, uintptr_t scale_factors,
           int rows, int cols, uintptr_t stream) {
            awq_bias_gelu_quant_bf16_to_nvfp4_swizzled(
                typed_ptr<__nv_bfloat16>(input),
                typed_ptr<__nv_bfloat16>(bias),
                typed_ptr<__nv_bfloat16>(inv_s),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, to_stream(stream));
        },
        py::arg("input"), py::arg("bias"), py::arg("inv_s"),
        py::arg("fp4_data"), py::arg("scale_factors"),
        py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("bias_gelu_quant_cached_bf16_to_nvfp4_swizzled",
        [](uintptr_t input, uintptr_t bias,
           uintptr_t fp4_data, uintptr_t scale_factors,
           int rows, int cols, uintptr_t stream) {
            bias_gelu_quant_cached_bf16_to_nvfp4_swizzled(
                typed_ptr<__nv_bfloat16>(input),
                typed_ptr<__nv_bfloat16>(bias),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, to_stream(stream));
        },
        py::arg("input"), py::arg("bias"),
        py::arg("fp4_data"), py::arg("scale_factors"),
        py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("awq_bias_gelu_quant_cached_bf16_to_nvfp4_swizzled",
        [](uintptr_t input, uintptr_t bias, uintptr_t inv_s,
           uintptr_t fp4_data, uintptr_t scale_factors,
           int rows, int cols, uintptr_t stream) {
            awq_bias_gelu_quant_cached_bf16_to_nvfp4_swizzled(
                typed_ptr<__nv_bfloat16>(input),
                typed_ptr<__nv_bfloat16>(bias),
                typed_ptr<__nv_bfloat16>(inv_s),
                reinterpret_cast<uint8_t*>(fp4_data),
                reinterpret_cast<uint8_t*>(scale_factors),
                rows, cols, to_stream(stream));
        },
        py::arg("input"), py::arg("bias"), py::arg("inv_s"),
        py::arg("fp4_data"), py::arg("scale_factors"),
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

    // Fused: affine LayerNorm(x, weight, bias) -> nvfp4 packed +
    // swizzled SF. Used by Motus cross-attn norm3 -> Q NVFP4 path.
    m.def("layer_norm_to_nvfp4_swizzled_bf16",
        [](uintptr_t x, uintptr_t weight, uintptr_t bias,
           uintptr_t packed, uintptr_t sf_swz,
           int rows, int cols, float eps, uintptr_t stream) {
            layer_norm_to_nvfp4_swizzled_bf16(
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(weight),
                typed_ptr<__nv_bfloat16>(bias),
                reinterpret_cast<uint8_t*>(packed),
                reinterpret_cast<uint8_t*>(sf_swz),
                rows, cols, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("weight"), py::arg("bias"),
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

    m.def("ada_rms_norm_style_fp16",
          [](uintptr_t x, uintptr_t weight, uintptr_t style,
             uintptr_t out, uintptr_t gate_out,
             int seq_len, int dim, float eps, uintptr_t stream) {
        ada_rms_norm_style_fp16(
            reinterpret_cast<const __half*>(x),
            reinterpret_cast<const __half*>(weight),
            reinterpret_cast<const __half*>(style),
            reinterpret_cast<__half*>(out),
            reinterpret_cast<__half*>(gate_out),
            seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("style"),
       py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("stream") = 0);

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

    // QKV Split + RoPE + KV Cache (FP16) with runtime device K/V row offset.
    // Same shape contract as qkv_split_rope_kvcache_fp16; K/V row =
    // devpos[0] + token row inside the layer slab.
    m.def("qkv_split_rope_kvcache_fp16_devpos", [](uintptr_t qkv, uintptr_t rope,
                                              uintptr_t Q, uintptr_t Kc, uintptr_t Vc,
                                              uintptr_t devpos,
                                              int S, int Q_dim, int K_dim, int HD, int qkv_stride,
                                              int kc_offset, int kc_stride, uintptr_t stream) {
        qkv_split_rope_kvcache_fp16_devpos(reinterpret_cast<const __half*>(qkv),
                                     reinterpret_cast<const __half*>(rope),
                                     reinterpret_cast<__half*>(Q),
                                     reinterpret_cast<__half*>(Kc),
                                     reinterpret_cast<__half*>(Vc),
                                     reinterpret_cast<const int*>(devpos),
                                     S, Q_dim, K_dim, HD, qkv_stride,
                                     kc_offset, kc_stride, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope"), py::arg("Q"), py::arg("Kc"), py::arg("Vc"),
       py::arg("devpos"),
       py::arg("S"), py::arg("Q_dim"), py::arg("K_dim"), py::arg("HD"), py::arg("qkv_stride"),
       py::arg("kc_offset"), py::arg("kc_stride"), py::arg("stream") = 0);

    // Elementwise FP16
    m.def("bias_residual_fp16", [](uintptr_t residual, uintptr_t x, uintptr_t bias,
                                    int seq_len, int dim, uintptr_t stream) {
        bias_residual_fp16(reinterpret_cast<__half*>(residual), reinterpret_cast<const __half*>(x),
                            reinterpret_cast<const __half*>(bias), seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("bias_residual_strict_fp16",
          [](uintptr_t residual, uintptr_t x, uintptr_t bias,
             int seq_len, int dim, uintptr_t stream) {
        bias_residual_strict_fp16(
            reinterpret_cast<__half*>(residual),
            reinterpret_cast<const __half*>(x),
            reinterpret_cast<const __half*>(bias),
            seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("bias_residual_layer_norm_fp16",
          [](uintptr_t residual, uintptr_t x, uintptr_t bias_pre,
             uintptr_t ln_weight, uintptr_t ln_bias, uintptr_t out,
             int seq_len, int dim, float eps, uintptr_t stream) {
        bias_residual_layer_norm_fp16(
            reinterpret_cast<__half*>(residual),
            reinterpret_cast<const __half*>(x),
            reinterpret_cast<const __half*>(bias_pre),
            reinterpret_cast<const __half*>(ln_weight),
            reinterpret_cast<const __half*>(ln_bias),
            reinterpret_cast<__half*>(out),
            seq_len, dim, eps, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias_pre"),
       py::arg("ln_weight"), py::arg("ln_bias"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("stream") = 0);

    m.def("residual_add_fp16", [](uintptr_t residual, uintptr_t x, int n, uintptr_t stream) {
        residual_add_fp16(reinterpret_cast<__half*>(residual), reinterpret_cast<const __half*>(x),
                           n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("gate_mul_residual_fp16",
          [](uintptr_t residual, uintptr_t x, uintptr_t gate,
             int n, uintptr_t stream) {
        gate_mul_residual_fp16(
            reinterpret_cast<__half*>(residual),
            reinterpret_cast<const __half*>(x),
            reinterpret_cast<const __half*>(gate),
            n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"),
       py::arg("n"), py::arg("stream") = 0);

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

    m.def("bias_gelu_fp16_strict", [](uintptr_t x, uintptr_t bias,
                                       int seq_len, int dim, uintptr_t stream) {
        cudaStream_t s = to_stream(stream);
        add_bias_fp16(reinterpret_cast<__half*>(x),
                      reinterpret_cast<const __half*>(bias),
                      seq_len, dim, s);
        gelu_inplace_fp16(reinterpret_cast<__half*>(x), seq_len * dim, s);
    }, py::arg("x"), py::arg("bias"), py::arg("seq_len"), py::arg("dim"),
       py::arg("stream") = 0);

    m.def("gate_geglu_merged_fp16", [](uintptr_t merged, uintptr_t out,
                                        int seq, int half_dim, uintptr_t stream) {
        gate_silu_mul_merged_fp16(reinterpret_cast<const __half*>(merged),
                                   reinterpret_cast<__half*>(out), seq, half_dim, to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"), py::arg("stream") = 0);

    m.def("mul_fp16", [](uintptr_t a, uintptr_t b, uintptr_t out,
                         int n, uintptr_t stream) {
        mul_fp16(reinterpret_cast<const __half*>(a),
                 reinterpret_cast<const __half*>(b),
                 reinterpret_cast<__half*>(out), n, to_stream(stream));
    }, py::arg("a"), py::arg("b"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

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

    m.def("gmm_fp16_alpha", [](FvkContext& ctx, uintptr_t A, uintptr_t B, uintptr_t C,
                               int M, int N, int K, float alpha, float beta,
                               uintptr_t stream) {
        gmm_fp16_alpha(ctx.cublas_handle,
                       reinterpret_cast<const __half*>(A),
                       reinterpret_cast<const __half*>(B),
                       reinterpret_cast<__half*>(C),
                       M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("ctx"), py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"),
       py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("gmm_fp16_out_fp32", [](FvkContext& ctx, uintptr_t A, uintptr_t B, uintptr_t C,
                                  int M, int N, int K, uintptr_t stream) {
        gmm_fp16_out_fp32(ctx.cublas_handle,
                          reinterpret_cast<const __half*>(A),
                          reinterpret_cast<const __half*>(B),
                          reinterpret_cast<float*>(C),
                          M, N, K, to_stream(stream));
    }, py::arg("ctx"), py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def("action_update_from_fp32", [](uintptr_t delta, uintptr_t bias, uintptr_t noise,
                                        int S, int D, float dt, bool residual,
                                        uintptr_t stream) {
        action_update_from_fp32(reinterpret_cast<const float*>(delta),
                                reinterpret_cast<const __half*>(bias),
                                reinterpret_cast<__half*>(noise),
                                S, D, dt, residual, to_stream(stream));
    }, py::arg("delta"), py::arg("bias"), py::arg("noise"),
       py::arg("S"), py::arg("D"), py::arg("dt"),
       py::arg("residual") = true, py::arg("stream") = 0);

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

    // Fixed-shape cuBLAS decomposed attention. S_kv_max is graph-captured;
    // seqused_k is a device int32[1] containing the valid K/V rows.
    m.def("attention_qkv_fp16_seqused", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S, int S_kv_max, int NH, int HD,
                                    uintptr_t seqused_k, float attn_scale, uintptr_t stream) {
        attention_qkv_fp16_seqused(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S, S_kv_max, NH, HD,
                            reinterpret_cast<const int*>(seqused_k),
                            attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S"), py::arg("S_kv_max"), py::arg("NH"), py::arg("HD"),
       py::arg("seqused_k"), py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

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

    // ── CUTLASS FP16 GEMMs (FP16 path, NT layout — B is [N,K] row-major) ──
    m.def("cutlass_fp16_plain", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_plain(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp16_sq", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                  int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_sq(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp16_t1", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                  int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_t1(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp16_wide", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                    int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_wide(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp16_k64", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                  int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_k64(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp16_2sm21", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                    int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_2sm21(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp16_k64_gelu", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                       int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_k64_gelu(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp16_sq_gelu", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                      int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_sq_gelu(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp16_k64_mul_aux", [](uintptr_t A, uintptr_t B, uintptr_t Aux, uintptr_t D,
                                          int M, int N, int K, uintptr_t stream) {
        return cutlass_fp16_k64_mul_aux(to_ptr(A), to_ptr(B), to_ptr(Aux), to_ptr(D),
                                         M, N, K, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("Aux"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    // Path C foundation: vendored production sm100_gemm_tma_warpspecialized
    // kernel struct, single GEMM.  Must match cutlass_fp16_sq isolation
    // perf before extending to multi-mainloop megakernel.
    m.def("flashrt_megakernel_single_fp16",
          [](uintptr_t A, uintptr_t B, uintptr_t D, int M, int N, int K,
             float alpha, float beta, uintptr_t stream) {
              return flashrt_megakernel_single_fp16(
                  to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                  alpha, beta, to_stream(stream));
          },
          py::arg("A"), py::arg("B"), py::arg("D"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("stream") = 0);

    // GeGLU megakernel: one launch computes hidden = GELU(X @ W_gate) * (X @ W_up).
    // The two sub-GEMMs share one tcgen05.alloc; each CTA work tile runs the
    // gate GEMM then the up GEMM in series via a single AccumulatorPipeline.
    m.def("flashrt_megakernel_geglu_fp16",
          [](uintptr_t X, uintptr_t W_gate, uintptr_t W_up,
             uintptr_t D_gate_scratch, uintptr_t hidden,
             int M, int N, int K, uintptr_t stream) {
              return flashrt_megakernel_geglu_fp16(
                  to_ptr(X), to_ptr(W_gate), to_ptr(W_up),
                  to_ptr(D_gate_scratch), to_ptr(hidden),
                  M, N, K, to_stream(stream));
          },
          py::arg("X"), py::arg("W_gate"), py::arg("W_up"),
          py::arg("D_gate_scratch"), py::arg("hidden"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("stream") = 0);

    // Fused GeGLU + down-proj with residual: the GeGLU megakernel and the
    // down GEMM (beta=1) are bundled inside one .cu entry (two launches).
    m.def("flashrt_megakernel_geglu_g8_fp16",
          [](uintptr_t X, uintptr_t W_gate, uintptr_t W_up,
             uintptr_t W_down,
             uintptr_t hidden_scratch, uintptr_t x_inout,
             int M, int H, int D, uintptr_t stream) {
              return flashrt_megakernel_geglu_g8_fp16(
                  to_ptr(X), to_ptr(W_gate), to_ptr(W_up),
                  to_ptr(W_down),
                  to_ptr(hidden_scratch), to_ptr(x_inout),
                  M, H, D, to_stream(stream));
          },
          py::arg("X"), py::arg("W_gate"), py::arg("W_up"),
          py::arg("W_down"),
          py::arg("hidden_scratch"), py::arg("x_inout"),
          py::arg("M"), py::arg("H"), py::arg("D"),
          py::arg("stream") = 0);

    // Bundle: rms_norm + QKV (k64) in one C entry.
    m.def("flashrt_rms_qkv_fp16",
          [](uintptr_t x, uintptr_t rms_weight, uintptr_t x_norm,
             uintptr_t qkv_weight, uintptr_t qkv_out,
             int M, int D, int N_qkv, float rms_eps, uintptr_t stream) {
              return flashrt_rms_qkv_fp16(
                  to_ptr(x), to_ptr(rms_weight), to_ptr(x_norm),
                  to_ptr(qkv_weight), to_ptr(qkv_out),
                  M, D, N_qkv, rms_eps, to_stream(stream));
          },
          py::arg("x"), py::arg("rms_weight"), py::arg("x_norm"),
          py::arg("qkv_weight"), py::arg("qkv_out"),
          py::arg("M"), py::arg("D"), py::arg("N_qkv"),
          py::arg("rms_eps") = 1e-6f, py::arg("stream") = 0);

    // Bundled FFN block: rms_norm + GeGLU megakernel + down-proj + residual
    // in one C entry.
    m.def("flashrt_encoder_ffn_block_fp16",
          [](uintptr_t x_resid, uintptr_t rms_weight, uintptr_t x_norm,
             uintptr_t W_gate, uintptr_t W_up, uintptr_t W_down,
             uintptr_t gate_scratch, uintptr_t hidden_scratch,
             int M, int H, int D, float rms_eps, uintptr_t stream) {
              return flashrt_encoder_ffn_block_fp16(
                  to_ptr(x_resid), to_ptr(rms_weight), to_ptr(x_norm),
                  to_ptr(W_gate), to_ptr(W_up), to_ptr(W_down),
                  to_ptr(gate_scratch), to_ptr(hidden_scratch),
                  M, H, D, rms_eps, to_stream(stream));
          },
          py::arg("x_resid"), py::arg("rms_weight"), py::arg("x_norm"),
          py::arg("W_gate"), py::arg("W_up"), py::arg("W_down"),
          py::arg("gate_scratch"), py::arg("hidden_scratch"),
          py::arg("M"), py::arg("H"), py::arg("D"),
          py::arg("rms_eps") = 1e-6f, py::arg("stream") = 0);

#ifdef FLASHRT_HAVE_SM100_ENCODER_MLP
    // Path C encoder MLP megakernel scaffold (WIP, off by default; built only
    // with -DFLASHRT_BUILD_SM100_ENCODER_MLP=ON).
    m.def("encoder_mlp_fused_fp16",
          [](uintptr_t X, uintptr_t W_gate, uintptr_t W_up, uintptr_t W_down,
             uintptr_t gate_buf, uintptr_t up_buf, uintptr_t hid_buf, uintptr_t out,
             int M, int N_out, int K, int H, uintptr_t stream) {
              return encoder_mlp_fused_fp16(
                  to_ptr(X), to_ptr(W_gate), to_ptr(W_up), to_ptr(W_down),
                  to_ptr(gate_buf), to_ptr(up_buf), to_ptr(hid_buf), to_ptr(out),
                  M, N_out, K, H, to_stream(stream));
          },
          py::arg("X"), py::arg("W_gate"), py::arg("W_up"), py::arg("W_down"),
          py::arg("gate_buf"), py::arg("up_buf"), py::arg("hid_buf"), py::arg("out"),
          py::arg("M"), py::arg("N_out"), py::arg("K"), py::arg("H"),
          py::arg("stream") = 0);
#endif

#ifdef FLASHRT_HAVE_SM100_SWEEP
    // FP16 tile-sweep bench dispatch (off by default; built only with
    // -DFLASHRT_BUILD_SM100_SWEEP=ON).
    m.def("cutlass_fp16_sweep", [](int variant, uintptr_t A, uintptr_t B, uintptr_t D,
                                    int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp16_sweep(variant, to_ptr(A), to_ptr(B), to_ptr(D),
                                  M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("variant"), py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);
    m.def("cutlass_fp16_sweep_count", []() { return cutlass_fp16_sweep_count(); });
#endif

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

    m.def("concat2_bf16", [](uintptr_t a, uintptr_t b, uintptr_t out,
                             int rows, int cols_a, int cols_b,
                             uintptr_t stream) {
        concat2_bf16(
            reinterpret_cast<const __nv_bfloat16*>(a),
            reinterpret_cast<const __nv_bfloat16*>(b),
            reinterpret_cast<__nv_bfloat16*>(out),
            rows, cols_a, cols_b, to_stream(stream));
    }, py::arg("a"), py::arg("b"), py::arg("out"),
       py::arg("rows"), py::arg("cols_a"), py::arg("cols_b"),
       py::arg("stream") = 0);

    m.def("cuda_read_i32_sync", [](uintptr_t src, uintptr_t stream) {
        int host = 0;
        cudaStream_t st = to_stream(stream);
        cudaError_t err = cudaMemcpyAsync(
            &host, reinterpret_cast<const void*>(src), sizeof(int),
            cudaMemcpyDeviceToHost, st);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(st);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        return host;
    }, py::arg("src"), py::arg("stream") = 0);

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

    m.def("relu_inplace_bf16", [](uintptr_t x, int n, uintptr_t stream) {
        extern void relu_inplace_bf16(__nv_bfloat16*, int, cudaStream_t);
        relu_inplace_bf16(reinterpret_cast<__nv_bfloat16*>(x), n, to_stream(stream));
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
    //  FP8 block-128 dequantization + GEMM.
    //  Used by Qwen3.6-27B as a portable fallback when native
    //  block-scaled FP8 GEMM is unavailable or not selected.
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

    m.def("tq_fp32_gemm_fp32",
        [](uintptr_t a_fp32, uintptr_t b_fp32, uintptr_t c_fp32,
           int M, int N, int K, uintptr_t stream) {
            tq_fp32_gemm_fp32_launch(
                to_ptr(a_fp32), to_ptr(b_fp32),
                reinterpret_cast<void*>(c_fp32),
                M, N, K, to_stream(stream));
        },
        py::arg("a_fp32"), py::arg("b_fp32"), py::arg("c_fp32"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    m.def("tq_fp32_gemm_fp32_bt",
        [](uintptr_t a_fp32, uintptr_t b_fp32, uintptr_t c_fp32,
           int M, int N, int K, uintptr_t stream) {
            tq_fp32_gemm_fp32_bt_launch(
                to_ptr(a_fp32), to_ptr(b_fp32),
                reinterpret_cast<void*>(c_fp32),
                M, N, K, to_stream(stream));
        },
        py::arg("a_fp32"), py::arg("b_fp32"), py::arg("c_fp32"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    m.def("tq_fp32_gemm_lt",
        [](uintptr_t a_fp32, uintptr_t b_fp32, uintptr_t c_fp32,
           int M, int N, int K, uintptr_t stream) {
            tq_fp32_gemm_lt_launch(
                to_ptr(a_fp32), to_ptr(b_fp32),
                reinterpret_cast<void*>(c_fp32),
                M, N, K, to_stream(stream));
        },
        py::arg("a_fp32"), py::arg("b_fp32"), py::arg("c_fp32"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    m.def("tq_fp32_gemm_lt_bt",
        [](uintptr_t a_fp32, uintptr_t b_fp32, uintptr_t c_fp32,
           int M, int N, int K, uintptr_t stream) {
            tq_fp32_gemm_lt_bt_launch(
                to_ptr(a_fp32), to_ptr(b_fp32),
                reinterpret_cast<void*>(c_fp32),
                M, N, K, to_stream(stream));
        },
        py::arg("a_fp32"), py::arg("b_fp32"), py::arg("c_fp32"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    m.def("tq_fp32_gemm_lt_algo",
        [](uintptr_t a_fp32, uintptr_t b_fp32, uintptr_t c_fp32,
           int M, int N, int K, int algo_idx, uintptr_t stream) {
            tq_fp32_gemm_lt_algo_launch(
                to_ptr(a_fp32), to_ptr(b_fp32),
                reinterpret_cast<void*>(c_fp32),
                M, N, K, algo_idx, to_stream(stream));
        },
        py::arg("a_fp32"), py::arg("b_fp32"), py::arg("c_fp32"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("algo_idx"), py::arg("stream") = 0);

    m.def("tq_fp32_gemm_lt_bt_algo",
        [](uintptr_t a_fp32, uintptr_t b_fp32, uintptr_t c_fp32,
           int M, int N, int K, int algo_idx, uintptr_t stream) {
            tq_fp32_gemm_lt_bt_algo_launch(
                to_ptr(a_fp32), to_ptr(b_fp32),
                reinterpret_cast<void*>(c_fp32),
                M, N, K, algo_idx, to_stream(stream));
        },
        py::arg("a_fp32"), py::arg("b_fp32"), py::arg("c_fp32"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("algo_idx"), py::arg("stream") = 0);

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

    // G7.7 — Fused IM2COL + FP8 e4m3 quantize for 3x3x3 stride-1
    // already-padded Conv3d. Caller pads x with F.pad to (T_pad, H_pad, W_pad)
    // first; this kernel emits col_fp8 (M=B*To*Ho*Wo, K=27*Ci) ready for
    // GemmRunner.fp8_nn_dev. Per-tensor act_scale (device fp32 scalar).

    // G7.21 — IM2COL+FP8 v2 with shared-memory tile (sm_120-tuned).

    // G7.10 — Fused (add_bias + GELU(tanh) + per-tensor FP8 quantize)
    // for FP8 FFN intermediate. bias may be 0 (passed as null pointer).
    m.def("bias_gelu_quantize_fp8_static_bf16",
        [](uintptr_t in_bf16, uintptr_t bias_bf16, uintptr_t out_fp8,
           uintptr_t act_scale, long long M, int N, uintptr_t stream) {
            flash_rt::quantize::bias_gelu_quantize_fp8_static_bf16(
                to_ptr(in_bf16),
                bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                to_ptr(out_fp8),
                reinterpret_cast<const float*>(act_scale),
                M, N, to_stream(stream));
        },
        py::arg("in_bf16"), py::arg("bias_bf16"),
        py::arg("out_fp8"), py::arg("act_scale"),
        py::arg("M"), py::arg("N"),
        py::arg("stream") = 0);

    // G7.15 — Fused 3D RoPE apply (bf16 → fp32) replacing 5-6 Python
    // launches per call with one CUDA kernel.
    m.def("rope_apply_bf16_to_fp32",
        [](uintptr_t in_bf16, uintptr_t freqs_re, uintptr_t freqs_im,
           uintptr_t out_fp32, int B, int T, int N, int head_dim,
           int seq_len, uintptr_t stream) {
            flash_rt::quantize::rope_apply_bf16_to_fp32(
                to_ptr(in_bf16),
                reinterpret_cast<const float*>(freqs_re),
                reinterpret_cast<const float*>(freqs_im),
                to_ptr(out_fp32),
                B, T, N, head_dim, seq_len, to_stream(stream));
        },
        py::arg("in_bf16"), py::arg("freqs_re"), py::arg("freqs_im"),
        py::arg("out_fp32"), py::arg("B"), py::arg("T"), py::arg("N"),
        py::arg("head_dim"), py::arg("seq_len"),
        py::arg("stream") = 0);

    // G7.16 — bf16 output variant of rope_apply (keeps cat in bf16
    // so FA2 dispatches its bf16 tensor-core fast path).
    m.def("rope_apply_bf16_to_bf16",
        [](uintptr_t in_bf16, uintptr_t freqs_re, uintptr_t freqs_im,
           uintptr_t out_bf16, int B, int T, int N, int head_dim,
           int seq_len, uintptr_t stream) {
            flash_rt::quantize::rope_apply_bf16_to_bf16(
                to_ptr(in_bf16),
                reinterpret_cast<const float*>(freqs_re),
                reinterpret_cast<const float*>(freqs_im),
                to_ptr(out_bf16),
                B, T, N, head_dim, seq_len, to_stream(stream));
        },
        py::arg("in_bf16"), py::arg("freqs_re"), py::arg("freqs_im"),
        py::arg("out_bf16"), py::arg("B"), py::arg("T"), py::arg("N"),
        py::arg("head_dim"), py::arg("seq_len"),
        py::arg("stream") = 0);

    // G7.17 — Fused AdaLayerNorm + per-tensor FP8 quantize. Replaces
    // the motus 2-launch chain (ada_layer_norm_bf16 + quantize_fp8_static)
    // with one kernel; eliminates the bf16 intermediate buffer
    // round-trip (memory-bound, dominant cost at T=2520).
    m.def("ada_layer_norm_fp8",
        [](uintptr_t x_bf16, uintptr_t scale_bf16, uintptr_t shift_bf16,
           uintptr_t out_fp8, uintptr_t act_scale,
           int seq_len, int dim, float eps, uintptr_t stream) {
            flash_rt::quantize::ada_layer_norm_fp8(
                to_ptr(x_bf16), to_ptr(scale_bf16), to_ptr(shift_bf16),
                to_ptr(out_fp8),
                reinterpret_cast<const float*>(act_scale),
                seq_len, dim, eps, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("scale_bf16"), py::arg("shift_bf16"),
        py::arg("out_fp8"), py::arg("act_scale"),
        py::arg("seq_len"), py::arg("dim"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("ada_layer_norm_nvfp4_swizzled",
        [](uintptr_t x_bf16, uintptr_t scale_bf16, uintptr_t shift_bf16,
           uintptr_t packed_u8, uintptr_t sf_swizzled_u8,
           int seq_len, int dim, float eps, uintptr_t stream) {
            flash_rt::quantize::ada_layer_norm_nvfp4_swizzled(
                to_ptr(x_bf16), to_ptr(scale_bf16), to_ptr(shift_bf16),
                to_ptr(packed_u8), to_ptr(sf_swizzled_u8),
                seq_len, dim, eps, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("scale_bf16"), py::arg("shift_bf16"),
        py::arg("packed_u8"), py::arg("sf_swizzled_u8"),
        py::arg("seq_len"), py::arg("dim"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // G7.23 — Fused {RMS_norm + SiLU + per-tensor FP8 quantize + NCDHW→NDHWC}
    // for motus VAE ResidualBlock chain. Replaces 3 BF16 launches + 1 permute
    // with one kernel; halves memory traffic on the VAE feature path.

    // G7.23 v19 — bare BF16 -> FP8 quant with NCDHW -> NDHWC permute.
    // Used by the (1,1,1) shortcut conv FP8 path (no RMS / SiLU / gamma).
    m.def("bf16_quant_fp8_ncdhw_to_ndhwc",
        [](uintptr_t x_bf16, uintptr_t y_fp8,
           int B, int C, int T, int H, int W,
           float act_scale, uintptr_t stream) {
            return flash_rt::quantize::bf16_quant_fp8_ncdhw_to_ndhwc(
                to_ptr(x_bf16), to_ptr(y_fp8),
                B, C, T, H, W, act_scale, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("y_fp8"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("act_scale"), py::arg("stream") = 0);

    // G7.23 — fast 5D BF16 NDHWC -> NCDHW transpose (replaces aten's
    // generic .permute().contiguous() copy on the v17 conv output).
    m.def("bf16_ndhwc_to_ncdhw_transpose",
        [](uintptr_t x_NDHWC, uintptr_t y_NCDHW,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            return flash_rt::quantize::bf16_ndhwc_to_ncdhw_transpose(
                to_ptr(x_NDHWC), to_ptr(y_NCDHW),
                B, C, T, H, W, to_stream(stream));
        },
        py::arg("x_NDHWC"), py::arg("y_NCDHW"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0);

    m.def("bf16_ndhwc_to_ncdhw_bias_bf16",
        [](uintptr_t x_NDHWC, uintptr_t bias_C, uintptr_t y_NCDHW,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            return flash_rt::quantize::bf16_ndhwc_to_ncdhw_bias_bf16(
                to_ptr(x_NDHWC), to_ptr(bias_C), to_ptr(y_NCDHW),
                B, C, T, H, W, to_stream(stream));
        },
        py::arg("x_NDHWC"), py::arg("bias_C"), py::arg("y_NCDHW"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0);

    // G7.23 v4 — v3 + x cached in regs (bf162 packed); 2-4× CTAs/SM gain.
    m.def("bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4",
        [](uintptr_t x_bf16, uintptr_t gamma_bf16, uintptr_t y_fp8,
           int B, int C, int T, int H, int W,
           float act_scale, float eps, uintptr_t stream) {
            return flash_rt::quantize::bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4(
                to_ptr(x_bf16), to_ptr(gamma_bf16), to_ptr(y_fp8),
                B, C, T, H, W, act_scale, eps, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("gamma_bf16"), py::arg("y_fp8"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("act_scale"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // G7.23 v3 — v2 + uint32 vec smem read (eliminates 4-8 way bank conflict).

    // G7.23 v2 — single-pass + smem-transposed coalesced write. Same API.

    // G7.19 — Fused QKV split + WanRMSNorm + 3D RoPE for Wan video Q/K.
    // Replaces 5+ launches (split×3 + norm_q + norm_k + rope×2) with 1.
    m.def("qkv_split_norm_rope_bf16",
        [](uintptr_t packed_qkv, uintptr_t norm_q_w, uintptr_t norm_k_w,
           uintptr_t freqs_re, uintptr_t freqs_im,
           uintptr_t q_rope_out, uintptr_t k_rope_out,
           int B, int L_v, int N, int D_h, int seq_len, float eps,
           uintptr_t stream) {
            flash_rt::quantize::qkv_split_norm_rope_bf16(
                to_ptr(packed_qkv), to_ptr(norm_q_w), to_ptr(norm_k_w),
                reinterpret_cast<const float*>(freqs_re),
                reinterpret_cast<const float*>(freqs_im),
                to_ptr(q_rope_out), to_ptr(k_rope_out),
                B, L_v, N, D_h, seq_len, eps, to_stream(stream));
        },
        py::arg("packed_qkv"), py::arg("norm_q_w"), py::arg("norm_k_w"),
        py::arg("freqs_re"), py::arg("freqs_im"),
        py::arg("q_rope_out"), py::arg("k_rope_out"),
        py::arg("B"), py::arg("L_v"), py::arg("N"), py::arg("D_h"),
        py::arg("seq_len"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    // In-place RMSnorm + RoPE on joint Q/K (pairs with qkv_scatter_mega)

    // G7.14 — Fused (per-K AWQ inv_s mul + per-tensor static FP8
    // quantize) for AWQ FP8 sites in action_expert + und_expert.
    m.def("awq_quant_fp8_static_bf16",
        [](uintptr_t in_bf16, uintptr_t inv_s_bf16, uintptr_t out_fp8,
           uintptr_t act_scale, long long M, int K, uintptr_t stream) {
            flash_rt::quantize::awq_quant_fp8_static_bf16(
                to_ptr(in_bf16),
                to_ptr(inv_s_bf16),
                to_ptr(out_fp8),
                reinterpret_cast<const float*>(act_scale),
                M, K, to_stream(stream));
        },
        py::arg("in_bf16"), py::arg("inv_s_bf16"),
        py::arg("out_fp8"), py::arg("act_scale"),
        py::arg("M"), py::arg("K"),
        py::arg("stream") = 0);

    // Motus 205ms path bindings. These are the production fused kernels
    // used by the cleaned Motus frontend; probe/test kernels stay archived.
    m.def("dequantize_fp8_static_bf16",
        [](uintptr_t input, uintptr_t output, uintptr_t d_scale,
           int n, uintptr_t stream) {
            dequantize_fp8_static_bf16(
                typed_ptr<__nv_fp8_e4m3>(input),
                typed_ptr<__nv_bfloat16>(output),
                reinterpret_cast<const float*>(d_scale),
                n, to_stream(stream));
        },
        py::arg("input"), py::arg("output"), py::arg("d_scale"),
        py::arg("n"), py::arg("stream") = 0);

    m.def("dequantize_fp8_static_bf16_2",
        [](uintptr_t in0, uintptr_t in1,
           uintptr_t out0, uintptr_t out1,
           uintptr_t s0, uintptr_t s1,
           int n, uintptr_t stream) {
            dequantize_fp8_static_bf16_2(
                typed_ptr<__nv_fp8_e4m3>(in0),
                typed_ptr<__nv_fp8_e4m3>(in1),
                typed_ptr<__nv_bfloat16>(out0),
                typed_ptr<__nv_bfloat16>(out1),
                reinterpret_cast<const float*>(s0),
                reinterpret_cast<const float*>(s1),
                n, to_stream(stream));
        },
        py::arg("in0"), py::arg("in1"),
        py::arg("out0"), py::arg("out1"),
        py::arg("s0"), py::arg("s1"),
        py::arg("n"), py::arg("stream") = 0);

    m.def("dequantize_fp8_static_bf16_6",
        [](uintptr_t in0, uintptr_t in1, uintptr_t in2,
           uintptr_t in3, uintptr_t in4, uintptr_t in5,
           uintptr_t out0, uintptr_t out1, uintptr_t out2,
           uintptr_t out3, uintptr_t out4, uintptr_t out5,
           uintptr_t s0, uintptr_t s1, uintptr_t s2,
           uintptr_t s3, uintptr_t s4, uintptr_t s5,
           int n, uintptr_t stream) {
            dequantize_fp8_static_bf16_6(
                typed_ptr<__nv_fp8_e4m3>(in0),
                typed_ptr<__nv_fp8_e4m3>(in1),
                typed_ptr<__nv_fp8_e4m3>(in2),
                typed_ptr<__nv_fp8_e4m3>(in3),
                typed_ptr<__nv_fp8_e4m3>(in4),
                typed_ptr<__nv_fp8_e4m3>(in5),
                typed_ptr<__nv_bfloat16>(out0),
                typed_ptr<__nv_bfloat16>(out1),
                typed_ptr<__nv_bfloat16>(out2),
                typed_ptr<__nv_bfloat16>(out3),
                typed_ptr<__nv_bfloat16>(out4),
                typed_ptr<__nv_bfloat16>(out5),
                reinterpret_cast<const float*>(s0),
                reinterpret_cast<const float*>(s1),
                reinterpret_cast<const float*>(s2),
                reinterpret_cast<const float*>(s3),
                reinterpret_cast<const float*>(s4),
                reinterpret_cast<const float*>(s5),
                n, to_stream(stream));
        },
        py::arg("in0"), py::arg("in1"), py::arg("in2"),
        py::arg("in3"), py::arg("in4"), py::arg("in5"),
        py::arg("out0"), py::arg("out1"), py::arg("out2"),
        py::arg("out3"), py::arg("out4"), py::arg("out5"),
        py::arg("s0"), py::arg("s1"), py::arg("s2"),
        py::arg("s3"), py::arg("s4"), py::arg("s5"),
        py::arg("n"), py::arg("stream") = 0);

    m.def("adaln_modulation6_bf16",
        [](uintptr_t adaln_params, uintptr_t layer_modulation,
           uintptr_t out0, uintptr_t out1, uintptr_t out2,
           uintptr_t out3, uintptr_t out4, uintptr_t out5,
           int B, int S, int D, uintptr_t stream) {
            adaln_modulation6_bf16(
                reinterpret_cast<const float*>(adaln_params),
                reinterpret_cast<const float*>(layer_modulation),
                typed_ptr<__nv_bfloat16>(out0),
                typed_ptr<__nv_bfloat16>(out1),
                typed_ptr<__nv_bfloat16>(out2),
                typed_ptr<__nv_bfloat16>(out3),
                typed_ptr<__nv_bfloat16>(out4),
                typed_ptr<__nv_bfloat16>(out5),
                B, S, D, to_stream(stream));
        },
        py::arg("adaln_params"), py::arg("layer_modulation"),
        py::arg("out0"), py::arg("out1"), py::arg("out2"),
        py::arg("out3"), py::arg("out4"), py::arg("out5"),
        py::arg("B"), py::arg("S"), py::arg("D"),
        py::arg("stream") = 0);

    m.def("ada_layer_norm_bf16_per_token",
        [](uintptr_t x, uintptr_t scale, uintptr_t shift, uintptr_t out,
           int seq_len, int dim, float eps, uintptr_t stream) {
            ada_layer_norm_bf16_per_token(
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(scale),
                typed_ptr<__nv_bfloat16>(shift),
                typed_ptr<__nv_bfloat16>(out),
                seq_len, dim, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("scale"), py::arg("shift"), py::arg("out"),
        py::arg("seq_len"), py::arg("dim"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("ada_layer_norm_fp8_modfp8",
        [](uintptr_t x_bf16, uintptr_t scale_fp8, uintptr_t shift_fp8,
           uintptr_t scale_deq, uintptr_t shift_deq,
           uintptr_t out_fp8, uintptr_t act_scale,
           int seq_len, int dim, float eps, uintptr_t stream) {
            flash_rt::quantize::ada_layer_norm_fp8_modfp8(
                to_ptr(x_bf16), to_ptr(scale_fp8), to_ptr(shift_fp8),
                reinterpret_cast<const float*>(scale_deq),
                reinterpret_cast<const float*>(shift_deq),
                to_ptr(out_fp8),
                reinterpret_cast<const float*>(act_scale),
                seq_len, dim, eps, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("scale_fp8"), py::arg("shift_fp8"),
        py::arg("scale_deq"), py::arg("shift_deq"),
        py::arg("out_fp8"), py::arg("act_scale"),
        py::arg("seq_len"), py::arg("dim"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("ada_layer_norm_nvfp4_swizzled_modfp8",
        [](uintptr_t x_bf16, uintptr_t scale_fp8, uintptr_t shift_fp8,
           uintptr_t scale_deq, uintptr_t shift_deq,
           uintptr_t packed_u8, uintptr_t sf_swizzled_u8,
           int seq_len, int dim, float eps, uintptr_t stream) {
            flash_rt::quantize::ada_layer_norm_nvfp4_swizzled_modfp8(
                to_ptr(x_bf16), to_ptr(scale_fp8), to_ptr(shift_fp8),
                reinterpret_cast<const float*>(scale_deq),
                reinterpret_cast<const float*>(shift_deq),
                to_ptr(packed_u8), to_ptr(sf_swizzled_u8),
                seq_len, dim, eps, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("scale_fp8"), py::arg("shift_fp8"),
        py::arg("scale_deq"), py::arg("shift_deq"),
        py::arg("packed_u8"), py::arg("sf_swizzled_u8"),
        py::arg("seq_len"), py::arg("dim"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("awq_ada_layer_norm_fp8",
        [](uintptr_t x_bf16, uintptr_t scale_bf16, uintptr_t shift_bf16,
           uintptr_t inv_s_bf16, uintptr_t out_fp8, uintptr_t act_scale,
           int seq_len, int dim, float eps, uintptr_t stream) {
            flash_rt::quantize::awq_ada_layer_norm_fp8(
                to_ptr(x_bf16), to_ptr(scale_bf16), to_ptr(shift_bf16),
                to_ptr(inv_s_bf16), to_ptr(out_fp8),
                reinterpret_cast<const float*>(act_scale),
                seq_len, dim, eps, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("scale_bf16"), py::arg("shift_bf16"),
        py::arg("inv_s_bf16"), py::arg("out_fp8"), py::arg("act_scale"),
        py::arg("seq_len"), py::arg("dim"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("awq_quant2_fp8_static_bf16",
        [](uintptr_t in0_bf16, uintptr_t inv_s0_bf16, uintptr_t out0_fp8,
           uintptr_t act_scale0, long long M0, int K0,
           uintptr_t in1_bf16, uintptr_t inv_s1_bf16, uintptr_t out1_fp8,
           uintptr_t act_scale1, long long M1, int K1, uintptr_t stream) {
            flash_rt::quantize::awq_quant2_fp8_static_bf16(
                to_ptr(in0_bf16), to_ptr(inv_s0_bf16), to_ptr(out0_fp8),
                reinterpret_cast<const float*>(act_scale0), M0, K0,
                to_ptr(in1_bf16), to_ptr(inv_s1_bf16), to_ptr(out1_fp8),
                reinterpret_cast<const float*>(act_scale1), M1, K1,
                to_stream(stream));
        },
        py::arg("in0_bf16"), py::arg("inv_s0_bf16"),
        py::arg("out0_fp8"), py::arg("act_scale0"),
        py::arg("M0"), py::arg("K0"),
        py::arg("in1_bf16"), py::arg("inv_s1_bf16"),
        py::arg("out1_fp8"), py::arg("act_scale1"),
        py::arg("M1"), py::arg("K1"), py::arg("stream") = 0);

    m.def("layer_norm_no_affine_fp8_static_bf16",
        [](uintptr_t x, uintptr_t out, uintptr_t d_scale,
           int seq_len, int dim, float eps, uintptr_t stream) {
            layer_norm_no_affine_fp8_static_bf16(
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_fp8_e4m3>(out),
                reinterpret_cast<const float*>(d_scale),
                seq_len, dim, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("out"), py::arg("d_scale"),
        py::arg("seq_len"), py::arg("dim"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("add_bf16_out",
        [](uintptr_t a, uintptr_t b, uintptr_t out, int n, uintptr_t stream) {
            add_bf16_out(typed_ptr<__nv_bfloat16>(a),
                         typed_ptr<__nv_bfloat16>(b),
                         typed_ptr<__nv_bfloat16>(out),
                         n, to_stream(stream));
        },
        py::arg("a"), py::arg("b"), py::arg("out"),
        py::arg("n"), py::arg("stream") = 0);

    m.def("euler_step_bf16_out",
        [](uintptr_t latent, uintptr_t velocity, uintptr_t out,
           float dt, int n, uintptr_t stream) {
            euler_step_bf16_out(typed_ptr<__nv_bfloat16>(latent),
                                typed_ptr<__nv_bfloat16>(velocity),
                                typed_ptr<__nv_bfloat16>(out),
                                dt, n, to_stream(stream));
        },
        py::arg("latent"), py::arg("velocity"), py::arg("out"),
        py::arg("dt"), py::arg("n"), py::arg("stream") = 0);

    m.def("teacher_force_first_frame_bf16",
        [](uintptr_t video_latent, uintptr_t cond_latent,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            teacher_force_first_frame_bf16(
                typed_ptr<__nv_bfloat16>(video_latent),
                typed_ptr<__nv_bfloat16>(cond_latent),
                B, C, T, H, W, to_stream(stream));
        },
        py::arg("video_latent"), py::arg("cond_latent"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0);

    m.def("motus_decode_postprocess_bf16_to_fp32",
        [](uintptr_t decoded, uintptr_t out,
           int B, int C, int T_in, int H, int W, uintptr_t stream) {
            motus_decode_postprocess_bf16_to_fp32(
                typed_ptr<__nv_bfloat16>(decoded),
                reinterpret_cast<float*>(out),
                B, C, T_in, H, W, to_stream(stream));
        },
        py::arg("decoded"), py::arg("out"),
        py::arg("B"), py::arg("C"), py::arg("T_in"),
        py::arg("H"), py::arg("W"), py::arg("stream") = 0);

    m.def("cast_bf16_to_fp32",
        [](uintptr_t src, uintptr_t dst, int n, uintptr_t stream) {
            cast_bf16_to_fp32(typed_ptr<__nv_bfloat16>(src),
                              reinterpret_cast<float*>(dst),
                              n, to_stream(stream));
        },
        py::arg("src"), py::arg("dst"), py::arg("n"),
        py::arg("stream") = 0);

    m.def("ncdhw_to_blc_bf16",
        [](uintptr_t x, uintptr_t out,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            ncdhw_to_blc_bf16(typed_ptr<__nv_bfloat16>(x),
                              typed_ptr<__nv_bfloat16>(out),
                              B, C, T, H, W, to_stream(stream));
        },
        py::arg("x"), py::arg("out"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0);

    m.def("dup_up3d_bf16",
        [](uintptr_t x, uintptr_t out,
           int B, int Cin, int Cout, int T, int H, int W,
           int factor_t, int factor_s, int repeats, int first_chunk,
           uintptr_t stream) {
            dup_up3d_bf16(typed_ptr<__nv_bfloat16>(x),
                          typed_ptr<__nv_bfloat16>(out),
                          B, Cin, Cout, T, H, W,
                          factor_t, factor_s, repeats, first_chunk,
                          to_stream(stream));
        },
        py::arg("x"), py::arg("out"),
        py::arg("B"), py::arg("Cin"), py::arg("Cout"),
        py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("factor_t"), py::arg("factor_s"),
        py::arg("repeats"), py::arg("first_chunk"),
        py::arg("stream") = 0);

    m.def("time_unshuffle2_bf16",
        [](uintptr_t x, uintptr_t out,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            time_unshuffle2_bf16(typed_ptr<__nv_bfloat16>(x),
                                 typed_ptr<__nv_bfloat16>(out),
                                 B, C, T, H, W, to_stream(stream));
        },
        py::arg("x"), py::arg("out"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0);

    m.def("add_bias_ncdhw_bf16",
        [](uintptr_t x, uintptr_t bias,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            add_bias_ncdhw_bf16(typed_ptr<__nv_bfloat16>(x),
                                typed_ptr<__nv_bfloat16>(bias),
                                B, C, T, H, W, to_stream(stream));
        },
        py::arg("x"), py::arg("bias"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0);

    m.def("update_cache2_ncdhw_bf16",
        [](uintptr_t cur, uintptr_t prev, uintptr_t out,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            update_cache2_ncdhw_bf16(
                typed_ptr<__nv_bfloat16>(cur),
                typed_ptr<__nv_bfloat16>(prev),
                typed_ptr<__nv_bfloat16>(out),
                B, C, T, H, W, to_stream(stream));
        },
        py::arg("cur"), py::arg("prev"), py::arg("out"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0);

    m.def("gate_mul_residual_out_bf16",
        [](uintptr_t residual, uintptr_t x, uintptr_t gate,
           uintptr_t out, int n, uintptr_t stream) {
            gate_mul_residual_out_bf16(
                typed_ptr<__nv_bfloat16>(residual),
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(gate),
                typed_ptr<__nv_bfloat16>(out),
                n, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("gate"),
        py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    m.def("gate_mul_residual_out_bf16_g1d",
        [](uintptr_t residual, uintptr_t x, uintptr_t gate_1d,
           uintptr_t out, int seq_len, int dim, uintptr_t stream) {
            gate_mul_residual_out_bf16_g1d(
                typed_ptr<__nv_bfloat16>(residual),
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(gate_1d),
                typed_ptr<__nv_bfloat16>(out),
                seq_len, dim, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("gate_1d"),
        py::arg("out"), py::arg("seq_len"), py::arg("dim"),
        py::arg("stream") = 0);

    m.def("gate_mul_residual_out_bf16_gate_fp8",
        [](uintptr_t residual, uintptr_t x, uintptr_t gate,
           uintptr_t gate_scale, uintptr_t out, int n, uintptr_t stream) {
            gate_mul_residual_out_bf16_gate_fp8(
                typed_ptr<__nv_bfloat16>(residual),
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_fp8_e4m3>(gate),
                reinterpret_cast<const float*>(gate_scale),
                typed_ptr<__nv_bfloat16>(out),
                n, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("gate"),
        py::arg("gate_scale"), py::arg("out"),
        py::arg("n"), py::arg("stream") = 0);

    m.def("bias_residual_out_bf16",
        [](uintptr_t residual, uintptr_t x, uintptr_t bias,
           uintptr_t out, int seq_len, int dim, uintptr_t stream) {
            bias_residual_out_bf16(
                typed_ptr<__nv_bfloat16>(residual),
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(bias),
                typed_ptr<__nv_bfloat16>(out),
                seq_len, dim, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("bias"),
        py::arg("out"), py::arg("seq_len"), py::arg("dim"),
        py::arg("stream") = 0);

    m.def("bias_gate_mul_residual_out_bf16",
        [](uintptr_t residual, uintptr_t x, uintptr_t bias, uintptr_t gate,
           uintptr_t out, int seq_len, int dim, uintptr_t stream) {
            bias_gate_mul_residual_out_bf16(
                typed_ptr<__nv_bfloat16>(residual),
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(bias),
                typed_ptr<__nv_bfloat16>(gate),
                typed_ptr<__nv_bfloat16>(out),
                seq_len, dim, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("bias"),
        py::arg("gate"), py::arg("out"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("bias_gate_mul_residual_out_bf16_g1d",
        [](uintptr_t residual, uintptr_t x, uintptr_t bias, uintptr_t gate_1d,
           uintptr_t out, int seq_len, int dim, uintptr_t stream) {
            bias_gate_mul_residual_out_bf16_g1d(
                typed_ptr<__nv_bfloat16>(residual),
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(bias),
                typed_ptr<__nv_bfloat16>(gate_1d),
                typed_ptr<__nv_bfloat16>(out),
                seq_len, dim, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("bias"),
        py::arg("gate_1d"), py::arg("out"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("bias_gate_mul_residual_out_bf16_gate_fp8",
        [](uintptr_t residual, uintptr_t x, uintptr_t bias,
           uintptr_t gate, uintptr_t gate_scale, uintptr_t out,
           int seq_len, int dim, uintptr_t stream) {
            bias_gate_mul_residual_out_bf16_gate_fp8(
                typed_ptr<__nv_bfloat16>(residual),
                typed_ptr<__nv_bfloat16>(x),
                typed_ptr<__nv_bfloat16>(bias),
                typed_ptr<__nv_fp8_e4m3>(gate),
                reinterpret_cast<const float*>(gate_scale),
                typed_ptr<__nv_bfloat16>(out),
                seq_len, dim, to_stream(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("bias"),
        py::arg("gate"), py::arg("gate_scale"), py::arg("out"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("bf16_rms_silu_ncdhw",
        [](uintptr_t x_bf16, uintptr_t gamma_bf16, uintptr_t y_bf16,
           uintptr_t prev_cache_bf16, uintptr_t next_cache_bf16,
           int B, int C, int T, int H, int W, float eps,
           uintptr_t stream) {
            return flash_rt::quantize::bf16_rms_silu_ncdhw(
                to_ptr(x_bf16), to_ptr(gamma_bf16), to_ptr(y_bf16),
                prev_cache_bf16 ? to_ptr(prev_cache_bf16) : nullptr,
                next_cache_bf16 ? to_ptr(next_cache_bf16) : nullptr,
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("gamma_bf16"), py::arg("y_bf16"),
        py::arg("prev_cache_bf16"), py::arg("next_cache_bf16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("bf16_rms_norm_ncdhw",
        [](uintptr_t x_bf16, uintptr_t gamma_bf16, uintptr_t bias_bf16,
           uintptr_t y_bf16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::quantize::bf16_rms_norm_ncdhw(
                to_ptr(x_bf16), to_ptr(gamma_bf16),
                bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                to_ptr(y_bf16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("gamma_bf16"), py::arg("bias_bf16"),
        py::arg("y_bf16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("bf16_pack_t1_cache3_nchw_channels_last",
        [](uintptr_t prev_cache_bf16, uintptr_t cur_bf16, uintptr_t out_bf16,
           int C, int H, int W, uintptr_t stream) {
            return flash_rt::quantize::bf16_pack_t1_cache3_nchw_channels_last(
                to_ptr(prev_cache_bf16), to_ptr(cur_bf16), to_ptr(out_bf16),
                C, H, W, to_stream(stream));
        },
        py::arg("prev_cache_bf16"), py::arg("cur_bf16"), py::arg("out_bf16"),
        py::arg("C"), py::arg("H"), py::arg("W"), py::arg("stream") = 0);

    m.def("bf16_ndhwc_to_ncdhw_add_bf16",
        [](uintptr_t x_NDHWC, uintptr_t residual_NCDHW, uintptr_t y_NCDHW,
           int B, int C, int T, int H, int W,
           long long rs_b, long long rs_c, long long rs_t,
           long long rs_h, long long rs_w, uintptr_t stream) {
            return flash_rt::quantize::bf16_ndhwc_to_ncdhw_add_bf16(
                to_ptr(x_NDHWC), to_ptr(residual_NCDHW), to_ptr(y_NCDHW),
                B, C, T, H, W, rs_b, rs_c, rs_t, rs_h, rs_w,
                to_stream(stream));
        },
        py::arg("x_NDHWC"), py::arg("residual_NCDHW"), py::arg("y_NCDHW"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("rs_b"), py::arg("rs_c"), py::arg("rs_t"),
        py::arg("rs_h"), py::arg("rs_w"), py::arg("stream") = 0);

    m.def("bf16_upsample2x_quant_fp8_nchw_to_nhwc",
        [](uintptr_t x_bf16, uintptr_t y_fp8,
           int N, int C, int H, int W, float act_scale, uintptr_t stream) {
            return flash_rt::quantize::bf16_upsample2x_quant_fp8_nchw_to_nhwc(
                to_ptr(x_bf16), to_ptr(y_fp8),
                N, C, H, W, act_scale, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("y_fp8"),
        py::arg("N"), py::arg("C"), py::arg("H"), py::arg("W"),
        py::arg("act_scale"), py::arg("stream") = 0);

    m.def("qkv_split_bias_norm_rope_v_bf16",
        [](uintptr_t packed_qkv, uintptr_t qkv_bias,
           uintptr_t norm_q_w, uintptr_t norm_k_w,
           uintptr_t freqs_re, uintptr_t freqs_im,
           uintptr_t q_rope_out, uintptr_t k_rope_out, uintptr_t v_out,
           int B, int L_v, int N, int D_h, int seq_len,
           float eps, uintptr_t stream) {
            flash_rt::quantize::qkv_split_bias_norm_rope_v_bf16(
                to_ptr(packed_qkv), to_ptr(qkv_bias),
                to_ptr(norm_q_w), to_ptr(norm_k_w),
                reinterpret_cast<const float*>(freqs_re),
                reinterpret_cast<const float*>(freqs_im),
                to_ptr(q_rope_out), to_ptr(k_rope_out), to_ptr(v_out),
                B, L_v, N, D_h, seq_len, eps, to_stream(stream));
        },
        py::arg("packed_qkv"), py::arg("qkv_bias"),
        py::arg("norm_q_w"), py::arg("norm_k_w"),
        py::arg("freqs_re"), py::arg("freqs_im"),
        py::arg("q_rope_out"), py::arg("k_rope_out"), py::arg("v_out"),
        py::arg("B"), py::arg("L_v"), py::arg("N"), py::arg("D_h"),
        py::arg("seq_len"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    m.def("qkv_split_bias_norm_rope_v_cat_bf16",
        [](uintptr_t packed_qkv, uintptr_t qkv_bias,
           uintptr_t norm_q_w, uintptr_t norm_k_w,
           uintptr_t freqs_re, uintptr_t freqs_im,
           uintptr_t q_cat_out, uintptr_t k_cat_out, uintptr_t v_cat_out,
           int B, int total_L, int video_offset, int L_v,
           int N, int D_h, int seq_len, float eps, uintptr_t stream) {
            flash_rt::quantize::qkv_split_bias_norm_rope_v_cat_bf16(
                to_ptr(packed_qkv), to_ptr(qkv_bias),
                to_ptr(norm_q_w), to_ptr(norm_k_w),
                reinterpret_cast<const float*>(freqs_re),
                reinterpret_cast<const float*>(freqs_im),
                to_ptr(q_cat_out), to_ptr(k_cat_out), to_ptr(v_cat_out),
                B, total_L, video_offset, L_v, N, D_h, seq_len, eps,
                to_stream(stream));
        },
        py::arg("packed_qkv"), py::arg("qkv_bias"),
        py::arg("norm_q_w"), py::arg("norm_k_w"),
        py::arg("freqs_re"), py::arg("freqs_im"),
        py::arg("q_cat_out"), py::arg("k_cat_out"), py::arg("v_cat_out"),
        py::arg("B"), py::arg("total_L"), py::arg("video_offset"),
        py::arg("L_v"), py::arg("N"), py::arg("D_h"),
        py::arg("seq_len"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    m.def("qkv_split_norm2_bf16",
        [](uintptr_t packed_a, uintptr_t norm_a_q_w, uintptr_t norm_a_k_w,
           uintptr_t q_a_out, uintptr_t k_a_out,
           int B, int L_a, int N, int D_h, float eps_a,
           uintptr_t packed_u, uintptr_t norm_u_q_w, uintptr_t norm_u_k_w,
           uintptr_t q_u_out, uintptr_t k_u_out,
           int L_u, float eps_u, uintptr_t stream) {
            flash_rt::quantize::qkv_split_norm2_bf16(
                to_ptr(packed_a), to_ptr(norm_a_q_w), to_ptr(norm_a_k_w),
                to_ptr(q_a_out), to_ptr(k_a_out),
                B, L_a, N, D_h, eps_a,
                to_ptr(packed_u), to_ptr(norm_u_q_w), to_ptr(norm_u_k_w),
                to_ptr(q_u_out), to_ptr(k_u_out), L_u, eps_u,
                to_stream(stream));
        },
        py::arg("packed_a"), py::arg("norm_a_q_w"), py::arg("norm_a_k_w"),
        py::arg("q_a_out"), py::arg("k_a_out"),
        py::arg("B"), py::arg("L_a"), py::arg("N"), py::arg("D_h"),
        py::arg("eps_a"),
        py::arg("packed_u"), py::arg("norm_u_q_w"), py::arg("norm_u_k_w"),
        py::arg("q_u_out"), py::arg("k_u_out"),
        py::arg("L_u"), py::arg("eps_u"), py::arg("stream") = 0);

    m.def("qkv_split_norm2_cat_bf16",
        [](uintptr_t packed_a, uintptr_t norm_a_q_w, uintptr_t norm_a_k_w,
           uintptr_t packed_u, uintptr_t norm_u_q_w, uintptr_t norm_u_k_w,
           uintptr_t q_cat_out, uintptr_t k_cat_out, uintptr_t v_cat_out,
           int B, int total_L, int L_v, int L_a, int L_u,
           int N, int D_h, float eps_a, float eps_u, uintptr_t stream) {
            flash_rt::quantize::qkv_split_norm2_cat_bf16(
                to_ptr(packed_a), to_ptr(norm_a_q_w), to_ptr(norm_a_k_w),
                to_ptr(packed_u), to_ptr(norm_u_q_w), to_ptr(norm_u_k_w),
                to_ptr(q_cat_out), to_ptr(k_cat_out), to_ptr(v_cat_out),
                B, total_L, L_v, L_a, L_u, N, D_h, eps_a, eps_u,
                to_stream(stream));
        },
        py::arg("packed_a"), py::arg("norm_a_q_w"), py::arg("norm_a_k_w"),
        py::arg("packed_u"), py::arg("norm_u_q_w"), py::arg("norm_u_k_w"),
        py::arg("q_cat_out"), py::arg("k_cat_out"), py::arg("v_cat_out"),
        py::arg("B"), py::arg("total_L"), py::arg("L_v"),
        py::arg("L_a"), py::arg("L_u"), py::arg("N"), py::arg("D_h"),
        py::arg("eps_a"), py::arg("eps_u"), py::arg("stream") = 0);

    m.def("qkv_split_joint3_cat_bf16",
        [](uintptr_t packed_v, uintptr_t qkv_v_bias,
           uintptr_t norm_v_q_w, uintptr_t norm_v_k_w,
           uintptr_t freqs_re, uintptr_t freqs_im,
           uintptr_t packed_a, uintptr_t norm_a_q_w, uintptr_t norm_a_k_w,
           uintptr_t packed_u, uintptr_t norm_u_q_w, uintptr_t norm_u_k_w,
           uintptr_t q_cat_out, uintptr_t k_cat_out, uintptr_t v_cat_out,
           int B, int total_L, int L_v, int L_a, int L_u,
           int N, int D_h, int seq_len,
           float eps_v, float eps_a, float eps_u, uintptr_t stream) {
            flash_rt::quantize::qkv_split_joint3_cat_bf16(
                to_ptr(packed_v), to_ptr(qkv_v_bias),
                to_ptr(norm_v_q_w), to_ptr(norm_v_k_w),
                reinterpret_cast<const float*>(freqs_re),
                reinterpret_cast<const float*>(freqs_im),
                to_ptr(packed_a), to_ptr(norm_a_q_w), to_ptr(norm_a_k_w),
                to_ptr(packed_u), to_ptr(norm_u_q_w), to_ptr(norm_u_k_w),
                to_ptr(q_cat_out), to_ptr(k_cat_out), to_ptr(v_cat_out),
                B, total_L, L_v, L_a, L_u, N, D_h, seq_len,
                eps_v, eps_a, eps_u, to_stream(stream));
        },
        py::arg("packed_v"), py::arg("qkv_v_bias"),
        py::arg("norm_v_q_w"), py::arg("norm_v_k_w"),
        py::arg("freqs_re"), py::arg("freqs_im"),
        py::arg("packed_a"), py::arg("norm_a_q_w"), py::arg("norm_a_k_w"),
        py::arg("packed_u"), py::arg("norm_u_q_w"), py::arg("norm_u_k_w"),
        py::arg("q_cat_out"), py::arg("k_cat_out"), py::arg("v_cat_out"),
        py::arg("B"), py::arg("total_L"), py::arg("L_v"),
        py::arg("L_a"), py::arg("L_u"), py::arg("N"), py::arg("D_h"),
        py::arg("seq_len"), py::arg("eps_v"), py::arg("eps_a"),
        py::arg("eps_u"), py::arg("stream") = 0);

    m.def("concat3_qkv_bf16_fast",
        [](uintptr_t q0, uintptr_t q1, uintptr_t q2,
           uintptr_t k0, uintptr_t k1, uintptr_t k2,
           uintptr_t v0, uintptr_t v1, uintptr_t v2,
           uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
           int B, int L0, int L1, int L2, int H, int D,
           long long q0s0, long long q0s1,
           long long q1s0, long long q1s1,
           long long q2s0, long long q2s1,
           long long k0s0, long long k0s1,
           long long k1s0, long long k1s1,
           long long k2s0, long long k2s1,
           long long v0s0, long long v0s1,
           long long v1s0, long long v1s1,
           long long v2s0, long long v2s1,
           uintptr_t stream) {
            concat3_qkv_bf16_fast(
                typed_ptr<__nv_bfloat16>(q0), typed_ptr<__nv_bfloat16>(q1),
                typed_ptr<__nv_bfloat16>(q2), typed_ptr<__nv_bfloat16>(k0),
                typed_ptr<__nv_bfloat16>(k1), typed_ptr<__nv_bfloat16>(k2),
                typed_ptr<__nv_bfloat16>(v0), typed_ptr<__nv_bfloat16>(v1),
                typed_ptr<__nv_bfloat16>(v2), typed_ptr<__nv_bfloat16>(q_out),
                typed_ptr<__nv_bfloat16>(k_out), typed_ptr<__nv_bfloat16>(v_out),
                B, L0, L1, L2, H, D,
                q0s0, q0s1, q1s0, q1s1, q2s0, q2s1,
                k0s0, k0s1, k1s0, k1s1, k2s0, k2s1,
                v0s0, v0s1, v1s0, v1s1, v2s0, v2s1,
                to_stream(stream));
        },
        py::arg("q0"), py::arg("q1"), py::arg("q2"),
        py::arg("k0"), py::arg("k1"), py::arg("k2"),
        py::arg("v0"), py::arg("v1"), py::arg("v2"),
        py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
        py::arg("B"), py::arg("L0"), py::arg("L1"), py::arg("L2"),
        py::arg("H"), py::arg("D"),
        py::arg("q0s0"), py::arg("q0s1"), py::arg("q1s0"), py::arg("q1s1"),
        py::arg("q2s0"), py::arg("q2s1"), py::arg("k0s0"), py::arg("k0s1"),
        py::arg("k1s0"), py::arg("k1s1"), py::arg("k2s0"), py::arg("k2s1"),
        py::arg("v0s0"), py::arg("v0s1"), py::arg("v1s0"), py::arg("v1s1"),
        py::arg("v2s0"), py::arg("v2s1"), py::arg("stream") = 0);

    m.def("concat3_qkv_bf16",
        [](uintptr_t q0, uintptr_t q1, uintptr_t q2,
           uintptr_t k0, uintptr_t k1, uintptr_t k2,
           uintptr_t v0, uintptr_t v1, uintptr_t v2,
           uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
           int B, int L0, int L1, int L2, int H, int D,
           long long q0s0, long long q0s1, long long q0s2,
           long long q1s0, long long q1s1, long long q1s2,
           long long q2s0, long long q2s1, long long q2s2,
           long long k0s0, long long k0s1, long long k0s2,
           long long k1s0, long long k1s1, long long k1s2,
           long long k2s0, long long k2s1, long long k2s2,
           long long v0s0, long long v0s1, long long v0s2,
           long long v1s0, long long v1s1, long long v1s2,
           long long v2s0, long long v2s1, long long v2s2,
           uintptr_t stream) {
            concat3_qkv_bf16(
                typed_ptr<__nv_bfloat16>(q0), typed_ptr<__nv_bfloat16>(q1),
                typed_ptr<__nv_bfloat16>(q2), typed_ptr<__nv_bfloat16>(k0),
                typed_ptr<__nv_bfloat16>(k1), typed_ptr<__nv_bfloat16>(k2),
                typed_ptr<__nv_bfloat16>(v0), typed_ptr<__nv_bfloat16>(v1),
                typed_ptr<__nv_bfloat16>(v2), typed_ptr<__nv_bfloat16>(q_out),
                typed_ptr<__nv_bfloat16>(k_out), typed_ptr<__nv_bfloat16>(v_out),
                B, L0, L1, L2, H, D,
                q0s0, q0s1, q0s2, q1s0, q1s1, q1s2,
                q2s0, q2s1, q2s2, k0s0, k0s1, k0s2,
                k1s0, k1s1, k1s2, k2s0, k2s1, k2s2,
                v0s0, v0s1, v0s2, v1s0, v1s1, v1s2,
                v2s0, v2s1, v2s2, to_stream(stream));
        });

    m.def("quant_per_warp_int8_bf16_d128",
        [](uintptr_t x, uintptr_t out, uintptr_t scale,
           int B, int L, int H, uintptr_t stream) {
            quant_per_warp_int8_bf16_d128(
                typed_ptr<__nv_bfloat16>(x), reinterpret_cast<int8_t*>(out),
                reinterpret_cast<float*>(scale), B, L, H, to_stream(stream));
        },
        py::arg("x"), py::arg("out"), py::arg("scale"),
        py::arg("B"), py::arg("L"), py::arg("H"), py::arg("stream") = 0);

#ifdef ENABLE_MOTUS_SAGE2_RAW
    m.def("sage2_qk_int8_sv_f8_bf16_nhd_d128",
        [](uintptr_t q_int8, uintptr_t k_int8, uintptr_t v_fp8,
           uintptr_t out_bf16, uintptr_t q_scale, uintptr_t k_scale,
           uintptr_t v_scale, int B, int Lq, int Lk, int H,
           float softmax_scale, uintptr_t stream) {
            return flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128(
                to_ptr(q_int8), to_ptr(k_int8), to_ptr(v_fp8),
                to_ptr(out_bf16), to_ptr(q_scale), to_ptr(k_scale),
                to_ptr(v_scale), B, Lq, Lk, H, softmax_scale,
                to_stream(stream));
        },
        py::arg("q_int8"), py::arg("k_int8"), py::arg("v_fp8"),
        py::arg("out_bf16"), py::arg("q_scale"), py::arg("k_scale"),
        py::arg("v_scale"), py::arg("B"), py::arg("Lq"),
        py::arg("Lk"), py::arg("H"), py::arg("softmax_scale"),
        py::arg("stream") = 0);

    m.def("sage2_qk_int8_sv_f16_bf16_nhd_d128",
        [](uintptr_t q_int8, uintptr_t k_int8, uintptr_t v_half,
           uintptr_t out_bf16, uintptr_t q_scale, uintptr_t k_scale,
           int B, int Lq, int Lk, int H, float softmax_scale,
           uintptr_t stream) {
            return flash_rt::attention::sage2::qk_int8_sv_f16_bf16_nhd_d128(
                to_ptr(q_int8), to_ptr(k_int8), to_ptr(v_half),
                to_ptr(out_bf16), to_ptr(q_scale), to_ptr(k_scale),
                B, Lq, Lk, H, softmax_scale, to_stream(stream));
        },
        py::arg("q_int8"), py::arg("k_int8"), py::arg("v_half"),
        py::arg("out_bf16"), py::arg("q_scale"), py::arg("k_scale"),
        py::arg("B"), py::arg("Lq"), py::arg("Lk"), py::arg("H"),
        py::arg("softmax_scale"), py::arg("stream") = 0);

    m.def("sage2_qk_int8_sv_f16_bf16_gqa_nhd_d256",
        [](uintptr_t q_int8, uintptr_t k_int8, uintptr_t v_half,
           uintptr_t out_bf16, uintptr_t q_scale, uintptr_t k_scale,
           int B, int Lq, int Lk, int Hq, int Hkv, float softmax_scale,
           uintptr_t stream) {
            return flash_rt::attention::sage2::qk_int8_sv_f16_bf16_gqa_nhd_d256(
                to_ptr(q_int8), to_ptr(k_int8), to_ptr(v_half),
                to_ptr(out_bf16), to_ptr(q_scale), to_ptr(k_scale),
                B, Lq, Lk, Hq, Hkv, softmax_scale, to_stream(stream));
        },
        py::arg("q_int8"), py::arg("k_int8"), py::arg("v_half"),
        py::arg("out_bf16"), py::arg("q_scale"), py::arg("k_scale"),
        py::arg("B"), py::arg("Lq"), py::arg("Lk"),
        py::arg("Hq"), py::arg("Hkv"), py::arg("softmax_scale"),
        py::arg("stream") = 0);

    m.def("sage2_qk_int8_sv_f8_bf16_gqa_nhd_d256",
        [](uintptr_t q_int8, uintptr_t k_int8, uintptr_t v_fp8,
           uintptr_t out_bf16, uintptr_t q_scale, uintptr_t k_scale,
           uintptr_t v_scale, int B, int Lq, int Lk, int Hq, int Hkv,
           float softmax_scale, uintptr_t stream) {
            return flash_rt::attention::sage2::qk_int8_sv_f8_bf16_gqa_nhd_d256(
                to_ptr(q_int8), to_ptr(k_int8), to_ptr(v_fp8),
                to_ptr(out_bf16), to_ptr(q_scale), to_ptr(k_scale),
                to_ptr(v_scale), B, Lq, Lk, Hq, Hkv, softmax_scale,
                to_stream(stream));
        },
        py::arg("q_int8"), py::arg("k_int8"), py::arg("v_fp8"),
        py::arg("out_bf16"), py::arg("q_scale"), py::arg("k_scale"),
        py::arg("v_scale"), py::arg("B"), py::arg("Lq"),
        py::arg("Lk"), py::arg("Hq"), py::arg("Hkv"),
        py::arg("softmax_scale"), py::arg("stream") = 0);
#endif

    m.def("quant_per_block_int8_bf16_d128",
        [](uintptr_t x, uintptr_t out, uintptr_t scale,
           int B, int L, int H, uintptr_t stream) {
            quant_per_block_int8_bf16_d128(
                typed_ptr<__nv_bfloat16>(x), reinterpret_cast<int8_t*>(out),
                reinterpret_cast<float*>(scale), B, L, H, to_stream(stream));
        },
        py::arg("x"), py::arg("out"), py::arg("scale"),
        py::arg("B"), py::arg("L"), py::arg("H"), py::arg("stream") = 0);

    m.def("concat3_v_transpose_pad_permute_bf16_d128",
        [](uintptr_t v0, uintptr_t v1, uintptr_t v2, uintptr_t v_tpp_out,
           int B, int L0, int L1, int L2, int H,
           long long v0s0, long long v0s1,
           long long v1s0, long long v1s1,
           long long v2s0, long long v2s1,
           uintptr_t stream) {
            concat3_v_transpose_pad_permute_bf16_d128(
                typed_ptr<__nv_bfloat16>(v0), typed_ptr<__nv_bfloat16>(v1),
                typed_ptr<__nv_bfloat16>(v2), typed_ptr<__nv_bfloat16>(v_tpp_out),
                B, L0, L1, L2, H,
                v0s0, v0s1, v1s0, v1s1, v2s0, v2s1,
                to_stream(stream));
        },
        py::arg("v0"), py::arg("v1"), py::arg("v2"), py::arg("v_tpp_out"),
        py::arg("B"), py::arg("L0"), py::arg("L1"), py::arg("L2"),
        py::arg("H"), py::arg("v0s0"), py::arg("v0s1"),
        py::arg("v1s0"), py::arg("v1s1"),
        py::arg("v2s0"), py::arg("v2s1"), py::arg("stream") = 0);

    m.def("v_tpp_bf16_quant_fp8_d128",
        [](uintptr_t v_tpp, uintptr_t v_fp8, uintptr_t v_scale,
           int B, int L, int H, uintptr_t stream) {
            v_tpp_bf16_quant_fp8_d128(
                typed_ptr<__nv_bfloat16>(v_tpp),
                reinterpret_cast<int8_t*>(v_fp8),
                reinterpret_cast<float*>(v_scale),
                B, L, H, to_stream(stream));
        },
        py::arg("v_tpp"), py::arg("v_fp8"), py::arg("v_scale"),
        py::arg("B"), py::arg("L"), py::arg("H"), py::arg("stream") = 0);

    m.def("concat3_qk_int8_v_fp16_d128",
        [](uintptr_t q0, uintptr_t q1, uintptr_t q2,
           uintptr_t k0, uintptr_t k1, uintptr_t k2,
           uintptr_t v0, uintptr_t v1, uintptr_t v2,
           uintptr_t q_out, uintptr_t k_out, uintptr_t v_fp16_out,
           uintptr_t q_scale, uintptr_t k_scale,
           int B, int L0, int L1, int L2, int H,
           long long q0s0, long long q0s1,
           long long q1s0, long long q1s1,
           long long q2s0, long long q2s1,
           long long k0s0, long long k0s1,
           long long k1s0, long long k1s1,
           long long k2s0, long long k2s1,
           long long v0s0, long long v0s1,
           long long v1s0, long long v1s1,
           long long v2s0, long long v2s1,
           uintptr_t stream) {
            concat3_qk_int8_v_fp16_d128(
                typed_ptr<__nv_bfloat16>(q0), typed_ptr<__nv_bfloat16>(q1),
                typed_ptr<__nv_bfloat16>(q2), typed_ptr<__nv_bfloat16>(k0),
                typed_ptr<__nv_bfloat16>(k1), typed_ptr<__nv_bfloat16>(k2),
                typed_ptr<__nv_bfloat16>(v0), typed_ptr<__nv_bfloat16>(v1),
                typed_ptr<__nv_bfloat16>(v2),
                reinterpret_cast<int8_t*>(q_out), reinterpret_cast<int8_t*>(k_out),
                reinterpret_cast<__half*>(v_fp16_out),
                reinterpret_cast<float*>(q_scale),
                reinterpret_cast<float*>(k_scale),
                B, L0, L1, L2, H,
                q0s0, q0s1, q1s0, q1s1, q2s0, q2s1,
                k0s0, k0s1, k1s0, k1s1, k2s0, k2s1,
                v0s0, v0s1, v1s0, v1s1, v2s0, v2s1,
                to_stream(stream));
        });

    m.def("concat3_qk_int8_v_fp8_d128",
        [](uintptr_t q0, uintptr_t q1, uintptr_t q2,
           uintptr_t k0, uintptr_t k1, uintptr_t k2,
           uintptr_t v0, uintptr_t v1, uintptr_t v2,
           uintptr_t q_out, uintptr_t k_out, uintptr_t v_fp8_out,
           uintptr_t q_scale, uintptr_t k_scale, uintptr_t v_scale,
           int B, int L0, int L1, int L2, int H,
           long long q0s0, long long q0s1,
           long long q1s0, long long q1s1,
           long long q2s0, long long q2s1,
           long long k0s0, long long k0s1,
           long long k1s0, long long k1s1,
           long long k2s0, long long k2s1,
           long long v0s0, long long v0s1,
           long long v1s0, long long v1s1,
           long long v2s0, long long v2s1,
           uintptr_t stream) {
            concat3_qk_int8_v_fp8_d128(
                typed_ptr<__nv_bfloat16>(q0), typed_ptr<__nv_bfloat16>(q1),
                typed_ptr<__nv_bfloat16>(q2), typed_ptr<__nv_bfloat16>(k0),
                typed_ptr<__nv_bfloat16>(k1), typed_ptr<__nv_bfloat16>(k2),
                typed_ptr<__nv_bfloat16>(v0), typed_ptr<__nv_bfloat16>(v1),
                typed_ptr<__nv_bfloat16>(v2),
                reinterpret_cast<int8_t*>(q_out),
                reinterpret_cast<int8_t*>(k_out),
                reinterpret_cast<int8_t*>(v_fp8_out),
                reinterpret_cast<float*>(q_scale),
                reinterpret_cast<float*>(k_scale),
                reinterpret_cast<float*>(v_scale),
                B, L0, L1, L2, H,
                q0s0, q0s1, q1s0, q1s1, q2s0, q2s1,
                k0s0, k0s1, k1s0, k1s1, k2s0, k2s1,
                v0s0, v0s1, v1s0, v1s1, v2s0, v2s1,
                to_stream(stream));
        },
        py::arg("q0"), py::arg("q1"), py::arg("q2"),
        py::arg("k0"), py::arg("k1"), py::arg("k2"),
        py::arg("v0"), py::arg("v1"), py::arg("v2"),
        py::arg("q_out"), py::arg("k_out"), py::arg("v_fp8_out"),
        py::arg("q_scale"), py::arg("k_scale"), py::arg("v_scale"),
        py::arg("B"), py::arg("L0"), py::arg("L1"), py::arg("L2"),
        py::arg("H"),
        py::arg("q0s0"), py::arg("q0s1"), py::arg("q1s0"), py::arg("q1s1"),
        py::arg("q2s0"), py::arg("q2s1"), py::arg("k0s0"), py::arg("k0s1"),
        py::arg("k1s0"), py::arg("k1s1"), py::arg("k2s0"), py::arg("k2s1"),
        py::arg("v0s0"), py::arg("v0s1"), py::arg("v1s0"), py::arg("v1s1"),
        py::arg("v2s0"), py::arg("v2s1"), py::arg("stream") = 0);

    m.def("motus_joint_residual3_out_bf16",
        [](uintptr_t v_residual, uintptr_t v_x, uintptr_t v_bias,
           uintptr_t v_gate, uintptr_t v_out, int v_n, int v_dim,
           uintptr_t a_residual, uintptr_t a_x, uintptr_t a_bias,
           uintptr_t a_gate, uintptr_t a_out, int a_n, int a_dim,
           uintptr_t u_residual, uintptr_t u_x, uintptr_t u_out,
           int u_n, int u_dim, uintptr_t stream) {
            motus_joint_residual3_out_bf16(
                typed_ptr<__nv_bfloat16>(v_residual),
                typed_ptr<__nv_bfloat16>(v_x),
                typed_ptr<__nv_bfloat16>(v_bias),
                typed_ptr<__nv_bfloat16>(v_gate),
                typed_ptr<__nv_bfloat16>(v_out), v_n, v_dim,
                typed_ptr<__nv_bfloat16>(a_residual),
                typed_ptr<__nv_bfloat16>(a_x),
                typed_ptr<__nv_bfloat16>(a_bias),
                typed_ptr<__nv_bfloat16>(a_gate),
                typed_ptr<__nv_bfloat16>(a_out), a_n, a_dim,
                typed_ptr<__nv_bfloat16>(u_residual),
                typed_ptr<__nv_bfloat16>(u_x),
                typed_ptr<__nv_bfloat16>(u_out), u_n, u_dim,
                to_stream(stream));
        });

    m.def("motus_joint_residual3_out_bf16_vgate_fp8",
        [](uintptr_t v_residual, uintptr_t v_x, uintptr_t v_bias,
           uintptr_t v_gate, uintptr_t v_gate_scale, uintptr_t v_out,
           int v_n, int v_dim,
           uintptr_t a_residual, uintptr_t a_x, uintptr_t a_bias,
           uintptr_t a_gate, uintptr_t a_out, int a_n, int a_dim,
           uintptr_t u_residual, uintptr_t u_x, uintptr_t u_out,
           int u_n, int u_dim, uintptr_t stream) {
            motus_joint_residual3_out_bf16_vgate_fp8(
                typed_ptr<__nv_bfloat16>(v_residual),
                typed_ptr<__nv_bfloat16>(v_x),
                typed_ptr<__nv_bfloat16>(v_bias),
                typed_ptr<__nv_fp8_e4m3>(v_gate),
                reinterpret_cast<const float*>(v_gate_scale),
                typed_ptr<__nv_bfloat16>(v_out), v_n, v_dim,
                typed_ptr<__nv_bfloat16>(a_residual),
                typed_ptr<__nv_bfloat16>(a_x),
                typed_ptr<__nv_bfloat16>(a_bias),
                typed_ptr<__nv_bfloat16>(a_gate),
                typed_ptr<__nv_bfloat16>(a_out), a_n, a_dim,
                typed_ptr<__nv_bfloat16>(u_residual),
                typed_ptr<__nv_bfloat16>(u_x),
                typed_ptr<__nv_bfloat16>(u_out), u_n, u_dim,
                to_stream(stream));
        });

    m.def("motus_joint_residual3_out_bf16_action_nobias",
        [](uintptr_t v_residual, uintptr_t v_x, uintptr_t v_bias,
           uintptr_t v_gate, uintptr_t v_out, int v_n, int v_dim,
           uintptr_t a_residual, uintptr_t a_x, uintptr_t a_gate,
           uintptr_t a_out, int a_n, int a_dim,
           uintptr_t u_residual, uintptr_t u_x, uintptr_t u_out,
           int u_n, int u_dim, uintptr_t stream) {
            motus_joint_residual3_out_bf16_action_nobias(
                typed_ptr<__nv_bfloat16>(v_residual),
                typed_ptr<__nv_bfloat16>(v_x),
                typed_ptr<__nv_bfloat16>(v_bias),
                typed_ptr<__nv_bfloat16>(v_gate),
                typed_ptr<__nv_bfloat16>(v_out), v_n, v_dim,
                typed_ptr<__nv_bfloat16>(a_residual),
                typed_ptr<__nv_bfloat16>(a_x),
                typed_ptr<__nv_bfloat16>(a_gate),
                typed_ptr<__nv_bfloat16>(a_out), a_n, a_dim,
                typed_ptr<__nv_bfloat16>(u_residual),
                typed_ptr<__nv_bfloat16>(u_x),
                typed_ptr<__nv_bfloat16>(u_out), u_n, u_dim,
                to_stream(stream));
        });

    m.def("motus_joint_residual3_out_bf16_g1d_action_nobias",
        [](uintptr_t v_residual, uintptr_t v_x, uintptr_t v_bias,
           uintptr_t v_gate_1d, uintptr_t v_out, int v_n, int v_dim,
           uintptr_t a_residual, uintptr_t a_x, uintptr_t a_gate_1d,
           uintptr_t a_out, int a_n, int a_dim,
           uintptr_t u_residual, uintptr_t u_x, uintptr_t u_out,
           int u_n, int u_dim, uintptr_t stream) {
            motus_joint_residual3_out_bf16_g1d_action_nobias(
                typed_ptr<__nv_bfloat16>(v_residual),
                typed_ptr<__nv_bfloat16>(v_x),
                typed_ptr<__nv_bfloat16>(v_bias),
                typed_ptr<__nv_bfloat16>(v_gate_1d),
                typed_ptr<__nv_bfloat16>(v_out), v_n, v_dim,
                typed_ptr<__nv_bfloat16>(a_residual),
                typed_ptr<__nv_bfloat16>(a_x),
                typed_ptr<__nv_bfloat16>(a_gate_1d),
                typed_ptr<__nv_bfloat16>(a_out), a_n, a_dim,
                typed_ptr<__nv_bfloat16>(u_residual),
                typed_ptr<__nv_bfloat16>(u_x),
                typed_ptr<__nv_bfloat16>(u_out), u_n, u_dim,
                to_stream(stream));
        });

    m.def("motus_joint_residual3_out_bf16_vgate_fp8_action_nobias",
        [](uintptr_t v_residual, uintptr_t v_x, uintptr_t v_bias,
           uintptr_t v_gate, uintptr_t v_gate_scale, uintptr_t v_out,
           int v_n, int v_dim,
           uintptr_t a_residual, uintptr_t a_x, uintptr_t a_gate,
           uintptr_t a_out, int a_n, int a_dim,
           uintptr_t u_residual, uintptr_t u_x, uintptr_t u_out,
           int u_n, int u_dim, uintptr_t stream) {
            motus_joint_residual3_out_bf16_vgate_fp8_action_nobias(
                typed_ptr<__nv_bfloat16>(v_residual),
                typed_ptr<__nv_bfloat16>(v_x),
                typed_ptr<__nv_bfloat16>(v_bias),
                typed_ptr<__nv_fp8_e4m3>(v_gate),
                reinterpret_cast<const float*>(v_gate_scale),
                typed_ptr<__nv_bfloat16>(v_out), v_n, v_dim,
                typed_ptr<__nv_bfloat16>(a_residual),
                typed_ptr<__nv_bfloat16>(a_x),
                typed_ptr<__nv_bfloat16>(a_gate),
                typed_ptr<__nv_bfloat16>(a_out), a_n, a_dim,
                typed_ptr<__nv_bfloat16>(u_residual),
                typed_ptr<__nv_bfloat16>(u_x),
                typed_ptr<__nv_bfloat16>(u_out), u_n, u_dim,
                to_stream(stream));
        });

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

    m.def("causal_conv1d_qwen36_update_chunk_bf16",
        [](uintptr_t x, uintptr_t w, uintptr_t bias,
           uintptr_t out, uintptr_t state,
           int B, int S, int conv_dim, int k, bool apply_silu,
           uintptr_t stream) {
            flash_rt::kernels::causal_conv1d_qwen36_update_chunk_bf16(
                to_ptr(x), to_ptr(w),
                bias ? to_ptr(bias) : nullptr,
                to_ptr(out), to_ptr(state),
                B, S, conv_dim, k, apply_silu, to_stream(stream));
        },
        py::arg("x"), py::arg("w"), py::arg("bias"),
        py::arg("out"), py::arg("state"),
        py::arg("B"), py::arg("S"), py::arg("conv_dim"), py::arg("k"),
        py::arg("apply_silu") = true, py::arg("stream") = 0);

    m.def("causal_conv1d_qwen36_update_chunk_parallel_bf16",
        [](uintptr_t x, uintptr_t w, uintptr_t bias,
           uintptr_t out, uintptr_t state,
           int B, int S, int conv_dim, int k, bool apply_silu,
           uintptr_t stream) {
            flash_rt::kernels::causal_conv1d_qwen36_update_chunk_parallel_bf16(
                to_ptr(x), to_ptr(w),
                bias ? to_ptr(bias) : nullptr,
                to_ptr(out), to_ptr(state),
                B, S, conv_dim, k, apply_silu, to_stream(stream));
        },
        py::arg("x"), py::arg("w"), py::arg("bias"),
        py::arg("out"), py::arg("state"),
        py::arg("B"), py::arg("S"), py::arg("conv_dim"), py::arg("k"),
        py::arg("apply_silu") = true, py::arg("stream") = 0);

    m.def("causal_conv1d_qwen36_update_chunk_parallel_gqa_bf16",
        [](uintptr_t x, uintptr_t w, uintptr_t bias,
           uintptr_t q16, uintptr_t k16, uintptr_t v48, uintptr_t state,
           int B, int S, int conv_dim, int k, bool apply_silu,
           uintptr_t stream) {
            flash_rt::kernels::causal_conv1d_qwen36_update_chunk_parallel_gqa_bf16(
                to_ptr(x), to_ptr(w),
                bias ? to_ptr(bias) : nullptr,
                to_ptr(q16), to_ptr(k16), to_ptr(v48), to_ptr(state),
                B, S, conv_dim, k, apply_silu, to_stream(stream));
        },
        py::arg("x"), py::arg("w"), py::arg("bias"),
        py::arg("q16"), py::arg("k16"), py::arg("v48"), py::arg("state"),
        py::arg("B"), py::arg("S"), py::arg("conv_dim"), py::arg("k"),
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

    // Thor-only MTP prompt-tail fc matmul. Returns 0 on success, a
    // non-zero value if the device cannot support the kernel's
    // shared-memory configuration; the caller must fall back to
    // bf16_matmul_qwen36_bf16 in that case.
    m.def("bf16_matmul_qwen36_thor_mtp_fc_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t out,
           int M, int N, uintptr_t stream) -> int {
            return flash_rt::kernels::bf16_matmul_qwen36_thor_mtp_fc_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W),
                reinterpret_cast<__nv_bfloat16*>(out),
                M, N, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("out"),
        py::arg("M"), py::arg("N"), py::arg("stream") = 0);

    m.def("bf16_matmul_qwen36_ab96_bf16",
        [](uintptr_t x, uintptr_t W_ab, uintptr_t out_ab,
           int M, uintptr_t stream) {
            flash_rt::kernels::bf16_matmul_qwen36_ab96_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W_ab),
                reinterpret_cast<__nv_bfloat16*>(out_ab),
                M, to_stream(stream));
        },
        py::arg("x"), py::arg("W_ab"), py::arg("out_ab"),
        py::arg("M"), py::arg("stream") = 0);

    m.def("bf16_matmul_qwen36_ab96_m4_bf16",
        [](uintptr_t x, uintptr_t W_ab, uintptr_t out_ab,
           int M, uintptr_t stream) {
            flash_rt::kernels::bf16_matmul_qwen36_ab96_m4_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W_ab),
                reinterpret_cast<__nv_bfloat16*>(out_ab),
                M, to_stream(stream));
        },
        py::arg("x"), py::arg("W_ab"), py::arg("out_ab"),
        py::arg("M"), py::arg("stream") = 0);

    m.def("bf16_matmul_qwen36_ab96_m4_pair_bf16",
        [](uintptr_t x, uintptr_t W_ab, uintptr_t out_ab,
           int M, uintptr_t stream) {
            flash_rt::kernels::bf16_matmul_qwen36_ab96_m4_pair_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W_ab),
                reinterpret_cast<__nv_bfloat16*>(out_ab),
                M, to_stream(stream));
        },
        py::arg("x"), py::arg("W_ab"), py::arg("out_ab"),
        py::arg("M"), py::arg("stream") = 0);

    m.def("bf16_matmul_qwen36_ab96_lt_bf16",
        [](uintptr_t x, uintptr_t W_ab, uintptr_t out_ab,
           int M, uintptr_t stream) {
            flash_rt::kernels::bf16_matmul_qwen36_ab96_lt_bf16(
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(W_ab),
                reinterpret_cast<__nv_bfloat16*>(out_ab),
                M, to_stream(stream));
        },
        py::arg("x"), py::arg("W_ab"), py::arg("out_ab"),
        py::arg("M"), py::arg("stream") = 0);

    m.def("qwen36_embedding_lookup_bf16",
        [](uintptr_t token_ids, uintptr_t embed, uintptr_t out,
           int rows, int hidden, uintptr_t stream) {
            flash_rt::kernels::qwen36_embedding_lookup_bf16(
                reinterpret_cast<const int64_t*>(token_ids),
                reinterpret_cast<const __nv_bfloat16*>(embed),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows, hidden, to_stream(stream));
        },
        py::arg("token_ids"), py::arg("embed"), py::arg("out"),
        py::arg("rows"), py::arg("hidden"), py::arg("stream") = 0);

    m.def("qwen36_partial_rope_qk_bf16",
        [](uintptr_t q_in, uintptr_t k_in, uintptr_t cos, uintptr_t sin,
           uintptr_t q_out, uintptr_t k_out, int rows, int q_heads,
           int k_heads, int head_dim, int rope_dim, uintptr_t stream) {
            flash_rt::kernels::qwen36_partial_rope_qk_bf16(
                reinterpret_cast<const __nv_bfloat16*>(q_in),
                reinterpret_cast<const __nv_bfloat16*>(k_in),
                reinterpret_cast<const __nv_bfloat16*>(cos),
                reinterpret_cast<const __nv_bfloat16*>(sin),
                reinterpret_cast<__nv_bfloat16*>(q_out),
                reinterpret_cast<__nv_bfloat16*>(k_out),
                rows, q_heads, k_heads, head_dim, rope_dim,
                to_stream(stream));
        },
        py::arg("q_in"), py::arg("k_in"), py::arg("cos"), py::arg("sin"),
        py::arg("q_out"), py::arg("k_out"), py::arg("rows"),
        py::arg("q_heads"), py::arg("k_heads"), py::arg("head_dim"),
        py::arg("rope_dim"), py::arg("stream") = 0);

    m.def("qwen36_tq_prepare_scalars",
        [](uintptr_t k_norm, uintptr_t k_rnorm, uintptr_t v_norm,
           uintptr_t norm_k, uintptr_t coef_rnorm, uintptr_t norm_v,
           int n, float coef, uintptr_t stream) {
            flash_rt::kernels::qwen36_tq_prepare_scalars(
                reinterpret_cast<const __half*>(k_norm),
                reinterpret_cast<const __half*>(k_rnorm),
                reinterpret_cast<const __half*>(v_norm),
                reinterpret_cast<float*>(norm_k),
                reinterpret_cast<float*>(coef_rnorm),
                reinterpret_cast<float*>(norm_v),
                n, coef, to_stream(stream));
        },
        py::arg("k_norm"), py::arg("k_rnorm"), py::arg("v_norm"),
        py::arg("norm_k"), py::arg("coef_rnorm"), py::arg("norm_v"),
        py::arg("n"), py::arg("coef"), py::arg("stream") = 0);

    m.def("qwen36_argmax_bf16",
        [](uintptr_t logits, uintptr_t argmax_out,
           int rows, int vocab, uintptr_t stream) {
            flash_rt::kernels::qwen36_argmax_bf16(
                reinterpret_cast<const __nv_bfloat16*>(logits),
                reinterpret_cast<int64_t*>(argmax_out),
                rows, vocab, to_stream(stream));
        },
        py::arg("logits"), py::arg("argmax_out"),
        py::arg("rows"), py::arg("vocab"), py::arg("stream") = 0);

    m.def("qwen36_spec_accept_greedy_bf16",
        [](uintptr_t logits, uintptr_t drafts, uintptr_t argmax_out,
           uintptr_t accept_n, int rows, int vocab, int spec_k,
           uintptr_t stream) {
            flash_rt::kernels::qwen36_spec_accept_greedy_bf16(
                reinterpret_cast<const __nv_bfloat16*>(logits),
                reinterpret_cast<const int64_t*>(drafts),
                reinterpret_cast<int64_t*>(argmax_out),
                reinterpret_cast<int*>(accept_n),
                rows, vocab, spec_k, to_stream(stream));
        },
        py::arg("logits"), py::arg("drafts"), py::arg("argmax_out"),
        py::arg("accept_n"), py::arg("rows"), py::arg("vocab"),
        py::arg("spec_k"), py::arg("stream") = 0);

    m.def("qwen36_spec_accept_partitioned_bf16",
        [](uintptr_t logits, uintptr_t drafts, uintptr_t argmax_out,
           uintptr_t accept_n, uintptr_t partial_vals,
           uintptr_t partial_idx, int rows, int vocab, int spec_k,
           int parts, uintptr_t stream) {
            flash_rt::kernels::qwen36_spec_accept_partitioned_bf16(
                reinterpret_cast<const __nv_bfloat16*>(logits),
                reinterpret_cast<const int64_t*>(drafts),
                reinterpret_cast<int64_t*>(argmax_out),
                reinterpret_cast<int*>(accept_n),
                reinterpret_cast<float*>(partial_vals),
                reinterpret_cast<int*>(partial_idx),
                rows, vocab, spec_k, parts, to_stream(stream));
        },
        py::arg("logits"), py::arg("drafts"), py::arg("argmax_out"),
        py::arg("accept_n"), py::arg("partial_vals"),
        py::arg("partial_idx"), py::arg("rows"), py::arg("vocab"),
        py::arg("spec_k"), py::arg("parts"), py::arg("stream") = 0);

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

    m.def("sigmoid_mul_qwen36_bf16",
        [](uintptr_t gate, uintptr_t x, uintptr_t out,
           int n, uintptr_t stream) {
            flash_rt::kernels::sigmoid_mul_qwen36_bf16(
                reinterpret_cast<const __nv_bfloat16*>(gate),
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<__nv_bfloat16*>(out),
                n, to_stream(stream));
        },
        py::arg("gate"), py::arg("x"), py::arg("out"),
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

    // FP32-state variant (Thor K-row no-LSB-jitter path).
    m.def("gated_deltanet_recurrent_qwen36_f32state_bf16io",
        [](uintptr_t q, uintptr_t k, uintptr_t v,
           uintptr_t g, uintptr_t beta,
           uintptr_t state_f32, uintptr_t out,
           int B, int num_v_heads, int head_k_dim, int head_v_dim,
           bool use_qk_l2norm, uintptr_t stream) {
            flash_rt::kernels::gated_deltanet_recurrent_qwen36_f32state_bf16io(
                to_ptr(q), to_ptr(k), to_ptr(v),
                to_ptr(g), to_ptr(beta),
                to_ptr(state_f32), to_ptr(out),
                B, num_v_heads, head_k_dim, head_v_dim,
                use_qk_l2norm, to_stream(stream));
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("g"), py::arg("beta"),
        py::arg("state_f32"), py::arg("out"),
        py::arg("B"), py::arg("num_v_heads"),
        py::arg("head_k_dim"), py::arg("head_v_dim"),
        py::arg("use_qk_l2norm") = true, py::arg("stream") = 0);

    m.def("gated_deltanet_chunk_qwen36_bf16",
        [](uintptr_t q, uintptr_t k, uintptr_t v,
           uintptr_t g, uintptr_t beta,
           uintptr_t state, uintptr_t out,
           int S, int num_v_heads, int head_k_dim, int head_v_dim,
           bool use_qk_l2norm, uintptr_t stream) {
            flash_rt::kernels::gated_deltanet_chunk_qwen36_bf16(
                to_ptr(q), to_ptr(k), to_ptr(v),
                to_ptr(g), to_ptr(beta),
                to_ptr(state), to_ptr(out),
                S, num_v_heads, head_k_dim, head_v_dim,
                use_qk_l2norm, to_stream(stream));
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("g"), py::arg("beta"),
        py::arg("state"), py::arg("out"),
        py::arg("S"), py::arg("num_v_heads"),
        py::arg("head_k_dim"), py::arg("head_v_dim"),
        py::arg("use_qk_l2norm") = true, py::arg("stream") = 0);

    m.def("gated_deltanet_chunk_smem_qwen36_bf16",
        [](uintptr_t q, uintptr_t k, uintptr_t v,
           uintptr_t g, uintptr_t beta,
           uintptr_t state, uintptr_t out,
           int S, int num_v_heads, int head_k_dim, int head_v_dim,
           bool use_qk_l2norm, uintptr_t stream) {
            flash_rt::kernels::gated_deltanet_chunk_smem_qwen36_bf16(
                to_ptr(q), to_ptr(k), to_ptr(v),
                to_ptr(g), to_ptr(beta),
                to_ptr(state), to_ptr(out),
                S, num_v_heads, head_k_dim, head_v_dim,
                use_qk_l2norm, to_stream(stream));
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("g"), py::arg("beta"),
        py::arg("state"), py::arg("out"),
        py::arg("S"), py::arg("num_v_heads"),
        py::arg("head_k_dim"), py::arg("head_v_dim"),
        py::arg("use_qk_l2norm") = true, py::arg("stream") = 0);

    m.def("qwen36_lin_split_qkv_broadcast_bf16",
        [](uintptr_t conv_out, uintptr_t q48,
           uintptr_t k48, uintptr_t v48,
           int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_lin_split_qkv_broadcast_bf16(
                to_ptr(conv_out), to_ptr(q48),
                to_ptr(k48), to_ptr(v48),
                S, to_stream(stream));
        },
        py::arg("conv_out"), py::arg("q48"),
        py::arg("k48"), py::arg("v48"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("qwen36_lin_split_qkv_gqa_bf16",
        [](uintptr_t conv_out, uintptr_t q16,
           uintptr_t k16, uintptr_t v48,
           int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_lin_split_qkv_gqa_bf16(
                to_ptr(conv_out), to_ptr(q16),
                to_ptr(k16), to_ptr(v48),
                S, to_stream(stream));
        },
        py::arg("conv_out"), py::arg("q16"),
        py::arg("k16"), py::arg("v48"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("qwen36_split_q_gate_bf16",
        [](uintptr_t q_proj, uintptr_t q_pre,
           uintptr_t gate, int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_split_q_gate_bf16(
                to_ptr(q_proj), to_ptr(q_pre), to_ptr(gate),
                S, to_stream(stream));
        },
        py::arg("q_proj"), py::arg("q_pre"), py::arg("gate"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("nexn2_lin_split_qkv_broadcast_bf16",
        [](uintptr_t conv_out, uintptr_t q32,
           uintptr_t k32, uintptr_t v32,
           int S, uintptr_t stream) {
            flash_rt::kernels::nexn2_lin_split_qkv_broadcast_bf16(
                to_ptr(conv_out), to_ptr(q32),
                to_ptr(k32), to_ptr(v32),
                S, to_stream(stream));
        },
        py::arg("conv_out"), py::arg("q32"),
        py::arg("k32"), py::arg("v32"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("nexn2_split_q_gate_bf16",
        [](uintptr_t q_proj, uintptr_t q_pre,
           uintptr_t gate, int S, uintptr_t stream) {
            flash_rt::kernels::nexn2_split_q_gate_bf16(
                to_ptr(q_proj), to_ptr(q_pre), to_ptr(gate),
                S, to_stream(stream));
        },
        py::arg("q_proj"), py::arg("q_pre"), py::arg("gate"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("nexn2_bf16_matvec_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t out,
           int N, int K, uintptr_t stream) -> int {
            return flash_rt::kernels::nexn2_bf16_matvec_bf16(
                to_ptr(x), to_ptr(W), to_ptr(out), N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("out"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def("nexn2_w4a16_matvec_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t sfb, uintptr_t out,
           int N, int K, float alpha, uintptr_t stream) -> int {
            return flash_rt::kernels::nexn2_w4a16_matvec_bf16(
                to_ptr(x), to_ptr(W), to_ptr(sfb), to_ptr(out),
                N, K, alpha, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("sfb"), py::arg("out"),
        py::arg("N"), py::arg("K"), py::arg("alpha"), py::arg("stream") = 0);

    m.def("nexn2_moe_grouped_gemv_bf16",
        [](uintptr_t A_stack, uintptr_t B_stack, uintptr_t D,
           uintptr_t SFA_stack, uintptr_t SFB_stack,
           uintptr_t alpha_stack, uintptr_t expert_idx,
           int E, int N, int K,
           long a_stride, long sfa_stride, long w_stride, long sfb_stride,
           uintptr_t stream) -> int {
            return flash_rt::gemm::nexn2_moe_grouped_gemv_bf16(
                to_ptr(A_stack), to_ptr(B_stack), to_ptr(D),
                to_ptr(SFA_stack), to_ptr(SFB_stack),
                to_ptr(alpha_stack), to_ptr(expert_idx),
                E, N, K, a_stride, sfa_stride, w_stride, sfb_stride,
                to_stream(stream));
        },
        py::arg("A_stack"), py::arg("B_stack"), py::arg("D"),
        py::arg("SFA_stack"), py::arg("SFB_stack"),
        py::arg("alpha_stack"), py::arg("expert_idx"),
        py::arg("E"), py::arg("N"), py::arg("K"),
        py::arg("a_stride"), py::arg("sfa_stride"),
        py::arg("w_stride"), py::arg("sfb_stride"),
        py::arg("stream") = 0);

    m.def("qwen36_gdn_gating_bf16",
        [](uintptr_t a, uintptr_t b,
           uintptr_t neg_exp_A_log, uintptr_t dt_bias,
           uintptr_t g_out, uintptr_t beta_out,
           int S, int num_heads, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_gating_bf16(
                to_ptr(a), to_ptr(b),
                reinterpret_cast<const float*>(neg_exp_A_log),
                reinterpret_cast<const float*>(dt_bias),
                to_ptr(g_out), to_ptr(beta_out),
                S, num_heads, to_stream(stream));
        },
        py::arg("a"), py::arg("b"),
        py::arg("neg_exp_A_log"), py::arg("dt_bias"),
        py::arg("g_out"), py::arg("beta_out"),
        py::arg("S"), py::arg("num_heads"),
        py::arg("stream") = 0);

    m.def("qwen36_gdn_gating_strided_bf16",
        [](uintptr_t a, uintptr_t b,
           uintptr_t neg_exp_A_log, uintptr_t dt_bias,
           uintptr_t g_out, uintptr_t beta_out,
           int S, int num_heads, int a_stride, int b_stride,
           uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_gating_strided_bf16(
                to_ptr(a), to_ptr(b),
                reinterpret_cast<const float*>(neg_exp_A_log),
                reinterpret_cast<const float*>(dt_bias),
                to_ptr(g_out), to_ptr(beta_out),
                S, num_heads, a_stride, b_stride, to_stream(stream));
        },
        py::arg("a"), py::arg("b"),
        py::arg("neg_exp_A_log"), py::arg("dt_bias"),
        py::arg("g_out"), py::arg("beta_out"),
        py::arg("S"), py::arg("num_heads"),
        py::arg("a_stride"), py::arg("b_stride"),
        py::arg("stream") = 0);

    m.def("qwen36_gdn_chunk_from_conv_smem_bf16",
        [](uintptr_t conv_out, uintptr_t a, uintptr_t b,
           uintptr_t neg_exp_A_log, uintptr_t dt_bias,
           uintptr_t state, uintptr_t out,
           int S, int num_v_heads, bool use_qk_l2norm,
           uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_chunk_from_conv_smem_bf16(
                to_ptr(conv_out), to_ptr(a), to_ptr(b),
                reinterpret_cast<const float*>(neg_exp_A_log),
                reinterpret_cast<const float*>(dt_bias),
                to_ptr(state), to_ptr(out),
                S, num_v_heads, use_qk_l2norm, to_stream(stream));
        },
        py::arg("conv_out"), py::arg("a"), py::arg("b"),
        py::arg("neg_exp_A_log"), py::arg("dt_bias"),
        py::arg("state"), py::arg("out"),
        py::arg("S"), py::arg("num_v_heads"),
        py::arg("use_qk_l2norm") = true, py::arg("stream") = 0);

    m.def("qwen36_gdn_chunk_from_conv_smem_strided_bf16",
        [](uintptr_t conv_out, uintptr_t a, uintptr_t b,
           uintptr_t neg_exp_A_log, uintptr_t dt_bias,
           uintptr_t state, uintptr_t out,
           int S, int num_v_heads, int a_stride, int b_stride,
           bool use_qk_l2norm, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_chunk_from_conv_smem_strided_bf16(
                to_ptr(conv_out), to_ptr(a), to_ptr(b),
                reinterpret_cast<const float*>(neg_exp_A_log),
                reinterpret_cast<const float*>(dt_bias),
                to_ptr(state), to_ptr(out),
                S, num_v_heads, a_stride, b_stride,
                use_qk_l2norm, to_stream(stream));
        },
        py::arg("conv_out"), py::arg("a"), py::arg("b"),
        py::arg("neg_exp_A_log"), py::arg("dt_bias"),
        py::arg("state"), py::arg("out"),
        py::arg("S"), py::arg("num_v_heads"),
        py::arg("a_stride"), py::arg("b_stride"),
        py::arg("use_qk_l2norm") = true, py::arg("stream") = 0);

    m.def("qwen36_gdn_wy_norm_cumsum_bf16",
        [](uintptr_t q16, uintptr_t k16, uintptr_t g,
           uintptr_t q16_l2, uintptr_t k16_l2, uintptr_t g_cumsum,
           int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_wy_norm_cumsum_bf16(
                to_ptr(q16), to_ptr(k16), to_ptr(g),
                to_ptr(q16_l2), to_ptr(k16_l2), to_ptr(g_cumsum),
                S, to_stream(stream));
        },
        py::arg("q16"), py::arg("k16"), py::arg("g"),
        py::arg("q16_l2"), py::arg("k16_l2"), py::arg("g_cumsum"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("qwen36_gdn_wy_norm_cumsum_pack_q_bf16",
        [](uintptr_t q16, uintptr_t k16, uintptr_t g,
           uintptr_t q16_l2, uintptr_t k16_l2,
           uintptr_t q_pack_hv, uintptr_t g_cumsum,
           int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_wy_norm_cumsum_pack_q_bf16(
                to_ptr(q16), to_ptr(k16), to_ptr(g),
                to_ptr(q16_l2), to_ptr(k16_l2),
                to_ptr(q_pack_hv), to_ptr(g_cumsum),
                S, to_stream(stream));
        },
        py::arg("q16"), py::arg("k16"), py::arg("g"),
        py::arg("q16_l2"), py::arg("k16_l2"),
        py::arg("q_pack_hv"), py::arg("g_cumsum"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("qwen36_gdn_wy_norm_cumsum_pack_qk_bf16",
        [](uintptr_t q16, uintptr_t k16, uintptr_t g,
           uintptr_t q16_l2, uintptr_t k16_l2,
           uintptr_t q_pack_hv, uintptr_t k_pack_hk,
           uintptr_t g_cumsum, int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_wy_norm_cumsum_pack_qk_bf16(
                to_ptr(q16), to_ptr(k16), to_ptr(g),
                to_ptr(q16_l2), to_ptr(k16_l2),
                to_ptr(q_pack_hv), to_ptr(k_pack_hk),
                to_ptr(g_cumsum), S, to_stream(stream));
        },
        py::arg("q16"), py::arg("k16"), py::arg("g"),
        py::arg("q16_l2"), py::arg("k16_l2"),
        py::arg("q_pack_hv"), py::arg("k_pack_hk"),
        py::arg("g_cumsum"), py::arg("S"), py::arg("stream") = 0);

    m.def("qwen36_gdn_wy_kkt_b64_bf16",
        [](uintptr_t k16_l2, uintptr_t beta, uintptr_t g_cumsum,
           uintptr_t A, int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_wy_kkt_b64_bf16(
                to_ptr(k16_l2), to_ptr(beta), to_ptr(g_cumsum),
                to_ptr(A), S, to_stream(stream));
        },
        py::arg("k16_l2"), py::arg("beta"), py::arg("g_cumsum"),
        py::arg("A"), py::arg("S"), py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_kkt_b64_bf16_cublaslt",
        [](uintptr_t k_l2, uintptr_t beta, uintptr_t g_cumsum,
           uintptr_t k_pack, uintptr_t kkt_base, uintptr_t A,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::gdn_wy_kkt_b64_bf16_cublaslt(
                to_ptr(k_l2), to_ptr(beta), to_ptr(g_cumsum),
                to_ptr(k_pack), to_ptr(kkt_base), to_ptr(A),
                S, num_k_heads, num_v_heads, head_dim, qk_group,
                to_stream(stream));
        },
        py::arg("k_l2"), py::arg("beta"), py::arg("g_cumsum"),
        py::arg("k_pack"), py::arg("kkt_base"), py::arg("A"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_kkt_b64_bf16_cublaslt_packed_k",
        [](uintptr_t k_pack, uintptr_t beta, uintptr_t g_cumsum,
           uintptr_t kkt_base, uintptr_t A,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_kkt_b64_bf16_cublaslt_packed_k(
                    to_ptr(k_pack), to_ptr(beta), to_ptr(g_cumsum),
                    to_ptr(kkt_base), to_ptr(A),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_pack"), py::arg("beta"), py::arg("g_cumsum"),
        py::arg("kkt_base"), py::arg("A"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_kkt_b64_bf16_cublaslt_packed_k_only",
        [](uintptr_t k_pack, uintptr_t kkt_base,
           int S, int num_k_heads, int head_dim, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_kkt_b64_bf16_cublaslt_packed_k_only(
                    to_ptr(k_pack), to_ptr(kkt_base),
                    S, num_k_heads, head_dim, to_stream(stream));
        },
        py::arg("k_pack"), py::arg("kkt_base"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("head_dim"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_kkt_b64_bf16_cublaslt_nogate",
        [](uintptr_t k_l2, uintptr_t beta,
           uintptr_t k_pack, uintptr_t kkt_base, uintptr_t A,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_kkt_b64_bf16_cublaslt_nogate(
                    to_ptr(k_l2), to_ptr(beta), to_ptr(k_pack),
                    to_ptr(kkt_base), to_ptr(A),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("beta"),
        py::arg("k_pack"), py::arg("kkt_base"), py::arg("A"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt",
        [](uintptr_t k_l2, uintptr_t v, uintptr_t beta,
           uintptr_t g_cumsum, uintptr_t Ai,
           uintptr_t Ai_pack, uintptr_t rhs_w, uintptr_t rhs_u,
           uintptr_t w_pack, uintptr_t u_pack,
           uintptr_t w, uintptr_t u,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_recompute_wu_b64_bf16_cublaslt(
                    to_ptr(k_l2), to_ptr(v), to_ptr(beta),
                    to_ptr(g_cumsum), to_ptr(Ai),
                    to_ptr(Ai_pack), to_ptr(rhs_w), to_ptr(rhs_u),
                    to_ptr(w_pack), to_ptr(u_pack),
                    to_ptr(w), to_ptr(u),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("v"), py::arg("beta"),
        py::arg("g_cumsum"), py::arg("Ai"),
        py::arg("Ai_pack"), py::arg("rhs_w"), py::arg("rhs_u"),
        py::arg("w_pack"), py::arg("u_pack"),
        py::arg("w"), py::arg("u"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed",
        [](uintptr_t k_l2, uintptr_t v, uintptr_t beta,
           uintptr_t g_cumsum, uintptr_t Ai,
           uintptr_t Ai_pack, uintptr_t rhs_w, uintptr_t rhs_u,
           uintptr_t w_pack, uintptr_t u_pack,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_recompute_wu_b64_bf16_cublaslt_packed(
                    to_ptr(k_l2), to_ptr(v), to_ptr(beta),
                    to_ptr(g_cumsum), to_ptr(Ai),
                    to_ptr(Ai_pack), to_ptr(rhs_w), to_ptr(rhs_u),
                    to_ptr(w_pack), to_ptr(u_pack),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("v"), py::arg("beta"),
        py::arg("g_cumsum"), py::arg("Ai"),
        py::arg("Ai_pack"), py::arg("rhs_w"), py::arg("rhs_u"),
        py::arg("w_pack"), py::arg("u_pack"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs",
        [](uintptr_t k_l2, uintptr_t v, uintptr_t beta,
           uintptr_t g_cumsum, uintptr_t Ai_pack,
           uintptr_t rhs_w, uintptr_t rhs_u,
           uintptr_t w_pack, uintptr_t u_pack,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs(
                    to_ptr(k_l2), to_ptr(v), to_ptr(beta),
                    to_ptr(g_cumsum), to_ptr(Ai_pack),
                    to_ptr(rhs_w), to_ptr(rhs_u),
                    to_ptr(w_pack), to_ptr(u_pack),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("v"), py::arg("beta"),
        py::arg("g_cumsum"), py::arg("Ai_pack"),
        py::arg("rhs_w"), py::arg("rhs_u"),
        py::arg("w_pack"), py::arg("u_pack"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs_nogate",
        [](uintptr_t k_l2, uintptr_t v, uintptr_t beta,
           uintptr_t g_cumsum, uintptr_t Ai_pack,
           uintptr_t rhs_w, uintptr_t rhs_u,
           uintptr_t w_pack, uintptr_t u_pack,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs_nogate(
                    to_ptr(k_l2), to_ptr(v), to_ptr(beta),
                    to_ptr(g_cumsum), to_ptr(Ai_pack),
                    to_ptr(rhs_w), to_ptr(rhs_u),
                    to_ptr(w_pack), to_ptr(u_pack),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("v"), py::arg("beta"),
        py::arg("g_cumsum"), py::arg("Ai_pack"),
        py::arg("rhs_w"), py::arg("rhs_u"),
        py::arg("w_pack"), py::arg("u_pack"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_solve_tril_b64_f32_parallel",
        [](uintptr_t A, uintptr_t Ai, int S, int num_v_heads,
           uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_solve_tril_b64_f32_parallel(
                    to_ptr(A), to_ptr(Ai), S, num_v_heads,
                    to_stream(stream));
        },
        py::arg("A"), py::arg("Ai"), py::arg("S"),
        py::arg("num_v_heads"), py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_solve_tril_b64_f32_parallel_pack",
        [](uintptr_t A, uintptr_t Ai, uintptr_t Ai_pack,
           int S, int num_v_heads, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_solve_tril_b64_f32_parallel_pack(
                    to_ptr(A), to_ptr(Ai), to_ptr(Ai_pack),
                    S, num_v_heads, to_stream(stream));
        },
        py::arg("A"), py::arg("Ai"), py::arg("Ai_pack"),
        py::arg("S"), py::arg("num_v_heads"), py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack",
        [](uintptr_t A, uintptr_t Ai, uintptr_t Ai_pack,
           int S, int num_v_heads, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_solve_tril_b64_f32_fused_pack(
                    to_ptr(A), to_ptr(Ai), to_ptr(Ai_pack),
                    S, num_v_heads, to_stream(stream));
        },
        py::arg("A"), py::arg("Ai"), py::arg("Ai_pack"),
        py::arg("S"), py::arg("num_v_heads"), py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack_only",
        [](uintptr_t A, uintptr_t Ai_pack,
           int S, int num_v_heads, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_solve_tril_b64_f32_fused_pack_only(
                    to_ptr(A), to_ptr(Ai_pack),
                    S, num_v_heads, to_stream(stream));
        },
        py::arg("A"), py::arg("Ai_pack"),
        py::arg("S"), py::arg("num_v_heads"), py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_solve_tril_b64_from_kkt_pack_only",
        [](uintptr_t kkt_base, uintptr_t beta, uintptr_t g_cumsum,
           uintptr_t Ai_pack,
           int S, int num_k_heads, int num_v_heads, int qk_group,
           uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_solve_tril_b64_from_kkt_pack_only(
                    to_ptr(kkt_base), to_ptr(beta), to_ptr(g_cumsum),
                    to_ptr(Ai_pack),
                    S, num_k_heads, num_v_heads, qk_group,
                    to_stream(stream));
        },
        py::arg("kkt_base"), py::arg("beta"), py::arg("g_cumsum"),
        py::arg("Ai_pack"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("qk_group"), py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_output_o_b64_bf16_cublaslt",
        [](uintptr_t q_l2, uintptr_t k_l2, uintptr_t v_new,
           uintptr_t h0, uintptr_t g_cumsum,
           uintptr_t q_pack, uintptr_t k_pack_hv, uintptr_t v_pack,
           uintptr_t qk_base, uintptr_t local_a_pack,
           uintptr_t qh_pack, uintptr_t local_pack, uintptr_t out,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_output_o_b64_bf16_cublaslt(
                    to_ptr(q_l2), to_ptr(k_l2), to_ptr(v_new),
                    to_ptr(h0), to_ptr(g_cumsum),
                    to_ptr(q_pack), to_ptr(k_pack_hv), to_ptr(v_pack),
                    to_ptr(qk_base), to_ptr(local_a_pack),
                    to_ptr(qh_pack), to_ptr(local_pack), to_ptr(out),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("q_l2"), py::arg("k_l2"), py::arg("v_new"),
        py::arg("h0"), py::arg("g_cumsum"),
        py::arg("q_pack"), py::arg("k_pack_hv"), py::arg("v_pack"),
        py::arg("qk_base"), py::arg("local_a_pack"),
        py::arg("qh_pack"), py::arg("local_pack"), py::arg("out"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_k",
        [](uintptr_t q_l2, uintptr_t k_pack_hv, uintptr_t v_new,
           uintptr_t h0, uintptr_t g_cumsum,
           uintptr_t q_pack, uintptr_t v_pack,
           uintptr_t qk_base, uintptr_t local_a_pack,
           uintptr_t qh_pack, uintptr_t local_pack, uintptr_t out,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_output_o_b64_bf16_cublaslt_packed_k(
                    to_ptr(q_l2), to_ptr(k_pack_hv), to_ptr(v_new),
                    to_ptr(h0), to_ptr(g_cumsum),
                    to_ptr(q_pack), to_ptr(v_pack),
                    to_ptr(qk_base), to_ptr(local_a_pack),
                    to_ptr(qh_pack), to_ptr(local_pack), to_ptr(out),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("q_l2"), py::arg("k_pack_hv"), py::arg("v_new"),
        py::arg("h0"), py::arg("g_cumsum"),
        py::arg("q_pack"), py::arg("v_pack"),
        py::arg("qk_base"), py::arg("local_a_pack"),
        py::arg("qh_pack"), py::arg("local_pack"), py::arg("out"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_kv",
        [](uintptr_t q_l2, uintptr_t k_pack_hv, uintptr_t v_pack,
           uintptr_t h0, uintptr_t g_cumsum,
           uintptr_t q_pack, uintptr_t qk_base,
           uintptr_t local_a_pack, uintptr_t qh_pack,
           uintptr_t local_pack, uintptr_t out,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_output_o_b64_bf16_cublaslt_packed_kv(
                    to_ptr(q_l2), to_ptr(k_pack_hv), to_ptr(v_pack),
                    to_ptr(h0), to_ptr(g_cumsum),
                    to_ptr(q_pack), to_ptr(qk_base), to_ptr(local_a_pack),
                    to_ptr(qh_pack), to_ptr(local_pack), to_ptr(out),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("q_l2"), py::arg("k_pack_hv"), py::arg("v_pack"),
        py::arg("h0"), py::arg("g_cumsum"),
        py::arg("q_pack"), py::arg("qk_base"),
        py::arg("local_a_pack"), py::arg("qh_pack"),
        py::arg("local_pack"), py::arg("out"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv",
        [](uintptr_t q_pack, uintptr_t k_pack_hv, uintptr_t v_pack,
           uintptr_t h0, uintptr_t g_cumsum,
           uintptr_t qk_base, uintptr_t local_a_pack,
           uintptr_t qh_pack, uintptr_t local_pack, uintptr_t out,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv(
                    to_ptr(q_pack), to_ptr(k_pack_hv), to_ptr(v_pack),
                    to_ptr(h0), to_ptr(g_cumsum),
                    to_ptr(qk_base), to_ptr(local_a_pack),
                    to_ptr(qh_pack), to_ptr(local_pack), to_ptr(out),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("q_pack"), py::arg("k_pack_hv"), py::arg("v_pack"),
        py::arg("h0"), py::arg("g_cumsum"),
        py::arg("qk_base"), py::arg("local_a_pack"),
        py::arg("qh_pack"), py::arg("local_pack"), py::arg("out"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    // FLA-style hand-tuned recompute_wu. Fuses pack_rhs + 2 cublasLt mmas
    // into a single kernel; only writes w_pack / u_pack.
    m.def("linear_attn_gdn_wy_recompute_wu_b64_bf16_mma_fla",
        [](uintptr_t k_l2, uintptr_t v, uintptr_t beta, uintptr_t g_cumsum,
           uintptr_t Ai_pack, uintptr_t w_pack, uintptr_t u_pack,
           int S, int num_k_heads, int num_v_heads, int head_dim,
           int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_recompute_wu_b64_bf16_mma_fla(
                    to_ptr(k_l2), to_ptr(v), to_ptr(beta), to_ptr(g_cumsum),
                    to_ptr(Ai_pack), to_ptr(w_pack), to_ptr(u_pack),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("v"), py::arg("beta"), py::arg("g_cumsum"),
        py::arg("Ai_pack"), py::arg("w_pack"), py::arg("u_pack"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    // FLA-style hand-tuned output_o. Drop-in for output_o packed_qkv but
    // no scratch buffers needed.
    m.def("linear_attn_gdn_wy_output_o_b64_bf16_mma_fla",
        [](uintptr_t q_pack, uintptr_t k_pack_hv, uintptr_t v_pack,
           uintptr_t h, uintptr_t g_cumsum, uintptr_t out,
           int S, int num_v_heads, int head_dim, double scale,
           uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_output_o_b64_bf16_mma_fla(
                    to_ptr(q_pack), to_ptr(k_pack_hv), to_ptr(v_pack),
                    to_ptr(h), to_ptr(g_cumsum), to_ptr(out),
                    S, num_v_heads, head_dim,
                    static_cast<float>(scale), to_stream(stream));
        },
        py::arg("q_pack"), py::arg("k_pack_hv"), py::arg("v_pack"),
        py::arg("h"), py::arg("g_cumsum"), py::arg("out"),
        py::arg("S"), py::arg("num_v_heads"), py::arg("head_dim"),
        py::arg("scale"),
        py::arg("stream") = 0);

    // Raw-K variant: output_o reads k_l2 (S, num_k_heads, head_dim)
    // directly and does GQA expansion in-kernel.
    m.def("linear_attn_gdn_wy_output_o_b64_bf16_mma_fla_rawk",
        [](uintptr_t q_pack, uintptr_t k_l2, uintptr_t v_pack,
           uintptr_t h, uintptr_t g_cumsum, uintptr_t out,
           int S, int num_k_heads, int num_v_heads, int head_dim,
           int qk_group, double scale, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_output_o_b64_bf16_mma_fla_rawk(
                    to_ptr(q_pack), to_ptr(k_l2), to_ptr(v_pack),
                    to_ptr(h), to_ptr(g_cumsum), to_ptr(out),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    static_cast<float>(scale), to_stream(stream));
        },
        py::arg("q_pack"), py::arg("k_l2"), py::arg("v_pack"),
        py::arg("h"), py::arg("g_cumsum"), py::arg("out"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"), py::arg("scale"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt",
        [](uintptr_t k_l2, uintptr_t u, uintptr_t w,
           uintptr_t g_cumsum, uintptr_t state,
           uintptr_t h0, uintptr_t v_new,
           uintptr_t k_pack_hv, uintptr_t w_pack, uintptr_t u_pack,
           uintptr_t wh_pack, uintptr_t decayed_v_pack,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_chunk_h_b64_bf16_cublaslt(
                    to_ptr(k_l2), to_ptr(u), to_ptr(w), to_ptr(g_cumsum),
                    to_ptr(state), to_ptr(h0), to_ptr(v_new),
                    to_ptr(k_pack_hv), to_ptr(w_pack), to_ptr(u_pack),
                    to_ptr(wh_pack), to_ptr(decayed_v_pack),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("u"), py::arg("w"),
        py::arg("g_cumsum"), py::arg("state"),
        py::arg("h0"), py::arg("v_new"),
        py::arg("k_pack_hv"), py::arg("w_pack"), py::arg("u_pack"),
        py::arg("wh_pack"), py::arg("decayed_v_pack"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32state",
        [](uintptr_t k_l2, uintptr_t u, uintptr_t w,
           uintptr_t g_cumsum, uintptr_t state,
           uintptr_t h0, uintptr_t v_new,
           uintptr_t k_pack_hv, uintptr_t w_pack, uintptr_t u_pack,
           uintptr_t wh_pack, uintptr_t decayed_v_pack,
           uintptr_t state_f32, uintptr_t delta_f32,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_chunk_h_b64_bf16_cublaslt_f32state(
                    to_ptr(k_l2), to_ptr(u), to_ptr(w), to_ptr(g_cumsum),
                    to_ptr(state), to_ptr(h0), to_ptr(v_new),
                    to_ptr(k_pack_hv), to_ptr(w_pack), to_ptr(u_pack),
                    to_ptr(wh_pack), to_ptr(decayed_v_pack),
                    to_ptr(state_f32), to_ptr(delta_f32),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("u"), py::arg("w"),
        py::arg("g_cumsum"), py::arg("state"),
        py::arg("h0"), py::arg("v_new"),
        py::arg("k_pack_hv"), py::arg("w_pack"), py::arg("u_pack"),
        py::arg("wh_pack"), py::arg("decayed_v_pack"),
        py::arg("state_f32"), py::arg("delta_f32"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm",
        [](uintptr_t k_l2, uintptr_t u, uintptr_t w,
           uintptr_t g_cumsum, uintptr_t state,
           uintptr_t h0, uintptr_t v_new,
           uintptr_t k_pack_hv, uintptr_t w_pack, uintptr_t u_pack,
           uintptr_t wh_pack, uintptr_t decayed_v_pack,
           uintptr_t state_f32, uintptr_t chunk_f32, uintptr_t acc_f32,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm(
                    to_ptr(k_l2), to_ptr(u), to_ptr(w), to_ptr(g_cumsum),
                    to_ptr(state), to_ptr(h0), to_ptr(v_new),
                    to_ptr(k_pack_hv), to_ptr(w_pack), to_ptr(u_pack),
                    to_ptr(wh_pack), to_ptr(decayed_v_pack),
                    to_ptr(state_f32), to_ptr(chunk_f32), to_ptr(acc_f32),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("u"), py::arg("w"),
        py::arg("g_cumsum"), py::arg("state"),
        py::arg("h0"), py::arg("v_new"),
        py::arg("k_pack_hv"), py::arg("w_pack"), py::arg("u_pack"),
        py::arg("wh_pack"), py::arg("decayed_v_pack"),
        py::arg("state_f32"), py::arg("chunk_f32"), py::arg("acc_f32"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm_packed_wu",
        [](uintptr_t k_l2, uintptr_t w_pack, uintptr_t u_pack,
           uintptr_t g_cumsum, uintptr_t state,
           uintptr_t h0, uintptr_t v_new,
           uintptr_t k_pack_hv, uintptr_t decayed_v_pack,
           uintptr_t state_f32, uintptr_t chunk_f32, uintptr_t acc_f32,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm_packed_wu(
                    to_ptr(k_l2), to_ptr(w_pack), to_ptr(u_pack),
                    to_ptr(g_cumsum), to_ptr(state), to_ptr(h0),
                    to_ptr(v_new), to_ptr(k_pack_hv),
                    to_ptr(decayed_v_pack), to_ptr(state_f32),
                    to_ptr(chunk_f32), to_ptr(acc_f32),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("w_pack"), py::arg("u_pack"),
        py::arg("g_cumsum"), py::arg("state"),
        py::arg("h0"), py::arg("v_new"),
        py::arg("k_pack_hv"), py::arg("decayed_v_pack"),
        py::arg("state_f32"), py::arg("chunk_f32"), py::arg("acc_f32"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu",
        [](uintptr_t k_l2, uintptr_t w_pack, uintptr_t u_pack,
           uintptr_t g_cumsum, uintptr_t state,
           uintptr_t h0, uintptr_t v_new,
           uintptr_t k_pack_hv, uintptr_t wh_pack,
           uintptr_t decayed_v_pack,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu(
                    to_ptr(k_l2), to_ptr(w_pack), to_ptr(u_pack),
                    to_ptr(g_cumsum), to_ptr(state), to_ptr(h0),
                    to_ptr(v_new), to_ptr(k_pack_hv), to_ptr(wh_pack),
                    to_ptr(decayed_v_pack),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("w_pack"), py::arg("u_pack"),
        py::arg("g_cumsum"), py::arg("state"),
        py::arg("h0"), py::arg("v_new"),
        py::arg("k_pack_hv"), py::arg("wh_pack"),
        py::arg("decayed_v_pack"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    // FLA-style mma.sync + cp.async chunk_h kernel.
    // RAW input layout, head_dim must be 128. See .cuh for shape contract.
    // v_new_packed and k_pack_hv are optional packed side outputs (pass 0 to skip).
    m.def("linear_attn_gdn_wy_chunk_h_b64_bf16_mma_fla",
        [](uintptr_t k_l2, uintptr_t w, uintptr_t u,
           uintptr_t g_cumsum, uintptr_t state,
           uintptr_t h_out, uintptr_t v_new,
           uintptr_t v_new_packed, uintptr_t k_pack_hv,
           int S, int num_k_heads, int num_v_heads,
           int head_dim, int qk_group, uintptr_t stream) {
            flash_rt::kernels::linear_attention::
                gdn_wy_chunk_h_b64_bf16_mma_fla(
                    to_ptr(k_l2), to_ptr(w), to_ptr(u),
                    to_ptr(g_cumsum), to_ptr(state),
                    to_ptr(h_out), to_ptr(v_new),
                    to_ptr(v_new_packed), to_ptr(k_pack_hv),
                    S, num_k_heads, num_v_heads, head_dim, qk_group,
                    to_stream(stream));
        },
        py::arg("k_l2"), py::arg("w"), py::arg("u"),
        py::arg("g_cumsum"), py::arg("state"),
        py::arg("h_out"), py::arg("v_new"),
        py::arg("v_new_packed"), py::arg("k_pack_hv"),
        py::arg("S"), py::arg("num_k_heads"), py::arg("num_v_heads"),
        py::arg("head_dim"), py::arg("qk_group"),
        py::arg("stream") = 0);

    m.def("qwen36_gdn_wy_solve_tril_b64_f32",
        [](uintptr_t A, uintptr_t Ai, int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_wy_solve_tril_b64_f32(
                to_ptr(A), to_ptr(Ai), S, to_stream(stream));
        },
        py::arg("A"), py::arg("Ai"), py::arg("S"),
        py::arg("stream") = 0);

    m.def("qwen36_gdn_wy_recompute_wu_b64_bf16",
        [](uintptr_t k16_l2, uintptr_t v48, uintptr_t beta,
           uintptr_t g_cumsum, uintptr_t Ai,
           uintptr_t w48, uintptr_t u48,
           int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_wy_recompute_wu_b64_bf16(
                to_ptr(k16_l2), to_ptr(v48), to_ptr(beta),
                to_ptr(g_cumsum), to_ptr(Ai),
                to_ptr(w48), to_ptr(u48),
                S, to_stream(stream));
        },
        py::arg("k16_l2"), py::arg("v48"), py::arg("beta"),
        py::arg("g_cumsum"), py::arg("Ai"),
        py::arg("w48"), py::arg("u48"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("qwen36_gdn_wy_chunk_h_b64_bf16",
        [](uintptr_t k16_l2, uintptr_t u48, uintptr_t w48,
           uintptr_t g_cumsum, uintptr_t state,
           uintptr_t h0, uintptr_t v_new,
           int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_wy_chunk_h_b64_bf16(
                to_ptr(k16_l2), to_ptr(u48), to_ptr(w48),
                to_ptr(g_cumsum), to_ptr(state),
                to_ptr(h0), to_ptr(v_new),
                S, to_stream(stream));
        },
        py::arg("k16_l2"), py::arg("u48"), py::arg("w48"),
        py::arg("g_cumsum"), py::arg("state"),
        py::arg("h0"), py::arg("v_new"),
        py::arg("S"), py::arg("stream") = 0);

    m.def("qwen36_gdn_wy_output_o_b64_bf16",
        [](uintptr_t q16_l2, uintptr_t k16_l2, uintptr_t v_new,
           uintptr_t h0, uintptr_t g_cumsum, uintptr_t out,
           int S, uintptr_t stream) {
            flash_rt::kernels::qwen36_gdn_wy_output_o_b64_bf16(
                to_ptr(q16_l2), to_ptr(k16_l2), to_ptr(v_new),
                to_ptr(h0), to_ptr(g_cumsum), to_ptr(out),
                S, to_stream(stream));
        },
        py::arg("q16_l2"), py::arg("k16_l2"), py::arg("v_new"),
        py::arg("h0"), py::arg("g_cumsum"), py::arg("out"),
        py::arg("S"), py::arg("stream") = 0);

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


    // Hand-tuned inline-PTX FP8 GEMM (no cutlass scaffold).
#define BIND_HANDTUNED(NAME)                                                 \
    m.def("ht_" #NAME,                                                       \
        [](uintptr_t A, uintptr_t B, uintptr_t D,                            \
           int M, int N, int K, float alpha, uintptr_t stream) {             \
            return flash_rt::gemm::smallM_hand::NAME(                        \
                to_ptr(A), to_ptr(B), to_ptr(D),                             \
                M, N, K, alpha, to_stream(stream));                          \
        },                                                                   \
        py::arg("A"), py::arg("B"), py::arg("D"),                            \
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"),          \
        py::arg("stream") = 0)

    BIND_HANDTUNED(fp8_gemm_16x64x128_w4);
    BIND_HANDTUNED(fp8_gemm_16x128x128_w4);
    BIND_HANDTUNED(fp8_gemm_16x256x128_w8);
    BIND_HANDTUNED(fp8_gemm_32x64x128_w4);
    BIND_HANDTUNED(fp8_gemm_32x128x128_w4);
    BIND_HANDTUNED(fp8_gemm_32x128x128_w8);
    BIND_HANDTUNED(fp8_gemm_16x64x128_w4_s3);
    BIND_HANDTUNED(fp8_gemm_16x128x128_w4_s3);
    BIND_HANDTUNED(fp8_gemm_32x64x128_w4_s3);
    BIND_HANDTUNED(fp8_gemm_32x128x128_w4_s3);
    BIND_HANDTUNED(fp8_gemm_16x64x256_w4);
    BIND_HANDTUNED(fp8_gemm_16x128x256_w4);
    BIND_HANDTUNED(fp8_gemm_32x64x256_w4);
    BIND_HANDTUNED(fp8_gemm_32x128x256_w4);
    BIND_HANDTUNED(fp8_gemm_16x192x128_w4);
    BIND_HANDTUNED(fp8_gemm_16x192x128_w8);
    BIND_HANDTUNED(fp8_gemm_32x192x128_w4);
    BIND_HANDTUNED(fp8_gemm_16x64x128_w4_s4);
    BIND_HANDTUNED(fp8_gemm_32x64x128_w4_s4);
    BIND_HANDTUNED(fp8_gemm_16x384x128_w8);
    BIND_HANDTUNED(fp8_gemm_32x384x128_w8);
    BIND_HANDTUNED(fp8_gemm_16x64x128_w8);
    BIND_HANDTUNED(fp8_gemm_32x64x128_w8);
    BIND_HANDTUNED(fp8_gemm_32x64x128_w4_s5);
    BIND_HANDTUNED(fp8_gemm_16x64x64_w4);
    BIND_HANDTUNED(fp8_gemm_16x128x64_w4);
    BIND_HANDTUNED(fp8_gemm_32x64x64_w4);
    BIND_HANDTUNED(fp8_gemm_32x128x64_w4);
    BIND_HANDTUNED(fp8_gemm_16x64x64_w4_s3);
    BIND_HANDTUNED(fp8_gemm_16x64x64_w4_s4);
    BIND_HANDTUNED(fp8_gemm_16x384x128_w4_big);
    BIND_HANDTUNED(fp8_gemm_32x384x128_w4_big);
    BIND_HANDTUNED(fp8_gemm_16x512x128_w8_big);
    BIND_HANDTUNED(fp8_gemm_16x256x128_w4_big);
    BIND_HANDTUNED(fp8_gemm_32x256x128_w4_big);
    BIND_HANDTUNED(fp8_gemm_64x64x128_w4);
    BIND_HANDTUNED(fp8_gemm_64x128x128_w4);
    BIND_HANDTUNED(fp8_gemm_64x128x128_w8);
    BIND_HANDTUNED(fp8_gemm_128x64x128_w4);
    BIND_HANDTUNED(fp8_gemm_128x128x128_w4);
    BIND_HANDTUNED(fp8_gemm_128x128x128_w8);
    BIND_HANDTUNED(fp8_gemm_64x256x128_w4_big);
    BIND_HANDTUNED(fp8_gemm_64x256x128_w8_big);
    BIND_HANDTUNED(fp8_gemm_128x256x128_w8_big);

#undef BIND_HANDTUNED

    // ldmatrix + 128B swizzle variants (v2 hand-tuned).
#define BIND_LDMATRIX(NAME)                                                  \
    m.def("ht_" #NAME,                                                       \
        [](uintptr_t A, uintptr_t B, uintptr_t D,                            \
           int M, int N, int K, float alpha, uintptr_t stream) {             \
            return flash_rt::gemm::smallM_ld::NAME(                          \
                to_ptr(A), to_ptr(B), to_ptr(D),                             \
                M, N, K, alpha, to_stream(stream));                          \
        },                                                                   \
        py::arg("A"), py::arg("B"), py::arg("D"),                            \
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"),          \
        py::arg("stream") = 0)

    BIND_LDMATRIX(ld_fp8_gemm_16x64x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_16x128x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_16x256x128_w8);
    BIND_LDMATRIX(ld_fp8_gemm_32x64x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_32x128x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_32x128x128_w8);
    BIND_LDMATRIX(ld_fp8_gemm_16x64x128_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_16x128x128_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_32x64x128_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_32x128x128_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_16x192x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_32x192x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_16x64x128_w4_s4);
    BIND_LDMATRIX(ld_fp8_gemm_16x64x128_w4_s5);
    BIND_LDMATRIX(ld_fp8_gemm_32x64x128_w4_s4);
    BIND_LDMATRIX(ld_fp8_gemm_32x64x128_w4_s5);
    BIND_LDMATRIX(ld_fp8_gemm_16x128x128_w4_s4);
    BIND_LDMATRIX(ld_fp8_gemm_32x128x128_w4_s4);
    BIND_LDMATRIX(ld_fp8_gemm_16x64x256_w4);
    BIND_LDMATRIX(ld_fp8_gemm_16x128x256_w4);
    BIND_LDMATRIX(ld_fp8_gemm_32x64x256_w4);
    BIND_LDMATRIX(ld_fp8_gemm_32x128x256_w4);
    BIND_LDMATRIX(ld_fp8_gemm_16x64x256_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_16x64x64_w4);
    BIND_LDMATRIX(ld_fp8_gemm_16x128x64_w4);
    BIND_LDMATRIX(ld_fp8_gemm_32x64x64_w4);
    BIND_LDMATRIX(ld_fp8_gemm_16x64x64_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_16x64x64_w4_s4);
    // und_qkv attack variants (M=188, K=512)
    BIND_LDMATRIX(ld_fp8_gemm_64x64x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_64x128x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_64x64x256_w4);
    BIND_LDMATRIX(ld_fp8_gemm_64x128x256_w4);
    BIND_LDMATRIX(ld_fp8_gemm_64x64x256_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_32x64x256_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_32x128x256_w4_s3);
    BIND_LDMATRIX(ld_fp8_gemm_128x64x128_w4);
    BIND_LDMATRIX(ld_fp8_gemm_128x128x128_w4);
#undef BIND_LDMATRIX


    // SplitK variants — extra `k_split` int + scratch fp32 buffer ptr.
#define BIND_SPLITK(NAME)                                                    \
    m.def("ht_" #NAME,                                                       \
        [](uintptr_t A, uintptr_t B, uintptr_t D,                            \
           int M, int N, int K, int k_split, float alpha,                    \
           uintptr_t scratch, uintptr_t stream) {                            \
            return flash_rt::gemm::smallM_splitk::NAME(                      \
                to_ptr(A), to_ptr(B), to_ptr(D),                             \
                M, N, K, k_split, alpha, to_ptr(scratch),                    \
                to_stream(stream));                                          \
        },                                                                   \
        py::arg("A"), py::arg("B"), py::arg("D"),                            \
        py::arg("M"), py::arg("N"), py::arg("K"),                            \
        py::arg("k_split"), py::arg("alpha"), py::arg("scratch"),            \
        py::arg("stream") = 0)

    BIND_SPLITK(splitk_fp8_gemm_16x64x128_w4);
    BIND_SPLITK(splitk_fp8_gemm_16x64x256_w4);
    BIND_SPLITK(splitk_fp8_gemm_32x64x128_w4);

#undef BIND_SPLITK
#endif  // ENABLE_CUTLASS_SM120_BLOCK_FP8

#ifdef ENABLE_DECODE_GEMV_M1
    // Dedicated M=1 GEMV (FP8 + BF16), warp-per-output-row, no MMA padding tax.
    // Internal decode kernels: callers must pass M=1 and K aligned (FP8 K%16,
    // BF16 K%8). These hold for the fixed Higgs/Qwen decode shapes; the kernels
    // do not validate, so do not expose these to arbitrary external shapes.
#define BIND_GEMV_M1(NAME)                                                   \
    m.def("ht_" #NAME,                                                       \
        [](uintptr_t A, uintptr_t B, uintptr_t D,                            \
           int M, int N, int K, float alpha, uintptr_t stream) {             \
            return flash_rt::gemm::gemv_m1::NAME(                            \
                to_ptr(A), to_ptr(B), to_ptr(D),                             \
                M, N, K, alpha, to_stream(stream));                          \
        },                                                                   \
        py::arg("A"), py::arg("B"), py::arg("D"),                            \
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"),          \
        py::arg("stream") = 0)

    BIND_GEMV_M1(gemv_fp8_m1_w4);
    BIND_GEMV_M1(gemv_fp8_m1_w8);
    BIND_GEMV_M1(gemv_fp8_m1_w16);
    BIND_GEMV_M1(gemv_fp8_m1_resadd_w4);
    BIND_GEMV_M1(gemv_fp8_m1_resadd_w8);

    // Dedicated M=1 BF16 GEMV (all-BF16 decode; no smem A-stage -> full occ).
    BIND_GEMV_M1(gemv_bf16_m1_w4);
    BIND_GEMV_M1(gemv_bf16_m1_w8);
    BIND_GEMV_M1(gemv_bf16_m1_w16);

#undef BIND_GEMV_M1
#endif  // ENABLE_DECODE_GEMV_M1

#ifdef ENABLE_FP8_CONV3D_V17
    m.def("fp8_conv3d_v17_ndhwc_bf16out",
        [](uintptr_t cache_x_fp8, uintptr_t new_x_fp8,
           uintptr_t w_fp8, uintptr_t y_bf16,
           uintptr_t bias_bf16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::fp8_conv3d_v17_ndhwc_bf16out(
                to_ptr(cache_x_fp8), to_ptr(new_x_fp8),
                to_ptr(w_fp8), to_ptr(y_bf16),
                to_ptr(bias_bf16),
                N, T_cache, T_new, H, W, Ci, Co, alpha,
                to_stream(stream));
        },
        py::arg("cache_x_fp8"), py::arg("new_x_fp8"),
        py::arg("w_fp8"), py::arg("y_bf16"),
        py::arg("bias_bf16") = 0,
        py::arg("N"), py::arg("T_cache"), py::arg("T_new"),
        py::arg("H"), py::arg("W"),
        py::arg("Ci"), py::arg("Co"),
        py::arg("alpha") = 1.0f, py::arg("stream") = 0);

    m.def("fp8_conv3d_v17_anyco_ndhwc_bf16out",
        [](uintptr_t cache_x_fp8, uintptr_t new_x_fp8,
           uintptr_t w_fp8, uintptr_t y_bf16,
           uintptr_t bias_bf16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::fp8_conv3d_v17_anyco_ndhwc_bf16out(
                to_ptr(cache_x_fp8), to_ptr(new_x_fp8),
                to_ptr(w_fp8), to_ptr(y_bf16),
                to_ptr(bias_bf16),
                N, T_cache, T_new, H, W, Ci, Co, alpha,
                to_stream(stream));
        },
        py::arg("cache_x_fp8"), py::arg("new_x_fp8"),
        py::arg("w_fp8"), py::arg("y_bf16"),
        py::arg("bias_bf16") = 0,
        py::arg("N"), py::arg("T_cache"), py::arg("T_new"),
        py::arg("H"), py::arg("W"),
        py::arg("Ci"), py::arg("Co"),
        py::arg("alpha") = 1.0f, py::arg("stream") = 0);
#endif
#ifdef ENABLE_FP8_CONV3D_V18
    m.def("fp8_conv3d_v18_ncdhw_res_bf16out",
        [](uintptr_t cache_x_fp8, uintptr_t new_x_fp8,
           uintptr_t w_fp8, uintptr_t y_bf16,
           uintptr_t bias_bf16, uintptr_t residual_bf16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::fp8_conv3d_v18_ncdhw_res_bf16out(
                to_ptr(cache_x_fp8), to_ptr(new_x_fp8),
                to_ptr(w_fp8), to_ptr(y_bf16),
                to_ptr(bias_bf16), to_ptr(residual_bf16),
                N, T_cache, T_new, H, W, Ci, Co, alpha,
                to_stream(stream));
        },
        py::arg("cache_x_fp8"), py::arg("new_x_fp8"),
        py::arg("w_fp8"), py::arg("y_bf16"),
        py::arg("bias_bf16") = 0,
        py::arg("residual_bf16") = 0,
        py::arg("N"), py::arg("T_cache"), py::arg("T_new"),
        py::arg("H"), py::arg("W"),
        py::arg("Ci"), py::arg("Co"),
        py::arg("alpha") = 1.0f, py::arg("stream") = 0);
#endif
#ifdef ENABLE_FP8_CONV2D_3X3_V1
    m.def("fp8_conv2d_3x3_v1_nhwc_bf16out",
        [](uintptr_t x_fp8, uintptr_t w_fp8, uintptr_t y_bf16,
           uintptr_t bias_bf16,
           int N, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::fp8_conv2d_3x3_v1_nhwc_bf16out(
                to_ptr(x_fp8), to_ptr(w_fp8), to_ptr(y_bf16),
                to_ptr(bias_bf16),
                N, H, W, Ci, Co, alpha,
                to_stream(stream));
        },
        py::arg("x_fp8"), py::arg("w_fp8"), py::arg("y_bf16"),
        py::arg("bias_bf16") = 0,
        py::arg("N"), py::arg("H"), py::arg("W"),
        py::arg("Ci"), py::arg("Co"),
        py::arg("alpha") = 1.0f, py::arg("stream") = 0);
    m.def("fp8_conv2d_3x3_v2_nhwc_bf16out",
        [](uintptr_t x_fp8, uintptr_t w_fp8, uintptr_t y_bf16,
           uintptr_t bias_bf16,
           int N, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::fp8_conv2d_3x3_v2_nhwc_bf16out(
                to_ptr(x_fp8), to_ptr(w_fp8), to_ptr(y_bf16),
                to_ptr(bias_bf16),
                N, H, W, Ci, Co, alpha,
                to_stream(stream));
        },
        py::arg("x_fp8"), py::arg("w_fp8"), py::arg("y_bf16"),
        py::arg("bias_bf16") = 0,
        py::arg("N"), py::arg("H"), py::arg("W"),
        py::arg("Ci"), py::arg("Co"),
        py::arg("alpha") = 1.0f, py::arg("stream") = 0);
    m.def("fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out",
        [](uintptr_t x_fp8, uintptr_t w_fp8, uintptr_t y_bf16,
           uintptr_t bias_bf16,
           int B, int T, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out(
                to_ptr(x_fp8), to_ptr(w_fp8), to_ptr(y_bf16),
                to_ptr(bias_bf16),
                B, T, H, W, Ci, Co, alpha,
                to_stream(stream));
        },
        py::arg("x_fp8"), py::arg("w_fp8"), py::arg("y_bf16"),
        py::arg("bias_bf16") = 0,
        py::arg("B"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("Ci"), py::arg("Co"),
        py::arg("alpha") = 1.0f, py::arg("stream") = 0);
#endif
#ifdef ENABLE_CUDNN_FP8_CONV2D_3X3
    m.def("cudnn_fp8_conv2d_3x3_nhwc_bf16out",
        [](uintptr_t x_fp8, uintptr_t w_fp8, uintptr_t y_bf16,
           uintptr_t bias_bf16,
           int N, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::cudnn_fp8_conv2d_3x3_nhwc_bf16out(
                to_ptr(x_fp8), to_ptr(w_fp8), to_ptr(y_bf16),
                to_ptr(bias_bf16),
                N, H, W, Ci, Co, alpha,
                to_stream(stream));
        },
        py::arg("x_fp8"), py::arg("w_fp8"), py::arg("y_bf16"),
        py::arg("bias_bf16") = 0,
        py::arg("N"), py::arg("H"), py::arg("W"),
        py::arg("Ci"), py::arg("Co"),
        py::arg("alpha") = 1.0f, py::arg("stream") = 0);
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

    m.def("fp4_w4a16_gemm_sm120_bf16out_pingpong",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           int M, int N, int K,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_sm120_bf16out_pingpong(
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

    // Recipe C step 1: NVFP4 W4A16 GEMM with fused per-col bias + GELU(tanh)
    // epilogue, BF16 output. Replaces (cutlass GEMM_up + bias_gelu_inplace)
    // 2-launch chain in motus Wan FFN forward. Schedule:
    // KernelTmaWarpSpecializedPingpong + PersistentScheduler (sweep-winner
    // at M=360 K=3072 N=14336; pingpong vs cooperative ~+0.6 µs/call).
    m.def("fp4_w4a16_gemm_bias_gelu_bf16out_sm120",
        [](uintptr_t A_packed, uintptr_t B_packed,
           uintptr_t SFA, uintptr_t SFB,
           uintptr_t bias, uintptr_t D,
           int M, int N, int K,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_bias_gelu_bf16out_sm120(
                to_ptr(A_packed), to_ptr(B_packed),
                to_ptr(SFA), to_ptr(SFB),
                to_ptr(bias), to_ptr(D),
                M, N, K,
                alpha,
                to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("bias"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha") = 1.0f,
        py::arg("stream") = 0);

    // Recipe C step 2: NVFP4 W4A16 GEMM with fused per-col bias + GELU(tanh)
    // + per-block-16 NVFP4 quant epilogue. Replaces the 3-launch chain
    // (GEMM_up + bias_gelu_inplace + quantize_bf16_to_nvfp4_swizzled) with
    // one cutlass-fork kernel that outputs FP4 packed + cutlass-swizzled
    // UE4M3 SF directly. Used as the GEMM_up half of the motus Wan FFN
    // Task A+B chain.
    m.def("fp4_w4a16_gemm_bias_gelu_fp4out_sm120",
        [](uintptr_t A_packed, uintptr_t B_packed,
           uintptr_t SFA, uintptr_t SFB,
           uintptr_t bias, uintptr_t D_packed, uintptr_t SFD,
           int M, int N, int K,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
                to_ptr(A_packed), to_ptr(B_packed),
                to_ptr(SFA), to_ptr(SFB),
                to_ptr(bias),
                to_ptr(D_packed), to_ptr(SFD),
                M, N, K,
                alpha,
                to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("bias"), py::arg("D_packed"), py::arg("SFD"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha") = 1.0f,
        py::arg("stream") = 0);

    // Recipe C step 3: NVFP4 W4A16 GEMM_dn + per-col bias epilogue, BF16
    // output, **StreamK scheduler**. Replaces (GEMM_dn + add_bias_bf16)
    // chain. StreamK recovers SM utilization at motus Wan FFN GEMM_dn
    // shape (72 CTAs / 170 SMs = 0.42 wave → ~1.7 wave) for 1.277× over
    // the default PersistentScheduler.
    m.def("fp4_w4a16_gemm_dn_streamk_bias_bf16out_sm120",
        [](uintptr_t A_packed, uintptr_t B_packed,
           uintptr_t SFA, uintptr_t SFB,
           uintptr_t bias, uintptr_t D,
           int M, int N, int K,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_dn_streamk_bias_bf16out_sm120(
                to_ptr(A_packed), to_ptr(B_packed),
                to_ptr(SFA), to_ptr(SFB),
                to_ptr(bias), to_ptr(D),
                M, N, K,
                alpha,
                to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("bias"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha") = 1.0f,
        py::arg("stream") = 0);

    m.def("fp4_w4a16_gemm_dn_streamk_bf16out_sm120",
        [](uintptr_t A_packed, uintptr_t B_packed,
           uintptr_t SFA, uintptr_t SFB,
           uintptr_t D,
           int M, int N, int K,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_dn_streamk_bf16out_sm120(
                to_ptr(A_packed), to_ptr(B_packed),
                to_ptr(SFA), to_ptr(SFB),
                to_ptr(D),
                M, N, K,
                alpha,
                to_stream(stream));
        },
        py::arg("A_packed"), py::arg("B_packed"),
        py::arg("SFA"), py::arg("SFB"),
        py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
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

    // ── NVFP4 W4A4 M=1 matvec (custom, decode hot path) ──
    // Hand-rolled SM120 kernel specialized for M=1 LLM decode where
    // CUTLASS NVFP4 GEMM tiles assume M ≥ 16 and run at ~30% of HBM
    // BW. This kernel targets ~70%+ HBM BW utilization (~2× decode
    // speedup). K must be in {4096, 12288}; N must be a multiple of
    // 32. SF layouts identical to the existing fp4_w4a16 GEMM.
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
        py::arg("stream") = 0,
        R"pbdoc(
NVFP4 W4A4 M=1 matvec (custom SM120 kernel).

Drop-in for ``fp4_w4a16_gemm_sm120_bf16out`` at M=1, ~2× faster on
the LLM decode hot path. Supported K: {4096, 12288}; N must be a
multiple of 32. Returns 0 on success, nonzero argument-error code.
)pbdoc");

    m.def("fp4_w4a4_matvec_sm120_init",
        []() { flash_rt::gemm::fp4_w4a4_matvec_init_luts(); },
        "Idempotent UE4M3 LUT initialization for the matvec kernel.");

    // ── P2-S1: tensor-core NVFP4 W4A4 single-tile MMA ──
    // Uses cute SM120_16x8x64_TN_VS<e2m1, e2m1, float, ue4m3, VS=16>
    // atom (PTX mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec
    // ::4X.m16n8k64). Single-tile only (M_pad=16, N=8, K=64) — gate
    // for cos = 1.000 vs CUTLASS reference. Subsequent P2-Sx steps add
    // K accumulation, N-tile parallelism, cp.async pipelining, etc.
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
        py::arg("stream") = 0,
        R"pbdoc(
NVFP4 W4A4 single-tile tensor-core MMA (SM120, P2-S1).

Single (M=16 padded from M=1, N=8, K=64) tile. Inputs in linear
row/col-major byte form (caller responsibility). Used by gate-S1 to
verify the MMA atom invocation matches CUTLASS at iso shape.
)pbdoc");

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
        py::arg("alpha") = 1.0f,
        py::arg("K"),
        py::arg("stream") = 0,
        R"pbdoc(
NVFP4 W4A4 multi-K tensor-core MMA (SM120, P2-S2).

Single warp, single (N=8) col-tile, full K-loop in K_TILE=64 chunks
with f32 fragment accumulation across tiles. K must be a multiple
of 64. Used by gate-S2 to verify cos = 1.000 vs reference at the
production shapes K=4096 / K=12288.
)pbdoc");

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
        py::arg("stream") = 0,
        R"pbdoc(
NVFP4 W4A4 full-N tensor-core MMA (SM120, P2-S3).

Block: 4 warps × 8 N-cols/warp = 32 N-cols/block.
gridDim.x = ceil(N / 32). A and SFA shared across warps; B and SFB
per-warp. Drop-in replacement signature for fp4_w4a4_matvec_sm120
(R2 SIMT) — same args, ~targeting 2× speedup once perf gates close.
N must be a multiple of 32; K must be a multiple of 64.
)pbdoc");

    m.def("fp4_w4a4_mma_sm120_warpsplit_bf16out",
        [](uintptr_t A, uintptr_t B, uintptr_t D, int N, int K, uintptr_t SFA,
           uintptr_t SFB, float alpha, int warps, int stages,
           uintptr_t stream) -> int {
            return flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_bf16out(
                to_ptr(A), to_ptr(B), to_ptr(D), N, K, to_ptr(SFA),
                to_ptr(SFB), alpha, warps, stages, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"), py::arg("N"), py::arg("K"),
        py::arg("SFA"), py::arg("SFB"), py::arg("alpha") = 1.0f,
        py::arg("warps") = 4, py::arg("stages") = 4, py::arg("stream") = 0,
        R"pbdoc(
NVFP4 W4A4 warp-split-K M=1 GEMV (SM120) for long-K / small-N decode shapes
(mlp_down K=17408, out/o_proj K=6144) the single-warp full_n kernel underfills.
8 N-cols/block, `warps` warps each streaming K/warps, partials summed in shared
memory (intra-block) -> CUDA-graph-replay safe; fills the SMs. cos=1.000 vs
full_n. N%8==0; K%64==0; (K/64)%warps==0; warps in {2,4,8}; stages in {3,4,6}.
)pbdoc");

    m.def("fp4_w4a4_mma_sm120_warpsplit_mrows_bf16out",
        [](uintptr_t A, uintptr_t B, uintptr_t D, int M, int N, int K,
           uintptr_t SFA, uintptr_t SFB, float alpha, int warps, int stages,
           uintptr_t stream) -> int {
            return flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_mrows_bf16out(
                to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_ptr(SFA),
                to_ptr(SFB), alpha, warps, stages, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"), py::arg("M"), py::arg("N"),
        py::arg("K"), py::arg("SFA"), py::arg("SFB"), py::arg("alpha") = 1.0f,
        py::arg("warps") = 4, py::arg("stages") = 4, py::arg("stream") = 0,
        R"pbdoc(
NVFP4 W4A4 small-M (M<=16) warp-split-K GEMM (SM120) for the spec verify-path
long-K shapes (mlp_down K=17408 at M=K_draft+1). The 16x8x64 MMA atom computes
a full 16-row tile, so M<=16 rows cost the same weight HBM as M=1; combined with
warp-split-K (K split across `warps`, partials summed in shared memory ->
graph-replay safe) to fill the SMs on long K. M in 1..16; N%8==0; K%64==0;
(K/64)%warps==0; warps in {2,4,8}; stages in {3,4,6}.
)pbdoc");

#endif

// ─────────────────────────────────────────────────────────────────────
// SM100 NVFP4 W4A16 GEMM bindings (Thor SM110).
//
// Mutually exclusive with ENABLE_CUTLASS_SM120_NVFP4_W4A16; the CMake
// gates ensure only one of the two object libraries is linked into
// flash_rt_kernels for a given GPU_ARCH. The public Python names are
// kept identical to the SM120 surface ("fp4_w4a16_gemm_sm120_bf16out*")
// because that is the frozen ABI the Qwen3.6 frontend already calls;
// the underlying implementation is dispatched by build arch.
//
// On Thor only the three core W4A16 entries are needed by the Qwen
// frontend (default / widen / pingpong). The Motus-specific
// bias_gelu / streamk fp4-out variants are SM120-exclusive and stay
// out of the Thor surface.
// ─────────────────────────────────────────────────────────────────────
#ifdef ENABLE_CUTLASS_SM100_NVFP4_W4A16
    m.def("fp4_w4a16_gemm_sm120_bf16out",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           int M, int N, int K,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_sm100_bf16out(
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

    m.def("fp4_w4a16_gemm_sm120_bf16out_widen",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           int M, int N, int K,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_sm100_bf16out_widen(
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

    m.def("fp4_w4a16_gemm_sm120_bf16out_pingpong",
        [](uintptr_t A_packed, uintptr_t B_packed, uintptr_t D,
           int M, int N, int K,
           uintptr_t SFA, uintptr_t SFB,
           float alpha,
           uintptr_t stream) {
            flash_rt::gemm::fp4_w4a16_gemm_sm100_bf16out_pingpong(
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

    // SF reshape (memory permutation only, arch-agnostic). Same Python
    // names as the SM120 surface so the weight loader works unchanged.
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
#endif

#ifdef ENABLE_ACTION_FFN_MEGAKERNEL_V6T
    // Action FFN megakernel V6tuned (ku256_sd4_su3 tile). Fused FP8
    // W4A8 GEMM_up + bias + GELU + intermediate FP8 quant + GEMM_dn +
    // bias + gate * acc + residual_add for the Pi0.5 action expert.
    // Shape lock: M<=32, K_up=1024, N_up=4096, K_dn=4096, N_dn=1024.
    m.def("action_ffn_v6t_launch_sm120",
        [](uintptr_t x_fp8_in,
           uintptr_t up_w_NK, uintptr_t up_bias,
           uintptr_t dn_inv_s, uintptr_t dn_w_NK, uintptr_t dn_bias,
           uintptr_t gate, uintptr_t residual,
           uintptr_t y_out,
           uintptr_t up_fp8_scr,
           int M, int K_up, int N_up, int K_dn, int N_dn,
           float up_alpha, float dn_alpha, float dn_act_scale,
           uintptr_t stream) {
            return flash_rt::megakernel::action_ffn_v6t_launch_sm120(
                to_ptr(x_fp8_in),
                to_ptr(up_w_NK), to_ptr(up_bias),
                to_ptr(dn_inv_s), to_ptr(dn_w_NK), to_ptr(dn_bias),
                to_ptr(gate), to_ptr(residual),
                to_ptr(y_out),
                to_ptr(up_fp8_scr),
                M, K_up, N_up, K_dn, N_dn,
                up_alpha, dn_alpha, dn_act_scale,
                to_stream(stream));
        });
#endif

#ifdef ENABLE_UND_FFN_MEGAKERNEL_V5T
    // Und FFN megakernel V5tuned. Fused FP8 W4A8 norm + GEMM_up + GELU +
    // FP8 quant + GEMM_dn + bias + residual_add for the Pi0.5 understanding
    // module. Shape: M ≤ 144, K_up=512, N_up=2048, K_dn=2048, N_dn=512.
    m.def("und_ffn_v5t_launch_sm120",
        [](uintptr_t x_in, uintptr_t up_inv_s,
           uintptr_t up_w_NK, uintptr_t up_bias,
           uintptr_t dn_inv_s, uintptr_t dn_w_NK, uintptr_t dn_bias,
           uintptr_t residual_in,
           uintptr_t y_out,
           uintptr_t x_fp8_scr, uintptr_t up_fp8_scr,
           int M, int K_up, int N_up, int K_dn, int N_dn,
           float up_alpha, float dn_alpha,
           float up_act_scale, float dn_act_scale,
           uintptr_t barrier_state, uintptr_t stream) {
            return flash_rt::megakernel::und_ffn_v5t_launch_sm120(
                to_ptr(x_in), to_ptr(up_inv_s),
                to_ptr(up_w_NK), to_ptr(up_bias),
                to_ptr(dn_inv_s), to_ptr(dn_w_NK), to_ptr(dn_bias),
                to_ptr(residual_in),
                to_ptr(y_out),
                to_ptr(x_fp8_scr), to_ptr(up_fp8_scr),
                M, K_up, N_up, K_dn, N_dn,
                up_alpha, dn_alpha, up_act_scale, dn_act_scale,
                to_ptr(barrier_state), to_stream(stream));
        });

    // Stage3 und FFN split megakernel. Same math as V5t, split into
    // up/intermediate and down/residual launches for M=188.
    m.def("und_ffn_v5split_stage3_launch_sm120",
        [](uintptr_t x_in, uintptr_t up_inv_s,
           uintptr_t up_w_NK, uintptr_t up_bias,
           uintptr_t dn_inv_s, uintptr_t dn_w_NK, uintptr_t dn_bias,
           uintptr_t residual_in,
           uintptr_t y_out,
           uintptr_t x_fp8_scr, uintptr_t up_fp8_scr,
           int M, int K_up, int N_up, int K_dn, int N_dn,
           float up_alpha, float dn_alpha,
           float up_act_scale, float dn_act_scale,
           uintptr_t barrier_state, uintptr_t stream) {
            return flash_rt::megakernel::und_ffn_v5split_stage3_launch_sm120(
                to_ptr(x_in), to_ptr(up_inv_s),
                to_ptr(up_w_NK), to_ptr(up_bias),
                to_ptr(dn_inv_s), to_ptr(dn_w_NK), to_ptr(dn_bias),
                to_ptr(residual_in),
                to_ptr(y_out),
                to_ptr(x_fp8_scr), to_ptr(up_fp8_scr),
                M, K_up, N_up, K_dn, N_dn,
                up_alpha, dn_alpha, up_act_scale, dn_act_scale,
                to_ptr(barrier_state), to_stream(stream));
        });
#endif

#ifdef ENABLE_TINYFP8_KERNELS
    // tiny_fp8: 5 small-shape 2-stage FP8 GEMM variants for the motus
    // action-expert / und-module sites. D = alpha * (A_fp8 @ B_fp8^T) -> bf16
    // with B stored in (N, K) row-major (pre-transposed at install time).
    auto bind_tiny = [&m](const char* name, auto fnptr) {
        m.def(name,
            [fnptr](uintptr_t A, uintptr_t B, uintptr_t D,
                    int M, int N, int K, float alpha, uintptr_t stream) {
                return fnptr(to_ptr(A), to_ptr(B), to_ptr(D),
                              M, N, K, alpha, to_stream(stream));
            });
    };
    bind_tiny("tinyfp8_gemm_M8_N32_K128_sm120",
              flash_rt::megakernel::tinyfp8_gemm_M8_N32_K128_sm120);
    bind_tiny("tinyfp8_gemm_M8_N32_K256_sm120",
              flash_rt::megakernel::tinyfp8_gemm_M8_N32_K256_sm120);
    bind_tiny("tinyfp8_gemm_M8_N32_K512_sm120",
              flash_rt::megakernel::tinyfp8_gemm_M8_N32_K512_sm120);
    bind_tiny("tinyfp8_gemm_M16_N32_K64_sm120",
              flash_rt::megakernel::tinyfp8_gemm_M16_N32_K64_sm120);
    bind_tiny("tinyfp8_gemm_M16_N64_K64_sm120",
              flash_rt::megakernel::tinyfp8_gemm_M16_N64_K64_sm120);
    bind_tiny("tinyfp8_gemm_M32_N32_K128_sm120",
              flash_rt::megakernel::tinyfp8_gemm_M32_N32_K128_sm120);
    bind_tiny("tinyfp8_gemm_M32_N32_K512_sm120",
              flash_rt::megakernel::tinyfp8_gemm_M32_N32_K512_sm120);
    bind_tiny("tinyfp8_gemm3_M16_N64_K128_sm120",
              flash_rt::megakernel::tinyfp8_gemm3_M16_N64_K128_sm120);
#endif

    // ── P3A-S2 (F1-lite): fused qkv post-processing ──
    // Replaces (q_norm + RoPE + Q_buf copy) with one launch and
    // (k_norm + RoPE + K_cache write + V_cache write) with another.
    // head_dim hardcoded at 128 (Qwen3-8B); S=1 decode hot path only.
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

    m.def("silu_mul_merged_to_nvfp4_swizzled_bf16",
        [](uintptr_t merged_gate_up,
           uintptr_t packed, uintptr_t sf_swz,
           int rows, int cols, uintptr_t stream) -> int {
            return flash_rt::kernels::silu_mul_merged_to_nvfp4_swizzled_bf16(
                to_ptr(merged_gate_up),
                to_ptr(packed), to_ptr(sf_swz),
                rows, cols, to_stream(stream));
        },
        py::arg("merged_gate_up"),
        py::arg("packed"), py::arg("sf_swz"),
        py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("silu_mul_merged_to_nvfp4_swizzled_grouped_bf16",
        [](uintptr_t merged_gate_up,
           uintptr_t packed, uintptr_t sf_swz,
           int rows, int cols, uintptr_t stream) -> int {
            return flash_rt::kernels::silu_mul_merged_to_nvfp4_swizzled_grouped_bf16(
                to_ptr(merged_gate_up),
                to_ptr(packed), to_ptr(sf_swz),
                rows, cols, to_stream(stream));
        },
        py::arg("merged_gate_up"),
        py::arg("packed"), py::arg("sf_swz"),
        py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("silu_mul_merged_to_nvfp4_swizzled_grouped32_bf16",
        [](uintptr_t merged_gate_up,
           uintptr_t packed, uintptr_t sf_swz,
           int rows, int cols, uintptr_t stream) -> int {
            return flash_rt::kernels::silu_mul_merged_to_nvfp4_swizzled_grouped32_bf16(
                to_ptr(merged_gate_up),
                to_ptr(packed), to_ptr(sf_swz),
                rows, cols, to_stream(stream));
        },
        py::arg("merged_gate_up"),
        py::arg("packed"), py::arg("sf_swz"),
        py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

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

    m.def("qwen3_k_norm_rope_kvwrite_devpos_bf16",
        [](uintptr_t k_pre, uintptr_t v_pre, uintptr_t k_norm_w,
           uintptr_t cos, uintptr_t sin,
           uintptr_t k_cache_base, uintptr_t v_cache_base,
           uintptr_t cur_pos, int row_elems,
           int n_kv_heads, float eps, uintptr_t stream) -> int {
            return flash_rt::kernels::qwen3_k_norm_rope_kvwrite_devpos_bf16(
                to_ptr(k_pre), to_ptr(v_pre), to_ptr(k_norm_w),
                to_ptr(cos), to_ptr(sin),
                to_ptr(k_cache_base), to_ptr(v_cache_base),
                to_ptr(cur_pos), row_elems,
                n_kv_heads, eps, to_stream(stream));
        },
        py::arg("k_pre"), py::arg("v_pre"), py::arg("k_norm_w"),
        py::arg("cos"), py::arg("sin"),
        py::arg("k_cache_base"), py::arg("v_cache_base"),
        py::arg("cur_pos"), py::arg("row_elems"),
        py::arg("n_kv_heads"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    m.def("ada_rms_norm_style_int8", [](uintptr_t x, uintptr_t weight, uintptr_t style,
                                         uintptr_t out, uintptr_t gate_out,
                                         int seq_len, int dim, float eps,
                                         uintptr_t d_scales, uintptr_t stream) {
        ada_rms_norm_style_int8(
            typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
            typed_ptr<__nv_bfloat16>(style), typed_ptr<int8_t>(out),
            typed_ptr<__nv_bfloat16>(gate_out), seq_len, dim, eps,
            reinterpret_cast<float*>(d_scales), to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("style"),
       py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scales") = 0, py::arg("stream") = 0);

    m.def("avg_pool_vision_tokens", [](uintptr_t x, uintptr_t out,
                                        int nv, int H, int W, int dim,
                                        int pool_factor, uintptr_t stream) {
        avg_pool_vision_tokens(
            typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(out),
            nv, H, W, dim, pool_factor, to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("nv"), py::arg("H"), py::arg("W"),
       py::arg("dim"), py::arg("pool_factor"), py::arg("stream") = 0);

    m.def("rms_norm_int8_rowwise", [](uintptr_t x, uintptr_t weight,
                                       uintptr_t out, uintptr_t scales,
                                       int seq_len, int dim, float eps,
                                       uintptr_t stream) {
        rms_norm_int8_rowwise(
            typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
            typed_ptr<int8_t>(out), reinterpret_cast<float*>(scales),
            seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"), py::arg("scales"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("stream") = 0);

    m.def("residual_add_rms_norm_int8_rowwise", [](uintptr_t residual, uintptr_t x,
                                                    uintptr_t weight,
                                                    uintptr_t out, uintptr_t scales,
                                                    int seq_len, int dim, float eps,
                                                    uintptr_t stream) {
        residual_add_rms_norm_int8_rowwise(
            typed_ptr<__nv_bfloat16>(residual), typed_ptr<__nv_bfloat16>(x),
            typed_ptr<__nv_bfloat16>(weight),
            typed_ptr<int8_t>(out), reinterpret_cast<float*>(scales),
            seq_len, dim, eps, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("weight"),
       py::arg("out"), py::arg("scales"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("stream") = 0);

    m.def("bias_residual_layer_norm_bf16", [](uintptr_t residual, uintptr_t x,
                                                uintptr_t bias_pre,
                                                uintptr_t ln_weight, uintptr_t ln_bias,
                                                uintptr_t out,
                                                int seq_len, int dim, float eps,
                                                uintptr_t stream) {
        bias_residual_layer_norm_bf16(
            typed_ptr<__nv_bfloat16>(residual), typed_ptr<__nv_bfloat16>(x),
            typed_ptr<__nv_bfloat16>(bias_pre),
            typed_ptr<__nv_bfloat16>(ln_weight),
            typed_ptr<__nv_bfloat16>(ln_bias),
            typed_ptr<__nv_bfloat16>(out), seq_len, dim, eps,
            to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias_pre"),
       py::arg("ln_weight"), py::arg("ln_bias"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("stream") = 0);

    m.def("bias_gelu_bf16", [](uintptr_t x, uintptr_t bias,
                                int seq_len, int dim, uintptr_t stream) {
        bias_gelu_inplace_bf16(typed_ptr<__nv_bfloat16>(x),
                               typed_ptr<__nv_bfloat16>(bias),
                               seq_len, dim, to_stream(stream));
    }, py::arg("x"), py::arg("bias"), py::arg("seq_len"), py::arg("dim"),
       py::arg("stream") = 0);

    m.def("bias_gelu_bf16_strict", [](uintptr_t x, uintptr_t bias,
                                       int seq_len, int dim, uintptr_t stream) {
        bias_gelu_inplace_bf16_strict(typed_ptr<__nv_bfloat16>(x),
                                      typed_ptr<__nv_bfloat16>(bias),
                                      seq_len, dim, to_stream(stream));
    }, py::arg("x"), py::arg("bias"), py::arg("seq_len"), py::arg("dim"),
       py::arg("stream") = 0);

    m.def("gate_residual_ada_norm_int8", [](uintptr_t residual, uintptr_t x,
                                             uintptr_t gate, uintptr_t weight,
                                             uintptr_t style,
                                             uintptr_t out, uintptr_t gate_out,
                                             int seq_len, int dim, float eps,
                                             uintptr_t d_scales, uintptr_t stream) {
        gate_residual_ada_norm_int8(
            typed_ptr<__nv_bfloat16>(residual), typed_ptr<__nv_bfloat16>(x),
            typed_ptr<__nv_bfloat16>(gate), typed_ptr<__nv_bfloat16>(weight),
            typed_ptr<__nv_bfloat16>(style), typed_ptr<int8_t>(out),
            typed_ptr<__nv_bfloat16>(gate_out), seq_len, dim, eps,
            reinterpret_cast<float*>(d_scales), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("weight"),
       py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scales") = 0, py::arg("stream") = 0);

    m.def("quantize_int8_static", [](uintptr_t input, uintptr_t output,
                                      uintptr_t scale, int n, uintptr_t stream) {
        quantize_int8_static(typed_ptr<__nv_bfloat16>(input), typed_ptr<int8_t>(output),
                             reinterpret_cast<const float*>(scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("quantize_int8_device", [](uintptr_t input, uintptr_t output,
                                      uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_int8_device(typed_ptr<__nv_bfloat16>(input), typed_ptr<int8_t>(output),
                             reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("quantize_int8_rowwise", [](uintptr_t input, uintptr_t output,
                                       uintptr_t d_scales, int rows, int cols,
                                       uintptr_t stream) {
        quantize_int8_rowwise(typed_ptr<__nv_bfloat16>(input), typed_ptr<int8_t>(output),
                              reinterpret_cast<float*>(d_scales), rows, cols, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scales"), py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("quantize_int8_rowwise_static", [](uintptr_t input, uintptr_t output,
                                              uintptr_t d_scales, int rows, int cols,
                                              uintptr_t stream) {
        quantize_int8_rowwise_static(typed_ptr<__nv_bfloat16>(input), typed_ptr<int8_t>(output),
                                     reinterpret_cast<const float*>(d_scales), rows, cols, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scales"), py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("dequant_int32_to_bf16", [](uintptr_t input, uintptr_t output,
                                       uintptr_t d_act_scale, uintptr_t d_weight_scale,
                                       int n, uintptr_t stream) {
        dequant_int32_to_bf16(typed_ptr<int32_t>(input), typed_ptr<__nv_bfloat16>(output),
                              reinterpret_cast<const float*>(d_act_scale),
                              reinterpret_cast<const float*>(d_weight_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_act_scale"), py::arg("d_weight_scale"),
       py::arg("n"), py::arg("stream") = 0);

    m.def("cutlass_int8_silu_gated_bf16out",
          [](uintptr_t act, uintptr_t up_w, uintptr_t act_s, uintptr_t wt_s,
             uintptr_t gate, uintptr_t D, int M, int N, int K, uintptr_t stream) {
#ifdef ENABLE_SM80_INT8_CUTLASS
              return cutlass_int8_silu_gated_bf16out(to_ptr(act), to_ptr(up_w), to_ptr(act_s),
                  to_ptr(wt_s), to_ptr(gate), to_ptr(D), M, N, K, to_stream(stream));
#else
              throw std::runtime_error("cutlass_int8_silu_gated_bf16out was not built");
#endif
          }, py::arg("act"), py::arg("up_w"), py::arg("act_scale"), py::arg("wt_scale"),
             py::arg("gate_buf"), py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def("cutlass_int8_rowwise_bf16out",
          [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
             uintptr_t D, int M, int N, int K, uintptr_t stream) {
#ifdef ENABLE_SM80_INT8_CUTLASS
              return cutlass_int8_rowwise_bf16out(to_ptr(A), to_ptr(B), to_ptr(act_scale),
                  to_ptr(weight_scale), to_ptr(D), M, N, K, to_stream(stream));
#else
              throw std::runtime_error("cutlass_int8_rowwise_bf16out was not built");
#endif
          }, py::arg("A"), py::arg("B"), py::arg("act_scale"), py::arg("weight_scale"),
          py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def("cutlass_int8_rowwise_bf16out_t64x128",
          [](uintptr_t A, uintptr_t B, uintptr_t act_scale, uintptr_t weight_scale,
             uintptr_t D, int M, int N, int K, uintptr_t stream) {
#ifdef ENABLE_SM80_INT8_CUTLASS
              return cutlass_int8_rowwise_bf16out_t64x128(to_ptr(A), to_ptr(B), to_ptr(act_scale),
                  to_ptr(weight_scale), to_ptr(D), M, N, K, to_stream(stream));
#else
              throw std::runtime_error("cutlass_int8_rowwise_bf16out_t64x128 was not built");
#endif
          }, py::arg("A"), py::arg("B"), py::arg("act_scale"), py::arg("weight_scale"),
          py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);


#ifdef ENABLE_MOTUS
    m.def("motus_fp4_conv3d_v19sf_ndhwc_bf16out",
        [](uintptr_t cache_x_fp4, uintptr_t new_x_fp4, uintptr_t w_fp4,
           uintptr_t cache_sfa, uintptr_t new_sfa, uintptr_t w_sfb,
           uintptr_t y_bf16, uintptr_t bias_bf16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::motus_fp4_conv3d_v19sf_ndhwc_bf16out(
                to_ptr(cache_x_fp4), to_ptr(new_x_fp4), to_ptr(w_fp4),
                to_ptr(cache_sfa), to_ptr(new_sfa), to_ptr(w_sfb),
                to_ptr(y_bf16), bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                N, T_cache, T_new, H, W, Ci, Co, alpha, to_stream(stream));
        });
    m.def("motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2",
        [](uintptr_t cache_x_fp4, uintptr_t new_x_fp4, uintptr_t w_fp4,
           uintptr_t cache_sfa, uintptr_t new_sfa, uintptr_t w_sfb,
           uintptr_t outer_w_fp32, uintptr_t y_bf16, uintptr_t bias_bf16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2(
                to_ptr(cache_x_fp4), to_ptr(new_x_fp4), to_ptr(w_fp4),
                to_ptr(cache_sfa), to_ptr(new_sfa), to_ptr(w_sfb),
                to_ptr(outer_w_fp32), to_ptr(y_bf16),
                bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                N, T_cache, T_new, H, W, Ci, Co, alpha, to_stream(stream));
        });
    m.def("motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out",
        [](uintptr_t cache_x_fp4, uintptr_t new_x_fp4, uintptr_t w_fp4,
           uintptr_t cache_sfa, uintptr_t new_sfa, uintptr_t w_sfb,
           uintptr_t y_bf16, uintptr_t bias_bf16, uintptr_t residual_bf16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out(
                to_ptr(cache_x_fp4), to_ptr(new_x_fp4), to_ptr(w_fp4),
                to_ptr(cache_sfa), to_ptr(new_sfa), to_ptr(w_sfb),
                to_ptr(y_bf16), bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                residual_bf16 ? to_ptr(residual_bf16) : nullptr,
                N, T_cache, T_new, H, W, Ci, Co, alpha, to_stream(stream));
        });
    m.def("motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out_v2",
        [](uintptr_t cache_x_fp4, uintptr_t new_x_fp4, uintptr_t w_fp4,
           uintptr_t cache_sfa, uintptr_t new_sfa, uintptr_t w_sfb,
           uintptr_t outer_w_fp32, uintptr_t y_bf16,
           uintptr_t bias_bf16, uintptr_t residual_bf16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out_v2(
                to_ptr(cache_x_fp4), to_ptr(new_x_fp4), to_ptr(w_fp4),
                to_ptr(cache_sfa), to_ptr(new_sfa), to_ptr(w_sfb),
                to_ptr(outer_w_fp32), to_ptr(y_bf16),
                bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                residual_bf16 ? to_ptr(residual_bf16) : nullptr,
                N, T_cache, T_new, H, W, Ci, Co, alpha, to_stream(stream));
        });
    m.def("motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out",
        [](uintptr_t cache_x_fp4, uintptr_t new_x_fp4, uintptr_t w_fp4,
           uintptr_t cache_sfa, uintptr_t new_sfa, uintptr_t w_sfb,
           uintptr_t y_bf16, uintptr_t bias_bf16, uintptr_t residual_bf16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return ::motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out(
                to_ptr(cache_x_fp4), to_ptr(new_x_fp4), to_ptr(w_fp4),
                to_ptr(cache_sfa), to_ptr(new_sfa), to_ptr(w_sfb),
                to_ptr(y_bf16), bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                residual_bf16 ? to_ptr(residual_bf16) : nullptr,
                N, T_cache, T_new, H, W, Ci, Co, alpha, to_stream(stream));
        });
    m.def("motus_bf16_rms_silu_quant_nvfp4_to_ndhwc_v1",
        [](uintptr_t x_bf16, uintptr_t gamma_bf16, uintptr_t awq_inv_scale_fp32,
           uintptr_t y_fp4, uintptr_t y_sf,
           int B, int C, int T, int H, int W, float eps, uintptr_t stream) {
            return ::motus_bf16_rms_silu_quant_nvfp4_to_ndhwc_v1(
                to_ptr(x_bf16), to_ptr(gamma_bf16),
                awq_inv_scale_fp32 ? to_ptr(awq_inv_scale_fp32) : nullptr,
                to_ptr(y_fp4), to_ptr(y_sf),
                B, C, T, H, W, eps, to_stream(stream));
        });
#endif

#ifdef ENABLE_QWEN36_FLASHINFER_XQA
    m.def("qwen36_flashinfer_xqa_bf16_fp8kv_spec",
        [](uintptr_t q, uintptr_t k_cache, uintptr_t v_cache,
           uintptr_t page_table, uintptr_t seq_lens, uintptr_t mask,
           uintptr_t out, uintptr_t semaphores, uintptr_t scratch,
           int max_seq_len, int q_seq_len, int sm_count,
           float q_scale, float kv_scale, bool enable_pdl,
           int64_t k_stride_page, int64_t k_stride_token,
           int64_t k_stride_head, uintptr_t stream) {
            qwen36_flashinfer_xqa_bf16_fp8kv_spec(
                to_ptr(q), to_ptr(k_cache), to_ptr(v_cache),
                reinterpret_cast<const int32_t*>(to_ptr(page_table)),
                reinterpret_cast<const uint32_t*>(to_ptr(seq_lens)),
                reinterpret_cast<const uint32_t*>(to_ptr(mask)),
                to_ptr(out),
                reinterpret_cast<uint32_t*>(to_ptr(semaphores)),
                to_ptr(scratch),
                max_seq_len, q_seq_len, sm_count, q_scale, kv_scale,
                enable_pdl, k_stride_page, k_stride_token, k_stride_head,
                to_stream(stream));
        },
        py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
        py::arg("page_table"), py::arg("seq_lens"), py::arg("mask"),
        py::arg("out"), py::arg("semaphores"), py::arg("scratch"),
        py::arg("max_seq_len"), py::arg("q_seq_len"), py::arg("sm_count"),
        py::arg("q_scale"), py::arg("kv_scale"), py::arg("enable_pdl"),
        py::arg("k_stride_page"), py::arg("k_stride_token"),
        py::arg("k_stride_head"), py::arg("stream") = 0);
#endif

#ifdef ENABLE_LINGBOT
#include "kernels/lingbot_bindings.inc"   // m.def("lingbot_...", &lingbot_...) for the LingBot-VLA model
#endif
}
