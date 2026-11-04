// ================================================================
// FlashRT — LingBot/Wan kernel bindings (opt-in SM89 module)
//
// These entries were moved out of flash_rt_kernels so model-specific
// kernels do not compile into the shared module on every architecture.
// The module is built only when GPU_ARCH=89; see docs/lingbot_kernels.md.
// ================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "lingbot_va/norm.cuh"
#include "kernels/wan_adaln.cuh"
#include "kernels/flow_euler_window.cuh"
#include "kernels/xattn_probs.cuh"
#include "lingbot_va/fp8_block128_to_nvfp4_swizzled.cuh"
#include "quantize/nvfp4_pack_tile_major.cuh"
#include "quantize/bf16_norm_tail_ncdhw.cuh"
#include "quantize/bf16_avgdown3d_ndhwc.cuh"
#include "lingbot_va/fp8_per_token_block_quant.cuh"
#include "lingbot_va/qkv_split_norm_rope_bf16.cuh"
#include "lingbot_va/sage2_attn_raw.cuh"

namespace py = pybind11;

static void* to_ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }
template <typename T> static T* typed_ptr(uintptr_t addr) {
    return reinterpret_cast<T*>(addr);
}
static cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

PYBIND11_MODULE(flash_rt_lingbot_va_kernels, m) {
    m.doc() = "FlashRT LingBot/Wan SM89 kernels (opt-in)";

m.def("bias_rms_norm_bf16_q8", [](uintptr_t x, uintptr_t bias, uintptr_t weight,
                                      uintptr_t out, uintptr_t out_q8,
                                      int seq_len, int dim, float eps, float scale,
                                      uintptr_t stream) {
        bias_rms_norm_bf16_q8(typed_ptr<__nv_bfloat16>(x),
                              typed_ptr<__nv_bfloat16>(bias),
                              typed_ptr<__nv_bfloat16>(weight),
                              typed_ptr<__nv_bfloat16>(out),
                              reinterpret_cast<int8_t*>(out_q8),
                              seq_len, dim, eps, scale, to_stream(stream));
    }, py::arg("x"), py::arg("bias"), py::arg("weight"), py::arg("out"),
       py::arg("out_q8"), py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-6f, py::arg("scale") = 1.0f, py::arg("stream") = 0);

m.def("quantize_i8_f16", [](uintptr_t k, uintptr_t v, uintptr_t k8, uintptr_t v16,
                                long n, float scale, uintptr_t stream) {
        quantize_i8_f16(reinterpret_cast<const void*>(k),
                        reinterpret_cast<const void*>(v),
                        reinterpret_cast<void*>(k8), reinterpret_cast<void*>(v16),
                        n, scale, to_stream(stream));
    }, py::arg("k"), py::arg("v"), py::arg("k8"), py::arg("v16"),
       py::arg("n"), py::arg("scale") = 1.0f, py::arg("stream") = 0);

m.def("gated_residual_sstp_layer_norm_fp8_block128",
        [](uintptr_t residual, uintptr_t x, uintptr_t xbias, uintptr_t gate_ss_row,
           uintptr_t gate_tp, uintptr_t weight, uintptr_t bias, uintptr_t out,
           uintptr_t out_scale, int seq_len, int dim, float eps, uintptr_t stream) {
        gated_residual_sstp_layer_norm_fp8_block128(
            typed_ptr<__nv_bfloat16>(residual), typed_ptr<__nv_bfloat16>(x),
            typed_ptr<__nv_bfloat16>(xbias), reinterpret_cast<const float*>(gate_ss_row),
            reinterpret_cast<const float*>(gate_tp), typed_ptr<__nv_bfloat16>(weight),
            typed_ptr<__nv_bfloat16>(bias), reinterpret_cast<__nv_fp8_e4m3*>(out),
            reinterpret_cast<float*>(out_scale), seq_len, dim, eps, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("xbias"), py::arg("gate_ss_row"),
       py::arg("gate_tp"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("out_scale"), py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-6f, py::arg("stream") = 0,
       "Wan gated residual (sstp form) then layer_norm_fp8_block128 of the updated "
       "residual in one launch; bit-identical to the two launches.");

m.def("layer_norm_fp8_block128",
        [](uintptr_t x, uintptr_t weight, uintptr_t bias, uintptr_t out,
           uintptr_t out_scale, int seq_len, int dim, float eps, uintptr_t stream) {
        layer_norm_fp8_block128(typed_ptr<__nv_bfloat16>(x),
                                typed_ptr<__nv_bfloat16>(weight),
                                typed_ptr<__nv_bfloat16>(bias),
                                reinterpret_cast<__nv_fp8_e4m3*>(out),
                                reinterpret_cast<float*>(out_scale),
                                seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("out_scale"), py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-6f, py::arg("stream") = 0,
       "LayerNorm emitting FP8 with 128-wide per-token scales in one launch, "
       "bit-identical to layer_norm followed by "
       "fp8_per_token_block128_quant_bf16.");

m.def("wan_adaln_ln_fp8", [](uintptr_t x, uintptr_t ss_table, uintptr_t tproj,
                                  uintptr_t out_fp8, uintptr_t act_scale,
                                  int B, int S, int D, int scale_idx, int shift_idx,
                                  float eps, uintptr_t stream) {
        flash_rt::wan_adaln::wan_adaln_ln_fp8(
            x, ss_table, tproj, out_fp8, act_scale,
            B, S, D, scale_idx, shift_idx, eps, stream);
    }, py::arg("x"), py::arg("ss_table"), py::arg("tproj"),
       py::arg("out_fp8"), py::arg("act_scale"),
       py::arg("B"), py::arg("S"), py::arg("D"),
       py::arg("scale_idx"), py::arg("shift_idx"),
       py::arg("eps") = 1e-6f, py::arg("stream") = 0,
       "LayerNorm (no affine) + per-token AdaLN modulate + FP8 static quant.");

m.def("wan_adaln_residual_ln_fp8",
          [](uintptr_t residual, uintptr_t x, uintptr_t gate_vec,
             uintptr_t ss_table, uintptr_t tproj,
             uintptr_t out_fp8, uintptr_t act_scale,
             int B, int S, int D, int scale_idx, int shift_idx,
             float eps, uintptr_t stream) {
        flash_rt::wan_adaln::wan_adaln_residual_ln_fp8(
            residual, x, gate_vec, ss_table, tproj, out_fp8, act_scale,
            B, S, D, scale_idx, shift_idx, eps, stream);
    }, py::arg("residual"), py::arg("x"), py::arg("gate_vec_fp32"),
       py::arg("ss_table"), py::arg("tproj"),
       py::arg("out_fp8"), py::arg("act_scale"),
       py::arg("B"), py::arg("S"), py::arg("D"),
       py::arg("scale_idx"), py::arg("shift_idx"),
       py::arg("eps") = 1e-6f, py::arg("stream") = 0,
       "residual += x*gate, then AdaLN(LayerNorm) + FP8 static quant.");

m.def("wan_adaln_residual_ln_bf16",
          [](uintptr_t residual, uintptr_t x, uintptr_t gate_vec,
             uintptr_t ss_table, uintptr_t tproj, uintptr_t out_bf16,
             int B, int S, int D, int scale_idx, int shift_idx,
             float eps, uintptr_t stream) {
        flash_rt::wan_adaln::wan_adaln_residual_ln_bf16(
            residual, x, gate_vec, ss_table, tproj, out_bf16,
            B, S, D, scale_idx, shift_idx, eps, stream);
    }, py::arg("residual"), py::arg("x"), py::arg("gate_vec"),
       py::arg("ss_table"), py::arg("tproj"), py::arg("out_bf16"),
       py::arg("B"), py::arg("S"), py::arg("D"),
       py::arg("scale_idx"), py::arg("shift_idx"),
       py::arg("eps") = 1e-6f, py::arg("stream") = 0,
       "residual += x*gate, then AdaLN(LayerNorm) with bf16 output.");

m.def("wan_gated_residual_bf16",
          [](uintptr_t residual, uintptr_t x, uintptr_t gate_vec,
             int B, int S, int D, uintptr_t stream) {
        flash_rt::wan_adaln::wan_gated_residual_bf16(
            residual, x, gate_vec, B, S, D, stream);
    }, py::arg("residual"), py::arg("x"), py::arg("gate_vec_fp32"),
       py::arg("B"), py::arg("S"), py::arg("D"), py::arg("stream") = 0,
       "residual += x * gate_vec, bf16 round on the sum only. gate_vec is fp32 "
       "(the eager block never rounds it) and may be 0 for a plain add.");

m.def("wan_adaln_residual_ln_fp8_block128",
          [](uintptr_t residual, uintptr_t x, uintptr_t gate_vec, uintptr_t ss_table,
             uintptr_t tproj, uintptr_t out_fp8, uintptr_t out_scale,
             int B, int S, int D, int scale_idx, int shift_idx, float eps,
             uintptr_t stream, int tp_bcast) {
        flash_rt::wan_adaln::wan_adaln_residual_ln_fp8_block128(
            residual, x, gate_vec, ss_table, tproj, out_fp8, out_scale,
            B, S, D, scale_idx, shift_idx, eps, stream, tp_bcast);
    }, py::arg("residual"), py::arg("x"), py::arg("gate_vec_fp32"),
       py::arg("ss_table_fp32"), py::arg("tproj_bf16"), py::arg("out_fp8"),
       py::arg("out_scale_fp32"), py::arg("B"), py::arg("S"), py::arg("D"),
       py::arg("scale_idx"), py::arg("shift_idx"), py::arg("eps") = 1e-6f,
       py::arg("stream") = 0, py::arg("tp_bcast") = 0,
       "AdaLN + LayerNorm emitting FP8 with 128-wide per-token scales in one "
       "launch, bit-identical to this kernel followed by "
       "fp8_per_token_block128_quant_bf16.");

m.def("wan_gated_residual_sstp_adaln_fp8_block128",
          [](uintptr_t residual, uintptr_t x, uintptr_t xbias, uintptr_t gate_ss_row,
             uintptr_t gate_tp, uintptr_t ss_table, uintptr_t tproj, uintptr_t out_fp8,
             uintptr_t out_scale, int B, int S, int D, int scale_idx, int shift_idx,
             float eps, uintptr_t stream, int tp_bcast) {
        flash_rt::wan_adaln::wan_gated_residual_sstp_adaln_fp8_block128(
            residual, x, xbias, gate_ss_row, gate_tp, ss_table, tproj, out_fp8,
            out_scale, B, S, D, scale_idx, shift_idx, eps, stream, tp_bcast);
    }, py::arg("residual"), py::arg("x"), py::arg("xbias_bf16"),
       py::arg("gate_ss_row_fp32"), py::arg("gate_tp_fp32"), py::arg("ss_table_fp32"),
       py::arg("tproj_bf16"), py::arg("out_fp8"), py::arg("out_scale_fp32"),
       py::arg("B"), py::arg("S"), py::arg("D"), py::arg("scale_idx"),
       py::arg("shift_idx"), py::arg("eps") = 1e-6f, py::arg("stream") = 0,
       py::arg("tp_bcast") = 0,
       "wan_gated_residual_sstp_bias_bf16 then wan_adaln_residual_ln_fp8_block128 "
       "(x = 0) on the updated residual, in one launch; bit-identical to the pair.");

m.def("wan_gated_residual_sstp_bias_bf16",
          [](uintptr_t residual, uintptr_t x, uintptr_t bias_bf16, uintptr_t ss_row,
             uintptr_t tp, int B, int S, int D, uintptr_t stream, int tp_bcast) {
        flash_rt::wan_adaln::wan_gated_residual_sstp_bias_bf16(
            residual, x, bias_bf16, ss_row, tp, B, S, D, stream, tp_bcast);
    }, py::arg("residual"), py::arg("x"), py::arg("bias_bf16"), py::arg("ss_row_fp32"),
       py::arg("tp_fp32"), py::arg("B"), py::arg("S"), py::arg("D"),
       py::arg("stream") = 0, py::arg("tp_bcast") = 0,
       "As wan_gated_residual_sstp_bf16, with the producer's bias folded in: the "
       "caller skips its add_bias_bf16 launch over x. bf16(fp32(x) + fp32(bias[d])) "
       "then the same gate multiply and residual add, so it is exactly the two "
       "launches it replaces. bias_bf16 is [D] and may be 0.");

m.def("wan_gated_residual_bias_bf16",
          [](uintptr_t residual, uintptr_t x, uintptr_t bias_bf16, uintptr_t gate_vec,
             int B, int S, int D, uintptr_t stream) {
        flash_rt::wan_adaln::wan_gated_residual_bias_bf16(
            residual, x, bias_bf16, gate_vec, B, S, D, stream);
    }, py::arg("residual"), py::arg("x"), py::arg("bias_bf16"),
       py::arg("gate_vec_fp32"), py::arg("B"), py::arg("S"), py::arg("D"),
       py::arg("stream") = 0,
       "As wan_gated_residual_bf16, with the producer's bias folded in before the "
       "gate multiply. bias_bf16 is [D] and may be 0.");

m.def("fp8_block128_to_nvfp4_sf128_swizzled_bf16",
        [](uintptr_t w_fp8, uintptr_t w_block_scale_fp32,
           uintptr_t nvfp4_packed, uintptr_t nvfp4_sf_swizzled,
           uintptr_t scratch_global_amax, uintptr_t out_global_scale,
           int N, int K, uintptr_t stream) {
            flash_rt::quantize::fp8_block128_to_nvfp4_sf128_swizzled_bf16(
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

m.def("fp8_block128_to_nvfp4_sf128_plain_bf16",
        [](uintptr_t w_fp8, uintptr_t w_block_scale_fp32,
           uintptr_t nvfp4_packed, uintptr_t sf128_plain,
           uintptr_t scratch_global_amax, uintptr_t out_global_scale,
           int N, int K, uintptr_t stream) {
            flash_rt::quantize::fp8_block128_to_nvfp4_sf128_plain_bf16(
                to_ptr(w_fp8),
                reinterpret_cast<const float*>(w_block_scale_fp32),
                reinterpret_cast<uint8_t*>(nvfp4_packed),
                reinterpret_cast<uint8_t*>(sf128_plain),
                reinterpret_cast<float*>(scratch_global_amax),
                reinterpret_cast<float*>(out_global_scale),
                N, K, to_stream(stream));
        },
        py::arg("w_fp8"), py::arg("w_block_scale_fp32"),
        py::arg("nvfp4_packed"), py::arg("sf128_plain"),
        py::arg("scratch_global_amax"), py::arg("out_global_scale"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

m.def("nvfp4_pack_tile_major",
        [](uintptr_t packed, uintptr_t packed_tile_major, int N, int K,
           uintptr_t stream) {
            int rc = flash_rt::quantize::nvfp4_pack_tile_major(
                reinterpret_cast<const uint8_t*>(packed),
                reinterpret_cast<uint8_t*>(packed_tile_major),
                N, K, to_stream(stream));
            if (rc != 0)
                throw std::runtime_error(
                    "nvfp4_pack_tile_major: N must be a multiple of 64 and K "
                    "of 128");
        },
        py::arg("nvfp4_packed"), py::arg("nvfp4_packed_tile_major"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

m.def("fp8_per_token_block128_quant_warp_bf16",
        [](uintptr_t input, uintptr_t output_fp8, uintptr_t output_scale,
           int M, int K, uintptr_t stream) {
            flash_rt::quantize::fp8_per_token_block128_quant_warp_bf16(
                to_ptr(input), to_ptr(output_fp8),
                reinterpret_cast<float*>(output_scale),
                M, K, to_stream(stream));
        },
        py::arg("input"), py::arg("output_fp8"),
        py::arg("output_scale"),
        py::arg("M"), py::arg("K"), py::arg("stream") = 0);

m.def("gelu_tanh_bias_to_fp8_block128_warp_bf16",
        [](uintptr_t input, uintptr_t bias, uintptr_t output_fp8,
           uintptr_t output_scale, int M, int K, uintptr_t stream) {
            flash_rt::quantize::gelu_tanh_bias_to_fp8_block128_bf16(
                to_ptr(input), to_ptr(bias), to_ptr(output_fp8),
                reinterpret_cast<float*>(output_scale), M, K, to_stream(stream));
        },
        py::arg("input"), py::arg("bias"), py::arg("output_fp8"),
        py::arg("output_scale"), py::arg("M"), py::arg("K"),
        py::arg("stream") = 0);

m.def("bf16_norm_tail_ncdhw",
        [](uintptr_t x_bf16, uintptr_t nrm_fp32, uintptr_t gamma_bf16,
           uintptr_t bias_bf16, uintptr_t y_bf16, int B, int C, int T, int H,
           int W, float scale, uintptr_t stream) {
            return flash_rt::quantize::bf16_norm_tail_ncdhw(
                to_ptr(x_bf16), to_ptr(nrm_fp32), to_ptr(gamma_bf16),
                bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                to_ptr(y_bf16), B, C, T, H, W, scale, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("nrm_fp32"), py::arg("gamma_bf16"),
        py::arg("bias_bf16"), py::arg("y_bf16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("scale"), py::arg("stream") = 0);

m.def("bf16_norm_tail_ndhwc",
        [](uintptr_t x_bf16, uintptr_t nrm_fp32, uintptr_t gamma_bf16,
           uintptr_t bias_bf16, uintptr_t y_bf16, int B, int C, int T, int H,
           int W, float scale, uintptr_t stream) {
            return flash_rt::quantize::bf16_norm_tail_ndhwc(
                to_ptr(x_bf16), to_ptr(nrm_fp32), to_ptr(gamma_bf16),
                bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                to_ptr(y_bf16), B, C, T, H, W, scale, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("nrm_fp32"), py::arg("gamma_bf16"),
        py::arg("bias_bf16"), py::arg("y_bf16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("scale"), py::arg("stream") = 0);

m.def("bf16_avgdown3d_ndhwc",
        [](uintptr_t x, uintptr_t y, int B, int C, int T, int H, int W, int ft, int fs,
           int out_ch, int order, uintptr_t stream) {
            return flash_rt::quantize::bf16_avgdown3d_ndhwc(
                to_ptr(x), to_ptr(y), B, C, T, H, W, ft, fs, out_ch, order, to_stream(stream));
        },
        py::arg("x"), py::arg("y"), py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"),
        py::arg("W"), py::arg("ft"), py::arg("fs"), py::arg("out_ch"), py::arg("order") = 0,
        py::arg("stream") = 0);

m.def("bf16_norm_tail_ndhwc_v8",
        [](uintptr_t x_bf16, uintptr_t nrm_fp32, uintptr_t gamma_bf16,
           uintptr_t bias_bf16, uintptr_t y_bf16, int B, int C, int T, int H,
           int W, float scale, int silu, uintptr_t stream) {
            return flash_rt::quantize::bf16_norm_tail_ndhwc_v8(
                to_ptr(x_bf16), to_ptr(nrm_fp32), to_ptr(gamma_bf16),
                bias_bf16 ? to_ptr(bias_bf16) : nullptr,
                to_ptr(y_bf16), B, C, T, H, W, scale, silu, to_stream(stream));
        },
        py::arg("x_bf16"), py::arg("nrm_fp32"), py::arg("gamma_bf16"),
        py::arg("bias_bf16"), py::arg("y_bf16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("scale"), py::arg("silu"), py::arg("stream") = 0);

m.def("qkv_split_bias_norm_rope_v_cat2_bf16",
        [](uintptr_t packed_qkv, uintptr_t qkv_bias,
           uintptr_t norm_q_w, uintptr_t norm_k_w,
           uintptr_t freqs_re, uintptr_t freqs_im,
           uintptr_t q_cat_out, uintptr_t k_cat_out, uintptr_t v_cat_out,
           int B, int Lq, int off_q, int Lkv, int off_kv, int L_v,
           int N, int D_h, int seq_len, float eps, uintptr_t stream) {
            flash_rt::quantize::qkv_split_bias_norm_rope_v_cat2_bf16(
                to_ptr(packed_qkv), to_ptr(qkv_bias),
                to_ptr(norm_q_w), to_ptr(norm_k_w),
                reinterpret_cast<const float*>(freqs_re),
                reinterpret_cast<const float*>(freqs_im),
                to_ptr(q_cat_out), to_ptr(k_cat_out), to_ptr(v_cat_out),
                B, Lq, off_q, Lkv, off_kv, L_v, N, D_h, seq_len, eps,
                to_stream(stream));
        },
        py::arg("packed_qkv"), py::arg("qkv_bias"),
        py::arg("norm_q_w"), py::arg("norm_k_w"),
        py::arg("freqs_re"), py::arg("freqs_im"),
        py::arg("q_cat_out"), py::arg("k_cat_out"), py::arg("v_cat_out"),
        py::arg("B"), py::arg("Lq"), py::arg("off_q"), py::arg("Lkv"),
        py::arg("off_kv"), py::arg("L_v"), py::arg("N"), py::arg("D_h"),
        py::arg("seq_len"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

m.def("qkv_split_bias_norm_rope_v_cat2_sage",
        [](uintptr_t packed_qkv, uintptr_t qkv_bias,
           uintptr_t norm_q_w, uintptr_t norm_k_w,
           uintptr_t freqs_re, uintptr_t freqs_im,
           uintptr_t q_cat_out, uintptr_t k_cat_out, uintptr_t v_cat_out,
           int B, int Lq, int off_q, int Lkv, int off_kv, int L_v,
           int N, int D_h, int seq_len, float eps,
           float q_scale, float k_scale, uintptr_t stream,
           int v_out_rows, int off_v, uintptr_t q_scale_h, uintptr_t k_scale_h,
           uintptr_t ring_origin, int ring_R) {
            flash_rt::quantize::qkv_split_bias_norm_rope_v_cat2_sage(
                to_ptr(packed_qkv), to_ptr(qkv_bias),
                to_ptr(norm_q_w), to_ptr(norm_k_w),
                reinterpret_cast<const float*>(freqs_re),
                reinterpret_cast<const float*>(freqs_im),
                to_ptr(q_cat_out), to_ptr(k_cat_out), to_ptr(v_cat_out),
                B, Lq, off_q, Lkv, off_kv, L_v, N, D_h, seq_len, eps,
                q_scale, k_scale, v_out_rows, off_v, to_stream(stream),
                reinterpret_cast<const float*>(q_scale_h),
                reinterpret_cast<const float*>(k_scale_h),
                reinterpret_cast<const int*>(ring_origin), ring_R);
        },
        py::arg("packed_qkv"), py::arg("qkv_bias"),
        py::arg("norm_q_w"), py::arg("norm_k_w"),
        py::arg("freqs_re"), py::arg("freqs_im"),
        py::arg("q_cat_out"), py::arg("k_cat_out"), py::arg("v_cat_out"),
        py::arg("B"), py::arg("Lq"), py::arg("off_q"), py::arg("Lkv"),
        py::arg("off_kv"), py::arg("L_v"), py::arg("N"), py::arg("D_h"),
        py::arg("seq_len"), py::arg("eps") = 1e-6f,
        py::arg("q_scale") = 1.0f, py::arg("k_scale") = 1.0f,
        py::arg("stream") = 0,
        py::arg("v_out_rows") = -1, py::arg("off_v") = -1,
        py::arg("q_scale_h") = 0, py::arg("k_scale_h") = 0,
        py::arg("ring_origin") = 0, py::arg("ring_R") = 0,
        "QKV split+norm+RoPE emitting int8 q/k (static scales; per head when "
        "q_scale_h/k_scale_h are given) and fp16 v "
        "for the Sage2 SM89 attention core.");

m.def("v_transpose_f8",
        [](uintptr_t src, uintptr_t dst, long total, int H, int D, int LT,
           int off, int PAD, float scale, uintptr_t stream) {
            flash_rt::quantize::v_transpose_f8(
                to_ptr(src), to_ptr(dst), total, H, D, LT, off, PAD, scale,
                to_stream(stream));
        },
        py::arg("src"), py::arg("dst"), py::arg("total"), py::arg("H"),
        py::arg("D"), py::arg("LT"), py::arg("off"), py::arg("PAD"),
        py::arg("scale") = 1.0f, py::arg("stream") = 0,
        "fp16 V -> fp8 e4m3 transposed into [B, D, H, PAD] at KV offset off.");

m.def("kv_window_shift_sage",
        [](uintptr_t k, uintptr_t vt, int B, int total, int row_bytes, int pad,
           int keep, int drop, uintptr_t stream) {
            int rc = flash_rt::quantize::kv_window_shift_sage(
                to_ptr(k), to_ptr(vt), B, total, row_bytes, pad, keep, drop,
                to_stream(stream));
            if (rc != 0)
                throw std::runtime_error("kv_window_shift_sage failed (rc=" +
                                         std::to_string(rc) + ")");
        },
        py::arg("k"), py::arg("vt"), py::arg("B"), py::arg("total"),
        py::arg("row_bytes"), py::arg("pad"), py::arg("keep"), py::arg("drop"),
        py::arg("stream") = 0,
        "In-place one-pass window shift of a Sage int8-K / transposed fp8-V cache.");

m.def("v_transpose_f8_tile",
        [](uintptr_t src, uintptr_t dst, int B, int H, int D, int LT,
           int off, int PAD, float scale, uintptr_t stream) {
            flash_rt::quantize::v_transpose_f8_tile(
                to_ptr(src), to_ptr(dst), B, H, D, LT, off, PAD, scale,
                to_stream(stream));
        },
        py::arg("src"), py::arg("dst"), py::arg("B"), py::arg("H"),
        py::arg("D"), py::arg("LT"), py::arg("off"), py::arg("PAD"),
        py::arg("scale") = 1.0f, py::arg("stream") = 0,
        "Coalesced 32x32-tile fp16 V -> fp8 e4m3 transpose (off must be 16-aligned).");

m.def("kv_ring_advance",
        [](uintptr_t origin, int drop, int ring_R, uintptr_t stream) {
            int rc = flash_rt::quantize::kv_ring_advance(to_ptr(origin), drop, ring_R,
                                                         to_stream(stream));
            if (rc != 0)
                throw std::runtime_error("kv_ring_advance failed (rc=" + std::to_string(rc) + ")");
        },
        py::arg("origin"), py::arg("drop"), py::arg("ring_R"), py::arg("stream") = 0,
        "K-ring window move: origin = (origin + drop) % ring_R on the device.");

m.def("sage2_attn_self_f8",
        [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
           uintptr_t q_scale, uintptr_t k_scale, uintptr_t v_scale,
           int B, int Sq, int Skv, int H, float sm_scale,
           int q_bz, int k_bz, int v_padded_k, int o_bz, uintptr_t stream) {
            int rc = flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128_strided(
                to_ptr(Q), to_ptr(K), to_ptr(V), to_ptr(O),
                to_ptr(q_scale), to_ptr(k_scale), to_ptr(v_scale),
                B, Sq, Skv, H, sm_scale, q_bz, k_bz, v_padded_k, o_bz,
                to_stream(stream));
            if (rc != 0)
                throw std::runtime_error("sage2_attn_self_f8 launch failed");
        },
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("q_scale"), py::arg("k_scale"), py::arg("v_scale"),
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads"), py::arg("softmax_scale"),
        py::arg("q_batch_stride"), py::arg("k_batch_stride"),
        py::arg("v_padded_k"), py::arg("o_batch_stride"),
        py::arg("stream") = 0,
        "Sage2 SM89 attention, V in per-channel fp8 [B, D, H, PAD].");

m.def("flow_cfg_euler_patch_bf16",
        [](uintptr_t pred, uintptr_t lat, uintptr_t out, int C, int F, int H, int W,
           int ph, int pw, int ld, float g, float dsig, uintptr_t stream) {
            int rc = flash_rt::kernels::flow_cfg_euler_patch_bf16(
                to_ptr(pred), to_ptr(lat), to_ptr(out), C, F, H, W, ph, pw, ld, g, dsig,
                to_stream(stream));
            if (rc != 0)
                throw std::runtime_error("flow_cfg_euler_patch_bf16 failed (rc=" +
                                         std::to_string(rc) + ")");
        },
        py::arg("pred"), py::arg("lat"), py::arg("out"), py::arg("C"), py::arg("F"),
        py::arg("H"), py::arg("W"), py::arg("ph"), py::arg("pw"), py::arg("ld"),
        py::arg("g"), py::arg("dsig"), py::arg("stream") = 0,
        "CFG blend + Euler step from a [2, L, ld] bf16 patch sequence into a "
        "[C, F, H, W] latent (bf16 roundings of the tensor expressions).");

m.def("xattn_probs_fp8",
        [](uintptr_t q, int ld_q, uintptr_t keys, int ld_k, uintptr_t log2_mult,
           uintptr_t row_elem, uintptr_t elem_k0, uintptr_t elem_nk, uintptr_t elem_col0,
           uintptr_t col_scale, uintptr_t P, int ld_p, int L, int heads, int hd,
           float scale, float p_mul, uintptr_t stream) {
            int rc = flash_rt::kernels::xattn_probs_fp8(
                to_ptr(q), ld_q, to_ptr(keys), ld_k,
                reinterpret_cast<const float*>(log2_mult),
                reinterpret_cast<const int*>(row_elem), reinterpret_cast<const int*>(elem_k0),
                reinterpret_cast<const int*>(elem_nk), reinterpret_cast<const int*>(elem_col0),
                reinterpret_cast<const float*>(col_scale), to_ptr(P), ld_p, L, heads, hd,
                scale, p_mul, to_stream(stream));
            if (rc != 0)
                throw std::runtime_error("xattn_probs_fp8 failed (rc=" +
                                         std::to_string(rc) + ")");
        },
        py::arg("q"), py::arg("ld_q"), py::arg("keys"), py::arg("ld_k"),
        py::arg("log2_mult"), py::arg("row_elem"), py::arg("elem_k0"), py::arg("elem_nk"),
        py::arg("elem_col0"), py::arg("col_scale"), py::arg("P"), py::arg("ld_p"),
        py::arg("L"), py::arg("heads"), py::arg("hd"), py::arg("scale"), py::arg("p_mul"),
        py::arg("stream") = 0,
        "Cross-attention probabilities over a few distinct text keys as FP8 "
        "operands of the projected-value GEMM.");

m.def("flow_euler_window_bf16",
        [](uintptr_t xs, uintptr_t v, uintptr_t dsig, int s, int ws, int r, int n,
           int E, int v_ch, int v_ld, int v_pt, bool extrap, uintptr_t nl,
           uintptr_t mask, int ch_stride, int ws_next, uintptr_t stream) {
            int rc = flash_rt::kernels::flow_euler_window_bf16(
                to_ptr(xs), to_ptr(v), reinterpret_cast<const float*>(dsig), s, ws, r,
                n, E, v_ch, v_ld, v_pt, extrap, to_ptr(nl), to_ptr(mask), ch_stride,
                ws_next, to_stream(stream));
            if (rc != 0)
                throw std::runtime_error("flow_euler_window_bf16 failed (rc=" +
                                         std::to_string(rc) + ")");
        },
        py::arg("xs"), py::arg("v"), py::arg("dsig"), py::arg("s"), py::arg("ws"),
        py::arg("r"), py::arg("n"), py::arg("E"), py::arg("v_ch"), py::arg("v_ld"),
        py::arg("v_pt"), py::arg("extrap"), py::arg("nl"),
        py::arg("mask"), py::arg("ch_stride"), py::arg("ws_next"),
        py::arg("stream") = 0,
        "One sliding-window Picard sweep's Euler recurrence over bf16 latents "
        "(and the next sweep's masked inputs).");

m.def("sage2_attn_self_f8_lse",
        [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O, uintptr_t lse,
           uintptr_t q_scale, uintptr_t k_scale, uintptr_t v_scale,
           int B, int Sq, int Skv, int H, float sm_scale,
           int q_bz, int k_bz, int v_padded_k, int o_bz,
           uintptr_t stream, bool per_head_scales, int tile,
           uintptr_t ring_origin, int ring_R, int ring_start) {
            int rc = ring_R > 0
                ? flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128_strided_ring(
                      to_ptr(Q), to_ptr(K), to_ptr(V), to_ptr(O),
                      reinterpret_cast<float*>(lse),
                      to_ptr(q_scale), to_ptr(k_scale), to_ptr(v_scale),
                      B, Sq, Skv, H, sm_scale, q_bz, k_bz, v_padded_k, o_bz,
                      to_stream(stream), nullptr, nullptr, per_head_scales, tile,
                      reinterpret_cast<const int*>(ring_origin), ring_R, ring_start)
                : flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128_strided_lse(
                      to_ptr(Q), to_ptr(K), to_ptr(V), to_ptr(O),
                      reinterpret_cast<float*>(lse),
                      to_ptr(q_scale), to_ptr(k_scale), to_ptr(v_scale),
                      B, Sq, Skv, H, sm_scale, q_bz, k_bz, v_padded_k, o_bz,
                      to_stream(stream), per_head_scales, tile);
            if (rc != 0)
                throw std::runtime_error("sage2_attn_self_f8_lse launch failed");
        },
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("lse"),
        py::arg("q_scale"), py::arg("k_scale"), py::arg("v_scale"),
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads"), py::arg("softmax_scale"),
        py::arg("q_batch_stride"), py::arg("k_batch_stride"),
        py::arg("v_padded_k"), py::arg("o_batch_stride"),
        py::arg("stream") = 0, py::arg("per_head_scales") = false, py::arg("tile") = 0,
        py::arg("ring_origin") = 0, py::arg("ring_R") = 0, py::arg("ring_start") = 0,
        "Sage2 SM89 attention (tile 0: 128x32, 1: 32x16, 2: 128x16), V fp8 [B, D, H, PAD], "
        "also writing the log2-domain LSE [B, H, S_q]. ring_R > 0: K is a ring "
        "(see qk_int8_sv_f8_bf16_nhd_d128_strided_ring).");

m.def("sage2_tail_merge_fp8",
        [](uintptr_t q8, uintptr_t k8, uintptr_t v16, uintptr_t o_shared,
           uintptr_t lse_shared, uintptr_t q_scale, uintptr_t k_scale,
           bool per_head, float sm_scale, int groups, int group_rows, int H,
           uintptr_t out_fp8, uintptr_t out_scale, uintptr_t out_bf16,
           uintptr_t stream, int rs_reduce) {
            int rc = flash_rt::attention::sage2::sage2_tail_merge_fp8(
                to_ptr(q8), to_ptr(k8), to_ptr(v16), to_ptr(o_shared),
                reinterpret_cast<const float*>(lse_shared),
                reinterpret_cast<const float*>(q_scale),
                reinterpret_cast<const float*>(k_scale), per_head, sm_scale,
                groups, group_rows, H, to_ptr(out_fp8),
                reinterpret_cast<float*>(out_scale), to_ptr(out_bf16),
                to_stream(stream), rs_reduce);
            if (rc != 0)
                throw std::runtime_error("sage2_tail_merge_fp8 failed (rc=" +
                                         std::to_string(rc) + ")");
        },
        py::arg("q8"), py::arg("k8"), py::arg("v16"), py::arg("o_shared"),
        py::arg("lse_shared"), py::arg("q_scale"), py::arg("k_scale"),
        py::arg("per_head"), py::arg("sm_scale"), py::arg("groups"),
        py::arg("group_rows"), py::arg("num_heads"), py::arg("out_fp8"),
        py::arg("out_scale"), py::arg("out_bf16") = 0, py::arg("stream") = 0,
        py::arg("rs_reduce") = 0,
        "Private 16-key segment per 16-row query group merged with a shared "
        "segment (normalised output + log2 LSE); FP8 block-128 output.");

m.def("sage2_attn_self_f8_tile",
        [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
           uintptr_t q_scale, uintptr_t k_scale, uintptr_t v_scale,
           int B, int Sq, int Skv, int H, float sm_scale,
           int q_bz, int k_bz, int v_padded_k, int o_bz, int tile,
           uintptr_t stream, float pad_extra, bool pad_exact, uintptr_t out_fp8,
           uintptr_t out_fp8_scale, bool per_head_scales,
           uintptr_t ring_origin, int ring_R, int ring_start) {
            if (ring_R > 0) {
                if (pad_extra != 0.0f)
                    throw std::runtime_error("sage2_attn_self_f8_tile: no pad_extra on a K ring");
                int rc = flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128_strided_ring(
                    to_ptr(Q), to_ptr(K), to_ptr(V), to_ptr(O), nullptr,
                    to_ptr(q_scale), to_ptr(k_scale), to_ptr(v_scale),
                    B, Sq, Skv, H, sm_scale, q_bz, k_bz, v_padded_k, o_bz,
                    to_stream(stream), to_ptr(out_fp8),
                    reinterpret_cast<float*>(out_fp8_scale), per_head_scales, tile,
                    reinterpret_cast<const int*>(ring_origin), ring_R, ring_start);
                if (rc != 0)
                    throw std::runtime_error("sage2_attn_self_f8_tile (ring) launch failed (rc=" +
                                             std::to_string(rc) + ")");
                return;
            }
            auto fn = (tile == 1)
                          ? flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128_strided_sq
                          : (tile == 2)
                                ? flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128_strided_w16
                          : (tile == 3)
                                ? flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128_strided_q64
                                : flash_rt::attention::sage2::qk_int8_sv_f8_bf16_nhd_d128_strided;
            int rc = fn(to_ptr(Q), to_ptr(K), to_ptr(V), to_ptr(O),
                        to_ptr(q_scale), to_ptr(k_scale), to_ptr(v_scale),
                        B, Sq, Skv, H, sm_scale, q_bz, k_bz, v_padded_k, o_bz,
                        to_stream(stream), pad_extra, pad_exact, to_ptr(out_fp8),
                        reinterpret_cast<float*>(out_fp8_scale), per_head_scales);
            if (rc != 0)
                throw std::runtime_error("sage2_attn_self_f8_tile launch failed");
        },
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("q_scale"), py::arg("k_scale"), py::arg("v_scale"),
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads"), py::arg("softmax_scale"),
        py::arg("q_batch_stride"), py::arg("k_batch_stride"),
        py::arg("v_padded_k"), py::arg("o_batch_stride"),
        py::arg("tile") = 0, py::arg("stream") = 0, py::arg("pad_extra") = 0.0f, py::arg("pad_exact") = false,
        py::arg("out_fp8") = 0, py::arg("out_fp8_scale") = 0,
        py::arg("per_head_scales") = false,
        py::arg("ring_origin") = 0, py::arg("ring_R") = 0, py::arg("ring_start") = 0,
        "Sage2 SM89 attention with a query-tile selector: 0 = 128x32 (delivered),\n"
        "1 = 32x16 short-query tile (S_q <= 32), 2 = 128x16 eight-warp tile,\n"
        "3 = 64x16 four-warp tile.\n"
        "pad_extra > 0: the last key stands for pad_extra + 1 identical K/V rows;\n"
        "pad_exact weights all of them at the fp32 probability instead of its e4m3 rounding.\n"
        "out_fp8/out_fp8_scale: emit O as FP8 with one scale per (row, head) -- the\n"
        "fp8_per_token_block128_quant of the bf16 output -- instead of bf16.\n"
        "per_head_scales: q_scale/k_scale are [num_heads] static per-head scales.\n"
        "ring_R > 0: K is a ring (key j at row (ring_origin[0] + ring_start + j) % ring_R).");

m.def("sage2_attn_self",
        [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
           uintptr_t q_scale, uintptr_t k_scale,
           int B, int Sq, int Skv, int H, float sm_scale,
           int q_bz, int k_bz, int v_bz, int o_bz, uintptr_t stream) {
            int rc = flash_rt::attention::sage2::qk_int8_sv_f16_bf16_nhd_d128_strided(
                to_ptr(Q), to_ptr(K), to_ptr(V), to_ptr(O),
                to_ptr(q_scale), to_ptr(k_scale), B, Sq, Skv, H, sm_scale,
                q_bz, k_bz, v_bz, o_bz, to_stream(stream));
            if (rc != 0)
                throw std::runtime_error("sage2_attn_self launch failed");
        },
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("q_scale"), py::arg("k_scale"),
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads"), py::arg("softmax_scale"),
        py::arg("q_batch_stride"), py::arg("k_batch_stride"),
        py::arg("v_batch_stride"), py::arg("o_batch_stride"),
        py::arg("stream") = 0,
        "Sage2 SM89 self-attention (int8 QK, fp16 V), explicit batch strides "
        "so the KV cache's fixed-capacity prefix can be used in place.");
}
