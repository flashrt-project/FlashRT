#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <sstream>
#include <string>
#include <vector>
#include "kernels/pi05/thor/pi05_rowops.cuh"
#include "kernels/pi05/thor/pi05_rowops_swizzled.cuh"
#include "kernels/pi05/thor/l2_prefetch.cuh"
#include "kernels/pi05/thor/pdl.cuh"
#include "kernels/pi05/thor/pi05_action_edges.cuh"
#include "kernels/pi05/thor/attn_lt.cuh"
#include "kernels/pi05/thor/cutlass_fp4_gemm.cuh"
#include "kernels/pi05/thor/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#include "kernels/pi05/thor/cutlass_fp4_gemm_siglip_ffn_variants_sm100.cuh"
#include "kernels/pi05/thor/patch_embed.cuh"
#include "kernels/quantize.cuh"
#include "kernels/rope_vec.cuh"
#include "fused_fp4/norm_silu_fp4_sfa.cuh"
extern "C" int cutlass_fp8_sq(void*, void*, void*, int, int, int, float, float, cudaStream_t);
static void* to_ptr(uintptr_t pointer) { return reinterpret_cast<void*>(pointer); }
template <typename T>
static T* typed_ptr(uintptr_t pointer) { return reinterpret_cast<T*>(pointer); }
namespace py = pybind11;

static cudaStream_t to_stream(uintptr_t stream) {
  return reinterpret_cast<cudaStream_t>(stream);
}

static std::string fp4_kernel_shape(
    std::initializer_list<std::pair<const char*, long long>> dims) {
  std::ostringstream out;
  bool first = true;
  for (const auto& [name, value] : dims) {
    out << (first ? "" : ", ") << name << '=' << value;
    first = false;
  }
  return out.str();
}

static void require_fp4(bool condition, const char* kernel,
                        const std::string& reason, const std::string& shape) {
  if (!condition) {
    throw py::value_error(std::string(kernel) + ": " + reason +
                          " (" + shape + ")");
  }
}

static void require_fp4_ptrs(
    const char* kernel,
    std::initializer_list<std::pair<const char*, uintptr_t>> ptrs,
    const std::string& shape) {
  for (const auto& [name, value] : ptrs) {
    require_fp4(value != 0, kernel,
                std::string(name) + " pointer must be non-null", shape);
  }
}


PYBIND11_MODULE(flash_rt_pi05_thor, m) {
m.def("attention_qkv_fp16_lt",
        [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t logits, uintptr_t out, int S, int S_kv, int NH, int HD,
           float attn_scale, uintptr_t stream) -> int {
          return flash_rt::fp4::attention_qkv_fp16_lt(reinterpret_cast<const void*>(Q), reinterpret_cast<const void*>(K),
                                                      reinterpret_cast<const void*>(V), reinterpret_cast<void*>(logits),
                                                      reinterpret_cast<void*>(out), S, S_kv, NH, HD, attn_scale,
                                                      reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("logits"), py::arg("out"), py::arg("S"), py::arg("S_kv"),
        py::arg("NH"), py::arg("HD"), py::arg("attn_scale"), py::arg("stream") = 0,
        "attention_qkv_fp16 through cublasLt with per-shape autotuned heuristics (QK^T, softmax, PV).");

m.def("cutlass_fp4_gemm_bias_gelu_fp4out_v",
        [](int idx, uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB, uintptr_t bias,
           uintptr_t D, uintptr_t SFD, int M, int N, int K, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_bias_gelu_fp4out_v(idx,
              reinterpret_cast<void const*>(A), reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B), reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void const*>(bias), reinterpret_cast<void*>(D),
              reinterpret_cast<void*>(SFD), M, N, K, reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("idx"), py::arg("A"), py::arg("SFA"), py::arg("B"), py::arg("SFB"), py::arg("bias"),
        py::arg("D"), py::arg("SFD"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0,
        "SigLIP Up GEMM (bias + GELU + fp4/SFA out) with a selectable MMA tile.");

m.def("cutlass_fp4_gemm_bias_res_fp16_v",
        [](int idx, uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB, uintptr_t bias,
           uintptr_t C, uintptr_t D, int M, int N, int K, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_bias_res_fp16_v(idx,
              reinterpret_cast<void const*>(A), reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B), reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void const*>(bias), reinterpret_cast<void const*>(C),
              reinterpret_cast<void*>(D), M, N, K, reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("idx"), py::arg("A"), py::arg("SFA"), py::arg("B"), py::arg("SFB"), py::arg("bias"),
        py::arg("C"), py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0,
        "SigLIP Down GEMM (bias + residual, fp16 out) with a selectable MMA tile.");

m.def("cutlass_fp4_gemm_geglu_il_hw_nod",
        [](uintptr_t A_packed, uintptr_t SFA,
           uintptr_t B_packed, uintptr_t SFB,
           uintptr_t D_dummy, uintptr_t compact_packed, uintptr_t compact_sfa,
           int M, int N_il, int K, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"M", M}, {"N_il", N_il}, {"K", K}});
          require_fp4_ptrs("cutlass_fp4_gemm_geglu_il_hw_nod",
                           {{"A_packed", A_packed}, {"SFA", SFA},
                            {"B_packed", B_packed}, {"SFB", SFB},
                            {"D_dummy", D_dummy},
                            {"compact_packed", compact_packed},
                            {"compact_sfa", compact_sfa}}, shape);
          require_fp4(M > 0 && N_il > 0 && K > 0 && (N_il % 32) == 0 &&
                      (K % 16) == 0,
                      "cutlass_fp4_gemm_geglu_il_hw_nod",
                      "M must be positive, N_il a positive multiple of 32 "
                      "and K a positive multiple of 16",
                      shape);
          return flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_nod(
              reinterpret_cast<void const*>(A_packed),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B_packed),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D_dummy),
              reinterpret_cast<void*>(compact_packed),
              reinterpret_cast<void*>(compact_sfa),
              M, N_il, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A_packed"), py::arg("SFA"),
        py::arg("B_packed"), py::arg("SFB"),
        py::arg("D_dummy"), py::arg("compact_packed"), py::arg("compact_sfa"),
        py::arg("M"), py::arg("N_il"), py::arg("K"),
        py::arg("stream") = 0,
        R"pbdoc(
Half-width fused GeGLU GEMM with the collective's own D store elided:
compact_packed/compact_sfa are the only outputs and D_dummy is never
written (still validated; the host-side TMA descriptor needs a real
pointer).  Same contract as cutlass_fp4_gemm_geglu_il_hw otherwise.
)pbdoc");

m.def("cutlass_fp4_gemm_geglu_il_hw_nod_swap",
        [](uintptr_t A_packed, uintptr_t SFA,
           uintptr_t B_packed, uintptr_t SFB,
           uintptr_t D_dummy, uintptr_t compact_packed, uintptr_t compact_sfa,
           int M, int N_il, int K, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"M", M}, {"N_il", N_il}, {"K", K}});
          require_fp4_ptrs("cutlass_fp4_gemm_geglu_il_hw_nod_swap",
                           {{"A_packed", A_packed}, {"SFA", SFA},
                            {"B_packed", B_packed}, {"SFB", SFB},
                            {"D_dummy", D_dummy},
                            {"compact_packed", compact_packed},
                            {"compact_sfa", compact_sfa}}, shape);
          require_fp4(M > 0 && N_il > 0 && K > 0 && (N_il % 32) == 0 &&
                      (K % 16) == 0,
                      "cutlass_fp4_gemm_geglu_il_hw_nod_swap",
                      "M must be positive, N_il a positive multiple of 32 "
                      "and K a positive multiple of 16",
                      shape);
          return flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_nod_swap(
              reinterpret_cast<void const*>(A_packed),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B_packed),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D_dummy),
              reinterpret_cast<void*>(compact_packed),
              reinterpret_cast<void*>(compact_sfa),
              M, N_il, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A_packed"), py::arg("SFA"),
        py::arg("B_packed"), py::arg("SFB"),
        py::arg("D_dummy"), py::arg("compact_packed"), py::arg("compact_sfa"),
        py::arg("M"), py::arg("N_il"), py::arg("K"),
        py::arg("stream") = 0,
        R"pbdoc(
Skinny-M no-D-store fused GeGLU GEMM on the decoder tile (128x64x256);
same contract as cutlass_fp4_gemm_geglu_il_hw_nod.
)pbdoc");

m.def("cutlass_fp4_gemm_num_variants", &flash_rt::fp4::cutlass_fp4_gemm_num_variants,
        "Count of available GEMM variants.");

m.def("cutlass_fp4_gemm_variant",
        [](int idx, uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t D, int M, int N, int K, float alpha, float beta,
           uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_variant(
              idx, reinterpret_cast<void const*>(A), reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B), reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D), M, N, K, alpha, beta,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("idx"), py::arg("A"), py::arg("SFA"),
        py::arg("B"), py::arg("SFB"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("stream") = 0,
        "Call one of the NVFP4 GEMM variants by index. Used for tile/schedule tuning.");

m.def("l2_touch_fork",
        [](const std::vector<std::pair<uintptr_t, unsigned long long>>& regions,
           uintptr_t main_stream, int nctas, int hint, uintptr_t sink, int depth, unsigned pace_ns, int nthreads) -> int {
          if (regions.size() > static_cast<size_t>(flash_rt::fp4::kL2TouchMaxRegions)) return -1;
          flash_rt::fp4::L2PrefetchRegions r{};
          r.count = static_cast<int>(regions.size());
          for (size_t i = 0; i < regions.size(); ++i) {
            r.ptr[i] = reinterpret_cast<const void*>(regions[i].first);
            r.bytes[i] = regions[i].second;
          }
          return flash_rt::fp4::l2_touch_fork(r, reinterpret_cast<cudaStream_t>(main_stream), nctas, hint,
                                              reinterpret_cast<void*>(sink), depth, pace_ns, nthreads);
        }, py::arg("regions"), py::arg("main_stream") = 0, py::arg("nctas") = 4, py::arg("hint") = 1, py::arg("sink") = 0,
        py::arg("depth") = 16, py::arg("pace_ns") = 0, py::arg("nthreads") = 256,
        "Real-load L2 touch of up to 8 (ptr, bytes) regions on a side stream forked from main_stream "
        "(nctas CTAs; hint 1 = L2::evict_last).");

m.def("l2_touch_init", []() -> int { return flash_rt::fp4::l2_touch_init(); },
        "Create the L2 touch side stream and its fork/join events (call before any graph capture).");

m.def("l2_touch_join",
        [](uintptr_t main_stream) -> int { return flash_rt::fp4::l2_touch_join(reinterpret_cast<cudaStream_t>(main_stream)); },
        py::arg("main_stream") = 0, "Make main_stream wait for the L2 touch side stream.");

m.def("pi05_patch_embed_bias_pos", [](uintptr_t output, uintptr_t bias, uintptr_t pos_emb,
                                         int S, int D, int S_per_view, uintptr_t stream) {
        pi05_patch_embed_bias_pos(reinterpret_cast<half*>(output),
                                reinterpret_cast<const half*>(bias),
                                reinterpret_cast<const half*>(pos_emb),
                                S, D, S_per_view, to_stream(stream));
    }, py::arg("output"), py::arg("bias"), py::arg("pos_emb"),
       py::arg("S"), py::arg("D"), py::arg("S_per_view"), py::arg("stream") = 0);

m.def("pi05_patch_im2col_uint8_pitch", [](uintptr_t input, uintptr_t lut, uintptr_t output,
                                          int nv, int pitch, uintptr_t stream) {
        pi05_patch_im2col_uint8_pitch(reinterpret_cast<const uint8_t*>(input),
                                 reinterpret_cast<const half*>(lut),
                                 reinterpret_cast<half*>(output), nv, pitch, to_stream(stream));
    }, py::arg("input"), py::arg("lut"), py::arg("output"), py::arg("nv"), py::arg("pitch"),
       py::arg("stream") = 0);

m.def("pi05_action_in_fp16",
        [](uintptr_t noise, uintptr_t w, uintptr_t b, uintptr_t x, int S, int D, int A, uintptr_t stream) -> int {
          return flash_rt::fp4::pi05_action_in_fp16(reinterpret_cast<const void*>(noise), reinterpret_cast<const void*>(w),
                                                    reinterpret_cast<const void*>(b), reinterpret_cast<void*>(x), S, D, A,
                                                    reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("noise"), py::arg("w"), py::arg("b"), py::arg("x"), py::arg("S"), py::arg("D"), py::arg("A"),
        py::arg("stream") = 0, "Pi0.5 decoder action_in projection + bias in one launch (fp16).");

m.def("pi05_adarms_action_out_fp16",
        [](uintptr_t x, uintptr_t style, uintptr_t xn, uintptr_t gate, uintptr_t aow, uintptr_t aob, uintptr_t noise,
           int S, int D, int A, float dt, uintptr_t stream) -> int {
          return flash_rt::fp4::pi05_adarms_action_out_fp16(
              reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(style), reinterpret_cast<void*>(xn),
              reinterpret_cast<void*>(gate), reinterpret_cast<const void*>(aow), reinterpret_cast<const void*>(aob),
              reinterpret_cast<void*>(noise), S, D, A, dt, reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("x"), py::arg("style"), py::arg("xn"), py::arg("gate"), py::arg("aow"), py::arg("aob"), py::arg("noise"),
        py::arg("S"), py::arg("D"), py::arg("A"), py::arg("dt"), py::arg("stream") = 0,
        "Pi0.5 decoder final AdaRMS + action head + Euler update in one launch (fp16).");

m.def("pi05_row_layer_norm_fp16",
        [](uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t out, int S, int D, float eps,
           uintptr_t stream) -> int {
          return flash_rt::fused_fp4::pi05_row_layer_norm_fp16(
              reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(gamma),
              reinterpret_cast<const __half*>(beta), reinterpret_cast<void*>(out), S, D, eps,
              reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("out"), py::arg("S"),
        py::arg("D"), py::arg("eps"), py::arg("stream") = 0,
        "Warp-per-row LayerNorm -> fp16 (v2).");

m.def("pi05_row_layer_norm_fp8_swizzled",
        [](uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t out, int S, int D, float eps,
           uintptr_t stream) -> int {
          return flash_rt::fused_fp4::pi05_row_layer_norm_fp8_swizzled(
              reinterpret_cast<const __half*>(x), reinterpret_cast<const float*>(gamma),
              reinterpret_cast<const float*>(beta), reinterpret_cast<void*>(out), S, D, eps,
              reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("out"), py::arg("S"),
        py::arg("D"), py::arg("eps"), py::arg("stream") = 0,
        "Warp-per-row LayerNorm (fp32 tables) -> e4m3 (v5, slim).");

m.def("pi05_row_layer_norm_mul_fp4_sfa_swizzled",
        [](uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t inv_s, uintptr_t packed,
           uintptr_t sfa, int S, int D, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::pi05_row_layer_norm_mul_fp4_sfa_swizzled(
              reinterpret_cast<const __half*>(x), reinterpret_cast<const float*>(gamma),
              reinterpret_cast<const float*>(beta), reinterpret_cast<const float*>(inv_s),
              reinterpret_cast<void*>(packed), reinterpret_cast<void*>(sfa), S, D, eps,
              reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("inv_s"), py::arg("packed"),
        py::arg("sfa"), py::arg("S"), py::arg("D"), py::arg("eps"), py::arg("stream") = 0,
        "Warp-per-row LayerNorm (fp32 tables) [* inv_s] -> NVFP4 + SFA (v5, slim).");

m.def("pi05_row_quantize_fp4_sfa_swizzled",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa, int N, int D, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::pi05_row_quantize_fp4_sfa_swizzled(
              reinterpret_cast<const __half*>(src), reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa), N, D, reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("src"), py::arg("packed"), py::arg("sfa"), py::arg("N"), py::arg("D"),
        py::arg("stream") = 0, "Warp-per-row NVFP4 + SFA quantization (v5, slim).");

m.def("pi05_row_rms_fp8_swizzled",
        [](uintptr_t x, uintptr_t out, int S, int D, uintptr_t descale, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::pi05_row_rms_fp8_swizzled(
              reinterpret_cast<const __half*>(x), reinterpret_cast<void*>(out), S, D,
              reinterpret_cast<const float*>(descale), reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("x"), py::arg("out"), py::arg("S"), py::arg("D"), py::arg("descale"),
        py::arg("stream") = 0, "Warp-per-row RMSNorm -> e4m3 with static descale (v5, slim).");

m.def("pi05_row_rms_mul_fp4_sfa_swizzled",
        [](uintptr_t x, uintptr_t inv_s, uintptr_t packed, uintptr_t sfa, int S, int D,
           uintptr_t stream) -> int {
          return flash_rt::fused_fp4::pi05_row_rms_mul_fp4_sfa_swizzled(
              reinterpret_cast<const __half*>(x), reinterpret_cast<const float*>(inv_s),
              reinterpret_cast<void*>(packed), reinterpret_cast<void*>(sfa), S, D,
              reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("x"), py::arg("inv_s"), py::arg("packed"), py::arg("sfa"), py::arg("S"),
        py::arg("D"), py::arg("stream") = 0,
        "Warp-per-row RMSNorm [* fp32 inv_s] -> NVFP4 + SFA (v5, slim).");

m.def("set_pdl", [](bool on) { flash_rt::fp4::pdl_flag() = on; }, py::arg("on"),
          "Programmatic dependent launch for this module's rope/softmax/FP8 quantize kernels and the SM100 FP8 GEMM.");
m.def("pi05_quantize_fp8_static_fp16", [](uintptr_t input, uintptr_t output,
                                          uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_static_fp16(reinterpret_cast<const __half*>(input),
                                  typed_ptr<__nv_fp8_e4m3>(output),
                                  reinterpret_cast<const float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

m.def("pi05_cutlass_fp8_sq", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                 int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_sq(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

m.def("pi05_qkv_split_rope_kvcache_fp16_vec", [](uintptr_t qkv, uintptr_t rope,
                                              uintptr_t Q, uintptr_t Kc, uintptr_t Vc,
                                              int S, int Q_dim, int K_dim, int HD, int qkv_stride,
                                              long kc_offset, int kc_stride, uintptr_t stream) {
        return qkv_split_rope_kvcache_fp16_vec(
                                     reinterpret_cast<const __half*>(qkv),
                                     reinterpret_cast<const __half*>(rope),
                                     reinterpret_cast<__half*>(Q),
                                     reinterpret_cast<__half*>(Kc),
                                     reinterpret_cast<__half*>(Vc),
                                     S, Q_dim, K_dim, HD, qkv_stride,
                                     kc_offset, kc_stride, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope"), py::arg("Q"), py::arg("Kc"), py::arg("Vc"),
       py::arg("S"), py::arg("Q_dim"), py::arg("K_dim"), py::arg("HD"), py::arg("qkv_stride"),
       py::arg("kc_offset"), py::arg("kc_stride"), py::arg("stream") = 0);

m.def("pi05_adarms_fp4_sfa_native_fp16",
        [](uintptr_t x, uintptr_t style, uintptr_t packed, uintptr_t sfa,
           uintptr_t gate, int seq_len, int dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape(
              {{"seq_len", seq_len}, {"dim", dim}});
          require_fp4_ptrs("pi05_adarms_fp4_sfa_native_fp16",
                           {{"x", x}, {"style", style}, {"packed", packed},
                            {"sfa", sfa}, {"gate", gate}}, shape);
          require_fp4(seq_len == 10 && dim == 1024,
                      "pi05_adarms_fp4_sfa_native_fp16",
                      "the Pi0.5 decoder business shape is seq_len=10, dim=1024",
                      shape);
          flash_rt::fused_fp4::pi05_adarms_fp4_sfa_native_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(style),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              reinterpret_cast<__half*>(gate), seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("style"), py::arg("packed"), py::arg("sfa"),
        py::arg("gate"), py::arg("seq_len"), py::arg("dim"),
        py::arg("stream") = 0,
        "Pi0.5 AdaRMSNorm to NVFP4 using native E2M1x2 conversion.");

m.def("pi05_gate_res_adarms_fp4_sfa_native_fp16",
        [](uintptr_t x, uintptr_t prev_gate, uintptr_t residual,
           uintptr_t style, uintptr_t packed, uintptr_t sfa, uintptr_t gate,
           int seq_len, int dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape(
              {{"seq_len", seq_len}, {"dim", dim}});
          require_fp4_ptrs("pi05_gate_res_adarms_fp4_sfa_native_fp16",
                           {{"x", x}, {"prev_gate", prev_gate},
                            {"residual", residual}, {"style", style},
                            {"packed", packed}, {"sfa", sfa}, {"gate", gate}},
                           shape);
          require_fp4(seq_len == 10 && dim == 1024,
                      "pi05_gate_res_adarms_fp4_sfa_native_fp16",
                      "the Pi0.5 decoder business shape is seq_len=10, dim=1024",
                      shape);
          flash_rt::fused_fp4::pi05_gate_res_adarms_fp4_sfa_native_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(prev_gate),
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(style),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              reinterpret_cast<__half*>(gate), seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("prev_gate"), py::arg("residual"),
        py::arg("style"), py::arg("packed"), py::arg("sfa"), py::arg("gate"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "Pi0.5 gated residual + AdaRMSNorm with native E2M1x2 conversion.");
}
