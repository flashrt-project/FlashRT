// ============================================================================
//  FlashRT — pybind module for NVFP4 kernels.
//
//  Built as a SEPARATE .so from flash_rt_kernels.so (which stays untouched).
//  Python-side usage:
//
//      import flash_rt.flash_rt_kernels as fvk        # unchanged
//      import flash_rt.flash_rt_fp4    as fvk_fp4     # new, additive
//
//  All pointer args are passed as int (ctypes.c_void_p.value) to mirror the
//  existing fvk convention; everything is host/device pointer pass-through.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>

#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_fp4out.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#ifdef FLASHRT_HAVE_COSMOS3_EDGE
#include "gemm/fp4/cosmos3_edge_fp4_gemm_relu2_fp4out.cuh"
#endif
#include "quantize/quantize_fp4_dynamic.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "quantize/quantize_e0m3_sfa.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_e0m3w_sm100.cuh"
#include "fused_fp4/pi05_e0m3_act.cuh"
#include "fused_fp4/siglip_ln_vec.cuh"
#include "quantize/reshape_scales_sfa.cuh"
#include "fused_fp16/rms_norm_noweight_fp16.cuh"
#ifdef FLASHRT_HAVE_COSMOS3_EDGE
#include "fused_fp4/cosmos3_edge_fp4.cuh"
#endif
#include "fused_fp4/norm_silu_fp4_sfa.cuh"
#include "fused_fp4/silu_mul_two_fp4_to_fp4.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.cuh"
#include "quantize/quantize_fp4_sfa_bf16.cuh"
#include "fused_fp4/dit_norm_fp4_sfa.cuh"
#include "fused_fp4/silu_mul_fp4_sfa_bf16.cuh"
#include "fused_fp4/layer_norm_fp4_sfa.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cuh"

extern "C" int flash_rt_per_channel_mul_fp16(
    uintptr_t x, uintptr_t inv_s, int S, int D, uintptr_t stream);

namespace py = pybind11;

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

PYBIND11_MODULE(flash_rt_fp4, m) {
  m.doc() = "FlashRT — NVFP4 (Thor SM110) add-on kernels";

  // ── GEMM ──
  m.def("cutlass_fp4_sq_fp16",
        [](uintptr_t A, uintptr_t SFA,
           uintptr_t B, uintptr_t SFB,
           uintptr_t D, int M, int N, int K,
           float alpha, float beta, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_sq_fp16(
              reinterpret_cast<void const*>(A),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D),
              M, N, K, alpha, beta,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A"), py::arg("SFA"),
        py::arg("B"), py::arg("SFB"),
        py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("stream") = 0,
        R"pbdoc(
NVFP4 block-scaled GEMM:  D[M,N] (fp16) = A[M,K] (fp4) @ B[N,K]^T (fp4)

A and B are NVFP4 (e2m1) packed as 2 elements per byte, with per-16-element
UE4M3 block scales (SFA, SFB). All pointers are device-resident; int-typed
(e.g. t.data_ptr()). Returns 0 on success, nonzero on error.

NOTE: SFA/SFB must be in the CUTLASS Sm1xxBlkScaledConfig tile-interleaved
layout, NOT the linear [N, D/16] layout produced by quantize_fp4_dynamic_fp16.
Phase 4 will add the layout conversion helper.
)pbdoc");

  m.def("cutlass_fp4_gemm_bias_gelu_fp4out",
        [](uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t bias, uintptr_t D, uintptr_t D_SFD,
           int M, int N, int K, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_bias_gelu_fp4out(
              reinterpret_cast<void const*>(A),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void const*>(bias),
              reinterpret_cast<void*>(D),
              reinterpret_cast<void*>(D_SFD),
              M, N, K, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A"), py::arg("SFA"), py::arg("B"), py::arg("SFB"),
        py::arg("bias"), py::arg("D"), py::arg("D_SFD"),
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0,
        "NVFP4 GEMM, epilogue tanh-GELU(acc + bias[N]) with fp4+SFA output.");

  m.def("cutlass_fp4_gemm_bias_res_fp16",
        [](uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t bias, uintptr_t C, uintptr_t D,
           int M, int N, int K, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_bias_res_fp16(
              reinterpret_cast<void const*>(A),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void const*>(bias),
              reinterpret_cast<void const*>(C),
              reinterpret_cast<void*>(D),
              M, N, K, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A"), py::arg("SFA"), py::arg("B"), py::arg("SFB"),
        py::arg("bias"), py::arg("C"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0,
        "NVFP4 GEMM, epilogue acc + bias[N] + C (residual), fp16 output; "
        "C may alias D.");

  m.def("rms_norm_mul_fp4_sfa_fp16",
        [](uintptr_t x, uintptr_t inv_s, uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::rms_norm_mul_fp4_sfa_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("inv_s"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F2 + AWQ: weightless RMSNorm x per-channel inverse scale to "
        "NVFP4 + SFA.");

  m.def("layer_norm_fp8_vec_fp16",
        [](uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t out,
           int S, int D, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::layer_norm_fp8_vec_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(gamma),
              reinterpret_cast<const __half*>(beta),
              reinterpret_cast<void*>(out),
              S, D, eps, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("out"),
        py::arg("S"), py::arg("D"), py::arg("eps") = 1e-5f,
        py::arg("stream") = 0,
        "Vectorized LayerNorm to FP8 (register-resident single pass; "
        "reduction order differs from layer_norm_fp8 at ulp level).");

  m.def("layer_norm_mul_fp4_sfa_vec_fp16",
        [](uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t inv_s,
           uintptr_t packed, uintptr_t sfa,
           int S, int D, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::layer_norm_mul_fp4_sfa_vec_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(gamma),
              reinterpret_cast<const __half*>(beta),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              S, D, eps, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("inv_s"),
        py::arg("packed"), py::arg("sfa"),
        py::arg("S"), py::arg("D"), py::arg("eps") = 1e-5f,
        py::arg("stream") = 0,
        "Vectorized LayerNorm [x AWQ inv_s] to NVFP4+SFA (single pass).");

  m.def("layer_norm_mul_fp4_sfa_fp16",
        [](uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t inv_s,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::layer_norm_mul_fp4_sfa_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(gamma),
              reinterpret_cast<const __half*>(beta),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              seq_len, dim, eps,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("inv_s"),
        py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-5f,
        py::arg("stream") = 0,
        "Fused LayerNorm + per-channel AWQ inverse scale + NVFP4/SFA "
        "quantize (inv_s = 0 for the plain path).");

  // The plain (no inverse-scale) LayerNorm→NVFP4 variant is reachable
  // through layer_norm_mul_fp4_sfa_fp16 with inv_s = 0; it is deliberately
  // not exported a second time under its own name.

  // ── Dynamic quantize ──
  m.def("quantize_fp4_dynamic_fp16",
        [](uintptr_t src, uintptr_t packed, uintptr_t scales,
           int N, int D, uintptr_t stream) -> int {
          return flash_rt::fp4::quantize_fp4_dynamic_fp16(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(scales),
              N, D, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("scales"),
        py::arg("N"), py::arg("D"), py::arg("stream") = 0,
        R"pbdoc(
fp16 [N, D] → NVFP4 packed [N, D/2] uint8 + UE4M3 scales [N, D/16].
Linear (row-major) scale layout. For CUTLASS GEMM consumption, additional
tile-interleave conversion is required.
)pbdoc");

  m.def("dequantize_fp4_to_fp16",
        [](uintptr_t packed, uintptr_t scales, uintptr_t dst,
           int N, int D, uintptr_t stream) -> int {
          return flash_rt::fp4::dequantize_fp4_to_fp16(
              reinterpret_cast<void const*>(packed),
              reinterpret_cast<void const*>(scales),
              reinterpret_cast<void*>(dst),
              N, D, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("packed"), py::arg("scales"), py::arg("dst"),
        py::arg("N"), py::arg("D"), py::arg("stream") = 0,
        "Inverse of quantize_fp4_dynamic_fp16. Used for unit tests.");

  m.def("quantize_fp4_dynamic_sfa_mse_fp16",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"N", N}, {"D", D}});
          require_fp4_ptrs("quantize_fp4_dynamic_sfa_mse_fp16",
                           {{"src", src}, {"packed", packed}, {"sfa", sfa}}, shape);
          require_fp4(N > 0 && D > 0 && (D % 16) == 0,
                      "quantize_fp4_dynamic_sfa_mse_fp16",
                      "N must be positive and D must be a positive multiple of 16", shape);
          return flash_rt::fp4::quantize_fp4_dynamic_sfa_mse_fp16(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed), reinterpret_cast<void*>(sfa),
              N, D, is_sfb, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"), py::arg("stream") = 0,
        "FP16 to NVFP4 SFA/SFB quantization with per-block MSE scale search.");

  m.def("quantize_e0m3_dynamic_sfa_fp16",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"N", N}, {"D", D}});
          require_fp4_ptrs("quantize_e0m3_dynamic_sfa_fp16",
                           {{"src", src}, {"packed", packed}, {"sfa", sfa}}, shape);
          require_fp4(N > 0 && D > 0 && (D % 16) == 0,
                      "quantize_e0m3_dynamic_sfa_fp16",
                      "N must be positive and D must be a positive multiple of 16", shape);
          return flash_rt::fp4::quantize_e0m3_dynamic_sfa_fp16(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed), reinterpret_cast<void*>(sfa),
              N, D, is_sfb, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"), py::arg("stream") = 0,
        "FP16 to E0M3 (uniform INT4) quantization with per-16 UE4M3 SFA/SFB "
        "scales; packed/scale layouts match the NVFP4 quantizers.");

  m.def("quantize_fp4_dynamic_sfa_fp16",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, uintptr_t stream) -> int {
          return flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              N, D, is_sfb,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"), py::arg("stream") = 0,
        R"pbdoc(
Fused: fp16 [N, D] → NVFP4 packed [N, D/2] + CUTLASS tile-interleaved SFA/SFB.
Bit-exact equivalent of quantize_fp4_dynamic_fp16 followed by
reshape_linear_scales_to_sfa, in a single kernel launch.
)pbdoc");

  m.def("quantize_fp4_dynamic_sfa_fp16_vec",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, uintptr_t stream) -> int {
          return flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16_vec(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              N, D, is_sfb,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"), py::arg("stream") = 0,
        R"pbdoc(
Vectorized bit-exact variant of quantize_fp4_dynamic_sfa_fp16 (16B loads,
8B packed stores). Returns nonzero without launching on unaligned buffers;
callers fall back to the scalar kernel.
)pbdoc");

  m.def("reshape_linear_scales_to_sfa",
        [](uintptr_t src, uintptr_t dst, int rows, int D, bool is_sfb,
           uintptr_t stream) -> int {
          return flash_rt::fp4::reshape_linear_scales_to_sfa(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(dst),
              rows, D, is_sfb,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("dst"), py::arg("rows"), py::arg("D"),
        py::arg("is_sfb"), py::arg("stream") = 0,
        "Permute linear [rows, D/16] fp8 scales into CUTLASS SFA (is_sfb=False) "
        "or SFB (is_sfb=True) tile-interleaved layout.");

  m.def("sfa_size_bytes",
        &flash_rt::fp4::sfa_size_bytes,
        py::arg("rows"), py::arg("D"), py::arg("is_sfb"),
        "Byte size of the CUTLASS SFA (or SFB) buffer for the given problem.");

  // Tuning variants
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

  m.def("cutlass_fp4_gemm_e0m3w",
        [](uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t D, int M, int N, int K, float alpha, float beta,
           uintptr_t stream, int a_format) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_e0m3w(
              reinterpret_cast<void const*>(A), reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B), reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D), M, N, K, alpha, beta,
              reinterpret_cast<cudaStream_t>(stream), a_format);
        },
        py::arg("A"), py::arg("SFA"),
        py::arg("B"), py::arg("SFB"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("stream") = 0, py::arg("a_format") = 1,
        "Block-scaled GEMM with E0M3 (uniform INT4) weights via the SM110 "
        "runtime instruction descriptor (tile 128x64x256). a_format selects "
        "the activation element format: 1 = E2M1 (default), 0 = E0M3.");

  m.def("quantize_e0m3_dynamic_sfa_fp16_vec",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, int use_rht, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"N", N}, {"D", D}});
          require_fp4_ptrs("quantize_e0m3_dynamic_sfa_fp16_vec",
                           {{"src", src}, {"packed", packed}, {"sfa", sfa}}, shape);
          require_fp4(N > 0 && D > 0 && (D % 16) == 0,
                      "quantize_e0m3_dynamic_sfa_fp16_vec",
                      "N must be positive and D must be a positive multiple of 16", shape);
          return flash_rt::fp4::quantize_e0m3_dynamic_sfa_fp16_vec(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed), reinterpret_cast<void*>(sfa),
              N, D, is_sfb, use_rht, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"),
        py::arg("use_rht") = 0, py::arg("stream") = 0,
        "Vectorized FP16 to E0M3 activation quantize with optional per-16 "
        "Hadamard rotation.");

  m.def("pi05_adarms_e0m3_sfa_fp16",
        [](uintptr_t x, uintptr_t style, uintptr_t packed, uintptr_t sfa,
           uintptr_t gate, int S, int D, int use_rht, uintptr_t stream) {
          flash_rt::fused_fp4::pi05_adarms_e0m3_sfa_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(style),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              reinterpret_cast<__half*>(gate),
              S, D, use_rht, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("style"), py::arg("packed"), py::arg("sfa"),
        py::arg("gate"), py::arg("S"), py::arg("D"),
        py::arg("use_rht") = 0, py::arg("stream") = 0,
        "AdaRMS to E0M3 packed + SFA with optional per-16 Hadamard rotation.");

  m.def("pi05_gate_res_adarms_e0m3_sfa_fp16",
        [](uintptr_t x, uintptr_t prev_gate, uintptr_t residual,
           uintptr_t style, uintptr_t packed, uintptr_t sfa, uintptr_t gate,
           int S, int D, int use_rht, uintptr_t stream) {
          flash_rt::fused_fp4::pi05_gate_res_adarms_e0m3_sfa_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(prev_gate),
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(style),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              reinterpret_cast<__half*>(gate),
              S, D, use_rht, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("prev_gate"), py::arg("residual"),
        py::arg("style"), py::arg("packed"), py::arg("sfa"), py::arg("gate"),
        py::arg("S"), py::arg("D"),
        py::arg("use_rht") = 0, py::arg("stream") = 0,
        "Gated residual + AdaRMS to E0M3 packed + SFA with optional RHT.");

  m.def("gate_geglu_e0m3_sfa_vec_fp16",
        [](uintptr_t merged, uintptr_t packed, uintptr_t sfa,
           int S, int H, int use_rht, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::gate_geglu_e0m3_sfa_vec_fp16(
              reinterpret_cast<const __half*>(merged),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              S, H, use_rht, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("merged"), py::arg("packed"), py::arg("sfa"),
        py::arg("S"), py::arg("H"),
        py::arg("use_rht") = 0, py::arg("stream") = 0,
        "GeGLU to E0M3 packed + SFA with optional per-16 Hadamard rotation.");

  m.def("cutlass_fp4_gemm_variant_name", &flash_rt::fp4::cutlass_fp4_gemm_variant_name,
        py::arg("idx"), "Human-readable name of variant at index.");
  m.def("cutlass_fp4_gemm_num_variants", &flash_rt::fp4::cutlass_fp4_gemm_num_variants,
        "Count of available GEMM variants.");

  m.def("has_nvfp4", &flash_rt::fp4::has_nvfp4_sm110,
        "True iff this .so was built with CUTLASS SM100 support (NVFP4 usable).");

  // ── fp16-output fused norm kernels (additive, for FP4 frontend path) ──
  m.def("rms_norm_noweight_fp16",
        [](uintptr_t x, uintptr_t out, int seq_len, int dim,
           uintptr_t stream) {
          flash_rt::fused_fp16::rms_norm_noweight_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<__half*>(out),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
        py::arg("stream") = 0,
        "fp16 [S,D] → fp16 [S,D]. RMSNorm without weight, no descale. "
        "Bit-exact with rms_norm_fp8_noweight_fp16 when subsequently quantized "
        "to fp8 with the same descale factor.");

  m.def("residual_add_rms_norm_noweight_fp16",
        [](uintptr_t residual, uintptr_t x, uintptr_t out,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp16::residual_add_rms_norm_noweight_fp16(
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<__half*>(out),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("out"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "Residual += x (in place) then RMSNorm to fp16 (no descale). "
        "Bit-exact with residual_add_rms_norm_fp8_noweight_fp16 modulo fp8 cast.");

  // ── Fused FP4 pre-GEMM kernels (F2/F3/F4) ──
  m.def("rms_norm_fp4_sfa_fp16",
        [](uintptr_t x, uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::rms_norm_fp4_sfa_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F2: fused rms_norm + fp4_quant + SFA write.");

  m.def("residual_add_rms_norm_fp4_sfa_v2_fp16",
        [](uintptr_t residual, uintptr_t x,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::residual_add_rms_norm_fp4_sfa_v2_fp16(
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F3 v2 (register-only, 1 thread = 1 NVFP4 block): fused res+rms+fp4+SFA.");

  m.def("residual_add_rms_norm_fp4_sfa_fp16",
        [](uintptr_t residual, uintptr_t x,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::residual_add_rms_norm_fp4_sfa_fp16(
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F3: fused residual+rms_norm + fp4_quant + SFA write.");

  m.def("pi05_adarms_fp4_sfa_fp16",
        [](uintptr_t x, uintptr_t style, uintptr_t packed, uintptr_t sfa,
           uintptr_t gate, int seq_len, int dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape(
              {{"seq_len", seq_len}, {"dim", dim}});
          require_fp4_ptrs("pi05_adarms_fp4_sfa_fp16",
                           {{"x", x}, {"style", style}, {"packed", packed},
                            {"sfa", sfa}, {"gate", gate}}, shape);
          require_fp4(seq_len == 10 && dim == 1024,
                      "pi05_adarms_fp4_sfa_fp16",
                      "the Pi0.5 decoder business shape is seq_len=10, dim=1024",
                      shape);
          flash_rt::fused_fp4::pi05_adarms_fp4_sfa_fp16(
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
        "Pi0.5 AdaRMSNorm(style) to NVFP4 SFA plus fp16 gate.");

  m.def("pi05_gate_res_adarms_fp4_sfa_fp16",
        [](uintptr_t x, uintptr_t prev_gate, uintptr_t residual,
           uintptr_t style, uintptr_t packed, uintptr_t sfa, uintptr_t gate,
           int seq_len, int dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape(
              {{"seq_len", seq_len}, {"dim", dim}});
          require_fp4_ptrs("pi05_gate_res_adarms_fp4_sfa_fp16",
                           {{"x", x}, {"prev_gate", prev_gate},
                            {"residual", residual}, {"style", style},
                            {"packed", packed}, {"sfa", sfa}, {"gate", gate}},
                           shape);
          require_fp4(seq_len == 10 && dim == 1024,
                      "pi05_gate_res_adarms_fp4_sfa_fp16",
                      "the Pi0.5 decoder business shape is seq_len=10, dim=1024",
                      shape);
          flash_rt::fused_fp4::pi05_gate_res_adarms_fp4_sfa_fp16(
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
        "Pi0.5 gated residual + AdaRMSNorm to NVFP4 SFA plus next fp16 gate.");

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

#ifdef FLASHRT_HAVE_COSMOS3_EDGE
  // Cosmos3-Edge model-specific fused FP4 quant kernels (bf16 residual chain).
  m.def("cosmos3_edge_res_rms_fp4_sfa_bf16",
        [](uintptr_t residual, uintptr_t x, uintptr_t weight,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, float eps, uintptr_t stream) {
          const auto shape = fp4_kernel_shape({{"seq_len", seq_len}, {"dim", dim}});
          require_fp4_ptrs("cosmos3_edge_res_rms_fp4_sfa_bf16",
              {{"residual", residual}, {"x", x}, {"weight", weight},
               {"packed", packed}, {"sfa", sfa}}, shape);
          require_fp4(seq_len > 0 && dim > 0 && dim <= 16384 && (dim % 16) == 0,
                      "cosmos3_edge_res_rms_fp4_sfa_bf16",
                      "seq_len must be positive and dim must be a multiple of 16 in [16, 16384]",
                      shape);
          require_fp4(std::isfinite(eps) && eps > 0.0f,
                      "cosmos3_edge_res_rms_fp4_sfa_bf16",
                      "eps must be finite and positive", shape);
          flash_rt::fused_fp4::cosmos3_edge_res_rms_fp4_sfa_bf16(
              reinterpret_cast<__nv_bfloat16*>(residual),
              reinterpret_cast<const __nv_bfloat16*>(x),
              reinterpret_cast<const __nv_bfloat16*>(weight),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim, eps,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("weight"),
        py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("eps"), py::arg("stream") = 0,
        "Cosmos3-Edge: bf16 residual += x; weighted RMSNorm; NVFP4 quant + SFA.");

  m.def("cosmos3_edge_relu2_fp4_sfa_fp16",
        [](uintptr_t x, uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape({{"seq_len", seq_len}, {"dim", dim}});
          require_fp4_ptrs("cosmos3_edge_relu2_fp4_sfa_fp16",
                           {{"x", x}, {"packed", packed}, {"sfa", sfa}}, shape);
          require_fp4(seq_len > 0 && dim > 0 && (dim % 16) == 0,
                      "cosmos3_edge_relu2_fp4_sfa_fp16",
                      "seq_len must be positive and dim must be a positive multiple of 16",
                      shape);
          flash_rt::fused_fp4::cosmos3_edge_relu2_fp4_sfa_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "Cosmos3-Edge: relu(x)^2 (fp16 in) -> NVFP4 quant + SFA.");
#endif

  // GEGLU (tanh-approx GELU(gate) * up) fused FP4 kernels.
  m.def("gate_geglu_fp4_sfa_fp16",
        [](uintptr_t merged, uintptr_t packed, uintptr_t sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_fp16(
              reinterpret_cast<const __half*>(merged),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("merged"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "F4 v1 (smem-staged): fused GEGLU + fp4_quant + SFA write.");

  m.def("gate_geglu_fp4_sfa_v2_fp16",
        [](uintptr_t merged, uintptr_t packed, uintptr_t sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_v2_fp16(
              reinterpret_cast<const __half*>(merged),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("merged"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "F4 v2 (register-only, no smem): same semantics as v1, faster at H=8192.");

  m.def("gate_geglu_fp4_sfa_vec_fp16",
        [](uintptr_t merged, uintptr_t packed, uintptr_t sfa,
           int seq_len, int half_dim, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_vec_fp16(
              reinterpret_cast<const __half*>(merged),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("merged"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "Vectorized bit-exact variant of gate_geglu_fp4_sfa_v2_fp16; "
        "returns nonzero on unaligned buffers without launching.");

  // ── AWQ fused: F3 + per-channel-mul ──
  m.def("residual_add_rms_norm_mul_fp4_sfa_fp16",
        [](uintptr_t residual, uintptr_t x, uintptr_t inv_s,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::residual_add_rms_norm_mul_fp4_sfa_fp16(
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("inv_s"),
        py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F3 + AWQ: fused res+rms+inv_s_mul+fp4_quant+SFA (1 launch).");

  // ── AWQ fused: F4 v2 + per-channel-mul ──
  m.def("gate_geglu_mul_fp4_sfa_v2_fp16",
        [](uintptr_t merged, uintptr_t inv_s,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          flash_rt::fused_fp4::gate_silu_mul_mul_fp4_sfa_v2_fp16(
              reinterpret_cast<const __half*>(merged),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("merged"), py::arg("inv_s"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "F4 v2 + AWQ: fused GEGLU+inv_s_mul+fp4_quant+SFA (1 launch).");

  // ── AWQ per-channel inverse scale multiply ──
  m.def("per_channel_mul_fp16",
        [](uintptr_t x, uintptr_t inv_s, int S, int D, uintptr_t stream) {
          flash_rt_per_channel_mul_fp16(x, inv_s, S, D, stream);
        },
        py::arg("x"), py::arg("inv_s"), py::arg("S"), py::arg("D"),
        py::arg("stream") = 0,
        "x[i,k] *= inv_s[k] in-place. Per-input-channel activation scaling "
        "for AWQ-style FP4 inference (paired with offline weight pre-scaling).");

  // ── P1: NVFP4 GEMM with FP4 packed output (LinCombBlockScaleFactor epilogue) ──
  m.def("cutlass_fp4_gemm_fp4out",
        [](uintptr_t A_packed, uintptr_t SFA,
           uintptr_t B_packed, uintptr_t SFB,
           uintptr_t D_packed, uintptr_t D_SFD,
           int M, int N, int K, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"M", M}, {"N", N}, {"K", K}});
          require_fp4_ptrs("cutlass_fp4_gemm_fp4out",
                           {{"A_packed", A_packed}, {"SFA", SFA},
                            {"B_packed", B_packed}, {"SFB", SFB},
                            {"D_packed", D_packed}, {"D_SFD", D_SFD}}, shape);
          require_fp4(M > 0 && N > 0 && K > 0 && (N % 16) == 0 && (K % 16) == 0,
                      "cutlass_fp4_gemm_fp4out",
                      "M must be positive and N/K must be positive multiples of 16",
                      shape);
          return flash_rt::fp4::cutlass_fp4_gemm_fp4out(
              reinterpret_cast<void const*>(A_packed),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B_packed),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D_packed),
              reinterpret_cast<void*>(D_SFD),
              M, N, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A_packed"), py::arg("SFA"),
        py::arg("B_packed"), py::arg("SFB"),
        py::arg("D_packed"), py::arg("D_SFD"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0,
        R"pbdoc(
P1 NVFP4 GEMM:  D[M,N/2] (fp4) + D_SFD = LinCombBlockScaleFactor(A @ B^T).
Drop-in for cutlass_fp4_sq_fp16 when downstream consumes FP4 + SFA directly.
)pbdoc");

  m.def("cutlass_fp4_gemm_geglu_il",
        [](uintptr_t A_packed, uintptr_t SFA,
           uintptr_t B_packed, uintptr_t SFB,
           uintptr_t D_packed, uintptr_t D_SFD,
           int M, int N_il, int K, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"M", M}, {"N_il", N_il}, {"K", K}});
          require_fp4_ptrs("cutlass_fp4_gemm_geglu_il",
                           {{"A_packed", A_packed}, {"SFA", SFA},
                            {"B_packed", B_packed}, {"SFB", SFB},
                            {"D_packed", D_packed}, {"D_SFD", D_SFD}}, shape);
          require_fp4(M > 0 && N_il > 0 && K > 0 && (N_il % 32) == 0 &&
                      (K % 16) == 0,
                      "cutlass_fp4_gemm_geglu_il",
                      "M must be positive, N_il a positive multiple of 32 "
                      "and K a positive multiple of 16",
                      shape);
          return flash_rt::fp4::cutlass_fp4_gemm_geglu_il(
              reinterpret_cast<void const*>(A_packed),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B_packed),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D_packed),
              reinterpret_cast<void*>(D_SFD),
              M, N_il, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A_packed"), py::arg("SFA"),
        py::arg("B_packed"), py::arg("SFB"),
        py::arg("D_packed"), py::arg("D_SFD"),
        py::arg("M"), py::arg("N_il"), py::arg("K"),
        py::arg("stream") = 0,
        R"pbdoc(
P1 NVFP4 GEMM with fused GeGLU epilogue over a column-interleaved gate/up
weight (B_il[2j] = gate[j], B_il[2j+1] = up[j]). D[M, N_il] (fp4) + D_SFD hold
gelu(gate)*up duplicated into both columns of each pair; replaces the
gate GEMM + up GEMM + combiner chain.
)pbdoc");

  m.def("cutlass_fp4_gemm_geglu_il_hw",
        [](uintptr_t A_packed, uintptr_t SFA,
           uintptr_t B_packed, uintptr_t SFB,
           uintptr_t D_dummy, uintptr_t compact_packed, uintptr_t compact_sfa,
           int M, int N_il, int K, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"M", M}, {"N_il", N_il}, {"K", K}});
          require_fp4_ptrs("cutlass_fp4_gemm_geglu_il_hw",
                           {{"A_packed", A_packed}, {"SFA", SFA},
                            {"B_packed", B_packed}, {"SFB", SFB},
                            {"D_dummy", D_dummy},
                            {"compact_packed", compact_packed},
                            {"compact_sfa", compact_sfa}}, shape);
          require_fp4(M > 0 && N_il > 0 && K > 0 && (N_il % 32) == 0 &&
                      (K % 16) == 0,
                      "cutlass_fp4_gemm_geglu_il_hw",
                      "M must be positive, N_il a positive multiple of 32 "
                      "and K a positive multiple of 16",
                      shape);
          return flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw(
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
Half-width fused GeGLU interleaved GEMM: the epilogue quantizes
gelu(gate)*up at compact granularity and writes compact_packed
[M, N_il/2] + compact_sfa (SFA tile-atom layout) directly; D_dummy
[M, N_il] receives zeros and may be one buffer shared across layers.
The downstream GEMM keeps its original K = N_il/2 weight.
)pbdoc");

  m.def("cutlass_fp4_gemm_geglu_il_hw_v10",
        [](uintptr_t A_packed, uintptr_t SFA,
           uintptr_t B_packed, uintptr_t SFB,
           uintptr_t D_dummy, uintptr_t compact_packed, uintptr_t compact_sfa,
           int M, int N_il, int K, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"M", M}, {"N_il", N_il}, {"K", K}});
          require_fp4_ptrs("cutlass_fp4_gemm_geglu_il_hw_v10",
                           {{"A_packed", A_packed}, {"SFA", SFA},
                            {"B_packed", B_packed}, {"SFB", SFB},
                            {"D_dummy", D_dummy},
                            {"compact_packed", compact_packed},
                            {"compact_sfa", compact_sfa}}, shape);
          require_fp4(M > 0 && N_il > 0 && K > 0 && (N_il % 32) == 0 &&
                      (K % 16) == 0,
                      "cutlass_fp4_gemm_geglu_il_hw_v10",
                      "M must be positive, N_il a positive multiple of 32 "
                      "and K a positive multiple of 16",
                      shape);
          return flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_v10(
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
Skinny-M half-width fused GeGLU GEMM on the decoder tile (128x64x256);
same contract as cutlass_fp4_gemm_geglu_il_hw.
)pbdoc");

#ifdef FLASHRT_HAVE_COSMOS3_EDGE
  m.def("cosmos3_edge_fp4_gemm_relu2_fp4out",
        [](uintptr_t A_packed, uintptr_t SFA,
           uintptr_t B_packed, uintptr_t SFB,
           uintptr_t D_packed, uintptr_t D_SFD,
           int M, int N, int K, uintptr_t stream) -> int {
          const auto shape = fp4_kernel_shape({{"M", M}, {"N", N}, {"K", K}});
          require_fp4_ptrs("cosmos3_edge_fp4_gemm_relu2_fp4out",
              {{"A_packed", A_packed}, {"SFA", SFA}, {"B_packed", B_packed},
               {"SFB", SFB}, {"D_packed", D_packed}, {"D_SFD", D_SFD}}, shape);
          require_fp4(M > 0 && N > 0 && K > 0 && (N % 16) == 0 && (K % 16) == 0,
                      "cosmos3_edge_fp4_gemm_relu2_fp4out",
                      "M must be positive and N/K must be positive multiples of 16",
                      shape);
          return flash_rt::fp4::cosmos3_edge_fp4_gemm_relu2_fp4out(
              reinterpret_cast<void const*>(A_packed),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B_packed),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D_packed),
              reinterpret_cast<void*>(D_SFD), M, N, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A_packed"), py::arg("SFA"),
        py::arg("B_packed"), py::arg("SFB"),
        py::arg("D_packed"), py::arg("D_SFD"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0,
        "Cosmos3-Edge fused NVFP4 up GEMM + ReLU squared + FP4 quantization.");
#endif

  // ── P1 + AWQ: geglu_two_mul_fp4_to_fp4 — GEGLU combiner with Down inv_s mul ──
  m.def("geglu_two_mul_fp4_to_fp4",
        [](uintptr_t gate_packed, uintptr_t gate_sfa,
           uintptr_t up_packed,   uintptr_t up_sfa,
           uintptr_t inv_s,
           uintptr_t out_packed,  uintptr_t out_sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape({{"seq_len", seq_len},
                                                {"half_dim", half_dim}});
          require_fp4_ptrs("geglu_two_mul_fp4_to_fp4",
                           {{"gate_packed", gate_packed}, {"gate_sfa", gate_sfa},
                            {"up_packed", up_packed}, {"up_sfa", up_sfa},
                            {"inv_s", inv_s}, {"out_packed", out_packed},
                            {"out_sfa", out_sfa}}, shape);
          require_fp4(seq_len > 0 && half_dim > 0 && (half_dim % 16) == 0,
                      "geglu_two_mul_fp4_to_fp4",
                      "seq_len must be positive and half_dim must be a positive multiple of 16",
                      shape);
          flash_rt::fused_fp4::silu_mul_two_mul_fp4_to_fp4(
              reinterpret_cast<const uint8_t*>(gate_packed),
              reinterpret_cast<const uint8_t*>(gate_sfa),
              reinterpret_cast<const uint8_t*>(up_packed),
              reinterpret_cast<const uint8_t*>(up_sfa),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(out_packed),
              reinterpret_cast<uint8_t*>(out_sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("gate_packed"), py::arg("gate_sfa"),
        py::arg("up_packed"),   py::arg("up_sfa"),
        py::arg("inv_s"),
        py::arg("out_packed"),  py::arg("out_sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "P1 + AWQ-Down: GEGLU + per-input-channel inv_s mul → FP4 + SFA.");

  m.def("geglu_two_mul_fp4_to_fp4_lut",
        [](uintptr_t gate_packed, uintptr_t gate_sfa,
           uintptr_t up_packed,   uintptr_t up_sfa,
           uintptr_t inv_s,
           uintptr_t out_packed,  uintptr_t out_sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape({{"seq_len", seq_len},
                                                {"half_dim", half_dim}});
          require_fp4_ptrs("geglu_two_mul_fp4_to_fp4_lut",
                           {{"gate_packed", gate_packed}, {"gate_sfa", gate_sfa},
                            {"up_packed", up_packed}, {"up_sfa", up_sfa},
                            {"inv_s", inv_s}, {"out_packed", out_packed},
                            {"out_sfa", out_sfa}}, shape);
          require_fp4(seq_len > 0 && half_dim > 0 && (half_dim % 16) == 0,
                      "geglu_two_mul_fp4_to_fp4_lut",
                      "seq_len must be positive and half_dim must be a positive multiple of 16",
                      shape);
          flash_rt::fused_fp4::silu_mul_two_mul_fp4_to_fp4_lut(
              reinterpret_cast<const uint8_t*>(gate_packed),
              reinterpret_cast<const uint8_t*>(gate_sfa),
              reinterpret_cast<const uint8_t*>(up_packed),
              reinterpret_cast<const uint8_t*>(up_sfa),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(out_packed),
              reinterpret_cast<uint8_t*>(out_sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("gate_packed"), py::arg("gate_sfa"),
        py::arg("up_packed"),   py::arg("up_sfa"),
        py::arg("inv_s"),
        py::arg("out_packed"),  py::arg("out_sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "P1 + AWQ-Down: explicit gate-LUT GEGLU combiner to FP4 + SFA.");

  m.def("geglu_two_mul_fp4_to_fp4_lut_native",
        [](uintptr_t gate_packed, uintptr_t gate_sfa,
           uintptr_t up_packed,   uintptr_t up_sfa,
           uintptr_t inv_s,
           uintptr_t out_packed,  uintptr_t out_sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape({{"seq_len", seq_len},
                                                {"half_dim", half_dim}});
          require_fp4_ptrs("geglu_two_mul_fp4_to_fp4_lut_native",
                           {{"gate_packed", gate_packed}, {"gate_sfa", gate_sfa},
                            {"up_packed", up_packed}, {"up_sfa", up_sfa},
                            {"inv_s", inv_s}, {"out_packed", out_packed},
                            {"out_sfa", out_sfa}}, shape);
          require_fp4(seq_len > 0 && half_dim > 0 && (half_dim % 16) == 0,
                      "geglu_two_mul_fp4_to_fp4_lut_native",
                      "seq_len must be positive and half_dim must be a positive multiple of 16",
                      shape);
          flash_rt::fused_fp4::silu_mul_two_mul_fp4_to_fp4_lut_native(
              reinterpret_cast<const uint8_t*>(gate_packed),
              reinterpret_cast<const uint8_t*>(gate_sfa),
              reinterpret_cast<const uint8_t*>(up_packed),
              reinterpret_cast<const uint8_t*>(up_sfa),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(out_packed),
              reinterpret_cast<uint8_t*>(out_sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("gate_packed"), py::arg("gate_sfa"),
        py::arg("up_packed"),   py::arg("up_sfa"),
        py::arg("inv_s"),
        py::arg("out_packed"),  py::arg("out_sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "P1 + AWQ-Down: gate-LUT GEGLU with native SM110 FP4 conversion.");

  // ── P1: geglu_two_fp4_to_fp4 — combiner kernel for split-GU FFN path ──
  m.def("geglu_two_fp4_to_fp4",
        [](uintptr_t gate_packed, uintptr_t gate_sfa,
           uintptr_t up_packed,   uintptr_t up_sfa,
           uintptr_t out_packed,  uintptr_t out_sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          const auto shape = fp4_kernel_shape({{"seq_len", seq_len},
                                                {"half_dim", half_dim}});
          require_fp4_ptrs("geglu_two_fp4_to_fp4",
                           {{"gate_packed", gate_packed}, {"gate_sfa", gate_sfa},
                            {"up_packed", up_packed}, {"up_sfa", up_sfa},
                            {"out_packed", out_packed}, {"out_sfa", out_sfa}}, shape);
          require_fp4(seq_len > 0 && half_dim > 0 && (half_dim % 16) == 0,
                      "geglu_two_fp4_to_fp4",
                      "seq_len must be positive and half_dim must be a positive multiple of 16",
                      shape);
          flash_rt::fused_fp4::silu_mul_two_fp4_to_fp4(
              reinterpret_cast<const uint8_t*>(gate_packed),
              reinterpret_cast<const uint8_t*>(gate_sfa),
              reinterpret_cast<const uint8_t*>(up_packed),
              reinterpret_cast<const uint8_t*>(up_sfa),
              reinterpret_cast<uint8_t*>(out_packed),
              reinterpret_cast<uint8_t*>(out_sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("gate_packed"), py::arg("gate_sfa"),
        py::arg("up_packed"),   py::arg("up_sfa"),
        py::arg("out_packed"),  py::arg("out_sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "P1: GEGLU over two FP4 inputs → FP4 + SFA.");

  // ── bf16-activation NVFP4 path (GR00T N1.7 DiT) ─────────────────────
  m.def("quantize_fp4_dynamic_sfa_bf16_vec",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, uintptr_t stream) -> int {
          return flash_rt::fp4::quantize_fp4_dynamic_sfa_bf16_vec(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              N, D, is_sfb,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"), py::arg("stream") = 0,
        "Fused: bf16 [N, D] -> NVFP4 packed [N, D/2] + tile-interleaved "
        "SFA/SFB (vectorized).");

  m.def("cutlass_fp4_gemm_bias_bf16",
        [](uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t bias, uintptr_t D, int M, int N, int K,
           uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_bias_bf16(
              reinterpret_cast<void const*>(A),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void const*>(bias),
              reinterpret_cast<void*>(D), M, N, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A"), py::arg("SFA"), py::arg("B"), py::arg("SFB"),
        py::arg("bias"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0,
        "NVFP4 GEMM, bf16 out: D = A @ B^T + bias[N].");

  m.def("cutlass_fp4_gemm_bias_res_bf16",
        [](uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t bias, uintptr_t C, uintptr_t D,
           int M, int N, int K, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_bias_res_bf16(
              reinterpret_cast<void const*>(A),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void const*>(bias),
              reinterpret_cast<void const*>(C),
              reinterpret_cast<void*>(D), M, N, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A"), py::arg("SFA"), py::arg("B"), py::arg("SFB"),
        py::arg("bias"), py::arg("C"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0,
        "NVFP4 GEMM, bf16 out with residual: D = A @ B^T + bias[N] + C "
        "(C may alias D).");

  m.def("cutlass_fp4_gemm_bias_gelu_fp4out_bf16",
        [](uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t bias, uintptr_t D_packed, uintptr_t D_SFD,
           int M, int N, int K, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_bias_gelu_fp4out_bf16(
              reinterpret_cast<void const*>(A),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void const*>(bias),
              reinterpret_cast<void*>(D_packed),
              reinterpret_cast<void*>(D_SFD), M, N, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A"), py::arg("SFA"), py::arg("B"), py::arg("SFB"),
        py::arg("bias"), py::arg("D_packed"), py::arg("D_SFD"),
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0,
        "NVFP4 GEMM with fused bias + tanh-GELU + FP4/SFA output "
        "(bf16 bias).");

  m.def("ada_layer_norm_fp4_sfa_bf16",
        [](uintptr_t x, uintptr_t scale, uintptr_t shift,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::ada_layer_norm_fp4_sfa_bf16(
              reinterpret_cast<void const*>(x),
              reinterpret_cast<void const*>(scale),
              reinterpret_cast<void const*>(shift),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              seq_len, dim, eps,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("scale"), py::arg("shift"),
        py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-5f,
        py::arg("stream") = 0,
        "Fused AdaLN (bf16) -> NVFP4 packed + SFA.");

  m.def("layer_norm_no_affine_fp4_sfa_bf16",
        [](uintptr_t x, uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::layer_norm_no_affine_fp4_sfa_bf16(
              reinterpret_cast<void const*>(x),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              seq_len, dim, eps,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-5f,
        py::arg("stream") = 0,
        "Fused no-affine LayerNorm (bf16) -> NVFP4 packed + SFA.");

  m.def("rms_norm_weight_fp4_sfa_bf16",
        [](uintptr_t x, uintptr_t weight, uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::rms_norm_weight_fp4_sfa_bf16(
              reinterpret_cast<void const*>(x),
              reinterpret_cast<void const*>(weight),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              seq_len, dim, eps,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("weight"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-5f,
        py::arg("stream") = 0,
        "Fused weighted RMSNorm (bf16) -> NVFP4 packed + SFA.");

  m.def("silu_mul_fp4_sfa_bf16",
        [](uintptr_t gate, uintptr_t up, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, uintptr_t stream) -> int {
          return flash_rt::fused_fp4::silu_mul_fp4_sfa_bf16(
              reinterpret_cast<void const*>(gate),
              reinterpret_cast<void const*>(up),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              N, D, is_sfb,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("gate"), py::arg("up"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"), py::arg("stream") = 0,
        "Fused SiLU(gate)*up (bf16) -> NVFP4 packed + SFA.");

  m.attr("__version__") = "0.1.0-dev";
  m.attr("layout_note") = "scales are linear [N, D/16]; Phase 4 adds tile-interleave conversion";
}
