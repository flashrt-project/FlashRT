// ================================================================
// FlashRT AMD -- pybind11 bindings for RDNA 3.5 BF16 kernels
//
// The Python-facing names include `_rdna` so a caller cannot accidentally
// route an RDNA implementation through a CDNA capability check. Like the
// existing AMD module, this keeps a raw-pointer ABI and never accepts tensors.
// Tensor dtype, shape, contiguity, device, and storage lifetime are validated
// by `flash_rt.amd.hardware.rdna35` before entering this module. The binding
// layer only converts integer device addresses/stream handles and forwards to
// HIP, keeping it independent of the PyTorch C++ ABI.
// ================================================================

#include <pybind11/pybind11.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include "gemm/hipblaslt_runner_rdna.h"

namespace py = pybind11;

// ── Raw-pointer conversion helpers ──

template <typename T> static T *typed_ptr(uintptr_t address) {
  return reinterpret_cast<T *>(address);
}

template <typename T> static const T *const_typed_ptr(uintptr_t address) {
  return reinterpret_cast<const T *>(address);
}

static hipStream_t to_stream(uintptr_t address) {
  return reinterpret_cast<hipStream_t>(address);
}

static void *to_ptr(uintptr_t address) {
  return reinterpret_cast<void *>(address);
}

// ── Kernel declarations ──
// Definitions live beside their CDNA counterparts under kernels/, attention/,
// and gemm/. Only this reduced BF16 surface is built for gfx1151.

void qkv_rope_rdna(const __hip_bfloat16 *qkv, const __hip_bfloat16 *rope_cos,
                   const __hip_bfloat16 *rope_sin, __hip_bfloat16 *q_out,
                   __hip_bfloat16 *k_out, __hip_bfloat16 *v_out, int rows,
                   int position_start, hipStream_t stream);

void gelu_rdna(const __hip_bfloat16 *input, __hip_bfloat16 *output, int size,
               hipStream_t stream);
void gelu_mul_rdna(const __hip_bfloat16 *gate, const __hip_bfloat16 *up,
                   __hip_bfloat16 *output, int size, hipStream_t stream);
void gelu_mul_merged_rdna(const __hip_bfloat16 *gate_up, __hip_bfloat16 *output,
                          int rows, int width, hipStream_t stream);
void silu_rdna(const __hip_bfloat16 *input, __hip_bfloat16 *output, int size,
               hipStream_t stream);
void residual_rdna(const __hip_bfloat16 *update, const __hip_bfloat16 *residual,
                   const __hip_bfloat16 *gate, __hip_bfloat16 *output, int size,
                   int width, hipStream_t stream);
void layer_norm_rdna(const __hip_bfloat16 *input, const __hip_bfloat16 *weight,
                     const __hip_bfloat16 *bias, __hip_bfloat16 *output,
                     int rows, int width, float epsilon, hipStream_t stream);
void rms_norm_rdna(const __hip_bfloat16 *input, __hip_bfloat16 *output,
                   int rows, int width, float epsilon, hipStream_t stream);
void adarms_rdna(const __hip_bfloat16 *input, const __hip_bfloat16 *modulation,
                 __hip_bfloat16 *output, int rows, int width, float epsilon,
                 hipStream_t stream);
void residual_rms_rdna(const __hip_bfloat16 *update,
                       const __hip_bfloat16 *residual,
                       __hip_bfloat16 *output_sum, __hip_bfloat16 *output_norm,
                       int rows, int width, float epsilon, hipStream_t stream);
void residual_adarms_rdna(const __hip_bfloat16 *update,
                          const __hip_bfloat16 *residual,
                          const __hip_bfloat16 *gate,
                          const __hip_bfloat16 *modulation,
                          __hip_bfloat16 *output_sum,
                          __hip_bfloat16 *output_norm, int rows, int width,
                          float epsilon, hipStream_t stream);
void attention_decoder_gqa_rdna(const __hip_bfloat16 *query,
                                const __hip_bfloat16 *key,
                                const __hip_bfloat16 *value,
                                __hip_bfloat16 *output, int query_rows,
                                int kv_rows, int query_heads, int head_dim,
                                int valid_prefix, int suffix_start, float scale,
                                hipStream_t stream);
void attention_decoder_gqa_splitkey_rdna(
    const __hip_bfloat16 *query, const __hip_bfloat16 *key,
    const __hip_bfloat16 *value, __hip_bfloat16 *output, int query_rows,
    int kv_rows, int query_heads, int head_dim, int valid_prefix,
    int suffix_start, float scale, int keys_per_iteration, hipStream_t stream);
void attention_encoder_gqa_rdna(const __hip_bfloat16 *query,
                                const __hip_bfloat16 *key,
                                const __hip_bfloat16 *value,
                                __hip_bfloat16 *output, int query_rows,
                                int valid_kv_rows, int query_heads,
                                int head_dim, hipStream_t stream);
void smallm_wmma_bf16_rdna(const __hip_bfloat16 *input,
                           const __hip_bfloat16 *weight_nt,
                           const __hip_bfloat16 *bias, __hip_bfloat16 *output,
                           int rows, int columns, int inner,
                           hipStream_t stream);
void smallm_wmma_bf16_residual_rdna(const __hip_bfloat16 *input,
                                    const __hip_bfloat16 *weight_nt,
                                    const __hip_bfloat16 *bias,
                                    __hip_bfloat16 *output, int rows,
                                    int columns, int inner, hipStream_t stream);

PYBIND11_MODULE(flash_rt_amd_kernels, module) {
  module.doc() = "FlashRT AMD RDNA 3.5 BF16 kernels (raw-pointer ABI)";

  // ── Runtime/build identity ──

  module.def("build_info", []() {
    py::dict info;
    info["platform"] = "hip";
    info["backend"] = "rdna35";
    info["wave_size"] = 32;
#ifdef FLASHRT_AMD_GPU_ARCH
    info["gpu_arch"] = FLASHRT_AMD_GPU_ARCH;
#endif
    int runtime_version = 0;
    (void)hipRuntimeGetVersion(&runtime_version);
    info["hip_runtime_version"] = runtime_version;
    return info;
  });

  module.def("device_arch", []() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
      return std::string("none");
    }
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) != hipSuccess) {
      return std::string("unknown");
    }
    return std::string(properties.gcnArchName);
  });

  // ── Stateful hipBLASLt runner ──

  py::class_<RdnaGemmRunner>(module, "RdnaGemmRunner")
      .def(py::init<>())
      .def("enable_lazy_autotune",
           &RdnaGemmRunner::enable_lazy_autotune,
           py::arg("num_algorithms") = 16)
      .def("bf16_nn",
           [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
              uintptr_t output, int rows, int columns, int inner,
              int weight_stride, int output_stride, uintptr_t stream) {
             self.bf16_nn(to_ptr(input), to_ptr(weight), to_ptr(output), rows,
                          columns, inner, weight_stride, output_stride,
                          to_stream(stream));
           },
           py::arg("input"), py::arg("weight"), py::arg("output"),
           py::arg("rows"), py::arg("columns"), py::arg("inner"),
           py::arg("weight_stride"),
           py::arg("output_stride"),
           py::arg("stream") = 0)
      .def("bf16_nn_bias",
           [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
              uintptr_t output, uintptr_t bias, int rows, int columns,
              int inner, int weight_stride, int output_stride,
              uintptr_t stream) {
             self.bf16_nn_bias(to_ptr(input), to_ptr(weight), to_ptr(output),
                               to_ptr(bias), rows, columns, inner,
                               weight_stride, output_stride, to_stream(stream));
           },
           py::arg("input"), py::arg("weight"), py::arg("output"),
           py::arg("bias"), py::arg("rows"), py::arg("columns"),
           py::arg("inner"), py::arg("weight_stride"),
           py::arg("output_stride"),
           py::arg("stream") = 0)
      .def("autotune_bf16_nn",
           [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
              uintptr_t output, int rows, int columns, int inner,
              int weight_stride, int output_stride, int num_algorithms,
              uintptr_t stream) {
             self.autotune_bf16_nn(to_ptr(input), to_ptr(weight),
                                   to_ptr(output), rows, columns, inner,
                                   weight_stride, output_stride, num_algorithms,
                                   to_stream(stream));
           },
           py::arg("input"), py::arg("weight"), py::arg("output"),
           py::arg("rows"), py::arg("columns"), py::arg("inner"),
           py::arg("weight_stride"),
           py::arg("output_stride"),
           py::arg("num_algorithms") = 16, py::arg("stream") = 0)
      .def("autotune_bf16_nn_bias",
           [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
              uintptr_t output, uintptr_t bias, int rows, int columns,
              int inner, int weight_stride, int output_stride,
              int num_algorithms,
              uintptr_t stream) {
             self.autotune_bf16_nn_bias(
                 to_ptr(input), to_ptr(weight), to_ptr(output), to_ptr(bias),
                 rows, columns, inner, weight_stride, output_stride,
                 num_algorithms, to_stream(stream));
           },
           py::arg("input"), py::arg("weight"), py::arg("output"),
           py::arg("bias"), py::arg("rows"), py::arg("columns"),
           py::arg("inner"), py::arg("weight_stride"),
           py::arg("output_stride"),
           py::arg("num_algorithms") = 16,
           py::arg("stream") = 0);

  // ── Stateless native kernels ──

  module.def("qkv_rope_rdna",
             [](uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
                uintptr_t qkv, uintptr_t rope_cos, uintptr_t rope_sin, int rows,
                int position_start, uintptr_t stream) {
               qkv_rope_rdna(const_typed_ptr<__hip_bfloat16>(qkv),
                             const_typed_ptr<__hip_bfloat16>(rope_cos),
                             const_typed_ptr<__hip_bfloat16>(rope_sin),
                             typed_ptr<__hip_bfloat16>(q_out),
                             typed_ptr<__hip_bfloat16>(k_out),
                             typed_ptr<__hip_bfloat16>(v_out), rows,
                             position_start, to_stream(stream));
             });

  module.def("layer_norm_rdna", [](uintptr_t output, uintptr_t input,
                                   uintptr_t weight, uintptr_t bias, int rows,
                                   int width, float epsilon, uintptr_t stream) {
    layer_norm_rdna(const_typed_ptr<__hip_bfloat16>(input),
                    const_typed_ptr<__hip_bfloat16>(weight),
                    const_typed_ptr<__hip_bfloat16>(bias),
                    typed_ptr<__hip_bfloat16>(output), rows, width, epsilon,
                    to_stream(stream));
  });

  module.def("rms_norm_rdna", [](uintptr_t output, uintptr_t input, int rows,
                                 int width, float epsilon, uintptr_t stream) {
    rms_norm_rdna(const_typed_ptr<__hip_bfloat16>(input),
                  typed_ptr<__hip_bfloat16>(output), rows, width, epsilon,
                  to_stream(stream));
  });

  module.def("adarms_rdna",
             [](uintptr_t output, uintptr_t input, uintptr_t modulation,
                int rows, int width, float epsilon, uintptr_t stream) {
               adarms_rdna(const_typed_ptr<__hip_bfloat16>(input),
                           const_typed_ptr<__hip_bfloat16>(modulation),
                           typed_ptr<__hip_bfloat16>(output), rows, width,
                           epsilon, to_stream(stream));
             });

  module.def("gelu_rdna", [](uintptr_t output, uintptr_t input, int size,
                             uintptr_t stream) {
    gelu_rdna(const_typed_ptr<__hip_bfloat16>(input),
              typed_ptr<__hip_bfloat16>(output), size, to_stream(stream));
  });

  module.def("gelu_mul_rdna", [](uintptr_t output, uintptr_t gate, uintptr_t up,
                                 int size, uintptr_t stream) {
    gelu_mul_rdna(const_typed_ptr<__hip_bfloat16>(gate),
                  const_typed_ptr<__hip_bfloat16>(up),
                  typed_ptr<__hip_bfloat16>(output), size, to_stream(stream));
  });

  module.def("gelu_mul_merged_rdna", [](uintptr_t output, uintptr_t gate_up,
                                        int rows, int width, uintptr_t stream) {
    gelu_mul_merged_rdna(const_typed_ptr<__hip_bfloat16>(gate_up),
                         typed_ptr<__hip_bfloat16>(output), rows, width,
                         to_stream(stream));
  });

  module.def("silu_rdna", [](uintptr_t output, uintptr_t input, int size,
                             uintptr_t stream) {
    silu_rdna(const_typed_ptr<__hip_bfloat16>(input),
              typed_ptr<__hip_bfloat16>(output), size, to_stream(stream));
  });

  module.def("residual_rdna", [](uintptr_t output, uintptr_t update,
                                 uintptr_t residual, uintptr_t gate, int size,
                                 int width, uintptr_t stream) {
    residual_rdna(const_typed_ptr<__hip_bfloat16>(update),
                  const_typed_ptr<__hip_bfloat16>(residual),
                  gate == 0 ? nullptr : const_typed_ptr<__hip_bfloat16>(gate),
                  typed_ptr<__hip_bfloat16>(output), size, width,
                  to_stream(stream));
  });

  module.def("residual_rms_rdna",
             [](uintptr_t output_sum, uintptr_t output_norm, uintptr_t update,
                uintptr_t residual, int rows, int width, float epsilon,
                uintptr_t stream) {
               residual_rms_rdna(const_typed_ptr<__hip_bfloat16>(update),
                                 const_typed_ptr<__hip_bfloat16>(residual),
                                 typed_ptr<__hip_bfloat16>(output_sum),
                                 typed_ptr<__hip_bfloat16>(output_norm), rows,
                                 width, epsilon, to_stream(stream));
             });

  module.def("residual_adarms_rdna",
             [](uintptr_t output_sum, uintptr_t output_norm, uintptr_t update,
                uintptr_t residual, uintptr_t gate, uintptr_t modulation,
                int rows, int width, float epsilon, uintptr_t stream) {
               residual_adarms_rdna(const_typed_ptr<__hip_bfloat16>(update),
                                    const_typed_ptr<__hip_bfloat16>(residual),
                                    const_typed_ptr<__hip_bfloat16>(gate),
                                    const_typed_ptr<__hip_bfloat16>(modulation),
                                    typed_ptr<__hip_bfloat16>(output_sum),
                                    typed_ptr<__hip_bfloat16>(output_norm),
                                    rows, width, epsilon, to_stream(stream));
             });

  // ── Attention kernels ──

  module.def(
      "attention_decoder_gqa_rdna",
      [](uintptr_t output, uintptr_t query, uintptr_t key, uintptr_t value,
         int query_rows, int kv_rows, int query_heads, int head_dim,
         int valid_prefix, int suffix_start, float scale, uintptr_t stream) {
        attention_decoder_gqa_rdna(
            const_typed_ptr<__hip_bfloat16>(query),
            const_typed_ptr<__hip_bfloat16>(key),
            const_typed_ptr<__hip_bfloat16>(value),
            typed_ptr<__hip_bfloat16>(output), query_rows, kv_rows, query_heads,
            head_dim, valid_prefix, suffix_start, scale, to_stream(stream));
      });

  module.def("attention_decoder_gqa_splitkey_rdna",
             [](uintptr_t output, uintptr_t query, uintptr_t key,
                uintptr_t value, int query_rows, int kv_rows, int query_heads,
                int head_dim, int valid_prefix, int suffix_start, float scale,
                int keys_per_iteration, uintptr_t stream) {
               attention_decoder_gqa_splitkey_rdna(
                   const_typed_ptr<__hip_bfloat16>(query),
                   const_typed_ptr<__hip_bfloat16>(key),
                   const_typed_ptr<__hip_bfloat16>(value),
                   typed_ptr<__hip_bfloat16>(output), query_rows, kv_rows,
                   query_heads, head_dim, valid_prefix, suffix_start, scale,
                   keys_per_iteration, to_stream(stream));
             });

  module.def("attention_encoder_gqa_rdna",
             [](uintptr_t output, uintptr_t query, uintptr_t key,
                uintptr_t value, int query_rows, int valid_kv_rows,
                int query_heads, int head_dim, uintptr_t stream) {
               attention_encoder_gqa_rdna(
                   const_typed_ptr<__hip_bfloat16>(query),
                   const_typed_ptr<__hip_bfloat16>(key),
                   const_typed_ptr<__hip_bfloat16>(value),
                   typed_ptr<__hip_bfloat16>(output), query_rows,
                   valid_kv_rows, query_heads, head_dim, to_stream(stream));
             });

  // ── Small-M WMMA kernels ──

  module.def("smallm_wmma_bf16_rdna", [](uintptr_t output, uintptr_t input,
                                         uintptr_t weight_nt, uintptr_t bias,
                                         int rows, int columns, int inner,
                                         uintptr_t stream) {
    smallm_wmma_bf16_rdna(const_typed_ptr<__hip_bfloat16>(input),
                          const_typed_ptr<__hip_bfloat16>(weight_nt),
                          bias == 0 ? nullptr
                                    : const_typed_ptr<__hip_bfloat16>(bias),
                          typed_ptr<__hip_bfloat16>(output), rows, columns,
                          inner, to_stream(stream));
  });

  module.def("smallm_wmma_bf16_residual_rdna",
             [](uintptr_t output, uintptr_t input, uintptr_t weight_nt,
                uintptr_t bias, int rows, int columns, int inner,
                uintptr_t stream) {
               smallm_wmma_bf16_residual_rdna(
                   const_typed_ptr<__hip_bfloat16>(input),
                   const_typed_ptr<__hip_bfloat16>(weight_nt),
                   bias == 0 ? nullptr : const_typed_ptr<__hip_bfloat16>(bias),
                   typed_ptr<__hip_bfloat16>(output), rows, columns, inner,
                   to_stream(stream));
             });
}
