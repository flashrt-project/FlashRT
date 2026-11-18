// ================================================================
// FlashRT AMD -- pybind11 bindings for RDNA 3 / 3.5 / 4 BF16 kernels
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
#include <string_view>

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
// and gemm/. RDNA targets build this surface independently of CDNA kernels.

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

void moe_forward_rdna(const __hip_bfloat16 *input,
                      const __hip_bfloat16 *routing, const int *selected,
                      const __hip_bfloat16 *gate, const __hip_bfloat16 *up,
                      const __hip_bfloat16 *down, int *counts, int *rows,
                      int *slots, __hip_bfloat16 *intermediate,
                      float *routes_output, float *output, int tokens,
                      int hidden, int experts, int top_k, int width,
                      hipStream_t stream);
void moe_forward_packed_rdna(const __hip_bfloat16 *input,
                             const __hip_bfloat16 *routing, const int *selected,
                             const __hip_bfloat16 *gate_up,
                             const __hip_bfloat16 *down, int *counts, int *rows,
                             int *slots, __hip_bfloat16 *intermediate,
                             float *routes_output, float *output, int tokens,
                             int hidden, int experts, int top_k, int width,
                             hipStream_t stream);
void moe_topk_rdna(const float *logits, const float *bias,
                   __hip_bfloat16 *routing, int *selected, int rows,
                   int experts, int top_k, float scale, hipStream_t stream);
void moe_pack_compact_rdna(const __hip_bfloat16 *raw, unsigned char *blob,
                           const unsigned int *metadata, int tiles, bool gate,
                           hipStream_t stream);
void moe_forward_compact_rdna(
    const __hip_bfloat16 *input, const __hip_bfloat16 *routing,
    const int *selected, const unsigned char *gate_blob,
    const unsigned char *down_blob, const unsigned int *gate_metadata,
    const unsigned int *down_metadata, int *counts, int *rows, int *slots,
    __hip_bfloat16 *intermediate, float *routes_output, float *output,
    int tokens, int hidden, int experts, int top_k, int width,
    hipStream_t stream);

void q_norm_rope_pack_rdna(const __hip_bfloat16 *input,
                           const __hip_bfloat16 *weight, const float *cosine,
                           const float *sine, float *output, int rows,
                           int heads, int groups, float epsilon,
                           hipStream_t stream);
void k_norm_rope_cache_rdna(const __hip_bfloat16 *key,
                            const __hip_bfloat16 *value,
                            const __hip_bfloat16 *weight, const float *cosine,
                            const float *sine, const int64_t *row_map,
                            float *key_cache, float *value_cache, int prefix,
                            int heads, float epsilon, hipStream_t stream);
void residual_rms_router_rdna(const __hip_bfloat16 *update,
                              const __hip_bfloat16 *residual,
                              const __hip_bfloat16 *weight, const float *gamma,
                              const float *beta, __hip_bfloat16 *summed,
                              __hip_bfloat16 *output, float *router_input,
                              int rows, float epsilon, hipStream_t stream);
void rms_norm_affine_rdna(const __hip_bfloat16 *input,
                          const __hip_bfloat16 *weight, __hip_bfloat16 *output,
                          int rows, int width, float epsilon,
                          hipStream_t stream);
void ada_rms_norm_fp32_rdna(const __hip_bfloat16 *input,
                            const __hip_bfloat16 *weight, const float *gamma,
                            const float *beta, __hip_bfloat16 *output, int rows,
                            int width, float epsilon, hipStream_t stream);
void residual_rms_affine_rdna(const void *update,
                              const __hip_bfloat16 *residual,
                              const __hip_bfloat16 *shared,
                              const __hip_bfloat16 *weight, const float *gamma,
                              const float *beta, __hip_bfloat16 *summed,
                              __hip_bfloat16 *output, int rows, int width,
                              float epsilon, hipStream_t stream);
void rotate_half_rdna(const void *input, const void *cosine, const void *sine,
                      void *output, int rows, int heads, int dim,
                      int row_stride, int head_stride, int cosine_stride,
                      int sine_stride, bool input_fp32, bool coeff_fp32,
                      bool output_fp32, hipStream_t stream);
void silu_bf16_rdna(const __hip_bfloat16 *gate, const __hip_bfloat16 *up,
                    __hip_bfloat16 *output, int size, hipStream_t stream);
void attention_pack_kv_rdna(const float *key, const float *value,
                            const float *suffix_key,
                            const __hip_bfloat16 *suffix_value,
                            float *packed_key, float *packed_value,
                            int prefix_rows, int suffix_rows, int heads,
                            int kv_heads, int dim, int key_stride,
                            int value_stride, int suffix_key_stride,
                            int suffix_value_stride, hipStream_t stream);
void attention_softmax_fp32_rdna(const float *scores, const bool *mask,
                                 float *probs, int queries, int keys, int heads,
                                 float scale, hipStream_t stream);
void attention_store_bf16_rdna(const float *input, __hip_bfloat16 *output,
                               int queries, int heads, int dim,
                               hipStream_t stream);
void attention_pack_q_fp32_rdna(const float *input, float *output, int queries,
                                int heads, int groups, int dim,
                                int query_stride, hipStream_t stream);
void attention_rope_pack_fp32_rdna(
    const __hip_bfloat16 *query, const __hip_bfloat16 *key,
    const __hip_bfloat16 *value, const float *cosine, const float *sine,
    float *packed_query, float *packed_key, float *packed_value, int queries,
    int heads, int kv_heads, int dim, int query_stride, int key_stride,
    int value_stride, int cosine_stride, int sine_stride, hipStream_t stream);
void attention_softmax_gqa_fp32_rdna(const float *scores, const bool *mask,
                                     float *probs, int queries, int keys,
                                     int heads, int groups, float scale,
                                     hipStream_t stream);
void attention_store_gqa_bf16_rdna(const float *input, __hip_bfloat16 *output,
                                   int queries, int heads, int groups, int dim,
                                   hipStream_t stream);
void attention_softmax_bf16_rdna(const __hip_bfloat16 *scores,
                                 __hip_bfloat16 *probs, int queries, int keys,
                                 int heads, float scale, hipStream_t stream);
void attention_store_bshd_rdna(const __hip_bfloat16 *input,
                               __hip_bfloat16 *output, int queries, int heads,
                               int dim, hipStream_t stream);

PYBIND11_MODULE(flash_rt_amd_kernels, module) {
  module.doc() = "FlashRT AMD RDNA BF16 kernels (raw-pointer ABI)";

  // ── Runtime/build identity ──

  module.def("build_info", []() {
    py::dict info;
    info["platform"] = "hip";
    info["backend"] = "rdna";
    info["wave_size"] = 32;
#ifdef FLASHRT_AMD_GPU_ARCH
    constexpr std::string_view gpu_arch = FLASHRT_AMD_GPU_ARCH;
    static_assert(gpu_arch == "gfx1100" || gpu_arch == "gfx1151" ||
                      gpu_arch == "gfx1201",
                  "Unsupported RDNA build target");
    info["gpu_arch"] = std::string(gpu_arch);
    info["hardware"] = gpu_arch == "gfx1100" ? "amd_rdna3"
                       : gpu_arch == "gfx1151" ? "amd_rdna35"
                                               : "amd_rdna4";
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
      .def("enable_lazy_autotune", &RdnaGemmRunner::enable_lazy_autotune,
           py::arg("num_algorithms") = 16)
      .def("begin_bf16_nt_calibration",
           &RdnaGemmRunner::begin_bf16_nt_calibration, py::arg("rows"),
           py::arg("columns"), py::arg("inner"), py::arg("biased"),
           py::arg("num_algorithms") = 32)
      .def("end_bf16_nt_calibration", &RdnaGemmRunner::end_bf16_nt_calibration)
      .def("begin_fp32_gqa_pv_calibration",
           &RdnaGemmRunner::begin_fp32_gqa_pv_calibration,
           py::arg("num_algorithms") = 16)
      .def("end_fp32_gqa_pv_calibration",
           &RdnaGemmRunner::end_fp32_gqa_pv_calibration)
      .def(
          "bf16_bmm",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, int rows, int columns, int inner,
             int input_stride, int weight_stride, int output_stride,
             int input_batch_stride, int weight_batch_stride,
             int output_batch_stride, int batches, bool transposed,
             uintptr_t stream) {
            self.bf16_bmm(to_ptr(input), to_ptr(weight), to_ptr(output), rows,
                          columns, inner, input_stride, weight_stride,
                          output_stride, input_batch_stride,
                          weight_batch_stride, output_batch_stride, batches,
                          transposed, to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("rows"), py::arg("columns"), py::arg("inner"),
          py::arg("input_stride"), py::arg("weight_stride"),
          py::arg("output_stride"), py::arg("input_batch_stride"),
          py::arg("weight_batch_stride"), py::arg("output_batch_stride"),
          py::arg("batches"), py::arg("transposed"), py::arg("stream") = 0)
      .def(
          "fp32_bmm",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, int rows, int columns, int inner,
             int input_stride, int weight_stride, int output_stride,
             int input_batch_stride, int weight_batch_stride,
             int output_batch_stride, int batches, bool transposed,
             uintptr_t stream) {
            self.fp32_bmm(to_ptr(input), to_ptr(weight), to_ptr(output), rows,
                          columns, inner, input_stride, weight_stride,
                          output_stride, input_batch_stride,
                          weight_batch_stride, output_batch_stride, batches,
                          transposed,
                          to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("rows"), py::arg("columns"), py::arg("inner"),
          py::arg("input_stride"), py::arg("weight_stride"),
          py::arg("output_stride"), py::arg("input_batch_stride"),
          py::arg("weight_batch_stride"), py::arg("output_batch_stride"),
          py::arg("batches"), py::arg("transposed"), py::arg("stream") = 0)
      .def(
          "fp32_gqa_bmm",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, int queries, int keys, int dim, int heads,
             int kv_heads, int query_stride, bool transposed,
             uintptr_t stream) {
            self.fp32_gqa_bmm(to_ptr(input), to_ptr(weight), to_ptr(output),
                              queries, keys, dim, heads, kv_heads, query_stride,
                              transposed, to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("queries"), py::arg("keys"), py::arg("dim"), py::arg("heads"),
          py::arg("kv_heads"), py::arg("query_stride"), py::arg("transposed"),
          py::arg("stream") = 0)
      .def(
          "fp32_nt",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, int rows, int columns, int inner,
             int weight_stride, int output_stride, uintptr_t stream) {
            self.fp32_nt(to_ptr(input), to_ptr(weight), to_ptr(output), rows,
                         columns, inner, weight_stride, output_stride,
                         to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("rows"), py::arg("columns"), py::arg("inner"),
          py::arg("weight_stride"), py::arg("output_stride"),
          py::arg("stream") = 0)
      .def(
          "bf16_nt",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, uintptr_t bias, int rows, int columns, int inner,
             int weight_stride, int output_stride, uintptr_t stream) {
            self.bf16_nt(to_ptr(input), to_ptr(weight), to_ptr(output),
                         to_ptr(bias), rows, columns, inner, weight_stride,
                         output_stride, to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("bias"), py::arg("rows"), py::arg("columns"),
          py::arg("inner"), py::arg("weight_stride"), py::arg("output_stride"),
          py::arg("stream") = 0)
      .def(
          "bf16_nt_pair",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, int rows, int reference_rows, int columns,
             int inner, uintptr_t stream) {
            self.bf16_nt_pair(to_ptr(input), to_ptr(weight), to_ptr(output),
                              rows, reference_rows, columns, inner,
                              to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("rows"), py::arg("reference_rows"), py::arg("columns"),
          py::arg("inner"), py::arg("stream") = 0)
      .def(
          "fp32_gqa_bmm_rows",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, int queries, int reference_queries, int keys,
             int dim, int heads, int kv_heads, int query_stride,
             bool transposed, uintptr_t stream) {
            self.fp32_gqa_bmm_rows(to_ptr(input), to_ptr(weight),
                                   to_ptr(output), queries, reference_queries,
                                   keys, dim, heads, kv_heads, query_stride,
                                   transposed, to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("queries"), py::arg("reference_queries"), py::arg("keys"),
          py::arg("dim"), py::arg("heads"), py::arg("kv_heads"),
          py::arg("query_stride"), py::arg("transposed"), py::arg("stream") = 0)
      .def(
          "bf16_nt_rows",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, uintptr_t bias, int rows, int reference_rows,
             int columns, int inner, int weight_stride, int output_stride,
             uintptr_t stream) {
            self.bf16_nt_rows(to_ptr(input), to_ptr(weight), to_ptr(output),
                              to_ptr(bias), rows, reference_rows, columns,
                              inner, weight_stride, output_stride,
                              to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("bias"), py::arg("rows"), py::arg("reference_rows"),
          py::arg("columns"), py::arg("inner"), py::arg("weight_stride"),
          py::arg("output_stride"), py::arg("stream") = 0)
      .def(
          "bf16_nn",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, int rows, int columns, int inner,
             int weight_stride, int output_stride, uintptr_t stream) {
            self.bf16_nn(to_ptr(input), to_ptr(weight), to_ptr(output), rows,
                         columns, inner, weight_stride, output_stride,
                         to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("rows"), py::arg("columns"), py::arg("inner"),
          py::arg("weight_stride"), py::arg("output_stride"),
          py::arg("stream") = 0)
      .def(
          "bf16_nn_bias",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, uintptr_t bias, int rows, int columns, int inner,
             int weight_stride, int output_stride, uintptr_t stream) {
            self.bf16_nn_bias(to_ptr(input), to_ptr(weight), to_ptr(output),
                              to_ptr(bias), rows, columns, inner, weight_stride,
                              output_stride, to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("bias"), py::arg("rows"), py::arg("columns"),
          py::arg("inner"), py::arg("weight_stride"), py::arg("output_stride"),
          py::arg("stream") = 0)
      .def(
          "autotune_bf16_nn",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, int rows, int columns, int inner,
             int weight_stride, int output_stride, int num_algorithms,
             uintptr_t stream) {
            self.autotune_bf16_nn(to_ptr(input), to_ptr(weight), to_ptr(output),
                                  rows, columns, inner, weight_stride,
                                  output_stride, num_algorithms,
                                  to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("rows"), py::arg("columns"), py::arg("inner"),
          py::arg("weight_stride"), py::arg("output_stride"),
          py::arg("num_algorithms") = 16, py::arg("stream") = 0)
      .def(
          "autotune_bf16_nn_bias",
          [](RdnaGemmRunner &self, uintptr_t input, uintptr_t weight,
             uintptr_t output, uintptr_t bias, int rows, int columns, int inner,
             int weight_stride, int output_stride, int num_algorithms,
             uintptr_t stream) {
            self.autotune_bf16_nn_bias(
                to_ptr(input), to_ptr(weight), to_ptr(output), to_ptr(bias),
                rows, columns, inner, weight_stride, output_stride,
                num_algorithms, to_stream(stream));
          },
          py::arg("input"), py::arg("weight"), py::arg("output"),
          py::arg("bias"), py::arg("rows"), py::arg("columns"),
          py::arg("inner"), py::arg("weight_stride"), py::arg("output_stride"),
          py::arg("num_algorithms") = 16, py::arg("stream") = 0);

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

  module.def(
      "rotate_half_rdna",
      [](uintptr_t input, uintptr_t cosine, uintptr_t sine, uintptr_t output,
         int rows, int heads, int dim, int row_stride, int head_stride,
         int cosine_stride, int sine_stride, bool input_fp32, bool coeff_fp32,
         bool output_fp32, uintptr_t stream) {
        rotate_half_rdna(to_ptr(input), to_ptr(cosine), to_ptr(sine),
                         to_ptr(output), rows, heads, dim, row_stride,
                         head_stride, cosine_stride, sine_stride, input_fp32,
                         coeff_fp32, output_fp32, to_stream(stream));
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

  module.def("rms_norm_affine_rdna",
             [](uintptr_t input, uintptr_t weight, uintptr_t output, int rows,
                int width, float epsilon, uintptr_t stream) {
               rms_norm_affine_rdna(const_typed_ptr<__hip_bfloat16>(input),
                                    const_typed_ptr<__hip_bfloat16>(weight),
                                    typed_ptr<__hip_bfloat16>(output), rows,
                                    width, epsilon, to_stream(stream));
             });

  module.def("adarms_rdna",
             [](uintptr_t output, uintptr_t input, uintptr_t modulation,
                int rows, int width, float epsilon, uintptr_t stream) {
               adarms_rdna(const_typed_ptr<__hip_bfloat16>(input),
                           const_typed_ptr<__hip_bfloat16>(modulation),
                           typed_ptr<__hip_bfloat16>(output), rows, width,
                           epsilon, to_stream(stream));
             });

  module.def("ada_rms_norm_fp32_rdna", [](uintptr_t input, uintptr_t weight,
                                          uintptr_t gamma, uintptr_t beta,
                                          uintptr_t output, int rows, int width,
                                          float epsilon, uintptr_t stream) {
    ada_rms_norm_fp32_rdna(
        const_typed_ptr<__hip_bfloat16>(input),
        const_typed_ptr<__hip_bfloat16>(weight), const_typed_ptr<float>(gamma),
        const_typed_ptr<float>(beta), typed_ptr<__hip_bfloat16>(output), rows,
        width, epsilon, to_stream(stream));
  });

  module.def("residual_rms_affine_rdna",
             [](uintptr_t update, uintptr_t residual, uintptr_t shared,
                uintptr_t weight, uintptr_t gamma, uintptr_t beta,
                uintptr_t summed, uintptr_t output, int rows, int width,
                float epsilon, uintptr_t stream) {
               residual_rms_affine_rdna(
                   to_ptr(update), const_typed_ptr<__hip_bfloat16>(residual),
                   const_typed_ptr<__hip_bfloat16>(shared),
                   const_typed_ptr<__hip_bfloat16>(weight),
                   const_typed_ptr<float>(gamma), const_typed_ptr<float>(beta),
                   typed_ptr<__hip_bfloat16>(summed),
                   typed_ptr<__hip_bfloat16>(output), rows, width, epsilon,
                   to_stream(stream));
             });

  module.def("q_norm_rope_pack_rdna",
             [](uintptr_t input, uintptr_t weight, uintptr_t cosine,
                uintptr_t sine, uintptr_t output, int rows, int heads,
                int groups, float epsilon, uintptr_t stream) {
               q_norm_rope_pack_rdna(const_typed_ptr<__hip_bfloat16>(input),
                                     const_typed_ptr<__hip_bfloat16>(weight),
                                     const_typed_ptr<float>(cosine),
                                     const_typed_ptr<float>(sine),
                                     typed_ptr<float>(output), rows, heads,
                                     groups, epsilon, to_stream(stream));
             });

  module.def("k_norm_rope_cache_rdna",
             [](uintptr_t key, uintptr_t value, uintptr_t weight,
                uintptr_t cosine, uintptr_t sine, uintptr_t row_map,
                uintptr_t key_cache, uintptr_t value_cache, int prefix,
                int heads, float epsilon, uintptr_t stream) {
               k_norm_rope_cache_rdna(
                   const_typed_ptr<__hip_bfloat16>(key),
                   const_typed_ptr<__hip_bfloat16>(value),
                   const_typed_ptr<__hip_bfloat16>(weight),
                   const_typed_ptr<float>(cosine), const_typed_ptr<float>(sine),
                   const_typed_ptr<int64_t>(row_map),
                   typed_ptr<float>(key_cache), typed_ptr<float>(value_cache),
                   prefix, heads, epsilon, to_stream(stream));
             });

  module.def(
      "residual_rms_router_rdna",
      [](uintptr_t update, uintptr_t residual, uintptr_t weight,
         uintptr_t gamma, uintptr_t beta, uintptr_t summed, uintptr_t output,
         uintptr_t router_input, int rows, float epsilon, uintptr_t stream) {
        residual_rms_router_rdna(
            const_typed_ptr<__hip_bfloat16>(update),
            const_typed_ptr<__hip_bfloat16>(residual),
            const_typed_ptr<__hip_bfloat16>(weight),
            const_typed_ptr<float>(gamma), const_typed_ptr<float>(beta),
            typed_ptr<__hip_bfloat16>(summed),
            typed_ptr<__hip_bfloat16>(output), typed_ptr<float>(router_input),
            rows, epsilon, to_stream(stream));
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

  module.def("silu_bf16_rdna", [](uintptr_t gate, uintptr_t up,
                                  uintptr_t output, int size,
                                  uintptr_t stream) {
    silu_bf16_rdna(const_typed_ptr<__hip_bfloat16>(gate),
                   const_typed_ptr<__hip_bfloat16>(up),
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

  module.def("attention_softmax_bf16_rdna", [](uintptr_t scores,
                                               uintptr_t probs, int queries,
                                               int keys, int heads, float scale,
                                               uintptr_t stream) {
    attention_softmax_bf16_rdna(const_typed_ptr<__hip_bfloat16>(scores),
                                typed_ptr<__hip_bfloat16>(probs), queries, keys,
                                heads, scale, to_stream(stream));
  });
  module.def("attention_store_bshd_rdna", [](uintptr_t input, uintptr_t output,
                                             int queries, int heads, int dim,
                                             uintptr_t stream) {
    attention_store_bshd_rdna(const_typed_ptr<__hip_bfloat16>(input),
                              typed_ptr<__hip_bfloat16>(output), queries, heads,
                              dim, to_stream(stream));
  });
  module.def("attention_pack_kv_rdna",
             [](uintptr_t k, uintptr_t v, uintptr_t sk, uintptr_t sv,
                uintptr_t pk, uintptr_t pv, int prefix, int suffix, int heads,
                int kv_heads, int dim, int k_stride, int v_stride,
                int sk_stride, int sv_stride, uintptr_t stream) {
               attention_pack_kv_rdna(
                   const_typed_ptr<float>(k), const_typed_ptr<float>(v),
                   const_typed_ptr<float>(sk),
                   const_typed_ptr<__hip_bfloat16>(sv), typed_ptr<float>(pk),
                   typed_ptr<float>(pv), prefix, suffix, heads, kv_heads, dim,
                   k_stride, v_stride, sk_stride, sv_stride, to_stream(stream));
             });
  module.def("attention_softmax_fp32_rdna",
             [](uintptr_t scores, uintptr_t mask, uintptr_t probs, int queries,
                int keys, int heads, float scale, uintptr_t stream) {
               attention_softmax_fp32_rdna(
                   const_typed_ptr<float>(scores), const_typed_ptr<bool>(mask),
                   typed_ptr<float>(probs), queries, keys, heads, scale,
                   to_stream(stream));
             });
  module.def("attention_store_bf16_rdna", [](uintptr_t input, uintptr_t output,
                                             int queries, int heads, int dim,
                                             uintptr_t stream) {
    attention_store_bf16_rdna(const_typed_ptr<float>(input),
                              typed_ptr<__hip_bfloat16>(output), queries, heads,
                              dim, to_stream(stream));
  });

  module.def("attention_rope_pack_fp32_rdna",
             [](uintptr_t query, uintptr_t key, uintptr_t value,
                uintptr_t cosine, uintptr_t sine, uintptr_t packed_query,
                uintptr_t packed_key, uintptr_t packed_value, int queries,
                int heads, int kv_heads, int dim, int query_stride,
                int key_stride, int value_stride, int cosine_stride,
                int sine_stride, uintptr_t stream) {
               attention_rope_pack_fp32_rdna(
                   const_typed_ptr<__hip_bfloat16>(query),
                   const_typed_ptr<__hip_bfloat16>(key),
                   const_typed_ptr<__hip_bfloat16>(value),
                   const_typed_ptr<float>(cosine), const_typed_ptr<float>(sine),
                   typed_ptr<float>(packed_query), typed_ptr<float>(packed_key),
                   typed_ptr<float>(packed_value), queries, heads, kv_heads,
                   dim, query_stride, key_stride, value_stride, cosine_stride,
                   sine_stride, to_stream(stream));
             });
  module.def("attention_pack_q_fp32_rdna",
             [](uintptr_t input, uintptr_t output, int queries, int heads,
                int groups, int dim, int stride, uintptr_t stream) {
               attention_pack_q_fp32_rdna(
                   const_typed_ptr<float>(input), typed_ptr<float>(output),
                   queries, heads, groups, dim, stride, to_stream(stream));
             });
  module.def("attention_softmax_gqa_fp32_rdna",
             [](uintptr_t scores, uintptr_t mask, uintptr_t probs, int queries,
                int keys, int heads, int groups, float scale,
                uintptr_t stream) {
               attention_softmax_gqa_fp32_rdna(
                   const_typed_ptr<float>(scores), const_typed_ptr<bool>(mask),
                   typed_ptr<float>(probs), queries, keys, heads, groups, scale,
                   to_stream(stream));
             });
  module.def("attention_store_gqa_bf16_rdna",
             [](uintptr_t input, uintptr_t output, int queries, int heads,
                int groups, int dim, uintptr_t stream) {
               attention_store_gqa_bf16_rdna(const_typed_ptr<float>(input),
                                             typed_ptr<__hip_bfloat16>(output),
                                             queries, heads, groups, dim,
                                             to_stream(stream));
             });

  // ── Routed BF16 MoE ──

  module.def("moe_topk_rdna", [](uintptr_t logits, uintptr_t bias,
                                 uintptr_t routing, uintptr_t selected,
                                 int rows, int experts, int top_k, float scale,
                                 uintptr_t stream) {
    moe_topk_rdna(const_typed_ptr<float>(logits), const_typed_ptr<float>(bias),
                  typed_ptr<__hip_bfloat16>(routing), typed_ptr<int>(selected),
                  rows, experts, top_k, scale, to_stream(stream));
  });

  module.def("moe_forward_rdna",
             [](uintptr_t input, uintptr_t routing, uintptr_t selected,
                uintptr_t gate, uintptr_t up, uintptr_t down, uintptr_t counts,
                uintptr_t rows, uintptr_t slots, uintptr_t intermediate,
                uintptr_t routes_output, uintptr_t output, int tokens,
                int hidden, int experts, int top_k, int width,
                uintptr_t stream) {
               moe_forward_rdna(const_typed_ptr<__hip_bfloat16>(input),
                                const_typed_ptr<__hip_bfloat16>(routing),
                                const_typed_ptr<int>(selected),
                                const_typed_ptr<__hip_bfloat16>(gate),
                                const_typed_ptr<__hip_bfloat16>(up),
                                const_typed_ptr<__hip_bfloat16>(down),
                                typed_ptr<int>(counts), typed_ptr<int>(rows),
                                typed_ptr<int>(slots),
                                typed_ptr<__hip_bfloat16>(intermediate),
                                typed_ptr<float>(routes_output),
                                typed_ptr<float>(output), tokens, hidden,
                                experts, top_k, width, to_stream(stream));
             });

  module.def(
      "moe_forward_packed_rdna",
      [](uintptr_t input, uintptr_t routing, uintptr_t selected,
         uintptr_t gate_up, uintptr_t down, uintptr_t counts, uintptr_t rows,
         uintptr_t slots, uintptr_t intermediate, uintptr_t routes_output,
         uintptr_t output, int tokens, int hidden, int experts, int top_k,
         int width, uintptr_t stream) {
        moe_forward_packed_rdna(
            const_typed_ptr<__hip_bfloat16>(input),
            const_typed_ptr<__hip_bfloat16>(routing),
            const_typed_ptr<int>(selected),
            const_typed_ptr<__hip_bfloat16>(gate_up),
            const_typed_ptr<__hip_bfloat16>(down), typed_ptr<int>(counts),
            typed_ptr<int>(rows), typed_ptr<int>(slots),
            typed_ptr<__hip_bfloat16>(intermediate),
            typed_ptr<float>(routes_output), typed_ptr<float>(output), tokens,
            hidden, experts, top_k, width, to_stream(stream));
      });

  module.def("moe_pack_compact_rdna",
             [](uintptr_t raw, uintptr_t blob, uintptr_t metadata, int tiles,
                bool gate, uintptr_t stream) {
               moe_pack_compact_rdna(const_typed_ptr<__hip_bfloat16>(raw),
                                     typed_ptr<unsigned char>(blob),
                                     const_typed_ptr<unsigned int>(metadata),
                                     tiles, gate, to_stream(stream));
             });

  module.def(
      "moe_forward_compact_rdna",
      [](uintptr_t input, uintptr_t routing, uintptr_t selected,
         uintptr_t gate_blob, uintptr_t down_blob, uintptr_t gate_metadata,
         uintptr_t down_metadata, uintptr_t counts, uintptr_t rows,
         uintptr_t slots, uintptr_t intermediate, uintptr_t routes_output,
         uintptr_t output, int tokens, int hidden, int experts, int top_k,
         int width, uintptr_t stream) {
        moe_forward_compact_rdna(
            const_typed_ptr<__hip_bfloat16>(input),
            const_typed_ptr<__hip_bfloat16>(routing),
            const_typed_ptr<int>(selected),
            const_typed_ptr<unsigned char>(gate_blob),
            const_typed_ptr<unsigned char>(down_blob),
            const_typed_ptr<unsigned int>(gate_metadata),
            const_typed_ptr<unsigned int>(down_metadata),
            typed_ptr<int>(counts), typed_ptr<int>(rows), typed_ptr<int>(slots),
            typed_ptr<__hip_bfloat16>(intermediate),
            typed_ptr<float>(routes_output), typed_ptr<float>(output), tokens,
            hidden, experts, top_k, width, to_stream(stream));
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
