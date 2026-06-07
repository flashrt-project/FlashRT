#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gemm/hipblaslt_matmul.h"
#include "gemm/hipblaslt_probe.h"

namespace py = pybind11;

void launch_vector_add_f32(const float* a, const float* b, float* out,
                           std::size_t n, hipStream_t stream);
void launch_rms_norm_f32(const float* x, const float* weight, float* out,
                         int rows, int hidden, float eps, hipStream_t stream);
void launch_rms_norm_bf16(const void* x, const void* weight, void* out,
                          int rows, int hidden, float eps, hipStream_t stream);
void launch_rms_norm_bf16_f32w(const void* x, const float* weight, void* out,
                               int rows, int hidden, float eps, hipStream_t stream);
void launch_rms_norm_fp8_e4m3fnuz(
    const void* x, const void* weight, void* out,
    int rows, int hidden, float eps, const float* scale,
    hipStream_t stream);
void launch_layer_norm_bf16(const void* x, const void* weight, const void* bias,
                            void* out, int rows, int hidden, float eps,
                            hipStream_t stream);
void launch_layer_norm_fp8_e4m3fnuz(
    const void* x, const void* weight, const void* bias, void* out,
    const float* scale, int rows, int hidden, float eps, hipStream_t stream);
void launch_add_bias_bf16(void* x, const void* bias, int rows, int hidden,
                          hipStream_t stream);
void launch_bias_residual_bf16(void* residual, const void* x, const void* bias,
                               int rows, int hidden, hipStream_t stream);
void launch_residual_add_bf16(void* residual, const void* x,
                              std::size_t n, hipStream_t stream);
void launch_gate_mul_residual_bf16(void* residual, const void* x,
                                   const void* gate, std::size_t n,
                                   hipStream_t stream);
void launch_residual_add_rms_norm_bf16(
    void* residual, const void* x, const void* weight, void* out,
    int rows, int hidden, float eps, hipStream_t stream);
void launch_residual_add_rms_norm_fp8_e4m3fnuz(
    void* residual, const void* x, const void* weight, void* out,
    int rows, int hidden, float eps, const float* scale, hipStream_t stream);
void launch_ada_rms_norm_style_bf16(
    const void* x, const void* weight, const void* style, void* out,
    void* gate_out, int rows, int hidden, float eps, hipStream_t stream);
void launch_ada_rms_norm_style_fp8_e4m3fnuz(
    const void* x, const void* weight, const void* style, void* out,
    void* gate_out, int rows, int hidden, float eps, const float* scale,
    hipStream_t stream);
void launch_gate_residual_ada_norm_bf16(
    void* residual, const void* x, const void* gate,
    const void* weight, const void* style, void* out, void* gate_out,
    int rows, int hidden, float eps, hipStream_t stream);
void launch_gate_residual_ada_norm_fp8_e4m3fnuz(
    void* residual, const void* x, const void* gate,
    const void* weight, const void* style, void* out, void* gate_out,
    int rows, int hidden, float eps, const float* scale, hipStream_t stream);
void launch_bias_residual_layer_norm_bf16(
    void* residual, const void* x, const void* bias_pre,
    const void* norm_weight, const void* norm_bias, void* out,
    int rows, int hidden, float eps, hipStream_t stream);
void launch_bias_residual_layer_norm_fp8_e4m3fnuz(
    void* residual, const void* x, const void* bias_pre,
    const void* norm_weight, const void* norm_bias, void* out,
    const float* scale, int rows, int hidden, float eps, hipStream_t stream);
void launch_gelu_tanh_mul_bf16(const void* gate, const void* up, void* out,
                               std::size_t n, hipStream_t stream);
void launch_gelu_tanh_mul_quantize_fp8_e4m3fnuz(
    const void* gate, const void* up, const float* scale, void* out,
    std::size_t n, hipStream_t stream);
void launch_gelu_tanh_merged_bf16(
    const void* gate_up, void* out, int rows, int hidden, hipStream_t stream);
void launch_gelu_tanh_merged_quantize_fp8_e4m3fnuz(
    const void* gate_up, const float* scale, void* out, int rows, int hidden,
    hipStream_t stream);
void launch_gelu_tanh_bf16(const void* x, void* out, std::size_t n,
                           hipStream_t stream);
void launch_gelu_tanh_quantize_fp8_e4m3fnuz(
    const void* x, const float* scale, void* out, std::size_t n,
    hipStream_t stream);
void launch_silu_bf16(const void* x, void* out, std::size_t n,
                      hipStream_t stream);
void launch_quantize_f32_to_fp8_e4m3fnuz(const float* x, const float* scale,
                                         void* out, std::size_t n,
                                         hipStream_t stream);
void launch_quantize_bf16_to_fp8_e4m3fnuz(const void* x, const float* scale,
                                          void* out, std::size_t n,
                                          hipStream_t stream);
void launch_dynamic_quantize_f32_to_fp8_e4m3fnuz(
    const float* x, void* out, float* scale, float* partial,
    int partial_count, std::size_t n, hipStream_t stream);
void launch_dynamic_quantize_bf16_to_fp8_e4m3fnuz(
    const void* x, void* out, float* scale, float* partial,
    int partial_count, std::size_t n, hipStream_t stream);
void launch_patch_im2col_u16(const void* input, void* output, int nv,
                             hipStream_t stream);
void launch_patch_embed_bias_pos_bf16(void* output, const void* bias,
                                      const void* pos_emb, int s, int d,
                                      int s_per_view, hipStream_t stream);
void launch_qkv_split_bf16(const void* qkv, void* q, void* k, void* v,
                           int seq, int q_dim, int k_dim, int v_dim,
                           hipStream_t stream);
void launch_qkv_split_rope_bf16(const void* qkv, const void* rope,
                                void* q, void* k, void* v,
                                int seq, int q_dim, int k_dim, int v_dim,
                                int head_dim, hipStream_t stream);
namespace {

void hip_check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::ostringstream oss;
    oss << what << " failed: " << hipGetErrorString(status);
    throw std::runtime_error(oss.str());
  }
}

std::uintptr_t data_ptr(py::handle tensor) {
  return tensor.attr("data_ptr")().cast<std::uintptr_t>();
}

std::size_t numel(py::handle tensor) {
  return tensor.attr("numel")().cast<std::size_t>();
}

bool bool_attr(py::handle obj, const char* name) {
  return obj.attr(name).cast<bool>();
}

bool bool_method(py::handle obj, const char* name) {
  return obj.attr(name)().cast<bool>();
}

std::string str_attr(py::handle obj, const char* name) {
  return py::str(obj.attr(name)).cast<std::string>();
}

hipStream_t stream_from_uint(std::uintptr_t stream) {
  return reinterpret_cast<hipStream_t>(stream);
}

void require_cuda_tensor(py::handle tensor, const char* name) {
  if (!bool_attr(tensor, "is_cuda")) {
    std::ostringstream oss;
    oss << name << " must be a CUDA/HIP tensor";
    throw std::invalid_argument(oss.str());
  }
  if (!bool_method(tensor, "is_contiguous")) {
    std::ostringstream oss;
    oss << name << " must be contiguous";
    throw std::invalid_argument(oss.str());
  }
  const std::string dtype = str_attr(tensor, "dtype");
  if (dtype.find("float32") == std::string::npos) {
    std::ostringstream oss;
    oss << name << " must be torch.float32, got " << dtype;
    throw std::invalid_argument(oss.str());
  }
}

void require_contiguous_cuda_tensor(py::handle tensor, const char* name) {
  if (!bool_attr(tensor, "is_cuda")) {
    std::ostringstream oss;
    oss << name << " must be a CUDA/HIP tensor";
    throw std::invalid_argument(oss.str());
  }
  if (!bool_method(tensor, "is_contiguous")) {
    std::ostringstream oss;
    oss << name << " must be contiguous";
    throw std::invalid_argument(oss.str());
  }
}

void require_bfloat16_tensor(py::handle tensor, const char* name) {
  require_contiguous_cuda_tensor(tensor, name);
  const std::string dtype = str_attr(tensor, "dtype");
  if (dtype.find("bfloat16") == std::string::npos) {
    std::ostringstream oss;
    oss << name << " must be torch.bfloat16, got " << dtype;
    throw std::invalid_argument(oss.str());
  }
}

void require_float32_scalar_tensor(py::handle tensor, const char* name) {
  require_contiguous_cuda_tensor(tensor, name);
  const std::string dtype = str_attr(tensor, "dtype");
  if (dtype.find("float32") == std::string::npos) {
    std::ostringstream oss;
    oss << name << " must be torch.float32, got " << dtype;
    throw std::invalid_argument(oss.str());
  }
  if (numel(tensor) != 1) {
    std::ostringstream oss;
    oss << name << " must contain exactly one element";
    throw std::invalid_argument(oss.str());
  }
}

void require_float8_e4m3fnuz_tensor(py::handle tensor, const char* name) {
  require_contiguous_cuda_tensor(tensor, name);
  const std::string dtype = str_attr(tensor, "dtype");
  if (dtype.find("float8_e4m3fnuz") == std::string::npos) {
    std::ostringstream oss;
    oss << name << " must be torch.float8_e4m3fnuz, got " << dtype;
    throw std::invalid_argument(oss.str());
  }
}

int64_t dim(py::handle tensor) {
  return tensor.attr("dim")().cast<int64_t>();
}

int64_t size_at(py::handle tensor, int64_t axis) {
  return tensor.attr("size")(axis).cast<int64_t>();
}

py::object vector_add_f32(py::object a, py::object b) {
  require_cuda_tensor(a, "a");
  require_cuda_tensor(b, "b");

  const std::size_t n = numel(a);
  if (numel(b) != n) {
    throw std::invalid_argument("a and b must have the same number of elements");
  }

  py::module_ torch = py::module_::import("torch");
  py::object out = torch.attr("empty_like")(a);

  const auto* a_ptr = reinterpret_cast<const float*>(data_ptr(a));
  const auto* b_ptr = reinterpret_cast<const float*>(data_ptr(b));
  auto* out_ptr = reinterpret_cast<float*>(data_ptr(out));

  launch_vector_add_f32(a_ptr, b_ptr, out_ptr, n, nullptr);
  hip_check(hipGetLastError(), "vector_add_f32 launch");
  return out;
}

void vector_add_f32_ptr(std::uintptr_t a, std::uintptr_t b,
                        std::uintptr_t out, std::size_t n,
                        std::uintptr_t stream = 0) {
  launch_vector_add_f32(reinterpret_cast<const float*>(a),
                        reinterpret_cast<const float*>(b),
                        reinterpret_cast<float*>(out),
                        n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "vector_add_f32_ptr launch");
}

void patch_im2col_ptr(std::uintptr_t input, std::uintptr_t output, int nv,
                      std::uintptr_t stream = 0) {
  if (nv <= 0) {
    throw std::invalid_argument("nv must be positive");
  }
  launch_patch_im2col_u16(reinterpret_cast<const void*>(input),
                          reinterpret_cast<void*>(output),
                          nv, stream_from_uint(stream));
  hip_check(hipGetLastError(), "patch_im2col_ptr launch");
}

void patch_embed_bias_pos_bf16_ptr(std::uintptr_t output, std::uintptr_t bias,
                                   std::uintptr_t pos_emb, int s, int d,
                                   int s_per_view,
                                   std::uintptr_t stream = 0) {
  if (s <= 0 || d <= 0 || s_per_view <= 0) {
    throw std::invalid_argument("s, d, and s_per_view must be positive");
  }
  launch_patch_embed_bias_pos_bf16(
      reinterpret_cast<void*>(output),
      reinterpret_cast<const void*>(bias),
      reinterpret_cast<const void*>(pos_emb),
      s, d, s_per_view, stream_from_uint(stream));
  hip_check(hipGetLastError(), "patch_embed_bias_pos_bf16_ptr launch");
}

void layer_norm_bf16_ptr(std::uintptr_t x, std::uintptr_t weight,
                         std::uintptr_t bias, std::uintptr_t out,
                         int rows, int hidden, double eps,
                         std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  launch_layer_norm_bf16(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<const void*>(bias),
      reinterpret_cast<void*>(out),
      rows, hidden, static_cast<float>(eps), stream_from_uint(stream));
  hip_check(hipGetLastError(), "layer_norm_bf16_ptr launch");
}

void add_bias_bf16_ptr(std::uintptr_t x, std::uintptr_t bias,
                       int rows, int hidden, std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  launch_add_bias_bf16(
      reinterpret_cast<void*>(x),
      reinterpret_cast<const void*>(bias),
      rows, hidden, stream_from_uint(stream));
  hip_check(hipGetLastError(), "add_bias_bf16_ptr launch");
}

void bias_residual_bf16_ptr(std::uintptr_t residual, std::uintptr_t x,
                            std::uintptr_t bias, int rows, int hidden,
                            std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  launch_bias_residual_bf16(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(bias),
      rows, hidden, stream_from_uint(stream));
  hip_check(hipGetLastError(), "bias_residual_bf16_ptr launch");
}

void residual_add_bf16_ptr(std::uintptr_t residual, std::uintptr_t x,
                           std::size_t n, std::uintptr_t stream = 0) {
  launch_residual_add_bf16(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "residual_add_bf16_ptr launch");
}

void gate_mul_residual_bf16_ptr(std::uintptr_t residual, std::uintptr_t x,
                                std::uintptr_t gate, std::size_t n,
                                std::uintptr_t stream = 0) {
  launch_gate_mul_residual_bf16(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(gate),
      n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "gate_mul_residual_bf16_ptr launch");
}

void residual_add_rms_norm_bf16_ptr(
    std::uintptr_t residual, std::uintptr_t x, std::uintptr_t weight,
    std::uintptr_t out, int rows, int hidden, double eps,
    std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  launch_residual_add_rms_norm_bf16(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<void*>(out),
      rows, hidden, static_cast<float>(eps), stream_from_uint(stream));
  hip_check(hipGetLastError(), "residual_add_rms_norm_bf16_ptr launch");
}

void residual_add_rms_norm_fp8_e4m3fnuz_ptr(
    std::uintptr_t residual, std::uintptr_t x, std::uintptr_t weight,
    std::uintptr_t out, std::uintptr_t scale, int rows, int hidden,
    double eps, std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_residual_add_rms_norm_fp8_e4m3fnuz(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<void*>(out),
      rows, hidden, static_cast<float>(eps),
      reinterpret_cast<const float*>(scale), stream_from_uint(stream));
  hip_check(hipGetLastError(), "residual_add_rms_norm_fp8_e4m3fnuz_ptr launch");
}

void ada_rms_norm_style_bf16_ptr(
    std::uintptr_t x, std::uintptr_t weight, std::uintptr_t style,
    std::uintptr_t out, std::uintptr_t gate_out,
    int rows, int hidden, double eps, std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  launch_ada_rms_norm_style_bf16(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<const void*>(style),
      reinterpret_cast<void*>(out),
      reinterpret_cast<void*>(gate_out),
      rows, hidden, static_cast<float>(eps), stream_from_uint(stream));
  hip_check(hipGetLastError(), "ada_rms_norm_style_bf16_ptr launch");
}

void ada_rms_norm_style_fp8_e4m3fnuz_ptr(
    std::uintptr_t x, std::uintptr_t weight, std::uintptr_t style,
    std::uintptr_t out, std::uintptr_t gate_out, std::uintptr_t scale,
    int rows, int hidden, double eps, std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_ada_rms_norm_style_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<const void*>(style),
      reinterpret_cast<void*>(out),
      reinterpret_cast<void*>(gate_out),
      rows, hidden, static_cast<float>(eps),
      reinterpret_cast<const float*>(scale), stream_from_uint(stream));
  hip_check(hipGetLastError(), "ada_rms_norm_style_fp8_e4m3fnuz_ptr launch");
}

void gate_residual_ada_norm_bf16_ptr(
    std::uintptr_t residual, std::uintptr_t x, std::uintptr_t gate,
    std::uintptr_t weight, std::uintptr_t style,
    std::uintptr_t out, std::uintptr_t gate_out,
    int rows, int hidden, double eps, std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  launch_gate_residual_ada_norm_bf16(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(gate),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<const void*>(style),
      reinterpret_cast<void*>(out),
      reinterpret_cast<void*>(gate_out),
      rows, hidden, static_cast<float>(eps), stream_from_uint(stream));
  hip_check(hipGetLastError(), "gate_residual_ada_norm_bf16_ptr launch");
}

void gate_residual_ada_norm_fp8_e4m3fnuz_ptr(
    std::uintptr_t residual, std::uintptr_t x, std::uintptr_t gate,
    std::uintptr_t weight, std::uintptr_t style,
    std::uintptr_t out, std::uintptr_t gate_out, std::uintptr_t scale,
    int rows, int hidden, double eps, std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_gate_residual_ada_norm_fp8_e4m3fnuz(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(gate),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<const void*>(style),
      reinterpret_cast<void*>(out),
      reinterpret_cast<void*>(gate_out),
      rows, hidden, static_cast<float>(eps),
      reinterpret_cast<const float*>(scale), stream_from_uint(stream));
  hip_check(hipGetLastError(), "gate_residual_ada_norm_fp8_e4m3fnuz_ptr launch");
}

void bias_residual_layer_norm_bf16_ptr(
    std::uintptr_t residual, std::uintptr_t x, std::uintptr_t bias_pre,
    std::uintptr_t norm_weight, std::uintptr_t norm_bias, std::uintptr_t out,
    int rows, int hidden, double eps, std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  launch_bias_residual_layer_norm_bf16(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(bias_pre),
      reinterpret_cast<const void*>(norm_weight),
      reinterpret_cast<const void*>(norm_bias),
      reinterpret_cast<void*>(out),
      rows, hidden, static_cast<float>(eps), stream_from_uint(stream));
  hip_check(hipGetLastError(), "bias_residual_layer_norm_bf16_ptr launch");
}

void layer_norm_fp8_e4m3fnuz_ptr(
    std::uintptr_t x, std::uintptr_t weight, std::uintptr_t bias,
    std::uintptr_t out, std::uintptr_t scale, int rows, int hidden,
    double eps, std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_layer_norm_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<const void*>(bias),
      reinterpret_cast<void*>(out),
      reinterpret_cast<const float*>(scale),
      rows, hidden, static_cast<float>(eps), stream_from_uint(stream));
  hip_check(hipGetLastError(), "layer_norm_fp8_e4m3fnuz_ptr launch");
}

void bias_residual_layer_norm_fp8_e4m3fnuz_ptr(
    std::uintptr_t residual, std::uintptr_t x, std::uintptr_t bias_pre,
    std::uintptr_t norm_weight, std::uintptr_t norm_bias, std::uintptr_t out,
    std::uintptr_t scale, int rows, int hidden, double eps,
    std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_bias_residual_layer_norm_fp8_e4m3fnuz(
      reinterpret_cast<void*>(residual),
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(bias_pre),
      reinterpret_cast<const void*>(norm_weight),
      reinterpret_cast<const void*>(norm_bias),
      reinterpret_cast<void*>(out),
      reinterpret_cast<const float*>(scale),
      rows, hidden, static_cast<float>(eps), stream_from_uint(stream));
  hip_check(hipGetLastError(), "bias_residual_layer_norm_fp8_e4m3fnuz_ptr");
}

void qkv_split_bf16_ptr(std::uintptr_t qkv, std::uintptr_t q,
                        std::uintptr_t k, std::uintptr_t v,
                        int seq, int q_dim, int k_dim, int v_dim,
                        std::uintptr_t stream = 0) {
  if (seq <= 0 || q_dim <= 0 || k_dim <= 0 || v_dim <= 0) {
    throw std::invalid_argument("seq, q_dim, k_dim, and v_dim must be positive");
  }
  launch_qkv_split_bf16(
      reinterpret_cast<const void*>(qkv),
      reinterpret_cast<void*>(q),
      reinterpret_cast<void*>(k),
      reinterpret_cast<void*>(v),
      seq, q_dim, k_dim, v_dim, stream_from_uint(stream));
  hip_check(hipGetLastError(), "qkv_split_bf16_ptr launch");
}

void gelu_tanh_bf16_ptr(std::uintptr_t x, std::uintptr_t out, std::size_t n,
                        std::uintptr_t stream = 0) {
  launch_gelu_tanh_bf16(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<void*>(out),
      n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "gelu_tanh_bf16_ptr launch");
}

void gelu_tanh_quantize_fp8_e4m3fnuz_ptr(
    std::uintptr_t x, std::uintptr_t scale, std::uintptr_t out,
    std::size_t n, std::uintptr_t stream = 0) {
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_gelu_tanh_quantize_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<void*>(out),
      n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "gelu_tanh_quantize_fp8_e4m3fnuz_ptr");
}

void gelu_tanh_mul_bf16_ptr(std::uintptr_t gate, std::uintptr_t up,
                            std::uintptr_t out, std::size_t n,
                            std::uintptr_t stream = 0) {
  launch_gelu_tanh_mul_bf16(
      reinterpret_cast<const void*>(gate),
      reinterpret_cast<const void*>(up),
      reinterpret_cast<void*>(out),
      n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "gelu_tanh_mul_bf16_ptr launch");
}

void rms_norm_bf16_ptr(std::uintptr_t x, std::uintptr_t weight,
                       std::uintptr_t out, int rows, int hidden, double eps,
                       std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  launch_rms_norm_bf16(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<void*>(out),
      rows, hidden, static_cast<float>(eps), stream_from_uint(stream));
  hip_check(hipGetLastError(), "rms_norm_bf16_ptr launch");
}

void rms_norm_fp8_e4m3fnuz_ptr(
    std::uintptr_t x, std::uintptr_t weight, std::uintptr_t out,
    std::uintptr_t scale, int rows, int hidden, double eps,
    std::uintptr_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    throw std::invalid_argument("rows and hidden must be positive");
  }
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_rms_norm_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<void*>(out),
      rows, hidden, static_cast<float>(eps),
      reinterpret_cast<const float*>(scale), stream_from_uint(stream));
  hip_check(hipGetLastError(), "rms_norm_fp8_e4m3fnuz_ptr launch");
}

void qkv_split_rope_bf16_ptr(std::uintptr_t qkv, std::uintptr_t rope,
                             std::uintptr_t q, std::uintptr_t k,
                             std::uintptr_t v, int seq, int q_dim,
                             int k_dim, int v_dim, int head_dim,
                             std::uintptr_t stream = 0) {
  if (seq <= 0 || q_dim <= 0 || k_dim <= 0 || v_dim <= 0 || head_dim <= 0) {
    throw std::invalid_argument("qkv split rope dimensions must be positive");
  }
  launch_qkv_split_rope_bf16(
      reinterpret_cast<const void*>(qkv),
      reinterpret_cast<const void*>(rope),
      reinterpret_cast<void*>(q),
      reinterpret_cast<void*>(k),
      reinterpret_cast<void*>(v),
      seq, q_dim, k_dim, v_dim, head_dim, stream_from_uint(stream));
  hip_check(hipGetLastError(), "qkv_split_rope_bf16_ptr launch");
}

py::object rms_norm(py::object x, py::object weight, double eps) {
  require_contiguous_cuda_tensor(x, "x");
  require_contiguous_cuda_tensor(weight, "weight");

  if (dim(x) < 2) {
    throw std::invalid_argument("x must have at least 2 dimensions");
  }
  if (dim(weight) != 1) {
    throw std::invalid_argument("weight must be a 1D tensor");
  }

  const int64_t hidden64 = size_at(x, -1);
  if (size_at(weight, 0) != hidden64) {
    throw std::invalid_argument("weight length must match x.size(-1)");
  }
  const std::size_t n = numel(x);
  if (n % static_cast<std::size_t>(hidden64) != 0) {
    throw std::invalid_argument("x.numel() must be divisible by x.size(-1)");
  }

  py::module_ torch = py::module_::import("torch");
  py::object out = torch.attr("empty_like")(x);

  const int rows = static_cast<int>(n / static_cast<std::size_t>(hidden64));
  const int hidden = static_cast<int>(hidden64);
  const float eps_f = static_cast<float>(eps);
  const std::string x_dtype = str_attr(x, "dtype");
  const std::string w_dtype = str_attr(weight, "dtype");

  if (x_dtype.find("float32") != std::string::npos &&
      w_dtype.find("float32") != std::string::npos) {
    launch_rms_norm_f32(reinterpret_cast<const float*>(data_ptr(x)),
                        reinterpret_cast<const float*>(data_ptr(weight)),
                        reinterpret_cast<float*>(data_ptr(out)),
                        rows, hidden, eps_f, nullptr);
  } else if (x_dtype.find("bfloat16") != std::string::npos &&
             w_dtype.find("bfloat16") != std::string::npos) {
    launch_rms_norm_bf16(reinterpret_cast<const void*>(data_ptr(x)),
                         reinterpret_cast<const void*>(data_ptr(weight)),
                         reinterpret_cast<void*>(data_ptr(out)),
                         rows, hidden, eps_f, nullptr);
  } else if (x_dtype.find("bfloat16") != std::string::npos &&
             w_dtype.find("float32") != std::string::npos) {
    launch_rms_norm_bf16_f32w(reinterpret_cast<const void*>(data_ptr(x)),
                              reinterpret_cast<const float*>(data_ptr(weight)),
                              reinterpret_cast<void*>(data_ptr(out)),
                              rows, hidden, eps_f, nullptr);
  } else {
    std::ostringstream oss;
    oss << "unsupported dtype combination: x=" << x_dtype
        << ", weight=" << w_dtype
        << "; expected both float32 or both bfloat16";
    throw std::invalid_argument(oss.str());
  }

  hip_check(hipGetLastError(), "rms_norm launch");
  return out;
}

py::dict hipblaslt_probe_dict() {
  const HipblasLtProbeResult result = probe_hipblaslt();
  py::dict out;
  out["available"] = result.available;
  out["status_code"] = result.status_code;
  out["status_name"] = result.status_name;
  out["version"] = result.version;
  return out;
}

py::object hipblaslt_matmul_bf16_py(py::object a, py::object b) {
  require_bfloat16_tensor(a, "a");
  require_bfloat16_tensor(b, "b");
  if (dim(a) != 2 || dim(b) != 2) {
    throw std::invalid_argument("a and b must be 2D tensors");
  }

  const int64_t m = size_at(a, 0);
  const int64_t k = size_at(a, 1);
  const int64_t b_k = size_at(b, 0);
  const int64_t n = size_at(b, 1);
  if (b_k != k) {
    throw std::invalid_argument("a.size(1) must equal b.size(0)");
  }

  py::object out = a.attr("new_empty")(py::make_tuple(m, n));
  hipblaslt_matmul_bf16(reinterpret_cast<const void*>(data_ptr(a)),
                        reinterpret_cast<const void*>(data_ptr(b)),
                        reinterpret_cast<void*>(data_ptr(out)),
                        m, n, k, nullptr);
  hip_check(hipGetLastError(), "hipblaslt_matmul_bf16");
  return out;
}

py::object hipblaslt_matmul_fp8_e4m3fnuz_bf16_py(
    py::object a, py::object b, py::object a_scale, py::object b_scale) {
  require_float8_e4m3fnuz_tensor(a, "a");
  require_float8_e4m3fnuz_tensor(b, "b");
  require_float32_scalar_tensor(a_scale, "a_scale");
  require_float32_scalar_tensor(b_scale, "b_scale");
  if (dim(a) != 2 || dim(b) != 2) {
    throw std::invalid_argument("a and b must be 2D tensors");
  }

  const int64_t m = size_at(a, 0);
  const int64_t k = size_at(a, 1);
  const int64_t b_k = size_at(b, 0);
  const int64_t n = size_at(b, 1);
  if (b_k != k) {
    throw std::invalid_argument("a.size(1) must equal b.size(0)");
  }

  py::module_ torch = py::module_::import("torch");
  py::object out = torch.attr("empty")(
      py::make_tuple(m, n),
      py::arg("device") = a.attr("device"),
      py::arg("dtype") = torch.attr("bfloat16"));

  hipblaslt_matmul_fp8_e4m3fnuz_bf16(
      reinterpret_cast<const void*>(data_ptr(a)),
      reinterpret_cast<const void*>(data_ptr(b)),
      reinterpret_cast<const float*>(data_ptr(a_scale)),
      reinterpret_cast<const float*>(data_ptr(b_scale)),
      reinterpret_cast<void*>(data_ptr(out)),
      m, n, k, nullptr);
  hip_check(hipGetLastError(), "hipblaslt_matmul_fp8_e4m3fnuz_bf16");
  return out;
}

py::object hipblaslt_linear_bf16_py(py::object x, py::object weight,
                                    py::object bias) {
  require_bfloat16_tensor(x, "x");
  require_bfloat16_tensor(weight, "weight");
  const bool has_bias = !bias.is_none();
  if (has_bias) {
    require_bfloat16_tensor(bias, "bias");
  }
  if (dim(x) < 2) {
    throw std::invalid_argument("x must have at least 2 dimensions");
  }
  if (dim(weight) != 2) {
    throw std::invalid_argument("weight must be a 2D tensor");
  }

  const int64_t k = size_at(x, -1);
  const int64_t n = size_at(weight, 0);
  if (size_at(weight, 1) != k) {
    throw std::invalid_argument("weight.size(1) must equal x.size(-1)");
  }
  if (has_bias && (dim(bias) != 1 || size_at(bias, 0) != n)) {
    throw std::invalid_argument("bias must be 1D and match weight.size(0)");
  }

  py::tuple out_shape(dim(x));
  std::size_t rows = 1;
  for (int64_t axis = 0; axis < dim(x) - 1; ++axis) {
    const int64_t s = size_at(x, axis);
    out_shape[axis] = py::int_(s);
    rows *= static_cast<std::size_t>(s);
  }
  out_shape[dim(x) - 1] = py::int_(n);

  py::object out = x.attr("new_empty")(out_shape);
  hipblaslt_linear_bf16(
      reinterpret_cast<const void*>(data_ptr(x)),
      reinterpret_cast<const void*>(data_ptr(weight)),
      has_bias ? reinterpret_cast<const void*>(data_ptr(bias)) : nullptr,
      reinterpret_cast<void*>(data_ptr(out)),
      static_cast<int64_t>(rows), n, k, nullptr);
  hip_check(hipGetLastError(), "hipblaslt_linear_bf16");
  return out;
}

void hipblaslt_linear_bf16_ptr_py(
    std::uintptr_t x, std::uintptr_t weight, std::uintptr_t bias,
    std::uintptr_t out, int64_t m, int64_t n, int64_t k,
    std::uintptr_t stream = 0) {
  hipblaslt_linear_bf16(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      bias == 0 ? nullptr : reinterpret_cast<const void*>(bias),
      reinterpret_cast<void*>(out),
      m, n, k, stream_from_uint(stream));
  hip_check(hipGetLastError(), "hipblaslt_linear_bf16_ptr");
}

void hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr_py(
    std::uintptr_t x, std::uintptr_t weight,
    std::uintptr_t x_scale, std::uintptr_t weight_scale,
    std::uintptr_t bias, std::uintptr_t out,
    int64_t m, int64_t n, int64_t k,
    std::uintptr_t stream = 0) {
  if (x_scale == 0 || weight_scale == 0) {
    throw std::invalid_argument("FP8 linear requires non-null scale pointers");
  }
  hipblaslt_linear_fp8_e4m3fnuz_bf16(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const void*>(weight),
      reinterpret_cast<const float*>(x_scale),
      reinterpret_cast<const float*>(weight_scale),
      bias == 0 ? nullptr : reinterpret_cast<const void*>(bias),
      reinterpret_cast<void*>(out),
      m, n, k, stream_from_uint(stream));
  hip_check(hipGetLastError(), "hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr");
}

void hipblaslt_linear_bf16_out_py(
    py::object x, py::object weight, py::object out, py::object bias) {
  require_bfloat16_tensor(x, "x");
  require_bfloat16_tensor(weight, "weight");
  require_bfloat16_tensor(out, "out");
  const bool has_bias = !bias.is_none();
  if (has_bias) {
    require_bfloat16_tensor(bias, "bias");
  }
  if (dim(x) < 2) {
    throw std::invalid_argument("x must have at least 2 dimensions");
  }
  if (dim(weight) != 2) {
    throw std::invalid_argument("weight must be a 2D tensor");
  }
  if (dim(out) != dim(x)) {
    throw std::invalid_argument("out.dim() must match x.dim()");
  }

  const int64_t k = size_at(x, -1);
  const int64_t n = size_at(weight, 0);
  if (size_at(weight, 1) != k) {
    throw std::invalid_argument("weight.size(1) must equal x.size(-1)");
  }
  if (has_bias && (dim(bias) != 1 || size_at(bias, 0) != n)) {
    throw std::invalid_argument("bias must be 1D and match weight.size(0)");
  }

  std::size_t rows = 1;
  for (int64_t axis = 0; axis < dim(x) - 1; ++axis) {
    const int64_t s = size_at(x, axis);
    if (size_at(out, axis) != s) {
      throw std::invalid_argument("out shape prefix must match x shape prefix");
    }
    rows *= static_cast<std::size_t>(s);
  }
  if (size_at(out, dim(out) - 1) != n) {
    throw std::invalid_argument("out.size(-1) must match weight.size(0)");
  }

  hipblaslt_linear_bf16(
      reinterpret_cast<const void*>(data_ptr(x)),
      reinterpret_cast<const void*>(data_ptr(weight)),
      has_bias ? reinterpret_cast<const void*>(data_ptr(bias)) : nullptr,
      reinterpret_cast<void*>(data_ptr(out)),
      static_cast<int64_t>(rows), n, k, nullptr);
  hip_check(hipGetLastError(), "hipblaslt_linear_bf16_out");
}

py::object hipblaslt_linear_fp8_e4m3fnuz_bf16_py(
    py::object x, py::object weight, py::object x_scale,
    py::object weight_scale, py::object bias) {
  require_float8_e4m3fnuz_tensor(x, "x");
  require_float8_e4m3fnuz_tensor(weight, "weight");
  require_float32_scalar_tensor(x_scale, "x_scale");
  require_float32_scalar_tensor(weight_scale, "weight_scale");
  const bool has_bias = !bias.is_none();
  if (has_bias) {
    require_bfloat16_tensor(bias, "bias");
  }
  if (dim(x) < 2) {
    throw std::invalid_argument("x must have at least 2 dimensions");
  }
  if (dim(weight) != 2) {
    throw std::invalid_argument("weight must be a 2D tensor");
  }

  const int64_t k = size_at(x, -1);
  const int64_t n = size_at(weight, 0);
  if (size_at(weight, 1) != k) {
    throw std::invalid_argument("weight.size(1) must equal x.size(-1)");
  }
  if (has_bias && (dim(bias) != 1 || size_at(bias, 0) != n)) {
    throw std::invalid_argument("bias must be 1D and match weight.size(0)");
  }

  py::tuple out_shape(dim(x));
  std::size_t rows = 1;
  for (int64_t axis = 0; axis < dim(x) - 1; ++axis) {
    const int64_t s = size_at(x, axis);
    out_shape[axis] = py::int_(s);
    rows *= static_cast<std::size_t>(s);
  }
  out_shape[dim(x) - 1] = py::int_(n);

  py::module_ torch = py::module_::import("torch");
  py::object out = torch.attr("empty")(
      out_shape,
      py::arg("device") = x.attr("device"),
      py::arg("dtype") = torch.attr("bfloat16"));

  hipblaslt_linear_fp8_e4m3fnuz_bf16(
      reinterpret_cast<const void*>(data_ptr(x)),
      reinterpret_cast<const void*>(data_ptr(weight)),
      reinterpret_cast<const float*>(data_ptr(x_scale)),
      reinterpret_cast<const float*>(data_ptr(weight_scale)),
      has_bias ? reinterpret_cast<const void*>(data_ptr(bias)) : nullptr,
      reinterpret_cast<void*>(data_ptr(out)),
      static_cast<int64_t>(rows), n, k, nullptr);
  hip_check(hipGetLastError(), "hipblaslt_linear_fp8_e4m3fnuz_bf16");
  return out;
}

void hipblaslt_linear_fp8_e4m3fnuz_bf16_out_py(
    py::object x, py::object weight, py::object x_scale,
    py::object weight_scale, py::object out, py::object bias) {
  require_float8_e4m3fnuz_tensor(x, "x");
  require_float8_e4m3fnuz_tensor(weight, "weight");
  require_float32_scalar_tensor(x_scale, "x_scale");
  require_float32_scalar_tensor(weight_scale, "weight_scale");
  require_bfloat16_tensor(out, "out");

  const bool has_bias = !bias.is_none();
  if (has_bias) {
    require_bfloat16_tensor(bias, "bias");
  }
  if (dim(x) < 2) {
    throw std::invalid_argument("x must have at least 2 dimensions");
  }
  if (dim(weight) != 2) {
    throw std::invalid_argument("weight must be 2D");
  }

  const int64_t k = size_at(x, -1);
  const int64_t n = size_at(weight, 0);
  if (size_at(weight, 1) != k) {
    throw std::invalid_argument("x.size(-1) must equal weight.size(1)");
  }
  if (has_bias && (dim(bias) != 1 || size_at(bias, 0) != n)) {
    throw std::invalid_argument("bias must be 1D and match weight.size(0)");
  }
  if (dim(out) != dim(x)) {
    throw std::invalid_argument("out.dim() must match x.dim()");
  }

  std::size_t rows = 1;
  for (int64_t axis = 0; axis < dim(x) - 1; ++axis) {
    const int64_t s = size_at(x, axis);
    if (size_at(out, axis) != s) {
      throw std::invalid_argument("out shape prefix must match x shape prefix");
    }
    rows *= static_cast<std::size_t>(s);
  }
  if (size_at(out, dim(out) - 1) != n) {
    throw std::invalid_argument("out.size(-1) must match weight.size(0)");
  }

  hipblaslt_linear_fp8_e4m3fnuz_bf16(
      reinterpret_cast<const void*>(data_ptr(x)),
      reinterpret_cast<const void*>(data_ptr(weight)),
      reinterpret_cast<const float*>(data_ptr(x_scale)),
      reinterpret_cast<const float*>(data_ptr(weight_scale)),
      has_bias ? reinterpret_cast<const void*>(data_ptr(bias)) : nullptr,
      reinterpret_cast<void*>(data_ptr(out)),
      static_cast<int64_t>(rows), n, k, nullptr);
  hip_check(hipGetLastError(), "hipblaslt_linear_fp8_e4m3fnuz_bf16_out");
}

py::object gelu_tanh_mul_bf16(py::object gate, py::object up) {
  require_bfloat16_tensor(gate, "gate");
  require_bfloat16_tensor(up, "up");
  if (numel(gate) != numel(up)) {
    throw std::invalid_argument("gate and up must have the same number of elements");
  }
  py::object out = gate.attr("new_empty")(gate.attr("shape"));
  launch_gelu_tanh_mul_bf16(
      reinterpret_cast<const void*>(data_ptr(gate)),
      reinterpret_cast<const void*>(data_ptr(up)),
      reinterpret_cast<void*>(data_ptr(out)),
      numel(gate), nullptr);
  hip_check(hipGetLastError(), "gelu_tanh_mul_bf16");
  return out;
}

void gelu_tanh_mul_bf16_out(py::object gate, py::object up, py::object out) {
  require_bfloat16_tensor(gate, "gate");
  require_bfloat16_tensor(up, "up");
  require_bfloat16_tensor(out, "out");
  if (numel(gate) != numel(up)) {
    throw std::invalid_argument("gate and up must have the same number of elements");
  }
  if (numel(out) != numel(gate)) {
    throw std::invalid_argument("out.numel() must match gate.numel()");
  }
  launch_gelu_tanh_mul_bf16(
      reinterpret_cast<const void*>(data_ptr(gate)),
      reinterpret_cast<const void*>(data_ptr(up)),
      reinterpret_cast<void*>(data_ptr(out)),
      numel(gate), nullptr);
  hip_check(hipGetLastError(), "gelu_tanh_mul_bf16_out");
}

void gelu_tanh_mul_quantize_fp8_e4m3fnuz_out(
    py::object gate, py::object up, py::object scale, py::object out) {
  require_bfloat16_tensor(gate, "gate");
  require_bfloat16_tensor(up, "up");
  require_float32_scalar_tensor(scale, "scale");
  require_float8_e4m3fnuz_tensor(out, "out");
  if (numel(gate) != numel(up)) {
    throw std::invalid_argument("gate and up must have the same number of elements");
  }
  if (numel(out) != numel(gate)) {
    throw std::invalid_argument("out.numel() must match gate.numel()");
  }
  launch_gelu_tanh_mul_quantize_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(data_ptr(gate)),
      reinterpret_cast<const void*>(data_ptr(up)),
      reinterpret_cast<const float*>(data_ptr(scale)),
      reinterpret_cast<void*>(data_ptr(out)),
      numel(gate), nullptr);
  hip_check(hipGetLastError(), "gelu_tanh_mul_quantize_fp8_e4m3fnuz_out");
}

void gelu_tanh_mul_quantize_fp8_e4m3fnuz_ptr(
    std::uintptr_t gate, std::uintptr_t up, std::uintptr_t scale,
    std::uintptr_t out, std::size_t n, std::uintptr_t stream = 0) {
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_gelu_tanh_mul_quantize_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(gate),
      reinterpret_cast<const void*>(up),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<void*>(out),
      n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "gelu_tanh_mul_quantize_fp8_e4m3fnuz_ptr");
}

void gelu_tanh_merged_bf16_ptr(
    std::uintptr_t gate_up, std::uintptr_t out, int rows, int hidden,
    std::uintptr_t stream = 0) {
  launch_gelu_tanh_merged_bf16(
      reinterpret_cast<const void*>(gate_up),
      reinterpret_cast<void*>(out),
      rows, hidden, stream_from_uint(stream));
  hip_check(hipGetLastError(), "gelu_tanh_merged_bf16_ptr launch");
}

void gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr(
    std::uintptr_t gate_up, std::uintptr_t scale, std::uintptr_t out,
    int rows, int hidden, std::uintptr_t stream = 0) {
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_gelu_tanh_merged_quantize_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(gate_up),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<void*>(out),
      rows, hidden, stream_from_uint(stream));
  hip_check(hipGetLastError(), "gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr");
}

py::object silu_bf16(py::object x) {
  require_bfloat16_tensor(x, "x");
  py::object out = x.attr("new_empty")(x.attr("shape"));
  launch_silu_bf16(reinterpret_cast<const void*>(data_ptr(x)),
                   reinterpret_cast<void*>(data_ptr(out)),
                   numel(x), nullptr);
  hip_check(hipGetLastError(), "silu_bf16");
  return out;
}

py::object gelu_tanh_bf16(py::object x) {
  require_bfloat16_tensor(x, "x");
  py::object out = x.attr("new_empty")(x.attr("shape"));
  launch_gelu_tanh_bf16(reinterpret_cast<const void*>(data_ptr(x)),
                        reinterpret_cast<void*>(data_ptr(out)),
                        numel(x), nullptr);
  hip_check(hipGetLastError(), "gelu_tanh_bf16");
  return out;
}

py::object quantize_to_fp8_e4m3fnuz(py::object x, py::object scale) {
  require_contiguous_cuda_tensor(x, "x");
  require_float32_scalar_tensor(scale, "scale");

  py::module_ torch = py::module_::import("torch");
  py::object out = torch.attr("empty")(
      x.attr("shape"),
      py::arg("device") = x.attr("device"),
      py::arg("dtype") = torch.attr("float8_e4m3fnuz"));

  const std::string dtype = str_attr(x, "dtype");
  if (dtype.find("float32") != std::string::npos) {
    launch_quantize_f32_to_fp8_e4m3fnuz(
        reinterpret_cast<const float*>(data_ptr(x)),
        reinterpret_cast<const float*>(data_ptr(scale)),
        reinterpret_cast<void*>(data_ptr(out)),
        numel(x), nullptr);
  } else if (dtype.find("bfloat16") != std::string::npos) {
    launch_quantize_bf16_to_fp8_e4m3fnuz(
        reinterpret_cast<const void*>(data_ptr(x)),
        reinterpret_cast<const float*>(data_ptr(scale)),
        reinterpret_cast<void*>(data_ptr(out)),
        numel(x), nullptr);
  } else {
    std::ostringstream oss;
    oss << "x must be torch.float32 or torch.bfloat16, got " << dtype;
    throw std::invalid_argument(oss.str());
  }
  hip_check(hipGetLastError(), "quantize_to_fp8_e4m3fnuz");
  return out;
}

void quantize_to_fp8_e4m3fnuz_out(py::object x, py::object scale, py::object out) {
  require_contiguous_cuda_tensor(x, "x");
  require_float32_scalar_tensor(scale, "scale");
  require_float8_e4m3fnuz_tensor(out, "out");
  if (numel(out) != numel(x)) {
    throw std::invalid_argument("out.numel() must match x.numel()");
  }

  const std::string dtype = str_attr(x, "dtype");
  if (dtype.find("float32") != std::string::npos) {
    launch_quantize_f32_to_fp8_e4m3fnuz(
        reinterpret_cast<const float*>(data_ptr(x)),
        reinterpret_cast<const float*>(data_ptr(scale)),
        reinterpret_cast<void*>(data_ptr(out)),
        numel(x), nullptr);
  } else if (dtype.find("bfloat16") != std::string::npos) {
    launch_quantize_bf16_to_fp8_e4m3fnuz(
        reinterpret_cast<const void*>(data_ptr(x)),
        reinterpret_cast<const float*>(data_ptr(scale)),
        reinterpret_cast<void*>(data_ptr(out)),
        numel(x), nullptr);
  } else {
    std::ostringstream oss;
    oss << "x must be torch.float32 or torch.bfloat16, got " << dtype;
    throw std::invalid_argument(oss.str());
  }
  hip_check(hipGetLastError(), "quantize_to_fp8_e4m3fnuz_out");
}

void quantize_bf16_to_fp8_e4m3fnuz_ptr(
    std::uintptr_t x, std::uintptr_t scale, std::uintptr_t out,
    std::size_t n, std::uintptr_t stream = 0) {
  if (scale == 0) {
    throw std::invalid_argument("scale pointer must be non-null");
  }
  launch_quantize_bf16_to_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<void*>(out),
      n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "quantize_bf16_to_fp8_e4m3fnuz_ptr");
}

void dynamic_quantize_bf16_to_fp8_e4m3fnuz_ptr(
    std::uintptr_t x, std::uintptr_t out, std::uintptr_t scale,
    std::uintptr_t partial, int partial_count, std::size_t n,
    std::uintptr_t stream = 0) {
  if (scale == 0 || partial == 0) {
    throw std::invalid_argument("scale and partial pointers must be non-null");
  }
  if (partial_count <= 0) {
    throw std::invalid_argument("partial_count must be positive");
  }
  launch_dynamic_quantize_bf16_to_fp8_e4m3fnuz(
      reinterpret_cast<const void*>(x),
      reinterpret_cast<void*>(out),
      reinterpret_cast<float*>(scale),
      reinterpret_cast<float*>(partial),
      partial_count, n, stream_from_uint(stream));
  hip_check(hipGetLastError(), "dynamic_quantize_bf16_to_fp8_e4m3fnuz_ptr");
}

py::tuple dynamic_quantize_to_fp8_e4m3fnuz(py::object x) {
  require_contiguous_cuda_tensor(x, "x");

  py::module_ torch = py::module_::import("torch");
  py::object out = torch.attr("empty")(
      x.attr("shape"),
      py::arg("device") = x.attr("device"),
      py::arg("dtype") = torch.attr("float8_e4m3fnuz"));
  py::object scale = torch.attr("empty")(
      py::make_tuple(1),
      py::arg("device") = x.attr("device"),
      py::arg("dtype") = torch.attr("float32"));

  const std::size_t n = numel(x);
  const int blocks = static_cast<int>((n + 255) / 256);
  const int partial_count = blocks < 1 ? 1 : (blocks > 4096 ? 4096 : blocks);
  py::object partial = torch.attr("empty")(
      py::make_tuple(partial_count),
      py::arg("device") = x.attr("device"),
      py::arg("dtype") = torch.attr("float32"));

  const std::string dtype = str_attr(x, "dtype");
  if (dtype.find("float32") != std::string::npos) {
    launch_dynamic_quantize_f32_to_fp8_e4m3fnuz(
        reinterpret_cast<const float*>(data_ptr(x)),
        reinterpret_cast<void*>(data_ptr(out)),
        reinterpret_cast<float*>(data_ptr(scale)),
        reinterpret_cast<float*>(data_ptr(partial)),
        partial_count, n, nullptr);
  } else if (dtype.find("bfloat16") != std::string::npos) {
    launch_dynamic_quantize_bf16_to_fp8_e4m3fnuz(
        reinterpret_cast<const void*>(data_ptr(x)),
        reinterpret_cast<void*>(data_ptr(out)),
        reinterpret_cast<float*>(data_ptr(scale)),
        reinterpret_cast<float*>(data_ptr(partial)),
        partial_count, n, nullptr);
  } else {
    std::ostringstream oss;
    oss << "x must be torch.float32 or torch.bfloat16, got " << dtype;
    throw std::invalid_argument(oss.str());
  }
  hip_check(hipGetLastError(), "dynamic_quantize_to_fp8_e4m3fnuz");
  return py::make_tuple(out, scale);
}

}  // namespace

PYBIND11_MODULE(flash_rt_rocm_kernels, m) {
  m.doc() = "FlashRT ROCm kernels";

  m.def("has_rocm", []() { return true; });

  m.def("device_count", []() {
    int count = 0;
    hip_check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    return count;
  });

  m.def("device_name", [](int device) {
    hipDeviceProp_t props{};
    hip_check(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");
    return std::string(props.name);
  }, py::arg("device") = 0);

  m.def("hip_sync", []() { hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize"); });

  m.def("hipblaslt_probe", &hipblaslt_probe_dict,
        "Return hipBLASLt availability, status, and runtime version.");

  m.def("hipblaslt_available", []() { return probe_hipblaslt().available; },
        "Return true when a hipBLASLt handle can be created on the current device.");
  m.def("hipblaslt_algo_cache_size", &hipblaslt_algo_cache_size,
        "Return the number of cached hipBLASLt algorithm choices.");
  m.def("hipblaslt_algo_cache_keys", &hipblaslt_algo_cache_keys,
        "Return cached hipBLASLt algorithm keys.");
  m.def("hipblaslt_algo_cache_clear", &hipblaslt_algo_cache_clear,
        "Clear cached hipBLASLt algorithm choices.");
  m.def("hipblaslt_linear_plan_cache_size", &hipblaslt_linear_plan_cache_size,
        "Return the number of persistent BF16 Linear hipBLASLt plans.");
  m.def("hipblaslt_linear_plan_cache_keys", &hipblaslt_linear_plan_cache_keys,
        "Return persistent BF16 Linear hipBLASLt plan keys.");
  m.def("hipblaslt_linear_plan_cache_clear", &hipblaslt_linear_plan_cache_clear,
        "Clear persistent BF16 Linear hipBLASLt plans.");

  m.def("hipblaslt_matmul_bf16", &hipblaslt_matmul_bf16_py,
        py::arg("a"), py::arg("b"),
        "Return a @ b for contiguous row-major torch.bfloat16 HIP matrices.");

  m.def("hipblaslt_matmul_fp8_e4m3fnuz_bf16",
        &hipblaslt_matmul_fp8_e4m3fnuz_bf16_py,
        py::arg("a"), py::arg("b"), py::arg("a_scale"), py::arg("b_scale"),
        "Return dequantized a @ b for torch.float8_e4m3fnuz HIP matrices with scalar FP32 scales and BF16 output.");

  m.def("hipblaslt_linear_bf16", &hipblaslt_linear_bf16_py,
        py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
        "Return torch.nn.functional.linear(x, weight, bias) for contiguous BF16 HIP tensors.");

  m.def("hipblaslt_linear_bf16_ptr", &hipblaslt_linear_bf16_ptr_py,
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
        py::arg("m"), py::arg("n"), py::arg("k"), py::arg("stream") = 0,
        "Write BF16 linear output from raw HIP pointers.");

  m.def("hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr",
        &hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr_py,
        py::arg("x"), py::arg("weight"),
        py::arg("x_scale"), py::arg("weight_scale"),
        py::arg("bias"), py::arg("out"),
        py::arg("m"), py::arg("n"), py::arg("k"), py::arg("stream") = 0,
        "Write BF16 Linear output from raw FP8 HIP pointers and scalar scales.");

  m.def("hipblaslt_linear_bf16_out", &hipblaslt_linear_bf16_out_py,
        py::arg("x"), py::arg("weight"), py::arg("out"),
        py::arg("bias") = py::none(),
        "Write torch.nn.functional.linear(x, weight, bias) into a BF16 output tensor.");

  m.def("hipblaslt_linear_fp8_e4m3fnuz_bf16",
        &hipblaslt_linear_fp8_e4m3fnuz_bf16_py,
        py::arg("x"), py::arg("weight"),
        py::arg("x_scale"), py::arg("weight_scale"),
        py::arg("bias") = py::none(),
        "Return linear(x, weight, bias) for FP8 E4M3 FNUZ inputs/weights with scalar FP32 scales and BF16 output.");

  m.def("hipblaslt_linear_fp8_e4m3fnuz_bf16_out",
        &hipblaslt_linear_fp8_e4m3fnuz_bf16_out_py,
        py::arg("x"), py::arg("weight"),
        py::arg("x_scale"), py::arg("weight_scale"),
        py::arg("out"), py::arg("bias") = py::none(),
        "Write linear(x, weight, bias) for FP8 E4M3 FNUZ inputs/weights into a BF16 output tensor.");

  m.def("gelu_tanh_mul_bf16", &gelu_tanh_mul_bf16,
        py::arg("gate"), py::arg("up"),
        "Return gelu_tanh(gate) * up for contiguous BF16 HIP tensors.");

  m.def("gelu_tanh_mul_bf16_out", &gelu_tanh_mul_bf16_out,
        py::arg("gate"), py::arg("up"), py::arg("out"),
        "Write gelu_tanh(gate) * up into a BF16 output tensor.");

  m.def("gelu_tanh_mul_quantize_fp8_e4m3fnuz_out",
        &gelu_tanh_mul_quantize_fp8_e4m3fnuz_out,
        py::arg("gate"), py::arg("up"), py::arg("scale"), py::arg("out"),
        "Compute gelu_tanh(gate) * up and write a static FP8 E4M3 FNUZ output.");

  m.def("gelu_tanh_mul_quantize_fp8_e4m3fnuz_ptr",
        &gelu_tanh_mul_quantize_fp8_e4m3fnuz_ptr,
        py::arg("gate"), py::arg("up"), py::arg("scale"), py::arg("out"),
        py::arg("n"), py::arg("stream") = 0,
        "Compute gelu_tanh(gate) * up and write static FP8 output from raw HIP pointers.");

  m.def("gelu_tanh_merged_bf16_ptr",
        &gelu_tanh_merged_bf16_ptr,
        py::arg("gate_up"), py::arg("out"), py::arg("rows"),
        py::arg("hidden"), py::arg("stream") = 0,
        "Compute gelu_tanh(gate) * up from a row-major merged [gate, up] BF16 tensor.");

  m.def("gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr",
        &gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr,
        py::arg("gate_up"), py::arg("scale"), py::arg("out"),
        py::arg("rows"), py::arg("hidden"), py::arg("stream") = 0,
        "Compute merged GEGLU and write static FP8 output from raw HIP pointers.");

  m.def("silu_bf16", &silu_bf16, py::arg("x"),
        "Return silu(x) for a contiguous BF16 HIP tensor.");

  m.def("gelu_tanh_bf16", &gelu_tanh_bf16, py::arg("x"),
        "Return GELU tanh approximation for a contiguous BF16 HIP tensor.");

  m.def("quantize_to_fp8_e4m3fnuz", &quantize_to_fp8_e4m3fnuz,
        py::arg("x"), py::arg("scale"),
        "Return (x / scale).to(torch.float8_e4m3fnuz) for contiguous HIP tensors.");

  m.def("quantize_to_fp8_e4m3fnuz_out", &quantize_to_fp8_e4m3fnuz_out,
        py::arg("x"), py::arg("scale"), py::arg("out"),
        "Write (x / scale).to(torch.float8_e4m3fnuz) into a preallocated output tensor.");

  m.def("quantize_bf16_to_fp8_e4m3fnuz_ptr",
        &quantize_bf16_to_fp8_e4m3fnuz_ptr,
        py::arg("x"), py::arg("scale"), py::arg("out"),
        py::arg("n"), py::arg("stream") = 0,
        "Write static BF16 -> FP8 E4M3 FNUZ quantization from raw HIP pointers.");

  m.def("dynamic_quantize_bf16_to_fp8_e4m3fnuz_ptr",
        &dynamic_quantize_bf16_to_fp8_e4m3fnuz_ptr,
        py::arg("x"), py::arg("out"), py::arg("scale"), py::arg("partial"),
        py::arg("partial_count"), py::arg("n"), py::arg("stream") = 0,
        "Write dynamic BF16 -> FP8 quantization and scalar scale from raw HIP pointers.");

  m.def("dynamic_quantize_to_fp8_e4m3fnuz", &dynamic_quantize_to_fp8_e4m3fnuz,
        py::arg("x"),
        "Return (fp8, scale) where scale=max(abs(x))/240 computed on HIP.");

  m.def("vector_add_f32", &vector_add_f32, py::arg("a"), py::arg("b"),
        "Return a + b for contiguous torch.float32 HIP tensors.");
  m.def("vector_add_f32_ptr", &vector_add_f32_ptr,
        py::arg("a"), py::arg("b"), py::arg("out"), py::arg("n"),
        py::arg("stream") = 0,
        "Write a + b for raw float32 HIP pointers.");
  m.def("patch_im2col_ptr", &patch_im2col_ptr,
        py::arg("input"), py::arg("output"), py::arg("nv"),
        py::arg("stream") = 0,
        "Write SigLIP patch im2col from raw 16-bit NHWC image pointers.");
  m.def("patch_embed_bias_pos_bf16_ptr", &patch_embed_bias_pos_bf16_ptr,
        py::arg("output"), py::arg("bias"), py::arg("pos_emb"),
        py::arg("s"), py::arg("d"), py::arg("s_per_view"),
        py::arg("stream") = 0,
        "Add BF16 patch embedding bias and per-view positional embedding in place.");
  m.def("layer_norm_bf16_ptr", &layer_norm_bf16_ptr,
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-5,
        py::arg("stream") = 0,
        "Write BF16 LayerNorm output from raw HIP pointers.");
  m.def("add_bias_bf16_ptr", &add_bias_bf16_ptr,
        py::arg("x"), py::arg("bias"), py::arg("rows"), py::arg("hidden"),
        py::arg("stream") = 0,
        "Add a BF16 bias vector to a raw HIP matrix in place.");
  m.def("bias_residual_bf16_ptr", &bias_residual_bf16_ptr,
        py::arg("residual"), py::arg("x"), py::arg("bias"),
        py::arg("rows"), py::arg("hidden"), py::arg("stream") = 0,
        "Compute residual += x + bias for raw BF16 HIP pointers.");
  m.def("residual_add_bf16_ptr", &residual_add_bf16_ptr,
        py::arg("residual"), py::arg("x"), py::arg("n"),
        py::arg("stream") = 0,
        "Compute residual += x for raw BF16 HIP pointers.");
  m.def("gate_mul_residual_bf16_ptr", &gate_mul_residual_bf16_ptr,
        py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("n"),
        py::arg("stream") = 0,
        "Compute residual += x * gate for raw BF16 HIP pointers.");
  m.def("residual_add_rms_norm_bf16_ptr", &residual_add_rms_norm_bf16_ptr,
        py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-6,
        py::arg("stream") = 0,
        "Fuse residual += x and BF16 RMSNorm over the updated residual.");
  m.def("residual_add_rms_norm_fp8_e4m3fnuz_ptr",
        &residual_add_rms_norm_fp8_e4m3fnuz_ptr,
        py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
        py::arg("scale"), py::arg("rows"), py::arg("hidden"),
        py::arg("eps") = 1e-6, py::arg("stream") = 0,
        "Fuse residual += x and RMSNorm directly into static FP8 output.");
  m.def("ada_rms_norm_style_bf16_ptr", &ada_rms_norm_style_bf16_ptr,
        py::arg("x"), py::arg("weight"), py::arg("style"),
        py::arg("out"), py::arg("gate_out"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-6,
        py::arg("stream") = 0,
        "Write adaptive RMSNorm output and gate from precomputed BF16 style.");
  m.def("ada_rms_norm_style_fp8_e4m3fnuz_ptr",
        &ada_rms_norm_style_fp8_e4m3fnuz_ptr,
        py::arg("x"), py::arg("weight"), py::arg("style"),
        py::arg("out"), py::arg("gate_out"), py::arg("scale"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-6,
        py::arg("stream") = 0,
        "Write adaptive RMSNorm output directly into static FP8 plus BF16 gate.");
  m.def("gate_residual_ada_norm_bf16_ptr", &gate_residual_ada_norm_bf16_ptr,
        py::arg("residual"), py::arg("x"), py::arg("gate"),
        py::arg("weight"), py::arg("style"),
        py::arg("out"), py::arg("gate_out"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-6,
        py::arg("stream") = 0,
        "Fuse residual += x * gate and BF16 AdaRMSNorm over the updated residual.");
  m.def("gate_residual_ada_norm_fp8_e4m3fnuz_ptr",
        &gate_residual_ada_norm_fp8_e4m3fnuz_ptr,
        py::arg("residual"), py::arg("x"), py::arg("gate"),
        py::arg("weight"), py::arg("style"),
        py::arg("out"), py::arg("gate_out"), py::arg("scale"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-6,
        py::arg("stream") = 0,
        "Fuse residual += x * gate and AdaRMSNorm directly into static FP8 output.");
  m.def("bias_residual_layer_norm_bf16_ptr",
        &bias_residual_layer_norm_bf16_ptr,
        py::arg("residual"), py::arg("x"), py::arg("bias_pre"),
        py::arg("norm_weight"), py::arg("norm_bias"), py::arg("out"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-5,
        py::arg("stream") = 0,
        "Fuse BF16 residual update with LayerNorm over the updated residual.");
  m.def("layer_norm_fp8_e4m3fnuz_ptr",
        &layer_norm_fp8_e4m3fnuz_ptr,
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
        py::arg("scale"), py::arg("rows"), py::arg("hidden"),
        py::arg("eps") = 1e-5, py::arg("stream") = 0,
        "Write LayerNorm directly into static FP8 using the BF16 boundary contract.");
  m.def("bias_residual_layer_norm_fp8_e4m3fnuz_ptr",
        &bias_residual_layer_norm_fp8_e4m3fnuz_ptr,
        py::arg("residual"), py::arg("x"), py::arg("bias_pre"),
        py::arg("norm_weight"), py::arg("norm_bias"), py::arg("out"),
        py::arg("scale"), py::arg("rows"), py::arg("hidden"),
        py::arg("eps") = 1e-5, py::arg("stream") = 0,
        "Fuse BF16 residual update with LayerNorm directly into static FP8.");
  m.def("qkv_split_bf16_ptr", &qkv_split_bf16_ptr,
        py::arg("qkv"), py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"),
        py::arg("stream") = 0,
        "Split packed BF16 QKV rows into Q, K, and V raw HIP buffers.");
  m.def("gelu_tanh_bf16_ptr", &gelu_tanh_bf16_ptr,
        py::arg("x"), py::arg("out"), py::arg("n"), py::arg("stream") = 0,
        "Write GELU tanh approximation for raw BF16 HIP pointers.");
  m.def("gelu_tanh_quantize_fp8_e4m3fnuz_ptr",
        &gelu_tanh_quantize_fp8_e4m3fnuz_ptr,
        py::arg("x"), py::arg("scale"), py::arg("out"), py::arg("n"),
        py::arg("stream") = 0,
        "Write GELU tanh approximation directly into static FP8 from raw pointers.");
  m.def("gelu_tanh_mul_bf16_ptr", &gelu_tanh_mul_bf16_ptr,
        py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("n"),
        py::arg("stream") = 0,
        "Write gelu_tanh(gate) * up for raw BF16 HIP pointers.");
  m.def("rms_norm_bf16_ptr", &rms_norm_bf16_ptr,
        py::arg("x"), py::arg("weight"), py::arg("out"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-6,
        py::arg("stream") = 0,
        "Write BF16 RMSNorm output from raw HIP pointers.");
  m.def("rms_norm_fp8_e4m3fnuz_ptr", &rms_norm_fp8_e4m3fnuz_ptr,
        py::arg("x"), py::arg("weight"), py::arg("out"), py::arg("scale"),
        py::arg("rows"), py::arg("hidden"), py::arg("eps") = 1e-6,
        py::arg("stream") = 0,
        "Write RMSNorm output directly into static FP8 from raw HIP pointers.");
  m.def("qkv_split_rope_bf16_ptr", &qkv_split_rope_bf16_ptr,
        py::arg("qkv"), py::arg("rope"), py::arg("q"), py::arg("k"),
        py::arg("v"), py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"),
        py::arg("v_dim"), py::arg("head_dim"), py::arg("stream") = 0,
        "Split packed BF16 QKV rows, apply RoPE to Q/K, and write raw buffers.");
  m.def("rms_norm", &rms_norm, py::arg("x"), py::arg("weight"),
        py::arg("eps") = 1e-6,
        "RMSNorm over the last dimension: x * rsqrt(mean(x^2)+eps) * (1+weight).");
}
