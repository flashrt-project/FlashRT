#include "flashrt/cpp/models/pi05/targets/sm110/operation_driver.h"

#include "activation.cuh"
#include "attention_cublas.cuh"
#include "decoder_fused.cuh"
#include "elementwise.cuh"
#include "gemm_runner.h"
#include "flashrt/native_cpp/operations.h"
#include "norm.cuh"
#include "patch_embed.cuh"
#include "quantize.cuh"
#include "rope.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <climits>
#include <cmath>
#include <exception>
#include <new>
#include <stdexcept>

extern "C" int cutlass_fp8_sq(void*, void*, void*, int, int, int, float,
                               float, cudaStream_t);
extern "C" int cutlass_fp8_t1(void*, void*, void*, int, int, int, float,
                               float, cudaStream_t);
extern "C" int cutlass_fp8_wide(void*, void*, void*, int, int, int, float,
                                 float, cudaStream_t);
extern "C" int cutlass_fp8_plain(void*, void*, void*, int, int, int, float,
                                  float, cudaStream_t);
extern "C" int fmha_fp16_strided(
    const void*, const void*, const void*, void*, int, int, int, int, int,
    int, int, int, cudaStream_t);

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm110 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

modalities::Status launch_status() {
    const cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess ? modalities::Status::ok()
                             : backend(cudaGetErrorString(rc));
}

bool valid_gemm(void* a, void* b, void* output, int m, int n, int k) {
    return a && b && output && m > 0 && n > 0 && k > 0;
}

}  // namespace

struct Sm110OperationDriver::Impl {
    Impl() {
        const cublasStatus_t rc = cublasCreate(&cublas);
        if (rc != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("SM110 cuBLAS handle creation failed");
        }
    }
    ~Impl() {
        if (cublas) cublasDestroy(cublas);
    }

    GemmRunner gemm;
    cublasHandle_t cublas = nullptr;
};

Sm110OperationDriver::Sm110OperationDriver() noexcept {
    try {
        impl_.reset(new Impl());
    } catch (const std::exception& e) {
        error_ = e.what();
    } catch (...) {
        error_ = "SM110 kernel driver initialization failed";
    }
}

Sm110OperationDriver::~Sm110OperationDriver() = default;

modalities::Status Sm110OperationDriver::status() const {
    return impl_ ? modalities::Status::ok() : backend(error_);
}

modalities::Status Sm110OperationDriver::fp16_nn(
    void* a, void* b, void* output, int m, int n, int k,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!valid_gemm(a, b, output, m, n, k)) {
        return invalid("SM110 FP16 GEMM arguments are invalid");
    }
    try {
        impl_->gemm.fp16_nn(a, b, output, m, n, k,
                            reinterpret_cast<cudaStream_t>(stream));
        return modalities::Status::ok();
    } catch (const std::exception& e) {
        return backend(e.what());
    } catch (...) {
        return backend("SM110 FP16 GEMM launch failed");
    }
}

modalities::Status Sm110OperationDriver::fp8_nn_bias(
    void* a, void* b, void* output, void* bias, int m, int n, int k,
    float alpha, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!valid_gemm(a, b, output, m, n, k) || !bias ||
        !std::isfinite(alpha)) {
        return invalid("SM110 FP8 bias GEMM arguments are invalid");
    }
    try {
        impl_->gemm.fp8_nn_bias(
            a, b, output, bias, m, n, k, alpha,
            reinterpret_cast<cudaStream_t>(stream));
        return modalities::Status::ok();
    } catch (const std::exception& e) {
        return backend(e.what());
    } catch (...) {
        return backend("SM110 FP8 bias GEMM launch failed");
    }
}

modalities::Status Sm110OperationDriver::fp8_nn_bias_residual(
    void* a, void* b, void* output, void* bias, int m, int n, int k,
    float alpha, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!valid_gemm(a, b, output, m, n, k) || !bias ||
        !std::isfinite(alpha)) {
        return invalid("SM110 FP8 residual GEMM arguments are invalid");
    }
    try {
        impl_->gemm.fp8_nn_bias_res(
            a, b, output, bias, m, n, k, alpha,
            reinterpret_cast<cudaStream_t>(stream));
        return modalities::Status::ok();
    } catch (const std::exception& e) {
        return backend(e.what());
    } catch (...) {
        return backend("SM110 FP8 residual GEMM launch failed");
    }
}

modalities::Status Sm110OperationDriver::fp8_nn_gelu_bias(
    void* a, void* b, void* output, void* bias, int m, int n, int k,
    float alpha, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!valid_gemm(a, b, output, m, n, k) || !bias ||
        !std::isfinite(alpha)) {
        return invalid("SM110 FP8 GELU GEMM arguments are invalid");
    }
    try {
        impl_->gemm.fp8_nn_gelu_bias(
            a, b, output, bias, m, n, k, alpha,
            reinterpret_cast<cudaStream_t>(stream));
        return modalities::Status::ok();
    } catch (const std::exception& e) {
        return backend(e.what());
    } catch (...) {
        return backend("SM110 FP8 GELU GEMM launch failed");
    }
}

modalities::Status Sm110OperationDriver::fp8_cutlass(
    void* a, void* b, void* output, int m, int n, int k,
    float alpha, float beta, Sm110Fp8Tactic tactic,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!valid_gemm(a, b, output, m, n, k) || !std::isfinite(alpha) ||
        !std::isfinite(beta)) {
        return invalid("SM110 CUTLASS FP8 GEMM arguments are invalid");
    }
    using Fn = int (*)(void*, void*, void*, int, int, int, float, float,
                       cudaStream_t);
    Fn fn = nullptr;
    switch (tactic) {
        case Sm110Fp8Tactic::kSquare: fn = cutlass_fp8_sq; break;
        case Sm110Fp8Tactic::kT1: fn = cutlass_fp8_t1; break;
        case Sm110Fp8Tactic::kWide: fn = cutlass_fp8_wide; break;
        case Sm110Fp8Tactic::kPlain: fn = cutlass_fp8_plain; break;
    }
    if (!fn) return invalid("SM110 CUTLASS FP8 tactic is invalid");
    const int rc = fn(a, b, output, m, n, k, alpha, beta,
                      reinterpret_cast<cudaStream_t>(stream));
    return rc == 0 ? launch_status()
                   : backend("SM110 CUTLASS FP8 GEMM rejected the shape");
}

modalities::Status Sm110OperationDriver::fp8_descale(
    void* a, void* b, void* output, int m, int n, int k,
    const float* activation_scale, const float* weight_scale,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!valid_gemm(a, b, output, m, n, k) || !activation_scale ||
        !weight_scale) {
        return invalid("SM110 cuBLASLt FP8 GEMM arguments are invalid");
    }
    try {
        impl_->gemm.fp8_descale_fp16(
            a, b, output, m, n, k, const_cast<float*>(activation_scale),
            const_cast<float*>(weight_scale),
            reinterpret_cast<cudaStream_t>(stream));
        return modalities::Status::ok();
    } catch (const std::exception& e) {
        return backend(e.what());
    } catch (...) {
        return backend("SM110 cuBLASLt FP8 GEMM launch failed");
    }
}

modalities::Status Sm110OperationDriver::add_bias_fp16(
    void* values, const void* bias, int rows, int columns,
    std::uintptr_t stream) const {
    if (!values || !bias || rows <= 0 || columns <= 0) {
        return invalid("SM110 FP16 bias arguments are invalid");
    }
    ::add_bias_fp16(static_cast<__half*>(values),
                    static_cast<const __half*>(bias), rows, columns,
                    reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::layer_norm_fp16(
    const void* values, const void* weight, const void* bias, void* output,
    int rows, int columns, float epsilon, std::uintptr_t stream) const {
    if (!values || !weight || !bias || !output || rows <= 0 || columns <= 0 ||
        !(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        return invalid("SM110 FP16 LayerNorm arguments are invalid");
    }
    ::layer_norm_fp16(
        static_cast<const __half*>(values), static_cast<const __half*>(weight),
        static_cast<const __half*>(bias), static_cast<__half*>(output), rows,
        columns, epsilon, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::layer_norm_fp8(
    const void* values, void* output, const void* weight, const void* bias,
    int rows, int columns, float epsilon, std::uintptr_t stream) const {
    if (!values || !output || !weight || !bias || rows <= 0 || columns <= 0 ||
        !(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        return invalid("SM110 FP8 LayerNorm arguments are invalid");
    }
    ::layer_norm_fp8(
        static_cast<const __half*>(values),
        static_cast<__nv_fp8_e4m3*>(output),
        static_cast<const __half*>(weight), static_cast<const __half*>(bias),
        rows, columns, epsilon, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::rms_norm_fp16(
    const void* values, const void* weight, void* output, int rows,
    int columns, float epsilon, std::uintptr_t stream) const {
    if (!values || !weight || !output || rows <= 0 || columns <= 0 ||
        !(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        return invalid("SM110 FP16 RMSNorm arguments are invalid");
    }
    ::rms_norm_fp16(
        static_cast<const __half*>(values), static_cast<const __half*>(weight),
        static_cast<__half*>(output), rows, columns, epsilon,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::rms_norm_fp8_noweight(
    const void* values, void* output, int rows, int columns,
    const float* scale, std::uintptr_t stream) const {
    if (!values || !output || !scale || rows <= 0 || columns <= 0) {
        return invalid("SM110 FP8 RMSNorm arguments are invalid");
    }
    ::rms_norm_fp8_noweight_fp16(
        static_cast<const __half*>(values),
        static_cast<__nv_fp8_e4m3*>(output), rows, columns, scale,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::residual_rms_norm_fp8_noweight(
    void* residual, const void* values, void* output, int rows, int columns,
    const float* scale, std::uintptr_t stream) const {
    if (!residual || !values || !output || !scale || rows <= 0 ||
        columns <= 0) {
        return invalid("SM110 residual FP8 RMSNorm arguments are invalid");
    }
    ::residual_add_rms_norm_fp8_noweight_fp16(
        static_cast<__half*>(residual), static_cast<const __half*>(values),
        static_cast<__nv_fp8_e4m3*>(output), rows, columns, scale,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::quantize_fp8_static(
    const void* values, void* output, const float* scale, std::size_t elements,
    std::uintptr_t stream) const {
    if (!values || !output || !scale || !elements || elements > INT_MAX) {
        return invalid("SM110 static FP8 quantization arguments are invalid");
    }
    ::quantize_fp8_static_fp16(
        static_cast<const __half*>(values),
        static_cast<__nv_fp8_e4m3*>(output), scale,
        static_cast<int>(elements), reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::quantize_fp8_dynamic(
    const void* values, void* output, float* scale, std::size_t elements,
    std::uintptr_t stream) const {
    if (!values || !output || !scale || !elements || elements > INT_MAX) {
        return invalid("SM110 dynamic FP8 quantization arguments are invalid");
    }
    ::quantize_fp8_device_fp16(
        static_cast<const __half*>(values),
        static_cast<__nv_fp8_e4m3*>(output), scale,
        static_cast<int>(elements), reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::gelu_fp16(
    void* values, std::size_t elements, std::uintptr_t stream) const {
    if (!values || !elements || elements > INT_MAX) {
        return invalid("SM110 FP16 GELU arguments are invalid");
    }
    ::gelu_inplace_fp16(static_cast<__half*>(values),
                        static_cast<int>(elements),
                        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::silu_fp16(
    void* values, std::size_t elements, std::uintptr_t stream) const {
    if (!values || !elements) {
        return invalid("SM110 FP16 SiLU arguments are invalid");
    }
    if (elements > INT_MAX || elements % 2) {
        return invalid("SM110 FP16 SiLU element count is invalid");
    }
    ::silu_inplace_fp16(static_cast<__half*>(values),
                        static_cast<int>(elements),
                        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::gate_gelu_fp16(
    const void* merged, void* output, int rows, int hidden,
    std::uintptr_t stream) const {
    if (!merged || !output || rows <= 0 || hidden <= 0) {
        return invalid("SM110 FP16 gated GELU arguments are invalid");
    }
    ::gate_geglu_merged_fp16(
        static_cast<const __half*>(merged), static_cast<__half*>(output),
        rows, hidden, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::gate_gelu_fp8(
    const void* merged, void* output, int rows, int hidden,
    const float* scale, std::uintptr_t stream) const {
    if (!merged || !output || !scale || rows <= 0 || hidden <= 0) {
        return invalid("SM110 FP8 gated GELU arguments are invalid");
    }
    ::gate_geglu_merged_fp8_fp16(
        static_cast<const __half*>(merged),
        static_cast<__nv_fp8_e4m3*>(output), rows, hidden, scale,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::residual_add_fp16(
    void* residual, const void* values, std::size_t elements,
    std::uintptr_t stream) const {
    if (!residual || !values || !elements || elements > INT_MAX) {
        return invalid("SM110 FP16 residual arguments are invalid");
    }
    ::residual_add_fp16(
        static_cast<__half*>(residual), static_cast<const __half*>(values),
        static_cast<int>(elements), reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::fused_adarms_fp8(
    const void* values, const void* style, void* output, void* gate,
    int rows, int columns, const float* scale,
    std::uintptr_t stream) const {
    if (!values || !style || !output || !gate || !scale || rows <= 0 ||
        columns <= 0) {
        return invalid("SM110 fused AdaRMSNorm arguments are invalid");
    }
    ::fused_adarms_fp8_static_fp16(
        static_cast<const __half*>(values), static_cast<const __half*>(style),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<__half*>(gate), rows,
        columns, scale, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::gate_res_adarms_fp8(
    const void* gemm_output, const void* previous_gate, void* residual,
    const void* style, void* output, void* gate, int rows, int columns,
    const float* scale, std::uintptr_t stream) const {
    if (!gemm_output || !previous_gate || !residual || !style || !output ||
        !gate || !scale || rows <= 0 || columns <= 0) {
        return invalid("SM110 gated AdaRMSNorm arguments are invalid");
    }
    ::gate_res_adarms_fp8_static_fp16(
        static_cast<const __half*>(gemm_output),
        static_cast<const __half*>(previous_gate),
        static_cast<__half*>(residual), static_cast<const __half*>(style),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<__half*>(gate), rows,
        columns, scale, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::gate_res_fp16(
    const void* gemm_output, const void* gate, void* residual,
    std::size_t elements, std::uintptr_t stream) const {
    if (!gemm_output || !gate || !residual || !elements || elements > INT_MAX) {
        return invalid("SM110 gated residual arguments are invalid");
    }
    ::gate_res_fp16(
        static_cast<const __half*>(gemm_output),
        static_cast<const __half*>(gate), static_cast<__half*>(residual),
        static_cast<int>(elements), reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::adarms_fp16(
    const void* values, const void* style, void* output, void* gate,
    int rows, int columns, std::uintptr_t stream) const {
    if (!values || !style || !output || !gate || rows <= 0 || columns <= 0) {
        return invalid("SM110 FP16 AdaRMSNorm arguments are invalid");
    }
    ::adarms_fp16(
        static_cast<const __half*>(values), static_cast<const __half*>(style),
        static_cast<__half*>(output), static_cast<__half*>(gate), rows,
        columns, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::qkv_rope_cache_fp16(
    const void* qkv, const void* rope, void* query, void* key_cache,
    void* value_cache, int rows, int query_columns, int key_columns,
    int head_dimension, int qkv_stride, int cache_offset, int cache_stride,
    std::uintptr_t stream) const {
    if (!qkv || !rope || !query || !key_cache || !value_cache || rows <= 0 ||
        query_columns <= 0 || key_columns <= 0 || head_dimension <= 0 ||
        qkv_stride <= 0 || cache_offset < 0 || cache_stride <= 0) {
        return invalid("SM110 FP16 QKV cache arguments are invalid");
    }
    ::qkv_split_rope_kvcache_fp16(
        static_cast<const __half*>(qkv), static_cast<const __half*>(rope),
        static_cast<__half*>(query), static_cast<__half*>(key_cache),
        static_cast<__half*>(value_cache), rows, query_columns, key_columns,
        head_dimension, qkv_stride, cache_offset, cache_stride,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::qkv_rope_cache_devpos_fp16(
    const void* qkv, const void* rope, void* query, void* key_cache,
    void* value_cache, const int* device_position, int rows,
    int query_columns, int key_columns, int head_dimension, int qkv_stride,
    int cache_offset, int cache_stride, std::uintptr_t stream) const {
    if (!qkv || !rope || !query || !key_cache || !value_cache ||
        !device_position || rows <= 0 || query_columns <= 0 ||
        key_columns <= 0 || head_dimension <= 0 || qkv_stride <= 0 ||
        cache_offset < 0 || cache_stride <= 0) {
        return invalid("SM110 FP16 device-position QKV arguments are invalid");
    }
    ::qkv_split_rope_kvcache_fp16_devpos(
        static_cast<const __half*>(qkv), static_cast<const __half*>(rope),
        static_cast<__half*>(query), static_cast<__half*>(key_cache),
        static_cast<__half*>(value_cache), device_position, rows,
        query_columns, key_columns, head_dimension, qkv_stride, cache_offset,
        cache_stride, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::attention_seqused_fp16(
    const void* query, const void* key, const void* value, void* logits,
    void* output, int query_rows, int max_key_rows, int heads,
    int head_dimension, const int* valid_key_rows, float scale,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!query || !key || !value || !logits || !output || !valid_key_rows ||
        query_rows <= 0 || max_key_rows <= 0 || heads <= 0 ||
        head_dimension <= 0 || !(scale > 0.0f) || !std::isfinite(scale)) {
        return invalid("SM110 FP16 attention arguments are invalid");
    }
    ::flashrt_native_attention_qkv_fp16_seqused(
        impl_->cublas, static_cast<const __half*>(query),
        static_cast<const __half*>(key), static_cast<const __half*>(value),
        static_cast<__half*>(logits), static_cast<__half*>(output), query_rows,
        max_key_rows, heads, head_dimension, valid_key_rows, scale,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::vision_fmha_fp16(
    const void* query, const void* key, const void* value, void* output,
    int batch, int query_rows, int key_rows, int query_heads, int key_heads,
    int head_dimension, int query_stride, int key_stride,
    std::uintptr_t stream) const {
    if (!query || !key || !value || !output || batch <= 0 || query_rows <= 0 ||
        key_rows <= 0 || query_heads <= 0 || key_heads <= 0 ||
        head_dimension <= 0 || query_stride <= 0 || key_stride <= 0) {
        return invalid("SM110 vision FMHA arguments are invalid");
    }
    const int rc = ::fmha_fp16_strided(
        query, key, value, output, batch, query_rows, key_rows, query_heads,
        key_heads, head_dimension, query_stride, key_stride,
        reinterpret_cast<cudaStream_t>(stream));
    return rc == 0 ? launch_status()
                   : backend("SM110 vision FMHA rejected the shape");
}

modalities::Status Sm110OperationDriver::patch_im2col_fp16(
    const void* images, void* patches, int num_views,
    std::uintptr_t stream) const {
    if (!images || !patches || num_views <= 0 || num_views > 3) {
        return invalid("SM110 patch im2col arguments are invalid");
    }
    ::patch_im2col(static_cast<const __half*>(images),
                   static_cast<__half*>(patches), num_views,
                   reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::patch_bias_position_fp16(
    void* output, const void* bias, const void* position, int sequence,
    int columns, int positions_per_view, std::uintptr_t stream) const {
    if (!output || !bias || !position || sequence <= 0 || columns <= 0 ||
        positions_per_view <= 0) {
        return invalid("SM110 patch position arguments are invalid");
    }
    ::patch_embed_bias_pos(
        static_cast<__half*>(output), static_cast<const __half*>(bias),
        static_cast<const __half*>(position), sequence, columns,
        positions_per_view, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::bias_residual_fp16(
    void* residual, const void* values, const void* bias, int rows,
    int columns, std::uintptr_t stream) const {
    if (!residual || !values || !bias || rows <= 0 || columns <= 0) {
        return invalid("SM110 FP16 bias residual arguments are invalid");
    }
    ::bias_residual_strict_fp16(
        static_cast<__half*>(residual), static_cast<const __half*>(values),
        static_cast<const __half*>(bias), rows, columns,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::gmm_fp16(
    const void* a, const void* b, void* output, int m, int n, int k,
    float beta, std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!a || !b || !output || m <= 0 || n <= 0 || k <= 0 ||
        !std::isfinite(beta)) {
        return invalid("SM110 decoder FP16 GEMM arguments are invalid");
    }
    ::gmm_fp16(
        impl_->cublas, static_cast<const __half*>(a),
        static_cast<const __half*>(b), static_cast<__half*>(output),
        m, n, k, beta, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::gmm_fp16_out_fp32(
    const void* a, const void* b, float* output, int m, int n, int k,
    std::uintptr_t stream) const {
    if (!impl_) return backend(error_);
    if (!a || !b || !output || m <= 0 || n <= 0 || k <= 0) {
        return invalid("SM110 FP32-output GEMM arguments are invalid");
    }
    ::gmm_fp16_out_fp32(
        impl_->cublas, static_cast<const __half*>(a),
        static_cast<const __half*>(b), output, m, n, k,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status Sm110OperationDriver::action_update_fp16(
    const float* delta, const void* bias, void* noise, int rows, int columns,
    float dt, std::uintptr_t stream) const {
    if (!delta || !bias || !noise || rows <= 0 || columns <= 0 ||
        !std::isfinite(dt)) {
        return invalid("SM110 action update arguments are invalid");
    }
    ::action_update_from_fp32(
        delta, static_cast<const __half*>(bias), static_cast<__half*>(noise),
        rows, columns, dt, true, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt
