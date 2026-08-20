#include "flashrt/cpp/models/pi05/targets/sm120/target.h"

#include "activation.cuh"
#include "elementwise.cuh"
#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/model/frontend_ops.h"
#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"
#include "flashrt/cpp/models/pi05/support/native_weight_materializer.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention_driver.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_scratch.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_weight_packer.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_autotune.h"
#include "flashrt/native_cpp/operations.h"
#include "fusion.cuh"
#include "norm.cuh"
#include "patch_embed.cuh"
#include "quantize.cuh"
#include "rope.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <filesystem>
#include <limits>
#include <new>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status unsupported(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kUnsupported,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

modalities::Status launch_status() {
    const cudaError_t result = cudaGetLastError();
    return result == cudaSuccess ? modalities::Status::ok()
                                 : backend(cudaGetErrorString(result));
}

void set_status(modalities::Status* destination,
                modalities::Status status) {
    if (destination) *destination = std::move(status);
}

modalities::Status validate_device() {
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t result = cudaGetDevice(&device);
    if (result == cudaSuccess) {
        result = cudaGetDeviceProperties(&properties, device);
    }
    if (result != cudaSuccess) return backend(cudaGetErrorString(result));
    return properties.major == 12 && properties.minor == 0
               ? modalities::Status::ok()
               : unsupported("SM120 target requires compute capability 12.0");
}

bool vector_weight(const Pi05ResolvedWeight& weight, int width) {
    return width > 0 && weight.device_data &&
           weight.storage == Pi05WeightStorage::kBFloat16 &&
           weight.shape.rank == 1 &&
           weight.shape.dims[0] == static_cast<std::uint64_t>(width) &&
           weight.bytes ==
               static_cast<std::uint64_t>(width) * sizeof(std::uint16_t);
}

bool matrix_weight(const Pi05ResolvedWeight& weight,
                   int rows,
                   int columns) {
    return rows > 0 && columns > 0 && weight.device_data &&
           weight.storage == Pi05WeightStorage::kBFloat16 &&
           weight.shape.rank == 2 &&
           weight.shape.dims[0] == static_cast<std::uint64_t>(rows) &&
           weight.shape.dims[1] == static_cast<std::uint64_t>(columns) &&
           static_cast<std::uint64_t>(rows) <=
               std::numeric_limits<std::uint64_t>::max() /
                   static_cast<std::uint64_t>(columns) &&
           weight.bytes == static_cast<std::uint64_t>(rows) *
                               static_cast<std::uint64_t>(columns) *
                               sizeof(std::uint16_t);
}

modalities::Status derive_action_step_weights(
    frt_ctx context,
    const Pi05ResolvedShape& shape,
    Pi05ResolvedWeights* weights,
    frt_buffer* weight_backing,
    frt_buffer* bias_backing) {
    if (!context || !weights || !weight_backing || !bias_backing ||
        shape.num_steps <= 0 ||
        !matrix_weight(weights->decoder.action_out_weight,
                       kPi05ModelDims.decoder_width,
                       kPi05ModelDims.action_width) ||
        !vector_weight(weights->decoder.action_out_bias,
                       kPi05ModelDims.action_width)) {
        return invalid("SM120 action weight derivation is invalid");
    }
    const std::uint64_t weight_bytes =
        weights->decoder.action_out_weight.bytes;
    const std::uint64_t bias_bytes = weights->decoder.action_out_bias.bytes;
    frt_buffer derived_weight = frt_buffer_alloc(
        context, "pi05_sm120_action_out_step_weight", weight_bytes);
    frt_buffer derived_bias = frt_buffer_alloc(
        context, "pi05_sm120_action_out_step_bias", bias_bytes);
    if (!derived_weight || !derived_bias ||
        !frt_buffer_dptr(derived_weight) || !frt_buffer_dptr(derived_bias)) {
        return backend("SM120 action weight derivation allocation failed");
    }
    const float scale = -1.0f / static_cast<float>(shape.num_steps);
    flashrt_native_scale_bf16_weight(
        static_cast<const __nv_bfloat16*>(
            weights->decoder.action_out_weight.device_data),
        static_cast<__nv_bfloat16*>(frt_buffer_dptr(derived_weight)), scale,
        kPi05ModelDims.decoder_width * kPi05ModelDims.action_width);
    flashrt_native_scale_bf16_weight(
        static_cast<const __nv_bfloat16*>(
            weights->decoder.action_out_bias.device_data),
        static_cast<__nv_bfloat16*>(frt_buffer_dptr(derived_bias)), scale,
        kPi05ModelDims.action_width);
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) return backend(cudaGetErrorString(launched));
    const cudaError_t synchronized = cudaStreamSynchronize(nullptr);
    if (synchronized != cudaSuccess) {
        return backend(cudaGetErrorString(synchronized));
    }

    Pi05ResolvedWeight private_weight = weights->decoder.action_out_weight;
    Pi05ResolvedWeight private_bias = weights->decoder.action_out_bias;
    private_weight.device_data = frt_buffer_dptr(derived_weight);
    private_bias.device_data = frt_buffer_dptr(derived_bias);
    weights->decoder.action_out_weight = private_weight;
    weights->decoder.action_out_bias = private_bias;
    *weight_backing = derived_weight;
    *bias_backing = derived_bias;
    return modalities::Status::ok();
}

struct Sm120FrontendBindings final {
    const Pi05ResolvedShape* shape = nullptr;
    Sm120Bf16Linear* bf16_linear = nullptr;
    Sm120Fp8Linear* fp8_linear = nullptr;
    Sm120AttentionBacking* attention_buffers = nullptr;
    Sm120AttentionDriver* attention = nullptr;
};

Sm120FrontendBindings* frontend(void* state) {
    return static_cast<Sm120FrontendBindings*>(state);
}

void* frontend_data(const Pi05ResolvedBuffer& buffer) {
    return buffer.buffer ? frt_buffer_dptr(buffer.buffer) : nullptr;
}

bool frontend_static_fp8(const Sm120FrontendBindings* binding) {
    return binding && binding->fp8_linear &&
           !binding->fp8_linear->observing();
}

float* frontend_scale(Sm120FrontendBindings* binding,
                      Pi05LinearWeightKey key,
                      int step) {
    Pi05LinearActivationSite site;
    return binding && binding->fp8_linear && binding->shape &&
                   resolve_pi05_linear_activation_site(
                       key, step, *binding->shape, &site).ok_status()
               ? binding->fp8_linear->scale_data(site)
               : nullptr;
}

modalities::Status frontend_linear(
    void* state,
    const Pi05ResolvedWeight& weight,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !binding->bf16_linear) {
        return invalid("SM120 BF16 linear binding is invalid");
    }
    return binding->bf16_linear->run(
        weight, input, output, rows, input_width, output_width, stream);
}

modalities::Status apply_projected_epilogue(
    const Pi05LinearEpilogue& epilogue,
    void* output,
    int rows,
    int columns,
    Pi05Stream stream) {
    const bool has_bias = epilogue.bias != nullptr;
    const bool has_residual = epilogue.residual != nullptr;
    if (!output || rows <= 0 || columns <= 0 ||
        (has_bias && !vector_weight(*epilogue.bias, columns))) {
        return invalid("SM120 projected-linear epilogue is invalid");
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (epilogue.kind) {
        case Pi05LinearEpilogueKind::kNone:
            return !has_bias && !has_residual
                       ? modalities::Status::ok()
                       : invalid("SM120 empty epilogue has operands");
        case Pi05LinearEpilogueKind::kBias:
            if (!has_bias || has_residual) break;
            add_bias_bf16(
                static_cast<__nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(
                    epilogue.bias->device_data),
                rows, columns, cuda_stream);
            return launch_status();
        case Pi05LinearEpilogueKind::kBiasGelu:
            if (!has_bias || has_residual) break;
            add_bias_bf16(
                static_cast<__nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(
                    epilogue.bias->device_data),
                rows, columns, cuda_stream);
            {
                modalities::Status status = launch_status();
                if (!status.ok_status()) return status;
            }
            ::gelu_inplace(static_cast<__nv_bfloat16*>(output),
                           rows * columns, cuda_stream);
            return launch_status();
        case Pi05LinearEpilogueKind::kBiasResidual:
            if (!has_bias || !has_residual) break;
            ::bias_residual(
                static_cast<__nv_bfloat16*>(epilogue.residual),
                static_cast<const __nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(
                    epilogue.bias->device_data),
                rows, columns, cuda_stream);
            return launch_status();
    }
    return invalid("SM120 projected-linear epilogue operands are invalid");
}

modalities::Status frontend_projected_linear(
    void* state,
    const Pi05ResolvedWeight& weight,
    Pi05LinearWeightKey key,
    int step,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    bool prequantized,
    const Pi05LinearEpilogue& epilogue,
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !binding->shape || !binding->bf16_linear) {
        return invalid("SM120 projected-linear binding is invalid");
    }
    modalities::Status status;
    if (!binding->fp8_linear) {
        status = prequantized
                     ? invalid("SM120 BF16 input cannot be prequantized")
                     : binding->bf16_linear->run(
                           weight, input, output, rows, input_width,
                           output_width, stream);
    } else {
        Pi05LinearActivationSite site;
        status = resolve_pi05_linear_activation_site(
            key, step, *binding->shape, &site);
        if (status.ok_status()) {
            status = prequantized
                         ? binding->fp8_linear->run_prequantized(
                               weight, site, input, output, rows,
                               input_width, output_width, stream)
                         : binding->fp8_linear->run(
                               weight, site, input, output, rows,
                               input_width, output_width, stream);
        }
    }
    return status.ok_status()
               ? apply_projected_epilogue(
                     epilogue, output, rows, output_width, stream)
               : status;
}

modalities::Status frontend_add_bias(void*,
                                     void* values,
                                     const Pi05ResolvedWeight& bias,
                                     int rows,
                                     int columns,
                                     Pi05Stream stream) {
    if (!values || rows <= 0 || !vector_weight(bias, columns)) {
        return invalid("SM120 BF16 bias arguments are invalid");
    }
    add_bias_bf16(
        static_cast<__nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(bias.device_data), rows, columns,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_silu(void*,
                                void* values,
                                std::size_t elements,
                                Pi05Stream stream) {
    if (!values || elements <= 0) {
        return invalid("SM120 BF16 SiLU arguments are invalid");
    }
    flashrt_native_silu_inplace_bf16(
        static_cast<__nv_bfloat16*>(values), static_cast<int>(elements),
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_copy(void*,
                                 void* destination,
                                 const void* source,
                                 std::size_t bytes,
                                 Pi05Stream stream) {
    const cudaError_t result = cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(stream));
    return result == cudaSuccess
               ? modalities::Status::ok()
               : backend(cudaGetErrorString(result));
}

modalities::Status frontend_patchify(void*,
                                     const void* images,
                                     void* patches,
                                     int views,
                                     Pi05Stream stream) {
    ::patch_im2col(
        static_cast<const __half*>(images), static_cast<__half*>(patches),
        views, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_bias_residual(
    void*,
    void* residual,
    const void* values,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    Pi05Stream stream) {
    if (!residual || !values || !vector_weight(bias, columns)) {
        return invalid("SM120 bias-residual binding is invalid");
    }
    ::bias_residual(
        static_cast<__nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(bias.device_data), rows, columns,
        reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_layer_norm(
    void*,
    const void* values,
    const Pi05ResolvedWeight& weight,
    const Pi05ResolvedWeight& bias,
    void* output,
    int rows,
    int columns,
    float epsilon,
    bool,
    Pi05LinearInput* linear_input,
    Pi05Stream stream) {
    if (!values || !output || !vector_weight(weight, columns) ||
        !vector_weight(bias, columns) || !linear_input) {
        return invalid("SM120 layer-norm binding is invalid");
    }
    ::layer_norm(
        static_cast<const __nv_bfloat16*>(values),
        static_cast<const __nv_bfloat16*>(weight.device_data),
        static_cast<const __nv_bfloat16*>(bias.device_data),
        static_cast<__nv_bfloat16*>(output), rows, columns, epsilon,
        reinterpret_cast<cudaStream_t>(stream));
    *linear_input = {output, false};
    return launch_status();
}

modalities::Status frontend_qkv_split(
    void*,
    const void* qkv,
    void* query,
    void* key,
    void* value,
    int rows,
    int query_width,
    int key_width,
    int value_width,
    Pi05Stream stream) {
    ::qkv_split(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<__nv_bfloat16*>(query),
        static_cast<__nv_bfloat16*>(key),
        static_cast<__nv_bfloat16*>(value), rows, query_width, key_width,
        value_width, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_normalize_for_linear(
    void* state,
    void* residual,
    const void* update,
    const Pi05ResolvedBuffer& weight,
    void* normalized,
    Pi05LinearWeightKey key,
    int step,
    int rows,
    int columns,
    float epsilon,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !residual || !frontend_data(weight) || !normalized ||
        !linear_input || !prequantized || rows <= 0 || columns <= 0) {
        return invalid("SM120 normalization binding is invalid");
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (frontend_static_fp8(binding)) {
        float* scale = frontend_scale(binding, key, step);
        if (!scale) return invalid("SM120 normalization scale is invalid");
        void* output = binding->fp8_linear->scratch_data();
        if (update) {
            ::residual_add_rms_norm_fp8(
                static_cast<__nv_bfloat16*>(residual),
                static_cast<const __nv_bfloat16*>(update),
                static_cast<const __nv_bfloat16*>(frontend_data(weight)),
                static_cast<__nv_fp8_e4m3*>(output), rows, columns, epsilon,
                scale, cuda_stream);
        } else {
            ::rms_norm_fp8(
                static_cast<const __nv_bfloat16*>(residual),
                static_cast<const __nv_bfloat16*>(frontend_data(weight)),
                static_cast<__nv_fp8_e4m3*>(output), rows, columns, epsilon,
                scale, cuda_stream);
        }
        *linear_input = output;
        *prequantized = true;
        return launch_status();
    }
    if (update) {
        ::residual_add(static_cast<__nv_bfloat16*>(residual),
                       static_cast<const __nv_bfloat16*>(update),
                       rows * columns, cuda_stream);
        modalities::Status status = launch_status();
        if (!status.ok_status()) return status;
    }
    ::rms_norm(
        static_cast<const __nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(frontend_data(weight)),
        static_cast<__nv_bfloat16*>(normalized), rows, columns, epsilon,
        cuda_stream);
    *linear_input = normalized;
    *prequantized = false;
    return launch_status();
}

modalities::Status frontend_qkv_rope(
    void*,
    const void* qkv,
    const Pi05ResolvedBuffer& rope,
    const Pi05ResolvedBuffer* position,
    void* query,
    void* key,
    void* value,
    int rows,
    int query_width,
    int key_width,
    int value_width,
    int head_width,
    Pi05Stream stream) {
    if (!qkv || !frontend_data(rope) || !query || !key || !value) {
        return invalid("SM120 QKV/RoPE binding is invalid");
    }
    if (position) {
        if (!frontend_data(*position)) {
            return invalid("SM120 QKV/RoPE position is invalid");
        }
        ::qkv_split_rope_devpos(
            static_cast<const __nv_bfloat16*>(qkv),
            static_cast<const __nv_bfloat16*>(frontend_data(rope)),
            static_cast<__nv_bfloat16*>(query),
            static_cast<__nv_bfloat16*>(key),
            static_cast<__nv_bfloat16*>(value),
            static_cast<const int*>(frontend_data(*position)), rows,
            query_width, key_width, value_width, head_width,
            reinterpret_cast<cudaStream_t>(stream));
    } else {
        ::qkv_split_rope(
            static_cast<const __nv_bfloat16*>(qkv),
            static_cast<const __nv_bfloat16*>(frontend_data(rope)),
            static_cast<__nv_bfloat16*>(query),
            static_cast<__nv_bfloat16*>(key),
            static_cast<__nv_bfloat16*>(value), rows, query_width, key_width,
            value_width, head_width,
            reinterpret_cast<cudaStream_t>(stream));
    }
    return launch_status();
}

modalities::Status frontend_residual_accumulate(
    void*,
    void* residual,
    const void* update,
    const void* gate,
    std::size_t elements,
    Pi05Stream stream) {
    if (!residual || !update || !elements ||
        elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return invalid("SM120 residual update is invalid");
    }
    if (gate) {
        ::gate_mul_residual(
            static_cast<__nv_bfloat16*>(residual),
            static_cast<const __nv_bfloat16*>(update),
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<int>(elements),
            reinterpret_cast<cudaStream_t>(stream));
    } else {
        ::residual_add(
            static_cast<__nv_bfloat16*>(residual),
            static_cast<const __nv_bfloat16*>(update),
            static_cast<int>(elements),
            reinterpret_cast<cudaStream_t>(stream));
    }
    return launch_status();
}

modalities::Status frontend_diffusion_update(
    void* state,
    void* residual,
    const void* update,
    const Pi05ResolvedWeight& bias,
    int rows,
    int columns,
    int num_steps,
    Pi05Stream stream) {
    if (!update || num_steps <= 0 || !vector_weight(bias, columns)) {
        return invalid("SM120 diffusion update is invalid");
    }
    add_bias_bf16(
        const_cast<__nv_bfloat16*>(
            static_cast<const __nv_bfloat16*>(update)),
        static_cast<const __nv_bfloat16*>(bias.device_data), rows, columns,
        reinterpret_cast<cudaStream_t>(stream));
    modalities::Status status = launch_status();
    return status.ok_status()
               ? frontend_residual_accumulate(
                     state, residual, update, nullptr,
                     static_cast<std::size_t>(rows) * columns, stream)
               : status;
}

modalities::Status frontend_adaptive_normalize(
    void* state,
    void* residual,
    const void* update,
    const void* update_gate,
    const Pi05ResolvedBuffer& weight,
    const void* style,
    void* normalized,
    void* gate,
    Pi05LinearWeightKey key,
    int step,
    int rows,
    int columns,
    float epsilon,
    bool quantize,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !residual || !frontend_data(weight) || !style ||
        !normalized || !gate || !linear_input || !prequantized || rows <= 0 ||
        columns <= 0 || ((update == nullptr) != (update_gate == nullptr))) {
        return invalid("SM120 adaptive normalization is invalid");
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const bool static_quantized = quantize && frontend_static_fp8(binding);
    if (static_quantized) {
        float* scale = frontend_scale(binding, key, step);
        if (!scale) {
            return invalid("SM120 adaptive normalization scale is invalid");
        }
        void* output = binding->fp8_linear->scratch_data();
        if (update) {
            ::gate_residual_ada_norm_fp8(
                static_cast<__nv_bfloat16*>(residual),
                static_cast<const __nv_bfloat16*>(update),
                static_cast<const __nv_bfloat16*>(update_gate),
                static_cast<const __nv_bfloat16*>(frontend_data(weight)),
                static_cast<const __nv_bfloat16*>(style),
                static_cast<__nv_fp8_e4m3*>(output),
                static_cast<__nv_bfloat16*>(gate), rows, columns, epsilon,
                scale, cuda_stream);
        } else {
            ::ada_rms_norm_style_fp8(
                static_cast<const __nv_bfloat16*>(residual),
                static_cast<const __nv_bfloat16*>(frontend_data(weight)),
                static_cast<const __nv_bfloat16*>(style),
                static_cast<__nv_fp8_e4m3*>(output),
                static_cast<__nv_bfloat16*>(gate), rows, columns, epsilon,
                scale, cuda_stream);
        }
        *linear_input = output;
        *prequantized = true;
        return launch_status();
    }
    if (update) {
        modalities::Status status = frontend_residual_accumulate(
            state, residual, update, update_gate,
            static_cast<std::size_t>(rows) * columns, stream);
        if (!status.ok_status()) return status;
    }
    ::ada_rms_norm_style(
        static_cast<const __nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(frontend_data(weight)),
        static_cast<const __nv_bfloat16*>(style),
        static_cast<__nv_bfloat16*>(normalized),
        static_cast<__nv_bfloat16*>(gate), rows, columns, epsilon,
        cuda_stream);
    *linear_input = normalized;
    *prequantized = false;
    return launch_status();
}

modalities::Status frontend_attention(
    void* state,
    Pi05LinearDomain domain,
    int layer,
    const void* query,
    const void* key,
    const void* value,
    void* output,
    int batches,
    int query_rows,
    int key_rows,
    int query_heads,
    int key_heads,
    int head_width,
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !binding->shape || !binding->attention_buffers ||
        !binding->attention) {
        return invalid("SM120 attention binding is invalid");
    }
    if (domain == Pi05LinearDomain::kVision) {
        const Sm120VisionAttentionBuffers& buffers =
            binding->attention_buffers->vision();
        if (layer != -1 || query != buffers.query.device_data() ||
            key != buffers.key.device_data() ||
            value != buffers.value.device_data() ||
            output != binding->attention->vision_output() ||
            batches != binding->shape->num_views ||
            query_rows * batches != binding->shape->vision_sequence ||
            key_rows != query_rows ||
            query_heads != kPi05ModelDims.vision_heads ||
            key_heads != query_heads ||
            head_width != kPi05ModelDims.vision_head_dim) {
            return invalid("SM120 vision-attention contract is invalid");
        }
        return binding->attention->vision(stream);
    }
    if (domain == Pi05LinearDomain::kEncoder) {
        const Sm120EncoderAttentionBuffers& buffers =
            binding->attention_buffers->encoder();
        if (layer < 0 || layer >= kPi05ModelDims.encoder_layers - 1 ||
            query != buffers.query.device_data() ||
            key != binding->attention_buffers->key_layer_data(layer) ||
            value != binding->attention_buffers->value_layer_data(layer) ||
            output != binding->attention->encoder_output() || batches != 1 ||
            query_rows != binding->shape->encoder_sequence ||
            key_rows != query_rows ||
            query_heads != kPi05ModelDims.encoder_heads ||
            key_heads != kPi05ModelDims.encoder_kv_heads ||
            head_width != kPi05ModelDims.encoder_head_dim) {
            return invalid("SM120 encoder-attention contract is invalid");
        }
        return binding->attention->encoder(layer, stream);
    }
    if (domain == Pi05LinearDomain::kDecoder) {
        const Sm120DecoderAttentionBuffers& buffers =
            binding->attention_buffers->decoder();
        if (layer < 0 || layer >= kPi05ModelDims.decoder_layers ||
            query != buffers.query.device_data() ||
            key != binding->attention_buffers->key_layer_data(layer) ||
            value != binding->attention_buffers->value_layer_data(layer) ||
            output != binding->attention->decoder_output() || batches != 1 ||
            query_rows != binding->shape->chunk ||
            key_rows != binding->shape->total_attention_keys ||
            query_heads != kPi05ModelDims.decoder_heads ||
            key_heads != kPi05ModelDims.decoder_kv_heads ||
            head_width != kPi05ModelDims.decoder_head_dim) {
            return invalid("SM120 decoder-attention contract is invalid");
        }
        return binding->attention->decoder(layer, stream);
    }
    return unsupported("SM120 attention domain is not installed");
}

modalities::Status frontend_gate_up(
    void* state,
    const Pi05FeedForwardWeights& weights,
    Pi05LinearWeightKey key,
    int step,
    const void* input,
    bool prequantized,
    void* gate,
    void* up,
    int rows,
    int width,
    int hidden_width,
    bool* merged,
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !binding->bf16_linear || !input || !gate || !up ||
        !merged) {
        return invalid("SM120 gate-up binding is invalid");
    }
    const bool static_fp8 = frontend_static_fp8(binding);
    if (prequantized != static_fp8) {
        return invalid("SM120 gate-up input storage is invalid");
    }
    if (static_fp8 || binding->fp8_linear) {
        *merged = true;
        return frontend_projected_linear(
            state, weights.gate_up_weight, key, step, input, gate, rows,
            width, 2 * hidden_width, prequantized, {}, stream);
    }
    *merged = false;
    modalities::Status status = binding->bf16_linear->run(
        weights.gate_weight, input, gate, rows, width, hidden_width, stream);
    return status.ok_status()
               ? binding->bf16_linear->run(
                     weights.up_weight, input, up, rows, width, hidden_width,
                     stream)
               : status;
}

modalities::Status frontend_gated_activation(
    void* state,
    const void* gate,
    const void* up,
    bool merged,
    void* output,
    int rows,
    int hidden_width,
    Pi05LinearWeightKey output_key,
    int step,
    const void** linear_input,
    bool* prequantized,
    Pi05Stream stream) {
    Sm120FrontendBindings* binding = frontend(state);
    if (!binding || !gate || !up || !output || !linear_input ||
        !prequantized) {
        return invalid("SM120 gated-activation binding is invalid");
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (frontend_static_fp8(binding)) {
        if (!merged) {
            return invalid("SM120 static FP8 gate-up layout is invalid");
        }
        float* scale = frontend_scale(binding, output_key, step);
        if (!scale) return invalid("SM120 gated-activation scale is invalid");
        *linear_input = binding->fp8_linear->scratch_data();
        *prequantized = true;
        ::gate_geglu_merged_fp8(
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<__nv_fp8_e4m3*>(
                binding->fp8_linear->scratch_data()),
            rows, hidden_width, scale, cuda_stream);
        return launch_status();
    }
    *linear_input = output;
    *prequantized = false;
    if (merged) {
        ::gate_geglu_merged(
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<__nv_bfloat16*>(output), rows, hidden_width,
            cuda_stream);
    } else {
        ::gate_geglu(
            static_cast<const __nv_bfloat16*>(gate),
            static_cast<const __nv_bfloat16*>(up),
            static_cast<__nv_bfloat16*>(output), rows * hidden_width,
            cuda_stream);
    }
    const modalities::Status status = launch_status();
    return status;
}

modalities::Status frontend_gelu(void*,
                                 void* values,
                                 std::size_t elements,
                                 Pi05Stream stream) {
    if (!values || !elements ||
        elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return invalid("SM120 GELU binding is invalid");
    }
    ::gelu_inplace(static_cast<__nv_bfloat16*>(values),
                   static_cast<int>(elements),
                   reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

modalities::Status frontend_vision_pool(void*,
                                        const void* input,
                                        void* output,
                                        int views,
                                        int grid_height,
                                        int grid_width,
                                        int columns,
                                        int factor,
                                        Pi05Stream stream) {
    ::avg_pool_vision_tokens(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output), views, grid_height, grid_width,
        columns, factor, reinterpret_cast<cudaStream_t>(stream));
    return launch_status();
}

bool fp8(Sm120ExecutionMode mode) {
    return mode == Sm120ExecutionMode::kStaticFp8E4M3 ||
           mode == Sm120ExecutionMode::kObservedFp8E4M3;
}

bool static_fp8(Sm120ExecutionMode mode) {
    return mode == Sm120ExecutionMode::kStaticFp8E4M3;
}

bool observed_fp8(Sm120ExecutionMode mode) {
    return mode == Sm120ExecutionMode::kObservedFp8E4M3;
}

bool valid_config(const Sm120TargetConfig& config) {
    if (config.checkpoint_path.empty()) return false;
    switch (config.execution_mode) {
        case Sm120ExecutionMode::kBf16:
            return !config.calibration.has_value();
        case Sm120ExecutionMode::kStaticFp8E4M3:
            return config.calibration.has_value();
        case Sm120ExecutionMode::kObservedFp8E4M3:
            return !config.calibration.has_value();
    }
    return false;
}

}  // namespace

struct Sm120TargetBundle::Impl final {
    enum class State {
        kConstructed = 0,
        kResourcesInitialized,
        kResourcesResolved,
        kPrepareComplete,
        kSetupFinalized,
        kCaptureInputsInitialized,
        kFailed,
    };

    Impl(frt_ctx context,
         Pi05ResolvedShape resolved_shape,
         Sm120TargetConfig target_config)
        : shape(resolved_shape),
          config(std::move(target_config)),
          weights(context),
          workspace(context),
          attention(context),
          scratch(context) {}

    modalities::Status fail(modalities::Status status) {
        state = State::kFailed;
        return status;
    }

    Pi05ResolvedShape shape;
    Sm120TargetConfig config;
    NativeDeviceWeightStore weights;
    frt_buffer action_out_step_weight = nullptr;
    frt_buffer action_out_step_bias = nullptr;
    NativeWorkspace workspace;
    Sm120AttentionBacking attention;
    Sm120Bf16ScratchBacking scratch;
    std::unique_ptr<Sm120Bf16Linear> bf16_linear;
    Sm120FrontendBindings frontend;
    Pi05FrontendOps frontend_ops;
    std::unique_ptr<Sm120Fp8WeightPacker> fp8_packer;
    std::unique_ptr<Sm120Fp8ActivationBacking> fp8_activation;
    std::unique_ptr<Sm120Fp8Linear> fp8_linear;
    std::unique_ptr<Sm120AttentionDriver> attention_driver;
    Pi05ResolvedResources resources;
    Pi05NativeSupportBuffers support;
    std::size_t prepare_cursor = 0;
    int prompt_length = 0;
    State state = State::kConstructed;
};

Sm120TargetBundle::Sm120TargetBundle(frt_ctx context,
                                     std::unique_ptr<Impl> impl)
    : Pi05TargetBundle(context, false), impl_(std::move(impl)) {}

Sm120TargetBundle::~Sm120TargetBundle() = default;

std::unique_ptr<Sm120TargetBundle> Sm120TargetBundle::create(
    frt_ctx context,
    const Pi05ResolvedShape& shape,
    Sm120TargetConfig config,
    modalities::Status* status) {
    modalities::Status validation = validate_pi05_resolved_shape(shape);
    if (!context || !valid_config(config) || !validation.ok_status()) {
        set_status(status,
                   context && !validation.ok_status()
                       ? validation
                       : invalid("SM120 target configuration is invalid"));
        return nullptr;
    }
    try {
        std::unique_ptr<Impl> impl(
            new Impl(context, shape, std::move(config)));
        std::unique_ptr<Sm120TargetBundle> target(
            new Sm120TargetBundle(context, std::move(impl)));
        set_status(status, modalities::Status::ok());
        return target;
    } catch (const std::exception& error) {
        set_status(status, backend(error.what()));
    } catch (...) {
        set_status(status, backend("SM120 target allocation failed"));
    }
    return nullptr;
}

modalities::Status Sm120TargetBundle::initialize_resources() {
    if (!impl_ || impl_->state != Impl::State::kConstructed) {
        return invalid("SM120 target resource state is invalid");
    }
    try {
        modalities::Status status = validate_device();
        if (!status.ok_status()) return impl_->fail(std::move(status));
        if (fp8(impl_->config.execution_mode)) {
            status = Sm120Fp8Linear::runtime_status();
            if (!status.ok_status()) return impl_->fail(std::move(status));
        }

        const std::filesystem::path checkpoint =
            std::filesystem::path(impl_->config.checkpoint_path) /
            "model.safetensors";
        loader::SafetensorsFile source;
        if (!source.open(checkpoint.string())) {
            return impl_->fail(modalities::Status::error(
                modalities::StatusCode::kNotFound, source.error()));
        }
        NativeWeightMaterializer materializer(source, &impl_->weights);
        NativeMaterializationOptions options;
        options.num_steps = impl_->shape.num_steps;
        options.include_embedding = true;
        status = materializer.materialize_all(options);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        NativeWorkspaceConfig workspace_config;
        workspace_config.num_views = impl_->shape.num_views;
        workspace_config.max_prompt_tokens =
            impl_->shape.max_prompt_tokens;
        workspace_config.chunk_size = impl_->shape.chunk;
        workspace_config.num_steps = impl_->shape.num_steps;
        workspace_config.vision_pool_factor =
            impl_->shape.vision_pool_factor;
        NativeWorkspaceRequirements requirements;
        requirements.activation_dtype = modalities::DType::kBFloat16;
        status = impl_->workspace.allocate(workspace_config, requirements);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->workspace.expand_vision_position_embedding(
            impl_->weights);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->attention.allocate(impl_->shape);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->scratch.allocate(
            impl_->shape, fp8(impl_->config.execution_mode));
        if (!status.ok_status()) return impl_->fail(std::move(status));

        impl_->bf16_linear.reset(new (std::nothrow) Sm120Bf16Linear());
        if (!impl_->bf16_linear) {
            return impl_->fail(backend("SM120 BF16 linear allocation failed"));
        }
        status = impl_->bf16_linear->status();
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->attention.set_prompt_length(0);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = impl_->workspace.update_decoder_rope(0);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        impl_->state = Impl::State::kResourcesInitialized;
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return impl_->fail(backend(error.what()));
    } catch (...) {
        return impl_->fail(backend("SM120 resource initialization failed"));
    }
}

modalities::Status Sm120TargetBundle::resolve_resources(
    Pi05ResolvedResources* out) {
    if (!impl_ || !out ||
        impl_->state != Impl::State::kResourcesInitialized) {
        return invalid("SM120 target resolve state is invalid");
    }
    try {
        Pi05TargetBufferBindings target_bindings;
        modalities::Status status =
            impl_->attention.make_target_bindings(&target_bindings);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        Pi05ResolvedResources resolved;
        status = resolve_pi05_native_buffers(
            impl_->workspace, target_bindings, impl_->shape,
            &resolved.buffers);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        Pi05NativeWeightLayout layout;
        layout.encoder = NativeFeedForwardLayout::kSeparateGateUp;
        layout.decoder = NativeFeedForwardLayout::kSeparateGateUp;
        status = resolve_pi05_materialized_weights(
            impl_->weights, impl_->shape, modalities::DType::kBFloat16,
            layout, &resolved.weights);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        status = derive_action_step_weights(
            context(), impl_->shape, &resolved.weights,
            &impl_->action_out_step_weight, &impl_->action_out_step_bias);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        Pi05NativeSupportBuffers support;
        status = resolve_pi05_native_support_buffers(
            impl_->workspace, impl_->shape, &support);
        if (!status.ok_status()) return impl_->fail(std::move(status));
        if (fp8(impl_->config.execution_mode)) {
            impl_->fp8_packer.reset(new (std::nothrow)
                                        Sm120Fp8WeightPacker(context()));
            if (!impl_->fp8_packer) {
                return impl_->fail(
                    backend("SM120 FP8 weight packer allocation failed"));
            }
            status = visit_pi05_linear_weight_groups(
                &resolved.weights, impl_->fp8_packer.get());
            if (status.ok_status()) status = impl_->fp8_packer->finish();
            if (!status.ok_status()) return impl_->fail(std::move(status));
        }
        status = validate_pi05_resolved_resources(resolved, impl_->shape);
        if (!status.ok_status()) return impl_->fail(std::move(status));

        if (fp8(impl_->config.execution_mode)) {
            impl_->fp8_activation.reset(new (std::nothrow)
                                            Sm120Fp8ActivationBacking(
                                                context()));
            if (!impl_->fp8_activation) {
                return impl_->fail(
                    backend("SM120 FP8 activation allocation failed"));
            }
            if (static_fp8(impl_->config.execution_mode)) {
                status = impl_->fp8_activation->initialize_static(
                    impl_->shape, *impl_->config.calibration);
            } else {
                status = impl_->fp8_activation->initialize_observer(
                    impl_->shape);
            }
            if (!status.ok_status()) return impl_->fail(std::move(status));
            const Sm120Fp8ExecutionMode linear_mode =
                observed_fp8(impl_->config.execution_mode)
                    ? Sm120Fp8ExecutionMode::kObserve
                    : Sm120Fp8ExecutionMode::kStatic;
            impl_->fp8_linear.reset(new (std::nothrow) Sm120Fp8Linear(
                impl_->fp8_activation.get(), linear_mode));
            if (!impl_->fp8_linear) {
                return impl_->fail(
                    backend("SM120 FP8 linear allocation failed"));
            }
            status = impl_->fp8_linear->status();
            if (!status.ok_status()) return impl_->fail(std::move(status));
        }

        impl_->resources = resolved;
        impl_->support = support;
        impl_->frontend = {&impl_->shape, impl_->bf16_linear.get(),
                           impl_->fp8_linear.get(), &impl_->attention,
                           nullptr};
        impl_->frontend_ops.profile.activation_dtype =
            modalities::DType::kBFloat16;
        impl_->frontend_ops.profile.vision_final_norm_epsilon =
            kPi05ModelNumerics.vision_layer_norm_epsilon;
        Pi05PrimitiveSet& primitives = impl_->frontend_ops.bf16;
        primitives.state = &impl_->frontend;
        primitives.linear = frontend_linear;
        primitives.projected_linear = frontend_projected_linear;
        primitives.add_bias = frontend_add_bias;
        primitives.silu = frontend_silu;
        primitives.copy = frontend_copy;
        primitives.patchify = frontend_patchify;
        primitives.bias_residual = frontend_bias_residual;
        primitives.layer_norm = frontend_layer_norm;
        primitives.qkv_split = frontend_qkv_split;
        primitives.normalize_for_linear = frontend_normalize_for_linear;
        primitives.qkv_rope = frontend_qkv_rope;
        primitives.adaptive_normalize = frontend_adaptive_normalize;
        primitives.diffusion_update = frontend_diffusion_update;
        primitives.attention = frontend_attention;
        primitives.gate_up = frontend_gate_up;
        primitives.gated_activation = frontend_gated_activation;
        primitives.gelu = frontend_gelu;
        primitives.vision_pool = frontend_vision_pool;
        impl_->state = Impl::State::kResourcesResolved;
        *out = resolved;
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return impl_->fail(backend(error.what()));
    } catch (...) {
        return impl_->fail(backend("SM120 resource resolution failed"));
    }
}

modalities::Status Sm120TargetBundle::make_prepare_execution(
    Pi05PrepareExecution* out) {
    if (!impl_ || !out || impl_->state != Impl::State::kResourcesResolved ||
        !impl_->frontend_ops.bf16.linear) {
        return invalid("SM120 prepare execution state is invalid");
    }
    const Sm120DecoderBf16Scratch& decoder = impl_->scratch.decoder();
    *out = {&impl_->resources,
            &impl_->frontend_ops,
            {decoder.normalized.device_data(), decoder.gate.device_data(),
             decoder.normalized.bytes()}};
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::complete_prepare() {
    if (!impl_ || impl_->state != Impl::State::kResourcesResolved) {
        return invalid("SM120 prepare completion state is invalid");
    }
    impl_->prepare_cursor =
        static_cast<std::size_t>(impl_->shape.num_steps) *
        (2 + 2 * kPi05ModelDims.decoder_layers);
    impl_->state = Impl::State::kPrepareComplete;
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::finalize_setup() {
    if (!impl_ || impl_->state != Impl::State::kPrepareComplete) {
        return invalid("SM120 target prepare is incomplete");
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    if (synchronized != cudaSuccess) {
        return impl_->fail(backend(cudaGetErrorString(synchronized)));
    }
    impl_->attention_driver.reset(
        new (std::nothrow) Sm120AttentionDriver(&impl_->attention));
    if (!impl_->attention_driver) {
        return impl_->fail(backend("SM120 attention driver allocation failed"));
    }
    modalities::Status status = impl_->attention_driver->status();
    if (!status.ok_status()) return impl_->fail(std::move(status));
    impl_->frontend.attention = impl_->attention_driver.get();
    if (impl_->fp8_linear) {
        status = autotune_sm120_fp8(
            impl_->shape, &impl_->resources, impl_->scratch,
            impl_->fp8_linear.get());
        if (!status.ok_status()) return impl_->fail(std::move(status));
    }
    impl_->state = Impl::State::kSetupFinalized;
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::make_forward_execution(
    Pi05ForwardExecution* out) {
    if (!impl_ || !out || impl_->state != Impl::State::kSetupFinalized ||
        !impl_->attention_driver) {
        return invalid("SM120 forward execution state is invalid");
    }
    const Sm120VisionBf16Scratch& scratch = impl_->scratch.vision();
    const Sm120VisionAttentionBuffers& attention = impl_->attention.vision();
    const Sm120EncoderBf16Scratch& encoder_scratch =
        impl_->scratch.encoder();
    const Sm120EncoderAttentionBuffers& encoder_attention =
        impl_->attention.encoder();
    const Sm120DecoderBf16Scratch& decoder_scratch =
        impl_->scratch.decoder();
    const Sm120DecoderAttentionBuffers& decoder_attention =
        impl_->attention.decoder();
    auto data = [](const Pi05ResolvedBuffer& buffer) {
        return buffer.buffer ? frt_buffer_dptr(buffer.buffer) : nullptr;
    };
    out->resources = &impl_->resources;
    out->ops = &impl_->frontend_ops;
    out->vision = {
        data(impl_->support.vision_patches),
        data(impl_->support.expanded_vision_position),
        data(impl_->support.pooled_vision_state),
        scratch.normalized.device_data(),
        scratch.qkv.device_data(),
        scratch.hidden.device_data(),
        attention.query.device_data(),
        attention.key.device_data(),
        attention.value.device_data(),
        impl_->attention_driver->vision_output(),
    };
    out->encoder = {
        impl_->support.encoder_rms_weight,
        encoder_scratch.normalized.device_data(),
        encoder_scratch.qkv.device_data(),
        encoder_scratch.gate.device_data(),
        encoder_scratch.hidden.device_data(),
        encoder_attention.query.device_data(),
        impl_->attention_driver->encoder_output(),
    };
    out->decoder = {
        impl_->support.decoder_rms_weight,
        decoder_scratch.normalized.device_data(),
        decoder_scratch.gate.device_data(),
        decoder_scratch.qkv.device_data(),
        decoder_scratch.gate_projection.device_data(),
        decoder_scratch.hidden.device_data(),
        decoder_attention.query.device_data(),
        impl_->attention_driver->decoder_output(),
    };
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::initialize_capture_inputs() {
    if (!impl_ || impl_->state != Impl::State::kSetupFinalized) {
        return invalid("SM120 capture input state is invalid");
    }
    const Pi05ResolvedBuffer* buffers[] = {
        &impl_->resources.buffers.images,
        &impl_->resources.buffers.prompt_embedding,
        &impl_->resources.buffers.encoder_state,
        &impl_->resources.buffers.noise,
    };
    for (const Pi05ResolvedBuffer* buffer : buffers) {
        if (!buffer->buffer ||
            cudaMemset(frt_buffer_dptr(buffer->buffer), 0,
                       frt_buffer_bytes(buffer->buffer)) != cudaSuccess) {
            return impl_->fail(backend("SM120 capture input reset failed"));
        }
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    if (synchronized != cudaSuccess) {
        return impl_->fail(backend(cudaGetErrorString(synchronized)));
    }
    impl_->state = Impl::State::kCaptureInputsInitialized;
    return modalities::Status::ok();
}

modalities::Status Sm120TargetBundle::reset_after_warmup() {
    if (!impl_ ||
        impl_->state != Impl::State::kCaptureInputsInitialized ||
        !impl_->resources.buffers.noise.buffer) {
        return invalid("SM120 warmup reset state is invalid");
    }
    const cudaError_t result = cudaMemset(
        frt_buffer_dptr(impl_->resources.buffers.noise.buffer), 0,
        frt_buffer_bytes(impl_->resources.buffers.noise.buffer));
    return result == cudaSuccess
               ? modalities::Status::ok()
               : impl_->fail(backend(cudaGetErrorString(result)));
}

modalities::Status Sm120TargetBundle::set_prompt_length(
    int prompt_tokens) {
    if (!impl_ || impl_->state == Impl::State::kConstructed ||
        impl_->state == Impl::State::kFailed || prompt_tokens < 0 ||
        prompt_tokens > impl_->shape.max_prompt_tokens) {
        return invalid("SM120 prompt length is out of range");
    }
    const int previous = impl_->prompt_length;
    modalities::Status status =
        impl_->attention.set_prompt_length(prompt_tokens);
    if (!status.ok_status()) return status;
    status = impl_->workspace.update_decoder_rope(prompt_tokens);
    if (!status.ok_status()) {
        impl_->attention.set_prompt_length(previous);
        impl_->workspace.update_decoder_rope(previous);
        return status;
    }
    impl_->prompt_length = prompt_tokens;
    return modalities::Status::ok();
}

const Pi05ResolvedResources* Sm120TargetBundle::resolved_resources() const {
    if (!impl_ || impl_->state == Impl::State::kConstructed ||
        impl_->state == Impl::State::kResourcesInitialized ||
        impl_->state == Impl::State::kFailed) {
        return nullptr;
    }
    return &impl_->resources;
}

std::size_t Sm120TargetBundle::materialized_weight_count() const {
    return impl_ ? impl_->weights.size() : 0;
}

std::size_t Sm120TargetBundle::packed_weight_count() const {
    return impl_ && impl_->fp8_packer ? impl_->fp8_packer->size() : 0;
}

std::size_t Sm120TargetBundle::autotuned_shape_count() const {
    return impl_ && impl_->fp8_linear
               ? impl_->fp8_linear->autotuned_shape_count()
               : 0;
}

std::size_t Sm120TargetBundle::prepare_call_count() const {
    return impl_ ? impl_->prepare_cursor : 0;
}

Sm120ExecutionMode Sm120TargetBundle::execution_mode() const {
    return impl_ ? impl_->config.execution_mode
                 : Sm120ExecutionMode::kBf16;
}

bool Sm120TargetBundle::observes_activations() const {
    return impl_ && observed_fp8(impl_->config.execution_mode);
}

modalities::Status Sm120TargetBundle::reset_observer(Pi05Stream stream) {
    if (!impl_ || impl_->state != Impl::State::kCaptureInputsInitialized ||
        !observed_fp8(impl_->config.execution_mode) ||
        !impl_->fp8_activation) {
        return invalid("SM120 observer reset state is invalid");
    }
    return impl_->fp8_activation->reset_observer_scales(stream);
}

modalities::Status Sm120TargetBundle::download_observer(
    Pi05ObservedScales* out) const {
    if (!impl_ || impl_->state != Impl::State::kCaptureInputsInitialized ||
        !observed_fp8(impl_->config.execution_mode) ||
        !impl_->fp8_activation || !out) {
        return invalid("SM120 observer download state is invalid");
    }
    return impl_->fp8_activation->download_observer_scales(
        &out->vision, &out->encoder, &out->decoder);
}

bool Sm120TargetBundle::ready_for_capture() const {
    return impl_ &&
           impl_->state == Impl::State::kCaptureInputsInitialized;
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt
