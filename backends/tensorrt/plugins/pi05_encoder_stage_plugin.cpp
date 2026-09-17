// TensorRT IPluginV3 for the whole pi0.5 Thor encoder (all layers) on
// FlashRT kernels. Same numerics as a chain of Pi05EncoderLayer nodes; one
// plugin boundary for the stage.
//
// ONNX op: domain "flashrt", op_type "Pi05Encoder".
// Inputs:
//   0 x    fp16 [Se, D]
//   1 rope fp16 [Se, HD]
//   then for l in [0, L):      qkv_w (blob), qkv_scale (float [1])
//   then for l in [0, L - 1):  o_packed, o_sfb (blob), awq_inv_s (fp16 [D]),
//                              gu_il_packed, gu_il_sfb, down_packed, down_sfb (blob)
//   Blobs are byte buffers carried in INT32 tensors.
// Outputs: 0 x_out fp16 [Se, D], 1 k fp16 [L*Se, HD], 2 v fp16 [L*Se, HD]
//   K/V rows are layer-major. They are flattened to 2D because TensorRT 10.16
//   lays out dynamic 3D plugin tensors incorrectly at builder optimization
//   level 0; the decoder plugin takes its prefix rows in the same 2D form.
// Attributes: D, H, NH, HD, L, attn_o_variant, down_variant, qkv_alpha (float[L]).
#include "pi05_encoder_layer.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>

namespace flashrt_trt {
namespace pi05 {
namespace {

using namespace nvinfer1;

constexpr const char* kName = "Pi05Encoder";
constexpr const char* kVersion = "1";
constexpr const char* kNamespace = "";
constexpr int32_t kNbOutputs = 3;
constexpr int32_t kPerLayerTail = 7;

int32_t nb_inputs(int L) { return 2 + 2 * L + kPerLayerTail * (L - 1); }

DataType input_type(int L, int32_t pos) {
    if (pos <= 1) return DataType::kHALF;
    int32_t p = pos - 2;
    if (p < 2 * L) return (p % 2 == 0) ? DataType::kINT32 : DataType::kFLOAT;
    p -= 2 * L;
    return (p % kPerLayerTail == 2) ? DataType::kHALF : DataType::kINT32;
}

class EncoderStagePlugin final : public IPluginV3,
                                 public IPluginV3OneCore,
                                 public IPluginV3OneBuild,
                                 public IPluginV3OneRuntime {
public:
    EncoderStagePlugin(const EncoderLayerDims& dims, int L, std::vector<float> alphas)
        : dims_(dims), L_(L), alphas_(std::move(alphas)) {
        alphas_.resize(static_cast<size_t>(L_), 1.0f);
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new EncoderStagePlugin(dims_, L_, alphas_); }

    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t configurePlugin(const DynamicPluginTensorDesc*, int32_t, const DynamicPluginTensorDesc*,
                            int32_t) noexcept override {
        return 0;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t nbOutputs, const DataType*,
                               int32_t) const noexcept override {
        for (int32_t i = 0; i < nbOutputs; ++i) out[i] = DataType::kHALF;
        return 0;
    }
    int32_t getOutputShapes(const DimsExprs* inputs, int32_t nbInputs, const DimsExprs*, int32_t,
                            DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& expr) noexcept override {
        if (nbInputs != nb_inputs(L_) || nbOutputs != kNbOutputs) return -1;
        outputs[0] = inputs[0];
        for (int32_t i = 1; i < 3; ++i) {
            outputs[i].nbDims = 2;
            outputs[i].d[0] = expr.operation(DimensionOperation::kPROD, *expr.constant(L_),
                                             *inputs[0].d[0]);
            outputs[i].d[1] = expr.constant(dims_.HD);
        }
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (nbInputs != nb_inputs(L_) || nbOutputs != kNbOutputs || pos < 0 ||
            pos >= nbInputs + nbOutputs) {
            return false;
        }
        const DataType expected = pos < nbInputs ? input_type(L_, pos) : DataType::kHALF;
        return inOut[pos].desc.type == expected;
    }
    int32_t getNbOutputs() const noexcept override { return kNbOutputs; }
    size_t getWorkspaceSize(const DynamicPluginTensorDesc* in, int32_t, const DynamicPluginTensorDesc*,
                            int32_t) const noexcept override {
        const int64_t max_se = in[0].max.d[0];
        return max_se > 0 ? static_cast<size_t>(encoder_layer_scratch_bytes(dims_, static_cast<int>(max_se)))
                          : 0;
    }

    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc*,
                          int32_t) noexcept override {
        encoder_layer_set_pdl(true);
        return encoder_layer_load_kernels() == 0 ? 0 : -1;
    }
    int32_t enqueue(const PluginTensorDesc* inDesc, const PluginTensorDesc*,
                    const void* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        EncoderLayerDims d = dims_;
        d.Se = static_cast<int>(inDesc[0].dims.d[0]);
        EncoderLayerScratch s;
        encoder_layer_bind_scratch(d, d.Se, workspace, &s);
        const size_t x_bytes = static_cast<size_t>(d.Se) * d.D * 2;
        const size_t kv_bytes = static_cast<size_t>(d.Se) * d.HD * 2;
        if (outputs[0] != inputs[0] &&
            cudaMemcpyAsync(outputs[0], inputs[0], x_bytes, cudaMemcpyDeviceToDevice, stream) !=
                cudaSuccess) {
            return -1;
        }
        for (int l = 0; l < L_; ++l) {
            EncoderLayerDims dl = d;
            dl.last = (l == L_ - 1);
            EncoderLayerWeights w;
            w.rope = inputs[1];
            w.qkv_w = inputs[2 + 2 * l];
            w.qkv_act_scale = static_cast<const float*>(inputs[3 + 2 * l]);
            w.qkv_alpha = alphas_[static_cast<size_t>(l)];
            if (!dl.last) {
                const int b = 2 + 2 * L_ + kPerLayerTail * l;
                w.o_packed = inputs[b];
                w.o_sfb = inputs[b + 1];
                w.awq_inv_s_gu = inputs[b + 2];
                w.gu_il_packed = inputs[b + 3];
                w.gu_il_sfb = inputs[b + 4];
                w.down_packed = inputs[b + 5];
                w.down_sfb = inputs[b + 6];
            }
            if (encoder_layer_forward(dl, w, s, outputs[0], static_cast<char*>(outputs[1]) + l * kv_bytes,
                                      static_cast<char*>(outputs[2]) + l * kv_bytes, stream) != 0) {
                return -1;
            }
        }
        return 0;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    const PluginFieldCollection* getFieldsToSerialize() noexcept override {
        ints_ = {dims_.D, dims_.H, dims_.NH, dims_.HD, L_, dims_.attn_o_variant, dims_.down_variant};
        static const char* const names[] = {"D", "H", "NH", "HD", "L", "attn_o_variant", "down_variant"};
        fields_.clear();
        for (size_t i = 0; i < ints_.size(); ++i) {
            fields_.emplace_back(names[i], &ints_[i], PluginFieldType::kINT32, 1);
        }
        fields_.emplace_back("qkv_alpha", alphas_.data(), PluginFieldType::kFLOAT32,
                             static_cast<int32_t>(alphas_.size()));
        collection_.nbFields = static_cast<int32_t>(fields_.size());
        collection_.fields = fields_.data();
        return &collection_;
    }

private:
    EncoderLayerDims dims_;
    int L_;
    std::vector<float> alphas_;
    std::vector<int32_t> ints_;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

}  // namespace

class EncoderStageCreator final : public IPluginCreatorV3One {
public:
    EncoderStageCreator() {
        for (const char* n : {"D", "H", "NH", "HD", "L", "attn_o_variant", "down_variant"}) {
            names_.emplace_back(n, nullptr, PluginFieldType::kINT32, 1);
        }
        names_.emplace_back("qkv_alpha", nullptr, PluginFieldType::kFLOAT32, 0);
        collection_.nbFields = static_cast<int32_t>(names_.size());
        collection_.fields = names_.data();
    }
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &collection_; }

    IPluginV3* createPlugin(const char*, const PluginFieldCollection* fc,
                            TensorRTPhase) noexcept override {
        EncoderLayerDims d;
        int L = 18;
        std::vector<float> alphas;
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
            const PluginField& f = fc->fields[i];
            if (f.name == nullptr || f.data == nullptr) continue;
            const std::string name(f.name);
            auto as_int = [&f]() -> int {
                if (f.type == PluginFieldType::kINT64) return static_cast<int>(*static_cast<const int64_t*>(f.data));
                return *static_cast<const int32_t*>(f.data);
            };
            if (name == "D") d.D = as_int();
            else if (name == "H") d.H = as_int();
            else if (name == "NH") d.NH = as_int();
            else if (name == "HD") d.HD = as_int();
            else if (name == "L") L = as_int();
            else if (name == "attn_o_variant") d.attn_o_variant = as_int();
            else if (name == "down_variant") d.down_variant = as_int();
            else if (name == "qkv_alpha") {
                alphas.resize(static_cast<size_t>(f.length));
                for (int32_t k = 0; k < f.length; ++k) {
                    alphas[static_cast<size_t>(k)] =
                        f.type == PluginFieldType::kFLOAT64
                            ? static_cast<float>(static_cast<const double*>(f.data)[k])
                            : static_cast<const float*>(f.data)[k];
                }
            }
        }
        return new EncoderStagePlugin(d, L, alphas);
    }

private:
    std::vector<PluginField> names_;
    PluginFieldCollection collection_{};
};

}  // namespace pi05
}  // namespace flashrt_trt

nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_encoder_creator() {
    static flashrt_trt::pi05::EncoderStageCreator creator;
    return &creator;
}
