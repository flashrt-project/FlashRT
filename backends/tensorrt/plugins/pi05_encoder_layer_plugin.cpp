// TensorRT IPluginV3 for one pi0.5 Thor encoder layer on FlashRT kernels.
//
// ONNX op: domain "flashrt", op_type "Pi05EncoderLayer".
// Inputs:
//   0 x            fp16  [Se, D]      residual stream
//   1 rope         fp16  [Se, HD]
//   Byte blobs (e4m3 weights, NVFP4 codes, block scales) are carried in
//   INT32 tensors of ceil(bytes / 4) elements: TensorRT constants do not
//   accept UINT8, and INT32 storage is passed through bit-exactly.
//   2 qkv_w        blob  [2560, D]    e4m3 bytes
//   3 qkv_scale    float [1]          static activation scale
//   4 o_packed     blob  [D, D/2]
//   5 o_sfb        blob  [n]
//   6 awq_inv_s    fp16  [D]
//   7 gu_il_packed blob  [2H, D/2]
//   8 gu_il_sfb    blob  [n]
//   9 down_packed  blob  [D, H/2]
//  10 down_sfb     blob  [n]
// Outputs:
//   0 x_out fp16 [Se, D], 1 k fp16 [Se, HD], 2 v fp16 [Se, HD]
// Attributes: D, H, NH, HD, attn_o_variant, down_variant, last, qkv_alpha.
#include "pi05_encoder_layer.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace flashrt_trt {
namespace pi05 {
namespace {

using namespace nvinfer1;

constexpr const char* kName = "Pi05EncoderLayer";
constexpr const char* kVersion = "1";
constexpr const char* kNamespace = "";
constexpr int32_t kNbInputs = 11;
constexpr int32_t kNbOutputs = 3;

class EncoderLayerPlugin final : public IPluginV3,
                                 public IPluginV3OneCore,
                                 public IPluginV3OneBuild,
                                 public IPluginV3OneRuntime {
public:
    explicit EncoderLayerPlugin(const EncoderLayerDims& dims, float qkv_alpha)
        : dims_(dims), qkv_alpha_(qkv_alpha) {
        refresh_fields();
    }

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new EncoderLayerPlugin(dims_, qkv_alpha_); }

    // IPluginV3OneCore
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }

    // IPluginV3OneBuild
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
        if (nbInputs != kNbInputs || nbOutputs != kNbOutputs) return -1;
        outputs[0] = inputs[0];
        for (int32_t i = 1; i < 3; ++i) {
            outputs[i].nbDims = 2;
            outputs[i].d[0] = inputs[0].d[0];
            outputs[i].d[1] = expr.constant(dims_.HD);
        }
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (nbInputs != kNbInputs || nbOutputs != kNbOutputs || pos < 0 ||
            pos >= nbInputs + nbOutputs) {
            return false;
        }
        static const DataType want[kNbInputs] = {
            DataType::kHALF, DataType::kHALF, DataType::kINT32, DataType::kFLOAT,
            DataType::kINT32, DataType::kINT32, DataType::kHALF, DataType::kINT32,
            DataType::kINT32, DataType::kINT32, DataType::kINT32};
        const DataType expected = pos < nbInputs ? want[pos] : DataType::kHALF;
        return inOut[pos].desc.type == expected;
    }
    int32_t getNbOutputs() const noexcept override { return kNbOutputs; }
    size_t getWorkspaceSize(const DynamicPluginTensorDesc* in, int32_t, const DynamicPluginTensorDesc*,
                            int32_t) const noexcept override {
        const int64_t max_se = in[0].max.d[0];
        const size_t bytes = max_se > 0 ? static_cast<size_t>(
            encoder_layer_scratch_bytes(dims_, static_cast<int>(max_se))) : 0;
        return bytes;
    }
    // IPluginV3OneRuntime
    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc*,
                          int32_t) noexcept override {
        // Runs before execution and outside any CUDA graph capture.
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

        EncoderLayerWeights w;
        w.rope = inputs[1];
        w.qkv_w = inputs[2];
        w.qkv_act_scale = static_cast<const float*>(inputs[3]);
        w.qkv_alpha = qkv_alpha_;
        w.o_packed = inputs[4];
        w.o_sfb = inputs[5];
        w.awq_inv_s_gu = inputs[6];
        w.gu_il_packed = inputs[7];
        w.gu_il_sfb = inputs[8];
        w.down_packed = inputs[9];
        w.down_sfb = inputs[10];

        const size_t x_bytes = static_cast<size_t>(d.Se) * d.D * 2;
        if (outputs[0] != inputs[0] &&
            cudaMemcpyAsync(outputs[0], inputs[0], x_bytes, cudaMemcpyDeviceToDevice, stream) !=
                cudaSuccess) {
            return -1;
        }
        return encoder_layer_forward(d, w, s, outputs[0], outputs[1], outputs[2], stream) == 0
                   ? 0
                   : -1;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    const PluginFieldCollection* getFieldsToSerialize() noexcept override {
        refresh_fields();
        collection_.nbFields = static_cast<int32_t>(fields_.size());
        collection_.fields = fields_.data();
        return &collection_;
    }

private:
    void refresh_fields() {
        ints_ = {dims_.D, dims_.H, dims_.NH, dims_.HD, dims_.attn_o_variant, dims_.down_variant,
                 dims_.last ? 1 : 0};
        static const char* const names[] = {"D", "H", "NH", "HD", "attn_o_variant", "down_variant",
                                            "last"};
        fields_.clear();
        for (size_t i = 0; i < ints_.size(); ++i) {
            fields_.emplace_back(names[i], &ints_[i], PluginFieldType::kINT32, 1);
        }
        fields_.emplace_back("qkv_alpha", &qkv_alpha_, PluginFieldType::kFLOAT32, 1);
    }

    EncoderLayerDims dims_;
    float qkv_alpha_;
    std::vector<int32_t> ints_;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

}  // namespace

class EncoderLayerCreator final : public IPluginCreatorV3One {
public:
    EncoderLayerCreator() {
        for (const char* n : {"D", "H", "NH", "HD", "attn_o_variant", "down_variant", "last"}) {
            names_.emplace_back(n, nullptr, PluginFieldType::kINT32, 1);
        }
        names_.emplace_back("qkv_alpha", nullptr, PluginFieldType::kFLOAT32, 1);
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
        float alpha = 1.0f;
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
            const PluginField& f = fc->fields[i];
            if (f.name == nullptr || f.data == nullptr) continue;
            const std::string name(f.name);
            auto as_int = [&f]() -> int {
                switch (f.type) {
                case PluginFieldType::kINT64: return static_cast<int>(*static_cast<const int64_t*>(f.data));
                case PluginFieldType::kINT32: return *static_cast<const int32_t*>(f.data);
                default: return 0;
                }
            };
            if (name == "D") d.D = as_int();
            else if (name == "H") d.H = as_int();
            else if (name == "NH") d.NH = as_int();
            else if (name == "HD") d.HD = as_int();
            else if (name == "attn_o_variant") d.attn_o_variant = as_int();
            else if (name == "down_variant") d.down_variant = as_int();
            else if (name == "last") d.last = as_int() != 0;
            else if (name == "qkv_alpha") {
                alpha = f.type == PluginFieldType::kFLOAT64
                            ? static_cast<float>(*static_cast<const double*>(f.data))
                            : *static_cast<const float*>(f.data);
            }
        }
        return new EncoderLayerPlugin(d, alpha);
    }

private:
    std::vector<PluginField> names_;
    PluginFieldCollection collection_{};
};

}  // namespace pi05
}  // namespace flashrt_trt

nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_encoder_layer_creator() {
    static flashrt_trt::pi05::EncoderLayerCreator creator;
    return &creator;
}
