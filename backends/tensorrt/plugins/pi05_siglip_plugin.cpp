// TensorRT IPluginV3 plugins for the pi0.5 Thor SigLIP vision stage on
// FlashRT kernels.
//
// Pi05SiglipLayer (ONNX domain "flashrt"): one transformer layer.
//   Inputs:  0 x fp16 [S, D] (S = views * spv), then the 15 layer tensors below.
//   Outputs: 0 x_out fp16 [S, D]
//   Attributes: D, H_pad, NH, HD, spv, up_variant, qkv_alpha, o_alpha.
//
// Pi05Siglip: the whole stage, images to projected image tokens.
//   Inputs:  0 images fp16 [views, 224, 224, 3], HWC in [-1, 1] (TensorRT
//            10.16 plugins take no uint8 tensors; map uint8 pixels with
//            x / 127.5 - 1 in fp32 before the fp16 cast to match FlashRT's
//            uint8 path bit for bit)
//            1 pe_w fp16 [588, D], 2 pe_b fp16 [D], 3 pos_emb fp16 [spv, D],
//            4 postln_w, 5 postln_b fp16 [D], 6 proj_w fp16 [D, De],
//            7 proj_b fp16 [De],
//            then the 15 layer tensors for each of the L layers.
//   Outputs: 0 tokens fp16 [views * spv, De]
//   Attributes: D, H_pad, NH, HD, spv, De, up_variant, L,
//               alpha (float[2L]: qkv, o per layer).
//
// Layer tensors, in order (blobs are byte buffers carried in INT32 tensors):
//   ln_attn_w fp16 [D], ln_attn_b fp16 [D], qkv_w blob (e4m3 [D, 3D]),
//   qkv_b fp16 [3D], o_w blob (e4m3 [D, D]), o_b fp16 [D], ln_ffn_w fp16 [D],
//   ln_ffn_b fp16 [D], awq_inv_s fp16 [D], up_packed blob, up_sfb blob,
//   up_b fp16 [H_pad], down_packed blob, down_sfb blob, down_b fp16 [D]
#include "pi05_siglip.h"

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

constexpr const char* kLayerName = "Pi05SiglipLayer";
constexpr const char* kStageName = "Pi05Siglip";
constexpr const char* kVersion = "1";
constexpr const char* kNamespace = "";
constexpr int32_t kPerLayer = 15;
constexpr int32_t kStageHead = 8;

bool layer_input_is_blob(int32_t p) { return p == 2 || p == 4 || p == 9 || p == 10 || p == 12 || p == 13; }

SiglipLayerWeights bind_layer(const void* const* in, int32_t base, float qkv_alpha, float o_alpha) {
    SiglipLayerWeights w;
    w.ln_attn_w = in[base];
    w.ln_attn_b = in[base + 1];
    w.qkv_w = in[base + 2];
    w.qkv_b = in[base + 3];
    w.qkv_alpha = qkv_alpha;
    w.o_w = in[base + 4];
    w.o_b = in[base + 5];
    w.o_alpha = o_alpha;
    w.ln_ffn_w = in[base + 6];
    w.ln_ffn_b = in[base + 7];
    w.awq_inv_s = in[base + 8];
    w.up_packed = in[base + 9];
    w.up_sfb = in[base + 10];
    w.up_b = in[base + 11];
    w.down_packed = in[base + 12];
    w.down_sfb = in[base + 13];
    w.down_b = in[base + 14];
    return w;
}

class SiglipPlugin final : public IPluginV3,
                           public IPluginV3OneCore,
                           public IPluginV3OneBuild,
                           public IPluginV3OneRuntime {
public:
    SiglipPlugin(bool stage, const SiglipDims& dims, int L, std::vector<float> alpha)
        : stage_(stage), dims_(dims), L_(stage ? L : 1), alpha_(std::move(alpha)) {
        alpha_.resize(static_cast<size_t>(2 * L_), 1.0f);
    }
    ~SiglipPlugin() override { siglip_gemm_destroy(gemm_); }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new SiglipPlugin(stage_, dims_, L_, alpha_); }

    const char* getPluginName() const noexcept override { return stage_ ? kStageName : kLayerName; }
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
        if (nbInputs != nb_inputs() || nbOutputs != 1) return -1;
        if (!stage_) {
            outputs[0] = inputs[0];
            return 0;
        }
        outputs[0].nbDims = 2;
        outputs[0].d[0] = expr.operation(DimensionOperation::kPROD, *inputs[0].d[0],
                                         *expr.constant(dims_.spv));
        outputs[0].d[1] = expr.constant(dims_.De);
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (nbInputs != nb_inputs() || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs) {
            return false;
        }
        return inOut[pos].desc.type == expected_type(pos);
    }
    int32_t getNbOutputs() const noexcept override { return 1; }
    size_t getWorkspaceSize(const DynamicPluginTensorDesc* in, int32_t, const DynamicPluginTensorDesc*,
                            int32_t) const noexcept override {
        const int64_t rows = in[0].max.d[0];
        if (rows <= 0) return 0;
        const int64_t max_s = stage_ ? rows * dims_.spv : rows;
        return static_cast<size_t>(siglip_scratch_bytes(dims_, static_cast<int>(max_s)));
    }

    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc*,
                          int32_t) noexcept override {
        if (siglip_load_kernels() != 0) return -1;
        if (gemm_ == nullptr) gemm_ = siglip_gemm_create();
        return gemm_ != nullptr ? 0 : -1;
    }
    int32_t enqueue(const PluginTensorDesc* inDesc, const PluginTensorDesc*,
                    const void* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (gemm_ == nullptr) return -1;
        SiglipScratch s;
        if (!stage_) {
            const int S = static_cast<int>(inDesc[0].dims.d[0]);
            if (S % dims_.spv != 0) return -1;
            siglip_bind_scratch(dims_, S, workspace, &s);
            if (outputs[0] != inputs[0] &&
                cudaMemcpyAsync(outputs[0], inputs[0], static_cast<size_t>(S) * dims_.D * 2,
                                 cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return -1;
            }
            return siglip_layer_forward(dims_, S / dims_.spv, bind_layer(inputs, 1, alpha_[0], alpha_[1]),
                                        s, gemm_, outputs[0], stream);
        }
        const int nv = static_cast<int>(inDesc[0].dims.d[0]);
        const int S = nv * dims_.spv;
        siglip_bind_scratch(dims_, S, workspace, &s);
        SiglipEmbedWeights ew;
        ew.pe_w = inputs[1];
        ew.pe_b = inputs[2];
        ew.pos_emb = inputs[3];
        ew.postln_w = inputs[4];
        ew.postln_b = inputs[5];
        ew.proj_w = inputs[6];
        ew.proj_b = inputs[7];
        void* x = s.x;
        int rc = siglip_patch_embed(dims_, nv, ew, s, gemm_, inputs[0], false, x, stream);
        for (int l = 0; l < L_ && rc == 0; ++l) {
            rc = siglip_layer_forward(dims_, nv,
                                      bind_layer(inputs, kStageHead + kPerLayer * l,
                                                 alpha_[2 * l], alpha_[2 * l + 1]),
                                      s, gemm_, x, stream);
        }
        if (rc == 0) rc = siglip_post_project(dims_, S, ew, s, gemm_, x, outputs[0], stream);
        return rc;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    const PluginFieldCollection* getFieldsToSerialize() noexcept override {
        ints_ = {dims_.D, dims_.H_pad, dims_.NH, dims_.HD, dims_.spv, dims_.up_variant};
        static const char* const names[] = {"D", "H_pad", "NH", "HD", "spv", "up_variant", "De", "L"};
        if (stage_) {
            ints_.push_back(dims_.De);
            ints_.push_back(L_);
        }
        fields_.clear();
        for (size_t i = 0; i < ints_.size(); ++i) {
            fields_.emplace_back(names[i], &ints_[i], PluginFieldType::kINT32, 1);
        }
        if (stage_) {
            fields_.emplace_back("alpha", alpha_.data(), PluginFieldType::kFLOAT32,
                                 static_cast<int32_t>(alpha_.size()));
        } else {
            fields_.emplace_back("qkv_alpha", &alpha_[0], PluginFieldType::kFLOAT32, 1);
            fields_.emplace_back("o_alpha", &alpha_[1], PluginFieldType::kFLOAT32, 1);
        }
        collection_.nbFields = static_cast<int32_t>(fields_.size());
        collection_.fields = fields_.data();
        return &collection_;
    }

private:
    int32_t nb_inputs() const { return stage_ ? kStageHead + kPerLayer * L_ : 1 + kPerLayer; }

    DataType expected_type(int32_t pos) const {
        if (pos >= nb_inputs()) return DataType::kHALF;  // output
        if (!stage_) {
            return pos > 0 && layer_input_is_blob(pos - 1) ? DataType::kINT32 : DataType::kHALF;
        }
        if (pos < kStageHead) return DataType::kHALF;
        return layer_input_is_blob((pos - kStageHead) % kPerLayer) ? DataType::kINT32 : DataType::kHALF;
    }

    bool stage_;
    SiglipDims dims_;
    int L_;
    std::vector<float> alpha_;
    GemmRunner* gemm_ = nullptr;
    std::vector<int32_t> ints_;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

}  // namespace

class SiglipCreator final : public IPluginCreatorV3One {
public:
    explicit SiglipCreator(bool stage) : stage_(stage) {
        const std::vector<const char*> ints = stage ? std::vector<const char*>{"D", "H_pad", "NH", "HD", "spv",
                                                                               "up_variant", "De", "L"}
                                                    : std::vector<const char*>{"D", "H_pad", "NH", "HD", "spv",
                                                                               "up_variant"};
        for (const char* n : ints) names_.emplace_back(n, nullptr, PluginFieldType::kINT32, 1);
        if (stage) {
            names_.emplace_back("alpha", nullptr, PluginFieldType::kFLOAT32, 0);
        } else {
            names_.emplace_back("qkv_alpha", nullptr, PluginFieldType::kFLOAT32, 1);
            names_.emplace_back("o_alpha", nullptr, PluginFieldType::kFLOAT32, 1);
        }
        collection_.nbFields = static_cast<int32_t>(names_.size());
        collection_.fields = names_.data();
    }
    const char* getPluginName() const noexcept override { return stage_ ? kStageName : kLayerName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &collection_; }

    IPluginV3* createPlugin(const char*, const PluginFieldCollection* fc, TensorRTPhase) noexcept override {
        SiglipDims d;
        int L = 27;
        std::vector<float> alpha(2, 1.0f);
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
            const PluginField& f = fc->fields[i];
            if (f.name == nullptr || f.data == nullptr) continue;
            const std::string name(f.name);
            auto as_int = [&f]() -> int {
                if (f.type == PluginFieldType::kINT64) return static_cast<int>(*static_cast<const int64_t*>(f.data));
                return *static_cast<const int32_t*>(f.data);
            };
            auto as_float = [&f](int32_t k) -> float {
                return f.type == PluginFieldType::kFLOAT64 ? static_cast<float>(static_cast<const double*>(f.data)[k])
                                                           : static_cast<const float*>(f.data)[k];
            };
            if (name == "D") d.D = as_int();
            else if (name == "H_pad") d.H_pad = as_int();
            else if (name == "NH") d.NH = as_int();
            else if (name == "HD") d.HD = as_int();
            else if (name == "spv") d.spv = as_int();
            else if (name == "De") d.De = as_int();
            else if (name == "up_variant") d.up_variant = as_int();
            else if (name == "L") L = as_int();
            else if (name == "qkv_alpha") alpha[0] = as_float(0);
            else if (name == "o_alpha") alpha[1] = as_float(0);
            else if (name == "alpha") {
                alpha.resize(static_cast<size_t>(f.length));
                for (int32_t k = 0; k < f.length; ++k) alpha[static_cast<size_t>(k)] = as_float(k);
            }
        }
        return new SiglipPlugin(stage_, d, L, alpha);
    }

private:
    bool stage_;
    std::vector<PluginField> names_;
    PluginFieldCollection collection_{};
};

}  // namespace pi05
}  // namespace flashrt_trt

nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_siglip_layer_creator() {
    static flashrt_trt::pi05::SiglipCreator creator(false);
    return &creator;
}

nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_siglip_creator() {
    static flashrt_trt::pi05::SiglipCreator creator(true);
    return &creator;
}
