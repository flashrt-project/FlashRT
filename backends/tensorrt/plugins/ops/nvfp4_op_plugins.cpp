// TensorRT IPluginV3 shells over the FlashRT operator layer (kernels/frt_ops.h).
//
// These are the model-free operators, for hosts that want to place FlashRT
// kernels in their own graph rather than take a whole pi0.5 stage:
//
//   flashrt::FlashrtNvfp4Linear     quantize -> NVFP4 GEMM -> epilogue
//   flashrt::FlashrtNvfp4Mlp        quantize -> gate/up + activation -> down
//   flashrt::FlashrtFa4Attention    FlashAttention-4 (head_dim 256 GQA, 72 MHA)
//
// NVFP4 codes and block scales travel in INT32 tensors of ceil(bytes / 4)
// elements: TensorRT constants do not accept UINT8 and INT32 storage is passed
// through bit-exactly. Optional inputs follow the required ones in the order
// documented per operator, and `opt_mask` says which of them are present.
#include "frt_ops.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace flashrt_trt {
namespace ops {
namespace {

using namespace nvinfer1;

constexpr const char* kVersion = "1";
constexpr const char* kNamespace = "";

int field_int(const PluginField& f) {
    switch (f.type) {
    case PluginFieldType::kINT64: return static_cast<int>(*static_cast<const int64_t*>(f.data));
    case PluginFieldType::kINT32: return *static_cast<const int32_t*>(f.data);
    default: return 0;
    }
}

float field_float(const PluginField& f) {
    switch (f.type) {
    case PluginFieldType::kFLOAT64: return static_cast<float>(*static_cast<const double*>(f.data));
    case PluginFieldType::kFLOAT32: return *static_cast<const float*>(f.data);
    default: return 0.0f;
    }
}

// Attributes of the two GEMM operators. One struct keeps the creators short;
// each operator reads the fields it needs.
struct OpAttrs {
    int n = 0, k = 0, d = 0, h = 0;
    int norm_mode = FRT_NORM_NONE;
    int epilogue = FRT_EPI_NONE;
    int gate_mode = FRT_GATE_GEGLU_IL;
    int variant = 0, gate_variant = 0, down_variant = 0;
    int opt_mask = 0;
    int mode = 0, nh = 1, head_dim = 256, batch = 1;
    float eps = 1e-6f, scale = 0.0f;
};

// Index of an optional input, or -1 when the mask says it is absent.
int opt_index(int mask, int required, int bit) {
    if ((mask & (1 << bit)) == 0) return -1;
    int idx = required;
    for (int b = 0; b < bit; ++b) {
        if (mask & (1 << b)) ++idx;
    }
    return idx;
}


// ------------------------------------------------------------------------
//  FlashrtNvfp4Linear
//    required inputs: x fp16 [M, K], w_packed blob, w_sfb blob
//    optional inputs (bit order): gamma, beta, awq_inv_s, bias, residual
//    output: fp16 [M, N]
// ------------------------------------------------------------------------
class LinearPlugin final : public IPluginV3,
                           public IPluginV3OneCore,
                           public IPluginV3OneBuild,
                           public IPluginV3OneRuntime {
public:
    explicit LinearPlugin(const OpAttrs& a) : a_(a) { refresh_fields(); }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new LinearPlugin(a_); }
    const char* getPluginName() const noexcept override { return "FlashrtNvfp4Linear"; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t configurePlugin(const DynamicPluginTensorDesc*, int32_t,
                            const DynamicPluginTensorDesc*, int32_t) noexcept override {
        return 0;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t nbOutputs, const DataType*,
                               int32_t) const noexcept override {
        for (int32_t i = 0; i < nbOutputs; ++i) out[i] = DataType::kHALF;
        return 0;
    }
    int32_t getOutputShapes(const DimsExprs* inputs, int32_t, const DimsExprs*, int32_t,
                            DimsExprs* outputs, int32_t, IExprBuilder& expr) noexcept override {
        outputs[0].nbDims = 2;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = expr.constant(a_.n);
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
        // x is fp16, the two weight blobs are int32, every optional is fp16.
        const DataType expected = (pos == 1 || pos == 2) ? DataType::kINT32 : DataType::kHALF;
        return inOut[pos].desc.type == expected;
    }
    int32_t getNbOutputs() const noexcept override { return 1; }
    size_t getWorkspaceSize(const DynamicPluginTensorDesc* in, int32_t,
                            const DynamicPluginTensorDesc*, int32_t) const noexcept override {
        const int64_t m = in[0].max.d[0];
        return m > 0 ? frt_nvfp4_linear_workspace(static_cast<int32_t>(m), a_.k) : 0;
    }
    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc*,
                          int32_t) noexcept override {
        frt_set_pdl(1);
        return 0;
    }
    int32_t enqueue(const PluginTensorDesc* inDesc, const PluginTensorDesc*,
                    const void* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        const int32_t m = static_cast<int32_t>(inDesc[0].dims.d[0]);
        const int gamma = opt_index(a_.opt_mask, 3, 0), beta = opt_index(a_.opt_mask, 3, 1),
                  awq = opt_index(a_.opt_mask, 3, 2), bias = opt_index(a_.opt_mask, 3, 3),
                  res = opt_index(a_.opt_mask, 3, 4);
        const void* residual = res >= 0 ? inputs[res] : nullptr;

        // An accumulating epilogue adds into its output buffer; at operator
        // granularity the residual lives in another tensor, so it is copied in.
        if (a_.epilogue == FRT_EPI_ACCUM && residual != nullptr &&
            cudaMemcpyAsync(outputs[0], residual, static_cast<size_t>(m) * a_.n * 2,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            return -1;
        }
        return frt_nvfp4_linear(inputs[0], gamma >= 0 ? inputs[gamma] : nullptr,
                                beta >= 0 ? inputs[beta] : nullptr,
                                awq >= 0 ? inputs[awq] : nullptr, inputs[1], inputs[2],
                                bias >= 0 ? inputs[bias] : nullptr, residual, outputs[0], m,
                                a_.n, a_.k, a_.norm_mode, a_.epilogue, a_.variant, a_.eps,
                                workspace, stream) == 0
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
        ints_ = {a_.n, a_.k, a_.norm_mode, a_.epilogue, a_.variant, a_.opt_mask};
        static const char* const names[] = {"N", "K", "norm_mode", "epilogue", "variant",
                                            "opt_mask"};
        fields_.clear();
        for (size_t i = 0; i < ints_.size(); ++i) {
            fields_.emplace_back(names[i], &ints_[i], PluginFieldType::kINT32, 1);
        }
        fields_.emplace_back("eps", &a_.eps, PluginFieldType::kFLOAT32, 1);
    }

    OpAttrs a_;
    std::vector<int32_t> ints_;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

// ------------------------------------------------------------------------
//  FlashrtNvfp4Mlp
//    required: x fp16 [M, D], gate_up_packed, gate_up_sfb, down_packed, down_sfb
//    optional (bit order): gamma, beta, awq_inv_s, gate_up_bias, down_bias, residual
//    output: fp16 [M, D]
// ------------------------------------------------------------------------
class MlpPlugin final : public IPluginV3,
                        public IPluginV3OneCore,
                        public IPluginV3OneBuild,
                        public IPluginV3OneRuntime {
public:
    explicit MlpPlugin(const OpAttrs& a) : a_(a) { refresh_fields(); }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new MlpPlugin(a_); }
    const char* getPluginName() const noexcept override { return "FlashrtNvfp4Mlp"; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t configurePlugin(const DynamicPluginTensorDesc*, int32_t,
                            const DynamicPluginTensorDesc*, int32_t) noexcept override {
        return 0;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t nbOutputs, const DataType*,
                               int32_t) const noexcept override {
        for (int32_t i = 0; i < nbOutputs; ++i) out[i] = DataType::kHALF;
        return 0;
    }
    int32_t getOutputShapes(const DimsExprs* inputs, int32_t, const DimsExprs*, int32_t,
                            DimsExprs* outputs, int32_t, IExprBuilder& expr) noexcept override {
        outputs[0].nbDims = 2;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = expr.constant(a_.d);
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
        const bool blob = pos >= 1 && pos <= 4;  // the four weight tensors
        return inOut[pos].desc.type == (blob ? DataType::kINT32 : DataType::kHALF);
    }
    int32_t getNbOutputs() const noexcept override { return 1; }
    size_t getWorkspaceSize(const DynamicPluginTensorDesc* in, int32_t,
                            const DynamicPluginTensorDesc*, int32_t) const noexcept override {
        const int64_t m = in[0].max.d[0];
        return m > 0 ? frt_nvfp4_mlp_workspace(static_cast<int32_t>(m), a_.d, a_.h, a_.gate_mode)
                     : 0;
    }
    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc*,
                          int32_t) noexcept override {
        frt_set_pdl(1);
        return 0;
    }
    int32_t enqueue(const PluginTensorDesc* inDesc, const PluginTensorDesc*,
                    const void* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        const int32_t m = static_cast<int32_t>(inDesc[0].dims.d[0]);
        const int gamma = opt_index(a_.opt_mask, 5, 0), beta = opt_index(a_.opt_mask, 5, 1),
                  awq = opt_index(a_.opt_mask, 5, 2), gu_bias = opt_index(a_.opt_mask, 5, 3),
                  dn_bias = opt_index(a_.opt_mask, 5, 4), res = opt_index(a_.opt_mask, 5, 5);
        const void* residual = res >= 0 ? inputs[res] : nullptr;

        if (a_.epilogue == FRT_EPI_ACCUM && residual != nullptr &&
            cudaMemcpyAsync(outputs[0], residual, static_cast<size_t>(m) * a_.d * 2,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            return -1;
        }
        return frt_nvfp4_mlp(inputs[0], gamma >= 0 ? inputs[gamma] : nullptr,
                             beta >= 0 ? inputs[beta] : nullptr, awq >= 0 ? inputs[awq] : nullptr,
                             inputs[1], inputs[2], gu_bias >= 0 ? inputs[gu_bias] : nullptr,
                             inputs[3], inputs[4], dn_bias >= 0 ? inputs[dn_bias] : nullptr,
                             residual, outputs[0], m, a_.d, a_.h, a_.norm_mode, a_.gate_mode,
                             a_.gate_variant, a_.down_variant, a_.epilogue, a_.eps, workspace,
                             stream) == 0
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
        ints_ = {a_.d,            a_.h,        a_.norm_mode,    a_.gate_mode,
                 a_.gate_variant, a_.down_variant, a_.epilogue, a_.opt_mask};
        static const char* const names[] = {"D",            "H",            "norm_mode",
                                            "gate_mode",    "gate_variant", "down_variant",
                                            "epilogue",     "opt_mask"};
        fields_.clear();
        for (size_t i = 0; i < ints_.size(); ++i) {
            fields_.emplace_back(names[i], &ints_[i], PluginFieldType::kINT32, 1);
        }
        fields_.emplace_back("eps", &a_.eps, PluginFieldType::kFLOAT32, 1);
    }

    OpAttrs a_;
    std::vector<int32_t> ints_;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

// ------------------------------------------------------------------------
//  FlashrtFa4Attention
//    mode 0: GQA, head_dim 256, one KV head. q fp16 [Sq, NH*256],
//            k/v fp16 [Sk, 256]; output [Sq, NH*256].
//    mode 1: MHA, head_dim 72. q/k/v fp16 [B*S, NH*72]; output [B*S, NH*72].
// ------------------------------------------------------------------------
class AttentionPlugin final : public IPluginV3,
                              public IPluginV3OneCore,
                              public IPluginV3OneBuild,
                              public IPluginV3OneRuntime {
public:
    explicit AttentionPlugin(const OpAttrs& a) : a_(a) { refresh_fields(); }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new AttentionPlugin(a_); }
    const char* getPluginName() const noexcept override { return "FlashrtFa4Attention"; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }

    int32_t configurePlugin(const DynamicPluginTensorDesc*, int32_t,
                            const DynamicPluginTensorDesc*, int32_t) noexcept override {
        return 0;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t nbOutputs, const DataType*,
                               int32_t) const noexcept override {
        for (int32_t i = 0; i < nbOutputs; ++i) out[i] = DataType::kHALF;
        return 0;
    }
    int32_t getOutputShapes(const DimsExprs* inputs, int32_t, const DimsExprs*, int32_t,
                            DimsExprs* outputs, int32_t, IExprBuilder&) noexcept override {
        outputs[0] = inputs[0];
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override {
        if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
        return inOut[pos].desc.type == DataType::kHALF;
    }
    int32_t getNbOutputs() const noexcept override { return 1; }
    size_t getWorkspaceSize(const DynamicPluginTensorDesc*, int32_t,
                            const DynamicPluginTensorDesc*, int32_t) const noexcept override {
        return 0;
    }
    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc*,
                          int32_t) noexcept override {
        // Loads the ahead-of-time FA4 modules; never during graph capture.
        return frt_fa4_load() == 0 ? 0 : -1;
    }
    int32_t enqueue(const PluginTensorDesc* inDesc, const PluginTensorDesc*,
                    const void* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        const float scale = a_.scale > 0.0f ? a_.scale : frt_fa4_default_scale(a_.head_dim);
        const int32_t rows = static_cast<int32_t>(inDesc[0].dims.d[0]);
        if (a_.mode == 0) {
            const int32_t sk = static_cast<int32_t>(inDesc[1].dims.d[0]);
            return frt_fa4_gqa_hd256(inputs[0], inputs[1], inputs[2], outputs[0], rows, sk, a_.nh,
                                     scale, stream) == 0
                       ? 0
                       : -1;
        }
        const int32_t b = a_.batch > 0 ? a_.batch : 1;
        const int32_t s = rows / b;
        const int64_t row = static_cast<int64_t>(a_.nh) * a_.head_dim;
        return frt_fa4_mha_hd72(inputs[0], row, inputs[1], row, inputs[2], row, outputs[0], b, s,
                                a_.nh, scale, stream) == 0
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
        ints_ = {a_.mode, a_.nh, a_.head_dim, a_.batch};
        static const char* const names[] = {"mode", "NH", "head_dim", "batch"};
        fields_.clear();
        for (size_t i = 0; i < ints_.size(); ++i) {
            fields_.emplace_back(names[i], &ints_[i], PluginFieldType::kINT32, 1);
        }
        fields_.emplace_back("scale", &a_.scale, PluginFieldType::kFLOAT32, 1);
    }

    OpAttrs a_;
    std::vector<int32_t> ints_;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

// One creator template for the three operators: the attribute names are a
// superset and each plugin reads the ones it defines.
template <typename PluginT>
class OpCreator final : public IPluginCreatorV3One {
public:
    OpCreator(const char* name, std::vector<const char*> int_fields)
        : name_(name), int_names_(std::move(int_fields)) {
        for (const char* n : int_names_) {
            names_.emplace_back(n, nullptr, PluginFieldType::kINT32, 1);
        }
        names_.emplace_back("eps", nullptr, PluginFieldType::kFLOAT32, 1);
        names_.emplace_back("scale", nullptr, PluginFieldType::kFLOAT32, 1);
        collection_.nbFields = static_cast<int32_t>(names_.size());
        collection_.fields = names_.data();
    }
    const char* getPluginName() const noexcept override { return name_; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &collection_; }

    IPluginV3* createPlugin(const char*, const PluginFieldCollection* fc,
                            TensorRTPhase) noexcept override {
        OpAttrs a;
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
            const PluginField& f = fc->fields[i];
            if (f.name == nullptr || f.data == nullptr) continue;
            const std::string n(f.name);
            if (n == "N") a.n = field_int(f);
            else if (n == "K") a.k = field_int(f);
            else if (n == "D") a.d = field_int(f);
            else if (n == "H") a.h = field_int(f);
            else if (n == "norm_mode") a.norm_mode = field_int(f);
            else if (n == "epilogue") a.epilogue = field_int(f);
            else if (n == "gate_mode") a.gate_mode = field_int(f);
            else if (n == "variant") a.variant = field_int(f);
            else if (n == "gate_variant") a.gate_variant = field_int(f);
            else if (n == "down_variant") a.down_variant = field_int(f);
            else if (n == "opt_mask") a.opt_mask = field_int(f);
            else if (n == "mode") a.mode = field_int(f);
            else if (n == "NH") a.nh = field_int(f);
            else if (n == "head_dim") a.head_dim = field_int(f);
            else if (n == "batch") a.batch = field_int(f);
            else if (n == "eps") a.eps = field_float(f);
            else if (n == "scale") a.scale = field_float(f);
        }
        return new PluginT(a);
    }

private:
    const char* name_;
    std::vector<const char*> int_names_;
    std::vector<PluginField> names_;
    PluginFieldCollection collection_{};
};

}  // namespace
}  // namespace ops
}  // namespace flashrt_trt

nvinfer1::IPluginCreatorInterface* flashrt_trt_nvfp4_linear_creator() {
    static flashrt_trt::ops::OpCreator<flashrt_trt::ops::LinearPlugin> creator(
        "FlashrtNvfp4Linear", {"N", "K", "norm_mode", "epilogue", "variant", "opt_mask"});
    return &creator;
}

nvinfer1::IPluginCreatorInterface* flashrt_trt_nvfp4_mlp_creator() {
    static flashrt_trt::ops::OpCreator<flashrt_trt::ops::MlpPlugin> creator(
        "FlashrtNvfp4Mlp", {"D", "H", "norm_mode", "gate_mode", "gate_variant", "down_variant",
                            "epilogue", "opt_mask"});
    return &creator;
}

nvinfer1::IPluginCreatorInterface* flashrt_trt_fa4_attention_creator() {
    static flashrt_trt::ops::OpCreator<flashrt_trt::ops::AttentionPlugin> creator(
        "FlashrtFa4Attention", {"mode", "NH", "head_dim", "batch"});
    return &creator;
}
