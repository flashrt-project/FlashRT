// TensorRT IPluginV3 for one pi0.5 Thor denoise step on FlashRT kernels.
//
// Op: "Pi05DecoderStep", version "1".
// Inputs:
//   0 noise     fp16 [S, 32]
//   1 prefix_k  fp16 [L * E * HD] prefix K rows written by the encoder, flattened
//   2 prefix_v  fp16 [L * E * HD] (dynamic 3D plugin inputs are laid out wrongly
//                                  on TensorRT 10.16, see Pi05Decoder)
//   3 ain_w   fp16 [32, D]     4 ain_b fp16 [D]
//   5 aow     fp16 [D, 32]     6 aob   fp16 [32]
//   7 rope    fp16 [S, 256]
//   8 sa      fp16 [L*S*3D]    this step's attention styles
//   9 sf      fp16 [L*S*3D]    this step's FFN styles
//  10 fs      fp16 [S*3D]      this step's final style
//  11-18 per-layer NVFP4 blobs concatenated over layers, carried as INT32:
//        qw_fp4, qw_sfb, ow_fp4, ow_sfb, gwil_fp4, gwil_sfb, dw_fp4, dw_sfb
// Output: 0 noise_out fp16 [S, 32]
// Attributes: S, D, H, NH, HD, L, v_qkv, v_o, v_gu, v_down (int32), dt (float32).
//
// A step writes its suffix K/V rows and reads them back within the same call,
// so no cache state crosses steps: the cache [L, E + S, HD] lives in this
// plugin's workspace and the prefix is copied in on every call (TensorRT may
// hand the same workspace to other nodes between steps, and a plugin must not
// modify engine inputs). Chaining steps therefore costs one prefix copy per
// step over the Pi05Decoder stage plugin.
#include "pi05_decoder_step.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "fused_fp4/pdl.cuh"

namespace flashrt_trt {
namespace pi05 {
namespace {

using namespace nvinfer1;

constexpr const char* kName = "Pi05DecoderStep";
constexpr int32_t kNbInputs = 19;
constexpr int64_t kAlign = 64;

int64_t align_up(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

int64_t kv_bytes(const DecoderDims& d, int total_keys) {
    return static_cast<int64_t>(d.L) * total_keys * d.HD * 2;
}

// One cuBLAS handle per process: TensorRT clones plugins for every build-time
// measurement and execution context, and creating a handle each time is slow.
// The step binds its stream on every call.
cublasHandle_t shared_cublas() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        return cublasCreate(&h) == CUBLAS_STATUS_SUCCESS ? h : nullptr;
    }();
    return handle;
}

const char* const kIntFields[] = {"S", "D", "H", "NH", "HD", "L", "v_qkv", "v_o", "v_gu", "v_down"};

class DecoderStepPlugin final : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild,
                                public IPluginV3OneRuntime {
public:
    explicit DecoderStepPlugin(const DecoderDims& d) : d_(d) { refresh_fields(); }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new DecoderStepPlugin(d_); }

    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return ""; }

    int32_t configurePlugin(const DynamicPluginTensorDesc*, int32_t, const DynamicPluginTensorDesc*,
                            int32_t) noexcept override {
        return 0;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t, const DataType*, int32_t) const noexcept override {
        out[0] = DataType::kHALF;
        return 0;
    }
    int32_t getOutputShapes(const DimsExprs* in, int32_t nbInputs, const DimsExprs*, int32_t, DimsExprs* out,
                            int32_t nbOutputs, IExprBuilder&) noexcept override {
        if (nbInputs != kNbInputs || nbOutputs != 1) return -1;
        out[0] = in[0];
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* io, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override {
        if (nbInputs != kNbInputs || nbOutputs != 1 || pos < 0 || pos > nbInputs) return false;
        const DataType want = (pos >= 11 && pos < nbInputs) ? DataType::kINT32 : DataType::kHALF;
        return io[pos].desc.type == want;
    }
    int32_t getNbOutputs() const noexcept override { return 1; }
    size_t getWorkspaceSize(const DynamicPluginTensorDesc* in, int32_t, const DynamicPluginTensorDesc*,
                            int32_t) const noexcept override {
        const int64_t max_prefix = in[1].max.d[0] / (static_cast<int64_t>(d_.L) * d_.HD);
        if (max_prefix <= 0) return 0;
        const int max_t = static_cast<int>(max_prefix) + d_.S;
        return static_cast<size_t>(decoder_scratch_bytes(d_, max_t) + 2 * align_up(kv_bytes(d_, max_t)));
    }

    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc*, int32_t) noexcept override {
        // Runs before execution, outside any CUDA graph capture.
        flash_rt::fp4::pdl_flag() = true;  // programmatic dependent launch, as in FlashRT's Thor default
        return shared_cublas() != nullptr ? 0 : -1;
    }
    int32_t enqueue(const PluginTensorDesc* inDesc, const PluginTensorDesc*, const void* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        cublasHandle_t cublas = shared_cublas();
        if (cublas == nullptr) return -1;
        DecoderDims d = d_;
        if (inDesc[1].dims.nbDims != 1 || inDesc[1].dims.d[0] % (static_cast<int64_t>(d.L) * d.HD) != 0) return -1;
        const int prefix = static_cast<int>(inDesc[1].dims.d[0] / (static_cast<int64_t>(d.L) * d.HD));
        d.total_keys = prefix + d.S;

        char* base = static_cast<char*>(workspace);
        const int64_t kvb = kv_bytes(d, d.total_keys);
        void* kv_k = base;
        void* kv_v = base + align_up(kvb);
        DecoderScratch s;
        decoder_bind_scratch(d, d.total_keys, base + 2 * align_up(kvb), &s);

        const int64_t row = static_cast<int64_t>(d.HD) * 2;
        const int64_t prefix_layer = row * prefix;
        const int64_t cache_layer = row * d.total_keys;
        for (int l = 0; l < d.L; ++l) {
            if (cudaMemcpyAsync(static_cast<char*>(kv_k) + cache_layer * l,
                                static_cast<const char*>(inputs[1]) + prefix_layer * l, prefix_layer,
                                cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
                cudaMemcpyAsync(static_cast<char*>(kv_v) + cache_layer * l,
                                static_cast<const char*>(inputs[2]) + prefix_layer * l, prefix_layer,
                                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return -1;
            }
        }
        if (outputs[0] != inputs[0] &&
            cudaMemcpyAsync(outputs[0], inputs[0], static_cast<size_t>(d.S) * 32 * 2, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            return -1;
        }

        DecoderWeights w;
        w.ain_w = inputs[3]; w.ain_b = inputs[4]; w.aow = inputs[5]; w.aob = inputs[6];
        w.rope = inputs[7]; w.sa = inputs[8]; w.sf = inputs[9]; w.fs = inputs[10];
        w.qw_fp4 = inputs[11]; w.qw_sfb = inputs[12]; w.ow_fp4 = inputs[13]; w.ow_sfb = inputs[14];
        w.gwil_fp4 = inputs[15]; w.gwil_sfb = inputs[16]; w.dw_fp4 = inputs[17]; w.dw_sfb = inputs[18];
        return decoder_step_forward(d, w, s, cublas, outputs[0], kv_k, kv_v, stream) == 0 ? 0 : -1;
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
        ints_ = {d_.S, d_.D, d_.H, d_.NH, d_.HD, d_.L, d_.v_qkv, d_.v_o, d_.v_gu, d_.v_down};
        dt_ = d_.dt;
        fields_.clear();
        for (size_t i = 0; i < ints_.size(); ++i) {
            fields_.emplace_back(kIntFields[i], &ints_[i], PluginFieldType::kINT32, 1);
        }
        fields_.emplace_back("dt", &dt_, PluginFieldType::kFLOAT32, 1);
    }

    DecoderDims d_;
    std::vector<int32_t> ints_;
    float dt_ = 0.0f;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

}  // namespace

class DecoderStepCreator final : public IPluginCreatorV3One {
public:
    DecoderStepCreator() {
        for (const char* n : kIntFields) names_.emplace_back(n, nullptr, PluginFieldType::kINT32, 1);
        names_.emplace_back("dt", nullptr, PluginFieldType::kFLOAT32, 1);
        collection_.nbFields = static_cast<int32_t>(names_.size());
        collection_.fields = names_.data();
    }
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return ""; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &collection_; }

    IPluginV3* createPlugin(const char*, const PluginFieldCollection* fc, TensorRTPhase) noexcept override {
        DecoderDims d;
        int* slots[] = {&d.S, &d.D, &d.H, &d.NH, &d.HD, &d.L, &d.v_qkv, &d.v_o, &d.v_gu, &d.v_down};
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i) {
            const PluginField& f = fc->fields[i];
            if (f.name == nullptr || f.data == nullptr) continue;
            const std::string name(f.name);
            if (name == "dt") {
                d.dt = f.type == PluginFieldType::kFLOAT64 ? static_cast<float>(*static_cast<const double*>(f.data))
                                                           : *static_cast<const float*>(f.data);
                continue;
            }
            for (size_t k = 0; k < sizeof(kIntFields) / sizeof(kIntFields[0]); ++k) {
                if (name == kIntFields[k]) {
                    *slots[k] = f.type == PluginFieldType::kINT64
                                    ? static_cast<int>(*static_cast<const int64_t*>(f.data))
                                    : *static_cast<const int32_t*>(f.data);
                }
            }
        }
        return new DecoderStepPlugin(d);
    }

private:
    std::vector<PluginField> names_;
    PluginFieldCollection collection_{};
};

}  // namespace pi05
}  // namespace flashrt_trt

// Exported through getCreators() in pi05_plugins.cpp.
nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_decoder_step_creator() {
    static flashrt_trt::pi05::DecoderStepCreator creator;
    return &creator;
}
