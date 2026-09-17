// TensorRT IPluginV3 for the whole pi0.5 Thor action decoder (all denoise
// steps in one call) on FlashRT kernels.
//
// Op: "Pi05Decoder", version "1".
// Inputs:
//   0 noise     fp16 [S, 32]
//   1 prefix_k  fp16 [L * E * HD] prefix K rows written by the encoder, flattened
//   2 prefix_v  fp16 [L * E * HD] (a [L, E, HD] input reaches the plugin
//                                  with wrong dims and layout on TensorRT 10.16)
//   3 ain_w fp16 [32, D]   4 ain_b fp16 [D]   5 aow fp16 [D, 32]   6 aob fp16 [32]
//   7 rope  fp16 [S, 256]
//   8 sa    fp16 [steps*L*S*3D]   9 sf fp16 [steps*L*S*3D]   10 fs fp16 [steps*S*3D]
//   11-18 per-layer NVFP4 blobs concatenated over layers, carried as INT32:
//         qw_fp4, qw_sfb, ow_fp4, ow_sfb, gwil_fp4, gwil_sfb, dw_fp4, dw_sfb
// Output: 0 actions fp16 [S, 32] (raw, before unnormalisation)
// Attributes: S, D, H, NH, HD, L, steps, v_qkv, v_o, v_gu, v_down (int32), dt (float32).
//
// The K/V cache [L, E + S, HD] lives in the plugin workspace: the prefix is
// copied in once per call and every step writes its suffix rows there. This
// keeps the cache out of TensorRT's I/O tensors, which a plugin must not
// modify in place.
#include "pi05_decoder_step.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "fused_fp4/pdl.cuh"

namespace flashrt_trt {
namespace pi05 {
namespace {

using namespace nvinfer1;

constexpr const char* kName = "Pi05Decoder";
constexpr int32_t kNbInputs = 19;
constexpr int64_t kAlign = 64;
const char* const kIntFields[] = {"S", "D", "H", "NH", "HD", "L", "steps", "v_qkv", "v_o", "v_gu", "v_down"};

int64_t align_up(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

cublasHandle_t shared_cublas() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        return cublasCreate(&h) == CUBLAS_STATUS_SUCCESS ? h : nullptr;
    }();
    return handle;
}

struct DecoderConfig {
    DecoderDims dims;
    int steps = 10;
};

int64_t kv_bytes(const DecoderDims& d, int total_keys) {
    return static_cast<int64_t>(d.L) * total_keys * d.HD * 2;
}

class DecoderPlugin final : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild,
                            public IPluginV3OneRuntime {
public:
    explicit DecoderPlugin(const DecoderConfig& c) : c_(c) { refresh_fields(); }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new DecoderPlugin(c_); }
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
        const int64_t max_prefix = in[1].max.d[0] / (static_cast<int64_t>(c_.dims.L) * c_.dims.HD);
        if (max_prefix <= 0) return 0;
        const int max_t = static_cast<int>(max_prefix) + c_.dims.S;
        return static_cast<size_t>(decoder_scratch_bytes(c_.dims, max_t) + 2 * align_up(kv_bytes(c_.dims, max_t)));
    }
    int32_t onShapeChange(const PluginTensorDesc*, int32_t, const PluginTensorDesc*, int32_t) noexcept override {
        // Runs before execution, outside any CUDA graph capture.
        const char* pdl = std::getenv("FLASHRT_TRT_PDL");
        flash_rt::fp4::pdl_flag() = pdl == nullptr || pdl[0] != '0';
        return shared_cublas() != nullptr ? 0 : -1;
    }
    int32_t enqueue(const PluginTensorDesc* inDesc, const PluginTensorDesc*, const void* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        cublasHandle_t cublas = shared_cublas();
        if (cublas == nullptr) return -1;
        if (std::getenv("FLASHRT_TRT_CHECKSUM") != nullptr) {
            cudaStreamSynchronize(stream);
            std::fprintf(stderr, "[flashrt] workspace=%p\n", workspace);
            for (int i = 1; i <= 2; ++i) {
                std::fprintf(stderr, "[flashrt] in%d nbDims=%d d=[%lld,%lld,%lld,%lld] ptr=%p\n", i, inDesc[i].dims.nbDims,
                             static_cast<long long>(inDesc[i].dims.d[0]), static_cast<long long>(inDesc[i].dims.d[1]),
                             static_cast<long long>(inDesc[i].dims.d[2]), static_cast<long long>(inDesc[i].dims.d[3]),
                             static_cast<void*>(const_cast<void*>(inputs[i])));
            }
            for (int i = 0; i < kNbInputs; ++i) {
                int64_t n = 1;
                for (int k = 0; k < inDesc[i].dims.nbDims; ++k) n *= inDesc[i].dims.d[k];
                const int64_t bytes = n * (inDesc[i].type == DataType::kINT32 ? 4 : 2);
                std::vector<unsigned char> h(static_cast<size_t>(bytes));
                cudaMemcpy(h.data(), inputs[i], h.size(), cudaMemcpyDeviceToHost);
                uint64_t wsum = 0;
                for (size_t j = 0; j < h.size(); ++j) wsum += static_cast<uint64_t>(h[j]) * ((j % 65521) + 1);
                std::fprintf(stderr, "[flashrt] in%-2d fmt=%d bytes=%lld wsum=%llu head=", i,
                             static_cast<int>(inDesc[i].format), static_cast<long long>(bytes),
                             static_cast<unsigned long long>(wsum));
                for (int j = 0; j < 8 && j < static_cast<int>(h.size()); ++j) std::fprintf(stderr, "%02x", h[j]);
                std::fprintf(stderr, "\n");
            }
        }
        DecoderDims d = c_.dims;
        if (inDesc[1].dims.nbDims != 1 || inDesc[1].dims.d[0] % (static_cast<int64_t>(d.L) * d.HD) != 0) return -1;
        const int prefix = static_cast<int>(inDesc[1].dims.d[0] / (static_cast<int64_t>(d.L) * d.HD));
        d.total_keys = prefix + d.S;

        char* base = static_cast<char*>(workspace);
        const int64_t kvb = kv_bytes(d, d.total_keys);
        void* kv_k = base;
        void* kv_v = base + align_up(kvb);
        DecoderScratch s;
        decoder_bind_scratch(d, d.total_keys, base + 2 * align_up(kvb), &s);

        // Prefix rows of every layer into the cache.
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
        if (cudaMemcpyAsync(outputs[0], inputs[0], static_cast<size_t>(d.S) * 32 * 2, cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            return -1;
        }

        DecoderWeights w;
        w.ain_w = inputs[3]; w.ain_b = inputs[4]; w.aow = inputs[5]; w.aob = inputs[6]; w.rope = inputs[7];
        w.qw_fp4 = inputs[11]; w.qw_sfb = inputs[12]; w.ow_fp4 = inputs[13]; w.ow_sfb = inputs[14];
        w.gwil_fp4 = inputs[15]; w.gwil_sfb = inputs[16]; w.dw_fp4 = inputs[17]; w.dw_sfb = inputs[18];
        const int64_t style = static_cast<int64_t>(d.L) * d.S * 3 * d.D * 2;
        const int64_t final_style = static_cast<int64_t>(d.S) * 3 * d.D * 2;
        for (int st = 0; st < c_.steps; ++st) {
            w.sa = static_cast<const char*>(inputs[8]) + style * st;
            w.sf = static_cast<const char*>(inputs[9]) + style * st;
            w.fs = static_cast<const char*>(inputs[10]) + final_style * st;
            if (decoder_step_forward(d, w, s, cublas, outputs[0], kv_k, kv_v, stream) != 0) return -1;
        }
        return 0;
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
        const DecoderDims& d = c_.dims;
        ints_ = {d.S, d.D, d.H, d.NH, d.HD, d.L, c_.steps, d.v_qkv, d.v_o, d.v_gu, d.v_down};
        dt_ = d.dt;
        fields_.clear();
        for (size_t i = 0; i < ints_.size(); ++i) fields_.emplace_back(kIntFields[i], &ints_[i], PluginFieldType::kINT32, 1);
        fields_.emplace_back("dt", &dt_, PluginFieldType::kFLOAT32, 1);
    }

    DecoderConfig c_;
    std::vector<int32_t> ints_;
    float dt_ = 0.0f;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

}  // namespace

class DecoderCreator final : public IPluginCreatorV3One {
public:
    DecoderCreator() {
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
        DecoderConfig c;
        DecoderDims& d = c.dims;
        int* slots[] = {&d.S, &d.D, &d.H, &d.NH, &d.HD, &d.L, &c.steps, &d.v_qkv, &d.v_o, &d.v_gu, &d.v_down};
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
                    *slots[k] = f.type == PluginFieldType::kINT64 ? static_cast<int>(*static_cast<const int64_t*>(f.data))
                                                                  : *static_cast<const int32_t*>(f.data);
                }
            }
        }
        return new DecoderPlugin(c);
    }

private:
    std::vector<PluginField> names_;
    PluginFieldCollection collection_{};
};

}  // namespace pi05
}  // namespace flashrt_trt

nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_decoder_creator() {
    static flashrt_trt::pi05::DecoderCreator creator;
    return &creator;
}
