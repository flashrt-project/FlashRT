// Bitwise parity of the native pi0.5 encoder (all layers) against the FlashRT
// library encoder forward, plus a latency probe.
//
// usage: encoder_stage_parity <encoder_all.safetensors> [iters]
#include "pi05_encoder_layer.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct HostTensor {
    std::vector<uint8_t> bytes;
};

std::map<std::string, HostTensor> load_safetensors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    std::string header(header_len, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    const auto base = static_cast<std::streamoff>(8 + header_len);
    auto j = nlohmann::json::parse(header);
    std::map<std::string, HostTensor> out;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        auto off = it.value()["data_offsets"].get<std::vector<uint64_t>>();
        HostTensor t;
        t.bytes.resize(off[1] - off[0]);
        f.seekg(base + static_cast<std::streamoff>(off[0]));
        f.read(reinterpret_cast<char*>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
        out.emplace(it.key(), std::move(t));
    }
    return out;
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    explicit DeviceBuffer(size_t n) : bytes(n) {
        if (cudaMalloc(&ptr, std::max<size_t>(n, 1)) != cudaSuccess) throw std::runtime_error("cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(ptr); }
};

std::unique_ptr<DeviceBuffer> upload(const HostTensor& t) {
    auto b = std::make_unique<DeviceBuffer>(t.bytes.size());
    cudaMemcpy(b->ptr, t.bytes.data(), t.bytes.size(), cudaMemcpyHostToDevice);
    return b;
}

std::vector<uint8_t> download(void* p, size_t n) {
    std::vector<uint8_t> h(n);
    cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost);
    return h;
}

size_t diff_count(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t d = 0;
    for (size_t i = 0; i + 1 < a.size(); i += 2) d += (a[i] != b[i] || a[i + 1] != b[i + 1]);
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s encoder_all.safetensors [iters]\n", argv[0]);
        return 2;
    }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 100;
    auto T = load_safetensors(argv[1]);
    std::vector<int64_t> m(T.at("meta").bytes.size() / 8);
    std::memcpy(m.data(), T.at("meta").bytes.data(), T.at("meta").bytes.size());

    using namespace flashrt_trt::pi05;
    EncoderLayerDims base;
    base.Se = static_cast<int>(m[0]);
    base.D = static_cast<int>(m[1]);
    base.H = static_cast<int>(m[2]);
    base.NH = static_cast<int>(m[3]);
    base.HD = static_cast<int>(m[4]);
    base.attn_o_variant = static_cast<int>(m[6]);
    base.down_variant = static_cast<int>(m[7]);
    const int L = static_cast<int>(m[8]);
    const size_t kv_bytes = static_cast<size_t>(base.Se) * base.HD * 2;
    std::printf("Se=%d L=%d\n", base.Se, L);

    std::map<std::string, std::unique_ptr<DeviceBuffer>> dev;
    for (auto& [name, t] : T) {
        if (name.rfind("L", 0) == 0 && name.find(".k_out") == std::string::npos &&
            name.find(".v_out") == std::string::npos && name.find(".alpha_qkv") == std::string::npos) {
            dev[name] = upload(t);
        }
    }
    auto rope = upload(T.at("rope"));
    std::vector<EncoderLayerWeights> W(L);
    std::vector<EncoderLayerDims> Dm(L, base);
    for (int l = 0; l < L; ++l) {
        const std::string p = "L" + std::to_string(l) + ".";
        auto& w = W[l];
        w.qkv_w = dev.at(p + "qkv_w")->ptr;
        std::memcpy(&w.qkv_alpha, T.at(p + "alpha_qkv").bytes.data(), 4);
        w.qkv_act_scale = static_cast<const float*>(dev.at(p + "act_scale_qkv")->ptr);
        w.rope = rope->ptr;
        Dm[l].last = (l == L - 1);
        if (Dm[l].last) continue;
        w.o_packed = dev.at(p + "o_packed")->ptr;
        w.o_sfb = dev.at(p + "o_sfb")->ptr;
        w.awq_inv_s_gu = dev.at(p + "awq_inv_s_gu")->ptr;
        w.gu_il_packed = dev.at(p + "gu_il_packed")->ptr;
        w.gu_il_sfb = dev.at(p + "gu_il_sfb")->ptr;
        w.down_packed = dev.at(p + "down_packed")->ptr;
        w.down_sfb = dev.at(p + "down_sfb")->ptr;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    DeviceBuffer scratch(static_cast<size_t>(encoder_layer_scratch_bytes(base, base.Se)));
    EncoderLayerScratch s;
    encoder_layer_bind_scratch(base, base.Se, scratch.ptr, &s);
    const auto& x_in = T.at("x_in");
    DeviceBuffer x(x_in.bytes.size());
    DeviceBuffer K(kv_bytes * L), V(kv_bytes * L);
    if (encoder_layer_load_kernels() != 0) return 1;
    encoder_layer_set_pdl(true);

    auto run = [&]() {
        for (int l = 0; l < L; ++l) {
            const int rc = encoder_layer_forward(Dm[l], W[l], s, x.ptr,
                                                 static_cast<char*>(K.ptr) + l * kv_bytes,
                                                 static_cast<char*>(V.ptr) + l * kv_bytes, stream);
            if (rc != 0) return rc;
        }
        return 0;
    };
    cudaMemcpy(x.ptr, x_in.bytes.data(), x_in.bytes.size(), cudaMemcpyHostToDevice);
    if (int rc = run(); rc != 0) {
        std::fprintf(stderr, "forward rc=%d\n", rc);
        return 1;
    }
    cudaStreamSynchronize(stream);
    size_t dx = diff_count(download(x.ptr, x.bytes), T.at("x_out").bytes);
    size_t dkv = 0;
    auto kh = download(K.ptr, K.bytes), vh = download(V.ptr, V.bytes);
    for (int l = 0; l < L; ++l) {
        const std::string p = "L" + std::to_string(l) + ".";
        std::vector<uint8_t> kl(kh.begin() + l * kv_bytes, kh.begin() + (l + 1) * kv_bytes);
        std::vector<uint8_t> vl(vh.begin() + l * kv_bytes, vh.begin() + (l + 1) * kv_bytes);
        dkv += diff_count(kl, T.at(p + "k_out").bytes) + diff_count(vl, T.at(p + "v_out").bytes);
    }
    std::printf("encoder parity: x %zu diff | KV %zu diff\n", dx, dkv);
    std::printf("PARITY_%s\n", dx == 0 && dkv == 0 ? "BITWISE" : "MISMATCH");

    std::vector<double> ms;
    for (int i = 0; i < 10 + iters; ++i) {
        cudaMemcpy(x.ptr, x_in.bytes.data(), x_in.bytes.size(), cudaMemcpyHostToDevice);
        const auto t0 = std::chrono::steady_clock::now();
        run();
        cudaStreamSynchronize(stream);
        const auto t1 = std::chrono::steady_clock::now();
        if (i >= 10) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("encoder forward (eager): median %.3f ms p90 %.3f ms n=%zu\n", ms[ms.size() / 2],
                ms[ms.size() * 9 / 10], ms.size());
    return dx == 0 && dkv == 0 ? 0 : 3;
}
