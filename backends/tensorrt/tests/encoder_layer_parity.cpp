// Bitwise parity of the native pi0.5 encoder layer against the FlashRT
// Python reference dump (tools/reference/pi05_pipeline.py), plus a latency probe.
//
// usage: encoder_layer_parity <layer.safetensors> [iters]
#include "pi05_encoder_layer.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

std::map<std::string, HostTensor> load_safetensors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    std::string header(header_len, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    auto j = nlohmann::json::parse(header);
    std::map<std::string, HostTensor> out;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        HostTensor t;
        t.dtype = it.value()["dtype"].get<std::string>();
        t.shape = it.value()["shape"].get<std::vector<int64_t>>();
        auto off = it.value()["data_offsets"].get<std::vector<uint64_t>>();
        t.bytes.assign(data.begin() + off[0], data.begin() + off[1]);
        out.emplace(it.key(), std::move(t));
    }
    return out;
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    explicit DeviceBuffer(size_t n) : bytes(n) {
        if (cudaMalloc(&ptr, std::max<size_t>(n, 1)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

std::unique_ptr<DeviceBuffer> upload(const HostTensor& t) {
    auto b = std::make_unique<DeviceBuffer>(t.bytes.size());
    cudaMemcpy(b->ptr, t.bytes.data(), t.bytes.size(), cudaMemcpyHostToDevice);
    return b;
}

size_t compare(const std::vector<uint8_t>& got, const std::vector<uint8_t>& ref,
               double* max_abs_fp16) {
    size_t diff = 0;
    *max_abs_fp16 = 0.0;
    for (size_t i = 0; i + 1 < got.size(); i += 2) {
        uint16_t a, b;
        std::memcpy(&a, &got[i], 2);
        std::memcpy(&b, &ref[i], 2);
        if (a != b) {
            ++diff;
            auto h2f = [](uint16_t h) {
                const int s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
                double v = e == 0 ? std::ldexp(m, -24)
                                  : std::ldexp(1024 + m, e - 25);
                return s ? -v : v;
            };
            *max_abs_fp16 = std::max(*max_abs_fp16, std::fabs(h2f(a) - h2f(b)));
        }
    }
    return diff;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s layer.safetensors [iters]\n", argv[0]);
        return 2;
    }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    auto T = load_safetensors(argv[1]);
    const auto& meta = T.at("meta");
    std::vector<int64_t> m(meta.bytes.size() / 8);
    std::memcpy(m.data(), meta.bytes.data(), meta.bytes.size());

    using namespace flashrt_trt::pi05;
    EncoderLayerDims d;
    d.Se = static_cast<int>(m[0]);
    d.D = static_cast<int>(m[1]);
    d.H = static_cast<int>(m[2]);
    d.NH = static_cast<int>(m[3]);
    d.HD = static_cast<int>(m[4]);
    d.attn_o_variant = static_cast<int>(m[6]);
    d.down_variant = static_cast<int>(m[7]);
    std::printf("Se=%d D=%d H=%d NH=%d HD=%d o_variant=%d down_variant=%d\n",
                d.Se, d.D, d.H, d.NH, d.HD, d.attn_o_variant, d.down_variant);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    std::map<std::string, std::unique_ptr<DeviceBuffer>> dev;
    for (const char* name : {"qkv_w", "rope", "o_packed", "o_sfb", "awq_inv_s_gu",
                             "gu_il_packed", "gu_il_sfb", "down_packed",
                             "down_sfb", "act_scale_qkv"}) {
        dev[name] = upload(T.at(name));
    }
    float alpha = 0.0f;
    std::memcpy(&alpha, T.at("alpha_qkv").bytes.data(), 4);

    EncoderLayerWeights w;
    w.qkv_w = dev["qkv_w"]->ptr;
    w.qkv_alpha = alpha;
    w.qkv_act_scale = static_cast<const float*>(dev["act_scale_qkv"]->ptr);
    w.rope = dev["rope"]->ptr;
    w.o_packed = dev["o_packed"]->ptr;
    w.o_sfb = dev["o_sfb"]->ptr;
    w.awq_inv_s_gu = dev["awq_inv_s_gu"]->ptr;
    w.gu_il_packed = dev["gu_il_packed"]->ptr;
    w.gu_il_sfb = dev["gu_il_sfb"]->ptr;
    w.down_packed = dev["down_packed"]->ptr;
    w.down_sfb = dev["down_sfb"]->ptr;

    DeviceBuffer scratch(static_cast<size_t>(encoder_layer_scratch_bytes(d, d.Se)));
    EncoderLayerScratch s;
    encoder_layer_bind_scratch(d, d.Se, scratch.ptr, &s);

    const auto& x_in = T.at("x_in");
    DeviceBuffer x(x_in.bytes.size());
    DeviceBuffer k(T.at("k_out").bytes.size());
    DeviceBuffer v(T.at("v_out").bytes.size());

    if (int rc = encoder_layer_load_kernels(); rc != 0) {
        std::fprintf(stderr, "FA4 module load failed rc=%d\n", rc);
        return 1;
    }
    encoder_layer_set_pdl(true);

    auto run_once = [&]() {
        cudaMemcpy(x.ptr, x_in.bytes.data(), x_in.bytes.size(), cudaMemcpyHostToDevice);
        int rc = encoder_layer_forward(d, w, s, x.ptr, k.ptr, v.ptr, stream);
        cudaStreamSynchronize(stream);
        return rc;
    };

    if (int rc = run_once(); rc != 0) {
        std::fprintf(stderr, "forward failed rc=%d (cuda: %s)\n", rc,
                     cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    auto download = [&](const DeviceBuffer& b) {
        std::vector<uint8_t> h(b.bytes);
        cudaMemcpy(h.data(), b.ptr, b.bytes, cudaMemcpyDeviceToHost);
        return h;
    };
    double mx = 0, mk = 0, mv = 0;
    const size_t dx = compare(download(x), T.at("x_out").bytes, &mx);
    const size_t dk = compare(download(k), T.at("k_out").bytes, &mk);
    const size_t dv = compare(download(v), T.at("v_out").bytes, &mv);
    std::printf("parity: x %zu diff (max %.3g) | K %zu diff (max %.3g) | V %zu diff (max %.3g)\n",
                dx, mx, dk, mk, dv, mv);
    const bool bitwise = dx == 0 && dk == 0 && dv == 0;
    std::printf("PARITY_%s\n", bitwise ? "BITWISE" : "MISMATCH");

    // Latency: forward only (x restored outside the timed region).
    std::vector<double> ms;
    for (int i = 0; i < 20 + iters; ++i) {
        cudaMemcpy(x.ptr, x_in.bytes.data(), x_in.bytes.size(), cudaMemcpyHostToDevice);
        const auto t0 = std::chrono::steady_clock::now();
        encoder_layer_forward(d, w, s, x.ptr, k.ptr, v.ptr, stream);
        cudaStreamSynchronize(stream);
        const auto t1 = std::chrono::steady_clock::now();
        if (i >= 20) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("layer forward (eager, host launch included): median %.3f ms p90 %.3f ms n=%zu\n",
                ms[ms.size() / 2], ms[ms.size() * 9 / 10], ms.size());
    cudaStreamDestroy(stream);
    return bitwise ? 0 : 3;
}
