// Bitwise parity of the native pi0.5 SigLIP stage (patch embedding, all
// layers, post-LayerNorm projection) against the FlashRT library forward,
// plus a latency probe.
//
// usage: siglip_parity <siglip_all.safetensors> [iters]
#include "pi05_siglip.h"

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

std::map<std::string, std::vector<uint8_t>> load_safetensors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    std::string header(header_len, '\0');
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    const auto base = static_cast<std::streamoff>(8 + header_len);
    auto j = nlohmann::json::parse(header);
    std::map<std::string, std::vector<uint8_t>> out;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        auto off = it.value()["data_offsets"].get<std::vector<uint64_t>>();
        std::vector<uint8_t> b(off[1] - off[0]);
        f.seekg(base + static_cast<std::streamoff>(off[0]));
        f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
        out.emplace(it.key(), std::move(b));
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

std::vector<uint8_t> download(void* p, size_t n) {
    std::vector<uint8_t> h(n);
    cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost);
    return h;
}

size_t diff_count(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size()) return static_cast<size_t>(-1);
    size_t d = 0;
    for (size_t i = 0; i + 1 < a.size(); i += 2) d += (a[i] != b[i] || a[i + 1] != b[i + 1]);
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s siglip_all.safetensors [iters]\n", argv[0]);
        return 2;
    }
    const int iters = argc > 2 ? std::atoi(argv[2]) : 100;
    auto T = load_safetensors(argv[1]);
    std::vector<int64_t> m(T.at("meta").size() / 8);
    std::memcpy(m.data(), T.at("meta").data(), T.at("meta").size());
    std::vector<float> alpha(T.at("alpha").size() / 4);
    std::memcpy(alpha.data(), T.at("alpha").data(), T.at("alpha").size());

    using namespace flashrt_trt::pi05;
    SiglipDims d;
    const int S = static_cast<int>(m[0]);
    d.D = static_cast<int>(m[1]);
    d.NH = static_cast<int>(m[3]);
    d.HD = static_cast<int>(m[4]);
    const int L = static_cast<int>(m[5]);
    const int nv = static_cast<int>(m[6]);
    d.spv = static_cast<int>(m[7]);
    d.H_pad = static_cast<int>(m[8]);
    d.De = static_cast<int>(m[9]);
    d.up_variant = static_cast<int>(m[10]);
    std::printf("S=%d nv=%d L=%d H_pad=%d up_variant=%d down_variant=%lld\n", S, nv, L, d.H_pad,
                d.up_variant, static_cast<long long>(m[11]));
    if (m[11] != 0) {
        std::fprintf(stderr, "only down variant 0 is implemented\n");
        return 2;
    }

    std::map<std::string, std::unique_ptr<DeviceBuffer>> dev;
    auto up = [&](const std::string& name) {
        auto& t = T.at(name);
        auto b = std::make_unique<DeviceBuffer>(t.size());
        cudaMemcpy(b->ptr, t.data(), t.size(), cudaMemcpyHostToDevice);
        void* p = b->ptr;
        dev[name] = std::move(b);
        return p;
    };
    SiglipEmbedWeights ew;
    ew.lut = up("lut");
    ew.pe_w = up("pe_w");
    ew.pe_b = up("pe_b");
    ew.pos_emb = up("pos_emb");
    ew.postln_w = up("postln_w");
    ew.postln_b = up("postln_b");
    ew.proj_w = up("proj_w");
    ew.proj_b = up("proj_b");
    std::vector<SiglipLayerWeights> W(L);
    for (int l = 0; l < L; ++l) {
        const std::string p = "L" + std::to_string(l) + ".";
        auto& w = W[l];
        w.ln_attn_w = up(p + "ln_attn_w");
        w.ln_attn_b = up(p + "ln_attn_b");
        w.qkv_w = up(p + "qkv_w");
        w.qkv_b = up(p + "qkv_b");
        w.qkv_alpha = alpha[l * 4 + 0];
        w.o_w = up(p + "o_w");
        w.o_b = up(p + "o_b");
        w.o_alpha = alpha[l * 4 + 1];
        w.ln_ffn_w = up(p + "ln_ffn_w");
        w.ln_ffn_b = up(p + "ln_ffn_b");
        w.awq_inv_s = up(p + "awq_inv_s");
        w.up_packed = up(p + "up_packed");
        w.up_sfb = up(p + "up_sfb");
        w.up_b = up(p + "up_b");
        w.down_packed = up(p + "down_packed");
        w.down_sfb = up(p + "down_sfb");
        w.down_b = up(p + "down_b");
    }
    const void* images = up("images_u8");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    DeviceBuffer scratch(static_cast<size_t>(siglip_scratch_bytes(d, S)));
    SiglipScratch s;
    siglip_bind_scratch(d, S, scratch.ptr, &s);
    DeviceBuffer x(static_cast<size_t>(S) * d.D * 2), tokens(static_cast<size_t>(S) * d.De * 2);
    if (int rc = siglip_load_kernels()) {
        std::fprintf(stderr, "load rc=%d\n", rc);
        return 1;
    }
    GemmRunner* gemm = siglip_gemm_create();
    if (gemm == nullptr) return 1;

    int rc = siglip_patch_embed(d, nv, ew, s, gemm, images, true, x.ptr, stream);
    cudaStreamSynchronize(stream);
    const size_t d_embed = diff_count(download(x.ptr, x.bytes), T.at("x_embed"));
    size_t d_layers = 0;
    int first_bad = -1;
    for (int l = 0; l < L && rc == 0; ++l) {
        rc = siglip_layer_forward(d, nv, W[l], s, gemm, x.ptr, stream);
        cudaStreamSynchronize(stream);
        const size_t dl = diff_count(download(x.ptr, x.bytes), T.at("L" + std::to_string(l) + ".x_out"));
        if (dl != 0 && first_bad < 0) first_bad = l;
        d_layers += dl;
    }
    if (rc == 0) rc = siglip_post_project(d, S, ew, s, gemm, x.ptr, tokens.ptr, stream);
    cudaStreamSynchronize(stream);
    if (rc != 0) {
        std::fprintf(stderr, "forward rc=%d\n", rc);
        return 1;
    }
    const size_t d_tok = diff_count(download(tokens.ptr, tokens.bytes), T.at("tokens"));
    std::printf("siglip parity: embed %zu diff | layers %zu diff (first layer %d) | tokens %zu diff\n",
                d_embed, d_layers, first_bad, d_tok);
    const bool ok = d_embed == 0 && d_layers == 0 && d_tok == 0;
    std::printf("PARITY_%s\n", ok ? "BITWISE" : "MISMATCH");

    std::vector<double> ms;
    for (int i = 0; i < 10 + iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        siglip_patch_embed(d, nv, ew, s, gemm, images, true, x.ptr, stream);
        for (int l = 0; l < L; ++l) siglip_layer_forward(d, nv, W[l], s, gemm, x.ptr, stream);
        siglip_post_project(d, S, ew, s, gemm, x.ptr, tokens.ptr, stream);
        cudaStreamSynchronize(stream);
        const auto t1 = std::chrono::steady_clock::now();
        if (i >= 10) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("siglip stage (eager, host launch included): median %.3f ms p90 %.3f ms n=%zu\n",
                ms[ms.size() / 2], ms[ms.size() * 9 / 10], ms.size());
    siglip_gemm_destroy(gemm);
    return ok ? 0 : 3;
}
