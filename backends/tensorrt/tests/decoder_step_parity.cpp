// Bitwise parity and latency of the native pi0.5 decoder step against the
// FlashRT reference dumped by tools/reference/dump_decoder.py (10 denoise steps).
#include "pi05_decoder_step.h"
#include "fused_fp4/pdl.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace flashrt_trt::pi05;

struct HostTensor { std::vector<int64_t> shape; std::vector<char> bytes; };

static std::map<std::string, HostTensor> load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n = 0; f.read(reinterpret_cast<char*>(&n), 8);
    std::string h(n, '\0'); f.read(&h[0], n);
    auto j = nlohmann::json::parse(h);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::map<std::string, HostTensor> out;
    for (auto& [name, v] : j.items()) {
        if (name == "__metadata__") continue;
        HostTensor t; t.shape = v["shape"].get<std::vector<int64_t>>();
        size_t a = v["data_offsets"][0], b = v["data_offsets"][1];
        t.bytes.assign(blob.begin() + a, blob.begin() + b);
        out[name] = std::move(t);
    }
    return out;
}

static void* up(const HostTensor& t) {
    void* p = nullptr; cudaMalloc(&p, std::max<size_t>(t.bytes.size(), 1));
    cudaMemcpy(p, t.bytes.data(), t.bytes.size(), cudaMemcpyHostToDevice); return p;
}

static size_t diff(void* dev, const char* ref, size_t n) {
    std::vector<char> h(n); cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost);
    size_t c = 0; for (size_t i = 0; i < n; ++i) c += h[i] != ref[i]; return c;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s decoder_steps.safetensors [iters]\n", argv[0]); return 2; }
    int iters = argc > 2 ? std::atoi(argv[2]) : 50;
    auto T = load(argv[1]);
    const int64_t* m = reinterpret_cast<const int64_t*>(T["meta"].bytes.data());
    DecoderDims d;
    d.S = m[0]; d.D = m[1]; d.H = m[2]; d.NH = m[3]; d.HD = m[4]; d.L = m[5];
    const int steps = m[6], enc_seq = m[7]; d.total_keys = m[8];
    d.v_qkv = m[9]; d.v_o = m[10]; d.v_gu = m[11]; d.v_down = m[12];
    std::memcpy(&d.dt, T["dt"].bytes.data(), 4);
    std::printf("S=%d D=%d H=%d L=%d steps=%d enc_seq=%d total_keys=%d variants=%d/%d/%d/%d dt=%g\n", d.S, d.D,
                d.H, d.L, steps, enc_seq, d.total_keys, d.v_qkv, d.v_o, d.v_gu, d.v_down, d.dt);

    DecoderWeights w;
    w.ain_w = up(T["ain_w"]); w.ain_b = up(T["ain_b"]); w.aow = up(T["aow"]); w.aob = up(T["aob"]);
    w.rope = up(T["rope"]);
    void* sa = up(T["sa"]); void* sf = up(T["sf"]); void* fs = up(T["fs"]);
    w.qw_fp4 = up(T["qw_fp4"]); w.qw_sfb = up(T["qw_sfb"]); w.ow_fp4 = up(T["ow_fp4"]); w.ow_sfb = up(T["ow_sfb"]);
    w.gwil_fp4 = up(T["gwil_fp4"]); w.gwil_sfb = up(T["gwil_sfb"]); w.dw_fp4 = up(T["dw_fp4"]); w.dw_sfb = up(T["dw_sfb"]);

    const size_t row = static_cast<size_t>(d.HD) * 2, layer_bytes = row * d.total_keys;
    const size_t kv_bytes = layer_bytes * d.L;
    std::vector<char> kv_k0(kv_bytes, 0), kv_v0(kv_bytes, 0);
    for (int l = 0; l < d.L; ++l) {
        auto& pk = T["L" + std::to_string(l) + ".k_prefix"]; auto& pv = T["L" + std::to_string(l) + ".v_prefix"];
        std::memcpy(kv_k0.data() + layer_bytes * l, pk.bytes.data(), pk.bytes.size());
        std::memcpy(kv_v0.data() + layer_bytes * l, pv.bytes.data(), pv.bytes.size());
    }
    void* kv_k = nullptr, *kv_v = nullptr; cudaMalloc(&kv_k, kv_bytes); cudaMalloc(&kv_v, kv_bytes);
    void* noise = up(T["noise_in"]);

    void* ws = nullptr; const int64_t ws_bytes = decoder_scratch_bytes(d, d.total_keys);
    cudaMalloc(&ws, ws_bytes); cudaMemset(ws, 0, ws_bytes);
    DecoderScratch s; decoder_bind_scratch(d, d.total_keys, ws, &s);
    cublasHandle_t cublas; cublasCreate(&cublas);
    cudaStream_t stream = nullptr; cudaStreamCreate(&stream);
    flash_rt::fp4::pdl_flag() = true;

    const int64_t style_step = static_cast<int64_t>(d.L) * d.S * 3 * d.D * 2;
    const int64_t fs_step = static_cast<int64_t>(d.S) * 3 * d.D * 2;
    auto run_all = [&](bool check) {
        cudaMemcpy(kv_k, kv_k0.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(kv_v, kv_v0.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(noise, T["noise_in"].bytes.data(), T["noise_in"].bytes.size(), cudaMemcpyHostToDevice);
        bool ok = true;
        for (int st = 0; st < steps; ++st) {
            w.sa = static_cast<char*>(sa) + style_step * st;
            w.sf = static_cast<char*>(sf) + style_step * st;
            w.fs = static_cast<char*>(fs) + fs_step * st;
            int rc = decoder_step_forward(d, w, s, cublas, noise, kv_k, kv_v, stream);
            if (rc != 0) { std::printf("step %d failed rc=%d\n", st, rc); return false; }
            if (check) {
                cudaStreamSynchronize(stream);
                auto& ref = T["step" + std::to_string(st) + ".noise_out"];
                size_t c = diff(noise, ref.bytes.data(), ref.bytes.size());
                std::printf("step %d noise: %zu bytes differ\n", st, c);
                ok &= c == 0;
            }
        }
        cudaStreamSynchronize(stream);
        if (check) {
            size_t ck = 0, cv = 0;
            for (int l = 0; l < d.L; ++l) {
                auto& rk = T["L" + std::to_string(l) + ".k_suffix"]; auto& rv = T["L" + std::to_string(l) + ".v_suffix"];
                ck += diff(static_cast<char*>(kv_k) + layer_bytes * l + row * enc_seq, rk.bytes.data(), rk.bytes.size());
                cv += diff(static_cast<char*>(kv_v) + layer_bytes * l + row * enc_seq, rv.bytes.data(), rv.bytes.size());
            }
            size_t cn = diff(noise, T["noise_out"].bytes.data(), T["noise_out"].bytes.size());
            std::printf("final: actions %zu bytes differ | suffix K %zu | suffix V %zu\n", cn, ck, cv);
            ok &= cn == 0 && ck == 0 && cv == 0;
        }
        return ok;
    };
    bool ok = run_all(true);
    std::vector<double> ms;
    for (int i = 0; i < iters; ++i) {
        cudaMemcpy(kv_k, kv_k0.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, stream);
        for (int st = 0; st < steps; ++st) {
            w.sa = static_cast<char*>(sa) + style_step * st; w.sf = static_cast<char*>(sf) + style_step * st;
            w.fs = static_cast<char*>(fs) + fs_step * st;
            decoder_step_forward(d, w, s, cublas, noise, kv_k, kv_v, stream);
        }
        cudaEventRecord(e1, stream); cudaStreamSynchronize(stream);
        float e = 0; cudaEventElapsedTime(&e, e0, e1); ms.push_back(e);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("10-step decoder (eager, host launch included): median %.3f ms p90 %.3f ms n=%zu\n",
                ms[ms.size() / 2], ms[ms.size() * 9 / 10], ms.size());
    std::printf("%s\n", ok ? "DECODER_PARITY_BITWISE" : "DECODER_PARITY_FAIL");
    return ok ? 0 : 1;
}
