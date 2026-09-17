// Decoder attention kernel choice: FlashRT cuBLAS decomposition
// (attention_qkv_fp16) vs FA4 (fa4_hd256_q1_fwd) at the pi0.5 decoder shape.
// Reports per-call wall (host launch + device, synchronized) and device-only
// time of a 180-call burst (18 layers x 10 steps), plus output agreement.
#include "fa4_attention.h"

#include "kernels/attention_cublas.cuh"
#include "fused_fp4/pdl.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using h16 = __half;

static void* dev(size_t n) { void* p; cudaMalloc(&p, n); return p; }

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    const int S = argc > 1 ? atoi(argv[1]) : 10;
    const int Skv = argc > 2 ? atoi(argv[2]) : 986;
    const int NH = 8, HD = 256, BURST = 180, REPS = 30;
    cudaSetDevice(0);
    flash_rt::fp4::pdl_flag() = true;  // as the decoder plugin runs
    cudaStream_t st; cudaStreamCreate(&st);
    cublasHandle_t cb; cublasCreate(&cb); cublasSetStream(cb, st);
    if (int rc = flashrt_trt::fa4_load()) { printf("fa4 load %d\n", rc); return 1; }

    std::mt19937 g(1); std::normal_distribution<float> nd(0, 1);
    auto host_rand = [&](size_t n) { std::vector<uint16_t> v(n); for (auto& x : v) { float f = nd(g);
        int16_t e = 15; uint32_t bits = 0; (void)e; // encode via float->half
        uint32_t fb; memcpy(&fb, &f, 4); uint32_t s = (fb >> 16) & 0x8000; int32_t ex = ((fb >> 23) & 0xff) - 127 + 15;
        uint32_t m = (fb >> 13) & 0x3ff; bits = ex <= 0 ? s : (ex >= 31 ? (s | 0x7c00) : (s | (ex << 10) | m)); x = bits; } return v; };
    const size_t qn = (size_t)S * NH * HD, kn = (size_t)Skv * HD;
    auto hq = host_rand(qn), hk = host_rand(kn), hv = host_rand(kn);
    void* q = dev(qn * 2), *k = dev(kn * 2), *v = dev(kn * 2);
    void* o_cb = dev(qn * 2), *o_fa = dev(qn * 2);
    void* logits = dev((size_t)S * NH * (Skv + 1) * 2 + 64);
    cudaMemcpy(q, hq.data(), qn * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(k, hk.data(), kn * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(v, hv.data(), kn * 2, cudaMemcpyHostToDevice);
    const float scale = 1.0f / std::sqrt((float)HD);

    auto run_cb = [&]() { attention_qkv_fp16(cb, (const h16*)q, (const h16*)k, (const h16*)v, (h16*)logits,
                                             (h16*)o_cb, S, Skv, NH, HD, scale, st); return 0; };
    auto run_fa = [&]() { return flashrt_trt::fa4_hd256_gqa(q, k, v, o_fa, S, Skv, NH, scale, st); };

    for (int i = 0; i < 50; ++i) { run_cb(); if (run_fa()) { printf("fa4 fail\n"); return 1; } }
    cudaStreamSynchronize(st);

    // agreement
    std::vector<uint16_t> a(qn), b(qn);
    cudaMemcpy(a.data(), o_cb, qn * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), o_fa, qn * 2, cudaMemcpyDeviceToHost);
    auto h2f = [](uint16_t h) { int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
        float f = e == 0 ? std::ldexp((float)m, -24) : std::ldexp((float)(m + 1024), e - 25); return s ? -f : f; };
    double dot = 0, na = 0, nb = 0, mx = 0; size_t neq = 0;
    for (size_t i = 0; i < qn; ++i) { double x = h2f(a[i]), y = h2f(b[i]); dot += x * y; na += x * x; nb += y * y;
        mx = std::max(mx, std::fabs(x - y)); neq += a[i] != b[i]; }
    printf("S=%d Skv=%d: cos=%.8f max|d|=%.3g differing=%zu/%zu\n", S, Skv, dot / std::sqrt(na * nb), mx, neq, qn);

    for (int which = 0; which < 2; ++which) {
        std::vector<double> call_ms, burst_wall, burst_dev;
        for (int r = 0; r < REPS; ++r) {
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            auto t0 = std::chrono::steady_clock::now();
            cudaEventRecord(e0, st);
            for (int i = 0; i < BURST; ++i) which ? run_fa() : run_cb();
            cudaEventRecord(e1, st);
            cudaStreamSynchronize(st);
            auto t1 = std::chrono::steady_clock::now();
            float ev; cudaEventElapsedTime(&ev, e0, e1);
            burst_wall.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            burst_dev.push_back(ev);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        for (int i = 0; i < 500; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            which ? run_fa() : run_cb();
            cudaStreamSynchronize(st);
            call_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("%-6s per call (sync) median %.3f ms | %d-call burst wall %.2f ms, device %.2f ms\n",
               which ? "FA4" : "cuBLAS", median(call_ms), BURST, median(burst_wall), median(burst_dev));
    }
    return 0;
}
