// SPDX-License-Identifier: Apache-2.0
// FP16 GEMM with FP16 accumulation (cuBLASLt CUBLAS_COMPUTE_16F) for the
// LaWAM vision / language stacks. GeForce Blackwell runs FP16 MMA with FP16
// accumulators at twice the FP32-accumulate rate. See lawam_kernels.cuh.
#include "lawam_kernels.cuh"

#include <cublasLt.h>

#include <map>
#include <mutex>
#include <tuple>

namespace flash_rt {
namespace lawam {
namespace {

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

struct Ctx {
    cublasLtHandle_t handle = nullptr;
    size_t ws_bytes = 32u << 20;
    std::map<cudaStream_t, void*> stream_ws;    // concurrent graph branches must not share one
    std::map<std::tuple<int, int, int, int>, Plan> plans;
    std::mutex mu;
    bool autotune = false;
};

Ctx& ctx() {
    static Ctx c;
    return c;
}

// Workspace of ``stream`` (allocated on its first, uncaptured, use).
void* workspace_for(Ctx& c, cudaStream_t stream) {
    auto it = c.stream_ws.find(stream);
    if (it != c.stream_ws.end()) return it->second;
    void* w = nullptr;
    if (cudaMalloc(&w, c.ws_bytes) != cudaSuccess) return nullptr;
    c.stream_ws[stream] = w;
    return w;
}

}  // namespace

int gemm_f16acc(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, bool gelu,
                cudaStream_t stream) {
    if (gelu && bias == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    Ctx& c = ctx();
    std::lock_guard<std::mutex> lock(c.mu);
    if (c.handle == nullptr) {
        if (cublasLtCreate(&c.handle) != CUBLAS_STATUS_SUCCESS) return static_cast<int>(cudaErrorUnknown);
    }
    void* ws = workspace_for(c, stream);
    if (ws == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    const int ep_kind = gelu ? 2 : (bias != nullptr ? 1 : 0);
    const auto key = std::make_tuple(M, N, K, ep_kind);
    auto it = c.plans.find(key);
    if (it == c.plans.end()) {
        Plan p;
        // row-major D[M,N] = A[M,K] B[K,N]  <=>  column-major D^T = B^T A^T
        if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_16F, CUDA_R_16F) != CUBLAS_STATUS_SUCCESS)
            return static_cast<int>(cudaErrorUnknown);
        cublasOperation_t opn = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &opn, sizeof(opn));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn));
        if (bias != nullptr) {
            // GELU_BIAS: the tanh-approximated GELU of (A @ B + bias)
            cublasLtEpilogue_t ep = gelu ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
            cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
            cudaDataType_t bt = CUDA_R_16F;
            cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt));
        }
        cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, N, K, N);
        cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, K, M, K);
        cublasLtMatrixLayoutCreate(&p.d, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.ws_bytes,
                                             sizeof(c.ws_bytes));
        // Autotune: time the heuristic's top candidates on the caller's
        // buffers (first call happens before graph capture) and keep the fastest.
        cublasLtMatmulHeuristicResult_t cand[16];
        int n = 0;
        const cublasStatus_t st =
            cublasLtMatmulAlgoGetHeuristic(c.handle, p.op, p.a, p.b, p.d, p.d, pref, 16, cand, &n);
        cublasLtMatmulPreferenceDestroy(pref);
        if (st != CUBLAS_STATUS_SUCCESS || n == 0) return static_cast<int>(cudaErrorNotSupported);
        if (bias != nullptr)
            cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        cublasLtMatmulHeuristicResult_t res = cand[0];
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (c.autotune && n > 1 && cap == cudaStreamCaptureStatusNone) {
            const __half one = __float2half(1.0f), zero = __float2half(0.0f);
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            float best = 1e30f;
            for (int i = 0; i < n; ++i) {
                // split-K would add FP16 partial reductions on top of the FP16 accumulation
                int split = 1;
                size_t sz = 0;
                cublasLtMatmulAlgoConfigGetAttribute(&cand[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split,
                                                     sizeof(split), &sz);
                if (split > 1) continue;
                bool ok = true;
                for (int r = 0; r < 3 && ok; ++r)   // warm-up
                    ok = cublasLtMatmul(c.handle, p.op, &one, B, p.a, A, p.b, &zero, D, p.d, D, p.d, &cand[i].algo,
                                        ws, c.ws_bytes, stream) == CUBLAS_STATUS_SUCCESS;
                if (!ok) continue;
                cudaEventRecord(e0, stream);
                for (int r = 0; r < 10; ++r)
                    cublasLtMatmul(c.handle, p.op, &one, B, p.a, A, p.b, &zero, D, p.d, D, p.d, &cand[i].algo,
                                   ws, c.ws_bytes, stream);
                cudaEventRecord(e1, stream);
                cudaEventSynchronize(e1);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) {
                    best = ms;
                    res = cand[i];
                }
            }
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
        p.algo = res.algo;
        it = c.plans.emplace(key, p).first;
    }
    Plan& p = it->second;
    if (bias != nullptr)
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    const __half one = __float2half(1.0f), zero = __float2half(0.0f);
    const cublasStatus_t st = cublasLtMatmul(c.handle, p.op, &one, B, p.a, A, p.b, &zero, D, p.d, D, p.d, &p.algo,
                                             ws, c.ws_bytes, stream);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(cudaErrorUnknown);
}

int gemm_f16acc_split(int M, int N, int K, bool bias, bool gelu) {
    Ctx& c = ctx();
    std::lock_guard<std::mutex> lock(c.mu);
    if (c.handle == nullptr) {
        if (cublasLtCreate(&c.handle) != CUBLAS_STATUS_SUCCESS) return -1;
    }
    // the descriptor gemm_f16acc builds for this shape, first heuristic candidate
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_16F, CUDA_R_16F) != CUBLAS_STATUS_SUCCESS) return -1;
    cublasOperation_t opn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opn, sizeof(opn));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn));
    if (bias) {
        cublasLtEpilogue_t ep = gelu ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
        cudaDataType_t bt = CUDA_R_16F;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt));
    }
    cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&d, CUDA_R_16F, N, M, N);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.ws_bytes,
                                         sizeof(c.ws_bytes));
    cublasLtMatmulHeuristicResult_t res{};
    int n = 0;
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(c.handle, op, a, b, d, d, pref, 1, &res, &n);
    int split = -1;
    if (st == CUBLAS_STATUS_SUCCESS && n > 0) {
        split = 1;
        size_t sz = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&res.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split), &sz);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(d);
    cublasLtMatmulDescDestroy(op);
    return split;
}

void gemm_f16acc_set_autotune(bool enable) {
    Ctx& c = ctx();
    std::lock_guard<std::mutex> lock(c.mu);
    c.autotune = enable;
}

int gemm_bf16_f32out(const __nv_bfloat16* A, const __nv_bfloat16* W, float* D, int M, int N, int K,
                     cudaStream_t stream) {
    Ctx& c = ctx();
    std::lock_guard<std::mutex> lock(c.mu);
    if (c.handle == nullptr) {
        if (cublasLtCreate(&c.handle) != CUBLAS_STATUS_SUCCESS) return static_cast<int>(cudaErrorUnknown);
    }
    void* ws = workspace_for(c, stream);
    if (ws == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    const auto key = std::make_tuple(M, N, K, 16);
    auto it = c.plans.find(key);
    if (it == c.plans.end()) {
        Plan p;
        // row-major D[M,N] = A[M,K] W[N,K]^T  <=>  column-major D^T = W A^T  (W as stored: op T)
        if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
            return static_cast<int>(cudaErrorUnknown);
        cublasOperation_t opt = CUBLAS_OP_T, opn = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &opt, sizeof(opt));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn));
        cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K);
        cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, K);
        cublasLtMatrixLayoutCreate(&p.d, CUDA_R_32F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &c.ws_bytes,
                                             sizeof(c.ws_bytes));
        cublasLtMatmulHeuristicResult_t res{};
        int n = 0;
        const cublasStatus_t st =
            cublasLtMatmulAlgoGetHeuristic(c.handle, p.op, p.a, p.b, p.d, p.d, pref, 1, &res, &n);
        cublasLtMatmulPreferenceDestroy(pref);
        if (st != CUBLAS_STATUS_SUCCESS || n == 0) return static_cast<int>(cudaErrorNotSupported);
        p.algo = res.algo;
        it = c.plans.emplace(key, p).first;
    }
    Plan& p = it->second;
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t st = cublasLtMatmul(c.handle, p.op, &one, W, p.a, A, p.b, &zero, D, p.d, D, p.d, &p.algo,
                                             ws, c.ws_bytes, stream);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(cudaErrorUnknown);
}

}  // namespace lawam
}  // namespace flash_rt
