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
    void* workspace = nullptr;
    size_t ws_bytes = 32u << 20;
    std::map<std::tuple<int, int, int, bool>, Plan> plans;
    std::mutex mu;
};

Ctx& ctx() {
    static Ctx c;
    return c;
}

}  // namespace

int gemm_f16acc(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K,
                cudaStream_t stream) {
    Ctx& c = ctx();
    std::lock_guard<std::mutex> lock(c.mu);
    if (c.handle == nullptr) {
        if (cublasLtCreate(&c.handle) != CUBLAS_STATUS_SUCCESS) return static_cast<int>(cudaErrorUnknown);
        if (cudaMalloc(&c.workspace, c.ws_bytes) != cudaSuccess) return static_cast<int>(cudaErrorMemoryAllocation);
    }
    const auto key = std::make_tuple(M, N, K, bias != nullptr);
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
            cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
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
    if (bias != nullptr)
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    const __half one = __float2half(1.0f), zero = __float2half(0.0f);
    const cublasStatus_t st = cublasLtMatmul(c.handle, p.op, &one, B, p.a, A, p.b, &zero, D, p.d, D, p.d, &p.algo,
                                             c.workspace, c.ws_bytes, stream);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : static_cast<int>(cudaErrorUnknown);
}

}  // namespace lawam
}  // namespace flash_rt
