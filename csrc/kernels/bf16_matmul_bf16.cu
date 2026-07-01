// Generic bf16 row-major matmul (small-M) — see header for design notes.
// Warp-per-output pattern: block grid (ceil(N/8), M), one block computes
// 8 output elements for one (m, n-tile). W is read once per (n-tile) block
// and reused across M rows via L2.
//
// Moved here from bf16_matmul_qwen36.cu as part of the generic-helper
// ownership cleanup (#112). The kernel algorithm is unchanged; only the file
// and the public symbol name (bf16_matmul_bf16) are neutral now.

#include "bf16_matmul_bf16.cuh"

#include <cublasLt.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;  // 256

#define FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(expr)                         \
    do {                                                                 \
        cublasStatus_t _st = (expr);                                     \
        if (_st != CUBLAS_STATUS_SUCCESS) {                              \
            throw std::runtime_error(                                    \
                std::string("cuBLASLt error ") +                        \
                std::to_string(static_cast<int>(_st)) +                  \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));   \
        }                                                                \
    } while (0)

struct Bf16LtPlan {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool autotuned = false;
};

struct Bf16LtKey {
    int M;
    int N;
    int K;

    bool operator==(const Bf16LtKey& other) const {
        return M == other.M && N == other.N && K == other.K;
    }
};

struct Bf16LtKeyHash {
    size_t operator()(const Bf16LtKey& key) const {
        size_t h = static_cast<size_t>(key.M);
        h = h * 1315423911u + static_cast<size_t>(key.N);
        h = h * 1315423911u + static_cast<size_t>(key.K);
        return h;
    }
};

static cublasLtHandle_t g_bf16_lt = nullptr;
static void* g_bf16_workspace = nullptr;
static size_t g_bf16_workspace_size = 32 * 1024 * 1024;
static std::mutex g_bf16_mu;
static std::unordered_map<Bf16LtKey, Bf16LtPlan, Bf16LtKeyHash>
    g_bf16_plans;

static int get_bf16_autotune_algos() {
    const char* env = std::getenv("FLASHRT_BF16_CUBLASLT_AUTOTUNE_ALGOS");
    if (!env || !*env) return 8;
    return std::clamp(std::atoi(env), 1, 32);
}

static bool get_bf16_autotune_verbose() {
    const char* env = std::getenv("FLASHRT_BF16_CUBLASLT_AUTOTUNE_VERBOSE");
    return env && std::atoi(env) != 0;
}

static void ensure_bf16_lt() {
    if (g_bf16_lt) return;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtCreate(&g_bf16_lt));
    cudaError_t err = cudaMalloc(&g_bf16_workspace, g_bf16_workspace_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMalloc failed for BF16 cuBLASLt workspace: ") +
            cudaGetErrorString(err));
    }
}

static Bf16LtPlan& get_bf16_lt_plan(int M, int N, int K) {
    std::lock_guard<std::mutex> lock(g_bf16_mu);
    ensure_bf16_lt();
    Bf16LtKey key{M, N, K};
    auto it = g_bf16_plans.find(key);
    if (it != g_bf16_plans.end()) return it->second;

    Bf16LtPlan plan;
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasOperation_t op_t = CUBLAS_OP_T;
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;

    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(
        cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_t, sizeof(op_t)));

    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &plan.a_desc, CUDA_R_16BF, M, K, K));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
        sizeof(row_order)));

    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &plan.b_desc, CUDA_R_16BF, N, K, K));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
        sizeof(row_order)));

    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &plan.c_desc, CUDA_R_16BF, M, N, N));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
        sizeof(row_order)));

    cublasLtMatmulPreference_t pref;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_bf16_workspace_size, sizeof(g_bf16_workspace_size)));
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        g_bf16_lt, plan.desc, plan.a_desc, plan.b_desc,
        plan.c_desc, plan.c_desc, pref, 1, &heuristic, &returned));
    cublasLtMatmulPreferenceDestroy(pref);
    if (returned == 0) {
        throw std::runtime_error("cuBLASLt: no generic BF16 GEMM algorithm");
    }
    plan.algo = heuristic.algo;

    auto [inserted, _] = g_bf16_plans.emplace(key, plan);
    return inserted->second;
}

static void autotune_bf16_lt_plan(
    Bf16LtPlan& plan,
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    if (plan.autotuned) return;
    const int num_algos = get_bf16_autotune_algos();
    if (num_algos <= 1) {
        plan.autotuned = true;
        return;
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t capture_err = cudaStreamIsCapturing(stream, &capture_status);
    if (capture_err == cudaSuccess && capture_status != cudaStreamCaptureStatusNone) {
        return;
    }

    cublasLtMatmulPreference_t pref;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_bf16_workspace_size, sizeof(g_bf16_workspace_size)));

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(num_algos);
    int returned = 0;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        g_bf16_lt, plan.desc, plan.a_desc, plan.b_desc,
        plan.c_desc, plan.c_desc, pref, num_algos, heuristics.data(),
        &returned));
    cublasLtMatmulPreferenceDestroy(pref);
    if (returned <= 1) {
        plan.autotuned = true;
        return;
    }

    cudaEvent_t start, stop;
    cudaError_t err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaEventCreate failed: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        throw std::runtime_error(std::string("cudaEventCreate failed: ") +
                                 cudaGetErrorString(err));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    constexpr int kWarmup = 1;
    constexpr int kBench = 3;
    float best_ms = 1.0e30f;
    int best_idx = -1;

    for (int i = 0; i < returned; ++i) {
        bool ok = true;
        for (int w = 0; w < kWarmup; ++w) {
            cublasStatus_t st = cublasLtMatmul(
                g_bf16_lt, plan.desc, &alpha,
                x, plan.a_desc,
                W, plan.b_desc,
                &beta,
                out, plan.c_desc,
                out, plan.c_desc,
                &heuristics[i].algo, g_bf16_workspace,
                g_bf16_workspace_size, stream);
            if (st != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            throw std::runtime_error(std::string("cudaEventRecord failed: ") +
                                     cudaGetErrorString(err));
        }
        for (int b = 0; b < kBench; ++b) {
            cublasStatus_t st = cublasLtMatmul(
                g_bf16_lt, plan.desc, &alpha,
                x, plan.a_desc,
                W, plan.b_desc,
                &beta,
                out, plan.c_desc,
                out, plan.c_desc,
                &heuristics[i].algo, g_bf16_workspace,
                g_bf16_workspace_size, stream);
            if (st != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        err = cudaEventRecord(stop, stream);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            throw std::runtime_error(std::string("cudaEventRecord failed: ") +
                                     cudaGetErrorString(err));
        }
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            throw std::runtime_error(std::string("cudaEventSynchronize failed: ") +
                                     cudaGetErrorString(err));
        }
        float ms = 0.0f;
        err = cudaEventElapsedTime(&ms, start, stop);
        if (err != cudaSuccess) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            throw std::runtime_error(std::string("cudaEventElapsedTime failed: ") +
                                     cudaGetErrorString(err));
        }
        ms /= static_cast<float>(kBench);
        if (ms < best_ms) {
            best_ms = ms;
            best_idx = i;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (best_idx >= 0) {
        plan.algo = heuristics[best_idx].algo;
    }
    plan.autotuned = true;
    if (best_idx >= 0 && get_bf16_autotune_verbose()) {
        std::cout << "  bf16_matmul_cublaslt autotune: shape=("
                  << M << "," << N << "," << K << ") tested=" << returned
                  << " best=" << best_idx << " (" << best_ms * 1000.0f
                  << " us)" << std::endl;
    }
}

// Vectorized: each thread reads 8 bf16 = 16 bytes per iter via int4.
// Block grid: (ceil(N/8), M). Each block handles one M row × 8 N elements.
// W is shared across M rows (read once per (n-tile) block; M blocks load
// the same w_row for the same n-tile, so the L2 cache absorbs reuse).
template<int K_FIXED>
__global__ void bf16_matmul_warp_kernel(
    const __nv_bfloat16* __restrict__ x,        // (M, K)
    const __nv_bfloat16* __restrict__ W,        // (N, K)
    __nv_bfloat16* __restrict__ out,            // (M, N)
    int M, int N) {
    __shared__ __nv_bfloat16 x_sh[K_FIXED];

    const int m = blockIdx.y;
    if (m >= M) return;

    // Cooperative load of x[m, :] into smem.
    const int4* x_i4 = reinterpret_cast<const int4*>(x + m * K_FIXED);
    int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);
    const int K_int4 = K_FIXED / 8;
    #pragma unroll 1
    for (int j = threadIdx.x; j < K_int4; j += kThreads) {
        x_sh_i4[j] = x_i4[j];
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;
    if (n >= N) return;

    const int4* w_row_i4 = reinterpret_cast<const int4*>(W + n * K_FIXED);

    float acc = 0.0f;
    #pragma unroll 1
    for (int i4 = lane; i4 < K_int4; i4 += 32) {
        int4 wv = w_row_i4[i4];
        int4 xv = x_sh_i4[i4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 wb = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&wv)[k]));
            __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&xv)[k]));
            float2 wf = __bfloat1622float2(wb);
            float2 xf = __bfloat1622float2(xb);
            acc = fmaf(xf.x, wf.x, acc);
            acc = fmaf(xf.y, wf.y, acc);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        out[m * N + n] = __float2bfloat16(acc);
    }
}

// Generic-K fallback (chunked smem). Same warp pattern.
__global__ void bf16_matmul_warp_kernel_generic(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
    extern __shared__ __nv_bfloat16 x_sh[];
    const int K_chunk_max = 4096;

    const int m = blockIdx.y;
    if (m >= M) return;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;

    float acc = 0.0f;

    for (int k_off = 0; k_off < K; k_off += K_chunk_max) {
        const int chunk = min(K_chunk_max, K - k_off);
        for (int j = threadIdx.x; j < chunk; j += kThreads) {
            x_sh[j] = x[m * K + k_off + j];
        }
        __syncthreads();

        if (n < N) {
            const __nv_bfloat16* w_row = W + n * K + k_off;
            #pragma unroll 1
            for (int j = lane; j < chunk; j += 32) {
                float xv = static_cast<float>(x_sh[j]);
                float wv = static_cast<float>(w_row[j]);
                acc = fmaf(xv, wv, acc);
            }
        }
        __syncthreads();
    }

    if (n >= N) return;

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        out[m * N + n] = __float2bfloat16(acc);
    }
}

}  // namespace

void bf16_matmul_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M, int N, int K,
    cudaStream_t stream) {
    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock, M);
    // Specialization set must match the bf16_matvec sibling so the
    // M=1 reference and the M=K test produce bit-identical reductions
    // (different chunking -> different fma order -> bf16 drift). matvec
    // specializes K=5120 and K=4096; everything else (incl. K=6144 for
    // lin_K out_proj) falls to the generic chunked path.
    if (K == 5120) {
        bf16_matmul_warp_kernel<5120>
            <<<grid, kThreads, 0, stream>>>(x, W, out, M, N);
    } else if (K == 4096) {
        bf16_matmul_warp_kernel<4096>
            <<<grid, kThreads, 0, stream>>>(x, W, out, M, N);
    } else {
        const int smem_bytes = 4096 * sizeof(__nv_bfloat16);
        bf16_matmul_warp_kernel_generic
            <<<grid, kThreads, smem_bytes, stream>>>(x, W, out, M, N, K);
    }
}

void bf16_matmul_cublaslt_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M, int N, int K,
    cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    Bf16LtPlan& plan = get_bf16_lt_plan(M, N, K);
    if (!plan.autotuned) {
        std::lock_guard<std::mutex> lock(g_bf16_mu);
        autotune_bf16_lt_plan(plan, x, W, out, M, N, K, stream);
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmul(
        g_bf16_lt, plan.desc, &alpha,
        x, plan.a_desc,
        W, plan.b_desc,
        &beta,
        out, plan.c_desc,
        out, plan.c_desc,
        &plan.algo, g_bf16_workspace, g_bf16_workspace_size, stream));
}

}  // namespace flash_rt::kernels
