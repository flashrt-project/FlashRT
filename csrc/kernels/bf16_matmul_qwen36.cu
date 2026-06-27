// Qwen3.6 AB96 bf16 matmul kernels (in_proj qkv/z, out_proj at K=5120,
// N=96) — see header for design notes. The generic small-M bf16 matmul and
// the cuBLASLt BF16 GEMM that used to live here moved to bf16_matmul_bf16.cu
// as part of the generic-helper ownership cleanup (#112); this file now keeps
// only the Qwen3.6-specific AB96 kernels plus a thin legacy wrapper
// (bf16_matmul_qwen36_bf16 -> bf16_matmul_bf16) so existing call sites and
// bindings keep working unchanged.

#include "bf16_matmul_qwen36.cuh"

#include "bf16_matmul_bf16.cuh"

#include <cublasLt.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;  // 256
constexpr int kAb96N = 96;
constexpr int kAb96K = 5120;

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

struct Ab96LtPlan {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

static cublasLtHandle_t g_ab96_lt = nullptr;
static void* g_ab96_workspace = nullptr;
static size_t g_ab96_workspace_size = 16 * 1024 * 1024;
static std::mutex g_ab96_mu;
static std::unordered_map<int, Ab96LtPlan> g_ab96_plans;

static void ensure_ab96_lt() {
    if (g_ab96_lt) return;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtCreate(&g_ab96_lt));
    cudaError_t err = cudaMalloc(&g_ab96_workspace, g_ab96_workspace_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMalloc failed for AB96 cuBLASLt workspace: ") +
            cudaGetErrorString(err));
    }
}

static Ab96LtPlan& get_ab96_lt_plan(int M) {
    std::lock_guard<std::mutex> lock(g_ab96_mu);
    ensure_ab96_lt();
    auto it = g_ab96_plans.find(M);
    if (it != g_ab96_plans.end()) return it->second;

    Ab96LtPlan plan;
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
        &plan.a_desc, CUDA_R_16BF, M, kAb96K, kAb96K));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
        sizeof(row_order)));

    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &plan.b_desc, CUDA_R_16BF, kAb96N, kAb96K, kAb96K));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
        sizeof(row_order)));

    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &plan.c_desc, CUDA_R_16BF, M, kAb96N, kAb96N));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
        sizeof(row_order)));

    cublasLtMatmulPreference_t pref;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_ab96_workspace_size, sizeof(g_ab96_workspace_size)));
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        g_ab96_lt, plan.desc, plan.a_desc, plan.b_desc,
        plan.c_desc, plan.c_desc, pref, 1, &heuristic, &returned));
    cublasLtMatmulPreferenceDestroy(pref);
    if (returned == 0) {
        throw std::runtime_error("cuBLASLt: no Qwen3.6 AB96 BF16 algorithm");
    }
    plan.algo = heuristic.algo;

    auto [inserted, _] = g_ab96_plans.emplace(M, plan);
    return inserted->second;
}

template<int K_FIXED>
__global__ void bf16_matmul_ab96_pair_kernel(
    const __nv_bfloat16* __restrict__ x,        // (M, K)
    const __nv_bfloat16* __restrict__ W,        // (96, K)
    __nv_bfloat16* __restrict__ out,            // (M, 96)
    int M) {
    __shared__ __nv_bfloat16 x_sh[K_FIXED];

    const int m = blockIdx.y;
    if (m >= M) return;

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
    const int n0 = blockIdx.x * (kWarpsPerBlock * 2) + warp_id * 2;
    const int n1 = n0 + 1;
    if (n0 >= 96) return;

    const int4* w0_row_i4 = reinterpret_cast<const int4*>(W + n0 * K_FIXED);
    const int4* w1_row_i4 = reinterpret_cast<const int4*>(W + n1 * K_FIXED);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    #pragma unroll 1
    for (int i4 = lane; i4 < K_int4; i4 += 32) {
        int4 xv = x_sh_i4[i4];
        int4 w0v = w0_row_i4[i4];
        int4 w1v = w1_row_i4[i4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&xv)[k]));
            __nv_bfloat162 w0b = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&w0v)[k]));
            __nv_bfloat162 w1b = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&w1v)[k]));
            float2 xf = __bfloat1622float2(xb);
            float2 w0f = __bfloat1622float2(w0b);
            float2 w1f = __bfloat1622float2(w1b);
            acc0 = fmaf(xf.x, w0f.x, acc0);
            acc0 = fmaf(xf.y, w0f.y, acc0);
            acc1 = fmaf(xf.x, w1f.x, acc1);
            acc1 = fmaf(xf.y, w1f.y, acc1);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, off);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, off);
    }
    if (lane == 0) {
        out[m * 96 + n0] = __float2bfloat16(acc0);
        out[m * 96 + n1] = __float2bfloat16(acc1);
    }
}

template<int K_FIXED, int M_TILE>
__global__ void bf16_matmul_ab96_mtile_kernel(
    const __nv_bfloat16* __restrict__ x,        // (M, K)
    const __nv_bfloat16* __restrict__ W,        // (96, K)
    __nv_bfloat16* __restrict__ out,            // (M, 96)
    int M) {
    extern __shared__ __align__(16) __nv_bfloat16 x_sh[];

    const int m0 = blockIdx.y * M_TILE;
    const int K_int4 = K_FIXED / 8;
    const int4* x_i4 = reinterpret_cast<const int4*>(x);
    int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);

    const int x_i4_total = M_TILE * K_int4;
    for (int j = threadIdx.x; j < x_i4_total; j += kThreads) {
        int mt = j / K_int4;
        int ki4 = j - mt * K_int4;
        int m = m0 + mt;
        if (m < M) {
            x_sh_i4[j] = x_i4[m * K_int4 + ki4];
        } else {
            x_sh_i4[j] = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;
    if (n >= 96) return;

    const int4* w_row_i4 = reinterpret_cast<const int4*>(W + n * K_FIXED);
    float acc[M_TILE];
#pragma unroll
    for (int mt = 0; mt < M_TILE; ++mt) acc[mt] = 0.0f;

    #pragma unroll 1
    for (int i4 = lane; i4 < K_int4; i4 += 32) {
        int4 wv = w_row_i4[i4];
        int4 xv[M_TILE];
#pragma unroll
        for (int mt = 0; mt < M_TILE; ++mt) {
            xv[mt] = x_sh_i4[mt * K_int4 + i4];
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 wb = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&wv)[k]));
            float2 wf = __bfloat1622float2(wb);
#pragma unroll
            for (int mt = 0; mt < M_TILE; ++mt) {
                __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
                    &(reinterpret_cast<int*>(&xv[mt])[k]));
                float2 xf = __bfloat1622float2(xb);
                acc[mt] = fmaf(xf.x, wf.x, acc[mt]);
                acc[mt] = fmaf(xf.y, wf.y, acc[mt]);
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
#pragma unroll
        for (int mt = 0; mt < M_TILE; ++mt) {
            acc[mt] += __shfl_xor_sync(0xffffffff, acc[mt], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int mt = 0; mt < M_TILE; ++mt) {
            int m = m0 + mt;
            if (m < M) {
                out[m * 96 + n] = __float2bfloat16(acc[mt]);
            }
        }
    }
}

template<int K_FIXED, int M_TILE>
__global__ void bf16_matmul_ab96_mtile_pair_kernel(
    const __nv_bfloat16* __restrict__ x,        // (M, K)
    const __nv_bfloat16* __restrict__ W,        // (96, K)
    __nv_bfloat16* __restrict__ out,            // (M, 96)
    int M) {
    extern __shared__ __align__(16) __nv_bfloat16 x_sh[];

    const int m0 = blockIdx.y * M_TILE;
    const int K_int4 = K_FIXED / 8;
    const int4* x_i4 = reinterpret_cast<const int4*>(x);
    int4* x_sh_i4 = reinterpret_cast<int4*>(x_sh);

    const int x_i4_total = M_TILE * K_int4;
    for (int j = threadIdx.x; j < x_i4_total; j += kThreads) {
        int mt = j / K_int4;
        int ki4 = j - mt * K_int4;
        int m = m0 + mt;
        if (m < M) {
            x_sh_i4[j] = x_i4[m * K_int4 + ki4];
        } else {
            x_sh_i4[j] = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * (kWarpsPerBlock * 2) + warp_id * 2;
    const int n1 = n0 + 1;
    if (n0 >= 96) return;

    const int4* w0_row_i4 = reinterpret_cast<const int4*>(W + n0 * K_FIXED);
    const int4* w1_row_i4 = reinterpret_cast<const int4*>(W + n1 * K_FIXED);
    float acc0[M_TILE];
    float acc1[M_TILE];
#pragma unroll
    for (int mt = 0; mt < M_TILE; ++mt) {
        acc0[mt] = 0.0f;
        acc1[mt] = 0.0f;
    }

    #pragma unroll 1
    for (int i4 = lane; i4 < K_int4; i4 += 32) {
        int4 w0v = w0_row_i4[i4];
        int4 w1v = w1_row_i4[i4];
        int4 xv[M_TILE];
#pragma unroll
        for (int mt = 0; mt < M_TILE; ++mt) {
            xv[mt] = x_sh_i4[mt * K_int4 + i4];
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 w0b = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&w0v)[k]));
            __nv_bfloat162 w1b = *reinterpret_cast<__nv_bfloat162*>(
                &(reinterpret_cast<int*>(&w1v)[k]));
            float2 w0f = __bfloat1622float2(w0b);
            float2 w1f = __bfloat1622float2(w1b);
#pragma unroll
            for (int mt = 0; mt < M_TILE; ++mt) {
                __nv_bfloat162 xb = *reinterpret_cast<__nv_bfloat162*>(
                    &(reinterpret_cast<int*>(&xv[mt])[k]));
                float2 xf = __bfloat1622float2(xb);
                acc0[mt] = fmaf(xf.x, w0f.x, acc0[mt]);
                acc0[mt] = fmaf(xf.y, w0f.y, acc0[mt]);
                acc1[mt] = fmaf(xf.x, w1f.x, acc1[mt]);
                acc1[mt] = fmaf(xf.y, w1f.y, acc1[mt]);
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
#pragma unroll
        for (int mt = 0; mt < M_TILE; ++mt) {
            acc0[mt] += __shfl_xor_sync(0xffffffff, acc0[mt], off);
            acc1[mt] += __shfl_xor_sync(0xffffffff, acc1[mt], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int mt = 0; mt < M_TILE; ++mt) {
            int m = m0 + mt;
            if (m < M) {
                out[m * 96 + n0] = __float2bfloat16(acc0[mt]);
                if (n1 < 96) {
                    out[m * 96 + n1] = __float2bfloat16(acc1[mt]);
                }
            }
        }
    }
}

}  // namespace

// Thin legacy wrapper: bf16_matmul_qwen36_bf16 is the historical name for the
// model-neutral small-M matmul, kept so Qwen3.6 call sites and the existing
// binding stay unchanged. The implementation lives in bf16_matmul_bf16.cu.
void bf16_matmul_qwen36_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M, int N, int K,
    cudaStream_t stream) {
    bf16_matmul_bf16(x, W, out, M, N, K, stream);
}

void bf16_matmul_qwen36_ab96_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_ab,
    __nv_bfloat16* out_ab,
    int M,
    cudaStream_t stream) {
    dim3 grid((96 + (kWarpsPerBlock * 2) - 1) / (kWarpsPerBlock * 2), M);
    bf16_matmul_ab96_pair_kernel<5120>
        <<<grid, kThreads, 0, stream>>>(x, W_ab, out_ab, M);
}

void bf16_matmul_qwen36_ab96_m4_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_ab,
    __nv_bfloat16* out_ab,
    int M,
    cudaStream_t stream) {
    constexpr int kMTile = 4;
    dim3 grid((96 + kWarpsPerBlock - 1) / kWarpsPerBlock,
              (M + kMTile - 1) / kMTile);
    size_t smem = kMTile * 5120 * sizeof(__nv_bfloat16);
    bf16_matmul_ab96_mtile_kernel<5120, kMTile>
        <<<grid, kThreads, smem, stream>>>(x, W_ab, out_ab, M);
}

void bf16_matmul_qwen36_ab96_m4_pair_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_ab,
    __nv_bfloat16* out_ab,
    int M,
    cudaStream_t stream) {
    constexpr int kMTile = 4;
    dim3 grid((96 + (kWarpsPerBlock * 2) - 1) / (kWarpsPerBlock * 2),
              (M + kMTile - 1) / kMTile);
    size_t smem = kMTile * 5120 * sizeof(__nv_bfloat16);
    bf16_matmul_ab96_mtile_pair_kernel<5120, kMTile>
        <<<grid, kThreads, smem, stream>>>(x, W_ab, out_ab, M);
}

void bf16_matmul_qwen36_ab96_lt_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W_ab,
    __nv_bfloat16* out_ab,
    int M,
    cudaStream_t stream) {
    if (M <= 0) return;
    Ab96LtPlan& plan = get_ab96_lt_plan(M);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    FLASHRT_BF16_MATMUL_CUBLASLT_CHECK(cublasLtMatmul(
        g_ab96_lt, plan.desc, &alpha,
        x, plan.a_desc,
        W_ab, plan.b_desc,
        &beta,
        out_ab, plan.c_desc,
        out_ab, plan.c_desc,
        &plan.algo, g_ab96_workspace, g_ab96_workspace_size, stream));
}

}  // namespace flash_rt::kernels
