#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <functional>

// Check cuBLAS status
#define CUBLAS_CHECK(expr)                                              \
    do {                                                                \
        cublasStatus_t status = (expr);                                 \
        if (status != CUBLAS_STATUS_SUCCESS) {                          \
            throw std::runtime_error(                                   \
                std::string("cuBLAS error at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " code=" +                   \
                std::to_string(static_cast<int>(status)));              \
        }                                                               \
    } while (0)

#define CUDA_CHECK(expr)                                                \
    do {                                                                \
        cudaError_t err = (expr);                                       \
        if (err != cudaSuccess) {                                       \
            throw std::runtime_error(                                   \
                std::string("CUDA error at ") + __FILE__ + ":" +        \
                std::to_string(__LINE__) + ": " +                       \
                cudaGetErrorString(err));                                \
        }                                                               \
    } while (0)

class GemmRunner {
public:
    GemmRunner();
    ~GemmRunner();

    // BF16 GEMM: D = alpha * A @ B^T + beta * C
    // A: (M, K) row-major bf16
    // B: (N, K) row-major bf16 (transposed as col-major in GEMM)
    // D: (M, N) row-major bf16
    float bf16_gemm(void* A, void* B, void* D,
                    int M, int N, int K,
                    float alpha, float beta,
                    int warmup, int iters);

    // FP8 GEMM: D = scale_a * scale_b * (A_fp8 @ B_fp8^T)
    // A: (M, K) row-major fp8_e4m3
    // B: (N, K) row-major fp8_e4m3
    // D: (M, N) row-major bf16
    float fp8_gemm(void* A, void* B, void* D,
                   int M, int N, int K,
                   float scale_a, float scale_b,
                   int warmup, int iters);

#ifdef ENABLE_NVFP4
    // NVFP4 Block-Scaled GEMM: D = A_fp4 @ B_fp4^T (with per-16-element UE4M3 scales)
    float fp4_gemm(void* A, void* SFA, void* B, void* SFB, void* D,
                   int M, int N, int K,
                   int warmup, int iters);
#endif

    // ── Inference (no timing, no sync, stream-based) ──

    // BF16: D = A(M,K) @ B(N,K)^T  (row-major, B transposed)
    void bf16_run(void* A, void* B, void* D,
                  int M, int N, int K,
                  cudaStream_t stream = 0);

    // BF16: D = A(M,K) @ B(K,N)  (row-major, no transpose)
    void bf16_nn(void* A, void* B, void* D,
                 int M, int N, int K,
                 cudaStream_t stream = 0);

    // FP16: D = A(M,K) @ B(K,N)  (row-major, no transpose)
    void fp16_nn(void* A, void* B, void* D,
                 int M, int N, int K,
                 cudaStream_t stream = 0);

    // BF16 with residual: D += A(M,K) @ B(K,N)  (row-major, no transpose)
    // Fuses residual add into GEMM accumulator (FP32), avoiding BF16 round-trip
    void bf16_nn_res(void* A, void* B, void* D,
                     int M, int N, int K,
                     cudaStream_t stream = 0);

    // BF16 + BIAS epilogue: D = A(M,K) @ B(K,N) + bias(N)
    void bf16_nn_bias(void* A, void* B, void* D, void* bias,
                       int M, int N, int K,
                       cudaStream_t stream = 0);

    // BF16 + BIAS + GELU epilogue: D = GELU(A(M,K) @ B(K,N) + bias(N))
    void bf16_nn_bias_gelu(void* A, void* B, void* D, void* bias,
                            int M, int N, int K,
                            cudaStream_t stream = 0);

    // BF16 + BIAS + residual: D += A(M,K) @ B(K,N) + bias(N)
    void bf16_nn_bias_res(void* A, void* B, void* D, void* bias,
                           int M, int N, int K,
                           cudaStream_t stream = 0);

    // FP8: D_bf16 = scale_a * scale_b * A_fp8(M,K) @ B_fp8(N,K)^T
    void fp8_run(void* A, void* B, void* D,
                 int M, int N, int K,
                 float scale_a, float scale_b,
                 cudaStream_t stream = 0);

    // FP8 with device scale pointers (CUDA Graph compatible)
    // d_scale_a, d_scale_b: device float* (already on GPU)
    void fp8_run_dev(void* A, void* B, void* D,
                     int M, int N, int K,
                     float* d_scale_a, float* d_scale_b,
                     cudaStream_t stream = 0);

    // FP8 no-transpose: D_bf16 = A_fp8(M,K) @ B_fp8(K,N) with device scale pointers
    // Matches bf16_nn layout — B stored as (K,N), no transpose
    void fp8_nn_dev(void* A, void* B, void* D,
                    int M, int N, int K,
                    float* d_scale_a, float* d_scale_b,
                    cudaStream_t stream = 0);

    // FP8 with host alpha + BIAS epilogue: D = alpha * A_fp8 @ B_fp8 + bias
    // Matches pi05 gmm_fp8_kn_bias: host scalar alpha, per-GEMM bias vector
    void fp8_nn_bias(void* A, void* B, void* D, void* bias,
                     int M, int N, int K, float alpha,
                     cudaStream_t stream = 0);

    // G6.7: FP8 GEMM with host alpha + BIAS epilogue, BF16 output
    // Same as fp8_nn_bias but with BF16 output and BF16 bias dtype.
    // Targets Motus Wan FP8 sites (which run bf16 throughout; the fp16
    // variant above forces an extra cast). Throws if cuBLASLt sm_120
    // returns no algorithm for the requested shape (caller must fall
    // back to fp8_nn_dev + add_bias_bf16).
    void fp8_nn_bias_bf16(void* A, void* B, void* D, void* bias,
                           int M, int N, int K, float alpha,
                           cudaStream_t stream = 0);

    // FP8 with host alpha + BIAS + residual: D += alpha * A_fp8 @ B_fp8 + bias
    // Matches pi05 gmm_fp8_kn_bias_res: beta=1 for residual accumulate
    void fp8_nn_bias_res(void* A, void* B, void* D, void* bias,
                         int M, int N, int K, float alpha,
                         cudaStream_t stream = 0);

    // FP8 with host alpha + GELU + BIAS epilogue: D = GELU(alpha * A_fp8 @ B_fp8 + bias)
    // Matches pi05 gmm_fp8_kn_gelu_bias: FP16 output with GELU activation
    void fp8_nn_gelu_bias(void* A, void* B, void* D, void* bias,
                          int M, int N, int K, float alpha,
                          cudaStream_t stream = 0);

    // FP8 with device descale → FP16 output (matches pi05 gmm_fp8_kn_descale)
    // Uses shared handle_ for consistent algo selection
    void fp8_descale_fp16(void* A, void* B, void* D,
                           int M, int N, int K,
                           float* act_descale, float* w_descale,
                           cudaStream_t stream = 0);

#ifdef ENABLE_NVFP4
    // NVFP4 block-scaled: D_bf16 = A_fp4(M,K) @ B_fp4(K,N) with per-16-block UE4M3 scales
    void fp4_nn_dev(void* A_fp4, void* SFA, void* B_fp4, void* SFB,
                    void* D, int M, int N, int K,
                    cudaStream_t stream = 0);
#endif

    // ── Autotune: benchmark top-N algorithms and cache the best ──
    // Call before CUDA Graph capture. Uses dummy data at the provided pointers.
    void autotune_bf16_nn(void* A, void* B, void* D,
                          int M, int N, int K, int num_algos = 16);
    void autotune_fp8_nn_dev(void* A, void* B, void* D,
                             int M, int N, int K,
                             float* d_scale_a, float* d_scale_b,
                             int num_algos = 16);
#ifdef ENABLE_NVFP4
    void autotune_fp4_nn_dev(void* A_fp4, void* SFA, void* B_fp4, void* SFB,
                              void* D, int M, int N, int K,
                              int num_algos = 16);
#endif

private:
    cublasLtHandle_t handle_;
    void* workspace_;
    size_t workspace_size_;
    // Pre-allocated device scale storage for fp8_run
    float* d_scale_a_;
    float* d_scale_b_;

    // ── GEMM descriptor + algorithm cache ──
    enum GemmType { BF16_NN = 0, BF16_NN_RES = 1, FP8_NN_DEV = 2, FP16_NN = 4
#ifdef ENABLE_NVFP4
        , FP4_NN_DEV = 3
#endif
    };

    struct GemmKey {
        int type, M, N, K;
        bool operator==(const GemmKey& o) const {
            return type == o.type && M == o.M && N == o.N && K == o.K;
        }
    };

    struct GemmKeyHash {
        size_t operator()(const GemmKey& k) const {
            size_t h = std::hash<int>()(k.type);
            h ^= std::hash<int>()(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct CachedGemm {
        cublasLtMatmulDesc_t matmul_desc;
        cublasLtMatrixLayout_t A_desc, B_desc, C_desc, D_desc;
        cublasLtMatmulAlgo_t algo;
        bool has_C_desc = false;  // FP4 needs separate C descriptor (BF16 != FP4)
    };

    std::unordered_map<GemmKey, CachedGemm, GemmKeyHash> gemm_cache_;

    // Setup descriptors for a given GEMM type and shape, store in cache
    CachedGemm& get_or_create_cached(GemmType type, int M, int N, int K);
    // Autotune helper: benchmark algorithms and pick the best
    void autotune_cached(CachedGemm& entry, void* A, void* B, void* D,
                         float alpha, float beta, int num_algos,
                         float* d_scale_a = nullptr, float* d_scale_b = nullptr);
};
