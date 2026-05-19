#include "gemm_runner.h"
#include <iostream>
#include <vector>

// ================================================================
// GemmRunner: cuBLASLt-based GEMM for FP8/NVFP4 on Blackwell
// Direct C++ calls, zero Python overhead
// ================================================================

GemmRunner::GemmRunner() {
    CUBLAS_CHECK(cublasLtCreate(&handle_));
    workspace_size_ = 256 * 1024 * 1024;  // 256 MB workspace (enables split-K etc.)
    CUDA_CHECK(cudaMalloc(&workspace_, workspace_size_));
    // Pre-allocate device scale storage for fp8_run
    CUDA_CHECK(cudaMalloc(&d_scale_a_, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_b_, sizeof(float)));
}

GemmRunner::~GemmRunner() {
    // Destroy cached descriptors
    for (auto& [key, entry] : gemm_cache_) {
        cublasLtMatrixLayoutDestroy(entry.A_desc);
        cublasLtMatrixLayoutDestroy(entry.B_desc);
        if (entry.has_C_desc) cublasLtMatrixLayoutDestroy(entry.C_desc);
        cublasLtMatrixLayoutDestroy(entry.D_desc);
        cublasLtMatmulDescDestroy(entry.matmul_desc);
    }
    gemm_cache_.clear();
    if (d_scale_b_) cudaFree(d_scale_b_);
    if (d_scale_a_) cudaFree(d_scale_a_);
    if (workspace_) cudaFree(workspace_);
    if (handle_) cublasLtDestroy(handle_);
}

// ================================================================
// Descriptor cache: create once, reuse across all calls
// ================================================================
GemmRunner::CachedGemm& GemmRunner::get_or_create_cached(GemmType type, int M, int N, int K) {
    GemmKey key{static_cast<int>(type), M, N, K};
    auto it = gemm_cache_.find(key);
    if (it != gemm_cache_.end()) return it->second;

    CachedGemm entry;
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;

#ifdef ENABLE_NVFP4
    if (type == FP4_NN_DEV) {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t op_T = CUBLAS_OP_T;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_T, sizeof(op_T)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

        int32_t block_scale = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &block_scale, sizeof(block_scale)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &block_scale, sizeof(block_scale)));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_4F_E2M1, K, N, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_4F_E2M1, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.C_desc, CUDA_R_16BF, N, M, N));
        entry.has_C_desc = true;
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16BF, N, M, N));
    } else
#endif
    if (type == FP8_NN_DEV) {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_8F_E4M3, M, K, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_8F_E4M3, K, N, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16BF, M, N, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    } else if (type == FP16_NN) {
        // FP16 no-transpose: D(M,N) = A(M,K) @ B(K,N)
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_16F, M, K, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_16F, K, N, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16F, M, N, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    } else {
        // BF16_NN or BF16_NN_RES
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_16BF, M, K, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_16BF, K, N, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16BF, M, N, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(entry.D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    }

    // Get heuristic top-1 algorithm as default
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    auto C_layout = entry.has_C_desc ? entry.C_desc : entry.D_desc;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc,
        entry.A_desc, entry.B_desc, C_layout, entry.D_desc,
        preference, 1, &heuristic, &returned_results));
    cublasLtMatmulPreferenceDestroy(preference);

    if (returned_results == 0) {
        throw std::runtime_error("cuBLASLt: no algorithm found for cached GEMM");
    }
    entry.algo = heuristic.algo;

    auto [inserted_it, _] = gemm_cache_.emplace(key, entry);
    return inserted_it->second;
}

// ================================================================
// Autotune: benchmark top-N algorithms and pick the fastest
// ================================================================
void GemmRunner::autotune_cached(CachedGemm& entry, void* A, void* B, void* D,
                                  float alpha, float beta, int num_algos,
                                  float* d_scale_a, float* d_scale_b) {
    // Update scale pointers if FP8
    if (d_scale_a) {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));
    }

    auto C_layout = entry.has_C_desc ? entry.C_desc : entry.D_desc;

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(num_algos);
    int returned_results = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc,
        entry.A_desc, entry.B_desc, C_layout, entry.D_desc,
        preference, num_algos, heuristics.data(), &returned_results));
    cublasLtMatmulPreferenceDestroy(preference);

    if (returned_results == 0) {
        std::cerr << "  autotune: no algorithms found, keeping default" << std::endl;
        return;
    }

    // Benchmark each algorithm
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float best_ms = 1e9f;
    int best_idx = 0;
    const int warmup_iters = 3;
    const int bench_iters = 10;

    for (int i = 0; i < returned_results; ++i) {
        // Warmup
        bool algo_ok = true;
        for (int w = 0; w < warmup_iters; ++w) {
            cublasStatus_t st = cublasLtMatmul(handle_, entry.matmul_desc,
                &alpha, A, entry.A_desc, B, entry.B_desc,
                &beta, D, C_layout, D, entry.D_desc,
                &heuristics[i].algo, workspace_, workspace_size_, 0);
            if (st != CUBLAS_STATUS_SUCCESS) { algo_ok = false; break; }
        }
        if (!algo_ok) continue;
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        CUDA_CHECK(cudaEventRecord(start));
        for (int b = 0; b < bench_iters; ++b) {
            cublasLtMatmul(handle_, entry.matmul_desc,
                &alpha, A, entry.A_desc, B, entry.B_desc,
                &beta, D, C_layout, D, entry.D_desc,
                &heuristics[i].algo, workspace_, workspace_size_, 0);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= bench_iters;

        if (ms < best_ms) {
            best_ms = ms;
            best_idx = i;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    entry.algo = heuristics[best_idx].algo;
    std::cout << "  autotune: tested " << returned_results << " algos, best="
              << best_idx << " (" << best_ms * 1000.0f << " us)" << std::endl;
}

void GemmRunner::autotune_bf16_nn(void* A, void* B, void* D,
                                   int M, int N, int K, int num_algos) {
    auto& entry = get_or_create_cached(BF16_NN, M, N, K);
    autotune_cached(entry, A, B, D, 1.0f, 0.0f, num_algos);
}

void GemmRunner::autotune_fp8_nn_dev(void* A, void* B, void* D,
                                      int M, int N, int K,
                                      float* d_scale_a, float* d_scale_b,
                                      int num_algos) {
    auto& entry = get_or_create_cached(FP8_NN_DEV, M, N, K);
    autotune_cached(entry, A, B, D, 1.0f, 0.0f, num_algos, d_scale_a, d_scale_b);
}

// ================================================================
//  BF16 GEMM via cuBLASLt
//  Computes D = alpha * A @ B^T + beta * D
//  A: (M,K) row-major, B: (N,K) row-major → B^T is (K,N)
//
//  cuBLASLt uses column-major convention:
//    Row-major A(M,K) = Col-major A^T(K,M)
//    Row-major B(N,K) = Col-major B^T(K,N)
//    Goal: D = A @ B^T
//    In col-major: D^T = B @ A^T
//    So: cublasLt(B_col, A_col) where B_col = B^T(K,N), A_col = A^T(K,M)
//    op_B = CUBLAS_OP_T on B_row → gives B^T = col(K,N)  -> NO
//    Actually simpler: treat row-major as transposed col-major
// ================================================================
float GemmRunner::bf16_gemm(void* A, void* B, void* D,
                            int M, int N, int K,
                            float alpha, float beta,
                            int warmup, int iters) {
    // cuBLASLt is column-major. For row-major:
    //   D_row(M,N) = A_row(M,K) @ B_row(N,K)^T
    // Equivalent in col-major:
    //   D_col(N,M) = B_col(N,K) @ A_col(K,M)
    // where B_col(N,K) is B_row(N,K) interpreted as col-major with ld=K
    // and A_col(K,M) is A_row(M,K) interpreted as col-major with ld=K
    // But that gives D in (N,M) col-major = (M,N) row-major. Perfect.

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    // Create matmul descriptor
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // For col-major convention: D(N,M) = B(N,K) @ A(K,M)
    // B_row(N,K) in row-major = B_col with transB=T, or direct col with ld=K
    // A_row(M,K) in row-major = A_col with transA=T
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasOperation_t op_T = CUBLAS_OP_T;

    // D(N,M)_col = B(N,K) @ A^T(K,M)
    // B: row-major (N,K) → col-major: need to transpose. Set transa=T with shape (K,N)
    // Actually let's use the standard approach:
    //   In col-major: C(M,N) = op(A)(M,K) @ op(B)(K,N)
    //   With row-major data, flip M↔N:
    //   cublasLt: C_col(N,M) = B_col(N,K) * A_col(K,M)
    //   B_row(N,K) → col-major(K,N) with transa=CUBLAS_OP_T → op(B_col) = B^T(N,K)
    //   Hmm, this is getting confusing. Let me use the simple recipe:

    // Simple recipe for row-major: swap A and B, use CUBLAS_OP_T for what was row-major
    // D_row(M,N) = A_row(M,K) @ B_row(N,K)^T
    // cuBLASLt col-major: C(N,M) = B(N,K) @ A^T(K,M)
    //   matA_for_cublas = B_row, shape in col-major desc: (N, K), order=row → desc(K, N, ld=K) with op=T? No...

    // Simplest: use cublasLt with row-major layout descriptors
    // cuBLASLt supports CUBLASLT_ORDER_ROW since CUDA 11+

    // A: (M, K) row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, M, K, K));  // rows=M, cols=K, ld=K
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    // B: (N, K) row-major, need B^T(K, N) in the GEMM
    // So describe B as (K, N) col-major? Or (N, K) row-major with transB?
    // With row-major: B(N, K) → transpose gives (K, N)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16BF, N, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    // D: (M, N) row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16BF, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    // Set transB = T (we want A @ B^T)
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_T, sizeof(op_T)));

    // Find best algo
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        throw std::runtime_error("cuBLASLt: no suitable algorithm found for BF16 GEMM");
    }

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
            &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
            &heuristic.algo, workspace_, workspace_size_, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
            &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
            &heuristic.algo, workspace_, workspace_size_, 0));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);

    return ms / iters;
}

// ================================================================
//  FP8 GEMM via cuBLASLt
//  D_bf16 = (scale_a * scale_b) * A_fp8 @ B_fp8^T
//  A: (M,K) row-major fp8_e4m3
//  B: (N,K) row-major fp8_e4m3
//  D: (M,N) row-major bf16
// ================================================================
float GemmRunner::fp8_gemm(void* A, void* B, void* D,
                           int M, int N, int K,
                           float scale_a, float scale_b,
                           int warmup, int iters) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    // FP8 compute in FP32
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasOperation_t op_T = CUBLAS_OP_T;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_T, sizeof(op_T)));

    // A: (M, K) fp8_e4m3 row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8F_E4M3, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    // B: (N, K) fp8_e4m3 row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_8F_E4M3, N, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    // D: (M, N) bf16 row-major
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16BF, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    // Set scale pointers (device memory)
    float* d_scale_a;
    float* d_scale_b;
    CUDA_CHECK(cudaMalloc(&d_scale_a, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_b, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scale_a, &scale_a, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_b, &scale_b, sizeof(float), cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));

    // Find best algo
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        // Cleanup before throwing
        cudaFree(d_scale_a);
        cudaFree(d_scale_b);
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt: no suitable algorithm found for FP8 GEMM");
    }

    // alpha/beta for the outer scaling (scale_a * scale_b is handled by cuBLASLt internally)
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
            &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
            &heuristic.algo, workspace_, workspace_size_, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
            &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
            &heuristic.algo, workspace_, workspace_size_, 0));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_scale_a);
    cudaFree(d_scale_b);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);

    return ms / iters;
}

#ifdef ENABLE_NVFP4
// ================================================================
//  NVFP4 Block-Scaled GEMM via cuBLASLt
//  Matching NVIDIA's LtNvfp4Matmul example exactly.
//
//  Col-major convention. Computes: D(m,n) = op(A)(m,k) @ op(B)(k,n)
//  with transa=OP_T, transb=OP_N.
//
//  For row-major weight-only quantization:
//    Weights are pre-converted to col-major FP4 format.
//    Activations are dynamically quantized to col-major FP4.
//
//  All inputs/outputs are col-major FP4 packed data.
//  A: col-major (K, M) FP4 with transa=OP_T → gives A^T = (M, K)
//  B: col-major (K, N) FP4 with transb=OP_N → gives (K, N)
//  C: col-major (M, N) BF16
//  D: col-major (M, N) FP4
//
//  SFA: scale factors for A (one UE4M3 per 16 elements in K-dim)
//  SFB: scale factors for B (one UE4M3 per 16 elements in K-dim)
// ================================================================
float GemmRunner::fp4_gemm(void* A, void* SFA, void* B, void* SFB, void* D,
                           int M, int N, int K,
                           int warmup, int iters) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, C_desc, D_desc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Match NVIDIA example exactly: transa=OP_T, transb=OP_N
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasOperation_t op_T = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_T, sizeof(op_T)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Block-scaled mode: VEC16_UE4M3 for all (matching NVIDIA example)
    int32_t block_scale = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &block_scale, sizeof(block_scale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &block_scale, sizeof(block_scale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &block_scale, sizeof(block_scale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &block_scale, sizeof(block_scale)));

    // Scale pointers (device memory)
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SFA, sizeof(SFA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SFB, sizeof(SFB)));

    // D scale factors (output quantization scales)
    // For benchmarking, allocate dummy scales
    int d_scale_elems = (M * N + 15) / 16;  // one UE4M3 per 16 output elements
    void* d_scale_ptr;
    void* d_out_scale_ptr;
    CUDA_CHECK(cudaMalloc(&d_scale_ptr, d_scale_elems));
    CUDA_CHECK(cudaMemset(d_scale_ptr, 0x3C, d_scale_elems));
    CUDA_CHECK(cudaMalloc(&d_out_scale_ptr, d_scale_elems));
    CUDA_CHECK(cudaMemset(d_out_scale_ptr, 0x3C, d_scale_elems));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_ptr, sizeof(d_scale_ptr)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_out_scale_ptr, sizeof(d_out_scale_ptr)));

    // Matrix layouts (col-major, default cuBLASLt convention)
    // A: stored as col-major (K, M), ld = K
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_4F_E2M1, K, M, K));
    // B: stored as col-major (K, N), ld = K
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_4F_E2M1, K, N, K));
    // C: col-major (M, N), BF16, ld = M
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_16BF, M, N, M));
    // D: col-major (M, N), FP4, ld = M
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_4F_E2M1, M, N, M));

    // Find best algo
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, C_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        cudaFree(d_scale_ptr);
        cudaFree(d_out_scale_ptr);
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(C_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt: no suitable algorithm found for FP4 block-scaled GEMM");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate C (BF16 accumulator) and D_fp4 output
    void* C_buf;
    CUDA_CHECK(cudaMalloc(&C_buf, (size_t)M * N * 2));
    CUDA_CHECK(cudaMemset(C_buf, 0, (size_t)M * N * 2));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
            &alpha, A, A_desc, B, B_desc, &beta, C_buf, C_desc, D, D_desc,
            &heuristic.algo, workspace_, workspace_size_, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
            &alpha, A, A_desc, B, B_desc, &beta, C_buf, C_desc, D, D_desc,
            &heuristic.algo, workspace_, workspace_size_, 0));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(C_buf);
    cudaFree(d_scale_ptr);
    cudaFree(d_out_scale_ptr);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(C_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);

    return ms / iters;
}
#endif  // ENABLE_NVFP4

// ================================================================
//  BF16 inference run: D = A(M,K) @ B(N,K)^T  (no timing, no sync)
// ================================================================
void GemmRunner::bf16_run(void* A, void* B, void* D,
                          int M, int N, int K, cudaStream_t stream) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasOperation_t op_T = CUBLAS_OP_T;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_T, sizeof(op_T)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16BF, N, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16BF, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt bf16_run: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
        &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  BF16 no-transpose: D(M,N) = A(M,K) @ B(K,N)  (row-major)
//  B is NOT transposed. For Pi05 weights stored as (K, N).
// ================================================================
void GemmRunner::bf16_nn(void* A, void* B, void* D,
                         int M, int N, int K, cudaStream_t stream) {
    auto& entry = get_or_create_cached(BF16_NN, M, N, K);
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, entry.matmul_desc,
        &alpha, A, entry.A_desc, B, entry.B_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  FP16 no-transpose: D(M,N) = A(M,K) @ B(K,N)
//  Same as bf16_nn but with FP16 (CUDA_R_16F) data types.
// ================================================================
void GemmRunner::fp16_nn(void* A, void* B, void* D,
                         int M, int N, int K, cudaStream_t stream) {
    auto& entry = get_or_create_cached(FP16_NN, M, N, K);
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, entry.matmul_desc,
        &alpha, A, entry.A_desc, B, entry.B_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  BF16 no-transpose with fused residual: D(M,N) += A(M,K) @ B(K,N)
//  Uses beta=1 to accumulate matmul result into D in FP32, avoiding
//  a BF16 round-trip that separate matmul+residual_add would cause.
// ================================================================
void GemmRunner::bf16_nn_res(void* A, void* B, void* D,
                              int M, int N, int K, cudaStream_t stream) {
    auto& entry = get_or_create_cached(BF16_NN_RES, M, N, K);
    float alpha = 1.0f, beta = 1.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, entry.matmul_desc,
        &alpha, A, entry.A_desc, B, entry.B_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  BF16 no-transpose with BIAS epilogue: D(M,N) = A(M,K) @ B(K,N) + bias(N)
//  Uses cuBLASLt EPILOGUE_BIAS to fuse bias addition into GEMM.
// ================================================================
void GemmRunner::bf16_nn_bias(void* A, void* B, void* D, void* bias,
                               int M, int N, int K, cudaStream_t stream) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Set BIAS epilogue
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    cudaDataType_t bias_type = CUDA_R_16BF;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16BF, K, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16BF, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt bf16_nn_bias: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
        &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  BF16 GEMM + BIAS + GELU epilogue: D(M,N) = GELU(A(M,K) @ B(K,N) + bias(N))
//  Fuses GEMM + bias + GELU into single cuBLASLt launch.
// ================================================================
void GemmRunner::bf16_nn_bias_gelu(void* A, void* B, void* D, void* bias,
                                    int M, int N, int K, cudaStream_t stream) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Set GELU_BIAS epilogue
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    cudaDataType_t bias_type = CUDA_R_16BF;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16BF, K, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16BF, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt bf16_nn_bias_gelu: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
        &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  BF16 GEMM + BIAS + residual epilogue: D(M,N) = A(M,K) @ B(K,N) + bias(N) + D
//  Combines bias and residual accumulation in one GEMM launch.
// ================================================================
void GemmRunner::bf16_nn_bias_res(void* A, void* B, void* D, void* bias,
                                   int M, int N, int K, cudaStream_t stream) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Set BIAS epilogue with beta=1 for residual
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    cudaDataType_t bias_type = CUDA_R_16BF;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16BF, K, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16BF, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt bf16_nn_bias_res: no algorithm found");
    }

    float alpha = 1.0f, beta = 1.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
        &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  FP8 inference run: D_bf16 = scale_a * scale_b * A_fp8 @ B_fp8^T
// ================================================================
void GemmRunner::fp8_run(void* A, void* B, void* D,
                         int M, int N, int K,
                         float scale_a, float scale_b,
                         cudaStream_t stream) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasOperation_t op_T = CUBLAS_OP_T;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_T, sizeof(op_T)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8F_E4M3, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_8F_E4M3, N, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16BF, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    CUDA_CHECK(cudaMemcpyAsync(d_scale_a_, &scale_a, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_scale_b_, &scale_b, sizeof(float), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a_, sizeof(d_scale_a_)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b_, sizeof(d_scale_b_)));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt fp8_run: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
        &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  FP8 inference with device scale pointers (CUDA Graph compatible)
//  D_bf16 = scale_a * scale_b * A_fp8(M,K) @ B_fp8(N,K)^T
//  d_scale_a, d_scale_b are device float* (no host-device memcpy)
// ================================================================
void GemmRunner::fp8_run_dev(void* A, void* B, void* D,
                              int M, int N, int K,
                              float* d_scale_a, float* d_scale_b,
                              cudaStream_t stream) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasOperation_t op_T = CUBLAS_OP_T;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_T, sizeof(op_T)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8F_E4M3, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_8F_E4M3, N, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16BF, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    // Device scale pointers — no memcpy needed, CUDA Graph compatible
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt fp8_run_dev: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, matmul_desc,
        &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// FP8 no-transpose: D_bf16 = A_fp8(M,K) @ B_fp8(K,N) with device scale pointers
// Matches bf16_nn layout — B stored as (K,N) row-major, no transpose
void GemmRunner::fp8_nn_dev(void* A, void* B, void* D,
                             int M, int N, int K,
                             float* d_scale_a, float* d_scale_b,
                             cudaStream_t stream) {
    auto& entry = get_or_create_cached(FP8_NN_DEV, M, N, K);
    // Update scale pointers per-call (different layers have different scales)
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, entry.matmul_desc,
        &alpha, A, entry.A_desc, B, entry.B_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  FP8 GEMM + BIAS epilogue: D = alpha * A_fp8(M,K) @ B_fp8(K,N) + bias(N)
//  Cached descriptor (pi05 pattern): same shape → same algo. Bias ptr set per-call.
// ================================================================
void GemmRunner::fp8_nn_bias(void* A, void* B, void* D, void* bias,
                              int M, int N, int K, float alpha,
                              cudaStream_t stream) {
    // Cache key: use N+2000000 to distinguish from other FP8 GEMM variants (same as pi05)
    GemmKey key{100, M, N + 2000000, K};
    auto it = gemm_cache_.find(key);
    if (it == gemm_cache_.end()) {
        CachedGemm entry;
        cublasOperation_t opN = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_8F_E4M3, N, K, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_8F_E4M3, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16F, N, M, N));
        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc, entry.A_desc, entry.B_desc, entry.D_desc, entry.D_desc, pref, 1, &result, &ret));
        entry.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        gemm_cache_[key] = entry;
        it = gemm_cache_.find(key);
    }
    auto& e = it->second;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(e.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, e.matmul_desc, &alpha, B, e.A_desc, A, e.B_desc,
        &beta, D, e.D_desc, D, e.D_desc, &e.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  G6.7: FP8 GEMM + BIAS epilogue, BF16 output (and BF16 bias dtype).
//  Same logic as fp8_nn_bias above but D and bias are __nv_bfloat16.
//  cuBLASLt sm_120 historically returned NOT_SUPPORTED with FP8+bias
//  on Wan FFN shapes under CUDA 12; CUDA 13 may behave differently.
//  cublasLtMatmulAlgoGetHeuristic returns 0 algos → propagated as
//  cuBLAS error so caller can fall back.
// ================================================================
void GemmRunner::fp8_nn_bias_bf16(void* A, void* B, void* D, void* bias,
                                   int M, int N, int K, float alpha,
                                   cudaStream_t stream) {
    GemmKey key{103, M, N + 5000000, K};
    auto it = gemm_cache_.find(key);
    if (it == gemm_cache_.end()) {
        CachedGemm entry;
        cublasOperation_t opN = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        cudaDataType_t btype = CUDA_R_16BF;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_8F_E4M3, N, K, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_8F_E4M3, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16BF, N, M, N));
        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc, entry.A_desc, entry.B_desc, entry.D_desc, entry.D_desc, pref, 1, &result, &ret));
        if (ret == 0) {
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatmulDescDestroy(entry.matmul_desc);
            cublasLtMatrixLayoutDestroy(entry.A_desc);
            cublasLtMatrixLayoutDestroy(entry.B_desc);
            cublasLtMatrixLayoutDestroy(entry.D_desc);
            throw std::runtime_error(
                "fp8_nn_bias_bf16: cuBLASLt returned 0 algorithms (NOT_SUPPORTED on this shape)");
        }
        entry.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        gemm_cache_[key] = entry;
        it = gemm_cache_.find(key);
    }
    auto& e = it->second;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(e.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, e.matmul_desc, &alpha, B, e.A_desc, A, e.B_desc,
        &beta, D, e.D_desc, D, e.D_desc, &e.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  FP8 GEMM + BIAS + residual: D += alpha * A_fp8(M,K) @ B_fp8(K,N) + bias(N)
//  Cached descriptor. beta=1.
// ================================================================
void GemmRunner::fp8_nn_bias_res(void* A, void* B, void* D, void* bias,
                                  int M, int N, int K, float alpha,
                                  cudaStream_t stream) {
    GemmKey key{101, M, N + 4000000, K};
    auto it = gemm_cache_.find(key);
    if (it == gemm_cache_.end()) {
        CachedGemm entry;
        cublasOperation_t opN = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_8F_E4M3, N, K, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_8F_E4M3, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16F, N, M, N));
        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc, entry.A_desc, entry.B_desc, entry.D_desc, entry.D_desc, pref, 1, &result, &ret));
        entry.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        gemm_cache_[key] = entry;
        it = gemm_cache_.find(key);
    }
    auto& e = it->second;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(e.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    float beta = 1.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, e.matmul_desc, &alpha, B, e.A_desc, A, e.B_desc,
        &beta, D, e.D_desc, D, e.D_desc, &e.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  FP8 GEMM + GELU + BIAS epilogue: D = GELU(alpha * A_fp8(M,K) @ B_fp8(K,N) + bias(N))
//  Matches pi05 gmm_fp8_kn_gelu_bias. Output is FP16.
// ================================================================
void GemmRunner::fp8_nn_gelu_bias(void* A, void* B, void* D, void* bias,
                                   int M, int N, int K, float alpha,
                                   cudaStream_t stream) {
    GemmKey key{102, M, N + 3000000, K};
    auto it = gemm_cache_.find(key);
    if (it == gemm_cache_.end()) {
        CachedGemm entry;
        cublasOperation_t opN = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        cudaDataType_t btype = CUDA_R_16F;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_8F_E4M3, N, K, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_8F_E4M3, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16F, N, M, N));
        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc, entry.A_desc, entry.B_desc, entry.D_desc, entry.D_desc, pref, 1, &result, &ret));
        entry.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        gemm_cache_[key] = entry;
        it = gemm_cache_.find(key);
    }
    auto& e = it->second;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(e.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, e.matmul_desc, &alpha, B, e.A_desc, A, e.B_desc,
        &beta, D, e.D_desc, D, e.D_desc, &e.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  FP8 GEMM with device descale → FP16 output
//  Uses GemmRunner's handle_ for consistent algo. Col-major layout matching pi05.
// ================================================================
void GemmRunner::fp8_descale_fp16(void* A, void* B, void* D,
                                    int M, int N, int K,
                                    float* act_descale, float* w_descale,
                                    cudaStream_t stream) {
    GemmKey key{103, M, N + 6000000, K};
    auto it = gemm_cache_.find(key);
    if (it == gemm_cache_.end()) {
        CachedGemm entry;
        cublasOperation_t opN = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_8F_E4M3, N, K, N));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_8F_E4M3, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16F, N, M, N));
        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc, entry.A_desc, entry.B_desc, entry.D_desc, entry.D_desc, pref, 1, &result, &ret));
        entry.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        gemm_cache_[key] = entry;
        it = gemm_cache_.find(key);
    }
    auto& e = it->second;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(e.matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(e.matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale)));
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, e.matmul_desc, &alpha, B, e.A_desc, A, e.B_desc,
        &beta, D, e.D_desc, D, e.D_desc, &e.algo, workspace_, workspace_size_, stream));
}

#ifdef ENABLE_NVFP4
// ================================================================
//  NVFP4 block-scaled inference: D_bf16(M,N) = Act_fp4(M,K) × W_fp4(K,N)
// ================================================================
void GemmRunner::fp4_nn_dev(void* A_fp4, void* SFA, void* B_fp4, void* SFB,
                             void* D, int M, int N, int K,
                             cudaStream_t stream) {
    auto& entry = get_or_create_cached(FP4_NN_DEV, M, N, K);
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SFA, sizeof(SFA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SFB, sizeof(SFB)));

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle_, entry.matmul_desc,
        &alpha, A_fp4, entry.A_desc, B_fp4, entry.B_desc,
        &beta, D, entry.C_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

void GemmRunner::autotune_fp4_nn_dev(void* A_fp4, void* SFA, void* B_fp4, void* SFB,
                                       void* D, int M, int N, int K,
                                       int num_algos) {
    auto& entry = get_or_create_cached(FP4_NN_DEV, M, N, K);
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SFA, sizeof(SFA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(entry.matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SFB, sizeof(SFB)));
    autotune_cached(entry, A_fp4, B_fp4, D, 1.0f, 0.0f, num_algos);
}
#endif  // ENABLE_NVFP4
