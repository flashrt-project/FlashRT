#include "hipblaslt_runner.h"
#include <iostream>
#include <vector>

// ================================================================
// GemmRunner (AMD): hipBLASLt-based GEMM for BF16/FP8 on CDNA
// Direct C++ calls, zero Python overhead.
//
// Port of csrc/gemm/gemm_runner.cu (cuBLASLt). Math semantics are
// identical; the one structural difference is the layout encoding:
//
//   cuBLASLt side: descriptors carry CUBLASLT_ORDER_ROW, so the
//   row-major matrices are described directly and operands are
//   passed in FlashRT order (A first, B second).
//
//   hipBLASLt side: matmul supports column-major layouts only, so
//   every entry point here uses the standard col-major swap, which
//   is bit-for-bit the same computation:
//       D_row(M,N) = A_row(M,K) @ B_row(K,N)
//   ==  D_col(N,M) = B_col(N,K) @ A_col(K,M)
//   where X_col is the same memory as X_row reinterpreted col-major.
//   Consequently hipBLASLt operand 0 ("A" in the library API) is the
//   FlashRT B matrix and operand 1 ("B") is the FlashRT A matrix.
//   The CUDA file itself already uses this exact trick for the pi05
//   FP8 epilogue paths (gemm_runner.cu:1139-1141 create A_desc from
//   the B matrix and call cublasLtMatmul(..., B, e.A_desc, A,
//   e.B_desc, ...) at gemm_runner.cu:1155).
//
// FP8 descale semantics (mirrors gemm_runner.cu:1089-1092):
//   cuBLASLt applies *(A_SCALE_POINTER) to operand A and
//   *(B_SCALE_POINTER) to operand B as per-tensor FP32 multipliers on
//   the FP32 accumulator — NOT via alpha and NOT in the epilogue.
//   hipBLASLt's HIPBLASLT_MATMUL_DESC_A/B_SCALE_POINTER attributes
//   have the same contract. Because of the operand swap above, the
//   FlashRT d_scale_a (activations) is attached to hipBLASLt operand
//   "B" and d_scale_b (weights) to operand "A". The product
//   scale_a * scale_b * (A@B) is unchanged.
//
// FP8 dtype: OCP e4m3 (HIP_R_8F_E4M3) — gfx950 native. Never the
// fnuz variant. BF16 = HIP_R_16BF. Compute type = HIPBLAS_COMPUTE_32F
// with FP32 scale type, matching CUBLAS_COMPUTE_32F / CUDA_R_32F.
// ================================================================

GemmRunner::GemmRunner() {
    HIPBLASLT_CHECK(hipblasLtCreate(&handle_));
    workspace_size_ = 256 * 1024 * 1024;  // 256 MB workspace (enables split-K etc.)
    HIP_CHECK(hipMalloc(&workspace_, workspace_size_));
    // NOTE: the CUDA ctor also pre-allocates d_scale_a_/d_scale_b_ device
    // scalars for fp8_run (host-scale variant). fp8_run is not in the
    // ported subset, so that scratch is omitted here.
}

GemmRunner::~GemmRunner() {
    // Destroy cached descriptors
    for (auto& [key, entry] : gemm_cache_) {
        hipblasLtMatrixLayoutDestroy(entry.op0_desc);
        hipblasLtMatrixLayoutDestroy(entry.op1_desc);
        hipblasLtMatrixLayoutDestroy(entry.D_desc);
        hipblasLtMatmulDescDestroy(entry.matmul_desc);
    }
    gemm_cache_.clear();
    if (workspace_) hipFree(workspace_);
    if (handle_) hipblasLtDestroy(handle_);
}

// ================================================================
// Descriptor cache: create once, reuse across all calls
//
// Col-major swap layouts (op0 = FlashRT B, op1 = FlashRT A):
//   NN  D_row(M,N) = A_row(M,K) @ B_row(K,N)
//       op0 = B_col(N,K) ld=N, TRANSA=N
//       op1 = A_col(K,M) ld=K, TRANSB=N
//       D   = D_col(N,M) ld=N
//   NT  D_row(M,N) = A_row(M,K) @ B_row(N,K)^T
//       op0 = B_col(K,N) ld=K, TRANSA=T  → op(A)=(N,K)
//       op1 = A_col(K,M) ld=K, TRANSB=N  → op(B)=(K,M)
//       D   = D_col(N,M) ld=N
// ================================================================
GemmRunner::CachedGemm& GemmRunner::get_or_create_cached(GemmType type, int M, int N, int K) {
    GemmKey key{static_cast<int>(type), M, N, K};
    auto it = gemm_cache_.find(key);
    if (it != gemm_cache_.end()) return it->second;

    CachedGemm entry;
    hipblasOperation_t op_N = HIPBLAS_OP_N;
    hipblasOperation_t op_T = HIPBLAS_OP_T;

    if (type == FP8_NN_DEV) {
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, HIP_R_8F_E4M3, N, K, N));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, HIP_R_8F_E4M3, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16BF, N, M, N));
    } else if (type == FP8_NT_DEV) {
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_T, sizeof(op_T)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, HIP_R_8F_E4M3, K, N, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, HIP_R_8F_E4M3, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16BF, N, M, N));
    } else if (type == MXFP4_NT_DEV) {
        // MXFP4 a4w4: same NT col-major swap as FP8_NT_DEV, which lands on
        // library-TN (TRANSA=T, TRANSB=N) — the canonical (and, per the
        // hipBLASLt MX samples, only exercised) layout for fp4 GEMM.
        // Element type HIP_R_4F_E2M1_EXT (hipblaslt-types.h): rows/cols/ld
        // stay in ELEMENTS; the 2-per-byte packing along the innermost
        // (stride-1) dimension is implied by the datatype.
        // K % 32 == 0 enforced by the public entry points.
        hipDataType fp4_type = static_cast<hipDataType>(HIP_R_4F_E2M1_EXT);
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_T, sizeof(op_T)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        // Block-scale mode MUST be on the descriptor before the heuristic
        // query so hipBLASLt only proposes MX-capable solutions. VEC32_UE8M0
        // = one uint8 e8m0 scale per 32-element block in the innermost
        // dimension of the corresponding data tensor (hipblaslt.h:172).
        // Both operands' innermost dim here is K (see layout note below),
        // matching the per-1x32-along-K quantization.
        hipblasLtMatmulMatrixScale_t scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, fp4_type, K, N, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, fp4_type, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16BF, N, M, N));
    } else if (type == FP16_NN) {
        // FP16 in/out, same NN col-major swap as BF16_NN.
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, HIP_R_16F, N, K, N));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, HIP_R_16F, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16F, N, M, N));
    } else if (type == FP8_NN_DEV_FP16) {
        // FP8 inputs with device descale, FP16 output (fp8_descale_fp16).
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, HIP_R_8F_E4M3, N, K, N));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, HIP_R_8F_E4M3, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16F, N, M, N));
    } else {
        // BF16_NN or BF16_NN_RES (identical descriptors — only the
        // runtime beta differs; separate cache slots keep the tuned
        // algos independent)
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&entry.matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op0_desc, HIP_R_16BF, N, K, N));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.op1_desc, HIP_R_16BF, K, M, K));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&entry.D_desc, HIP_R_16BF, N, M, N));
    }

    // Get heuristic top-1 algorithm as default
    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc,
        entry.op0_desc, entry.op1_desc, entry.D_desc, entry.D_desc,
        preference, 1, &heuristic, &returned_results));
    hipblasLtMatmulPreferenceDestroy(preference);

    if (returned_results == 0) {
        throw std::runtime_error("hipBLASLt: no algorithm found for cached GEMM");
    }
    entry.algo = heuristic.algo;

    auto [inserted_it, _] = gemm_cache_.emplace(key, entry);
    return inserted_it->second;
}

// ================================================================
// Autotune: benchmark top-N algorithms and pick the fastest
// (mirrors gemm_runner.cu:148-227; op0/op1 are in hipBLASLt operand
// order, i.e. already swapped by the public wrappers)
// ================================================================
void GemmRunner::autotune_cached(CachedGemm& entry, void* op0, void* op1, void* D,
                                  float alpha, float beta, int num_algos,
                                  float* scale_op0, float* scale_op1) {
    // Update scale pointers if FP8
    if (scale_op0) {
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
            HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_op0, sizeof(scale_op0)));
        HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
            HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_op1, sizeof(scale_op1)));
    }

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics(num_algos);
    int returned_results = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, entry.matmul_desc,
        entry.op0_desc, entry.op1_desc, entry.D_desc, entry.D_desc,
        preference, num_algos, heuristics.data(), &returned_results));
    hipblasLtMatmulPreferenceDestroy(preference);

    if (returned_results == 0) {
        std::cerr << "  autotune: no algorithms found, keeping default" << std::endl;
        return;
    }

    // Benchmark each algorithm
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    float best_ms = 1e9f;
    int best_idx = 0;
    const int warmup_iters = 3;
    const int bench_iters = 10;

    for (int i = 0; i < returned_results; ++i) {
        // Warmup
        bool algo_ok = true;
        for (int w = 0; w < warmup_iters; ++w) {
            hipblasStatus_t st = hipblasLtMatmul(handle_, entry.matmul_desc,
                &alpha, op0, entry.op0_desc, op1, entry.op1_desc,
                &beta, D, entry.D_desc, D, entry.D_desc,
                &heuristics[i].algo, workspace_, workspace_size_, 0);
            if (st != HIPBLAS_STATUS_SUCCESS) { algo_ok = false; break; }
        }
        if (!algo_ok) continue;
        HIP_CHECK(hipDeviceSynchronize());

        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        for (int b = 0; b < bench_iters; ++b) {
            hipblasLtMatmul(handle_, entry.matmul_desc,
                &alpha, op0, entry.op0_desc, op1, entry.op1_desc,
                &beta, D, entry.D_desc, D, entry.D_desc,
                &heuristics[i].algo, workspace_, workspace_size_, 0);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        ms /= bench_iters;

        if (ms < best_ms) {
            best_ms = ms;
            best_idx = i;
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    entry.algo = heuristics[best_idx].algo;
    entry.tuned = true;
    std::cout << "  autotune: tested " << returned_results << " algos, best="
              << best_idx << " (" << best_ms * 1000.0f << " us)" << std::endl;
}

void GemmRunner::autotune_bf16_nn(void* A, void* B, void* D,
                                   int M, int N, int K, int num_algos) {
    auto& entry = get_or_create_cached(BF16_NN, M, N, K);
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos);
}

void GemmRunner::autotune_fp8_nn_dev(void* A, void* B, void* D,
                                      int M, int N, int K,
                                      float* d_scale_a, float* d_scale_b,
                                      int num_algos) {
    auto& entry = get_or_create_cached(FP8_NN_DEV, M, N, K);
    // Operand swap: op0 = B (weights, scale_b), op1 = A (acts, scale_a)
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos, d_scale_b, d_scale_a);
}

void GemmRunner::autotune_fp8_nt_dev(void* A, void* B, void* D,
                                      int M, int N, int K,
                                      float* d_scale_a, float* d_scale_b,
                                      int num_algos) {
    auto& entry = get_or_create_cached(FP8_NT_DEV, M, N, K);
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos, d_scale_b, d_scale_a);
}

void GemmRunner::autotune_mxfp4_nt_dev(void* A, void* A_scales,
                                        void* B, void* B_scales, void* D,
                                        int M, int N, int K,
                                        int num_algos) {
    if (K % 32 != 0) {
        throw std::runtime_error("mxfp4_nt_dev: K must be a multiple of 32 (MX block size)");
    }
    auto& entry = get_or_create_cached(MXFP4_NT_DEV, M, N, K);
    // Operand swap: library A = FlashRT B (gets B_scales), library B =
    // FlashRT A (gets A_scales). autotune_cached only forwards the scale
    // pointers into the A/B_SCALE_POINTER attributes (opaque void* writes),
    // so the uint8 ue8m0 tensors ride through its float* parameters
    // unchanged — the SCALE_MODE already on the descriptor tells hipBLASLt
    // how to interpret them.
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos,
                    reinterpret_cast<float*>(B_scales),
                    reinterpret_cast<float*>(A_scales));
}

// ================================================================
//  BF16 inference run: D = A(M,K) @ B(N,K)^T  (no timing, no sync)
//  Mirrors gemm_runner.cu:670-720 (per-call descriptors, heuristic
//  top-1, alpha=1 beta=0).
// ================================================================
void GemmRunner::bf16_run(void* A, void* B, void* D,
                          int M, int N, int K, hipStream_t stream) {
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t op_N = HIPBLAS_OP_N;
    hipblasOperation_t op_T = HIPBLAS_OP_T;

    // Col-major swap, NT: op0 = B_col(K,N) with OP_T, op1 = A_col(K,M) with OP_N
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_T, sizeof(op_T)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op0_desc, HIP_R_16BF, K, N, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op1_desc, HIP_R_16BF, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        op0_desc, op1_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(op0_desc);
        hipblasLtMatrixLayoutDestroy(op1_desc);
        hipblasLtMatrixLayoutDestroy(D_desc);
        hipblasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("hipBLASLt bf16_run: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, matmul_desc,
        &alpha, B, op0_desc, A, op1_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(op0_desc);
    hipblasLtMatrixLayoutDestroy(op1_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  BF16 no-transpose: D(M,N) = A(M,K) @ B(K,N)  (row-major)
//  B is NOT transposed. For Pi05 weights stored as (K, N).
//  Mirrors gemm_runner.cu:726-734 (cached descriptor + algo).
// ================================================================
void GemmRunner::bf16_nn(void* A, void* B, void* D,
                         int M, int N, int K, hipStream_t stream) {
    auto& entry = get_or_create_cached(BF16_NN, M, N, K);
    if (lazy_autotune_ && !entry.tuned)
        autotune_cached(entry, B, A, D, 1.0f, 0.0f, lazy_pool_);
    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  BF16 no-transpose with fused residual: D(M,N) += A(M,K) @ B(K,N)
//  beta=1.0 accumulates the matmul into D on the FP32 accumulator,
//  avoiding the BF16 round-trip of separate matmul + residual_add.
//  Mirrors gemm_runner.cu:755-763 (cached BF16_NN_RES descriptor).
//  Lazy autotune runs with beta=1.0 so the timed pick reflects the
//  production epilogue; NOTE the timed loop's repeated D += A@B
//  accumulates garbage into D — acceptable because first-shape calls
//  happen only during pre-capture warmup whose outputs are discarded
//  (see the header note).
// ================================================================
void GemmRunner::bf16_nn_res(void* A, void* B, void* D,
                             int M, int N, int K, hipStream_t stream) {
    auto& entry = get_or_create_cached(BF16_NN_RES, M, N, K);
    if (lazy_autotune_ && !entry.tuned)
        autotune_cached(entry, B, A, D, 1.0f, 1.0f, lazy_pool_);
    float alpha = 1.0f, beta = 1.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  BF16 no-transpose with BIAS epilogue: D(M,N) = A(M,K) @ B(K,N) + bias(N)
//  Uses EPILOGUE_BIAS to fuse bias addition into the GEMM.
//  With the col-major swap, D_col is (N,M), so the epilogue bias vector
//  (length = rows of D_col = N, broadcast across columns) lands exactly
//  as bias(N) broadcast over the M rows — same math as the CUDA
//  row-major-order version (gemm_runner.cu:769-825).
// ================================================================
void GemmRunner::bf16_nn_bias(void* A, void* B, void* D, void* bias,
                               int M, int N, int K, hipStream_t stream) {
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t op_N = HIPBLAS_OP_N;

    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Set BIAS epilogue
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    hipDataType bias_type = HIP_R_16BF;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op0_desc, HIP_R_16BF, N, K, N));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op1_desc, HIP_R_16BF, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        op0_desc, op1_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(op0_desc);
        hipblasLtMatrixLayoutDestroy(op1_desc);
        hipblasLtMatrixLayoutDestroy(D_desc);
        hipblasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("hipBLASLt bf16_nn_bias: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, matmul_desc,
        &alpha, B, op0_desc, A, op1_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(op0_desc);
    hipblasLtMatrixLayoutDestroy(op1_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  BF16 GEMM + BIAS + GELU epilogue: D(M,N) = GELU(A(M,K) @ B(K,N) + bias(N))
//  Fuses GEMM + bias + GELU into a single hipBLASLt launch.
//  GELU_BIAS in both libraries computes gelu(x + bias) with the tanh
//  approximation (mirrors gemm_runner.cu:831-887).
// ================================================================
void GemmRunner::bf16_nn_bias_gelu(void* A, void* B, void* D, void* bias,
                                    int M, int N, int K, hipStream_t stream) {
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t op_N = HIPBLAS_OP_N;

    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Set GELU_BIAS epilogue
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    hipDataType bias_type = HIP_R_16BF;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op0_desc, HIP_R_16BF, N, K, N));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op1_desc, HIP_R_16BF, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        op0_desc, op1_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(op0_desc);
        hipblasLtMatrixLayoutDestroy(op1_desc);
        hipblasLtMatrixLayoutDestroy(D_desc);
        hipblasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("hipBLASLt bf16_nn_bias_gelu: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, matmul_desc,
        &alpha, B, op0_desc, A, op1_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(op0_desc);
    hipblasLtMatrixLayoutDestroy(op1_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  BF16 GEMM + BIAS + residual: D(M,N) += A(M,K) @ B(K,N) + bias(N)
//  EPILOGUE_BIAS combined with beta=1.0 — bias and the residual read
//  of D both land on the FP32 accumulator before the single bf16
//  round. Mirrors gemm_runner.cu:893-949 (per-call descriptors +
//  heuristic top-1, same style as bf16_nn_bias above).
// ================================================================
void GemmRunner::bf16_nn_bias_res(void* A, void* B, void* D, void* bias,
                                   int M, int N, int K, hipStream_t stream) {
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    hipblasOperation_t op_N = HIPBLAS_OP_N;

    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    // Set BIAS epilogue with beta=1 for residual
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    hipDataType bias_type = HIP_R_16BF;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op0_desc, HIP_R_16BF, N, K, N));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op1_desc, HIP_R_16BF, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, HIP_R_16BF, N, M, N));

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_, sizeof(workspace_size_)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle_, matmul_desc,
        op0_desc, op1_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(op0_desc);
        hipblasLtMatrixLayoutDestroy(op1_desc);
        hipblasLtMatrixLayoutDestroy(D_desc);
        hipblasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("hipBLASLt bf16_nn_bias_res: no algorithm found");
    }

    float alpha = 1.0f, beta = 1.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, matmul_desc,
        &alpha, B, op0_desc, A, op1_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace_, workspace_size_, stream));

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(op0_desc);
    hipblasLtMatrixLayoutDestroy(op1_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);
}

// ================================================================
//  FP8 no-transpose: D_bf16 = A_fp8(M,K) @ B_fp8(K,N)
//  with device scale pointers (HIP Graph compatible).
//  Matches bf16_nn layout — B stored as (K,N) row-major, no transpose.
//  Mirrors gemm_runner.cu:1083-1099: descale enters via the library's
//  A/B_SCALE_POINTER accumulator scaling, alpha stays 1.0.
// ================================================================
void GemmRunner::fp8_nn_dev(void* A, void* B, void* D,
                             int M, int N, int K,
                             float* d_scale_a, float* d_scale_b,
                             hipStream_t stream) {
    auto& entry = get_or_create_cached(FP8_NN_DEV, M, N, K);
    if (lazy_autotune_ && !entry.tuned)
        autotune_cached(entry, B, A, D, 1.0f, 0.0f, lazy_pool_,
                        d_scale_b, d_scale_a);
    // Update scale pointers per-call (different layers have different scales).
    // Operand swap: hipBLASLt operand A = FlashRT B → gets d_scale_b.
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  FP8 transpose-B path: D_bf16 = A_fp8(M,K) @ B_fp8(N,K)^T
//  with device scale pointers. B is stored as (N,K) row-major.
//  Mirrors gemm_runner.cu:1104-1119.
// ================================================================
void GemmRunner::fp8_nt_dev(void* A, void* B, void* D,
                             int M, int N, int K,
                             float* d_scale_a, float* d_scale_b,
                             hipStream_t stream) {
    auto& entry = get_or_create_cached(FP8_NT_DEV, M, N, K);
    if (lazy_autotune_ && !entry.tuned)
        autotune_cached(entry, B, A, D, 1.0f, 0.0f, lazy_pool_,
                        d_scale_b, d_scale_a);
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
//  MXFP4 a4w4 transpose-B path: D_bf16 = A_fp4(M,K) @ B_fp4(N,K)^T
//  (OCP MX: E2M1 elements + per-1x32-block UE8M0 scales, gfx950)
//
//  How the col-major swap maps the SCALE tensors:
//    op0 (library "A") = FlashRT B, stored (N, K) row-major = B_col(K, N)
//        ld=K, TRANSA=T. Its innermost (stride-1) dim is K, so
//        VEC32_UE8M0 wants one e8m0 byte per 32 elements along K —
//        i.e. a scale tensor shaped col-major (K/32, N). That is
//        byte-identical to the row-major (N, K/32) tensor produced by
//        quantizing B per 1x32 blocks along K. Zero reshuffling.
//    op1 (library "B") = FlashRT A, stored (M, K) row-major = A_col(K, M)
//        ld=K, TRANSB=N. Innermost dim is again K → scale tensor
//        col-major (K/32, M) == row-major (M, K/32). Zero reshuffling.
//  So under the swap: library A_SCALE_POINTER ← FlashRT B_scales and
//  library B_SCALE_POINTER ← FlashRT A_scales, exactly mirroring the
//  data-operand swap. Dequant semantics: value = e2m1 * 2^(byte - 127),
//  applied by hipBLASLt inside the FP32 accumulation.
//
//  Packing: HIP_R_4F_E2M1_EXT layouts keep rows/cols/ld in ELEMENTS;
//  two elements share a byte along the innermost dim (element 2i in the
//  low nibble, 2i+1 in the high nibble — hip_fp4 convention). K % 32
//  == 0 covers both the packing and the block-scale granularity.
//
//  Layout support note: this NT entry point lands on library-TN, the
//  canonical fp4 layout in the hipBLASLt MX samples. If a future need
//  arises for other trans combos, probe support with the heuristic
//  query first — TN is the only one exercised here.
// ================================================================
void GemmRunner::mxfp4_nt_dev(void* A, void* A_scales,
                               void* B, void* B_scales, void* D,
                               int M, int N, int K,
                               hipStream_t stream) {
    if (K % 32 != 0) {
        throw std::runtime_error("mxfp4_nt_dev: K must be a multiple of 32 (MX block size)");
    }
    auto& entry = get_or_create_cached(MXFP4_NT_DEV, M, N, K);
    // Update scale-tensor pointers per call (different layers, different
    // scale buffers). Operand swap: library A = FlashRT B → B_scales.
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &B_scales, sizeof(B_scales)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &A_scales, sizeof(A_scales)));

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// ================================================================
// GROOT N1.7 surface: FP16 GEMM + FP8 epilogue variants
// (signature-identical to the CUDA class; see header note)
// ================================================================

void GemmRunner::fp16_nn(void* A, void* B, void* D,
                         int M, int N, int K, hipStream_t stream) {
    auto& entry = get_or_create_cached(FP16_NN, M, N, K);
    if (lazy_autotune_ && !entry.tuned)
        autotune_cached(entry, B, A, D, 1.0f, 0.0f, lazy_pool_);
    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

void GemmRunner::fp8_descale_fp16(void* A, void* B, void* D,
                                  int M, int N, int K,
                                  float* act_descale, float* w_descale,
                                  hipStream_t stream) {
    auto& entry = get_or_create_cached(FP8_NN_DEV_FP16, M, N, K);
    if (lazy_autotune_ && !entry.tuned)
        autotune_cached(entry, B, A, D, 1.0f, 0.0f, lazy_pool_,
                        w_descale, act_descale);
    // Operand swap: hipBLASLt operand A = FlashRT B (weights) → w_descale.
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(entry.matmul_desc,
        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale)));

    float alpha = 1.0f, beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle_, entry.matmul_desc,
        &alpha, B, entry.op0_desc, A, entry.op1_desc,
        &beta, D, entry.D_desc, D, entry.D_desc,
        &entry.algo, workspace_, workspace_size_, stream));
}

// Shared body for the two fp8 host-alpha epilogue entry points.
// Per-call descriptors (bias pointer + alpha are baked per call, same
// style as bf16_nn_bias above); D/bias are FP16.
static void fp8_epilogue_nn(hipblasLtHandle_t handle,
                            void* workspace, size_t workspace_size,
                            void* A, void* B, void* D, void* bias,
                            int M, int N, int K, float alpha,
                            hipblasLtEpilogue_t epilogue_kind,
                            const char* what, hipStream_t stream) {
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;

    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t op_N = HIPBLAS_OP_N;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_kind, sizeof(epilogue_kind)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    hipDataType bias_type = HIP_R_16F;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op0_desc, HIP_R_8F_E4M3, N, K, N));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&op1_desc, HIP_R_8F_E4M3, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, HIP_R_16F, N, M, N));

    hipblasLtMatmulPreference_t preference;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
    size_t ws = workspace_size;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    int returned_results = 0;
    hipblasLtMatmulHeuristicResult_t heuristic;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle, matmul_desc,
        op0_desc, op1_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(op0_desc);
        hipblasLtMatrixLayoutDestroy(op1_desc);
        hipblasLtMatrixLayoutDestroy(D_desc);
        hipblasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error(std::string("hipBLASLt ") + what + ": no algorithm found");
    }

    float beta = 0.0f;
    HIPBLASLT_CHECK(hipblasLtMatmul(handle, matmul_desc,
        &alpha, B, op0_desc, A, op1_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace, workspace_size, stream));

    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(op0_desc);
    hipblasLtMatrixLayoutDestroy(op1_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);
}

void GemmRunner::fp8_nn_bias(void* A, void* B, void* D, void* bias,
                             int M, int N, int K, float alpha,
                             hipStream_t stream) {
    fp8_epilogue_nn(handle_, workspace_, workspace_size_, A, B, D, bias,
                    M, N, K, alpha, HIPBLASLT_EPILOGUE_BIAS,
                    "fp8_nn_bias", stream);
}

void GemmRunner::fp8_nn_gelu_bias(void* A, void* B, void* D, void* bias,
                                  int M, int N, int K, float alpha,
                                  hipStream_t stream) {
    fp8_epilogue_nn(handle_, workspace_, workspace_size_, A, B, D, bias,
                    M, N, K, alpha, HIPBLASLT_EPILOGUE_GELU_BIAS,
                    "fp8_nn_gelu_bias", stream);
}

void GemmRunner::autotune_fp16_nn(void* A, void* B, void* D,
                                  int M, int N, int K, int num_algos) {
    auto& entry = get_or_create_cached(FP16_NN, M, N, K);
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos);
}

void GemmRunner::autotune_fp8_descale_fp16(void* A, void* B, void* D,
                                           int M, int N, int K,
                                           float* act_descale, float* w_descale,
                                           int num_algos) {
    auto& entry = get_or_create_cached(FP8_NN_DEV_FP16, M, N, K);
    autotune_cached(entry, B, A, D, 1.0f, 0.0f, num_algos, w_descale, act_descale);
}


void GemmRunner::enable_lazy_autotune(int num_algos) {
    lazy_autotune_ = true;
    lazy_pool_ = num_algos;
}
