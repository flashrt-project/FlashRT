// ================================================================
// FlashRT — CUTLASS FP16 GEMM implementations for SM100/SM110
//
// FP16 mirror of cutlass_sm100.cu. Tile configurations match the FP8
// variants so we can compare cuBLASLt vs CUTLASS on the same shapes.
//
// Layout: A row-major (M, K), B column-major (K, N) [stored as
// [N, K] row-major in memory — same as PyTorch nn.Linear weights],
// D row-major (M, N).
// ================================================================

#include "gemm_types_sm100_fp16.h"
#include "cutlass/util/device_memory.h"
#include <cuda_runtime.h>
#include <cstdio>

// ── Generic runner ──
template <typename GemmOp>
static int cutlass_run_impl_fp16(void* A, void* B, void* D,
                                  int M, int N, int K,
                                  float alpha, float beta,
                                  cudaStream_t stream) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(ElementA*)A, stride_A, (ElementB*)B, stride_B},
        {{alpha, beta}, (ElementD*)D, stride_D, (ElementD*)D, stride_D}
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS-FP16] cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    status = gemm.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS-FP16] init failed: M=%d N=%d K=%d\n", M, N, K);
        return -2;
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS-FP16] run failed: M=%d N=%d K=%d\n", M, N, K);
        return -3;
    }
    return 0;
}

extern "C" {

int cutlass_fp16_plain(void* A, void* B, void* D, int M, int N, int K,
                        float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl_fp16<sm100_fp16_plain::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp16_sq(void* A, void* B, void* D, int M, int N, int K,
                     float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl_fp16<sm100_fp16_sq::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp16_t1(void* A, void* B, void* D, int M, int N, int K,
                     float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl_fp16<sm100_fp16_t1::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp16_wide(void* A, void* B, void* D, int M, int N, int K,
                       float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl_fp16<sm100_fp16_wide::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp16_k64(void* A, void* B, void* D, int M, int N, int K,
                      float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl_fp16<sm100_fp16_k64::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp16_2sm21(void* A, void* B, void* D, int M, int N, int K,
                        float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl_fp16<sm100_fp16_2sm21::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp16_k64_gelu(void* A, void* B, void* D, int M, int N, int K,
                           float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl_fp16<sm100_fp16_k64_gelu::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp16_sq_gelu(void* A, void* B, void* D, int M, int N, int K,
                          float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl_fp16<sm100_fp16_sq_gelu::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

// R3.1 Phase 2 runner: GEMM with LinCombDeEltAct epilogue that multiplies
// the accumulator by an auxiliary tensor (gate_buf [M, N] row-major).
// D[m,n] = (acc[m,n]) * Aux[m,n].
int cutlass_fp16_k64_mul_aux(void* A, void* B, void* Aux, void* D,
                              int M, int N, int K, cudaStream_t stream) {
    using GemmOp = sm100_fp16_k64_mul_aux::Gemm;
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;

    auto sA = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto sB = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto sD = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    // Aux is row-major [M, N] with the same shape as D.
    using AuxStride = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
    auto dAux = cutlass::make_cute_packed_stride(AuxStride{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(ElementA*)A, sA, (ElementB*)B, sB},
        {
            {  // FusionCallbacks::Arguments (LinCombDeEltAct)
                1.0f,                                  // alpha
                0.0f,                                  // beta
                nullptr, nullptr,                      // alpha_ptr, beta_ptr
                {}, {},                                // dAlpha, dBeta (broadcast stride)
                {},                                    // activation args (multiplies has none)
                (cutlass_fp16_t const*)Aux,            // aux_ptr
                dAux                                   // dAux stride
            },
            nullptr, {},                               // ptr_C (unused since beta=0), dC
            (ElementD*)D, sD
        }
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[k64_mul_aux] cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    if (gemm.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[k64_mul_aux] init failed: M=%d N=%d K=%d\n", M, N, K);
        return -2;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[k64_mul_aux] run failed: M=%d N=%d K=%d\n", M, N, K);
        return -3;
    }
    return 0;
}

}  // extern "C"
