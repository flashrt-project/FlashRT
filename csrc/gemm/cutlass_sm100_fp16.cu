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

}  // extern "C"
