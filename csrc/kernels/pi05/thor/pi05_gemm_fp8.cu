#include "gemm/gemm_types_sm100.h"
#include "kernels/pi05/thor/pdl.cuh"
#include "cutlass/util/device_memory.h"
#include <cuda_runtime.h>
#include <cstdio>

template <typename GemmOp>
static int cutlass_run_impl(void* A, void* B, void* D,
                             int M, int N, int K,
                             float alpha, float beta,
                             cudaStream_t stream) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;

    // CUTLASS stride computation
    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},  // problem size
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
        fprintf(stderr, "[CUTLASS] cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }

    status = gemm.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] init failed: M=%d N=%d K=%d\n", M, N, K);
        return -2;
    }

    status = gemm.run(stream, nullptr, flash_rt::fp4::pdl_launch());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] run failed: M=%d N=%d K=%d\n", M, N, K);
        return -3;
    }
    return 0;
}

extern "C" {
int cutlass_fp8_sq(void* A, void* B, void* D, int M, int N, int K,
                    float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_sq::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}
}
