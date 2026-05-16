// ================================================================
// FlashRT — single-GEMM kernel using the vendored production
// sm100_gemm_tma_warpspecialized.hpp kernel struct (now
// flashrt::megakernel::FlashRtMegakernelGemm).
//
// Purpose: prove that vendoring the production SM100 kernel into our
// codebase preserves its perf characteristics for a single GEMM.
// This is the foundation for the megakernel work in subsequent
// commits — the operator() body in flashrt_megakernel_kernel.hpp
// will be modified to chain multiple CollectiveMma instances with
// GeGLU in SMEM between them.  The header is a literal copy of
// CUTLASS's production sm100_gemm_tma_warpspecialized.hpp with only
// the namespace and class name changed; no functional modification yet.
//
// Validation gate: at the split-G7 shape (M=1024, K=2048, N=16384)
// this must match cutlass_fp16_sq isolation perf (~750 us at n=500).
// If it doesn't, the vendor port is broken and we cannot proceed
// to megakernel extension.
// ================================================================

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/integral_constant.hpp"

#include "flashrt_megakernel_kernel.hpp"

#include <cuda_runtime.h>
#include <cstdio>

using namespace cute;
using fp16_t = cutlass::half_t;

namespace {

// Same tile/cluster/schedule as production sm100_fp16_sq (the current
// split-G7 winner).  This isolates the vendor port: any perf delta
// here is from the kernel struct itself, not the tile selection.
using Tile    = Shape<_256, _256, _128>;
using Cluster = Shape<_2, _2, _1>;

using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, fp16_t, float>;

using CollectiveEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;

using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

// Use the vendored kernel struct in place of CUTLASS's GemmUniversal.
// Note: the SFINAE enable_if_t inside FlashRtMegakernelGemm matches on
// CollectiveMainloop's Schedule tag (KernelTmaWarpSpecializedSm100),
// so any Sm100 TMA WS mainloop instantiation is accepted.
using GemmKernel = cutlass::gemm::kernel::FlashRtMegakernelGemm<
    Shape<int, int, int, int>,
    CollectiveMma,
    CollectiveEpi,
    void>;  // TileSchedulerTag = void → default scheduler

using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // anonymous namespace

extern "C" int flashrt_megakernel_single_fp16(
    void* A, void* B, void* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;

    auto sA = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto sB = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto sD = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(ElementA*)A, sA, (ElementB*)B, sB},
        {{alpha, beta}, (ElementD*)D, sD, (ElementD*)D, sD}
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_single] cannot implement M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    if (gemm.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_single] init failed\n");
        return -2;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_single] run failed\n");
        return -3;
    }
    return 0;
}
