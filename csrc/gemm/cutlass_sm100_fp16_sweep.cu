// ================================================================
// FlashRT — CUTLASS FP16 GEMM TILE SWEEP (transient infrastructure)
//
// Provides ~8 tile/cluster/schedule variants in a single dispatch
// entry point, used to find the best CUTLASS configuration per encoder
// GEMM shape.  Developer-only: built behind -DFLASHRT_BUILD_SM100_SWEEP=ON.
//
// Layout matches cutlass_sm100_fp16.cu:
//   A row-major (M, K), B column-major (K, N) stored as [N, K]
//   row-major, D row-major (M, N).  FP16 in/out, FP32 accumulate.
//
// Once the sweep concludes, the winning variant should be promoted
// into gemm_types_sm100_fp16.h and this file deleted.
// ================================================================

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include <cuda_runtime.h>
#include <cstdio>

using namespace cute;
using fp16_t = cutlass::half_t;

namespace sweep_fp16 {

template <class TileShape_, class ClusterShape_,
          class KernelSched_ = cutlass::gemm::collective::KernelScheduleAuto,
          class EpiSched_    = cutlass::epilogue::collective::EpilogueScheduleAuto>
struct Variant {
    using Tile    = TileShape_;
    using Cluster = ClusterShape_;
    using Fusion  = cutlass::epilogue::fusion::LinCombEltAct<
        cutlass::epilogue::thread::Identity, fp16_t, float>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
        float, float, fp16_t, cutlass::layout::RowMajor, 8,
        fp16_t, cutlass::layout::RowMajor, 8,
        EpiSched_, Fusion>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        fp16_t, cutlass::layout::RowMajor, 8,
        fp16_t, cutlass::layout::ColumnMajor, 8,
        float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        KernelSched_>::CollectiveOp;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
        cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
};

// V0: Baseline = current `sq`. (256, 256, 128) cluster (2, 2, 1) auto.
using V0 = Variant<Shape<_256,_256,_128>, Shape<_2,_2,_1>>;
// V1: Smaller K-tile, expect more pipelined stages.
using V1 = Variant<Shape<_256,_256,_64>,  Shape<_2,_2,_1>>;
// V2: Narrower M (good when M=1024 << N).
using V2 = Variant<Shape<_128,_256,_128>, Shape<_2,_2,_1>>;
// V3: Narrower N (good for K-heavy shapes like G8).
using V3 = Variant<Shape<_256,_128,_128>, Shape<_2,_2,_1>>;
// V4: 1SM mode, no cluster (Tile_M <= 128 and reduced N/K so >=2 stages fit).
using V4 = Variant<Shape<_128,_128,_128>, Shape<_1,_1,_1>>;
// V5: Tile match, 2x1 cluster (TMA 2SM friendly).
using V5 = Variant<Shape<_256,_256,_128>, Shape<_2,_1,_1>,
                   cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                   cutlass::epilogue::TmaWarpSpecialized2Sm>;
// V6: Larger M-tile (test 256x128 narrow N pattern with 2x2).
using V6 = Variant<Shape<_128,_128,_256>, Shape<_2,_2,_1>>;
// V7: 4x2 cluster (more SMs per group; 8 SMs needed).
using V7 = Variant<Shape<_256,_256,_128>, Shape<_4,_2,_1>>;
// V8: Explicit 1SM small-M tile (probe whether tall-skinny tile helps M=1024).
using V8 = Variant<Shape<_64,_128,_128>, Shape<_1,_1,_1>,
                   cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
                   cutlass::epilogue::TmaWarpSpecialized1Sm>;

}  // namespace sweep_fp16

template <typename GemmOp>
static int run_impl(void* A, void* B, void* D, int M, int N, int K,
                    float alpha, float beta, cudaStream_t stream) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;
    auto sA = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto sB = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto sD = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});
    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {(ElementA*)A, sA, (ElementB*)B, sB},
        {{alpha, beta}, (ElementD*)D, sD, (ElementD*)D, sD}
    };
    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -10;
    if (gemm.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) return -11;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return -12;
    return 0;
}

extern "C" int cutlass_fp16_sweep(int variant, void* A, void* B, void* D,
                                  int M, int N, int K,
                                  float alpha, float beta, cudaStream_t stream) {
    switch (variant) {
        case 0: return run_impl<sweep_fp16::V0::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        case 1: return run_impl<sweep_fp16::V1::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        case 2: return run_impl<sweep_fp16::V2::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        case 3: return run_impl<sweep_fp16::V3::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        case 4: return run_impl<sweep_fp16::V4::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        case 5: return run_impl<sweep_fp16::V5::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        case 6: return run_impl<sweep_fp16::V6::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        case 7: return run_impl<sweep_fp16::V7::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        case 8: return run_impl<sweep_fp16::V8::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
        default: return -1;
    }
}

extern "C" int cutlass_fp16_sweep_count() { return 9; }
