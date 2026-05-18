// ============================================================================
// FlashRT — encoder split-G7 megakernel (Stage B prototype).
//
// Stage B-v2: visitor-owned SMEM (mirrors Sm90AuxStore pattern).
//   - Phase 1 (gate) fusion = Sm90EVT<Sm100SmemAuxStore, Sm90EVT<Sm90Compute<GELU>,
//                                     Sm90LinearCombination>>
//     captures post-GELU to phase 1 visitor's OWN SharedStorage.
//   - Phase 2 (up) fusion = Sm90EVT<Sm90Compute<multiplies>, Sm90LinearCombination,
//                                   Sm100SmemAuxLoad>
//     loads aux from phase 2 visitor's OWN SharedStorage (NOT phase 1's).
//
// Cross-phase data sharing is NOT YET wired at this stage — phase 1 writes
// to phase1.epilogue.fusion.smem_aux; phase 2 reads from
// phase2.epilogue_2.fusion.smem_aux (different memory regions).  So numerical
// output will be wrong, but the test is: does the kernel RUN without
// illegal address?  That validates the visitor mechanic isolated from
// the cross-phase plumbing.
//
// Once kernel runs: add a kernel-body S2S copy from phase1's smem_aux to
// phase2's smem_aux at phase transition, OR overlay the two via kernel
// TensorStorage union.
// ============================================================================

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/integral_constant.hpp"

#include "sm100_smem_aux_visitor.hpp"
#include "flashrt_megakernel_geglu_kernel.hpp"

#include <cuda_runtime.h>
#include <cstdio>

using namespace cute;
using fp16_t = cutlass::half_t;

namespace {

// Stage E3 best tile: (128, 128, 128) Cluster (2,2,1) with shared SMEM_A.
// Per-CTA (64, 64, 128) — TileK=128 (production sq's K).
// Low Thor regime: 1.06-1.07x faster than production back-to-back.
using Tile    = Shape<_128, _128, _128>;
using Cluster = Shape<_2, _2, _1>;

using FusionGate = flashrt::megakernel::fusion::LinCombEltActSmemAuxStore<
    cutlass::epilogue::thread::GELU_taylor, fp16_t, float, fp16_t>;

using FusionUp = flashrt::megakernel::fusion::LinCombDeEltActSmemAuxLoad<
    cutlass::multiplies, fp16_t, float, fp16_t>;

using CollectiveEpiGate = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionGate>::CollectiveOp;

using CollectiveEpiUp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionUp>::CollectiveOp;

using CollectiveMmaGate = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using CollectiveMmaUp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::FlashRtMegakernelGeGLUFusedGemm<
    Shape<int, int, int, int>,
    CollectiveMmaGate, CollectiveEpiGate,
    CollectiveMmaUp,   CollectiveEpiUp,
    void>;

using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // anonymous namespace

extern "C" int flashrt_megakernel_geglu_fp16(
    void* X, void* W_gate, void* W_up,
    void* D_gate_scratch, void* hidden,
    int M, int N, int K,
    cudaStream_t stream)
{
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;

    auto sA = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto sB = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto sD = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(ElementA*)X, sA, (ElementB*)W_gate, sB},
        {
            // Stage B-v2: visitor has empty Arguments (owns SMEM internally)
            { 1.0f, 0.0f, nullptr, nullptr, {}, {}, {} },
            nullptr, {},
            (ElementD*)D_gate_scratch, sD
        },
        {(ElementA*)X, sA, (ElementB*)W_up, sB},
        {
            { 1.0f, 0.0f, nullptr, nullptr, {}, {}, {} },
            nullptr, {},
            (ElementD*)hidden, sD
        }
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }

    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_geglu] cannot implement M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    if (gemm.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_geglu] init failed\n");
        return -2;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_geglu] run failed\n");
        return -3;
    }
    return 0;
}
