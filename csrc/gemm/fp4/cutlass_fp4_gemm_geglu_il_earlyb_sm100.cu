// Fused GeGLU decoder gate_up (the v10 128x64x256 tile, compact store, no D
// store) with the weight k-tiles streamed before the programmatic-dependency
// wait: the mainloop is the EarlyB fork (sm100_blockscaled_mma_earlyb.hpp),
// tile and epilogue are those of cutlass_fp4_gemm_geglu_il_hw_nod_v10, so
// the hid_fp4 / hid_sfa bytes are identical. Lets gate_up pull its weights
// while the preceding AdaRMS kernel runs.
#undef CUTLASS_ENABLE_GDC_FOR_SM100
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_variants_earlyb.cuh"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "gemm/fp4/sm100_gelu_mul_blockscale_visitor.hpp"
#include "gemm/fp4/sm100_epilogue_nod.hpp"

namespace flash_rt {
namespace fp4 {
namespace geglu_il_earlyb {
using namespace cute;

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD     = cutlass::float_e2m1_t;
using ElementC     = ElementD;
using LayoutDTag   = cutlass::layout::RowMajor;
using LayoutCTag   = LayoutDTag;
constexpr int AlignmentD = 32;
constexpr int AlignmentC = AlignmentD;
using ElementSFD     = cutlass::float_ue4m3_t;
using LayoutSFDTag   = LayoutDTag;
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
constexpr int OutputSFVectorSize = 16;

using MmaTileShapeV10 = Shape<_128, _64, _256>;
using ClusterShape = Shape<_1, _1, _1>;

using FusionOperationHw = cutlass::epilogue::fusion::GeluMulCompactBlockScaleFactor<
    OutputSFVectorSize, ElementD, ElementCompute, ElementSFD, LayoutSFDTag, ElementC>;

using CollectiveEpilogueHwV10 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShapeV10, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionOperationHw>::CollectiveOp;

using BaseMainloopV10 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, MmaTileShapeV10, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogueHwV10::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

template <class BuiltEpilogue> struct MakeNoD;
template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore, class... Rest>
struct MakeNoD<cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>, Rest...>> {
  using type = cutlass::epilogue::collective::CollectiveEpilogueNoD<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, Rest...>;
};
using CollectiveEpilogueNoDV10 = typename MakeNoD<CollectiveEpilogueHwV10>::type;
static_assert(sizeof(typename CollectiveEpilogueNoDV10::SharedStorage) == sizeof(typename CollectiveEpilogueHwV10::SharedStorage),
              "NoD epilogue must keep the builder's smem footprint");

// Weights (B) streamed for EarlyStages k-tiles before the wait; activations (A) after it.
template <int EarlyStages>
using MainloopEarlyB = typename variants_earlyb::ToEarlyB<BaseMainloopV10, false, 0, false, EarlyStages, false>::type;
template <int EarlyStages>
using KernelEarlyB = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, MainloopEarlyB<EarlyStages>, CollectiveEpilogueNoDV10, void>;
template <int EarlyStages>
using GemmEarlyB = cutlass::gemm::device::GemmUniversalAdapter<KernelEarlyB<EarlyStages>>;

template <class GemmT>
static int run(void const* A_packed, void const* SFA, void const* B_packed, void const* SFB, void* D_dummy,
               void* compact_packed, void* compact_sfa, int M, int N_il, int K, cudaStream_t stream) {
  using StrideAT = typename GemmT::GemmKernel::StrideA;
  using StrideBT = typename GemmT::GemmKernel::StrideB;
  using StrideCT = typename GemmT::GemmKernel::StrideC;
  using StrideDT = typename GemmT::GemmKernel::StrideD;
  using CfgT = typename GemmT::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto stride_A = cutlass::make_cute_packed_stride(StrideAT{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideBT{}, {N_il, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideCT{}, {M, N_il, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideDT{}, {M, N_il, 1});
  cute::get<0>(stride_D) = 0;
  auto layout_SFA = CfgT::tile_atom_to_shape_SFA(make_shape(M, N_il, K, 1));
  auto layout_SFB = CfgT::tile_atom_to_shape_SFB(make_shape(M, N_il, K, 1));
  using EA = typename ElementA::DataType; using SA = typename ElementA::ScaleFactorType;
  using EB = typename ElementB::DataType; using SB = typename ElementB::ScaleFactorType;
  typename GemmT::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N_il, K, 1},
      { reinterpret_cast<EA const*>(A_packed), stride_A, reinterpret_cast<EB const*>(B_packed), stride_B,
        reinterpret_cast<SA const*>(SFA), layout_SFA, reinterpret_cast<SB const*>(SFB), layout_SFB },
      { { 1.0f, 0.0f }, reinterpret_cast<ElementC*>(D_dummy), stride_C, reinterpret_cast<ElementD*>(D_dummy), stride_D }
  };
  args.mainloop.weight_evict_first = get_weight_evict_first();
  args.epilogue.thread.compact_ptr    = reinterpret_cast<uint8_t*>(compact_packed);
  args.epilogue.thread.compact_sf_ptr = reinterpret_cast<uint8_t*>(compact_sfa);
  GemmT gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
  size_t ws_sz = GemmT::get_workspace_size(args);
  void* ws = nullptr;
  if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
  st = gemm.initialize(args, ws, stream);
  if (st != cutlass::Status::kSuccess) { if (ws) cudaFree(ws); return static_cast<int>(st) | 0x20000; }
  st = gemm.run(stream, nullptr, flash_rt::fp4::pdl_launch());
  if (ws) cudaFree(ws);
  return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace geglu_il_earlyb

int cutlass_fp4_gemm_geglu_il_hw_nod_v10_earlyb(
    void const* A_packed, void const* SFA, void const* B_packed, void const* SFB, void* D_dummy,
    void* compact_packed, void* compact_sfa, int M, int N_il, int K, cudaStream_t stream, int early_stages) {
  switch (early_stages) {
    case 7: return geglu_il_earlyb::run<geglu_il_earlyb::GemmEarlyB<7>>(A_packed, SFA, B_packed, SFB, D_dummy, compact_packed, compact_sfa, M, N_il, K, stream);
    case 5: return geglu_il_earlyb::run<geglu_il_earlyb::GemmEarlyB<5>>(A_packed, SFA, B_packed, SFB, D_dummy, compact_packed, compact_sfa, M, N_il, K, stream);
    default: return geglu_il_earlyb::run<geglu_il_earlyb::GemmEarlyB<3>>(A_packed, SFA, B_packed, SFB, D_dummy, compact_packed, compact_sfa, M, N_il, K, stream);
  }
}

}  // namespace fp4
}  // namespace flash_rt
