// NVFP4 GEMM variants whose mainloop streams the weight tiles before the
// programmatic-dependency wait (see sm100_blockscaled_mma_earlyb.hpp).
// The kernel-level GDC waits are compiled out here on purpose: the fork's
// load() performs the wait after the weight prefetch.
#undef CUTLASS_ENABLE_GDC_FOR_SM100
#include "fused_fp4/pdl.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"
#include "gemm/fp4/sm100_blockscaled_mma_earlyb.hpp"

namespace flash_rt {
namespace fp4 {
namespace variants_earlyb {
using namespace cute;

template <class T> struct ToEarlyB;
template <int S, int SP, int AP, class CS, class... Rest>
struct ToEarlyB<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<S, SP, AP, CS>, Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledEarlyB<S, SP, AP, CS>, Rest...>;
};

template <class MmaTile, class Cluster>
struct Variant {
  using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;
  using ElementD   = cutlass::half_t;
  using ElementC   = cutlass::half_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ElementAccumulator = float;
  using ArchTag            = cutlass::arch::Sm100;
  using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB, ElementAccumulator, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename ToEarlyB<BaseMainloop>::type;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  static int run(void const* A, void const* SFA, void const* B, void const* SFB,
                 void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    using EA = typename ElementA::DataType; using SA = typename ElementA::ScaleFactorType;
    using EB = typename ElementB::DataType; using SB = typename ElementB::ScaleFactorType;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        { reinterpret_cast<EA const*>(A), stride_A, reinterpret_cast<EB const*>(B), stride_B,
          reinterpret_cast<SA const*>(SFA), layout_SFA, reinterpret_cast<SB const*>(SFB), layout_SFB },
        { {alpha, beta}, reinterpret_cast<ElementC*>(D), stride_C, reinterpret_cast<ElementD*>(D), stride_D }
    };
    Gemm gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
    size_t ws_sz = Gemm::get_workspace_size(args);
    void* ws = nullptr;
    if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) { if (ws) cudaFree(ws); return static_cast<int>(st) | 0x20000; }
    st = gemm.run(stream, nullptr, flash_rt::fp4::pdl_launch());
    if (ws) cudaFree(ws);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
  }
};
using E0 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>>;   // v10 tile
using E1 = Variant<Shape<_128,_128,_256>, Shape<_1,_1,_1>>;   // v7 tile
using E2 = Variant<Shape<_128,_256,_256>, Shape<_1,_1,_1>>;   // v8 tile
}  // namespace variants_earlyb

int cutlass_fp4_gemm_variant_earlyb(int idx, void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
  using namespace variants_earlyb;
  switch (idx) {
    case 0: return E0::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 1: return E1::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 2: return E2::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    default: return -99;
  }
}
}  // namespace fp4
}  // namespace flash_rt
