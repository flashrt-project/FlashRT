// Operand-swapped fused GeGLU GEMM for the decoder FFN (Thor, M_act = 10).
//
// D^T = W_il * X^T with the interleaved gate/up weights as the A operand
// (every streamed row useful), the activations as the 64-wide B tile, the
// 2-SM UMMA tile of the swapped plain GEMMs and the weight k-tiles streamed
// before the programmatic-dependency wait. The epilogue is the column
// compact store (sm100_gelu_mul_blockscale_visitor.hpp): it writes the same
// hid_fp4 / hid_sfa bytes as cutlass_fp4_gemm_geglu_il_hw_nod_v10.
#undef CUTLASS_ENABLE_GDC_FOR_SM100
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"
#include "fused_fp4/pdl.cuh"
#include <utility>
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"
#include "gemm/fp4/sm100_gelu_mul_blockscale_visitor.hpp"
#include "gemm/fp4/sm100_epilogue_nod.hpp"
#include "gemm/fp4/sm100_blockscaled_mma_earlyb.hpp"

namespace flash_rt {
namespace fp4 {
namespace geglu_il_swap {
using namespace cute;

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // W_il (N_il x K)
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // X (M_act x K)
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
// The D path is elided by the NoD epilogue; its layout only shapes an unused
// descriptor, which still has to be encodable: column-major over the swapped
// (N_il, M_act) problem gives a 16-byte-aligned pitch for M_act = 10.
using ElementD     = cutlass::half_t;
using ElementC     = void;   // no source operand: keeps the builder from reusing C smem for D (NoD requirement)
using LayoutDTag   = cutlass::layout::ColumnMajor;
using LayoutCTag   = LayoutDTag;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;
using ElementSFD     = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
constexpr int SFVectorSize = 16;

using MmaTileShape = Shape<_256, _64, _256>;
using ClusterShape = Shape<_2, _1, _1>;

// The scale-factor layout tag only keys the FusionCallbacks specialization (RowMajor).
using FusionOperation = cutlass::epilogue::fusion::GeluMulCompactColBlockScaleFactor<
    SFVectorSize, ElementD, ElementCompute, ElementSFD, cutlass::layout::RowMajor, ElementD>;

using CollectiveEpilogueBase = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;

template <class T> struct MakeNoD;
template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore, class... Rest>
struct MakeNoD<cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>, Rest...>> {
  using type = cutlass::epilogue::collective::CollectiveEpilogueNoD<
      StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, Rest...>;
};
using CollectiveEpilogue = typename MakeNoD<CollectiveEpilogueBase>::type;
static_assert(sizeof(typename CollectiveEpilogue::SharedStorage) == sizeof(typename CollectiveEpilogueBase::SharedStorage),
              "NoD epilogue must keep the builder's smem footprint");

using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogueBase::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

// Same adaptation as cutlass_fp4_gemm_variants_earlyb.cu: weights (A) early, 3 k-tiles, trigger from the load warp.
template <class T> struct ToEarlyA;
template <int S, int SP, int AP, class CS, class... Rest>
struct ToEarlyA<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<S, SP, AP, CS>, Rest...>> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledEarlyB<S, SP, AP, CS,
          cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<SP, AP>, true, 3, false>, Rest...>;
};
using CollectiveMainloop = typename ToEarlyA<BaseMainloop>::type;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

}  // namespace geglu_il_swap

int cutlass_fp4_gemm_geglu_il_hw_nod_swap(
    void const* A_packed, void const* SFA,      // activations (M x K) + their SFA
    void const* B_packed, void const* SFB,      // interleaved gate/up weights (N_il x K) + their SFB
    void*       D_dummy,
    void*       compact_packed,
    void*       compact_sfa,
    int M, int N_il, int K,
    cudaStream_t stream) {
  using namespace geglu_il_swap;
  // Swapped problem: rows = N_il (weights), columns = M (activations).
  const int Mw = N_il, Nx = M;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {Mw, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {Nx, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {Mw, Nx, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {Mw, Nx, 1});
  auto layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(Mw, Nx, K, 1));
  auto layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(Mw, Nx, K, 1));
  using EA = typename ElementA::DataType; using SA = typename ElementA::ScaleFactorType;
  using EB = typename ElementB::DataType; using SB = typename ElementB::ScaleFactorType;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {Mw, Nx, K, 1},
      { reinterpret_cast<EA const*>(B_packed), stride_A, reinterpret_cast<EB const*>(A_packed), stride_B,
        reinterpret_cast<SA const*>(SFB), layout_SFA, reinterpret_cast<SB const*>(SFA), layout_SFB },
      { { 1.0f, 0.0f }, nullptr, stride_C, reinterpret_cast<ElementD*>(D_dummy), stride_D }
  };
  args.epilogue.thread.compact_ptr    = reinterpret_cast<uint8_t*>(compact_packed);
  args.epilogue.thread.compact_sf_ptr = reinterpret_cast<uint8_t*>(compact_sfa);
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

}  // namespace fp4
}  // namespace flash_rt
