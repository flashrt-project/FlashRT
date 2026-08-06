// ============================================================================
//  FlashRT — NVFP4 GEMM + fused GeGLU epilogue over column-interleaved
//  gate/up weight, FP4 packed output + SFD.
//
//  Structure mirrors cutlass_fp4_gemm_fp4out.cu (the production P1 fp4out
//  GEMM); the only functional difference is the fusion operation, which
//  applies gelu(gate)*up on adjacent accumulator column pairs before the
//  block-scale-factor generation (sm100_gelu_mul_blockscale_visitor.hpp).
// ============================================================================
#include "gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cuh"

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

namespace flash_rt {
namespace fp4 {
namespace geglu_il {

using namespace cute;

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// FP4 output (e2m1 packed)
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

constexpr int InputSFVectorSize  = 16;
constexpr int OutputSFVectorSize = InputSFVectorSize;

// Same tile as the production fp4out GEMM (audit best for [968, 16384, 2048]).
using MmaTileShape = Shape<_128, _256, _256>;
using ClusterShape = Shape<_1, _1, _1>;

using FusionOperation = cutlass::epilogue::fusion::GeluMulBlockScaleFactor<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD, LayoutSFDTag,
    ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ── Half-width (compact-store) instantiation — only the fusion op differs ──
using FusionOperationHw = cutlass::epilogue::fusion::GeluMulCompactBlockScaleFactor<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD, LayoutSFDTag,
    ElementC>;

using CollectiveEpilogueHw = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperationHw
>::CollectiveOp;

using CollectiveMainloopHw = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogueHw::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernelHw = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopHw, CollectiveEpilogueHw, void>;

using GemmHw = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelHw>;

// ── Skinny-M compact-store instantiation (decoder FFN: tiny M, N=8192) ──
// Same tile as the production decoder GEMMs (v10, 128x64x256): the small
// N tile keeps enough CTAs in flight to stream the weight at DRAM rate.
using MmaTileShapeV10 = Shape<_128, _64, _256>;

using CollectiveEpilogueHwV10 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShapeV10, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperationHw
>::CollectiveOp;

using CollectiveMainloopHwV10 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShapeV10, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogueHwV10::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernelHwV10 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopHwV10, CollectiveEpilogueHwV10, void>;

using GemmHwV10 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelHwV10>;

}  // namespace geglu_il

int cutlass_fp4_gemm_geglu_il(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_packed,
    void*       D_SFD,
    int M, int N_il, int K,
    cudaStream_t stream) {
  using namespace geglu_il;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N_il, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N_il, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N_il, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N_il, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N_il, K, 1));

  using EA = typename ElementA::DataType;
  using SA = typename ElementA::ScaleFactorType;
  using EB = typename ElementB::DataType;
  using SB = typename ElementB::ScaleFactorType;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N_il, K, 1},
      { reinterpret_cast<EA const*>(A_packed), stride_A,
        reinterpret_cast<EB const*>(B_packed), stride_B,
        reinterpret_cast<SA const*>(SFA), layout_SFA,
        reinterpret_cast<SB const*>(SFB), layout_SFB },
      { /* thread args (FusionCallbacks::Arguments) */ { 1.0f, 0.0f },
        reinterpret_cast<ElementC*>(D_packed), stride_C,
        reinterpret_cast<ElementD*>(D_packed), stride_D }
  };
  // BlockScaleFactor needs a non-null norm_constant_ptr (kernel reads it
  // unconditionally). Allocate a single fp32 = 1.0 once on device.
  static float* d_norm = nullptr;
  if (!d_norm) {
    cudaMalloc(&d_norm, sizeof(float));
    float h = 1.0f;
    cudaMemcpyAsync(d_norm, &h, sizeof(float), cudaMemcpyHostToDevice, stream);
  }
  args.epilogue.thread.block_scale_factor_ptr = reinterpret_cast<ElementSFD*>(D_SFD);
  args.epilogue.thread.norm_constant_ptr      = d_norm;

  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
  size_t ws_sz = Gemm::get_workspace_size(args);
  void* ws = nullptr;
  if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
  st = gemm.initialize(args, ws, stream);
  if (st != cutlass::Status::kSuccess) {
    if (ws) cudaFree(ws);
    return static_cast<int>(st) | 0x20000;
  }
  st = gemm.run(stream);
  if (ws) cudaFree(ws);
  return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

namespace geglu_il {

template <class GemmT>
static int run_geglu_il_hw(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_dummy,
    void*       compact_packed,
    void*       compact_sfa,
    int M, int N_il, int K,
    cudaStream_t stream) {
  using StrideAT = typename GemmT::GemmKernel::StrideA;
  using StrideBT = typename GemmT::GemmKernel::StrideB;
  using StrideCT = typename GemmT::GemmKernel::StrideC;
  using StrideDT = typename GemmT::GemmKernel::StrideD;
  using CfgT = typename GemmT::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideAT{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideBT{}, {N_il, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideCT{}, {M, N_il, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideDT{}, {M, N_il, 1});
  auto layout_SFA = CfgT::tile_atom_to_shape_SFA(make_shape(M, N_il, K, 1));
  auto layout_SFB = CfgT::tile_atom_to_shape_SFB(make_shape(M, N_il, K, 1));

  using EA = typename ElementA::DataType;
  using SA = typename ElementA::ScaleFactorType;
  using EB = typename ElementB::DataType;
  using SB = typename ElementB::ScaleFactorType;

  typename GemmT::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N_il, K, 1},
      { reinterpret_cast<EA const*>(A_packed), stride_A,
        reinterpret_cast<EB const*>(B_packed), stride_B,
        reinterpret_cast<SA const*>(SFA), layout_SFA,
        reinterpret_cast<SB const*>(SFB), layout_SFB },
      { /* thread args (FusionCallbacks::Arguments) */ { 1.0f, 0.0f },
        reinterpret_cast<ElementC*>(D_dummy), stride_C,
        reinterpret_cast<ElementD*>(D_dummy), stride_D }
  };
  args.epilogue.thread.compact_ptr    = reinterpret_cast<uint8_t*>(compact_packed);
  args.epilogue.thread.compact_sf_ptr = reinterpret_cast<uint8_t*>(compact_sfa);

  GemmT gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
  size_t ws_sz = GemmT::get_workspace_size(args);
  void* ws = nullptr;
  if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
  st = gemm.initialize(args, ws, stream);
  if (st != cutlass::Status::kSuccess) {
    if (ws) cudaFree(ws);
    return static_cast<int>(st) | 0x20000;
  }
  st = gemm.run(stream);
  if (ws) cudaFree(ws);
  return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace geglu_il

int cutlass_fp4_gemm_geglu_il_hw(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_dummy,
    void*       compact_packed,
    void*       compact_sfa,
    int M, int N_il, int K,
    cudaStream_t stream) {
  return geglu_il::run_geglu_il_hw<geglu_il::GemmHw>(
      A_packed, SFA, B_packed, SFB, D_dummy, compact_packed, compact_sfa,
      M, N_il, K, stream);
}

int cutlass_fp4_gemm_geglu_il_hw_v10(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_dummy,
    void*       compact_packed,
    void*       compact_sfa,
    int M, int N_il, int K,
    cudaStream_t stream) {
  return geglu_il::run_geglu_il_hw<geglu_il::GemmHwV10>(
      A_packed, SFA, B_packed, SFB, D_dummy, compact_packed, compact_sfa,
      M, N_il, K, stream);
}

}  // namespace fp4
}  // namespace flash_rt
