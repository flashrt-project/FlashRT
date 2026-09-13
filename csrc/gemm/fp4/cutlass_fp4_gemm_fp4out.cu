// ============================================================================
//  FlashRT — NVFP4 GEMM with FP4 (e2m1) packed output + SFA tile-interleaved.
//
//  Strict copy of CUTLASS example 72b's NVFP4→NVFP4 GEMM, adapted to our
//  variant style. Used for both gate_proj and up_proj in the P1 split-GU
//  FFN path. Both inputs and output are NVFP4 packed; the output SFA can
//  feed directly into the next NVFP4 GEMM as A's SFA.
// ============================================================================
#include "gemm/fp4/cutlass_fp4_gemm_fp4out.cuh"
#include "fused_fp4/pdl.cuh"

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

namespace flash_rt {
namespace fp4 {
namespace fp4out {

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

// V8 shape (per audit best for [968, 16384, 2048]; for split half-N use V6 maybe later)
using MmaTileShape = Shape<_128, _256, _256>;
using ClusterShape = Shape<_1, _1, _1>;

// Proven LinCombBlockScaleFactor (examples 72b, 79b, 80b, 92).
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
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

}  // namespace fp4out

int cutlass_fp4_gemm_fp4out(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void*       D_packed,
    void*       D_SFD,
    int M, int N, int K,
    cudaStream_t stream) {
  using namespace fp4out;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  using EA = typename ElementA::DataType;
  using SA = typename ElementA::ScaleFactorType;
  using EB = typename ElementB::DataType;
  using SB = typename ElementB::ScaleFactorType;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
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
  st = gemm.run(stream, nullptr, flash_rt::fp4::pdl_launch());
  if (ws) cudaFree(ws);
  return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace fp4
}  // namespace flash_rt
