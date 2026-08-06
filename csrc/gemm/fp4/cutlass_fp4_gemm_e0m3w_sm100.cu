// ============================================================================
//  Block-scaled GEMM with E0M3 (uniform INT4) weights — SM110 runtime idesc.
//
//  Structure mirrors cutlass_fp4_gemm_variants.cu with two differences:
//    * ElementA/B are the CUTLASS runtime datatype
//      (type_erased_dynamic_nv_float4_t), which makes the collective read
//      the element format from mainloop arguments at run time and write it
//      straight into the UMMA instruction descriptor.
//    * runtime_data_type_a = E2M1 (1) and runtime_data_type_b = 0, the
//      undocumented E0M3 encoding validated by an element-level canary
//      (distinct payloads decode to the exact sign-magnitude INT4 dot
//      products, including negative codes).
// ============================================================================

#include "cutlass_fp4_gemm_e0m3w_sm100.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
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

namespace e0m3w {

using namespace cute;

struct RunnerV10 {
  using ElementA   = cutlass::type_erased_dynamic_nv_float4_t;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB   = cutlass::type_erased_dynamic_nv_float4_t;
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
  // Production decoder projection tile (variant v10).
  using MmaTile            = Shape<_128, _64, _256>;
  using Cluster            = Shape<_1, _1, _1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      MmaTile, Cluster,
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
  using RuntimeA = typename Gemm::GemmKernel::CollectiveMainloop::RuntimeDataTypeA;
  using RuntimeB = typename Gemm::GemmKernel::CollectiveMainloop::RuntimeDataTypeB;
  using ArrayElementA =
      typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB =
      typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

  // Instruction-descriptor element-format values (3-bit field).
  static constexpr int kFormatE2M1 = 1;
  static constexpr int kFormatE0M3 = 0;

  static int run(void const* A, void const* SFA, void const* B, void const* SFB,
                 void* D, int M, int N, int K, float alpha, float beta,
                 cudaStream_t stream, int a_format) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        { reinterpret_cast<ArrayElementA const*>(A), stride_A,
          reinterpret_cast<ArrayElementB const*>(B), stride_B,
          reinterpret_cast<cutlass::float_ue4m3_t const*>(SFA), layout_SFA,
          reinterpret_cast<cutlass::float_ue4m3_t const*>(SFB), layout_SFB },
        { {alpha, beta},
          reinterpret_cast<ElementC*>(D), stride_C,
          reinterpret_cast<ElementD*>(D), stride_D }
    };
    args.mainloop.runtime_data_type_a =
        static_cast<RuntimeA>(a_format & 0b111);
    args.mainloop.runtime_data_type_b = static_cast<RuntimeB>(kFormatE0M3);

    Gemm gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
    size_t ws_sz = Gemm::get_workspace_size(args);
    void* ws = nullptr;
    if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) { if (ws) cudaFree(ws); return static_cast<int>(st) | 0x20000; }
    st = gemm.run(stream);
    if (ws) cudaFree(ws);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
  }
};

}  // namespace e0m3w

int cutlass_fp4_gemm_e0m3w(
    void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta,
    cudaStream_t stream, int a_format) {
  return e0m3w::RunnerV10::run(A, SFA, B, SFB, D, M, N, K, alpha, beta,
                               stream, a_format);
}

}  // namespace fp4
}  // namespace flash_rt
