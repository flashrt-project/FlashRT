// NVFP4 GEMM variants with the operands swapped for small-M problems.
//
// For M=10 activations the stock kernel streams a 128-row A box per k-tile of
// which 118 rows are out of bounds; the zero-filled rows still cost TMA/L2
// bandwidth and end up being two thirds of the traffic. These variants compute
// D^T = W * X^T instead: the weights are the A operand (every row useful), the
// activations are the B operand (N-tile 64, 54 rows wasted instead of 118) and
// D is stored column-major, which is byte-for-byte the row-major (M, N) output
// the callers already expect. The public entry keeps the (activation, weight)
// argument order of cutlass_fp4_gemm_variant.
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

namespace flash_rt {
namespace fp4 {
namespace variants_swap {
using namespace cute;

template <class MmaTile, class Cluster>
struct Variant {
  using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // weights (N_out x K), K contiguous
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // activations (M_act x K), K contiguous
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;
  using ElementD   = cutlass::half_t;
  using ElementC   = cutlass::half_t;
  using LayoutCTag = cutlass::layout::ColumnMajor;   // (N_out, M_act) column-major == (M_act, N_out) row-major
  using LayoutDTag = cutlass::layout::ColumnMajor;
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
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB, ElementAccumulator, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  // A = weights (Mw = N_out rows), B = activations (Nx = M_act rows). D is (Mw, Nx) column-major.
  static int run(void const* W, void const* SFW, void const* X, void const* SFX,
                 void* D, int Mw, int Nx, int K, float alpha, float beta, cudaStream_t stream) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {Mw, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {Nx, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {Mw, Nx, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {Mw, Nx, 1});
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(Mw, Nx, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(Mw, Nx, K, 1));
    using EA = typename ElementA::DataType; using SA = typename ElementA::ScaleFactorType;
    using EB = typename ElementB::DataType; using SB = typename ElementB::ScaleFactorType;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {Mw, Nx, K, 1},
        { reinterpret_cast<EA const*>(W), stride_A, reinterpret_cast<EB const*>(X), stride_B,
          reinterpret_cast<SA const*>(SFW), layout_SFA, reinterpret_cast<SB const*>(SFX), layout_SFB },
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
using S0 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>>;   // weights 128 rows / CTA
using S1 = Variant<Shape<_256, _64,_256>, Shape<_2,_1,_1>>;   // 2-SM UMMA: 256 weight rows / CTA pair, activations multicast
using S2 = Variant<Shape<_128,_128,_256>, Shape<_1,_1,_1>>;   // wider activation tile (reference)
using S3 = Variant<Shape<_128, _64,_128>, Shape<_1,_1,_1>>;   // shorter k-tile, deeper pipeline
}  // namespace variants_swap

// Public entry keeps the (activation A, weight B, M, N, K) convention of cutlass_fp4_gemm_variant.
int cutlass_fp4_gemm_variant_swap(int idx, void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
  using namespace variants_swap;
  switch (idx) {
    case 0: return S0::run(B, SFB, A, SFA, D, N, M, K, alpha, beta, stream);
    case 1: return S1::run(B, SFB, A, SFA, D, N, M, K, alpha, beta, stream);
    case 2: return S2::run(B, SFB, A, SFA, D, N, M, K, alpha, beta, stream);
    case 3: return S3::run(B, SFB, A, SFA, D, N, M, K, alpha, beta, stream);
    default: return -99;
  }
}
}  // namespace fp4
}  // namespace flash_rt
