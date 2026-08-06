// ============================================================================
//  FlashRT — parametric NVFP4 GEMM kernel variants for small-M tuning.
//
//  Allows instantiating multiple (MmaTileShape, ClusterShape) configs at
//  compile time. Each variant exported as a separate extern "C"-style runner
//  that we benchmark from Python to pick the best.
// ============================================================================

#include "cutlass_fp4_gemm.cuh"

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
namespace variants {

using namespace cute;

// ── Parametric variant ─────────────────────────────────────────────────────
// Template on MmaTileShape + ClusterShape.
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

  static int run(void const* A, void const* SFA, void const* B, void const* SFB,
                 void* D, int M, int N, int K, float alpha, float beta,
                 cudaStream_t stream) {
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
        { reinterpret_cast<EA const*>(A), stride_A,
          reinterpret_cast<EB const*>(B), stride_B,
          reinterpret_cast<SA const*>(SFA), layout_SFA,
          reinterpret_cast<SB const*>(SFB), layout_SFB },
        { {alpha, beta},
          reinterpret_cast<ElementC*>(D), stride_C,
          reinterpret_cast<ElementD*>(D), stride_D }
    };
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

// ── Concrete variants we want to benchmark ───────────────────────────────
// v0: baseline (production kernel)
using V0 = Variant<Shape<_128,_128,_128>, Shape<_2,_1,_1>>;
// v1: wider N tile, small cluster
using V1 = Variant<Shape<_128,_256,_128>, Shape<_2,_1,_1>>;
// v2: ex72a default cluster
using V2 = Variant<Shape<_128,_128,_128>, Shape<_2,_4,_1>>;
// v3: wide N + 2x N cluster
using V3 = Variant<Shape<_128,_256,_128>, Shape<_2,_2,_1>>;
// v4: BEST from first sweep — cluster 1x1x1
using V4 = Variant<Shape<_128,_128,_128>, Shape<_1,_1,_1>>;
// v5: narrower N tile for small-N proj (QKV=2560, O=2048)
using V5 = Variant<Shape<_128, _64,_128>, Shape<_1,_1,_1>>;
// v6: wider N with min cluster (for Gate+Up N=16384)
using V6 = Variant<Shape<_128,_256,_128>, Shape<_1,_1,_1>>;
// v7: K tile 256 (fewer main-loop iterations on K=2048)
using V7 = Variant<Shape<_128,_128,_256>, Shape<_1,_1,_1>>;
// v8: wider K, wider N
using V8 = Variant<Shape<_128,_256,_256>, Shape<_1,_1,_1>>;
// v9: cluster x2 in M (exploit M parallelism despite small M)
using V9 = Variant<Shape<_128,_128,_128>, Shape<_2,_1,_1>>;  // same as V0 → sanity
// Note: MMA instruction forces tile_M=128 minimum for NVFP4 SM100 block-scaled.
// Cannot reduce tile_M further without switching MMA primitive (GEMV path).

// v10: narrow N + wide K → fewer mainloop iters, same block count as v5
using V10 = Variant<Shape<_128, _64,_256>, Shape<_1,_1,_1>>;

}  // namespace variants

// Dispatch by index (exposed via pybind).
int cutlass_fp4_gemm_variant(int idx,
    void const* A, void const* SFA, void const* B, void const* SFB,
    void* D, int M, int N, int K, float alpha, float beta,
    cudaStream_t stream) {
  using namespace variants;
  switch (idx) {
    case 0: return V0::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 1: return V1::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 2: return V2::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 3: return V3::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 4: return V4::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 5: return V5::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 6: return V6::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 7: return V7::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 8: return V8::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 9: return V9::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    case 10: return V10::run(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
    default: return -99;
  }
}

const char* cutlass_fp4_gemm_variant_name(int idx) {
  switch (idx) {
    case 0: return "tile128x128x128 cluster2x1x1 (old baseline)";
    case 1: return "tile128x256x128 cluster2x1x1";
    case 2: return "tile128x128x128 cluster2x4x1 (ex72a default)";
    case 3: return "tile128x256x128 cluster2x2x1";
    case 4: return "tile128x128x128 cluster1x1x1";
    case 5: return "tile128x64x128  cluster1x1x1 (narrow N)";
    case 6: return "tile128x256x128 cluster1x1x1 (wide N)";
    case 7: return "tile128x128x256 cluster1x1x1 (wide K)";
    case 8: return "tile128x256x256 cluster1x1x1 (wide N+K)";
    case 9: return "tile128x128x128 cluster2x1x1 (sanity)";
    case 10: return "tile128x64x256  cluster1x1x1 (narrow N + wide K)";
    default: return "<invalid>";
  }
}

int cutlass_fp4_gemm_num_variants() { return 11; }

}  // namespace fp4
}  // namespace flash_rt
