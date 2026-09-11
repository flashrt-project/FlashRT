// ============================================================================
//  FlashRT — NVFP4 GEMM with fp32 per-column bias and fp16 output
//  (SM100/SM110). See header for the contract.
// ============================================================================

#include "gemm/fp4/cutlass_fp4_gemm_bias_f32b_f16out_sm100.cuh"

#include "cutlass/cutlass.h"
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

#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace fp4 {

namespace bias_f16out {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ElementD = cutlass::half_t;
using ElementC = cutlass::half_t;
constexpr int AlignmentCD = 8;

using MmaTileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

// per-shape CUTLASS workspace cache (capture-safe: growth happens during
// the uncaptured warmup evaluation)
struct ws_key {
    int M, N, K;
    bool operator==(const ws_key & o) const { return M == o.M && N == o.N && K == o.K; }
};
struct ws_key_hash {
    size_t operator()(const ws_key & k) const noexcept {
        return (size_t) k.M * 1315423911u ^ (size_t) k.N * 2654435761u ^ (size_t) k.K;
    }
};
inline void * get_ws(int M, int N, int K, size_t needed) {
    static std::unordered_map<ws_key, std::pair<void *, size_t>, ws_key_hash> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto & e = cache[ws_key{M, N, K}];
    if (e.second < needed) {
        if (e.first) { cudaFree(e.first); }
        cudaMalloc(&e.first, needed);
        e.second = needed;
    }
    return e.first;
}

using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBias<
    ElementD, ElementCompute, float, ElementC, ElementCompute>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, cutlass::layout::RowMajor, AlignmentCD,
        ElementD, cutlass::layout::RowMajor, AlignmentCD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace bias_f16out

int gemm_bias_f16out(
    const void * A_packed, const void * SFA,
    const void * B_packed, const void * SFB,
    const void * bias_f32, void * D_f16,
    int M, int N, int K,
    cudaStream_t stream) {
  using namespace bias_f16out;

  auto stride_A = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
  using Cfg =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  using EA = typename ElementA::DataType;
  using SA = typename ElementA::ScaleFactorType;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA const*>(A_packed), stride_A,
       reinterpret_cast<EA const*>(B_packed), stride_B,
       reinterpret_cast<SA const*>(SFA), layout_SFA,
       reinterpret_cast<SA const*>(SFB), layout_SFB},
      {{},
       reinterpret_cast<ElementC const*>(D_f16), stride_C,
       reinterpret_cast<ElementD*>(D_f16), stride_D}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.thread.bias_ptr = reinterpret_cast<float const*>(bias_f32);

  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
  size_t ws_sz = Gemm::get_workspace_size(args);
  void* ws = ws_sz > 0 ? get_ws(M, N, K, ws_sz) : nullptr;
  st = gemm.initialize(args, ws, stream);
  if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x20000;
  st = gemm.run(stream);
  return (st == cutlass::Status::kSuccess) ? 0
                                           : (static_cast<int>(st) | 0x30000);
}

}  // namespace fp4
}  // namespace flash_rt
