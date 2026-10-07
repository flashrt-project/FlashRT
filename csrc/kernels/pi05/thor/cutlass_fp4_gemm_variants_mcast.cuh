// Shared template for the encoder-shape NVFP4 GEMM variants: optional operand
// swap (784 activation rows on the N axis, D column-major) and wider TMA
// multicast clusters on the 2-SM 256x256 tile. See the .cu files for the
// rationale of each instantiation.
#pragma once
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include "kernels/pi05/thor/pdl.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace flash_rt {
namespace fp4 {
namespace variants_mcast {
using namespace cute;

// Swapped = true computes D^T = W * X^T with D stored column-major (the row-major
// (M, N) output the callers expect); Swapped = false is the plain D = X * W^T.
template <class MmaTile, class Cluster, bool Swapped, class Sched = void,
          class EpiTile = cutlass::epilogue::collective::EpilogueTileAuto>
struct Variant {
  using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // K contiguous
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // K contiguous
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;
  using ElementD   = cutlass::half_t;
  using ElementC   = cutlass::half_t;
  using LayoutCTag = std::conditional_t<Swapped, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using LayoutDTag = LayoutCTag;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ElementAccumulator = float;
  using ArchTag            = cutlass::arch::Sm100;
  using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
  static constexpr bool kStreamK = std::is_same_v<Sched, cutlass::gemm::StreamKScheduler>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, Cluster,
      EpiTile,
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
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, Sched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  // Stream-K needs a reduction workspace; cache it per problem so the launch is graph-capturable.
  static void* workspace_for(size_t bytes, int M, int N, int K) {
    static std::map<std::tuple<int,int,int>, std::pair<void*, size_t>> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    auto key = std::make_tuple(M, N, K);
    auto it = cache.find(key);
    if (it != cache.end() && it->second.second >= bytes) return it->second.first;
    void* p = nullptr;
    if (bytes > 0 && cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    cache[key] = {p, bytes};
    return p;
  }

  // Public (activation A, weight B, M, N, K) convention; the swap happens here.
  static int run(void const* A, void const* SFA, void const* B, void const* SFB,
                 void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
    if constexpr (Swapped) return run_impl(B, SFB, A, SFA, D, N, M, K, alpha, beta, stream);
    else return run_impl(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream);
  }
  static int run_impl(void const* W, void const* SFW, void const* X, void const* SFX,
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
    if constexpr (kStreamK) {
      using DecompositionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
      using ReductionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;
      args.scheduler.splits = 1;
      args.scheduler.decomposition_mode = DecompositionMode::StreamK;
      args.scheduler.reduction_mode = ReductionMode::Deterministic;
    }
    Gemm gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
    size_t ws_sz = Gemm::get_workspace_size(args);
    void* ws = nullptr;
    if constexpr (kStreamK) {
      ws = workspace_for(ws_sz, Mw, Nx, K);
      if (ws_sz > 0 && ws == nullptr) return -1;
    } else {
      if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
    }
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) { if (!kStreamK && ws) cudaFree(ws); return static_cast<int>(st) | 0x20000; }
    st = gemm.run(stream, nullptr, flash_rt::fp4::pdl_launch());
    if (!kStreamK && ws) cudaFree(ws);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
  }
};
}  // namespace variants_mcast
}  // namespace fp4
}  // namespace flash_rt
