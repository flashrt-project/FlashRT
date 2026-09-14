// Shared template for the forked-mainloop NVFP4 GEMM variants (see
// cutlass_fp4_gemm_variants_earlyb.cu): ToEarlyB policy rewrite + Variant<>.
#pragma once
#undef CUTLASS_ENABLE_GDC_FOR_SM100
#include "fused_fp4/pdl.cuh"
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <type_traits>
#include "cutlass/kernel_hardware_info.hpp"
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
#include "gemm/fp4/sm100_gemm_seq_kernel.hpp"
#include "gemm/fp4/sm100_gemm_phase_cta.hpp"
#include "gemm/fp4/fp4_runtime_knobs.cuh"

namespace flash_rt {
namespace fp4 {
namespace variants_earlyb {
using namespace cute;

template <class T, bool Seq, int StagesOverride = 0, bool EarlyA = false, int EarlyStages = 0, bool TriggerInMma = false> struct ToEarlyB;
template <int S, int SP, int AP, class CS, class... Rest, int SO, bool EA, int ES, bool TM>
struct ToEarlyB<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<S, SP, AP, CS>, Rest...>, false, SO, EA, ES, TM> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledEarlyB<(SO ? SO : S), SP, AP, CS,
          cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<SP, AP>, EA, ES, TM>, Rest...>;
};
template <int S, int SP, int AP, class CS, class... Rest, int SO, bool EA, int ES, bool TM>
struct ToEarlyB<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<S, SP, AP, CS>, Rest...>, true, SO, EA, ES, TM> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledEarlyB<(SO ? SO : S), SP, AP, CS,
          cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100Seq<SP, AP>, EA, ES, TM>, Rest...>;
};

template <class MmaTile, class Cluster, bool Seq = false, int StagesOverride = 0, bool Swapped = false,
          int EarlyStages = 0, bool EarlyA = Swapped, bool TriggerInMma = false, class Sched = void>
struct Variant {
  using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;
  using ElementD   = cutlass::half_t;
  using ElementC   = cutlass::half_t;
  // Swapped: D is (N_out, M_act) column-major == the (M_act, N_out) row-major buffer the callers expect.
  using LayoutCTag = std::conditional_t<Swapped, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using LayoutDTag = std::conditional_t<Swapped, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
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
  using CollectiveMainloop = typename ToEarlyB<BaseMainloop, Seq, StagesOverride, EarlyA, EarlyStages, TriggerInMma>::type;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, Sched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  static int run(void const* A, void const* SFA, void const* B, void const* SFB,
                 void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
    return run_impl(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream, nullptr);
  }
  // Forked kernel + static scheduler only: appends the phase CTAs to the grid and
  // sizes the persistent grid to one tile per cluster (the static scheduler
  // strides by gridDim, so no cluster may own more than one tile).
  static int run_phase(void const* A, void const* SFA, void const* B, void const* SFB,
                       void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream,
                       PhaseCtaParams const& phase) {
    static_assert(Seq && !std::is_void_v<Sched>, "run_phase needs the forked kernel with the static scheduler");
    return run_impl(A, SFA, B, SFB, D, M, N, K, alpha, beta, stream, &phase);
  }
  static int run_impl(void const* A, void const* SFA, void const* B, void const* SFB,
                      void* D, int M, int N, int K, float alpha, float beta, cudaStream_t stream,
                      PhaseCtaParams const* phase) {
    if constexpr (Swapped) { std::swap(A, B); std::swap(SFA, SFB); std::swap(M, N); }
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
    args.mainloop.weight_evict_first = get_weight_evict_first();
    if constexpr (!std::is_void_v<Sched>) {
      // The static persistent scheduler sizes its grid from hw_info (the CLC scheduler queries the device itself).
      static int sm_count = 0;
      if (sm_count == 0) {
        sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
        // Clusters larger than 2 cannot fill every SM (Thor: 20 SMs in GPCs of 8+6+6); size the
        // persistent grid from the number of co-resident clusters instead of the SM count.
        constexpr int cs = size(Cluster{});
        if constexpr (cs > 2) {
          int max_clusters = 0;
          cudaLaunchConfig_t cfg = {};
          cfg.gridDim = dim3(cs * 64); cfg.blockDim = dim3(Gemm::GemmKernel::MaxThreadsPerBlock);
          cfg.dynamicSmemBytes = Gemm::GemmKernel::SharedStorageSize;
          cudaLaunchAttribute at; at.id = cudaLaunchAttributeClusterDimension;
          at.val.clusterDim.x = size<0>(Cluster{}); at.val.clusterDim.y = size<1>(Cluster{}); at.val.clusterDim.z = 1;
          cfg.attrs = &at; cfg.numAttrs = 1;
          cudaFuncSetAttribute(cutlass::device_kernel<typename Gemm::GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, Gemm::GemmKernel::SharedStorageSize);
          if (cudaOccupancyMaxActiveClusters(&max_clusters, cutlass::device_kernel<typename Gemm::GemmKernel>, &cfg) == cudaSuccess && max_clusters > 0)
            sm_count = max_clusters * cs;
        }
      }
      args.hw_info.sm_count = sm_count;
    }
    if constexpr (Seq && !std::is_void_v<Sched>) {
      if (phase != nullptr) {
        args.phase = *phase;
        args.hw_info.sm_count = phase->num_gemm_ctas;   // one tile per cluster
      }
      if (phase != nullptr && std::getenv("FLASHRT_PHASE_DEBUG")) {
        auto params = GemmKernel::to_underlying_arguments(args, nullptr);
        const dim3 g = GemmKernel::get_grid_shape(params);
        const dim3 gg = GemmKernel::get_grid_shape_gemm(params);
        std::fprintf(stderr, "[phase] grid (%u,%u,%u) gemm grid (%u,%u,%u) num_gemm_ctas %d num_phase_ctas %d kind %d dbg %d smem %d\n",
                     g.x, g.y, g.z, gg.x, gg.y, gg.z, phase->num_gemm_ctas, phase->num_phase_ctas, phase->kind, phase->dbg, GemmKernel::SharedStorageSize);
      }
    }
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
}  // namespace variants_earlyb
}  // namespace fp4
}  // namespace flash_rt
