// Persistent dependent-GEMM sequence for the Thor decoder (NVFP4, operand-swapped
// 2-SM tile, weights streamed ahead of the barrier). Host side of
// sm100_gemm_seq_persistent_kernel.hpp.
#undef CUTLASS_ENABLE_GDC_FOR_SM100
#include "gemm/fp4/cutlass_fp4_gemm_seq_sm100.cuh"
#include "fused_fp4/pdl.cuh"
#include <utility>
#include <type_traits>
#include <cuda_fp16.h>
#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/device_kernel.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "gemm/fp4/sm100_gelu_mul_blockscale_visitor.hpp"
#include "gemm/fp4/sm100_blockscaled_mma_earlyb.hpp"
#include "gemm/fp4/sm100_gemm_seq_persistent_kernel.hpp"

namespace flash_rt {
namespace fp4 {
namespace seq {
using namespace cute;

template <class T, int ES> struct ToEarlyA;
template <int S, int SP, int AP, class CS, class... Rest, int ES>
struct ToEarlyA<cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<S, SP, AP, CS>, Rest...>, ES> {
  using type = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaledEarlyB<S, SP, AP, CS,
          cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100<SP, AP>, /*EarlyA*/ true, ES, /*TriggerInMma*/ false>, Rest...>;
};

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // weights (N_out x K)
using LayoutATag = cutlass::layout::RowMajor;
using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // activations (M_act x K)
using LayoutBTag = cutlass::layout::ColumnMajor;
using ElementD   = cutlass::half_t;
using ElementC   = void;                           // no source operand
using LayoutCTag = cutlass::layout::ColumnMajor;   // (N_out, M_act) column-major == (M_act, N_out) row-major
using LayoutDTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 32, AlignmentB = 32;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = AlignmentD;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTile = Shape<_256, _64, _256>;
using Cluster = Shape<_2, _1, _1>;
using ElementSFD = cutlass::float_ue4m3_t;
// One epilogue type for every problem of a sequence: fp16 pass-through store, or the GeGLU
// column compact store (gate_up) selected per problem at runtime.
using FusionOperation = cutlass::epilogue::fusion::GeluMulCompactColOrPassBlockScaleFactor<
    16, ElementD, ElementAccumulator, ElementSFD, cutlass::layout::RowMajor, ElementD>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;
using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB, ElementAccumulator, MmaTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
template <int ES> using MainloopES = typename ToEarlyA<BaseMainloop, ES>::type;
template <int ES> using KernelES = cutlass::gemm::kernel::GemmSeqPersistent<MainloopES<ES>, CollectiveEpilogue, kSeqMaxProblems>;
using CollectiveMainloop = MainloopES<3>;
using StrideA = typename CollectiveMainloop::StrideA;
using StrideB = typename CollectiveMainloop::StrideB;
using StrideC = typename CollectiveEpilogue::StrideC;
using StrideD = typename CollectiveEpilogue::StrideD;
using Cfg = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

template <class Kernel>
static int run_seq(int n, const SeqGemmDesc* d, int* counter, cudaStream_t stream, int* grid_out, int flags, int early_first, int early_rest) {
  using CollectiveMainloopK = typename Kernel::CollectiveMainloop;

  if (n < 1 || n > kSeqMaxProblems || !counter) return -1;
  static int sm_count = 0;
  if (sm_count == 0) sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;
  typename Kernel::Params params;
  params.num_problems = n;
  params.counter = counter;
  params.flags = flags;
  params.early_first = early_first;
  params.early_rest = early_rest;
  dim3 grid(1, 1, 1);
  for (int i = 0; i < n; ++i) {
    const int Mw = d[i].N, Nx = d[i].M, K = d[i].K;
    if ((reinterpret_cast<uintptr_t>(d[i].A) & 31) || (reinterpret_cast<uintptr_t>(d[i].B) & 31) || (K % 256) || (Mw % 256)) return -2;
    auto shape = make_shape(Mw, Nx, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {Mw, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {Nx, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {Mw, Nx, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {Mw, Nx, 1});
    auto layout_SFA = Cfg::tile_atom_to_shape_SFA(shape);
    auto layout_SFB = Cfg::tile_atom_to_shape_SFB(shape);
    using EA = typename ElementA::DataType; using SA = typename ElementA::ScaleFactorType;
    using EB = typename ElementB::DataType; using SB = typename ElementB::ScaleFactorType;
    typename Kernel::MainloopArguments margs{
        reinterpret_cast<EA const*>(d[i].B), stride_A, reinterpret_cast<EB const*>(d[i].A), stride_B,
        reinterpret_cast<SA const*>(d[i].SFB), layout_SFA, reinterpret_cast<SB const*>(d[i].SFA), layout_SFB };
    typename Kernel::EpilogueArguments eargs{
        {}, nullptr, stride_C, reinterpret_cast<ElementD*>(d[i].D), stride_D };
    eargs.thread.alpha = d[i].alpha;
    eargs.thread.beta = 0.f;
    if (d[i].geglu) {
      if ((Mw % 32) || !d[i].compact_packed || !d[i].compact_sfa) return -10;
      eargs.thread.mode = 1;
      eargs.thread.compact_ptr = static_cast<uint8_t*>(d[i].compact_packed);
      eargs.thread.compact_sf_ptr = static_cast<uint8_t*>(d[i].compact_sfa);
    }
    if (!CollectiveMainloopK::can_implement(shape, margs) || !CollectiveEpilogue::can_implement(shape, eargs)) return -3;
    params.prob[i].shape = shape;
    params.prob[i].mainloop = CollectiveMainloopK::to_underlying_arguments(shape, margs, nullptr, hw_info);
    params.prob[i].epilogue = CollectiveEpilogue::to_underlying_arguments(shape, eargs, nullptr);
      typename Kernel::TileSchedulerArguments sargs{};
    params.prob[i].scheduler = Kernel::TileScheduler::to_underlying_arguments(
        shape, typename Kernel::TileShape{}, typename Kernel::AtomThrShapeMNK{}, typename Kernel::ClusterShape{}, hw_info, sargs, nullptr);
    params.prob[i].phase = d[i].phase;
    if (d[i].phase != 0) {
      if (d[i].phase != 1 || d[i].ph_D != 1024 || d[i].ph_S < 1 || d[i].ph_S > 32 || d[i].ph_S != Nx) return -6;
      if (i + 1 == n) return -7;   // a phase must feed a following problem
      auto& pa = params.prob[i].phase_args;
      pa.x = static_cast<const __half*>(d[i].D);
      pa.prev_gate = static_cast<const __half*>(d[i].ph_prev_gate);
      pa.residual = static_cast<__half*>(d[i].ph_residual);
      pa.style = static_cast<const __half*>(d[i].ph_style);
      pa.packed = static_cast<uint8_t*>(d[i].ph_packed);
      pa.sfa = static_cast<uint8_t*>(d[i].ph_sfa);
      pa.gate = static_cast<__half*>(d[i].ph_gate);
      pa.S = d[i].ph_S; pa.D = d[i].ph_D;
      if (!pa.prev_gate || !pa.residual || !pa.style || !pa.packed || !pa.sfa || !pa.gate) return -8;
    }
    dim3 g = Kernel::TileScheduler::get_grid_shape(params.prob[i].scheduler, shape, typename Kernel::TileShape{},
        typename Kernel::AtomThrShapeMNK{}, typename Kernel::ClusterShape{}, hw_info);
    if (g.x * g.y * g.z > grid.x * grid.y * grid.z) grid = g;
  }
  params.num_ctas = static_cast<int>(grid.x * grid.y * grid.z);
  if (grid_out) { grid_out[0] = grid.x; grid_out[1] = grid.y; grid_out[2] = grid.z; }
  static bool attr_set = false;   // per Kernel instantiation (template static)
  if (!attr_set) {
    if (cudaFuncSetAttribute(cutlass::device_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SharedStorageSize) != cudaSuccess) return -4;
    attr_set = true;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(Kernel::MaxThreadsPerBlock, 1, 1);
  cfg.dynamicSmemBytes = Kernel::SharedStorageSize;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  int na = 0;
  attrs[na].id = cudaLaunchAttributeClusterDimension;
  attrs[na].val.clusterDim.x = size<0>(Cluster{}); attrs[na].val.clusterDim.y = size<1>(Cluster{}); attrs[na].val.clusterDim.z = size<2>(Cluster{});
  ++na;
  if (pdl_launch()) {
    attrs[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attrs; cfg.numAttrs = na;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, cutlass::device_kernel<Kernel>, params);
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace seq

int cutlass_fp4_gemm_seq_run(int n, const SeqGemmDesc* d, int* counter, cudaStream_t stream, int* grid_out, int variant, int flags) {
  using namespace seq;
  // One instantiation; the early-stream depth is a runtime choice per tile position.
  switch (variant) {
    case 0: return run_seq<KernelES<3>>(n, d, counter, stream, grid_out, flags, 3, 3);   // as the standalone variant 28
    case 1: return run_seq<KernelES<3>>(n, d, counter, stream, grid_out, flags, 8, 0);   // deep across the barrier, interleaved inside
    case 2: return run_seq<KernelES<3>>(n, d, counter, stream, grid_out, flags, 5, 0);
    case 3: return run_seq<KernelES<3>>(n, d, counter, stream, grid_out, flags, 8, 3);
    case 4: return run_seq<KernelES<3>>(n, d, counter, stream, grid_out, flags, 3, 0);
    default: return -5;
  }
}

}  // namespace fp4
}  // namespace flash_rt
