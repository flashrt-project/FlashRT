// ============================================================================
//  NVFP4 GEMMs for the SigLIP FFN. See header for the contract.
//
//  Both kernels reuse the proven block-scaled mainloop configuration of
//  cutlass_fp4_gemm_fp4out.cu (Sm100, 128x256x256, cluster 1x1x1); only
//  the fusion operations differ:
//    Up:   LinCombPerColBiasEltActBlockScaleFactor<GELU_taylor>
//    Down: LinCombPerColBias (fp16 out, beta = 1 residual source)
// ============================================================================
#include "gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
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

namespace siglip_ffn {

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
constexpr int SFVecSize = 16;

using UpTileShape = Shape<_128, _256, _256>;
using DownTileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

// ── Up: bias + tanh-GELU + fp4/SFA output ──────────────────────────────────
namespace up {

using ElementD = cutlass::float_e2m1_t;
using ElementC = ElementD;
using ElementSFD = cutlass::float_ue4m3_t;
constexpr int AlignmentD = 32;

using MmaTileShape = UpTileShape;

using FusionOperation =
    cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
        cutlass::epilogue::thread::GELU_taylor, SFVecSize,
        ElementD, ElementCompute, ElementSFD, cutlass::layout::RowMajor,
        cutlass::half_t, ElementC, ElementCompute>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, cutlass::layout::RowMajor, AlignmentD,
        ElementD, cutlass::layout::RowMajor, AlignmentD,
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

}  // namespace up

// ── Down: bias + residual source, fp16 output ──────────────────────────────
namespace down {

using ElementD = cutlass::half_t;
using ElementC = cutlass::half_t;
constexpr int AlignmentCD = 8;

using MmaTileShape = DownTileShape;

using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBias<
    ElementD, ElementCompute, cutlass::half_t, ElementC, ElementCompute>;

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

}  // namespace down

}  // namespace siglip_ffn

int cutlass_fp4_gemm_bias_gelu_fp4out(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_fp16,
    void* D_packed, void* D_SFD,
    int M, int N, int K,
    cudaStream_t stream) {
  using namespace siglip_ffn;
  using Gemm = up::Gemm;

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
       reinterpret_cast<up::ElementC const*>(D_packed), stride_C,
       reinterpret_cast<up::ElementD*>(D_packed), stride_D}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.thread.bias_ptr =
      reinterpret_cast<cutlass::half_t const*>(bias_fp16);
  static float* d_norm = nullptr;
  if (!d_norm) {
    if (cudaMalloc(&d_norm, sizeof(float)) != cudaSuccess) return -1;
    float h = 1.0f;
    cudaMemcpyAsync(d_norm, &h, sizeof(float), cudaMemcpyHostToDevice,
                    stream);
  }
  args.epilogue.thread.block_scale_factor_ptr =
      reinterpret_cast<up::ElementSFD*>(D_SFD);
  args.epilogue.thread.norm_constant_ptr = d_norm;

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
  return (st == cutlass::Status::kSuccess) ? 0
                                           : (static_cast<int>(st) | 0x30000);
}

int cutlass_fp4_gemm_bias_res_fp16(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_fp16,
    void const* C_fp16, void* D_fp16,
    int M, int N, int K,
    cudaStream_t stream) {
  using namespace siglip_ffn;
  using Gemm = down::Gemm;

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
       reinterpret_cast<down::ElementC const*>(C_fp16), stride_C,
       reinterpret_cast<down::ElementD*>(D_fp16), stride_D}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 1.0f;
  args.epilogue.thread.bias_ptr =
      reinterpret_cast<cutlass::half_t const*>(bias_fp16);

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
  return (st == cutlass::Status::kSuccess) ? 0
                                           : (static_cast<int>(st) | 0x30000);
}

}  // namespace fp4
}  // namespace flash_rt
