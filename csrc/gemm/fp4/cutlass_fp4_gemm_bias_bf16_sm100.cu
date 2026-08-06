// ============================================================================
//  NVFP4 GEMMs with bf16 fused-bias epilogues. See header for the contract.
//
//  Three fusions over one shared skinny-M mainloop config (Sm100,
//  tile 128x64x256, cluster 1x1x1 — the narrow-N + wide-K shape that wins
//  at small M where the GEMM is weight-bandwidth-bound):
//    bias:      LinCombPerColBias  (bf16 out, beta = 0)
//    bias+res:  LinCombPerColBias  (bf16 out, beta = 1, C = residual)
//    bias+gelu: LinCombPerColBiasEltActBlockScaleFactor<GELU_taylor>
//               (fp4 + SFA out for the following NVFP4 GEMM)
// ============================================================================
#include "gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.cuh"

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

namespace bias_bf16_gemm {

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

using MmaTileShape = Shape<_128, _64, _256>;
using ClusterShape = Shape<_1, _1, _1>;

template <class FusionOp, class ElemC, class ElemD, int AlignCD>
struct BiasGemm {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator,
          ElemC, cutlass::layout::RowMajor, AlignCD,
          ElemD, cutlass::layout::RowMajor, AlignCD,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          FusionOp>::CollectiveOp;

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
};

// ── bias / bias+res: bf16 out ──────────────────────────────────────────────
using ElementCD = cutlass::bfloat16_t;
using FusionBias = cutlass::epilogue::fusion::LinCombPerColBias<
    ElementCD, ElementCompute, ElementCD, ElementCD, ElementCompute>;
using GemmBias = BiasGemm<FusionBias, ElementCD, ElementCD, 8>::Gemm;

// ── bias+gelu: fp4 + SFA out ───────────────────────────────────────────────
using ElementDQ = cutlass::float_e2m1_t;
using ElementSFD = cutlass::float_ue4m3_t;
using FusionGelu =
    cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
        cutlass::epilogue::thread::GELU_taylor, SFVecSize,
        ElementDQ, ElementCompute, ElementSFD, cutlass::layout::RowMajor,
        ElementCD, ElementDQ, ElementCompute>;
using GemmGelu = BiasGemm<FusionGelu, ElementDQ, ElementDQ, 32>::Gemm;

template <class Gemm>
static int run_gemm(typename Gemm::Arguments& args, cudaStream_t stream) {
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

template <class Gemm, class ElemC, class ElemD>
static typename Gemm::Arguments make_args(
    void const* A, void const* SFA, void const* B, void const* SFB,
    void const* C, void* D, int M, int N, int K) {
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

  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA const*>(A), stride_A,
       reinterpret_cast<EA const*>(B), stride_B,
       reinterpret_cast<SA const*>(SFA), layout_SFA,
       reinterpret_cast<SA const*>(SFB), layout_SFB},
      {{},
       reinterpret_cast<ElemC const*>(C), stride_C,
       reinterpret_cast<ElemD*>(D), stride_D}};
}

}  // namespace bias_bf16_gemm

int cutlass_fp4_gemm_bias_bf16(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_bf16,
    void* D_bf16,
    int M, int N, int K, cudaStream_t stream) {
  using namespace bias_bf16_gemm;
  auto args = make_args<GemmBias, ElementCD, ElementCD>(
      A_packed, SFA, B_packed, SFB, D_bf16, D_bf16, M, N, K);
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.thread.bias_ptr =
      reinterpret_cast<ElementCD const*>(bias_bf16);
  return run_gemm<GemmBias>(args, stream);
}

int cutlass_fp4_gemm_bias_res_bf16(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_bf16,
    void const* C_bf16, void* D_bf16,
    int M, int N, int K, cudaStream_t stream) {
  using namespace bias_bf16_gemm;
  auto args = make_args<GemmBias, ElementCD, ElementCD>(
      A_packed, SFA, B_packed, SFB, C_bf16, D_bf16, M, N, K);
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 1.0f;
  args.epilogue.thread.bias_ptr =
      reinterpret_cast<ElementCD const*>(bias_bf16);
  return run_gemm<GemmBias>(args, stream);
}

int cutlass_fp4_gemm_bias_gelu_fp4out_bf16(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* bias_bf16,
    void* D_packed, void* D_SFD,
    int M, int N, int K, cudaStream_t stream) {
  using namespace bias_bf16_gemm;
  auto args = make_args<GemmGelu, ElementDQ, ElementDQ>(
      A_packed, SFA, B_packed, SFB, D_packed, D_packed, M, N, K);
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.thread.bias_ptr =
      reinterpret_cast<ElementCD const*>(bias_bf16);
  // The block-scale epilogue divides by a device-resident norm constant;
  // 1.0 keeps the native per-16 dynamic scale. Allocated once, first call
  // must happen before any CUDA Graph capture (warmup covers this).
  static float* d_norm = nullptr;
  if (!d_norm) {
    if (cudaMalloc(&d_norm, sizeof(float)) != cudaSuccess) return -1;
    float h = 1.0f;
    cudaMemcpyAsync(d_norm, &h, sizeof(float), cudaMemcpyHostToDevice,
                    stream);
  }
  args.epilogue.thread.block_scale_factor_ptr =
      reinterpret_cast<bias_bf16_gemm::ElementSFD*>(D_SFD);
  args.epilogue.thread.norm_constant_ptr = d_norm;
  return run_gemm<GemmGelu>(args, stream);
}

}  // namespace fp4
}  // namespace flash_rt
