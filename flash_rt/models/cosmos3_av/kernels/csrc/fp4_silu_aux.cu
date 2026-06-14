#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <unordered_map>

namespace cutlass::epilogue::fusion {

template<int SFVecSize_, class ElementOutput_, class ElementCompute_,
  class ElementBlockScaleFactor_, class ElementAux_ = cutlass::bfloat16_t,
  class GmemLayoutTagAux_ = cutlass::layout::RowMajor, int AlignmentAux_ = 8,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct LinCombSiLuAuxMulBSF : FusionOperation {
  using ElementOutput = ElementOutput_; using ElementCompute = ElementCompute_;
  using ElementSource = ElementOutput_; using ElementScalar = ElementCompute_;
  using ElementAux = ElementAux_; using ElementBlockScaleFactor = ElementBlockScaleFactor_;
  using GmemLayoutTagAux = GmemLayoutTagAux_;
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;
  static constexpr int SFVecSize = SFVecSize_;
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr FloatRoundStyle RoundStyle = RoundStyle_;
  static constexpr bool IsSourceSupported = false;
  static constexpr bool IsAuxOutSupported = false;
  static constexpr bool IsAuxInSupported = true;
  static constexpr bool IsBlockScaleSupported = true;
};

template<int Stages, int SFVecSize, class EpilogueTile, class CtaTileShapeMNK,
  int FragmentSize, class ElementOutput,
  class ElementCompute, class ElementBlockScaleFactor, class ElementAux,
  class StrideAuxMNL, class SmemLayoutAtomAux, class CopyOpS2RAux,
  int AlignmentAux, FloatRoundStyle RoundStyle>
using Sm120SiLuAuxMulRowBSF =
  Sm90EVT<
    Sm120BlockScaleFactorRowStore<SFVecSize, EpilogueTile, CtaTileShapeMNK,
        FragmentSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>,
    Sm90EVT<
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
      Sm90EVT<
        Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute, RoundStyle>,
        Sm90AuxLoad<Stages, EpilogueTile, ElementAux, StrideAuxMNL,
            SmemLayoutAtomAux, CopyOpS2RAux, AlignmentAux, false>>,
      Sm90AccFetch>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
  bool DelayTmaStore, int SFVecSize, class ElementOutput, class ElementCompute,
  class ElementBlockScaleFactor, class ElementAux, class GmemLayoutTagAux,
  int AlignmentAux, FloatRoundStyle RoundStyle, class CtaTileShapeMNK,
  class EpilogueTile, class SmemLayoutAtomAux, class CopyOpS2RAux>
struct FusionCallbacks<
    epilogue::Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombSiLuAuxMulBSF<SFVecSize, ElementOutput, ElementCompute,
        ElementBlockScaleFactor, ElementAux, GmemLayoutTagAux, AlignmentAux,
        cutlass::layout::RowMajor, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile, SmemLayoutAtomAux, CopyOpS2RAux>
  : Sm120SiLuAuxMulRowBSF<StagesC, SFVecSize, EpilogueTile, CtaTileShapeMNK, FragmentSize,
        typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
        ElementCompute, ElementBlockScaleFactor, ElementAux,
        cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
        SmemLayoutAtomAux, CopyOpS2RAux, AlignmentAux, RoundStyle>
{
  using Impl = Sm120SiLuAuxMulRowBSF<StagesC, SFVecSize, EpilogueTile, CtaTileShapeMNK, FragmentSize,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
      ElementCompute, ElementBlockScaleFactor, ElementAux,
      cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtomAux, CopyOpS2RAux, AlignmentAux, RoundStyle>;
  using Operation = fusion::LinCombSiLuAuxMulBSF<SFVecSize, ElementOutput,
      ElementCompute, ElementBlockScaleFactor, ElementAux, GmemLayoutTagAux,
      AlignmentAux, cutlass::layout::RowMajor, RoundStyle>;
  using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
  struct Arguments {
    ElementBlockScaleFactor* block_scale_factor_ptr = nullptr;
    using StrideNormConst = Stride<_0, _0, int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};
    ElementAux const* aux_ptr = nullptr;
    ElementAux aux_null = ElementAux(0);
    StrideAux dAux = {};
    operator typename Impl::Arguments() const {
      // Impl = Sm90EVT<BSFRowStore, Sm90EVT<mult, Sm90EVT<SiLu,AuxLoad>, AccFetch>>
      return { { { {aux_ptr, aux_null, dAux}, {} }, {}, {} },
               { block_scale_factor_ptr, norm_constant_ptr, dNormConst } };
    }
  };
  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace silu_aux_jit {
using namespace cute;
using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementD = cutlass::float_e2m1_t;          // FP4 packed out
using ElementC = cutlass::bfloat16_t;
using ElementSFD = cutlass::float_ue4m3_t;   // NVFP4 output SF (matches Motus working kernel)
using ElementSF = cutlass::float_ue4m3_t;     // INPUT SF (A/B nvfp4) ue4m3
using ElementAux = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 32;
using TileShape = Shape<_128, _128, _256>;       // K=256 (matches Motus working fp4-out)
using ClusterShape = Shape<_1, _1, _1>;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
constexpr int OutputSFVectorSize = 16;

using FusionOperation = cutlass::epilogue::fusion::LinCombSiLuAuxMulBSF<
    OutputSFVectorSize, ElementD, ElementCompute, ElementSFD,
    ElementAux, cutlass::layout::RowMajor, 8,
    cutlass::layout::RowMajor, cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairA, LayoutA, AlignmentA, ElementPairB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideAux = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;
}  // namespace silu_aux_jit

static float* _norm_buf(float v) {
  // one device float per distinct value (cache so graph capture sees a stable ptr)
  static std::unordered_map<float, float*>* m = new std::unordered_map<float, float*>();
  auto it = m->find(v);
  if (it != m->end()) return it->second;
  float* p = nullptr; cudaMalloc(&p, sizeof(float)); cudaMemcpy(p, &v, 4, cudaMemcpyHostToDevice);
  (*m)[v] = p; return p;
}

int fp4_silu_aux(int64_t A_packed, int64_t SFA, int64_t B_packed, int64_t SFB,
                 int64_t aux_gate, int64_t D_packed, int64_t D_SFD,
                 int M, int N, int K, double norm_const, int64_t stream) {
  using namespace silu_aux_jit;
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SC = typename Gemm::GemmKernel::StrideC;
  using SD = typename Gemm::GemmKernel::StrideD;
  auto sA = cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(SC{}, cute::make_shape(M, N, 1));
  auto sD = cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1));
  auto sAux = cutlass::make_cute_packed_stride(StrideAux{}, cute::make_shape(M, N, 1));
  auto mnkl = cute::make_shape(M, N, K, 1);
  auto lSFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(mnkl);
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(mnkl);
  using AEA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using AEB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      { reinterpret_cast<AEA const*>(A_packed), sA,
        reinterpret_cast<AEB const*>(B_packed), sB,
        reinterpret_cast<ElementSF const*>(SFA), lSFA,
        reinterpret_cast<ElementSF const*>(SFB), lSFB },
      { {}, nullptr, sC, reinterpret_cast<ElementD*>(D_packed), sD } };
  args.epilogue.thread.block_scale_factor_ptr = reinterpret_cast<ElementSFD*>(D_SFD);
  args.epilogue.thread.norm_constant_ptr = _norm_buf((float)norm_const);
  args.epilogue.thread.aux_ptr = reinterpret_cast<ElementAux const*>(aux_gate);
  args.epilogue.thread.dAux = sAux;
  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return int(st) | 0x10000;
  static void* ws = nullptr; static size_t wsz = 0;
  size_t need = Gemm::get_workspace_size(args);
  if (need > wsz) { if (ws) cudaFree(ws); cudaMalloc(&ws, need); wsz = need; }
  st = gemm.initialize(args, ws, (cudaStream_t)stream);
  if (st != cutlass::Status::kSuccess) return int(st) | 0x20000;
  st = gemm.run((cudaStream_t)stream);
  return st == cutlass::Status::kSuccess ? 0 : (int(st) | 0x30000);
}
