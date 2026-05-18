// ============================================================================
// FlashRT — MK-1 Stage B SMEM-only EVT visitors + FusionOperation tags
// + FusionCallbacks specializations.
//
// Two visitor classes:
//   Sm100SmemAuxStore  — phase 1 (gate) post-activation fragment R2S-stored
//                        to an external full-tile SMEM buffer (smem_gate),
//                        sized by the kernel's SharedStorage.
//   Sm100SmemAuxLoad   — phase 2 (up) epilogue previsit S2R-loads from the
//                        same SMEM buffer.
//
// Two FusionOperation tags (FlashRT-specific):
//   LinCombEltActSmemAuxStore  — `D = act(alpha*acc + beta*C)`, AND
//                                aux = D captured into SMEM.
//   LinCombDeEltActSmemAuxLoad — `D = activation(beta*C + alpha*acc, aux)`
//                                with aux loaded from SMEM.
//
// FusionCallbacks<Sm90TmaWarpSpecialized<...>, OurOp, ...> specializations
// stitch these into the standard CollectiveEpilogue dispatch path.  Sm100
// dispatch inherits from Sm90 automatically (sm100_callbacks_tma_warpspecialized.hpp).
//
// Synchronization between phase 1 R2S complete and phase 2 S2R start is
// the kernel body's responsibility — NamedBarrier on NumEpilogueThreads
// between the two collective_epilogue.store() calls.
// ============================================================================

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"

namespace flashrt::megakernel::fusion {

using namespace cute;
using namespace cutlass::epilogue::fusion;

// ============================================================================
// Visitor 1: Sm100SmemAuxStore
//   - In the EVT chain, sits on top of an upstream LinComb+Act node.
//   - visit() captures the post-activation fragment into RMEM.
//   - postreduce() R2S-stores RMEM into smem_aux at (epi_m, epi_n) within
//     the full CTA-tile buffer.
//   - No producer load, no TMA store.  The smem_aux buffer is OWNED by
//     the calling kernel (not the visitor's SharedStorage) and passed
//     via Arguments::smem_aux_ptr.
// ============================================================================
template <
  class CtaTileShape_MN_,
  class EpilogueTile_,
  class Element_,
  class SmemLayoutAtom_,
  class CopyOpR2S_,
  cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest
>
struct Sm100SmemAuxStore {
  using CtaTileShape_MN = CtaTileShape_MN_;
  using EpilogueTile = EpilogueTile_;
  using Element      = Element_;
  using ElementAux   = Element;
  using SmemLayoutAtom = SmemLayoutAtom_;
  using CopyOpR2S    = CopyOpR2S_;

  using SmemLayout = decltype(tile_to_shape(
      SmemLayoutAtom{},
      cute::shape(CtaTileShape_MN{}),
      Step<_1,_2>{}));

  // Visitor OWNS its SMEM (mirrors Sm90AuxStore pattern).  Cross-phase
  // sharing handled at kernel TensorStorage layout level (see kernel hdr).
  struct SharedStorage {
    alignas(128) cute::array_aligned<Element, cute::cosize_v<SmemLayout>> smem_aux;
  };

  struct Arguments { };
  struct Params { };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const&, Arguments const&, void*) {
    return Params{};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) { return true; }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const&, Arguments const&, void*,
                       cudaStream_t = nullptr, cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE Sm100SmemAuxStore() { }
  CUTLASS_HOST_DEVICE
  Sm100SmemAuxStore(Params const&, SharedStorage const& shared_storage)
      : smem_aux(const_cast<Element*>(shared_storage.smem_aux.data())) { }

  Element* smem_aux;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class RTensor, class TiledR2S, class STensorR2S>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(RTensor&& tC_rAux_, TiledR2S tiled_r2s_, STensorR2S&& tRS_sAux_)
      : tC_rAux(cute::forward<RTensor>(tC_rAux_)),
        tiled_r2s(tiled_r2s_),
        tRS_sAux(cute::forward<STensorR2S>(tRS_sAux_)) { }

    RTensor    tC_rAux;
    TiledR2S   tiled_r2s;
    STensorR2S tRS_sAux;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const&,
          int epi_v, int, int,
          cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      using ConvertInput = cutlass::NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};
      Tensor tC_rAux_frg = recast<cutlass::Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(epi_v) = convert_input(frg_input);
      return frg_input;
    }

    CUTLASS_DEVICE void
    postreduce(int epi_m, int epi_n, int, bool issue_smem_store) {
      if (issue_smem_store) {
        using RLayoutR2S = decltype(cute::layout(TiledR2S{}.get_slice(0).retile_S(RTensor{})));
        Tensor tRS_rAux = make_tensor(tC_rAux.data(), RLayoutR2S{});
        copy(tiled_r2s, tRS_rAux, tRS_sAux(_,_,_,epi_m,epi_n));
      }
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Tensor sAux_full = cute::as_position_independent_swizzle_tensor(
                         make_tensor(make_smem_ptr(smem_aux), SmemLayout{}));
    Tensor sAux_epi = flat_divide(sAux_full, args.epi_tile);

    auto tiled_r2s = conditional_return<ReferenceSrc>(
        make_tiled_copy_S(Copy_Atom<CopyOpR2S, Element>{}, args.tiled_copy),
        make_tiled_copy_D(Copy_Atom<CopyOpR2S, Element>{}, args.tiled_copy));

    auto thr_r2s = tiled_r2s.get_slice(args.thread_idx);
    Tensor tRS_sAux = thr_r2s.partition_D(sAux_epi);
    Tensor tC_rAux = make_tensor<Element>(take<0,3>(shape(tRS_sAux)));

    return ConsumerStoreCallbacks<decltype(tC_rAux), decltype(tiled_r2s), decltype(tRS_sAux)>(
        cute::move(tC_rAux), tiled_r2s, cute::move(tRS_sAux));
  }
};

// ============================================================================
// Visitor 2: Sm100SmemAuxLoad
// ============================================================================
template <
  class CtaTileShape_MN_,
  class EpilogueTile_,
  class Element_,
  class SmemLayoutAtom_,
  class CopyOpS2R_
>
struct Sm100SmemAuxLoad {
  using CtaTileShape_MN = CtaTileShape_MN_;
  using EpilogueTile = EpilogueTile_;
  using Element      = Element_;
  using ElementAux   = Element;
  using SmemLayoutAtom = SmemLayoutAtom_;
  using CopyOpS2R    = CopyOpS2R_;

  using SmemLayout = decltype(tile_to_shape(
      SmemLayoutAtom{},
      cute::shape(CtaTileShape_MN{}),
      Step<_1,_2>{}));

  struct SharedStorage {
    alignas(128) cute::array_aligned<Element, cute::cosize_v<SmemLayout>> smem_aux;
  };

  struct Arguments { };
  struct Params { };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const&, Arguments const&, void*) {
    return Params{};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) { return true; }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const&, Arguments const&, void*,
                       cudaStream_t = nullptr, cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE Sm100SmemAuxLoad() { }
  CUTLASS_HOST_DEVICE
  Sm100SmemAuxLoad(Params const&, SharedStorage const& shared_storage)
      : smem_aux(const_cast<Element*>(shared_storage.smem_aux.data())) { }

  Element* smem_aux;

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class RTensor, class TiledS2R, class STensorS2R>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(RTensor&& tC_rAux_, TiledS2R tiled_s2r_, STensorS2R&& tSR_sAux_)
      : tC_rAux(cute::forward<RTensor>(tC_rAux_)),
        tiled_s2r(tiled_s2r_),
        tSR_sAux(cute::forward<STensorS2R>(tSR_sAux_)) { }

    RTensor    tC_rAux;
    TiledS2R   tiled_s2r;
    STensorS2R tSR_sAux;

    CUTLASS_DEVICE void
    previsit(int epi_m, int epi_n, int, bool) {
      using RLayoutS2R = decltype(cute::layout(TiledS2R{}.get_slice(0).retile_S(RTensor{})));
      Tensor tSR_rAux = make_tensor(tC_rAux.data(), RLayoutS2R{});
      copy(tiled_s2r, tSR_sAux(_,_,_,epi_m,epi_n), tSR_rAux);
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<Element, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const&,
          int epi_v, int, int) {
      Tensor tC_rAux_frg = recast<cutlass::Array<Element, FragmentSize>>(coalesce(tC_rAux));
      return tC_rAux_frg(epi_v);
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Tensor sAux_full = cute::as_position_independent_swizzle_tensor(
                         make_tensor(make_smem_ptr(smem_aux), SmemLayout{}));
    Tensor sAux_epi = flat_divide(sAux_full, args.epi_tile);

    auto tiled_s2r = conditional_return<ReferenceSrc>(
        make_tiled_copy_S(Copy_Atom<CopyOpS2R, Element>{}, args.tiled_copy),
        make_tiled_copy_D(Copy_Atom<CopyOpS2R, Element>{}, args.tiled_copy));

    auto thr_s2r = tiled_s2r.get_slice(args.thread_idx);
    Tensor tSR_sAux = thr_s2r.partition_S(sAux_epi);
    Tensor tC_rAux = make_tensor<Element>(take<0,3>(shape(tSR_sAux)));

    return ConsumerStoreCallbacks<decltype(tC_rAux), decltype(tiled_s2r), decltype(tSR_sAux)>(
        cute::move(tC_rAux), tiled_s2r, cute::move(tSR_sAux));
  }
};

// ============================================================================
// FusionOperation tags (FlashRT-specific).
//
// LinCombEltActSmemAuxStore inherits from LinCombEltAct and marks
// IsAuxOutSupported=true with ElementAux=ElementOutput so the
// CallbacksBuilder auto-dispatcher in sm100_builder.inl picks the
// "single aux out" specialization and routes the (SmemLayoutAtomAux,
// SmemCopyOpAux) types into our FusionCallbacks below.
// ============================================================================
template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  cutlass::FloatRoundStyle RoundStyle_ = cutlass::FloatRoundStyle::round_to_nearest
>
struct LinCombEltActSmemAuxStore
    : LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = cutlass::layout::RowMajor;  // unused (SMEM-only), but required by dispatcher
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxOutSupported = true;
  static constexpr bool IsAuxInSupported  = false;
};

template<
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux_ = ElementOutput_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentAux_ = 128 / cute::sizeof_bits_v<ElementAux_>,
  cutlass::FloatRoundStyle RoundStyle_ = cutlass::FloatRoundStyle::round_to_nearest
>
struct LinCombDeEltActSmemAuxLoad
    : cutlass::epilogue::fusion::LinearCombination<
          ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
  using ActivationFn = ActivationFn_<ElementCompute_>;
  static constexpr bool IsDeEltActSupported = true;

  using ElementAux = ElementAux_;
  using GmemLayoutTagAux = cutlass::layout::RowMajor;  // unused (SMEM-only), required by dispatcher
  static constexpr int AlignmentAux = AlignmentAux_;
  static constexpr bool IsAuxOutSupported = false;
  static constexpr bool IsAuxInSupported  = true;
};

// ============================================================================
// EVT tree aliases — what the FusionCallbacks specializations inherit from.
// ============================================================================
template <
  class CtaTileShapeMN, class EpilogueTile,
  class SmemLayoutAtom, class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput, class ElementCompute,
  class ElementAux,
  class ElementSource, class ElementScalar,
  cutlass::FloatRoundStyle RoundStyle
>
using Sm100LinCombEltActSmemAuxStoreTree =
  Sm90EVT<Sm100SmemAuxStore<CtaTileShapeMN, EpilogueTile, ElementAux, SmemLayoutAtom, CopyOpR2S, RoundStyle>,
    Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
      Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>
    >
  >;

template <
  class CtaTileShapeMN, class EpilogueTile,
  class SmemLayoutAtom, class CopyOpS2R,
  template <class> class ActivationFn,
  class ElementOutput, class ElementCompute,
  class ElementAux,
  class ElementSource, class ElementScalar,
  cutlass::FloatRoundStyle RoundStyle
>
using Sm100LinCombDeEltActSmemAuxLoadTree =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    Sm100SmemAuxLoad<CtaTileShapeMN, EpilogueTile, ElementAux, SmemLayoutAtom, CopyOpS2R>
  >;

} // namespace flashrt::megakernel::fusion

// ============================================================================
// FusionCallbacks specializations — must live in
// cutlass::epilogue::fusion namespace to be found by ADL during
// CollectiveBuilder dispatch.
// ============================================================================
namespace cutlass::epilogue::fusion {

// Phase 1 (gate): LinCombEltActSmemAuxStore -> EVT tree with Sm100SmemAuxStore
template <
  int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput, class ElementCompute,
  class ElementAux, class ElementSource, class ElementScalar,
  int AlignmentAux, cutlass::FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK, class EpilogueTile,
  class SmemLayoutAtom, class CopyOpR2S
>
struct FusionCallbacks<
    cutlass::epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    flashrt::megakernel::fusion::LinCombEltActSmemAuxStore<
        ActivationFn, ElementOutput, ElementCompute,
        ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile,
    SmemLayoutAtom, CopyOpR2S
> : flashrt::megakernel::fusion::Sm100LinCombEltActSmemAuxStoreTree<
        decltype(cute::take<0,2>(CtaTileShapeMNK{})),
        EpilogueTile,
        SmemLayoutAtom, CopyOpR2S,
        ActivationFn, ElementOutput, ElementCompute, ElementAux,
        ElementSource, ElementScalar, RoundStyle
    >
{
  using Impl = flashrt::megakernel::fusion::Sm100LinCombEltActSmemAuxStoreTree<
      decltype(cute::take<0,2>(CtaTileShapeMNK{})),
      EpilogueTile, SmemLayoutAtom, CopyOpR2S,
      ActivationFn, ElementOutput, ElementCompute, ElementAux,
      ElementSource, ElementScalar, RoundStyle>;

  using Operation = flashrt::megakernel::fusion::LinCombEltActSmemAuxStore<
      ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta  = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr  = nullptr;
    using StrideAlpha = cute::Stride<cute::_0, cute::_0, int64_t>;
    using StrideBeta  = cute::Stride<cute::_0, cute::_0, int64_t>;
    StrideAlpha dAlpha = {cute::_0{}, cute::_0{}, 0};
    StrideBeta  dBeta  = {cute::_0{}, cute::_0{}, 0};

    using ActivationArguments =
        typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      // Sm90VisitorImpl<children..., node>::Arguments — children FIRST,
      // node LAST.  Sm100SmemAuxStore now owns its SMEM via SharedStorage
      // (no external pointer); its Arguments is empty.
      return {
        // child of outer tree: inner Sm90EVT (LinComb + GELU)
        {
          // child of inner tree: Sm90LinearCombination
          {
            { {beta},  {beta_ptr},  {dBeta}  },
            {},
            {
              { {alpha}, {alpha_ptr}, {dAlpha} },
              {},
              {}
            },
            {}
          },
          activation
        },
        // node of outer tree: Sm100SmemAuxStore — Arguments is empty
        {}
      };
    }
  };

  using Impl::Impl;
};

// Phase 2 (up): LinCombDeEltActSmemAuxLoad -> EVT tree with Sm100SmemAuxLoad
template <
  int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput, class ElementCompute,
  class ElementAux, class ElementSource, class ElementScalar,
  int AlignmentAux, cutlass::FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK, class EpilogueTile,
  class SmemLayoutAtom, class CopyOpS2R
>
struct FusionCallbacks<
    cutlass::epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    flashrt::megakernel::fusion::LinCombDeEltActSmemAuxLoad<
        ActivationFn, ElementOutput, ElementCompute,
        ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile,
    SmemLayoutAtom, CopyOpS2R
> : flashrt::megakernel::fusion::Sm100LinCombDeEltActSmemAuxLoadTree<
        decltype(cute::take<0,2>(CtaTileShapeMNK{})),
        EpilogueTile,
        SmemLayoutAtom, CopyOpS2R,
        ActivationFn, ElementOutput, ElementCompute, ElementAux,
        ElementSource, ElementScalar, RoundStyle
    >
{
  using Impl = flashrt::megakernel::fusion::Sm100LinCombDeEltActSmemAuxLoadTree<
      decltype(cute::take<0,2>(CtaTileShapeMNK{})),
      EpilogueTile, SmemLayoutAtom, CopyOpS2R,
      ActivationFn, ElementOutput, ElementCompute, ElementAux,
      ElementSource, ElementScalar, RoundStyle>;

  using Operation = flashrt::megakernel::fusion::LinCombDeEltActSmemAuxLoad<
      ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta  = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr  = nullptr;
    using StrideAlpha = cute::Stride<cute::_0, cute::_0, int64_t>;
    using StrideBeta  = cute::Stride<cute::_0, cute::_0, int64_t>;
    StrideAlpha dAlpha = {cute::_0{}, cute::_0{}, 0};
    StrideBeta  dBeta  = {cute::_0{}, cute::_0{}, 0};

    using ActivationArguments =
        typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return {
        // child 0: Sm90LinearCombination
        {
          { {beta},  {beta_ptr},  {dBeta}  },
          {},
          {
            { {alpha}, {alpha_ptr}, {dAlpha} },
            {},
            {}
          },
          {}
        },
        // child 1: Sm100SmemAuxLoad — empty Arguments
        {},
        // node: Sm90Compute<ActivationFn>
        activation
      };
    }
  };

  using Impl::Impl;
};

} // namespace cutlass::epilogue::fusion
