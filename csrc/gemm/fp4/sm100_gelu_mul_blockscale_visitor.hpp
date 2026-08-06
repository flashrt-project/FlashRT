// SPDX-License-Identifier: Apache-2.0
//
// GeGLU fused blockscale-FP4 epilogue store node for SM100/SM110 — a fork of
// CUTLASS's Sm100BlockScaleFactorRowStore (sm100_visitor_store_tma_
// warpspecialized.hpp) whose visit() applies gelu(gate)*up on the ADJACENT
// interleaved accumulator columns (gate@even, up@odd) before the scale-factor
// generation, instead of a passthrough. The amax/SF-store/quantize machinery
// is verbatim, so the FP4 output + UE4M3 SFD layout match the stock kernel.
//
// Weight-side contract: B holds the gate and up projection rows pairwise
// interleaved along N (B_il[2j] = W_gate[j], B_il[2j+1] = W_up[j]), so each
// accumulator fragment carries [g0,u0,g1,u1,...]. Any per-output-column scale
// (e.g. the down-projection AWQ inv_s) must be folded into the up rows before
// quantization — the epilogue applies no per-column vector.
//
// Full-width output stage: the geglu result is duplicated into both columns
// of each pair, so N_out == N_il and the consumer's weight is K-expanded with
// zero odd columns. Half-width store compaction is a separate follow-up.
//
// Additive: extends cutlass::epilogue::fusion with a new op tag + callback
// specialization in our tree; the stock visitor is untouched.

#pragma once

#include <cuda_fp8.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"

namespace cutlass::epilogue::fusion {

using namespace cute;

// ── Forked store node: gelu(gate)*up over adjacent interleaved columns ───────
template <
  int SFVecSize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct Sm100GeluMulBlockScaleFactorRowStore {
  static_assert(size<1>(EpilogueTile{}) % SFVecSize == 0, "EpilogueTileN should be divisible by SFVecSize");
  static_assert(size<1>(EpilogueTile{}) / SFVecSize == 1 or
                size<1>(EpilogueTile{}) / SFVecSize == 2 or
                size<1>(EpilogueTile{}) / SFVecSize == 4 or
                size<1>(EpilogueTile{}) / SFVecSize == 8,
                "Possible store in interleaved 4B aligned format");
  using NormalConstStrideMNL = Stride<_0,_0,int64_t>;
  struct SharedStorage { };

  struct Arguments {
    ElementBlockScaleFactor* ptr_scale_factor = nullptr;
    ElementCompute const* norm_constant_ptr = nullptr;
    NormalConstStrideMNL norm_constant_stride = {};
  };

  using Params = Arguments;

  using UnderlyingElementBlockScaleFactor = cute::remove_pointer_t<ElementBlockScaleFactor>;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    // Pairwise column fold: N must also be even in units of SFVecSize.
    bool implementable = (N % SFVecSize == 0) && (N % 2 == 0);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: [EVT Sm100GeluMulBlockScaleFactorRowStore] N-dim should be divisible by SFVecSize.\n");
    }
    return implementable;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm100GeluMulBlockScaleFactorRowStore() { }

  CUTLASS_HOST_DEVICE
  Sm100GeluMulBlockScaleFactorRowStore(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params) { }

  Params const* params_ptr = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  // tanh-GELU in its sigmoid form — bit-matches the GeGLU combiner kernels
  // (gelu(g) = g * sigmoid(2*sqrt(2/pi)*g*(1 + 0.044715 g^2))).
  CUTLASS_DEVICE static float
  gelu_tanh_(float g) {
    return g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
  }

  template <
    class RTensor,
    class GTensor,
    class CoordGTensor,
    class ThrResidue,
    class EpiTileCoordMN,
    class ElementType
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
          RTensor&& tC_rSFD_,                   // (CPY,CPY_M,CPY_N)
          GTensor&& tC_gSFD_,                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,#EPI_Ms, #EPI_Ns)
          CoordGTensor tC_cSFD_,                // (m,n)
          ThrResidue residue_tC_cSFD_,          // (m,n)
          Params const* params_ptr_,
          EpiTileCoordMN epi_tile_coord_mn_,    // (epi_tile_coord_m, epi_tile_coord_n)
          ElementType norm_constant_,
          ElementType norm_constant_scaled_down_)
      : tC_rSFD(cute::forward<RTensor>(tC_rSFD_))
      , tC_gSFD(cute::forward<GTensor>(tC_gSFD_))
      , tC_cSFD(tC_cSFD_)
      , residue_tC_cSFD(residue_tC_cSFD_)
      , params_ptr(params_ptr_)
      , norm_constant(norm_constant_)
      , norm_constant_scaled_down(norm_constant_scaled_down_)
      , epi_tile_coord_mn(epi_tile_coord_mn_){}

    static_assert(is_same_v<ElementType, ElementCompute>);
    RTensor tC_rSFD;
    GTensor tC_gSFD;
    CoordGTensor tC_cSFD;
    ThrResidue residue_tC_cSFD;
    Params const* params_ptr;
    ElementCompute norm_constant;
    ElementCompute norm_constant_scaled_down;
    EpiTileCoordMN epi_tile_coord_mn;

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc,
          int epi_v,
          int epi_m,
          int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input)
    {
      static_assert(FragmentSize % SFVecSize == 0, "Scale factor vector size should divide FragmentSize");
      constexpr int NumVecs = FragmentSize / SFVecSize;
      Array<ElementCompute, FragmentSize> frg_compute;

      auto input_frgs = reinterpret_cast<Array< ElementInput, SFVecSize> const*>(frg_input.data());
      auto compute_frgs = reinterpret_cast<Array< ElementCompute, SFVecSize> *>(frg_compute.data());

      Tensor tC_rSFD_frg = recast<cutlass::Array<UnderlyingElementBlockScaleFactor, NumVecs>>(coalesce(filter(tC_rSFD)));               // (EPI_V)

      cutlass::maximum_absolute_value_reduction<Array<ElementCompute, SFVecSize>, true> amax_reduction;

      cutlass::Array<ElementCompute, NumVecs> vec_maxs;
      cutlass::Array<ElementCompute, NumVecs> pvscales;
      CUTLASS_PRAGMA_UNROLL
      for (int sf_v = 0; sf_v < NumVecs; ++sf_v) {
        compute_frgs[sf_v] = NumericArrayConverter<ElementCompute, ElementInput, SFVecSize>{}(input_frgs[sf_v]);
      }

      // GeGLU fold. The interleaved weight puts ADJACENT (gate, up) columns
      // in the fragment: [g0,u0,g1,u1,...]. Compute gelu(g)*u and duplicate
      // into both slots of the pair (full-width output), then run the stock
      // amax/SF/quantize pipeline on the folded values.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; i += 2) {
        float g = static_cast<float>(frg_compute[i]);
        float u = static_cast<float>(frg_compute[i + 1]);
        ElementCompute v = static_cast<ElementCompute>(gelu_tanh_(g) * u);
        frg_compute[i]     = v;
        frg_compute[i + 1] = v;
      }

      // SF generation
      CUTLASS_PRAGMA_UNROLL
      for (int sf_v = 0; sf_v < NumVecs; ++sf_v) {
        /// Step1: get max across a vector
        vec_maxs[sf_v] = amax_reduction(ElementCompute(0), compute_frgs[sf_v]);
      }

      /// Step2: Compute Scale
      pvscales = cutlass::multiplies<Array<ElementCompute, NumVecs>>{}(vec_maxs, norm_constant_scaled_down);

      tC_rSFD_frg(_0{}) = cutlass::NumericArrayConverter<UnderlyingElementBlockScaleFactor, ElementCompute, NumVecs>{}(pvscales);

      Tensor tCgSFD_flt = filter_zeros(tC_gSFD(_,_,_,_0{},_0{},get<0>(epi_tile_coord_mn) + epi_m, get<1>(epi_tile_coord_mn) + epi_n));
      Tensor tCrSFD_flt = filter_zeros(tC_rSFD);
      constexpr auto MCL = decltype(max_common_layout(tCgSFD_flt, tCrSFD_flt)){};
      constexpr int V = cute::min(4, size(MCL));
      using VecType = uint_bit_t<V * sizeof_bits_v<UnderlyingElementBlockScaleFactor>>;
      Tensor tCgSFD_vec = recast<VecType>(coalesce(tCgSFD_flt));
      Tensor tCrSFD_vec = recast<VecType>(coalesce(tCrSFD_flt));
      Tensor tCcSFD_pred = tC_cSFD(_,_,_, epi_m, epi_n);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrSFD_vec); i++){
        if (elem_less(tCcSFD_pred(i * SFVecSize * V), residue_tC_cSFD)) {
          tCgSFD_vec(i) = tCrSFD_vec(i);
        }
      }
      /// Step3: Compute quantized output values
      return detail::compute_quantized_with_row_scalefactor<SFVecSize, ElementOutput>(frg_compute, tC_rSFD_frg(_0{}), norm_constant);
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [tile_coord_m, tile_coord_n, tile_coord_k, tile_coord_l] = args.tile_coord_mnkl;
    using Sm1xxBlockScaledOutputConfig= cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize>;
    UnderlyingElementBlockScaleFactor* ptr_scale_factor = nullptr;
    // If Ptr-Array/Grouped GEMM with BlockScaleFactor per batch/group
    if constexpr (!cute::is_same_v<UnderlyingElementBlockScaleFactor, ElementBlockScaleFactor>) {
      ptr_scale_factor = params_ptr->ptr_scale_factor[tile_coord_l];
      tile_coord_l = 0;
    }
    else {
      ptr_scale_factor = params_ptr->ptr_scale_factor;
    }

    auto epi_tile_mn = shape<1>(zipped_divide(make_layout(take<0,2>(args.tile_shape_mnk)), args.epi_tile));
    Tensor mSFD = make_tensor(make_gmem_ptr(ptr_scale_factor), Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(args.problem_shape_mnkl));
    static_assert(size<1>(EpilogueTile{}) && ((size<1>(EpilogueTile{}) & (size<1>(EpilogueTile{}) - 1)) == 0), "Epilogue Tile N should be pow of 2");
    Tensor gSFD = local_tile(mSFD, args.epi_tile, make_coord(_,_,tile_coord_l));                   // (EPI_M,EPI_N, #EPI_Ms, #EPI_Ns)
    Tensor tCgSFD = sm90_partition_for_epilogue<ReferenceSrc>(                                     // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,#EPI_Ms, #EPI_Ns)
                        gSFD, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrSFD = make_tensor_like<UnderlyingElementBlockScaleFactor>(take<0,3>(cute::layout(tCgSFD)));    // (CPY,CPY_M,CPY_N)

    auto epi_tile_coord_mn = make_coord(tile_coord_m * size<0>(epi_tile_mn), tile_coord_n * size<1>(epi_tile_mn));

    // Fetch and compute these during initialization
    Tensor mNormConst= make_tensor(make_gmem_ptr(params_ptr->norm_constant_ptr), make_layout(make_shape(M, N, L), params_ptr->norm_constant_stride));
    ElementCompute norm_constant = mNormConst(_0{},_0{},tile_coord_l);
    ElementCompute fp_max = ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::max());
    ElementCompute scale_down_factor = cutlass::reciprocal_approximate_ftz<ElementCompute>{}(fp_max);
    ElementCompute norm_constant_scaled_down = cutlass::multiplies<ElementCompute>{}(norm_constant, scale_down_factor);

    return ConsumerStoreCallbacks(
      cute::move(tCrSFD),
      cute::move(tCgSFD),
      args.tCcD,
      args.residue_tCcD,
      params_ptr,
      epi_tile_coord_mn,
      norm_constant,
      norm_constant_scaled_down);

  }
};

// ── EVT tree alias: store node over (beta*C + alpha*acc), alpha kept = 1 ─────
template <
  int SFVecsize, class EpilogueTile, class ElementOutput, class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementSource = ElementOutput, class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100GeluMulRowBlockScaleFactor =
  Sm90EVT<Sm100GeluMulBlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>,
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>
  >;

// ── New FusionOperation tag (subclass of LinCombBlockScaleFactor) ────────────
template <
  int SFVecSize, class ElementOutput, class ElementCompute,
  class ElementBlockScaleFactor, class GmemLayoutTagScalefactor,
  class ElementSource = ElementOutput, class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct GeluMulBlockScaleFactor
    : LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute,
        ElementBlockScaleFactor, GmemLayoutTagScalefactor, ElementSource,
        ElementScalar, RoundStyle> {};

// ── FusionCallbacks specialization mapping the op tag -> forked EVT tree ─────
template <
  int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
  class ElementOutput, class ElementCompute, class ElementBlockScaleFactor,
  int SFVecSize, class ElementSource, class ElementScalar,
  FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    GeluMulBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100GeluMulRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm100GeluMulRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = GeluMulBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor* block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// ═════════════════════════════════════════════════════════════════════════════
// Half-width variant: the geglu result is quantized at compact granularity
// (16 unique values per scale block, matching the standalone combiner) and
// written by the visitor itself to a compact [M, N/2] FP4 buffer + SFD via
// direct gmem stores — the same self-partitioned store pattern the stock
// node uses for its scale factors. The collective's own D path receives
// garbage (zeros) and should be pointed at a small reusable dummy buffer;
// the downstream GEMM consumes only the compact outputs, so its weight
// keeps the original K (no K-expansion, no doubled weight streaming).
//
// Fragment contract (verified on the production tile): each visit covers
// FragmentSize contiguous columns of one row, FragmentSize % 32 == 0, and
// the fragment base column is 32-aligned, so every visit folds whole
// compact scale blocks. Threads may hold duplicate fragments; duplicate
// stores write identical bytes and are benign.
// ═════════════════════════════════════════════════════════════════════════════
template <
  int SFVecSize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct Sm100GeluMulCompactBlockScaleFactorRowStore {
  static_assert(size<1>(EpilogueTile{}) % SFVecSize == 0, "EpilogueTileN should be divisible by SFVecSize");
  using NormalConstStrideMNL = Stride<_0,_0,int64_t>;
  struct SharedStorage { };

  struct Arguments {
    ElementBlockScaleFactor* ptr_scale_factor = nullptr;  // unused (kept for arg shape)
    ElementCompute const* norm_constant_ptr = nullptr;    // unused
    NormalConstStrideMNL norm_constant_stride = {};
    uint8_t* compact_ptr = nullptr;     // packed e2m1 [M, N/2] row-major
    uint8_t* compact_sf_ptr = nullptr;  // UE4M3 SFA tile-atom layout on (M, N/2)
  };

  using Params = Arguments;

  using UnderlyingElementBlockScaleFactor = cute::remove_pointer_t<ElementBlockScaleFactor>;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    // Whole compact scale blocks per fragment: N in units of 2*SFVecSize.
    return (N % (2 * SFVecSize) == 0) && args.compact_ptr && args.compact_sf_ptr;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm100GeluMulCompactBlockScaleFactorRowStore() { }

  CUTLASS_HOST_DEVICE
  Sm100GeluMulCompactBlockScaleFactorRowStore(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params) { }

  Params const* params_ptr = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  CUTLASS_DEVICE static float
  gelu_tanh_(float g) {
    return g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
  }

  // Branch-ladder e2m1 round-to-nearest — matches the combiner kernels.
  CUTLASS_DEVICE static uint8_t
  fp32_to_e2m1_(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
  }

  template <
    class CoordGTensor,
    class ThrResidue,
    class LayoutSFC
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
          CoordGTensor tC_cD_,
          ThrResidue residue_tC_cD_,
          Params const* params_ptr_,
          LayoutSFC layout_sfc_,
          int n_compact_bytes_,
          int tile_row_off_,
          int tile_col_off_)
      : tC_cD(tC_cD_)
      , residue_tC_cD(residue_tC_cD_)
      , params_ptr(params_ptr_)
      , layout_sfc(layout_sfc_)
      , n_compact_bytes(n_compact_bytes_)
      , tile_row_off(tile_row_off_)
      , tile_col_off(tile_col_off_) {}

    CoordGTensor tC_cD;
    ThrResidue residue_tC_cD;
    Params const* params_ptr;
    LayoutSFC layout_sfc;
    int n_compact_bytes;  // (N/2)/2 bytes per compact row
    // The coordinate tensor is thread-relative (its layout drops the
    // per-thread iterator offset); the thread's global base coordinate is
    // recovered as (M, N) - residue, computed per tile by the caller.
    int tile_row_off;
    int tile_col_off;

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc,
          int epi_v,
          int epi_m,
          int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input)
    {
      static_assert(FragmentSize % (2 * SFVecSize) == 0,
                    "fragment must cover whole compact scale blocks");
      constexpr int CompactBlocks = FragmentSize / (2 * SFVecSize);

      Tensor pred = tC_cD(_, _, _, epi_m, epi_n);
      auto c0 = pred(0);
      if (elem_less(c0, residue_tC_cD)) {
        const int row = tile_row_off + get<0>(c0);
        const int n0  = tile_col_off + get<1>(c0);
        uint8_t* cptr = params_ptr->compact_ptr
                        + static_cast<long>(row) * n_compact_bytes + (n0 >> 2);
        // One compact scale block (16 unique values) at a time keeps the
        // working set register-resident.
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < CompactBlocks; ++j) {
          float vals[SFVecSize];
          float amax = 0.f;
          CUTLASS_PRAGMA_UNROLL
          for (int t = 0; t < SFVecSize; ++t) {
            float g = static_cast<float>(frg_input[2 * SFVecSize * j + 2 * t]);
            float u = static_cast<float>(
                frg_input[2 * SFVecSize * j + 2 * t + 1]);
            vals[t] = gelu_tanh_(g) * u;
            const float a = fabsf(vals[t]);
            if (a > amax) amax = a;
          }
          float desired = amax / 6.f;
          if (desired < 1e-12f) desired = 1e-12f;
          __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(desired);
          const float inv_bs = 1.f / static_cast<float>(bs_q);
          params_ptr->compact_sf_ptr[
              layout_sfc(row, (n0 >> 1) + j * SFVecSize, 0)] =
              *reinterpret_cast<uint8_t*>(&bs_q);
          // Hardware e2m1 conversion; the packed subbyte Array puts element
          // 2p in the low nibble of byte p, matching the consumer layout.
          Array<ElementCompute, SFVecSize> scaled;
          CUTLASS_PRAGMA_UNROLL
          for (int t = 0; t < SFVecSize; ++t) {
            scaled[t] = static_cast<ElementCompute>(vals[t] * inv_bs);
          }
          auto packed_e2m1 =
              NumericArrayConverter<cutlass::float_e2m1_t, ElementCompute,
                                    SFVecSize, RoundStyle>{}(scaled);
          *reinterpret_cast<uint2*>(cptr + j * 8) =
              *reinterpret_cast<uint2 const*>(&packed_e2m1);
        }
      }

      // The collective's D path lands in a dummy buffer; feed it zeros.
      Array<ElementOutput, FragmentSize> frg_output;
      frg_output.fill(ElementOutput(0));
      return frg_output;
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
    // SFA layout convention places the quantized axis in the K slot
    // (the compact buffer is consumed as the next GEMM's A operand).
    auto layout_sfc = Cfg::tile_atom_to_shape_SFA(make_shape(M, 1, N / 2, 1));
    return ConsumerStoreCallbacks<decltype(args.tCcD),
                                  decltype(args.residue_tCcD),
                                  decltype(layout_sfc)>(
        args.tCcD, args.residue_tCcD, params_ptr, layout_sfc, N / 4,
        M - static_cast<int>(get<0>(args.residue_tCcD)),
        N - static_cast<int>(get<1>(args.residue_tCcD)));
  }
};

// EVT tree + op tag + callbacks for the compact store.
template <
  int SFVecsize, class EpilogueTile, class ElementOutput, class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementSource = ElementOutput, class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100GeluMulCompactRowBlockScaleFactor =
  Sm90EVT<Sm100GeluMulCompactBlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>,
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>
  >;

template <
  int SFVecSize, class ElementOutput, class ElementCompute,
  class ElementBlockScaleFactor, class GmemLayoutTagScalefactor,
  class ElementSource = ElementOutput, class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct GeluMulCompactBlockScaleFactor
    : LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute,
        ElementBlockScaleFactor, GmemLayoutTagScalefactor, ElementSource,
        ElementScalar, RoundStyle> {};

template <
  int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
  class ElementOutput, class ElementCompute, class ElementBlockScaleFactor,
  int SFVecSize, class ElementSource, class ElementScalar,
  FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    GeluMulCompactBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100GeluMulCompactRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm100GeluMulCompactRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = GeluMulCompactBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor* block_scale_factor_ptr = nullptr;
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};
    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};
    uint8_t* compact_ptr = nullptr;
    uint8_t* compact_sf_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return
        {
          {
            {{beta}, {beta_ptr}, {dBeta}},
            {},
            {
              {{alpha}, {alpha_ptr}, {dAlpha}},
              {},
              {}
            },
            {}
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst,
           compact_ptr, compact_sf_ptr}
        };
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion
