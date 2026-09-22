// SPDX-License-Identifier: Apache-2.0
//
// SwiGLU fused blockscale-FP4 epilogue store node for SM120a — a fork of
// CUTLASS's Sm120BlockScaleFactorRowStore (sm120_visitor_store_tma_
// warpspecialized.hpp) whose visit() applies silu(alpha_gate*gate)*alpha_up*up
// on the ADJACENT interleaved accumulator columns (gate@even, up@odd) instead
// of a passthrough. The amax/quant/store machinery (reduce(), the SF store) is
// verbatim, so the FP4 output + UE4M3 SF layout match the stock kernel exactly.
//
// M2a stage: FULL-WIDTH output (the silu_mul result is duplicated into both
// columns of each pair). Down consumes one slot; the half-width compaction is
// M2b. Additive: lives in our tree, extends cutlass::epilogue::fusion with a new
// op tag + callback specialization; the stock visitor is untouched.

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"

namespace cutlass::epilogue::fusion {

// Spark's GELU is torch's exact erf form, evaluated in fp32 and rounded to the
// output type -- not the tanh approximation CUTLASS's Gelu_tanh implements.
__device__ __forceinline__ float spark_gelu_f32(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}


using namespace cute;
using namespace detail;

// ── Forked store node: silu_mul over adjacent (gate,up) columns ──────────────
template <
  int SFVecSize,
  class EpilogueTile,
  class CtaTileShapeMNK,
  int FragmentSize,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct Sm120GeluMulBlockScaleFactorRowStore {

  static_assert(size<1>(EpilogueTile{}) % SFVecSize == 0, "EpilogueTileN should be divisible by SFVecSize");
  static_assert(size<1>(EpilogueTile{}) / SFVecSize == 1 or
                size<1>(EpilogueTile{}) / SFVecSize == 2 or
                size<1>(EpilogueTile{}) / SFVecSize == 4 or
                size<1>(EpilogueTile{}) / SFVecSize == 8,
                "Possible store in interleaved 4B aligned format");

  static constexpr int NumWarpgroups = 2;
  static constexpr int NumSyncWarps = NumWarpsPerWarpGroup * NumWarpgroups;
  static constexpr int NumQuadsPerWarp = 8;
  static constexpr int NumSyncQuads = NumSyncWarps * NumQuadsPerWarp;
  struct SharedStorage {
    array_aligned<ElementCompute, NumSyncQuads> smem_aux;
  };
  using NormalConstStrideMNL = Stride<_0,_0,int64_t>;
  struct Arguments {
    ElementBlockScaleFactor* ptr_scale_factor = {};
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    ElementCompute const* norm_constant_ptr = {};
    NormalConstStrideMNL norm_constant_stride = {};
    ElementCompute alpha_gate = ElementCompute(1);
    ElementCompute alpha_up = ElementCompute(1);
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
    bool implementable = (N % SFVecSize == 0);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: [EVT Sm120GeluMulBlockScaleFactorRowStore] N-dim should be divisible by SFVecSize.\n");
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
  Sm120GeluMulBlockScaleFactorRowStore() { }

  CUTLASS_HOST_DEVICE
  Sm120GeluMulBlockScaleFactorRowStore(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params)
      , smem_aux(const_cast<ElementCompute*>(shared_storage.smem_aux.data())) { }

  Params const* params_ptr = nullptr;
  ElementCompute *smem_aux = nullptr;

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

  template <
    class RTensor,
    class GTensor,
    class STensor,
    class CoordGTensor,
    class ThrResidue,
    class TileCoordMN,
    class ElementType,
    class TiledCopy_
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
          RTensor&& tC_rSFD_,
          GTensor&& tC_gSFD_,
          STensor&& sAmaxs_,
          CoordGTensor tC_cSFD_,
          ThrResidue residue_tC_cSFD_,
          Params const* params_ptr_,
          TileCoordMN tile_coord_mn_,
          ElementType norm_constant_,
          ElementType norm_constant_scaled_down_,
          int thread_idx_,
          TiledCopy_ const&)
      : tC_rSFD(cute::forward<RTensor>(tC_rSFD_))
      , tC_gSFD(cute::forward<GTensor>(tC_gSFD_))
      , sAmaxs(cute::forward<STensor>(sAmaxs_))
      , tC_cSFD(tC_cSFD_)
      , residue_tC_cSFD(residue_tC_cSFD_)
      , params_ptr(params_ptr_)
      , norm_constant(norm_constant_)
      , norm_constant_scaled_down(norm_constant_scaled_down_)
      , tile_coord_mn(tile_coord_mn_)
      , thread_idx(thread_idx_) {}

    static_assert(is_same_v<ElementType, ElementCompute>);
    RTensor tC_rSFD;
    GTensor tC_gSFD;
    STensor sAmaxs;
    CoordGTensor tC_cSFD;
    ThrResidue residue_tC_cSFD;
    Params const* params_ptr;
    ElementCompute norm_constant;
    ElementCompute norm_constant_scaled_down;
    TileCoordMN tile_coord_mn;
    int thread_idx;
    static constexpr int NumCollaboratingThreads = decltype(size(TiledCopy_{}))::value;
    static_assert(NumCollaboratingThreads % NumThreadsPerWarpGroup == 0);
    static constexpr int NumCollaboratingWarpGroups = NumCollaboratingThreads / NumThreadsPerWarpGroup;
    static_assert(NumCollaboratingWarpGroups == 1 || NumCollaboratingWarpGroups == 2,
                  "SM120 epilogue currently only supports one or two warp groups collaborating.");

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc,
          int epi_v,
          int epi_m,
          int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      // SwiGLU fold. With the interleaved gate/up weight, frg_input holds
      // ADJACENT (gate,up) accumulator columns: [r0gate,r0up,r1gate,r1up] for
      // FragmentSize==4. Apply the per-projection scales and silu(gate)*up.
      // M2a: duplicate the result into both columns of each pair (full-width
      // output; M2b compacts the store to half width).
      Array<ElementInput, FragmentSize> r;
      ElementCompute ag = params_ptr->alpha_gate;
      ElementCompute au = params_ptr->alpha_up;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; i += 2) {
        ElementCompute g = ag * static_cast<ElementCompute>(frg_input[i]);
        ElementCompute u = au * static_cast<ElementCompute>(frg_input[i + 1]);
        ElementCompute sm = spark_gelu_f32(g) * u;
        r[i]     = static_cast<ElementInput>(sm);
        r[i + 1] = static_cast<ElementInput>(sm);
      }
      return r;
    }

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(SmemTensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      /*
      Accumulator fragments are distributed across quads in different warps.
      For SFVector = 16, we have:

         8 elements          8 elements       8 elements          8 elements
      <----------------><-----------------><-----------------><----------------->
        Warp 0 Quad 0      Warp 0 Quad 0      Warp 4 Quad 0      Warp 4 Quad 0
        Warp 0 Quad 1      Warp 0 Quad 1      Warp 4 Quad 1      Warp 4 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 0 Quad 7      Warp 4 Quad 7      Warp 4 Quad 7
        Warp 0 Quad 0      Warp 0 Quad 0      Warp 4 Quad 0      Warp 4 Quad 0
        Warp 0 Quad 1      Warp 0 Quad 1      Warp 4 Quad 1      Warp 4 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 0 Quad 7      Warp 4 Quad 7      Warp 4 Quad 7

        <same pattern for warps 1 and 5 for the next set of 16 rows>
        <same pattern for warps 2 and 6 for the next set of 16 rows>
        <same pattern for warps 3 and 7 for the next set of 16 rows>

      In this case, row-wise scale factors are cooperatively reduced across 4
      threads from 1 quad in 1 warp. Each quad computes its own, local absolute
      maximum without communicating with other warps through shared memory.

      For SFVector = 32, we have:
         8 elements        8 elements         8 elements         8 elements
      <----------------><-----------------><-----------------><----------------->
        Warp 0 Quad 0      Warp 4 Quad 0      Warp 0 Quad 0      Warp 4 Quad 0
        Warp 0 Quad 1      Warp 4 Quad 1      Warp 0 Quad 1      Warp 4 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 4 Quad 7      Warp 0 Quad 7      Warp 4 Quad 7
        Warp 0 Quad 0      Warp 4 Quad 0      Warp 0 Quad 0      Warp 4 Quad 0
        Warp 0 Quad 1      Warp 4 Quad 1      Warp 0 Quad 1      Warp 4 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 4 Quad 7      Warp 0 Quad 7      Warp 4 Quad 7

        <same pattern for warps 1 and 5 for the next set of 16 rows>
        <same pattern for warps 2 and 6 for the next set of 16 rows>
        <same pattern for warps 3 and 7 for the next set of 16 rows>

      For SFVector = 64, we have:
          8 elements        8 elements         8 elements         8 elements
      <----------------><-----------------><-----------------><----------------->
        Warp 0 Quad 0      Warp 2 Quad 0      Warp 4 Quad 0      Warp 6 Quad 0
        Warp 0 Quad 1      Warp 2 Quad 1      Warp 4 Quad 1      Warp 6 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 2 Quad 7      Warp 4 Quad 7      Warp 6 Quad 7
        Warp 0 Quad 0      Warp 2 Quad 0      Warp 4 Quad 0      Warp 6 Quad 0
        Warp 0 Quad 1      Warp 2 Quad 1      Warp 4 Quad 1      Warp 6 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 2 Quad 7      Warp 4 Quad 7      Warp 6 Quad 7

        <same pattern for warps 1, 3, 5 and 7 for the next set of 16 rows>

      Thus, rowwise scale factors are cooperatively reduced across 8 threads
      from two quads in two warps. Each quad first computes its own, local
      absolute maximum and then shares this with the corresponding quad in the
      other warp. In this case, a reduction through shared memory is needed.

      For a non-cooperative epilogue (in which each warpgroup computes a
      separate tile), the pattern is the same as that above, except that warps 0
      and 2 are in the same row, and 1 and 3 are in the same row, and warps 4-7
      are not included.
      */

      // Accumulator fragments consist of two elements from two different rows of a 16x8 MMA output
      static constexpr int ColsPerThreadAccFrag = 2;
      static constexpr int RowsPerThreadAccFrag = 2;
      static_assert(FragmentSize ==
                    (ColsPerThreadAccFrag * RowsPerThreadAccFrag));

      static constexpr int NumThreadsPerQuad = 4;
      static_assert(SFVecSize == 16 || SFVecSize == 32 || SFVecSize == 64, "SF vector size must be either 16, 32 or 64.");
      // A quad from two or four warps participate in computing each scale factor.
      constexpr int WarpsPerSF = SFVecSize / 16;
      static_assert(WarpsPerSF == 1 || WarpsPerSF == 2 || WarpsPerSF == 4, "Only one, two or four warps are allowed in reduction.");

      constexpr bool IsInterWarpReductionNeeded = (WarpsPerSF != 1);

      // Number of fragments for each thread that are needed for computing a scale factor
      static constexpr int AccFragsPerSF = SFVecSize / (ColsPerThreadAccFrag * NumThreadsPerQuad * WarpsPerSF);
      static_assert(size<2>(visit_results) % AccFragsPerSF == 0,
        "Fragments along N mode must be a multiple of the number of accumulator fragments needed per SF");

      auto warp_idx = thread_idx / NumThreadsPerWarp;
      auto warpgroup_idx = thread_idx / NumThreadsPerWarpGroup;
      auto quad_idx_in_warp = (thread_idx % NumThreadsPerWarp) / NumThreadsPerQuad;
      auto thread_idx_in_quad = thread_idx % NumThreadsPerQuad;

      cutlass::maximum_absolute_value_reduction<ElementCompute, true> amax_op;
      cutlass::multiplies<ElementCompute> mul;

      Tensor tC_rSFD_flt = filter_zeros(tC_rSFD);

      auto synchronize = [&] () {
        cutlass::arch::NamedBarrier::sync(NumCollaboratingThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      };

      CUTLASS_PRAGMA_UNROLL
      for (int sf_id = 0; sf_id < size(tC_rSFD_flt); ++sf_id) {

        auto coord = idx2crd(sf_id, tC_rSFD_flt.shape());
        auto row_in_acc = get<0,1,1>(coord);
        auto row = crd2idx(get<1>(coord), get<1>(tC_rSFD_flt.shape()));
        auto sf = crd2idx(get<2>(coord), get<2>(tC_rSFD_flt.shape()));

        //
        // Compute amax for this scale factor
        //
        ElementCompute amax{0};

        // Compute amax among vals owned by this thread for this vector
        auto acc_frag_row = row_in_acc * RowsPerThreadAccFrag;
        auto acc_frag_start_for_sf = sf * AccFragsPerSF;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccFragsPerSF; ++i) {
          auto acc_frg = visit_results(0, row, acc_frag_start_for_sf + i);
          amax = amax_op(amax, acc_frg[acc_frag_row]);
          amax = amax_op(amax, acc_frg[acc_frag_row + 1]);
        }

        // At this point, each thread has computed the amax of the values that it owns for this SF vector.
        // We now need to compute the amax across threads. Because the TiledMMA uses an MmaThrLayout of <4,1,1>,
        // we know that all fragments in this row will belong to threads in this warp. Furthermore, because
        // SM120 narrow-precision MMAs have 16x8 output size with a quad owning two rows, we know that a quad
        // will own all of the elements to be reduced via amax. Therefore, we can use warp shuffle intrinsics
        // among threads in one quad to compute the amax.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < 3; ++i) {
          auto amax_other = __shfl_xor_sync(0xffffffff, amax, i);
          amax = amax_op(amax, amax_other);
        }

        if constexpr (IsInterWarpReductionNeeded) {
          // At this point, all threads in the quad have the amax for the elements of the accumulator owned by its quad
          // that should be used in computing the amax for this SF. Threads 0 in each quad of warps 0 and 2
          // (similarly, 1 and 3) now exchange amaxes to compute the final amax.
          if (thread_idx_in_quad == 0) {
            sAmaxs(quad_idx_in_warp, warp_idx) = amax;
          }
          synchronize();

          // Get the amax broadcasted by the warp with which we share.
          // Work on 4 warps per SFD generation
          if constexpr (WarpsPerSF == 4) {
            if constexpr (NumCollaboratingWarpGroups == 2) {
              // This implementation assumes warp layout 2 x 4.
              // For cooperative kernels (NumCollaboratingWarpGroups=2),
              // warp 0 shares with 2 / 4 / 6, warp 1 shares with 3 / 5/ 7.
              auto amax_other2 = sAmaxs(quad_idx_in_warp, warp_idx ^ 2);
              auto amax_other4 = sAmaxs(quad_idx_in_warp, warp_idx ^ 4);
              auto amax_other6 = sAmaxs(quad_idx_in_warp, warp_idx ^ 6);
              synchronize();
              amax = amax_op(amax, amax_other2);
              amax = amax_op(amax, amax_other4);
              amax = amax_op(amax, amax_other6);
            }
            else {
              static_assert(cutlass::detail::dependent_false<TiledCopy_>, "Unsupported warp layout.");
            }
          }
          // Work on 2 warps per SFD generation
          else if constexpr(WarpsPerSF == 2) {
            // For cooperative kernels (NumCollaboratingWarpGroups=2), 0 shares
            // with 4, 1 shares with 5, etc. For non-cooperative kernels
            // (NumCollaboratingWarpGroups=1), 0 shares with 2, 1 shares with 3.
            auto amax_other = sAmaxs(
                quad_idx_in_warp, warp_idx ^ (1 << NumCollaboratingWarpGroups));
            synchronize();
            amax = amax_op(amax, amax_other);
          }
        }

        ElementCompute pvscale = mul(amax, norm_constant_scaled_down);
        UnderlyingElementBlockScaleFactor qpvscale = NumericConverter<UnderlyingElementBlockScaleFactor, ElementCompute>{}(pvscale);
        tC_rSFD_flt(coord) = qpvscale;

        //
        // Apply the scale factor to the output
        //
        ElementCompute qpvscale_rcp = [&]() {
          if constexpr (cute::is_same_v<UnderlyingElementBlockScaleFactor, float_ue8m0_t>) {
            // UE8M0: Use integer subtraction to do the fast rcp in ue8m0 and then convert to float.
            auto e8m0_qpvscale_rcp = cutlass::reciprocal_approximate<UnderlyingElementBlockScaleFactor>{}(qpvscale);
            return cutlass::NumericConverter<ElementCompute, UnderlyingElementBlockScaleFactor>{}(e8m0_qpvscale_rcp);
          }
          else {
            // UE4M3: Do the rcp in fp32 data type.
            auto qpvscale_up = cutlass::NumericConverter<ElementCompute, UnderlyingElementBlockScaleFactor>{}(qpvscale);
            return cutlass::reciprocal_approximate_ftz<decltype(qpvscale_up)>{}(qpvscale_up);
          }
        }();

        ElementCompute acc_scale = mul(norm_constant, qpvscale_rcp);
        acc_scale = cutlass::minimum_with_nan_propagation<ElementCompute>{}(acc_scale, cutlass::platform::numeric_limits<ElementCompute>::max());

        // Compute quantized output values
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccFragsPerSF; ++i) {
          auto acc_frag = visit_results(0, row, acc_frag_start_for_sf + i);
          visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row    ] = mul(acc_frag[acc_frag_row], acc_scale);
          visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row + 1] = mul(acc_frag[acc_frag_row + 1], acc_scale);
        }
      } // sf

      // Since scale factors are computed cooperatively across two quads from two warps, we only need one thread from the
      // set of 8 cooperating threads to write out the data. We do this with thread 0 in each quad of the first warp that collaborates.
      bool write_sf = (thread_idx_in_quad == 0);
      if constexpr (NumCollaboratingWarpGroups == 2) {
        // For cooperative kernels (NumCollaboratingWarpGroups=2), 0 shares with 4, 1 shares with 5, etc.
        // Thus, only the warps in the first warpgroup need to write out scale factors.
        if constexpr (IsInterWarpReductionNeeded) {
          write_sf &= warp_idx < NumWarpsPerWarpGroup;
        }
      }
      else {
        if constexpr (IsInterWarpReductionNeeded) {
          // When non-cooperative kernels apply inter warp reduce, they are with
          // SF output rule as below :
          // 1. warp 0 shares with 2 and 1 shares with 3 within each warpgroup.
          // 2. warps 0 and 1 of the first warpgroup and 4 and 5 of the second
          //   warpgroup need to write output sf.
          write_sf &= ((warp_idx < 2) || (warpgroup_idx == 1 && warp_idx < 6));
        }
      }

      if (write_sf && elem_less(tC_cSFD(_0{}, _0{}, _0{}, epi_m, epi_n), residue_tC_cSFD)) {
        copy_aligned(tC_rSFD, tC_gSFD(_, _, _, _0{}, _0{}, get<0>(tile_coord_mn) + epi_m, get<1>(tile_coord_mn) + epi_n));
      }
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    using Sm1xxBlockScaledOutputConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize>;
    UnderlyingElementBlockScaleFactor* ptr_scale_factor = nullptr;
    // If Ptr-Array/Grouped GEMM with BlockScaleFactor per batch/group
    if constexpr (!cute::is_same_v<UnderlyingElementBlockScaleFactor, ElementBlockScaleFactor>) {
      ptr_scale_factor = params_ptr->ptr_scale_factor[l];
      l = 0;
    }
    else {
      ptr_scale_factor = params_ptr->ptr_scale_factor;
    }

    auto epi_tile_mn = shape<1>(zipped_divide(make_layout(take<0,2>(args.tile_shape_mnk)), args.epi_tile));
    Tensor mSFD = make_tensor(make_gmem_ptr(ptr_scale_factor), Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(args.problem_shape_mnkl));

    static_assert(size<1>(EpilogueTile{}) && ((size<1>(EpilogueTile{}) & (size<1>(EpilogueTile{}) - 1)) == 0), "Epilogue Tile N should be pow of 2");
    Tensor gSFD = local_tile(mSFD, args.epi_tile, make_coord(_, _,l));                             // (EPI_M,EPI_N, #EPI_Ms, #EPI_Ns)
    Tensor tCgSFD = sm90_partition_for_epilogue<ReferenceSrc>(                                     // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,#EPI_Ms, #EPI_Ns)
                        gSFD, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrSFD = make_tensor_like<UnderlyingElementBlockScaleFactor>(take<0,3>(cute::layout(tCgSFD)));    // (CPY,CPY_M,CPY_N)

    auto tile_coord_mn = make_coord(m * size<0>(epi_tile_mn), n * size<1>(epi_tile_mn));

    // Fetch and compute these during initialization
    Tensor mNormConst= make_tensor(make_gmem_ptr(params_ptr->norm_constant_ptr), make_layout(make_shape(M, N, L), params_ptr->norm_constant_stride));
    ElementCompute norm_constant = mNormConst(_0{},_0{},l);
    ElementCompute fp_max = ElementCompute(cutlass::platform::numeric_limits<ElementOutput>::max());
    ElementCompute scale_down_factor = cutlass::reciprocal_approximate_ftz<ElementCompute>{}(fp_max);
    ElementCompute norm_constant_scaled_down = cutlass::multiplies<ElementCompute>{}(norm_constant, scale_down_factor);

    Tensor sAmaxs = make_tensor(
      make_smem_ptr(smem_aux),
      make_layout(make_shape(Int<NumQuadsPerWarp>{}, Int<NumSyncWarps>{}))
    );

    return ConsumerStoreCallbacks(
      cute::move(tCrSFD),
      cute::move(tCgSFD),
      cute::move(sAmaxs),
      args.tCcD,
      args.residue_tCcD,
      params_ptr,
      tile_coord_mn,
      norm_constant,
      norm_constant_scaled_down,
      args.thread_idx,
      args.tiled_copy);

  }
};

// ── EVT tree alias: store node over (beta*C + alpha*acc), alpha kept = 1 ──────
template <
  int SFVecsize, class EpilogueTile, class CtaTileShapeMNK, int FragmentSize,
  class ElementOutput, class ElementCompute, class ElementBlockScaleFactor,
  class ElementSource = ElementOutput, class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm120SiluMulRowBlockScaleFactor =
  Sm90EVT<Sm120GeluMulBlockScaleFactorRowStore<SFVecsize, EpilogueTile, CtaTileShapeMNK, FragmentSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>,
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
    epilogue::Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    GeluMulBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm120SiluMulRowBlockScaleFactor<SFVecSize, EpilogueTile, CtaTileShapeMNK, FragmentSize, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm120SiluMulRowBlockScaleFactor<SFVecSize, EpilogueTile, CtaTileShapeMNK, FragmentSize, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = GeluMulBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor* block_scale_factor_ptr = nullptr;
    ElementCompute const* norm_constant_ptr = nullptr;
    using StrideNormConst = Stride<_0,_0,int64_t>;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};
    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};
    // SwiGLU per-projection scales (applied in visit()).
    ElementCompute alpha_gate = ElementCompute(1);
    ElementCompute alpha_up   = ElementCompute(1);

    operator typename Impl::Arguments() const {
      return
        {
          {    // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}},
            {},
            {                   // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}},
              {},
              {}
            },
            {}
          },
          // store-node args: ptr_scale_factor, norm_constant_ptr,
          // norm_constant_stride, alpha_gate, alpha_up
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst, alpha_gate, alpha_up}
        };
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion
