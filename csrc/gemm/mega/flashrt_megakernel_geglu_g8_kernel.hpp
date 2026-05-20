/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

// ============================================================================
// FlashRtMegakernelGeGLUG8FusedGemm — the megakernel kernel struct for the
// Pi0.5 encoder split-G7 path.
//
// Stage A.0 (this commit): byte-for-byte copy of the vendored production
// SM100 single-GEMM kernel (`flashrt_megakernel_kernel.hpp`) with ONLY the
// class name renamed.  Functionally identical to single GEMM at this
// stage; serves as a build/wiring scaffold.
//
// Stage A.1 (next): extend struct to TWO Mainloop+Epilogue template
// params; operator() body chains phase1 (gate) + phase2 (up) serially
// per work tile, sharing one tcgen05.alloc.  Both phases still write to
// gmem (no SMEM_gate routing yet).
//
// Stage B: redirect phase1 epilogue store target from gmem D_gate to a
// SharedStorage::smem_gate buffer (suppress TMA store).
//
// Stage C: add `Sm100SmemGateLoad` EVT visitor to phase2 epilogue and
// fuse the gate × up multiply in-register.  This is the cosine-correct
// MK-1 ship candidate.
//
// Stage D: share SMEM_A across phases (X is the same input for both
// gate and up).  Performance optimization on top of Stage C.
// ============================================================================

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/mainloop_fusion_helper_scale_factor.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/mma_atom.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

// Bring CUTLASS identifiers (used unqualified in the vendored kernel
// body) into scope.  In production these are found via ADL because
// the kernel lives in cutlass::gemm::kernel; in flashrt::megakernel
// they need explicit using-declarations.
using cutlass::gemm::KernelTmaWarpSpecializedSm100;
using cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100;
using cutlass::NumThreadsPerWarp;
using cutlass::MinWorkspaceAlignment;

// Primary template (forward declaration).  Mirrors
// cutlass::gemm::kernel::GemmUniversal but takes TWO mainloop+epilogue
// collectives for the gate / up chain.  Per-phase types must be
// structurally identical (same Tile, Cluster, TiledMma, EpilogueTile);
// they differ only in their TMA descriptors (gate loads W_gate; up
// loads W_up) and epilogue activation (gate=GELU, up=Identity).
template <
  class ProblemShape_,
  class CollectiveMainloop_,    // phase 1 (gate)
  class CollectiveEpilogue_,    // phase 1 (gate) — applies GELU
  class CollectiveMainloop2_,   // phase 2 (up)
  class CollectiveEpilogue2_,   // phase 2 (up) — Identity (no activation)
  class TileSchedulerTag_ = void,
  class Enable = void
>
class FlashRtMegakernelGeGLUG8FusedGemm;

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class CollectiveMainloop2_,
  class CollectiveEpilogue2_,
  class TileSchedulerTag_
>
class FlashRtMegakernelGeGLUG8FusedGemm<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  CollectiveMainloop2_,
  CollectiveEpilogue2_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cute::disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedSm100>,
    cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelTmaWarpSpecializedBlockScaledSm100>>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using LayoutSFA = typename cutlass::detail::LayoutSFAType<CollectiveMainloop>::type;
  using LayoutSFB = typename cutlass::detail::LayoutSFBType<CollectiveMainloop>::type;
  using ElementSF = typename cutlass::detail::ElementSFType<CollectiveMainloop>::type;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static constexpr bool IsComplex = CollectiveEpilogue::NumAccumulatorMtxs == 2;

  // ============================================================================
  // Phase 2 (up) collectives.  Must match phase 1 structurally — same
  // TileShape, ClusterShape, TiledMma, EpilogueTile, ElementA — so a
  // single TMEM allocation can be serially reused (slice_accumulator
  // semantics are well-defined only when the underlying acc shape is
  // identical).  These asserts catch mismatches at compile time.
  // ============================================================================
  using CollectiveMainloop2 = CollectiveMainloop2_;
  using CollectiveEpilogue2 = CollectiveEpilogue2_;
  using MainloopArguments2 = typename CollectiveMainloop2::Arguments;
  using MainloopParams2    = typename CollectiveMainloop2::Params;
  using EpilogueArguments2 = typename CollectiveEpilogue2::Arguments;
  using EpilogueParams2    = typename CollectiveEpilogue2::Params;

  static_assert(cute::is_same_v<typename CollectiveMainloop2::TileShape, TileShape>,
                "Phase 2 mainloop TileShape must match phase 1 (shared TMEM constraint).");
  static_assert(cute::is_same_v<typename CollectiveMainloop2::DispatchPolicy::ClusterShape, ClusterShape>,
                "Phase 2 mainloop ClusterShape must match phase 1.");
  static_assert(cute::is_same_v<typename CollectiveMainloop2::TiledMma, TiledMma>,
                "Phase 2 mainloop TiledMma must match phase 1 (TMEM accumulator layout depends on TiledMma).");
  static_assert(cute::is_same_v<typename CollectiveMainloop2::ElementA, ElementA>,
                "Phase 2 mainloop ElementA must match phase 1.");
  static_assert(cute::is_same_v<typename CollectiveEpilogue2::EpilogueTile, EpilogueTile>,
                "Phase 2 EpilogueTile must match phase 1.");

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  // TileID scheduler
  // Get Blk and Scheduling tile shapes
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using TileSchedulerTag = TileSchedulerTag_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMMAThreads          = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueThreads     = CollectiveEpilogue::ThreadCount;
  static constexpr uint32_t NumEpilogueWarps       = NumEpilogueThreads / NumThreadsPerWarp;

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_load_pipe_increment(CtaShape_MNK{});

  // Fixup performed for split-/stream-K is done across warps in different CTAs
  // at epilogue subtile granularity. Thus, there must be one barrier per sub-tile per
  // epilogue warp.
  static constexpr uint32_t NumFixupBarriers = 1;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  // Phase 2 thread-count must match phase 1 for shared CLC + AccPipe arrival counts.
  static_assert(CollectiveEpilogue2::ThreadCount == CollectiveEpilogue::ThreadCount,
                "Phase 2 epilogue ThreadCount must match phase 1 (shared CLC pipeline arrivals).");

  // Pipeline and pipeline state types — phase 1 (gate)
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  // Pipeline types — phase 2 (up).  Same template instances as phase 1
  // when the underlying CollectiveBuilder produced compatible mainloop /
  // epilogue collectives; the doubling is for separate SMEM rings and
  // separate producer/consumer state machines.
  using MainloopPipeline2 = typename CollectiveMainloop2::MainloopPipeline;
  using MainloopPipelineState2 = typename CollectiveMainloop2::MainloopPipelineState;

  using EpiLoadPipeline2 = typename CollectiveEpilogue2::LoadPipeline;
  using EpiLoadPipelineState2 = typename CollectiveEpilogue2::LoadPipelineState;

  using EpiStorePipeline2 = typename CollectiveEpilogue2::StorePipeline;
  using EpiStorePipelineState2 = typename CollectiveEpilogue2::StorePipelineState;

  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using CLCThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage  = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage   = typename CollectiveEpilogue::PipelineStorage;
      using MainloopPipelineStorage2 = typename CollectiveMainloop2::PipelineStorage;
      using EpiLoadPipelineStorage2  = typename CollectiveEpilogue2::PipelineStorage;
      using LoadOrderBarrierStorage  = typename LoadOrderBarrier::SharedStorage;
      using CLCPipelineStorage       = typename CLCPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

      // Doubled (per phase) — separate SMEM rings + state machines.
      alignas(16) MainloopPipelineStorage  mainloop;
      alignas(16) MainloopPipelineStorage2 mainloop_2;
      alignas(16) EpiLoadPipelineStorage   epi_load;
      alignas(16) EpiLoadPipelineStorage2  epi_load_2;
      // Single (shared across phases) — natural slot-wrap serialization.
      alignas(16) LoadOrderBarrierStorage     load_order;
      alignas(16) CLCPipelineStorage          clc;
      alignas(16) AccumulatorPipelineStorage  accumulator;
      alignas(16) CLCThrottlePipelineStorage  clc_throttle;
      alignas(16) arch::ClusterBarrier        tmem_dealloc;
    } pipelines;

    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    uint32_t tmem_base_ptr;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorStorage   = typename CollectiveEpilogue::TensorStorage;
      using EpilogueTensorStorage2  = typename CollectiveEpilogue2::TensorStorage;

      EpilogueTensorStorage   epilogue;
      EpilogueTensorStorage2  epilogue_2;

      // Stage E3: shared smem_A + per-phase smem_B
      using SmemAllocTypeA  = typename CollectiveMainloop::SmemAllocTypeA;
      using SmemAllocTypeB  = typename CollectiveMainloop::SmemAllocTypeB;
      using SmemAllocTypeB2 = typename CollectiveMainloop2::SmemAllocTypeB;
      static constexpr int SmemA_Elems  = cute::cosize_v<typename CollectiveMainloop::SmemLayoutA>;
      static constexpr int SmemB1_Elems = cute::cosize_v<typename CollectiveMainloop::SmemLayoutB>;
      static constexpr int SmemB2_Elems = cute::cosize_v<typename CollectiveMainloop2::SmemLayoutB>;

      cute::ArrayEngine<SmemAllocTypeA,  SmemA_Elems>  shared_smem_A;
      cute::ArrayEngine<SmemAllocTypeB,  SmemB1_Elems> smem_B_phase1;
      cute::ArrayEngine<SmemAllocTypeB2, SmemB2_Elems> smem_B_phase2;
    } tensors;

    template <class SmemABuf, class SmemBBuf>
    struct PhaseTensorStorageView {
      SmemABuf& smem_A;
      SmemBBuf& smem_B;
    };
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Host facing host arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    // Phase 1 (gate)
    MainloopArguments  mainloop{};
    EpilogueArguments  epilogue{};
    // Phase 2 (up)
    MainloopArguments2 mainloop_2{};
    EpilogueArguments2 epilogue_2{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel device entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams  mainloop{};
    EpilogueParams  epilogue{};
    MainloopParams2 mainloop_2{};
    EpilogueParams2 epilogue_2{};
    TileSchedulerParams scheduler{};
    KernelHardwareInfo hw_info{};
  };

  enum class WarpCategory : int32_t {
    MMA          = 0,
    Sched        = 1,
    MainloopLoad = 2,
    EpilogueLoad = 3,
    Epilogue     = 4
  };

  struct IsParticipant {
    uint32_t mma       = false;
    uint32_t sched     = false;
    uint32_t main_load = false;
    uint32_t epi_load  = false;
    uint32_t epilogue  = false;
  };

  //
  // Methods
  //

  // Convert to underlying arguments.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    auto problem_shape = args.problem_shape;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count != 0) {
      CUTLASS_TRACE_HOST("  WARNING: SM100 tile scheduler does not allow for user specified SM counts.\n"
          "  To restrict a kernel's resource usage, consider using CUDA driver APIs instead (green contexts).");
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue phase 1 (gate)
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    // Epilogue phase 2 (up)
    void* epilogue_workspace_2 = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue2::get_workspace_size(args.problem_shape, args.epilogue_2);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;

    // Tile scheduler
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace, args.hw_info),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      CollectiveMainloop2::to_underlying_arguments(args.problem_shape, args.mainloop_2, mainloop_workspace, args.hw_info),
      CollectiveEpilogue2::to_underlying_arguments(args.problem_shape, args.epilogue_2, epilogue_workspace_2),
      TileScheduler::to_underlying_arguments(
        problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
        args.hw_info, args.scheduler, scheduler_workspace
      )
      ,args.hw_info
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= CollectiveMainloop2::can_implement(args.problem_shape, args.mainloop_2);
    implementable &= CollectiveEpilogue2::can_implement(args.problem_shape, args.epilogue_2);
    implementable &= TileScheduler::can_implement(args.scheduler);

    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
    }

    constexpr bool IsBlockscaled = !cute::is_void_v<ElementSF>;
    if constexpr (IsBlockscaled) {
      if constexpr (IsDynamicCluster) {
        implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
        // Special cluster shape check for scale factor multicasts. Due to limited size of scale factors, we can't multicast among
        // more than 4 CTAs
        implementable &= (args.hw_info.cluster_shape.x <= 4 && args.hw_info.cluster_shape.y <= 4 &&
                          args.hw_info.cluster_shape_fallback.x <= 4 && args.hw_info.cluster_shape_fallback.y <= 4);
      }
      else {
        // Special cluster shape check for scale factor multicasts. Due to limited size of scale factors, we can't multicast among
        // more than 4 CTAs
        implementable &= ((size<0>(ClusterShape{}) <= 4) && (size<1>(ClusterShape{}) <= 4));
      }
    }

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    // Epilogue phase 1 (gate)
    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    // Epilogue phase 2 (up)
    workspace_size += CollectiveEpilogue2::get_workspace_size(args.problem_shape, args.epilogue_2);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    // Tile scheduler
    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue phase 1 (gate)
    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    // Epilogue phase 2 (up)
    status = CollectiveEpilogue2::initialize_workspace(args.problem_shape, args.epilogue_2, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue2::get_workspace_size(args.problem_shape, args.epilogue_2);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    // Tile scheduler
    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // NOTE cluster_shape here is the major cluster shape, not fallback one
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_grid_shape(
        params.scheduler,
        problem_shape_MNKL,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator() (Params const& params, char* smem_buf) {

    using namespace cute;
    using X = Underscore;

    static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");
    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_shape_MNKL;

    // Account for more than one epilogue warp
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx)
                                                                                     : WarpCategory::Epilogue;

    uint32_t lane_predicate = cute::elect_one_sync();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});
    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_first_cta_in_cluster = cta_rank_in_cluster == 0;
    int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;
    constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;
    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop  collective_mainloop  (params.mainloop,   cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue  collective_epilogue  (params.epilogue,   shared_storage.tensors.epilogue);
    CollectiveMainloop2 collective_mainloop_2(params.mainloop_2, cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue2 collective_epilogue_2(params.epilogue_2, shared_storage.tensors.epilogue_2);

    // Issue Tma Descriptor Prefetch from a single thread (both phases).
    if ((warp_category == WarpCategory::Sched) && lane_predicate) {
      collective_mainloop.prefetch_tma_descriptors();
      collective_mainloop_2.prefetch_tma_descriptors();
    }
    if ((warp_category == WarpCategory::EpilogueLoad) && lane_predicate) {
      collective_epilogue.prefetch_tma_descriptors(params.epilogue);
      collective_epilogue_2.prefetch_tma_descriptors(params.epilogue_2);
    }

    // Do we load source tensor C or other aux inputs (either phase)
    bool is_epi_load_needed_1 = collective_epilogue.is_producer_load_needed();
    bool is_epi_load_needed_2 = collective_epilogue_2.is_producer_load_needed();
    bool is_epi_load_needed   = is_epi_load_needed_1 || is_epi_load_needed_2;
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                 // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopLoad),                        // main_load
      (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,  // epi_load
      (warp_category == WarpCategory::Epilogue)                             // epilogue
    };

    // Mainloop Load pipeline — phase 1 (gate)
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load;
    // Stage E3: shared pipeline carries (A + B_phase1 + B_phase2) per slot.
    mainloop_pipeline_params.transaction_bytes =
        CollectiveMainloop::TmaTransactionBytesA
      + CollectiveMainloop::TmaTransactionBytesB
      + CollectiveMainloop2::TmaTransactionBytesB;
    // Stage E3: 2 consumer groups per slot — phase 1 mma + phase 2 mma must
    // each release every slot (K-tile interleaved in MMA warp body below).
    mainloop_pipeline_params.num_consumers = 2;
    mainloop_pipeline_params.initializing_warp = 0;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                       mainloop_pipeline_params,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::false_type{}); // Delay mask calculation

    // Mainloop Load pipeline — phase 2 (up).  Separate SMEM ring + state
    // machine; both producer (main_load warp) and consumer (mma warp)
    // run their phase-2 calls right after their phase-1 calls in the
    // same persistent loop iteration.
    typename MainloopPipeline2::Params mainloop_pipeline_2_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      mainloop_pipeline_2_params.role = MainloopPipeline2::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_pipeline_2_params.role = MainloopPipeline2::ThreadCategory::Consumer;
    }
    mainloop_pipeline_2_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load;
    mainloop_pipeline_2_params.transaction_bytes = CollectiveMainloop2::TmaTransactionBytes;
    // Use a distinct initializing_warp index (6 — outside the {0..5}
    // already claimed by phase-1 pipelines + LoadOrder/CLC/Acc/Throttle)
    // so two warps don't race on the same barrier init.
    mainloop_pipeline_2_params.initializing_warp = 6;
    MainloopPipeline2 mainloop_pipeline_2(shared_storage.pipelines.mainloop_2,
                                          mainloop_pipeline_2_params,
                                          cluster_shape,
                                          cute::true_type{},
                                          cute::false_type{});

    // Epilogue Load pipeline — phase 1 (gate)
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cta_rank_in_cluster;
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    epi_load_pipeline_params.initializing_warp = 1;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Load pipeline — phase 2 (up)
    typename EpiLoadPipeline2::Params epi_load_pipeline_2_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_load_pipeline_2_params.role = EpiLoadPipeline2::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_load_pipeline_2_params.role = EpiLoadPipeline2::ThreadCategory::Consumer;
    }
    epi_load_pipeline_2_params.dst_blockid = cta_rank_in_cluster;
    epi_load_pipeline_2_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_2_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_2_params.transaction_bytes = CollectiveEpilogue2::TmaTransactionBytes;
    epi_load_pipeline_2_params.initializing_warp = 7;
    EpiLoadPipeline2 epi_load_pipeline_2(shared_storage.pipelines.epi_load_2, epi_load_pipeline_2_params);

    // Epilogue Store pipelines — local-scope, no SharedStorage state.
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename EpiStorePipeline2::Params epi_store_pipeline_2_params;
    epi_store_pipeline_2_params.always_wait = true;
    EpiStorePipeline2 epi_store_pipeline_2(epi_store_pipeline_2_params);

    // Load order barrier
    typename LoadOrderBarrier::Params load_order_barrier_params;
    load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopLoad) ? 0 : 1;
    load_order_barrier_params.group_size = NumMainloopLoadThreads;
    load_order_barrier_params.initializing_warp = 3;
    LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_barrier_params);

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    }
    else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.producer_blockid = 0;
    clc_pipeline_params.producer_arv_count = 1;
    clc_pipeline_params.consumer_arv_count = NumSchedThreads + cluster_size *
                                                 (NumMainloopLoadThreads + NumEpilogueThreads + NumMMAThreads);
    if (is_epi_load_needed) {
      clc_pipeline_params.consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
    }
    clc_pipeline_params.transaction_bytes = CLCResponseSize;
    clc_pipeline_params.initializing_warp = 4;
    CLCPipeline clc_pipeline(shared_storage.pipelines.clc, clc_pipeline_params, cluster_shape);

    // Mainloop-Epilogue pipeline
    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (WarpCategory::MMA == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
    }
    // Only one producer thread arrives on this barrier.
    accumulator_pipeline_params.producer_arv_count = 1;
    accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    accumulator_pipeline_params.initializing_warp = 5;
    AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator,
                                             accumulator_pipeline_params,
                                             cluster_shape,
                                             cute::true_type{},   // Perform barrier init
                                             cute::false_type{}); // Delay mask calculation

    // CLC throttle pipeline
    typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Sched == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Consumer;
    }
    clc_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
    clc_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
    clc_throttle_pipeline_params.dst_blockid = 0;
    clc_throttle_pipeline_params.initializing_warp = 3;
    CLCThrottlePipeline clc_throttle_pipeline(shared_storage.pipelines.clc_throttle, clc_throttle_pipeline_params);
    CLCThrottlePipelineState clc_pipe_throttle_consumer_state;
    CLCThrottlePipelineState clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();

    // Tmem allocator
    TmemAllocator tmem_allocator{};

    // Sync allocation status between MMA and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumMMAThreads + NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    if (WarpCategory::MMA == warp_category) {
      if constexpr(!IsOverlappingAccum) {
        if (has_mma_peer_cta && lane_predicate) {
          tmem_deallocation_result_barrier.init(NumMMAThreads);
        }
      }
      else {
        if (has_mma_peer_cta && lane_predicate) {
          tmem_deallocation_result_barrier.init(NumEpilogueThreads*2);
        }
        else if (lane_predicate) {
          tmem_deallocation_result_barrier.init(NumEpilogueThreads);
        }
      }
    }

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    // Stage E3: phase 1 and phase 2 mainloops both see shared smem_A.
    typename SharedStorage::template PhaseTensorStorageView<
        decltype(shared_storage.tensors.shared_smem_A),
        decltype(shared_storage.tensors.smem_B_phase1)>
      phase1_view{shared_storage.tensors.shared_smem_A, shared_storage.tensors.smem_B_phase1};
    typename SharedStorage::template PhaseTensorStorageView<
        decltype(shared_storage.tensors.shared_smem_A),
        decltype(shared_storage.tensors.smem_B_phase2)>
      phase2_view{shared_storage.tensors.shared_smem_A, shared_storage.tensors.smem_B_phase2};

    auto load_inputs   = collective_mainloop.load_init  (problem_shape_MNKL, phase1_view);
    auto load_inputs_2 = collective_mainloop_2.load_init(problem_shape_MNKL, phase2_view);

    MainloopPipelineState  mainloop_pipe_consumer_state;
    MainloopPipelineState  mainloop_pipe_producer_state   = cutlass::make_producer_start_state<MainloopPipeline>();
    MainloopPipelineState2 mainloop_pipe_consumer_state_2;
    MainloopPipelineState2 mainloop_pipe_producer_state_2 = cutlass::make_producer_start_state<MainloopPipeline2>();

    EpiLoadPipelineState  epi_load_pipe_consumer_state;
    EpiLoadPipelineState  epi_load_pipe_producer_state   = cutlass::make_producer_start_state<EpiLoadPipeline>();
    EpiLoadPipelineState2 epi_load_pipe_consumer_state_2;
    EpiLoadPipelineState2 epi_load_pipe_producer_state_2 = cutlass::make_producer_start_state<EpiLoadPipeline2>();

    // epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    EpiStorePipelineState  epi_store_pipe_producer_state   = cutlass::make_producer_start_state<EpiStorePipeline>();
    EpiStorePipelineState2 epi_store_pipe_producer_state_2 = cutlass::make_producer_start_state<EpiStorePipeline2>();

    CLCPipelineState clc_pipe_consumer_state;
    CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    AccumulatorPipelineState accumulator_pipe_consumer_state;
    AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

    dim3 block_id_in_cluster = cute::block_id_in_cluster();

    // Calculate mask after cluster barrier arrival (both mainloop pipelines).
    mainloop_pipeline.init_masks(cluster_shape, block_id_in_cluster);
    mainloop_pipeline_2.init_masks(cluster_shape, block_id_in_cluster);
    accumulator_pipeline.init_masks(cluster_shape, block_id_in_cluster);

    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    //
    // TMEM "Allocation"
    //
    auto tmem_storage = collective_mainloop.template init_tmem_tensors<EpilogueTile, IsOverlappingAccum>(EpilogueTile{});

    pipeline_init_wait(cluster_size);

    if (is_participant.main_load) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_arrive = is_epi_load_needed;
      bool requires_clc_query = true;

      do {
        // K-tile iterator/count: both phases share the same K dim (X has
        // the same K both times; W_gate and W_up have the same K dim).
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, load_inputs.k_tiles);
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);

        if constexpr (IsSchedDynamicPersistent) {
          if (is_first_cta_in_cluster && requires_clc_query) {
            clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
            clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);
            ++clc_pipe_throttle_producer_state;
          }
        }

        // -------- Stage E3: inline 3-TMA shared-pipeline main_load --------
        {
          auto& tma_load_a    = params.mainloop.tma_load_a;
          auto& tma_load_b_p1 = params.mainloop.tma_load_b;
          auto& tma_load_b_p2 = params.mainloop_2.tma_load_b;

          auto atom_thr_M = size(typename TiledMma::AtomThrID{});
          Tensor tAgA  = load_inputs.tAgA_mkl(cute::_, get<0>(cta_coord_mnkl) / atom_thr_M, cute::_, get<3>(cta_coord_mnkl));
          Tensor tBgB1 = load_inputs.tBgB_nkl(cute::_, get<1>(cta_coord_mnkl), cute::_, get<3>(cta_coord_mnkl));
          Tensor tBgB2 = load_inputs_2.tBgB_nkl(cute::_, get<1>(cta_coord_mnkl), cute::_, get<3>(cta_coord_mnkl));

          auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);
          int k_remaining = k_tile_count;
          bool did_load_order_arrive_local = false;

          CUTLASS_PRAGMA_NO_UNROLL
          while (k_remaining > 0) {
            mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);
            using BarrierType = typename MainloopPipeline::ProducerBarrierType;
            BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);
            int write_stage = mainloop_pipe_producer_state.index();
            ++mainloop_pipe_producer_state;
            barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

            if (cute::elect_one_sync()) {
              cute::copy(tma_load_a   .with(*tma_barrier, load_inputs.mcast_mask_a),   tAgA (cute::_, *k_tile_iter), load_inputs.tAsA (cute::_, write_stage));
              cute::copy(tma_load_b_p1.with(*tma_barrier, load_inputs.mcast_mask_b),   tBgB1(cute::_, *k_tile_iter), load_inputs.tBsB (cute::_, write_stage));
              cute::copy(tma_load_b_p2.with(*tma_barrier, load_inputs_2.mcast_mask_b), tBgB2(cute::_, *k_tile_iter), load_inputs_2.tBsB(cute::_, write_stage));
            }
            if (do_load_order_arrive && !did_load_order_arrive_local) {
              load_order_barrier.arrive();
              do_load_order_arrive = false;
              did_load_order_arrive_local = true;
            }
            --k_remaining;
            ++k_tile_iter;
          }
        }
        (void)k_tile_prologue;  // unused in Stage E3 inline main_load

        // Sync warp to prevent non-participating threads entering next wave early
        __syncwarp();
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        requires_clc_query = increment_pipe;
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());
      // Stage E3: single shared pipeline; mainloop_pipeline_2 unused.
      collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

    }

    else if (is_participant.sched) {
      if constexpr (IsSchedDynamicPersistent) {
        // Whether a new CLC query must be performed.
        // See comment below where this variable is updated for a description of
        // why this variable is needed.
        bool requires_clc_query = true;

        cutlass::arch::wait_on_dependent_grids();

        do {
          if (requires_clc_query) {
            // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
            clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
            clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
            ++clc_pipe_throttle_consumer_state;

            // Query next clcID and update producer state
            clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          }

          // Fetch next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
            work_tile_info,
            clc_pipeline,
            clc_pipe_consumer_state
          );

          // Only perform a new CLC query if we consumed a new CLC query result in
          // `fetch_next_work`. An example of a case in which CLC `fetch_next_work` does
          // not consume a new CLC query response is when processing stream-K units.
          // The current stream-K scheduler uses single WorkTileInfo to track multiple
          // (potentially-partial) tiles to be computed via stream-K. In this case,
          // `fetch_next_work` simply performs in-place updates on the existing WorkTileInfo,
          // rather than consuming a CLC query response.
          requires_clc_query = increment_pipe;
          if (increment_pipe) {
            ++clc_pipe_consumer_state;
          }

          work_tile_info = next_work_tile_info;
        } while (work_tile_info.is_valid());
        clc_pipeline.producer_tail(clc_pipe_producer_state);
      }
    }

    else if (is_participant.mma) {
      // TMEM allocation sequence — ONE allocate() per kernel.  Both
      // phases re-bind their logical tmem_storage to the same base
      // address; physical columns are time-multiplexed across phases
      // via consecutive AccPipe slot indices.
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop  .set_tmem_offsets(tmem_storage, tmem_base_ptr);

      typename SharedStorage::template PhaseTensorStorageView<
          decltype(shared_storage.tensors.shared_smem_A),
          decltype(shared_storage.tensors.smem_B_phase1)>
        phase1_view_mma{shared_storage.tensors.shared_smem_A, shared_storage.tensors.smem_B_phase1};
      typename SharedStorage::template PhaseTensorStorageView<
          decltype(shared_storage.tensors.shared_smem_A),
          decltype(shared_storage.tensors.smem_B_phase2)>
        phase2_view_mma{shared_storage.tensors.shared_smem_A, shared_storage.tensors.smem_B_phase2};

      auto mma_inputs   = collective_mainloop.mma_init  (tmem_storage, phase1_view_mma);
      auto mma_inputs_2 = collective_mainloop_2.mma_init(tmem_storage, phase2_view_mma);

      do {
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        // -------- Phase 1 (gate) MMA --------
        int acc_stage = [&] () {
          if constexpr (IsOverlappingAccum) {
            return accumulator_pipe_producer_state.phase() ^ 1;
          }
          else {
            return accumulator_pipe_producer_state.index();
          }
        }();

        // -------- Stage E3: inline K-tile interleaved phase1+phase2 MMA --------
        // Single shared mainloop_pipeline.  Per K-tile slot: phase 1 mma
        // (A, B_phase1) → acc_p1, then phase 2 mma (A, B_phase2) → acc_p2.
        // Both phases release the slot each iteration so pipeline reaches
        // the 2x consumer arrival count and producer can wrap.
        int acc_stage_2 = [&] () {
          // Compute phase 2's acc slot — uses NEXT acc_pipe slot from phase 1.
          if constexpr (IsOverlappingAccum) {
            return ((accumulator_pipe_producer_state.phase() ^ 1) + 1) % AccumulatorPipelineStageCount;
          }
          else {
            return (accumulator_pipe_producer_state.index() + 1) % AccumulatorPipelineStageCount;
          }
        }();

        if (is_mma_leader_cta) {
          // Acquire acc pipe slots for both phases.
          accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
          AccumulatorPipelineState acc_state_p2 = accumulator_pipe_producer_state;
          ++acc_state_p2;
          accumulator_pipeline.producer_acquire(acc_state_p2);

          // Get phase 1 and phase 2 accumulators (different TMEM column slots).
          auto acc_p1 = get<0>(collective_mainloop.slice_accumulator(tmem_storage, acc_stage));
          auto acc_p2 = get<0>(collective_mainloop_2.slice_accumulator(tmem_storage, acc_stage_2));

          // Unpack mma_inputs.
          auto [tiled_mma_p1, tCrA_p1, tCrB_p1] = mma_inputs;
          auto [tiled_mma_p2, tCrA_p2, tCrB_p2] = mma_inputs_2;
          tiled_mma_p1.accumulate_ = cute::UMMA::ScaleOut::Zero;
          tiled_mma_p2.accumulate_ = cute::UMMA::ScaleOut::Zero;

          // K-tile interleaved loop.
          uint32_t skip_wait = k_tile_count <= 0;
          auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);
          int k_remaining = k_tile_count;
          CUTLASS_PRAGMA_NO_UNROLL
          while (k_remaining > 0) {
            mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
            int read_stage = mainloop_pipe_consumer_state.index();
            auto cur_state = mainloop_pipe_consumer_state;
            ++mainloop_pipe_consumer_state;
            --k_remaining;
            skip_wait = k_remaining <= 0;
            barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

            // Phase 1 tcgen05.mma (A from shared, B from smem_B_phase1)
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tCrA_p1); ++k_block) {
              cute::gemm(tiled_mma_p1,
                         tCrA_p1(cute::_, cute::_, k_block, read_stage),
                         tCrB_p1(cute::_, cute::_, k_block, read_stage),
                         acc_p1);
              tiled_mma_p1.accumulate_ = cute::UMMA::ScaleOut::One;
            }
            // Phase 1 release this slot (1 of 2 consumer arrivals)
            mainloop_pipeline.consumer_release(cur_state);

            // Phase 2 tcgen05.mma (A from SAME shared, B from smem_B_phase2)
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tCrA_p2); ++k_block) {
              cute::gemm(tiled_mma_p2,
                         tCrA_p2(cute::_, cute::_, k_block, read_stage),
                         tCrB_p2(cute::_, cute::_, k_block, read_stage),
                         acc_p2);
              tiled_mma_p2.accumulate_ = cute::UMMA::ScaleOut::One;
            }
            // Phase 2 release this slot (2 of 2 consumer arrivals → slot free)
            mainloop_pipeline.consumer_release(cur_state);
          }

          // Commit both phases' acc slots.
          accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
          accumulator_pipeline.producer_commit(acc_state_p2);
        }
        // Advance acc pipe state by 2 (both phases committed) for next work tile.
        ++accumulator_pipe_producer_state;
        ++accumulator_pipe_producer_state;

        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      // Release the right to allocate before deallocations so that the next CTA can rasterize
      tmem_allocator.release_allocation_lock();

      if constexpr (!IsOverlappingAccum) {
        // Leader MMA waits for leader + peer epilogues to release accumulator stage
        if (is_mma_leader_cta) {
          accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
        }
        // Signal to peer MMA that entire tmem allocation can be deallocated
        if constexpr (has_mma_peer_cta) {
          // Leader does wait + arrive, follower does arrive + wait
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
          tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
        }
      }
      else {
        tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
      }

      // Free entire tmem allocation
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

    else if (is_participant.epi_load) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_wait = true;
      bool do_tail_load = false;
      int current_wave = 0;

      do {
        bool compute_epilogue = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

        // Get current work tile and fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        if (compute_epilogue) {
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          bool reverse_epi_n = IsOverlappingAccum && (current_wave % 2 == 0);

          // Phase 1 (gate) epilogue producer-load — only fires if the
          // phase-1 epilogue actually consumes source C/aux.  For
          // LinComb beta=0 path this is a fast no-op.
          if (is_epi_load_needed_1) {
            epi_load_pipe_producer_state = collective_epilogue.template load<IsOverlappingAccum>(
              epi_load_pipeline,
              epi_load_pipe_producer_state,
              problem_shape_MNKL,
              CtaShape_MNK{},
              cta_coord_mnkl,
              TileShape{},
              TiledMma{},
              shared_storage.tensors.epilogue,
              reverse_epi_n
            );
          }

          // Phase 2 (up) epilogue producer-load
          if (is_epi_load_needed_2) {
            epi_load_pipe_producer_state_2 = collective_epilogue_2.template load<IsOverlappingAccum>(
              epi_load_pipeline_2,
              epi_load_pipe_producer_state_2,
              problem_shape_MNKL,
              CtaShape_MNK{},
              cta_coord_mnkl,
              TileShape{},
              TiledMma{},
              shared_storage.tensors.epilogue_2,
              reverse_epi_n
            );
          }

          do_tail_load = true;
        }
        current_wave++;

        // Calculate the cta coordinates of the next work tile
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Only perform a tail load if one of the work units processed performed
      // an epilogue load.  Both phases drain.
      if (do_tail_load) {
        if (is_epi_load_needed_1) {
          collective_epilogue.load_tail(
            epi_load_pipeline, epi_load_pipe_producer_state,
            epi_store_pipeline, epi_store_pipe_producer_state);
        }
        if (is_epi_load_needed_2) {
          collective_epilogue_2.load_tail(
            epi_load_pipeline_2, epi_load_pipe_producer_state_2,
            epi_store_pipeline_2, epi_store_pipe_producer_state_2);
        }
      }
    }

    else if (is_participant.epilogue) {
      // Wait for tmem allocate here
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

      bool do_tail_store = false;
      do {
        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        // -------- Phase 1 (gate) epilogue --------
        int acc_stage = [&] () {
          if constexpr (IsOverlappingAccum) {
            return accumulator_pipe_consumer_state.phase();
          }
          else {
            return accumulator_pipe_consumer_state.index();
          }
        }();

        auto accumulator = get<0>(collective_mainloop.slice_accumulator(tmem_storage, acc_stage));
        accumulator_pipe_consumer_state = scheduler.template fixup<IsComplex>(
          TiledMma{},
          work_tile_info,
          accumulator,
          accumulator_pipeline,
          accumulator_pipe_consumer_state,
          typename CollectiveEpilogue::CopyOpT2R{}
        );

        if (scheduler.compute_epilogue(work_tile_info)) {
          auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue.template store<IsOverlappingAccum>(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            accumulator_pipeline,
            accumulator_pipe_consumer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            accumulator,
            shared_storage.tensors.epilogue
          );
          epi_load_pipe_consumer_state    = load_state_next;
          epi_store_pipe_producer_state   = store_state_next;
          accumulator_pipe_consumer_state = acc_state_next;
          do_tail_store = true;
        }

        // -------- Cross-phase SMEM bridge --------
        // Phase 1's Sm100SmemAuxStore visitor wrote gate values to its own
        // SharedStorage::smem_aux (inside collective_epilogue's TensorStorage).
        // Phase 2's Sm100SmemAuxLoad visitor reads from its own SharedStorage,
        // which is a DIFFERENT region.  Copy phase 1 → phase 2 here.
        // S2S is fast (~8 KB per tile).
        {
          // Access pointers via the FusionCallbacks visitor tree.  ops tuple
          // ordering: children-first, node-last.  For both phases the
          // SmemAux{Store,Load} visitor sits at ops index 1.
          auto& aux_store = cute::get<1>(collective_epilogue.fusion_callbacks.ops);
          auto& aux_load  = cute::get<1>(collective_epilogue_2.fusion_callbacks.ops);
          auto* src = aux_store.smem_aux;
          auto* dst = aux_load.smem_aux;
          using SmemLayout = typename cute::remove_reference_t<decltype(aux_store)>::SmemLayout;
          constexpr int N_elts = cute::cosize_v<SmemLayout>;
          // Epilogue threads are warps 4..(NumEpilogueWarps+3) — offset by
          // (4 * NumThreadsPerWarp).  Use epilogue-local index 0..255.
          int local_tid = threadIdx.x - (4 * NumThreadsPerWarp);
          // Real S2S copy: phase 1 smem_aux → phase 2 smem_aux.
          for (int i = local_tid; i < N_elts; i += NumEpilogueThreads) {
            dst[i] = src[i];
          }
          // Fence + barrier so dst writes visible before phase 2 epilogue reads.
          // Use a barrier id distinct from CUTLASS's EpilogueBarrier (used
          // internally by Sm100 epilogue's NamedBarrier::sync()).
          cutlass::arch::fence_view_async_shared();
          cutlass::arch::NamedBarrier(NumEpilogueThreads, /*barrier_id=*/0).sync();
        }

        // -------- Phase 2 (up) epilogue --------
        // Consumer state has advanced past phase 1's slot; phase 2
        // consumes the next AccPipe slot which holds phase 2's MMA result
        // for this work tile.
        int acc_stage_2 = [&] () {
          if constexpr (IsOverlappingAccum) {
            return accumulator_pipe_consumer_state.phase();
          }
          else {
            return accumulator_pipe_consumer_state.index();
          }
        }();

        auto accumulator_2 = get<0>(collective_mainloop_2.slice_accumulator(tmem_storage, acc_stage_2));
        accumulator_pipe_consumer_state = scheduler.template fixup<IsComplex>(
          TiledMma{},
          work_tile_info,
          accumulator_2,
          accumulator_pipeline,
          accumulator_pipe_consumer_state,
          typename CollectiveEpilogue2::CopyOpT2R{}
        );

        if (scheduler.compute_epilogue(work_tile_info)) {
          auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue_2.template store<IsOverlappingAccum>(
            epi_load_pipeline_2,
            epi_load_pipe_consumer_state_2,
            epi_store_pipeline_2,
            epi_store_pipe_producer_state_2,
            accumulator_pipeline,
            accumulator_pipe_consumer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            accumulator_2,
            shared_storage.tensors.epilogue_2
          );
          epi_load_pipe_consumer_state_2    = load_state_next;
          epi_store_pipe_producer_state_2   = store_state_next;
          accumulator_pipe_consumer_state   = acc_state_next;
          do_tail_store = true;
        }

        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

      } while (work_tile_info.is_valid());

      if constexpr (IsOverlappingAccum) {
        // Signal to peer MMA that Full TMEM alloc can be deallocated
        if constexpr (has_mma_peer_cta) {
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank);
        }
        tmem_deallocation_result_barrier.arrive();
      }

      // Tail store for both phases.
      if (do_tail_store) {
        collective_epilogue.store_tail(
          epi_load_pipeline, epi_load_pipe_consumer_state,
          epi_store_pipeline, epi_store_pipe_producer_state,
          CtaShape_MNK{});
        collective_epilogue_2.store_tail(
          epi_load_pipeline_2, epi_load_pipe_consumer_state_2,
          epi_store_pipeline_2, epi_store_pipe_producer_state_2,
          CtaShape_MNK{});
      }
    }

    else {
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
