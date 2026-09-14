// ============================================================================
//  FlashRT — persistent dependent-GEMM sequence kernel for SM100/SM110
//  block-scaled (NVFP4) mainloops.
//
//  One launch runs a list of GEMM problems back to back on a resident grid
//  (one CTA per SM). Between problems the CTAs synchronise on a global
//  counter; the load warp streams the next problem's weight k-tiles (the
//  early operand of the forked mainloop) before it waits, so the DRAM stream
//  stays busy across the boundary, and the epilogue warps may run an
//  elementwise phase after the barrier before the activation loads are
//  released. Warp roles, pipelines, TMEM handling and the tile loop are
//  those of the SM100 warp-specialized kernel (sm100_gemm_seq_kernel.hpp)
//  with the static persistent tile scheduler.
// ============================================================================
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/mma_atom.hpp"
#include "gemm/fp4/sm100_blockscaled_mma_earlyb.hpp"
#include "gemm/fp4/sm100_seq_phases.hpp"

namespace cutlass::gemm::kernel {

namespace seq_detail {
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
// Spin until the sequence counter reaches `target` (all CTAs arrived at an earlier point).
__device__ __forceinline__ void seq_wait(const int* counter, int target) {
  while (ld_acquire_gpu(counter) < target) { __nanosleep(128); }
}
}  // namespace seq_detail

template <
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  int kMaxProblems_
>
class GemmSeqPersistent {
public:
  using ProblemShape = cute::Shape<int, int, int, int>;
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using ElementSF = typename CollectiveMainloop::ElementSF;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;

  static constexpr int kMaxProblems = kMaxProblems_;
  static constexpr int SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr int AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  using TileScheduler = typename detail::TileSchedulerSelector<
    StaticPersistentScheduler, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  static_assert(!TileScheduler::IsDynamicPersistent, "the sequence kernel uses the static persistent scheduler");
  static_assert(cute::is_static_v<ClusterShape>, "static cluster shape only");

  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;
  static constexpr uint32_t NumMMAThreads          = NumThreadsPerWarp;
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueThreads     = CollectiveEpilogue::ThreadCount;
  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_load_pipe_increment(CtaShape_MNK{});
  static constexpr uint32_t NumFixupBarriers = 1;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;
  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;
  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;
  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;
  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;
  using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, cute::_1> {
      alignas(16) typename CollectiveMainloop::PipelineStorage mainloop;
      alignas(16) typename CollectiveEpilogue::PipelineStorage epi_load;
      alignas(16) typename LoadOrderBarrier::SharedStorage load_order;
      alignas(16) typename AccumulatorPipeline::SharedStorage accumulator;
      alignas(16) arch::ClusterBarrier tmem_dealloc;
    } pipelines;
    uint32_t tmem_base_ptr;
    SeqPhaseSmem phase;
    struct TensorStorage : cute::aligned_struct<128, cute::_1> {
      typename CollectiveEpilogue::TensorStorage epilogue;
      typename CollectiveMainloop::TensorStorage mainloop;
    } tensors;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Elementwise phase run by the epilogue warps after the barrier that closes a problem.
  enum PhaseKind : int { kPhaseNone = 0, kPhaseGateResAdarms = 1, kPhaseGateRes = 2, kPhasePrivateAdarms = 3 };

  struct Problem {
    ProblemShape shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    int phase = kPhaseNone;
    SeqPhaseArgs phase_args{};
    int b_batch_mode = 0;      // 1: B / SFB come from this cluster's slot (batch index = cluster id)
  };

  struct Params {
    int num_problems = 0;
    int num_ctas = 0;          // CTAs in the launched grid (all resident)
    int* counter = nullptr;    // zero-initialised sequence counter (reset by the kernel on exit)
    int flags = 0;             // timing probes only: bit0 skip grid barriers, bit1 skip phase work, bit2 epilogue warps
                               // do not wait, bit3 one sync point per boundary. Any probe that skips a wait leaves the
                               // counter dirty; zero it before the next synchronised run.
    int early_first = 8;       // weight k-tiles issued ahead of the wait for a problem's first tile (across the barrier)
    int early_rest = 0;        // ... for its remaining tiles (0 = interleaved with the activations)
    Problem prob[kMaxProblems];
  };

  enum class WarpCategory : int32_t { MMA = 0, Sched = 1, MainloopLoad = 2, EpilogueLoad = 3, Epilogue = 4 };

  // Named barrier among the epilogue threads (phase-internal syncs) and the load-warp hand-off.
  static constexpr int kEpilogueSyncBarrierId = 9;
  static constexpr int kPhaseBarrierId = 8;

  // Sync points before problem p may load its activations: one per closed problem, two when it ran a phase.
  static CUTLASS_DEVICE int sync_points_before(Params const& params, int p) {
    int n = 0;
    for (int q = 0; q < p; ++q) {
      const int ph = params.prob[q].phase;
      n += ((ph == kPhaseGateResAdarms || ph == kPhaseGateRes) && !(params.flags & 8)) ? 2 : 1;
    }
    return n;
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;
    static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");

    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx)
                                                                                     : WarpCategory::Epilogue;
    uint32_t lane_predicate = cute::elect_one_sync();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});
    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;
    constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;
    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int cta_linear_id = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
    const int cluster_linear_id = cta_linear_id / size(ClusterShape{});
    const int num_problems = params.num_problems;
    const int num_ctas = params.num_ctas;
    int* counter = params.counter;

    // Descriptor prefetch for every problem up front (one thread).
    if ((warp_category == WarpCategory::Sched) && lane_predicate) {
      for (int p = 0; p < num_problems; ++p) {
        CollectiveMainloop cm(params.prob[p].mainloop, cluster_shape, cta_rank_in_cluster);
        cm.prefetch_tma_descriptors();
      }
    }
    if ((warp_category == WarpCategory::EpilogueLoad) && lane_predicate) {
      for (int p = 0; p < num_problems; ++p) {
        CollectiveEpilogue ce(params.prob[p].epilogue, shared_storage.tensors.epilogue);
        ce.prefetch_tma_descriptors(params.prob[p].epilogue);
      }
    }

    // Pipelines (built once; their states carry across problems).
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    if (WarpCategory::MMA == warp_category) mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    mainloop_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && (warp_category == WarpCategory::MainloopLoad);
    mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    mainloop_pipeline_params.initializing_warp = 0;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, cluster_shape,
                                       cute::true_type{}, cute::false_type{});

    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    if (WarpCategory::Epilogue == warp_category) epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    epi_load_pipeline_params.dst_blockid = cta_rank_in_cluster;
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    epi_load_pipeline_params.initializing_warp = 1;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadOrderBarrier::Params load_order_barrier_params;
    load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopLoad) ? 0 : 1;
    load_order_barrier_params.group_size = NumMainloopLoadThreads;
    load_order_barrier_params.initializing_warp = 3;
    LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_barrier_params);

    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (WarpCategory::MMA == warp_category) accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
    if (WarpCategory::Epilogue == warp_category) accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
    accumulator_pipeline_params.producer_arv_count = 1;
    accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    accumulator_pipeline_params.initializing_warp = 5;
    AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator, accumulator_pipeline_params, cluster_shape,
                                             cute::true_type{}, cute::false_type{});

    TmemAllocator tmem_allocator{};
    arch::NamedBarrier tmem_allocation_result_barrier(NumMMAThreads + NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    if (WarpCategory::MMA == warp_category) {
      if constexpr (!IsOverlappingAccum) {
        if (has_mma_peer_cta && lane_predicate) tmem_deallocation_result_barrier.init(NumMMAThreads);
      }
      else {
        if (has_mma_peer_cta && lane_predicate) tmem_deallocation_result_barrier.init(NumEpilogueThreads * 2);
        else if (lane_predicate) tmem_deallocation_result_barrier.init(NumEpilogueThreads);
      }
    }
    pipeline_init_arrive_relaxed(cluster_size);

    MainloopPipelineState mainloop_pipe_consumer_state;
    MainloopPipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();
    AccumulatorPipelineState accumulator_pipe_consumer_state;
    AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    mainloop_pipeline.init_masks(cluster_shape, block_id_in_cluster);
    accumulator_pipeline.init_masks(cluster_shape, block_id_in_cluster);

    // TMEM tensors are laid out once (same tile for every problem).
    auto tmem_storage = [&]() {
      CollectiveMainloop cm(params.prob[0].mainloop, cluster_shape, cta_rank_in_cluster);
      return cm.template init_tmem_tensors<EpilogueTile, IsOverlappingAccum>(EpilogueTile{});
    }();
    pipeline_init_wait(cluster_size);

    if (warp_category == WarpCategory::MainloopLoad) {
      // ---------------- producer warp ----------------
      for (int p = 0; p < num_problems; ++p) {
        auto const& pr = params.prob[p];
        auto problem_shape_MNKL = pr.shape;
        CollectiveMainloop collective_mainloop(pr.mainloop, cluster_shape, cta_rank_in_cluster);
        TileScheduler scheduler(nullptr, pr.scheduler, block_id_in_cluster);
        auto work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, shared_storage.tensors.mainloop);
        bool first_tile = true;
        const bool last_problem = (p + 1 == num_problems);
        const int b_batch = pr.b_batch_mode ? cluster_linear_id : -1;
        while (work_tile_info.is_valid()) {
          auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, load_inputs.k_tiles);
          auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
          auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);
          auto [next_work_tile_info, unused_inc] = scheduler.fetch_next_work(work_tile_info);
          const bool last_tile = !next_work_tile_info.is_valid();
          // Wait hook for the first tile of a problem: PDL for problem 0, the sequence barrier otherwise.
          auto wait_fn = [&]() {
            if (!first_tile) return;
            if (p == 0) {
              cutlass::arch::wait_on_dependent_grids();
            } else {
              if (!(params.flags & 1)) {
                if (lane_predicate) seq_detail::seq_wait(counter, num_ctas * sync_points_before(params, p));
                __syncwarp();
              }
              if (params.prob[p - 1].phase == kPhasePrivateAdarms) {
                arch::NamedBarrier::sync(NumEpilogueThreads + NumMainloopLoadThreads, kPhaseBarrierId);
              }
              seq_detail::fence_proxy_async();
            }
          };
          auto [mainloop_producer_state_next, k_tile_iter_next] = collective_mainloop.load(
            mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl,
            k_tile_iter, k_tile_prologue, wait_fn, /*trigger=*/false, first_tile ? params.early_first : params.early_rest, b_batch);
          mainloop_pipe_producer_state = mainloop_producer_state_next;
          first_tile = false;
          auto [mainloop_producer_state_next_, unused_] = collective_mainloop.load(
            mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl,
            k_tile_iter_next, k_tile_count - k_tile_prologue, wait_fn, /*trigger=*/last_problem && last_tile, params.early_rest, b_batch);
          mainloop_pipe_producer_state = mainloop_producer_state_next_;
          __syncwarp();
          work_tile_info = next_work_tile_info;
          cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        }
        if (first_tile) {
          // No tile for this CTA in this problem: still observe the barrier / phase hand-off.
          wait_only(counter, num_ctas, p, params);
        }
      }
      // A CTA whose last problem had no tiles never issued the trigger; issue it now.
      cutlass::arch::launch_dependent_grids();
      collective_mainloop_load_tail(mainloop_pipeline, mainloop_pipe_producer_state, params, cluster_shape, cta_rank_in_cluster);
      // Final sync point (load warp's half): nobody in this CTA spins on the counter any more.
      __syncwarp();
      if (lane_predicate) final_arrive(params, counter, num_ctas);
    }
    else if (warp_category == WarpCategory::MMA) {
      // ---------------- MMA warp ----------------
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      for (int p = 0; p < num_problems; ++p) {
        auto const& pr = params.prob[p];
        auto problem_shape_MNKL = pr.shape;
        CollectiveMainloop collective_mainloop(pr.mainloop, cluster_shape, cta_rank_in_cluster);
        collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);
        auto mma_inputs = collective_mainloop.mma_init(tmem_storage, shared_storage.tensors.mainloop);
        TileScheduler scheduler(nullptr, pr.scheduler, block_id_in_cluster);
        auto work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        while (work_tile_info.is_valid()) {
          auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
          auto [next_work_tile_info, unused_inc] = scheduler.fetch_next_work(work_tile_info);
          int acc_stage = [&]() {
            if constexpr (IsOverlappingAccum) return accumulator_pipe_producer_state.phase() ^ 1;
            else return accumulator_pipe_producer_state.index();
          }();
          if (is_mma_leader_cta) {
            mainloop_pipe_consumer_state = collective_mainloop.mma(
              cute::make_tuple(mainloop_pipeline, accumulator_pipeline),
              cute::make_tuple(mainloop_pipe_consumer_state, accumulator_pipe_producer_state),
              collective_mainloop.slice_accumulator(tmem_storage, acc_stage),
              mma_inputs, cta_coord_mnkl, k_tile_count);
            accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
          }
          ++accumulator_pipe_producer_state;
          work_tile_info = next_work_tile_info;
          cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        }
      }
      tmem_allocator.release_allocation_lock();
      if constexpr (!IsOverlappingAccum) {
        if (is_mma_leader_cta) accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
        if constexpr (has_mma_peer_cta) {
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
          tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
        }
      }
      else {
        tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
      }
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
    else if (warp_category == WarpCategory::EpilogueLoad) {
      // ---------------- epilogue source loads (unused by these epilogues) ----------------
      bool do_load_order_wait = true;
      bool do_tail_load = false;
      for (int p = 0; p < num_problems; ++p) {
        auto const& pr = params.prob[p];
        CollectiveEpilogue collective_epilogue(pr.epilogue, shared_storage.tensors.epilogue);
        if (!collective_epilogue.is_producer_load_needed()) continue;
        auto problem_shape_MNKL = pr.shape;
        TileScheduler scheduler(nullptr, pr.scheduler, block_id_in_cluster);
        auto work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        int current_wave = 0;
        while (work_tile_info.is_valid()) {
          auto [next_work_tile_info, unused_inc] = scheduler.fetch_next_work(work_tile_info);
          if (do_load_order_wait) { load_order_barrier.wait(); do_load_order_wait = false; }
          bool reverse_epi_n = IsOverlappingAccum && (current_wave % 2 == 0);
          epi_load_pipe_producer_state = collective_epilogue.template load<IsOverlappingAccum>(
            epi_load_pipeline, epi_load_pipe_producer_state, problem_shape_MNKL, CtaShape_MNK{}, cta_coord_mnkl,
            TileShape{}, TiledMma{}, shared_storage.tensors.epilogue, reverse_epi_n);
          do_tail_load = true;
          current_wave++;
          work_tile_info = next_work_tile_info;
          cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        }
        if (do_tail_load) {
          collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state, epi_store_pipeline, epi_store_pipe_producer_state);
          do_tail_load = false;
        }
      }
    }
    else if (warp_category == WarpCategory::Epilogue) {
      // ---------------- epilogue warps ----------------
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      const int epi_thread = threadIdx.x - (MaxThreadsPerBlock - NumEpilogueThreads);
      for (int p = 0; p < num_problems; ++p) {
        auto const& pr = params.prob[p];
        auto problem_shape_MNKL = pr.shape;
        CollectiveMainloop collective_mainloop(pr.mainloop, cluster_shape, cta_rank_in_cluster);
        collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);
        CollectiveEpilogue collective_epilogue(pr.epilogue, shared_storage.tensors.epilogue);
        TileScheduler scheduler(nullptr, pr.scheduler, block_id_in_cluster);
        auto work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        bool do_tail_store = false;
        while (work_tile_info.is_valid()) {
          auto [next_work_tile_info, unused_inc] = scheduler.fetch_next_work(work_tile_info);
          int acc_stage = [&]() {
            if constexpr (IsOverlappingAccum) return accumulator_pipe_consumer_state.phase();
            else return accumulator_pipe_consumer_state.index();
          }();
          auto accumulator = get<0>(collective_mainloop.slice_accumulator(tmem_storage, acc_stage));
          auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue.template store<IsOverlappingAccum>(
            epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline, epi_store_pipe_producer_state,
            accumulator_pipeline, accumulator_pipe_consumer_state, problem_shape_MNKL, CtaShape_MNK{}, cta_coord_mnkl,
            TileShape{}, TiledMma{}, accumulator, shared_storage.tensors.epilogue);
          epi_load_pipe_consumer_state = load_state_next;
          epi_store_pipe_producer_state = store_state_next;
          accumulator_pipe_consumer_state = acc_state_next;
          do_tail_store = true;
          work_tile_info = next_work_tile_info;
          cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        }
        if (do_tail_store) {
          collective_epilogue.store_tail(epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline,
                                         epi_store_pipe_producer_state, CtaShape_MNK{});
        }
        // Close problem p: this CTA's stores are complete and visible.
        arch::NamedBarrier::sync(NumEpilogueThreads, kEpilogueSyncBarrierId);
        if (p + 1 < num_problems) {
          if (epi_thread == 0) {
            seq_detail::fence_proxy_async();
            __threadfence();
            atomicAdd(counter, 1);
          }
          if (pr.phase == kPhasePrivateAdarms) {
            // One sync point: once every CTA's stores (x_new, partials) are visible, this CTA
            // quantizes every row into its cluster's slot and hands the load warp over locally.
            if (!(params.flags & 5)) {
              if (epi_thread == 0) seq_detail::seq_wait(counter, num_ctas * (sync_points_before(params, p) + 1));
              arch::NamedBarrier::sync(NumEpilogueThreads, kEpilogueSyncBarrierId);
            }
            __threadfence();
            if (!(params.flags & 2)) run_phase(params, p, epi_thread, shared_storage.phase, cluster_linear_id, num_ctas);
            seq_detail::fence_proxy_async();
            __threadfence();
            arch::NamedBarrier::sync(NumEpilogueThreads + NumMainloopLoadThreads, kPhaseBarrierId);
          }
          else if (pr.phase != kPhaseNone) {
            // Every CTA waits until all stores of problem p are visible, runs its share of the
            // phase (rows strided over the grid) and arrives a second time; the load warps wait
            // for that second point before touching the phase outputs.
            if (!(params.flags & 5)) {
              if (epi_thread == 0) seq_detail::seq_wait(counter, num_ctas * (sync_points_before(params, p) + 1));
              arch::NamedBarrier::sync(NumEpilogueThreads, kEpilogueSyncBarrierId);
            }
            __threadfence();
            // Rows go to the highest CTAs first: those are idle in the small (O / down) problems.
            if (!(params.flags & 2)) run_phase(params, p, epi_thread, shared_storage.phase, num_ctas - 1 - cta_linear_id, num_ctas);
            seq_detail::fence_proxy_async();
            __threadfence();
            arch::NamedBarrier::sync(NumEpilogueThreads, kEpilogueSyncBarrierId);
            if (epi_thread == 0 && !(params.flags & 8)) atomicAdd(counter, 1);
          }
        }
        else {
          if (epi_thread == 0) {
            seq_detail::fence_proxy_async();
            __threadfence();
            final_arrive(params, counter, num_ctas);
          }
        }
      }
      if constexpr (IsOverlappingAccum) {
        if constexpr (has_mma_peer_cta) tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank);
        tmem_deallocation_result_barrier.arrive();
      }
    }
  }

private:
  CUTLASS_DEVICE void run_phase(Params const& params, int p, int epi_thread, SeqPhaseSmem& phase_smem, int cta, int num_ctas) {
    auto sync = []() { arch::NamedBarrier::sync(NumEpilogueThreads, kEpilogueSyncBarrierId); };
    if (params.prob[p].phase == kPhaseGateResAdarms) {
      seq_phase_gate_res_adarms(params.prob[p].phase_args, phase_smem, epi_thread, sync, cta, num_ctas);
    } else if (params.prob[p].phase == kPhaseGateRes) {
      seq_phase_gate_res(params.prob[p].phase_args, epi_thread, cta, num_ctas);
    } else if (params.prob[p].phase == kPhasePrivateAdarms) {
      seq_phase_private_adarms(params.prob[p].phase_args, phase_smem, epi_thread, sync, /*slot=*/cta);
    }
  }

  // Final sync point: two arrivals per CTA (load warp + epilogue); the last one resets the counter.
  static CUTLASS_DEVICE void final_arrive(Params const& params, int* counter, int num_ctas) {
    const int total = num_ctas * (sync_points_before(params, params.num_problems - 1) + 2);
    const int old = atomicAdd(counter, 1);
    if (old == total - 1) { *counter = 0; __threadfence(); }
  }

  // Load warp: the problem had no tile for this CTA; keep the barrier protocol consistent.
  CUTLASS_DEVICE void wait_only(int* counter, int num_ctas, int p, Params const& params) {
    if (p == 0) { cutlass::arch::wait_on_dependent_grids(); return; }
    if (!(params.flags & 1)) {
      if (cute::elect_one_sync()) seq_detail::seq_wait(counter, num_ctas * sync_points_before(params, p));
      __syncwarp();
    }
    if (params.prob[p - 1].phase == kPhasePrivateAdarms) {
      arch::NamedBarrier::sync(NumEpilogueThreads + NumMainloopLoadThreads, kPhaseBarrierId);
    }
  }
  template <class Pipe, class State, class CS>
  CUTLASS_DEVICE void collective_mainloop_load_tail(Pipe& pipe, State state, Params const& params, CS cluster_shape, uint32_t rank) {
    CollectiveMainloop cm(params.prob[params.num_problems - 1].mainloop, cluster_shape, rank);
    cm.load_tail(pipe, state);
  }
};

}  // namespace cutlass::gemm::kernel
