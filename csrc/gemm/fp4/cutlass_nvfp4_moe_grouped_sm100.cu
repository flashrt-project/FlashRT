// SPDX-License-Identifier: Apache-2.0
//
// Grouped NVFP4 block-scaled GEMM for sm_100-class Blackwell. See header.

#include "gemm/fp4/cutlass_nvfp4_moe_grouped_sm100.cuh"
#include <cstdlib>
#include <cstdio>

#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

namespace flash_rt {
namespace gemm {

namespace {

using namespace cute;

using ElementA           = cutlass::float_e2m1_t;
using ElementB           = cutlass::float_e2m1_t;
using ElementC           = cutlass::bfloat16_t;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementSF          = cutlass::float_ue4m3_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;

using ProblemShape  = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape  = Shape<int32_t, int32_t, _1>;
// N of 128 rather than 256, measured. A prefill routes about sixty-four rows
// to the average expert across 256 groups of unequal size, and the narrower
// tile gives the scheduler twice as many blocks to balance them across twenty
// SMs. Paired against N=256 on the same machine state: 377.9/378.2/377.9 ms
// against 429.6/386.3/387.6 -- the worst run of this tile beats the best run
// of the other, and the spread goes from 43 ms to 0.3.
using MmaTileShape  = Shape<_128, _128, _256>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape,
        Shape<_128, _64>,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutC*, AlignmentD,
        cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementPairA, LayoutA*, AlignmentA,
        ElementPairB, LayoutB*, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using ProblemShapeMNK = typename ProblemShape::UnderlyingProblemShape;

// Everything the launch needs, laid out back to back in one scratch buffer so
// the caller allocates once. Filled by a device kernel from the routing, which
// is what keeps the whole thing free of a host round trip.
struct GroupArgs {
  ProblemShapeMNK*  shapes;
  const ElementA**  ptr_A;
  const ElementB**  ptr_B;
  const ElementSF** ptr_SFA;
  const ElementSF** ptr_SFB;
  ElementD**        ptr_D;
  const float**     ptr_alpha;
  StrideA*          stride_A;
  StrideB*          stride_B;
  StrideC*          stride_C;
  StrideD*          stride_D;
  LayoutSFA*        layout_SFA;
  LayoutSFB*        layout_SFB;
};

constexpr size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

size_t args_bytes(int g) {
  size_t n = 0;
  n = align_up(n + sizeof(ProblemShapeMNK) * g, 256);
  n = align_up(n + sizeof(void*) * g * 6, 256);          // A B SFA SFB D alpha
  n = align_up(n + sizeof(StrideA) * g, 256);
  n = align_up(n + sizeof(StrideB) * g, 256);
  n = align_up(n + sizeof(StrideC) * g, 256);
  n = align_up(n + sizeof(StrideD) * g, 256);
  n = align_up(n + sizeof(LayoutSFA) * g, 256);
  n = align_up(n + sizeof(LayoutSFB) * g, 256);
  return n;
}

GroupArgs carve(void* base, int g) {
  auto* p = static_cast<uint8_t*>(base);
  size_t o = 0;
  auto take = [&](size_t bytes) {
    void* r = p + o;
    o = align_up(o + bytes, 256);
    return r;
  };
  GroupArgs a{};
  a.shapes    = static_cast<ProblemShapeMNK*>(take(sizeof(ProblemShapeMNK) * g));
  auto* ptrs  = static_cast<void**>(take(sizeof(void*) * g * 6));
  auto at = [&](int i) { return static_cast<void*>(ptrs + i * g); };
  a.ptr_A     = static_cast<const ElementA**>(at(0));
  a.ptr_B     = static_cast<const ElementB**>(at(1));
  a.ptr_SFA   = static_cast<const ElementSF**>(at(2));
  a.ptr_SFB   = static_cast<const ElementSF**>(at(3));
  a.ptr_D     = static_cast<ElementD**>(at(4));
  a.ptr_alpha = static_cast<const float**>(at(5));
  a.stride_A  = static_cast<StrideA*>(take(sizeof(StrideA) * g));
  a.stride_B  = static_cast<StrideB*>(take(sizeof(StrideB) * g));
  a.stride_C  = static_cast<StrideC*>(take(sizeof(StrideC) * g));
  a.stride_D  = static_cast<StrideD*>(take(sizeof(StrideD) * g));
  a.layout_SFA = static_cast<LayoutSFA*>(take(sizeof(LayoutSFA) * g));
  a.layout_SFB = static_cast<LayoutSFB*>(take(sizeof(LayoutSFB) * g));
  return a;
}

// One thread per group. Reads the routing (prefix sums of the per-expert token
// counts) and writes the descriptor arrays CUTLASS reads. No host involvement,
// which is the point: the launch shape below depends only on the group count.
__global__ void fill_group_args(
    GroupArgs a,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ SFA,
    const uint8_t* __restrict__ W_stack,
    const uint8_t* __restrict__ SFB_stack,
    const float* __restrict__ alpha,
    uint8_t* __restrict__ D,
    const int* __restrict__ group_off,
    const int* __restrict__ sfa_off,
    int groups, int N, int K, long w_stride, long sfb_stride) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= groups) return;

  const int off = group_off[e];
  const int m = group_off[e + 1] - off;

  a.shapes[e] = cute::make_shape(m, N, K);
  a.ptr_A[e] = reinterpret_cast<const ElementA*>(
      A_packed + static_cast<size_t>(off) * (K / 2));
  a.ptr_B[e] = reinterpret_cast<const ElementB*>(W_stack + e * w_stride);
  a.ptr_SFA[e] = reinterpret_cast<const ElementSF*>(SFA + sfa_off[e]);
  a.ptr_SFB[e] = reinterpret_cast<const ElementSF*>(SFB_stack + e * sfb_stride);
  a.ptr_D[e] = reinterpret_cast<ElementD*>(
      D + static_cast<size_t>(off) * N * sizeof(ElementD));
  a.ptr_alpha[e] = alpha + e;

  a.stride_A[e] = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, K, 1));
  a.stride_B[e] = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(N, K, 1));
  a.stride_C[e] = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(m, N, 1));
  a.stride_D[e] = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, N, 1));
  a.layout_SFA[e] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, N, K, 1));
  a.layout_SFB[e] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, N, K, 1));
}

}  // namespace

size_t moe_grouped_gemm_nvfp4_sm100_scratch_bytes(int groups) {
  if (groups <= 0) return 0;
  // The CUTLASS workspace for a grouped launch scales with the group count and
  // the scheduler, not with the token counts, so a bound taken at construction
  // stays valid however the routing falls.
  return args_bytes(groups) + static_cast<size_t>(groups) * 1024 + (1u << 20);
}

int moe_grouped_gemm_nvfp4_sm100_bf16out(
    const void*  A_packed,
    const void*  SFA,
    const void*  W_stack,
    const void*  SFB_stack,
    const void*  alpha_dev,
    void*        D,
    const void*  group_off,
    const void*  sfa_off,
    int          groups,
    int          N,
    int          K,
    long         w_stride,
    long         sfb_stride,
    void*        scratch,
    size_t       scratch_bytes,
    cudaStream_t stream) {
  if (!A_packed || !SFA || !W_stack || !SFB_stack || !alpha_dev || !D
      || !group_off || !sfa_off || !scratch) return 1;
  if (groups <= 0 || N <= 0 || K <= 0 || (K & 15) != 0) return 2;
  const size_t need = args_bytes(groups);
  if (scratch_bytes < need) return 3;

  GroupArgs ga = carve(scratch, groups);
  const int threads = 128;
  fill_group_args<<<(groups + threads - 1) / threads, threads, 0, stream>>>(
      ga,
      static_cast<const uint8_t*>(A_packed),
      static_cast<const uint8_t*>(SFA),
      static_cast<const uint8_t*>(W_stack),
      static_cast<const uint8_t*>(SFB_stack),
      static_cast<const float*>(alpha_dev),
      static_cast<uint8_t*>(D),
      static_cast<const int*>(group_off),
      static_cast<const int*>(sfa_off),
      groups, N, K, w_stride, sfb_stride);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  // The cluster is a runtime shape, so it was swept without recompiling:
  // (1,1) (2,1) (1,2) (2,2) (4,1) (1,4) at the prefill shape, then the best
  // two alternated three times each. The within-pair differences (+1.4%,
  // -0.4%, +0.8%) came out smaller than the drift between runs (427 to 385 ms
  // for the same setting), so the cluster shape does not move this. Left as a
  // knob with the result written down rather than as a knob to try again.
  static const dim3 kCluster = [] {
    const char* v = std::getenv("FLASHRT_MOE_GROUPED_CLUSTER");
    int x = 1, y = 1;
    if (v && std::sscanf(v, "%d,%d", &x, &y) == 2 && x >= 1 && y >= 1) {
      return dim3(x, y, 1);
    }
    return dim3(1, 1, 1);
  }();
  hw_info.cluster_shape = kCluster;
  hw_info.cluster_shape_fallback = dim3(1, 1, 1);

  typename Gemm::Arguments args_proto{};
  // The fusion argument type is reachable only through an Arguments instance,
  // which is how the CUTLASS example spells it too.
  decltype(args_proto.epilogue.thread) fusion_args;
  fusion_args.alpha = 0.0f;
  fusion_args.alpha_ptr_array = ga.ptr_alpha;
  fusion_args.dAlpha = {_0{}, _0{}, 1};
  fusion_args.beta = 0.0f;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dBeta = {_0{}, _0{}, 0};

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler{};

  // Host-side problem shapes are passed as nullptr deliberately: the shapes
  // live only on device, so nothing here depends on the routing and the call
  // is safe to capture.
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {groups, ga.shapes, nullptr},
      {ga.ptr_A, ga.stride_A, ga.ptr_B, ga.stride_B,
       ga.ptr_SFA, ga.layout_SFA, ga.ptr_SFB, ga.layout_SFB},
      {fusion_args, nullptr, ga.stride_C, ga.ptr_D, ga.stride_D},
      hw_info, scheduler};

  Gemm gemm;
  const size_t ws = Gemm::get_workspace_size(args);
  if (need + ws > scratch_bytes) return 4;
  void* ws_ptr = static_cast<uint8_t*>(scratch) + need;

  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[moe_grouped_gemm_nvfp4_sm100] can_implement FAIL groups=%d N=%d "
        "K=%d status=%d\n", groups, N, K, static_cast<int>(status));
    return 5;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) return 6;
  status = gemm.run(stream);
  return status == cutlass::Status::kSuccess ? 0 : 7;
}

}  // namespace gemm
}  // namespace flash_rt
