// SPDX-License-Identifier: Apache-2.0
//
// See header for the full GeGLU-fold design.  This translation unit currently
// implements MILESTONE 0 (the build/instantiation scaffold): a single NVFP4
// projection with a SiLu activation and FP4-packed blockscaled output, i.e.
//   D = pack_FP4( SiLu(alpha_gate * (A_fp4 @ Bgate_fp4^T)) ).
// Bup/SFBup/alpha_up are accepted (stable caller interface) but unused until
// the dual-accumulator mainloop lands.  The epilogue/quant path is identical
// to cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.cu with GELU swapped for SiLu
// and the per-col bias removed.

#include "cutlass_nvfp4_dual_gemm_geglu_fp4out_sm120.cuh"

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "sm120_gelu_mul_blockscale_visitor.hpp"

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace gemm {

namespace {
using namespace cute;

using ElementA           = cutlass::float_e2m1_t;
using ElementB           = cutlass::float_e2m1_t;
using ElementC           = cutlass::bfloat16_t;
using ElementD           = cutlass::float_e2m1_t;
using ElementSFD         = cutlass::float_ue4m3_t;
using ElementBias        = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using ElementSF          = cutlass::float_ue4m3_t;

using LayoutA      = cutlass::layout::RowMajor;
using LayoutB      = cutlass::layout::ColumnMajor;
using LayoutC      = cutlass::layout::ColumnMajor;
using LayoutD      = cutlass::layout::RowMajor;
using LayoutSFDTag = cutlass::layout::RowMajor;

using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape    = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// M2b: SF vector size 32 over the INTERLEAVED gate|up GEMM output (N=2*inter)
// == per-16-OUTPUT-col after the silu_mul 2->1 collapse. The swizzled SFD byte
// layout (rows M, n_blocks = 2*inter/32 = inter/16) is then BYTE-IDENTICAL to
// the down GEMM's per-16 SFA for [M, inter], so the SFD passes straight through;
// only the FP4 data needs the trivial even-column nibble compaction.
constexpr int OutputSFVectorSize = 32;

// GeGLU fold (M2a): gelu(alpha_gate*gate) * (alpha_up*up) on adjacent
// interleaved accumulator columns, then per-block-16 NVFP4 quant + FP4 out.
// The silu_mul is baked into the forked store node (visit()); full-width
// output (each pair duplicated) for M2a — M2b compacts the store.
using FusionOperation = cutlass::epilogue::fusion::GeluMulBlockScaleFactor<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD,
    LayoutSFDTag>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementPairA, LayoutA, AlignmentA,
        ElementPairB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;

struct ShapeKey {
  int M, N, K;
  bool operator==(const ShapeKey& o) const {
    return M == o.M && N == o.N && K == o.K;
  }
};
struct SHash {
  size_t operator()(const ShapeKey& k) const noexcept {
    return (size_t(k.M) * 1315423911u) ^ (size_t(k.N) * 2654435761u)
         ^ size_t(k.K);
  }
};
struct CachedWs { void* ptr = nullptr; size_t size = 0; };
std::unordered_map<ShapeKey, CachedWs, SHash> g_ws;
std::mutex g_mu;

void* get_ws(int M, int N, int K, size_t need) {
  std::lock_guard<std::mutex> lk(g_mu);
  ShapeKey k{M, N, K};
  auto it = g_ws.find(k);
  if (it != g_ws.end() && it->second.size >= need) return it->second.ptr;
  if (it != g_ws.end()) { cudaFree(it->second.ptr); g_ws.erase(it); }
  CachedWs w; w.size = need;
  if (need > 0) cudaMalloc(&w.ptr, need);
  g_ws[k] = w;
  return w.ptr;
}

float* get_norm_const_one() {
  static float* p = nullptr;
  if (p == nullptr) {
    cudaMalloc(&p, sizeof(float));
    float one = 1.0f;
    cudaMemcpy(p, &one, sizeof(float), cudaMemcpyHostToDevice);
  }
  return p;
}

// sm120 only specializes the bias+act blockscale fusion op, so a (zero) per-col
// bias pointer is required even though GeGLU has no bias. Cache one zeroed
// bf16 buffer sized to the largest N seen.
ElementBias* get_zero_bias(int N) {
  static ElementBias* p = nullptr;
  static int cap = 0;
  std::lock_guard<std::mutex> lk(g_mu);
  if (N > cap) {
    if (p) cudaFree(p);
    cudaMalloc(&p, (size_t)N * sizeof(ElementBias));
    cudaMemset(p, 0, (size_t)N * sizeof(ElementBias));
    cap = N;
  }
  return p;
}

}  // namespace

void fp4_w4a16_dual_gemm_geglu_fp4out_sm120(
    const void* A_packed, const void* Bgate_packed, const void* /*Bup_packed*/,
    const void* SFA,      const void* SFBgate,      const void* /*SFBup*/,
    void*       D_packed,
    void*       SFD,
    int M, int N, int K,
    float alpha_gate,
    float alpha_up,
    cudaStream_t stream)
{
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  StrideA strA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB strB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC strC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD strD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto problem_MNKL = cute::make_shape(M, N, K, 1);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_MNKL);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_MNKL);

  using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

  float* norm_const_dev = get_norm_const_one();

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ArrayElementA const*>(A_packed), strA,
          reinterpret_cast<ArrayElementB const*>(Bgate_packed), strB,
          reinterpret_cast<ElementSF const*>(SFA), layout_SFA,
          reinterpret_cast<ElementSF const*>(SFBgate), layout_SFB
      },
      {
          {1.0f, 0.0f},   // GEMM linear (alpha,beta); per-proj scales below
          nullptr, strC,
          reinterpret_cast<ElementD*>(D_packed), strD
      }
  };
  // Bgate_packed / SFBgate hold the INTERLEAVED gate|up weight (gate@2i,
  // up@2i+1); N = 2*intermediate. The forked epilogue applies the per-
  // projection scales and silu(gate)*up.
  args.epilogue.thread.block_scale_factor_ptr =
      reinterpret_cast<ElementSFD*>(SFD);
  args.epilogue.thread.norm_constant_ptr = norm_const_dev;
  args.epilogue.thread.alpha_gate = alpha_gate;
  args.epilogue.thread.alpha_up = alpha_up;

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_ws(M, N, K, ws_size);
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_dual_gemm_geglu_fp4out_sm120] can_implement FAIL "
        "M=%d N=%d K=%d status=%d\n", M, N, K, int(status));
    return;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_dual_gemm_geglu_fp4out_sm120] initialize FAIL "
        "M=%d N=%d K=%d status=%d\n", M, N, K, int(status));
    return;
  }
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_dual_gemm_geglu_fp4out_sm120] run FAIL status=%d\n",
        int(status));
  }
}

}  // namespace gemm
}  // namespace flash_rt
