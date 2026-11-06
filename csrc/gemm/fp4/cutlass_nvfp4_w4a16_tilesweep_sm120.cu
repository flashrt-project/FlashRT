// SPDX-License-Identifier: Apache-2.0
//
// See header.  Clones the production fp4_w4a16_gemm_sm120_bf16out kernel config
// exactly (arch::Sm120 + OpClassBlockScaledTensorOp + Cooperative + BF16 out),
// templated on TileShape, so a tile sweep is a faithful efficiency measurement
// of that kernel family (not a different arch/schedule).

#include "cutlass_nvfp4_w4a16_tilesweep_sm120.cuh"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

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
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using ElementSF          = cutlass::float_ue4m3_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

template <class TileShape, class ClusterShape,
          class Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative>
struct TileVariant {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute,
          ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto
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
          Schedule
      >::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static int run(const void* A, const void* B, void* D, int M, int N, int K,
                 const void* SFA, const void* SFB, float alpha,
                 cudaStream_t stream, void*& ws, size_t& ws_cap) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    auto MNKL = cute::make_shape(M, N, K, 1);
    auto lSFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(MNKL);
    auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(MNKL);
    using AEA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
    using AEB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        { reinterpret_cast<AEA const*>(A), sA, reinterpret_cast<AEB const*>(B), sB,
          reinterpret_cast<ElementSF const*>(SFA), lSFA,
          reinterpret_cast<ElementSF const*>(SFB), lSFB },
        { {alpha, 0.0f}, nullptr, sC, reinterpret_cast<ElementD*>(D), sD } };
    Gemm gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return int(st) | 0x10000;
    size_t need = Gemm::get_workspace_size(args);
    if (need > ws_cap) { if (ws) cudaFree(ws); cudaMalloc(&ws, need); ws_cap = need; }
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) return int(st) | 0x20000;
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (int(st) | 0x30000);
  }
};

// Per-variant workspace cache (idx-keyed; shapes within a sweep reuse).
void* g_ws[8] = {nullptr};
size_t g_ws_cap[8] = {0};
std::mutex g_mu;

using C111 = Shape<_1, _1, _1>;
// NOTE (measured 2026-06-26): the sm120 blockscaled collective requires
// BLOCK_N >= 128 and BLOCK_M >= 128 — a 128x64 / 64x128 tile fails to compile
// ("TMA: could not find a common tile-gmem vectorization" on the UE4M3 SF
// tensor, the 128-wide SF atom can't be sub-tiled). This is why a dual-
// accumulator SwiGLU mainloop (which needs a 128x64 per-accumulator tile to
// fit two accumulators in <=255 regs) is INFEASIBLE on this path. Only
// BLOCK_{M,N} in {128,256} are valid here.
using T0 = TileVariant<Shape<_128, _128, _256>, C111>;
using T1 = TileVariant<Shape<_128, _256, _128>, C111>;  // widen (valid)
using T2 = TileVariant<Shape<_128, _128, _128>, C111>;  // half-K, deeper pipeline
using T3 = TileVariant<Shape<_128, _128, _128>, C111,
                       cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
using T4 = TileVariant<Shape<_256, _128, _128>, C111>;  // tall

const char* kNames[8] = {
    "128x128x256 (baseline)", "128x256x128 (widen)",
    "128x128x128", "128x128x128 pingpong", "256x128x128", "", "", ""};

}  // namespace

int fp4_w4a16_tilesweep_sm120_bf16out(
    int v, const void* A, const void* B, void* D, int M, int N, int K,
    const void* SFA, const void* SFB, float alpha, cudaStream_t stream) {
  std::lock_guard<std::mutex> lk(g_mu);
  switch (v) {
    case 0: return T0::run(A, B, D, M, N, K, SFA, SFB, alpha, stream, g_ws[0], g_ws_cap[0]);
    case 1: return T1::run(A, B, D, M, N, K, SFA, SFB, alpha, stream, g_ws[1], g_ws_cap[1]);
    case 2: return T2::run(A, B, D, M, N, K, SFA, SFB, alpha, stream, g_ws[2], g_ws_cap[2]);
    case 3: return T3::run(A, B, D, M, N, K, SFA, SFB, alpha, stream, g_ws[3], g_ws_cap[3]);
    case 4: return T4::run(A, B, D, M, N, K, SFA, SFB, alpha, stream, g_ws[4], g_ws_cap[4]);
    default: return -99;
  }
}

int fp4_w4a16_tilesweep_sm120_num_variants() { return 5; }
const char* fp4_w4a16_tilesweep_sm120_name(int v) {
  return (v >= 0 && v < 8) ? kNames[v] : "<invalid>";
}

}  // namespace gemm
}  // namespace flash_rt
