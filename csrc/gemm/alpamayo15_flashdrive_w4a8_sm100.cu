// SPDX-License-Identifier: Apache-2.0
//
// FlashDrive (Alpamayo 1.5) W4A8 GEMM on tcgen05 INT8 MMA for Thor (sm_110a).
//
// Computes what vLLM's Marlin W4A8 kernel computes for FlashDrive's ParoQuant
// linears (UINT4 weights with zero points, group 128, per-token INT8
// activations, integer group scales from marlin_act_int8_process_scales):
//
//   out[m, n] = half( float( sum_g s[g, n] * sum_{k in g} x[m, k] * (w[k, n] - z[g, n]) ) * a_scale[m] )
//
// The group sums and their scaled total are exact int32 arithmetic (CUTLASS's
// sm100 groupwise-scaling mainloop with an int32 accumulator promotes each
// 128-deep partial with the integer scale); the total is converted to float
// once and multiplied by the per-token scale in the epilogue. Marlin computes
// the same integers but sums K slices in FP32, so the two agree except for
// rare one-ulp FP16 differences where Marlin's slice sum rounds differently.
//
// The problem is computed transposed (out^T = W x^T): MMA rows are output
// channels, so each TMEM lane carries one group scale per K block, and MMA
// columns are tokens. Operands:
//   xq       [M, K] int8, row-major (per-token quantized activations)
//   w8       [N, K] int8, row-major, w - z in [-15, 15]
//   sf_group [K/128, N] int32, s[g, n] at g * N + n
//   sf_ones  int32 ones (the token-side scale operand of the mainloop), see sizes()
//   a_scale  [M] fp32
//   out      [M, N] fp16, row-major

#include <cuda_runtime.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

using ElementW = int8_t;                               // MMA operand A: weights [N, K], K-major
using ElementX = int8_t;                               // MMA operand B: activations [M, K], K-major
using ElementD = cutlass::half_t;
using ElementAcc = int32_t;
using ElementCompute = float;
constexpr int kAlign8 = 16, kAlignD = 8;
constexpr int kGroup = 128;

using MmaTile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
// Channel-side scales per (channel, K group); token-side scales are ones over 32-token blocks.
using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<1, 32, kGroup>;
using LayoutSFW = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFX = decltype(ScaleConfig::deduce_layoutSFB());

// D^T[n, m] = half(a_scale[m] * float(acc[n, m])): a per-column broadcast of the transposed problem.
using TokenScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, MmaTile, float, ElementCompute,
                                                               Stride<_0, _1, int64_t>, 4>;
using Epilogue = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, ElementCompute,
                                           cutlass::FloatRoundStyle::round_to_nearest>,
    TokenScale, cutlass::epilogue::fusion::Sm90AccFetch>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute, void,
    cutlass::layout::ColumnMajor, kAlignD, ElementD, cutlass::layout::ColumnMajor, kAlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Epilogue>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementW, cute::tuple<cutlass::layout::RowMajor, LayoutSFW>,
    kAlign8, ElementX, cute::tuple<cutlass::layout::ColumnMajor, LayoutSFX>, kAlign8, ElementAcc, MmaTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

Gemm::Arguments make_args(const void* xq, const void* w8, const int* sf_group, const int* sf_ones,
                          const float* a_scale, void* out, int M, int N, int K) {
  auto sW = cutlass::make_cute_packed_stride(Gemm::GemmKernel::StrideA{}, make_shape(N, K, 1));
  auto sX = cutlass::make_cute_packed_stride(Gemm::GemmKernel::StrideB{}, make_shape(M, K, 1));
  auto sC = cutlass::make_cute_packed_stride(Gemm::GemmKernel::StrideC{}, make_shape(N, M, 1));
  auto sD = cutlass::make_cute_packed_stride(Gemm::GemmKernel::StrideD{}, make_shape(N, M, 1));
  return Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K, 1},
      {static_cast<const ElementW*>(w8), sW, static_cast<const ElementX*>(xq), sX, sf_group,
       ScaleConfig::tile_atom_to_shape_SFA(make_shape(N, M, K, 1)), sf_ones,
       ScaleConfig::tile_atom_to_shape_SFB(make_shape(N, M, K, 1))},
      {{{a_scale, 0.0f, {_0{}, _1{}, int64_t(M)}}, {}, {}}, nullptr, sC, static_cast<ElementD*>(out), sD}};
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

extern "C" {

// Workspace bytes and the number of int32 ones the token-side scale operand addresses.
int flashrt_fd_w4a8_gemm_sm100_sizes(int M, int N, int K, unsigned long long* workspace_bytes,
                                     long long* ones_elems) {
  if (M <= 0 || N <= 0 || K <= 0 || K % kGroup || N % kAlignD) return 2;
  auto args = make_args(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K);
  *workspace_bytes = Gemm::get_workspace_size(args);
  *ones_elems = size(filter_zeros(ScaleConfig::tile_atom_to_shape_SFB(make_shape(N, M, K, 1))));
  return 0;
}

int flashrt_fd_w4a8_gemm_sm100(const void* xq, const void* w8, const int* sf_group, const int* sf_ones,
                               const float* a_scale, void* out, int M, int N, int K, void* workspace,
                               unsigned long long workspace_bytes, cudaStream_t stream) {
  if (!xq || !w8 || !sf_group || !sf_ones || !a_scale || !out || (workspace_bytes && !workspace)) return 1;
  if (M <= 0 || N <= 0 || K <= 0 || K % kGroup || N % kAlignD) return 2;
  if (!aligned16(xq) || !aligned16(w8) || !aligned16(sf_group) || !aligned16(sf_ones) || !aligned16(a_scale) ||
      !aligned16(out))
    return 5;                                      // TMA and vectorized scale loads need 16-byte alignment
  auto args = make_args(xq, w8, sf_group, sf_ones, a_scale, out, M, N, K);
  if (Gemm::get_workspace_size(args) > workspace_bytes) return 3;
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 5;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return 6;
  if (gemm.run(stream) != cutlass::Status::kSuccess) return 7;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

}  // extern "C"
