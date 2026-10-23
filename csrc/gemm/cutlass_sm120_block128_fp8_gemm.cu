// SPDX-License-Identifier: Apache-2.0
//
// Path B implementation: CUTLASS block-128 FP8 GEMM for SM120a.
// Header: cutlass_sm120_block128_fp8_gemm.cuh.
//
// Kernel template ported from CUTLASS 4.x example 87b
// (third_party/cutlass/examples/87_blackwell_geforce_gemm_blockwise/
//  87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu).
//
// Two GEMM instantiations are kept live and dispatched by grid size:
//   * Pingpong (TileShape 64 x 128 x 128) — M <= 64, or when its grid
//     needs at most 7/4 the waves of the 128-row grid
//   * Cooperative (TileShape 128 x 128 x 128) — otherwise
//
// Per-shape Arguments + workspace are cached in two thread-safe maps.
// The kernel itself is fused (no dequant intermediate), removing the
// 3x memory bandwidth tax of Path D.

#include "cutlass_sm120_block128_fp8_gemm.cuh"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/packed_stride.hpp"

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace gemm {

namespace {

using namespace cute;

// ── Element / layout types (match 87b) ───────────────────────────
using ElementA           = cutlass::float_e4m3_t;
using LayoutA            = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB           = cutlass::float_e4m3_t;
using LayoutB            = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC           = cutlass::bfloat16_t;
using LayoutC            = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD           = ElementC;
using LayoutD            = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute     = float;

// DeepSeek / Qwen3.6 layout: per-token activation, 128x128 weight.
//
// majorSFA = majorSFB = K so the SFA tensor is laid out (M, K/128)
// row-major (the natural ckpt layout produced by HF dynamic FP8 quant)
// instead of (M, K/128) col-major (the CUTLASS MN-major default).
// Same for SFB: (N/128, K/128) row-major matches the safetensors
// weight_scale_inv on disk.
constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
using ScaleConfig =
    cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM,
                                                ScaleGranularityN,
                                                ScaleGranularityK,
                                                cute::UMMA::Major::K,
                                                cute::UMMA::Major::K>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// ── Two kernel variants (Pingpong for small M, Cooperative for larger M) ──
template <class MmaTileShape_, class Schedule_>
struct GemmInstance {
  using MmaTileShape  = MmaTileShape_;
  using ClusterShape  = Shape<_1, _1, _1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      Schedule_
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using PingpongMmaTileShape = Shape<_64, _128, _128>;
using CooperativeMmaTileShape = Shape<_128, _128, _128>;

using PingpongGemm =
    typename GemmInstance<PingpongMmaTileShape,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::Gemm;
using CooperativeGemm =
    typename GemmInstance<CooperativeMmaTileShape,
        cutlass::gemm::KernelScheduleSm120Blockwise>::Gemm;

// ── Per-shape workspace cache ───────────────────────────────────
struct ShapeKey {
  int M, N, K;
  bool operator==(const ShapeKey& o) const {
    return M == o.M && N == o.N && K == o.K;
  }
};
struct ShapeKeyHash {
  size_t operator()(const ShapeKey& k) const noexcept {
    return (static_cast<size_t>(k.M) * 1315423911u)
         ^ (static_cast<size_t>(k.N) * 2654435761u)
         ^ static_cast<size_t>(k.K);
  }
};

struct CachedWorkspace {
  void*  ptr = nullptr;
  size_t size = 0;
};

std::unordered_map<ShapeKey, CachedWorkspace, ShapeKeyHash> g_ws_cache;
std::mutex g_ws_mu;

void* get_workspace(int M, int N, int K, size_t needed) {
  std::lock_guard<std::mutex> lk(g_ws_mu);
  ShapeKey key{M, N, K};
  auto it = g_ws_cache.find(key);
  if (it != g_ws_cache.end() && it->second.size >= needed) {
    return it->second.ptr;
  }
  if (it != g_ws_cache.end()) {
    cudaFree(it->second.ptr);
    g_ws_cache.erase(it);
  }
  CachedWorkspace w;
  w.size = needed;
  if (needed > 0) {
    cudaMalloc(&w.ptr, needed);
  }
  g_ws_cache[key] = w;
  return w.ptr;
}

// SM count of the current device, queried once per device.
int num_sms() {
  static std::mutex mu;
  static std::unordered_map<int, int> counts;
  int device = 0;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lk(mu);
  auto it = counts.find(device);
  if (it != counts.end()) return it->second;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  counts[device] = sms;
  return sms;
}

template <class Gemm>
cutlass::Status run_gemm(
    const void* A_fp8, const void* B_fp8, void* D_bf16,
    int M, int N, int K,
    const float* act_scale, const float* w_scale,
    cudaStream_t stream)
{
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(M, N, 1));

  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ElementA const*>(A_fp8), stride_A,
          reinterpret_cast<ElementB const*>(B_fp8), stride_B,
          act_scale, layout_SFA,
          w_scale,   layout_SFB
      },
      {
          {1.0f, 0.0f},                    // epilogue.thread (alpha, beta)
          nullptr, stride_C,                // C unused (beta = 0)
          reinterpret_cast<ElementD*>(D_bf16), stride_D
      }
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_workspace(M, N, K, ws_size);

  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp8_block128_gemm_cutlass_sm120_bf16out] can_implement FAIL "
        "for M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp8_block128_gemm_cutlass_sm120_bf16out] initialize FAIL "
        "for M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  status = gemm.run(stream);
  return status;
}

}  // namespace

void fp8_block128_gemm_cutlass_sm120_bf16out(
    const void* A_fp8,
    const void* B_fp8,
    void*       D_bf16,
    int M, int N, int K,
    const float* act_block_scale,
    const float* w_block_scale,
    cudaStream_t stream)
{
  // Schedule selection: small M -> Pingpong (better latency at low
  // arithmetic intensity); large M -> Cooperative (better throughput
  // when there are enough M-tiles to fill the SMs). A larger M also takes
  // Pingpong when its 64-row grid needs at most 7/4 the waves of the
  // 128-row grid: a 64-row tile takes about 1/1.75 of a 128-row tile's
  // time, so an under-filled 128-row grid (a tail wave, or under one wave)
  // finishes sooner as 64-row tiles. Each output element accumulates the
  // same K blocks in the same order in both, so the result is
  // bit-identical. The 7/4 is measured on an RTX 5090 (170 SMs) over 1180
  // shapes: no shape is slower than Cooperative, 609 are faster (median
  // 1.52x); e.g. at M = 512, N = 2560, K = 9728 121 -> 72 us.
  const int sms = num_sms();
  const int n_tiles = (N + 127) / 128;
  const int waves_128 = (((M + 127) / 128) * n_tiles + sms - 1) / sms;
  const int waves_64 = (((M + 63) / 64) * n_tiles + sms - 1) / sms;
  const bool pingpong = M <= 64 || 4 * waves_64 <= 7 * waves_128;
  cutlass::Status status;
  if (pingpong) {
    status = run_gemm<PingpongGemm>(
        A_fp8, B_fp8, D_bf16, M, N, K,
        act_block_scale, w_block_scale, stream);
  } else {
    status = run_gemm<CooperativeGemm>(
        A_fp8, B_fp8, D_bf16, M, N, K,
        act_block_scale, w_block_scale, stream);
  }

  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp8_block128_gemm_cutlass_sm120_bf16out] run FAIL "
        "for M=%d N=%d K=%d (status=%d); D output undefined\n",
        M, N, K, static_cast<int>(status));
  }
}

}  // namespace gemm
}  // namespace flash_rt
