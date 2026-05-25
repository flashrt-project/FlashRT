// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS NVFP4 W4A16 block-scaled GEMM, SM120a, BF16 output.
// Header: cutlass_nvfp4_w4a16_gemm_sm120.cuh
//
// Template is a direct port of NVIDIA's verified unit test at
//   third_party/cutlass/test/unit/gemm/device/
//     sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_bf16.cu
// (one config: TileShape <128,128,256>, ClusterShape <1,1,1>,
//  KernelTmaWarpSpecializedCooperative, OpClassBlockScaledTensorOp).
//
// Why this is the "vendor-best" path (not hand-written):
//   * Uses CUTLASS 4.x's `CollectiveBuilder` for SM120 BlockScaled
//     mainloop + epilogue. NVIDIA tunes the mainloop schedule.
//   * Same family as the existing FP8 SM120 GEMM (cutlass_sm120_block128_
//     fp8_gemm.cu) — only the element types and `OpClass` differ.
//   * No handwritten PTX or custom layout swizzling.
//
// Per-shape argument cache + workspace match the FP8 path so the
// hot-path is launch-only.

#include "cutlass_nvfp4_w4a16_gemm_sm120.cuh"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
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

// ── Element / layout types (copy of unit test "kernel_1") ────────
using ElementA           = cutlass::float_e2m1_t;
using ElementB           = cutlass::float_e2m1_t;
using ElementC           = cutlass::bfloat16_t;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using ElementSF          = cutlass::float_ue4m3_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
// Public API hands us D in row-major (matches the FP8 sm120 kernel
// and what HF / our pipeline downstream expects for (M, N) tensors).
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// 16-byte alignments in bits = 16 * 8 = 128. Convert to element count.
constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;  // 32
constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;  // 32
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;     // 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;     // 8

using TileShape    = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

// SF tensor layout helper. Vector size = 16 (NVFP4 group) per ckpt.
// Sm1xxBlockScaledConfig generates the (M-blk, K-blk) atom layout
// CUTLASS expects on-device. Our weight loader and act quantizer
// must produce SF in this exact layout (transformed once).
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

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
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ── Per-shape workspace cache (mirrors FP8 path) ─────────────────
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
  void*  ptr  = nullptr;
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

cutlass::Status run_gemm(
    const void* A_packed, const void* B_packed, void* D_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha,
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

  auto problem_shape_MNKL = cute::make_shape(M, N, K, 1);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

  using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ArrayElementA const*>(A_packed), stride_A,
          reinterpret_cast<ArrayElementB const*>(B_packed), stride_B,
          reinterpret_cast<ElementSF const*>(SFA), layout_SFA,
          reinterpret_cast<ElementSF const*>(SFB), layout_SFB
      },
      {
          {alpha, 0.0f},                                      // (alpha, beta)
          nullptr, stride_C,                                  // C unused (beta=0)
          reinterpret_cast<ElementD*>(D_bf16), stride_D
      }
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_workspace(M, N, K, ws_size);

  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out] can_implement FAIL "
        "M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out] initialize FAIL "
        "M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  return gemm.run(stream);
}

}  // namespace

void fp4_w4a16_gemm_sm120_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream)
{
  cutlass::Status status = run_gemm(
      A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out] run FAIL "
        "M=%d N=%d K=%d (status=%d); D output undefined\n",
        M, N, K, static_cast<int>(status));
  }
}

// ============================================================
// WIDEN variant: TileShape <128, 256, 128>. Same kernel template
// machinery as above but a wider N tile + narrower K. Profiled
// faster on shapes with very large N (lm_head N=248320: 88% BW vs
// 64% baseline; MLP gate/up N=17408: 66% vs 56%). Slower on small/
// medium-N shapes (k/v_proj N=1024, lin in_proj_z N=6144, etc.) so
// callers dispatch by shape.
// ============================================================
namespace widen {

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

using TileShape    = Shape<_128, _256, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

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
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

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
struct CachedWorkspace { void* ptr = nullptr; size_t size = 0; };
std::unordered_map<ShapeKey, CachedWorkspace, ShapeKeyHash> g_ws_cache_widen;
std::mutex g_ws_mu_widen;

void* get_workspace_widen(int M, int N, int K, size_t needed) {
  std::lock_guard<std::mutex> lk(g_ws_mu_widen);
  ShapeKey key{M, N, K};
  auto it = g_ws_cache_widen.find(key);
  if (it != g_ws_cache_widen.end() && it->second.size >= needed) {
    return it->second.ptr;
  }
  if (it != g_ws_cache_widen.end()) {
    cudaFree(it->second.ptr);
    g_ws_cache_widen.erase(it);
  }
  CachedWorkspace w;
  w.size = needed;
  if (needed > 0) cudaMalloc(&w.ptr, needed);
  g_ws_cache_widen[key] = w;
  return w.ptr;
}

cutlass::Status run_gemm_widen(
    const void* A_packed, const void* B_packed, void* D_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha,
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

  auto problem_shape_MNKL = cute::make_shape(M, N, K, 1);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

  using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ArrayElementA const*>(A_packed), stride_A,
          reinterpret_cast<ArrayElementB const*>(B_packed), stride_B,
          reinterpret_cast<ElementSF const*>(SFA), layout_SFA,
          reinterpret_cast<ElementSF const*>(SFB), layout_SFB
      },
      {
          {alpha, 0.0f},
          nullptr, stride_C,
          reinterpret_cast<ElementD*>(D_bf16), stride_D
      }
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_workspace_widen(M, N, K, ws_size);

  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out_widen] can_implement FAIL "
        "M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out_widen] initialize FAIL "
        "M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  return gemm.run(stream);
}

}  // namespace widen

void fp4_w4a16_gemm_sm120_bf16out_widen(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream)
{
  cutlass::Status status = widen::run_gemm_widen(
      A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out_widen] run FAIL "
        "M=%d N=%d K=%d (status=%d); D output undefined\n",
        M, N, K, static_cast<int>(status));
  }
}

// ============================================================
// PINGPONG variant: default <128,128,256> tile with
// KernelTmaWarpSpecializedPingpong. This is intentionally separate from
// the production entrypoint so Qwen can A/B per shape before any dispatch
// policy is changed.
// ============================================================
namespace pingpong {

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

using TileShape    = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

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
        cutlass::gemm::KernelTmaWarpSpecializedPingpong
    >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

std::unordered_map<ShapeKey, CachedWorkspace, ShapeKeyHash> g_ws_cache_pingpong;
std::mutex g_ws_mu_pingpong;

void* get_workspace_pingpong(int M, int N, int K, size_t needed) {
  std::lock_guard<std::mutex> lk(g_ws_mu_pingpong);
  ShapeKey key{M, N, K};
  auto it = g_ws_cache_pingpong.find(key);
  if (it != g_ws_cache_pingpong.end() && it->second.size >= needed) {
    return it->second.ptr;
  }
  if (it != g_ws_cache_pingpong.end()) {
    cudaFree(it->second.ptr);
    g_ws_cache_pingpong.erase(it);
  }
  CachedWorkspace w;
  w.size = needed;
  if (needed > 0) cudaMalloc(&w.ptr, needed);
  g_ws_cache_pingpong[key] = w;
  return w.ptr;
}

cutlass::Status run_gemm_pingpong(
    const void* A_packed, const void* B_packed, void* D_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha,
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

  auto problem_shape_MNKL = cute::make_shape(M, N, K, 1);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

  using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ArrayElementA const*>(A_packed), stride_A,
          reinterpret_cast<ArrayElementB const*>(B_packed), stride_B,
          reinterpret_cast<ElementSF const*>(SFA), layout_SFA,
          reinterpret_cast<ElementSF const*>(SFB), layout_SFB
      },
      {
          {alpha, 0.0f},
          nullptr, stride_C,
          reinterpret_cast<ElementD*>(D_bf16), stride_D
      }
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_workspace_pingpong(M, N, K, ws_size);

  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out_pingpong] can_implement FAIL "
        "M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out_pingpong] initialize FAIL "
        "M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return status;
  }
  return gemm.run(stream);
}

}  // namespace pingpong

void fp4_w4a16_gemm_sm120_bf16out_pingpong(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int M, int N, int K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream)
{
  cutlass::Status status = pingpong::run_gemm_pingpong(
      A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_sm120_bf16out_pingpong] run FAIL "
        "M=%d N=%d K=%d (status=%d); D output undefined\n",
        M, N, K, static_cast<int>(status));
  }
}

}  // namespace gemm
}  // namespace flash_rt
