// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS NVFP4 W4A16 GEMM with fused per-col bias + GELU(tanh) +
// per-block-16 NVFP4 quantization epilogue, FP4 packed output, SM120a.
//
// Replaces the 3-launch chain
//   cutlass NVFP4 GEMM_up    (~33 µs)
// + bias_gelu_inplace_bf16   (~8  µs)
// + quantize_bf16_to_nvfp4   (~20 µs)
// with a single cutlass-fork kernel (~33 µs).
//
// Same TileShape <128,128,256> ClusterShape <1,1,1>
// KernelTmaWarpSpecializedPingpong as the bf16-out fork; the
// FusionOperation is swapped to LinCombPerColBiasEltActBlockScaleFactor
// which produces packed NVFP4 + UE4M3 SF in cutlass-swizzled layout
// (consumable directly by the downstream NVFP4 W4A16 GEMM_dn that reads
// the same SF layout).

#include "cutlass_nvfp4_gemm_bias_gelu_fp4out_sm120.cuh"

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

constexpr int OutputSFVectorSize = 16;

using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::GELU_taylor,
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD,
    LayoutSFDTag,
    ElementBias>;

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

}  // namespace

void fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
    const void* A_packed, const void* B_packed,
    const void* SFA,      const void* SFB,
    const void* bias_bf16,
    void*       D_packed,
    void*       SFD,
    int M, int N, int K,
    float alpha,
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
          reinterpret_cast<ArrayElementB const*>(B_packed), strB,
          reinterpret_cast<ElementSF const*>(SFA), layout_SFA,
          reinterpret_cast<ElementSF const*>(SFB), layout_SFB
      },
      {
          {alpha, 0.0f},
          nullptr, strC,
          reinterpret_cast<ElementD*>(D_packed), strD
      }
  };
  args.epilogue.thread.bias_ptr =
      reinterpret_cast<ElementBias const*>(bias_bf16);
  args.epilogue.thread.block_scale_factor_ptr =
      reinterpret_cast<ElementSFD*>(SFD);
  args.epilogue.thread.norm_constant_ptr = norm_const_dev;

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_ws(M, N, K, ws_size);
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_bias_gelu_fp4out_sm120] can_implement FAIL "
        "M=%d N=%d K=%d status=%d\n", M, N, K, int(status));
    return;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_bias_gelu_fp4out_sm120] initialize FAIL "
        "M=%d N=%d K=%d status=%d\n", M, N, K, int(status));
    return;
  }
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[fp4_w4a16_gemm_bias_gelu_fp4out_sm120] run FAIL status=%d\n",
        int(status));
  }
}

}  // namespace gemm
}  // namespace flash_rt
