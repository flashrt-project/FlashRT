// SPDX-License-Identifier: Apache-2.0
// CUTLASS NVFP4 GEMMs with GELU / GeGLU NVFP4-output epilogues and an explicit
// output global scale, SM120a. See the header.
#include "cutlass_nvfp4_gemm_glu_fp4out_sm120.cuh"

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

#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace gemm {
namespace {

using namespace cute;

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::float_e2m1_t;
using ElementSFD = cutlass::float_ue4m3_t;
using ElementBias = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementSF = cutlass::float_ue4m3_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using LayoutSFDTag = cutlass::layout::RowMajor;
using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 32;
using TileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

template <class Fusion>
struct Kernel {
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementPairA, LayoutA, AlignmentA,
        ElementPairB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue,
                                                            cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Per-column bias + GELU(tanh) + per-16 NVFP4 output.
using GeluFusion = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::GELU_taylor, 16, ElementD, ElementCompute, ElementSFD, LayoutSFDTag, ElementBias>;
using GeluGemm = Kernel<GeluFusion>::Gemm;

// GeGLU over interleaved gate|up columns: scale vector 32 on the interleaved
// width == 16 outputs after the 2 -> 1 collapse.
using GegluFusion = cutlass::epilogue::fusion::GeluMulBlockScaleFactor<32, ElementD, ElementCompute, ElementSFD,
                                                                       LayoutSFDTag>;
using GegluGemm = Kernel<GegluFusion>::Gemm;

struct WsKey {
    int M, N, K, s;
    bool operator==(const WsKey& o) const { return M == o.M && N == o.N && K == o.K && s == o.s; }
};
struct WsHash {
    size_t operator()(const WsKey& k) const noexcept {
        return (size_t(k.M) * 1315423911u) ^ (size_t(k.N) * 2654435761u) ^ (size_t(k.K) * 97u) ^ size_t(k.s);
    }
};
std::unordered_map<WsKey, std::pair<void*, size_t>, WsHash> g_ws;
std::mutex g_mu;

void* workspace(int M, int N, int K, int s, size_t need) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto& slot = g_ws[WsKey{M, N, K, s}];
    if (slot.second >= need) return slot.first;
    if (slot.first) cudaFree(slot.first);
    slot = {nullptr, need};
    if (need > 0 && cudaMalloc(&slot.first, need) != cudaSuccess) slot = {nullptr, 0};
    return slot.first;
}

template <class Gemm>
typename Gemm::Arguments make_args(const void* A, const void* B, const void* SFA, const void* SFB, void* D, int M,
                                   int N, int K) {
    using GK = typename Gemm::GemmKernel;
    auto strA = cutlass::make_cute_packed_stride(typename GK::StrideA{}, cute::make_shape(M, K, 1));
    auto strB = cutlass::make_cute_packed_stride(typename GK::StrideB{}, cute::make_shape(N, K, 1));
    auto strC = cutlass::make_cute_packed_stride(typename GK::StrideC{}, cute::make_shape(M, N, 1));
    auto strD = cutlass::make_cute_packed_stride(typename GK::StrideD{}, cute::make_shape(M, N, 1));
    auto problem = cute::make_shape(M, N, K, 1);
    using ArrayA = typename GK::CollectiveMainloop::ArrayElementA;
    using ArrayB = typename GK::CollectiveMainloop::ArrayElementB;
    return typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<ArrayA const*>(A), strA, reinterpret_cast<ArrayB const*>(B), strB,
         reinterpret_cast<ElementSF const*>(SFA), Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem),
         reinterpret_cast<ElementSF const*>(SFB), Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem)},
        {{1.0f, 0.0f}, nullptr, strC, reinterpret_cast<ElementD*>(D), strD}};
}

template <class Gemm>
int launch(typename Gemm::Arguments& args, int M, int N, int K, int s, cudaStream_t stream) {
    Gemm gemm;
    const size_t need = Gemm::get_workspace_size(args);
    void* ws = workspace(M, N, K, s, need);
    if (need > 0 && ws == nullptr) return static_cast<int>(cutlass::Status::kErrorWorkspaceNull);
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
    return static_cast<int>(gemm.run(stream));
}

bool shape_ok(int M, int N, int K) { return M > 0 && M % 128 == 0 && N % 32 == 0 && K % 128 == 0; }

}  // namespace

int nvfp4_gemm_bias_gelu_fp4out_sm120(const void* A, const void* B, const void* SFA, const void* SFB,
                                      const void* bias, void* D_packed, void* SFD, int M, int N, int K,
                                      const float* alpha, const float* norm_constant, cudaStream_t stream) {
    if (!A || !B || !SFA || !SFB || !bias || !D_packed || !SFD || !alpha || !norm_constant || !shape_ok(M, N, K))
        return static_cast<int>(cutlass::Status::kErrorInvalidProblem);
    auto args = make_args<GeluGemm>(A, B, SFA, SFB, D_packed, M, N, K);
    args.epilogue.thread.alpha_ptr = alpha;
    args.epilogue.thread.bias_ptr = reinterpret_cast<ElementBias const*>(bias);
    args.epilogue.thread.block_scale_factor_ptr = reinterpret_cast<ElementSFD*>(SFD);
    args.epilogue.thread.norm_constant_ptr = norm_constant;
    return launch<GeluGemm>(args, M, N, K, 0, stream);
}

int nvfp4_gemm_geglu_fold_fp4out_sm120(const void* A, const void* B_il, const void* SFA, const void* SFB_il,
                                       void* D_packed, void* SFD, int M, int N_il, int K, const float* alpha_gate,
                                       const float* alpha_up, const float* norm_constant, cudaStream_t stream) {
    if (!A || !B_il || !SFA || !SFB_il || !D_packed || !SFD || !alpha_gate || !alpha_up || !norm_constant ||
        !shape_ok(M, N_il, K) || N_il % 64 != 0)
        return static_cast<int>(cutlass::Status::kErrorInvalidProblem);
    auto args = make_args<GegluGemm>(A, B_il, SFA, SFB_il, D_packed, M, N_il, K);
    args.epilogue.thread.block_scale_factor_ptr = reinterpret_cast<ElementSFD*>(SFD);
    args.epilogue.thread.norm_constant_ptr = norm_constant;
    args.epilogue.thread.alpha_gate_ptr = alpha_gate;
    args.epilogue.thread.alpha_up_ptr = alpha_up;
    return launch<GegluGemm>(args, M, N_il, K, 1, stream);
}

}  // namespace gemm
}  // namespace flash_rt
