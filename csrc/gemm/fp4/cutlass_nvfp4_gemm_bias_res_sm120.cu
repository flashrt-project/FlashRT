// SPDX-License-Identifier: Apache-2.0
// CUTLASS NVFP4 GEMM + per-column bias + residual source, BF16 out, SM120a. See the header.
#include "cutlass_nvfp4_gemm_bias_res_sm120.cuh"

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

#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace gemm {
namespace {

using namespace cute;

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementBias = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementSF = cutlass::float_ue4m3_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;
using TileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

using Fusion = cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
    cutlass::epilogue::thread::Identity, ElementD, ElementCompute, ElementBias, ElementC>;

template <class MainloopSchedule, class TileScheduler>
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
        MainloopSchedule>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue,
                                                            TileScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using PingpongGemm =
    Kernel<cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::gemm::PersistentScheduler>::Gemm;
using StreamKGemm =
    Kernel<cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::gemm::StreamKScheduler>::Gemm;

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
int run(const void* A, const void* B, const void* SFA, const void* SFB, const void* bias, const void* C, void* D,
        int M, int N, int K, const float* alpha, float beta, int s, cudaStream_t stream) {
    using GK = typename Gemm::GemmKernel;
    auto strA = cutlass::make_cute_packed_stride(typename GK::StrideA{}, cute::make_shape(M, K, 1));
    auto strB = cutlass::make_cute_packed_stride(typename GK::StrideB{}, cute::make_shape(N, K, 1));
    auto strC = cutlass::make_cute_packed_stride(typename GK::StrideC{}, cute::make_shape(M, N, 1));
    auto strD = cutlass::make_cute_packed_stride(typename GK::StrideD{}, cute::make_shape(M, N, 1));
    auto problem = cute::make_shape(M, N, K, 1);
    using ArrayA = typename GK::CollectiveMainloop::ArrayElementA;
    using ArrayB = typename GK::CollectiveMainloop::ArrayElementB;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<ArrayA const*>(A), strA, reinterpret_cast<ArrayB const*>(B), strB,
         reinterpret_cast<ElementSF const*>(SFA), Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem),
         reinterpret_cast<ElementSF const*>(SFB), Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem)},
        {{1.0f, beta}, reinterpret_cast<ElementC const*>(C), strC, reinterpret_cast<ElementD*>(D), strD}};
    args.epilogue.thread.alpha_ptr = alpha;
    args.epilogue.thread.bias_ptr = reinterpret_cast<ElementBias const*>(bias);
    Gemm gemm;
    const size_t need = Gemm::get_workspace_size(args);
    void* ws = workspace(M, N, K, s, need);
    if (need > 0 && ws == nullptr) return static_cast<int>(cutlass::Status::kErrorWorkspaceNull);
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
    st = gemm.run(stream);
    return static_cast<int>(st);
}

}  // namespace

int nvfp4_gemm_bias_res_bf16out_sm120(const void* A, const void* B, const void* SFA, const void* SFB,
                                      const void* bias, const void* C, void* D, int M, int N, int K,
                                      const float* alpha, float beta, int schedule, cudaStream_t stream) {
    if (!A || !B || !SFA || !SFB || !bias || !D || !alpha || (beta != 0.f && !C) || M < 1 || N % 8 || K % 128)
        return static_cast<int>(cutlass::Status::kErrorInvalidProblem);
    if (schedule == 0)
        return run<PingpongGemm>(A, B, SFA, SFB, bias, C, D, M, N, K, alpha, beta, 0, stream);
    if (schedule == 1)
        return run<StreamKGemm>(A, B, SFA, SFB, bias, C, D, M, N, K, alpha, beta, 1, stream);
    return static_cast<int>(cutlass::Status::kErrorInvalidProblem);
}

}  // namespace gemm
}  // namespace flash_rt
