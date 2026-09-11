// Block-scaled NVFP4 GEMM for Thor SM110, fp32 output.
//
// Vendored from FlashRT's cutlass_nvfp4_w4a16_gemm_sm100.cu (Apache-2.0)
// with ElementD changed from bf16 to fp32 so the result lands directly in
// ggml's fp32 dst tensor. The Sm100 CollectiveBuilder dispatch under
// KernelScheduleAuto produces the block-scaled tcgen05 mainloop when built
// for sm_110a.

#include "fr_kernels.h"

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

namespace ggml_cuda_flashrt {

namespace {

using namespace cute;

template <class TileShape, class ClusterShapeT = Shape<_1, _1, _1>>
struct FrGemmConfig {
    using ElementA           = cutlass::float_e2m1_t;
    using ElementB           = cutlass::float_e2m1_t;
    using ElementC           = float;
    using ElementD           = float;
    using ElementAccumulator = float;
    using ElementCompute     = float;
    using ElementSF          = cutlass::float_ue4m3_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value; // 4
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value; // 4

    using ClusterShape = ClusterShapeT;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
            TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator, ElementCompute,
            ElementC, LayoutC, AlignmentC,
            ElementD, LayoutD, AlignmentD,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
            ElementPairA, LayoutA, AlignmentA,
            ElementPairB, LayoutB, AlignmentB,
            ElementAccumulator,
            TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// Per-shape CUTLASS workspace cache; entries live for the process lifetime
// (weight shapes are fixed per model).
struct ShapeKey {
    int M, N, K;
    bool operator==(const ShapeKey & o) const { return M == o.M && N == o.N && K == o.K; }
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey & k) const noexcept {
        return (static_cast<size_t>(k.M) * 1315423911u)
             ^ (static_cast<size_t>(k.N) * 2654435761u)
             ^ static_cast<size_t>(k.K);
    }
};
struct CachedWorkspace { void * ptr = nullptr; size_t size = 0; };

std::unordered_map<ShapeKey, CachedWorkspace, ShapeKeyHash> g_ws_cache;
std::mutex g_ws_mu;

void * get_workspace(int M, int N, int K, size_t needed) {
    std::lock_guard<std::mutex> lk(g_ws_mu);
    ShapeKey key{M, N, K};
    auto it = g_ws_cache.find(key);
    if (it != g_ws_cache.end() && it->second.size >= needed) return it->second.ptr;
    if (it != g_ws_cache.end()) { cudaFree(it->second.ptr); g_ws_cache.erase(it); }
    CachedWorkspace w; w.size = needed;
    if (needed > 0) cudaMalloc(&w.ptr, needed);
    g_ws_cache[key] = w;
    return w.ptr;
}

template <class Config>
int run_gemm(const void * A_packed, const void * SFA,
             const void * B_packed, const void * SFB,
             float * D, int M, int N, int K,
             float alpha, cudaStream_t stream) {
    using Gemm = typename Config::Gemm;
    using ElementSF = typename Config::ElementSF;
    using ElementD  = typename Config::ElementD;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto problem_shape_MNKL = cute::make_shape(M, N, K, 1);
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

    using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ArrayElementA const *>(A_packed), stride_A,
            reinterpret_cast<ArrayElementB const *>(B_packed), stride_B,
            reinterpret_cast<ElementSF const *>(SFA), layout_SFA,
            reinterpret_cast<ElementSF const *>(SFB), layout_SFB
        },
        {
            {alpha, 0.0f},
            nullptr, stride_C,
            reinterpret_cast<ElementD *>(D), stride_D
        }
    };

    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(args);
    void * ws_ptr = get_workspace(M, N, K, ws_size);

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[fr_gemm_f32out] can_implement FAIL M=%d N=%d K=%d (status=%d)\n",
                     M, N, K, static_cast<int>(status));
        return static_cast<int>(status);
    }
    status = gemm.initialize(args, ws_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[fr_gemm_f32out] initialize FAIL M=%d N=%d K=%d (status=%d)\n",
                     M, N, K, static_cast<int>(status));
        return static_cast<int>(status);
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[fr_gemm_f32out] run FAIL M=%d N=%d K=%d (status=%d)\n",
                     M, N, K, static_cast<int>(status));
        return static_cast<int>(status);
    }
    return 0;
}

using ConfigDefault = FrGemmConfig<Shape<_128, _128, _256>>;
// 2-SM tcgen05 tile: 11-15% faster than the 1-SM tile on Thor for every
// prefill shape measured (M >= ~256); slower at decode-sized M.
using ConfigLargeM  = FrGemmConfig<Shape<_256, _128, _256>, Shape<_2, _1, _1>>;

} // namespace

int gemm_f32out(const void * A_packed, const void * SFA,
                const void * B_packed, const void * SFB,
                float * D, int M, int N, int K,
                float alpha, bool widen, cudaStream_t stream) {
    (void) widen; // superseded by the M-based tile choice
    if (M >= 256) {
        return run_gemm<ConfigLargeM>(A_packed, SFA, B_packed, SFB, D, M, N, K, alpha, stream);
    }
    return run_gemm<ConfigDefault>(A_packed, SFA, B_packed, SFB, D, M, N, K, alpha, stream);
}

} // namespace ggml_cuda_flashrt
