// ================================================================
// FlashRT — CUTLASS SM80 INT8 rowwise GEMM with FP16 output
//
// Same math as cutlass_sm80_int8_rowwise (per-row activation scale +
// per-row weight scale INT32→FP32 dequant epilogue), but writes FP16
// directly instead of BF16. Skips the cast_bf16_to_fp16 that would
// otherwise follow every INT8 GEMM feeding an FP16 consumer.
//
// Savings on the Chameleon-7B path (Orin SM87):
//   - 224 GEMMs per forward (32 layers × 7 projections)
//   - Each cast is ~30-50 μs on the Orin bandwidth budget
//   - ~10-15 ms saved per E2E replay
//
// Optionally supports fused per-N bias add: y[m,n] += bias[n] as a
// third VisitorRowBroadcast in the epilogue chain, eliminating the
// separate add_bias_fp16 launch that follows Q/K/V/O projections.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

#include "cute/tensor.hpp"

namespace flash_rt {
namespace gemm {
namespace cutlass_int8_sm8x_fp16out {

using namespace cute;

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementOutput = cutlass::half_t;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = int32_t;
using ElementCompute = float;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 8;

using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
// Stages=5 measured best on Orin SM87 (64-69 vs 58-60 TOPS at s4):
// 80 KB smem/block still fits 2 blocks/SM (164 KB), deeper cp.async
// pipeline hides more DRAM latency on the 16-SM part.
constexpr int NumStages = 5;
constexpr int EVTEpilogueStages = 1;

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementOutput, AlignmentC, EVTEpilogueStages>;

// Rowwise scale visitors — identical to the BF16-out kernel.
using AccFetch = cutlass::epilogue::threadblock::VisitorAccFetch;
using ActScaleLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float, Stride<_1, _0, _0>>;
using WtScaleLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, Stride<_0, _1, int32_t>>;
using MulActScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using MulWtScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

// Bias load: FP16 [N] broadcast across M. Used only in the *_bias variant.
using BiasLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, cutlass::half_t, Stride<_0, _1, int32_t>>;
using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

// EVT chain (no bias): acc → mul act_scale → mul wt_scale → store fp16.
using EVT_AccMulAct = cutlass::epilogue::threadblock::Sm80EVT<
    MulActScale, AccFetch, ActScaleLoad>;
using EVT_MulBoth = cutlass::epilogue::threadblock::Sm80EVT<
    MulWtScale, EVT_AccMulAct, WtScaleLoad>;
using EVT_NoBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_MulBoth>;

// EVT chain (with bias): acc → mul act_scale → mul wt_scale → +bias → store fp16.
using EVT_AddBias = cutlass::epilogue::threadblock::Sm80EVT<
    AddBias, EVT_MulBoth, BiasLoad>;
using EVT_WithBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_AddBias>;

using GemmKernelNoBias = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EVT_NoBias,
    // Group-4 L2-aware rasterization: Orin's 16-SM waves re-streamed the
    // whole B (weight) matrix from DRAM once per tile-row under the
    // default identity swizzle (measured 16-44 TOPS vs 85 TOPS mma peak).
    // Grouping 4 tile-rows makes waves share A/B tiles in L2:
    // QKVO 3.7x, gate/up 1.3x, down 1.25x. Bit-identical output (block
    // scheduling order only; INT32 accumulation unchanged).
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EVTEpilogueStages
>::GemmKernel;

using GemmKernelWithBias = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EVT_WithBias,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EVTEpilogueStages
>::GemmKernel;

using GemmDeviceNoBias = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelNoBias>;
using GemmDeviceWithBias = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithBias>;

static int run_no_bias(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename EVT_NoBias::Arguments evt_args{
        {
            {
                {},
                {reinterpret_cast<float const*>(act_scale), 1.0f, {}},
                {}
            },
            {reinterpret_cast<float const*>(weight_scale), 1.0f, {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {reinterpret_cast<ElementOutput*>(D),
         {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}}
    };

    typename GemmDeviceNoBias::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        1,
        evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr,
        nullptr,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(N) * K,
        0,
        0,
        K,
        K,
        N,
        N
    );

    GemmDeviceNoBias gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "[cutlass_int8_fp16out] can_implement failed: M=%d N=%d K=%d code=%d\n",
                     M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    size_t ws_sz = GemmDeviceNoBias::get_workspace_size(args);
    static void* ws_ptr = nullptr;
    static size_t ws_cap = 0;
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) {
            ws_ptr = nullptr;
            ws_cap = 0;
            return -1;
        }
        ws_cap = ws_sz;
    }

    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) {
        return static_cast<int>(st) | 0x20000;
    }
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

static int run_with_bias(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void const* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename EVT_WithBias::Arguments evt_args{
        {
            {
                {
                    {},
                    {reinterpret_cast<float const*>(act_scale), 1.0f, {}},
                    {}
                },
                {reinterpret_cast<float const*>(weight_scale), 1.0f, {_0{}, _1{}, int32_t(N)}},
                {}
            },
            {reinterpret_cast<cutlass::half_t const*>(bias), cutlass::half_t(0), {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {reinterpret_cast<ElementOutput*>(D),
         {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}}
    };

    typename GemmDeviceWithBias::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        1,
        evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr,
        nullptr,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(N) * K,
        0,
        0,
        K,
        K,
        N,
        N
    );

    GemmDeviceWithBias gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "[cutlass_int8_fp16out_bias] can_implement failed: M=%d N=%d K=%d code=%d\n",
                     M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    size_t ws_sz = GemmDeviceWithBias::get_workspace_size(args);
    static void* ws_ptr = nullptr;
    static size_t ws_cap = 0;
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) {
            ws_ptr = nullptr;
            ws_cap = 0;
            return -1;
        }
        ws_cap = ws_sz;
    }

    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) {
        return static_cast<int>(st) | 0x20000;
    }
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace cutlass_int8_sm8x_fp16out
}  // namespace gemm
}  // namespace flash_rt

// Forward declarations for the alt-tile variant defined in
// cutlass_sm80_int8_rowwise_fp16out_t64x128.cu.
extern "C" int cutlass_int8_rowwise_fp16out_t64x128(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream);

extern "C" int cutlass_int8_rowwise_fp16out_bias_t64x128(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias, void* D,
    int M, int N, int K, cudaStream_t stream);

// Alt-tile for long-K large-M (cutlass_sm80_int8_rowwise_fp16out_t256x128.cu).
extern "C" int cutlass_int8_rowwise_fp16out_t256x128(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream);

// With the group-4 swizzle on the 128×128 kernel, only true small-M
// (decoder / action-head) work still benefits from the 64×128 tile.
// The old "N in (2048, 4096] → 64×128" clause was an Id1-swizzle-era
// artifact: 128×128+Id4 measures 0.69 ms vs 1.66 ms (64×128) on the
// (1214, 4096, 4096) QKVO shape.
static inline bool prefer_t64x128_for_fp16out(int M, int N) {
    (void)N;
    return M <= 64;
}

static bool fp16out_tile_dispatch_enabled() {
    static const int v = []() {
        const char* env = std::getenv("FVK_ORIN_INT8_NO_TILE_DISPATCH");
        return (env && env[0] == '1') ? 0 : 1;
    }();
    return v != 0;
}

extern "C" int cutlass_int8_rowwise_fp16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    if (fp16out_tile_dispatch_enabled() && prefer_t64x128_for_fp16out(M, N)) {
        return cutlass_int8_rowwise_fp16out_t64x128(
            A, B, act_scale, weight_scale, D, M, N, K, stream);
    }
    // Long-K large-M (FFN down, K=11008): 256×128 s5 tile measures +22-29%
    // over 128×128 s5 on Orin SM87 (fewer K-loop passes per output row).
    if (fp16out_tile_dispatch_enabled() && M >= 256 && K >= 8192) {
        return cutlass_int8_rowwise_fp16out_t256x128(
            A, B, act_scale, weight_scale, D, M, N, K, stream);
    }
    return flash_rt::gemm::cutlass_int8_sm8x_fp16out::run_no_bias(
        A, B, act_scale, weight_scale, D, M, N, K, stream);
}

extern "C" int cutlass_int8_rowwise_fp16out_bias(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void const* bias,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    if (fp16out_tile_dispatch_enabled() && prefer_t64x128_for_fp16out(M, N)) {
        return cutlass_int8_rowwise_fp16out_bias_t64x128(
            A, B, act_scale, weight_scale, bias, D, M, N, K, stream);
    }
    return flash_rt::gemm::cutlass_int8_sm8x_fp16out::run_with_bias(
        A, B, act_scale, weight_scale, bias, D, M, N, K, stream);
}
