// ================================================================
// FlashRT — CUTLASS SM80 INT8 rowwise GEMM with FP16 output (64×128)
//
// Alt-tile companion to cutlass_sm80_int8_rowwise_fp16out (128×128).
// Same math, smaller M-tile for shapes where 128 wastes wave packing:
//   - M ≤ 64 (decoder / action-head)
//   - Awkward N in (2048, 4096] (Chameleon QKV/O at N=4096)
// Selected by the runtime dispatcher (prefer_t64x128_for_shape) in the
// 128×128 file, mirroring the BF16-out layout.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

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
namespace cutlass_int8_sm8x_fp16out_t64x128 {

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
using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int NumStages = 4;
constexpr int EVTEpilogueStages = 1;

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementOutput, AlignmentC, EVTEpilogueStages>;

using AccFetch = cutlass::epilogue::threadblock::VisitorAccFetch;
using ActScaleLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float, Stride<_1, _0, _0>>;
using WtScaleLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, Stride<_0, _1, int32_t>>;
using MulActScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using MulWtScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using BiasLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, cutlass::half_t, Stride<_0, _1, int32_t>>;
using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

using EVT_AccMulAct = cutlass::epilogue::threadblock::Sm80EVT<
    MulActScale, AccFetch, ActScaleLoad>;
using EVT_MulBoth = cutlass::epilogue::threadblock::Sm80EVT<
    MulWtScale, EVT_AccMulAct, WtScaleLoad>;
using EVT_NoBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_MulBoth>;
using EVT_AddBias = cutlass::epilogue::threadblock::Sm80EVT<
    AddBias, EVT_MulBoth, BiasLoad>;
using EVT_WithBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_AddBias>;

using GemmKernelNoBias = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator, ElementCompute, OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVT_NoBias,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTEpilogueStages
>::GemmKernel;

using GemmKernelWithBias = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator, ElementCompute, OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVT_WithBias,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTEpilogueStages
>::GemmKernel;

using GemmDeviceNoBias = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelNoBias>;
using GemmDeviceWithBias = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelWithBias>;

static int run_no_bias(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename EVT_NoBias::Arguments evt_args{
        {
            {{}, {reinterpret_cast<float const*>(act_scale), 1.0f, {}}, {}},
            {reinterpret_cast<float const*>(weight_scale), 1.0f, {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {reinterpret_cast<ElementOutput*>(D),
         {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N}}
    };
    typename GemmDeviceNoBias::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, 1, evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr, nullptr,
        static_cast<int64_t>(M) * K, static_cast<int64_t>(N) * K, 0, 0,
        K, K, N, N);
    GemmDeviceNoBias gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[int8_fp16out_t64x128] can_implement failed: %d\n",
                     static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }
    size_t ws_sz = GemmDeviceNoBias::get_workspace_size(args);
    static void* ws_ptr = nullptr; static size_t ws_cap = 0;
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) { ws_ptr = nullptr; ws_cap = 0; return -1; }
        ws_cap = ws_sz;
    }
    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x20000;
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

static int run_with_bias(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias, void* D,
    int M, int N, int K, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename EVT_WithBias::Arguments evt_args{
        {
            {
                {{}, {reinterpret_cast<float const*>(act_scale), 1.0f, {}}, {}},
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
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, 1, evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr, nullptr,
        static_cast<int64_t>(M) * K, static_cast<int64_t>(N) * K, 0, 0,
        K, K, N, N);
    GemmDeviceWithBias gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[int8_fp16out_t64x128_bias] can_implement failed: %d\n",
                     static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }
    size_t ws_sz = GemmDeviceWithBias::get_workspace_size(args);
    static void* ws_ptr = nullptr; static size_t ws_cap = 0;
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) { ws_ptr = nullptr; ws_cap = 0; return -1; }
        ws_cap = ws_sz;
    }
    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x20000;
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace cutlass_int8_sm8x_fp16out_t64x128
}  // namespace gemm
}  // namespace flash_rt

extern "C" int cutlass_int8_rowwise_fp16out_t64x128(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream) {
    return flash_rt::gemm::cutlass_int8_sm8x_fp16out_t64x128::run_no_bias(
        A, B, act_scale, weight_scale, D, M, N, K, stream);
}

extern "C" int cutlass_int8_rowwise_fp16out_bias_t64x128(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias, void* D,
    int M, int N, int K, cudaStream_t stream) {
    return flash_rt::gemm::cutlass_int8_sm8x_fp16out_t64x128::run_with_bias(
        A, B, act_scale, weight_scale, bias, D, M, N, K, stream);
}
