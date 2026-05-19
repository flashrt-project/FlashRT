#include <cuda_runtime.h>
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
namespace cutlass_int8_sm8x {

using namespace cute;

using ElementA = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementOutput = cutlass::bfloat16_t;
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
using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

using EVT_AccMulAct = cutlass::epilogue::threadblock::Sm80EVT<
    MulActScale, AccFetch, ActScaleLoad>;
using EVT_MulBoth = cutlass::epilogue::threadblock::Sm80EVT<
    MulWtScale, EVT_AccMulAct, WtScaleLoad>;
using EVT_NoBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_MulBoth>;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
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
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EVTEpilogueStages
>::GemmKernel;

using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

static int run(
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

    typename GemmDevice::Arguments args(
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

    GemmDevice gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
                     "[cutlass_int8_sm8x] can_implement failed: M=%d N=%d K=%d code=%d\n",
                     M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    size_t ws_sz = GemmDevice::get_workspace_size(args);
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
        std::fprintf(stderr,
                     "[cutlass_int8_sm8x] init failed: M=%d N=%d K=%d code=%d\n",
                     M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x20000;
    }

    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace cutlass_int8_sm8x
}  // namespace gemm
}  // namespace flash_rt

// Forward declaration for the alt-tile variant (defined in
// cutlass_sm80_int8_rowwise_t64x128.cu). ABI: identical signature.
extern "C" int cutlass_int8_rowwise_bf16out_t64x128(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream);

// Per-Pi0.5-shape tile selection on Orin SM87. Measured (per-shape
// microbench on this branch, 200 iters each):
//
//   shape                       128×128   64×128    winner
//   enc_qkv  (522,2560,2048)    328 us    214 us    64×128  ⭐ 1.54×
//   enc_o    (522,2048,2048)     95 us    113 us    128×128
//   enc_gate (522,8192,2048)    337 us    398 us    128×128
//   enc_up   (522,8192,2048)    331 us    399 us    128×128
//   enc_down (522,2048,16384)  1126 us   1419 us    128×128
//   dec_*    (M=10, N≤8192)     ~50 us    ~46 us   64×128 ⭐ 1.08-1.12×
//
// Rule of thumb that captures every winning case here:
//   * M ≤ 64 (decoder small-M):         64×128 wins (always)
//   * M > 64 AND N in (2048, 4096]:     64×128 wins (qkv-shape — odd N
//     above the 2048 wave boundary causes a costly partial wave on 128
//     tile; the 64-M tile redistributes the slack)
//   * else (big-N or M-aligned):        128×128 wins (default)
static inline bool prefer_t64x128_for_shape(int M, int N) {
    if (M <= 64) return true;                        // decoder
    if (N > 2048 && N <= 4096) return true;          // qkv-like awkward N
    return false;
}

// Tile dispatch is BIT-EQUIVALENT to the default-128×128 baseline:
// CUTLASS INT8 GEMM accumulates in INT32 (associative integer math, no
// overflow for our shapes K ≤ 16384 × 127² ≈ 33M ≪ 2³¹), and the EVT
// fp32-dequant epilogue uses the same multiply order regardless of
// threadblock tile size. Verified empirically: 6/6 frames maxabs=0.
//
// To opt OUT (e.g. for debugging tile-shape regressions), set
// FVK_PI05_RTX_INT8_NO_TILE_DISPATCH=1.
static bool tile_dispatch_enabled() {
    static const int v = []() {
        const char* env = std::getenv("FVK_PI05_RTX_INT8_NO_TILE_DISPATCH");
        return (env && env[0] == '1') ? 0 : 1;
    }();
    return v != 0;
}

extern "C" int cutlass_int8_rowwise_bf16out(
    void const* A,
    void const* B,
    void const* act_scale,
    void const* weight_scale,
    void* D,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    if (tile_dispatch_enabled() && prefer_t64x128_for_shape(M, N)) {
        return cutlass_int8_rowwise_bf16out_t64x128(
            A, B, act_scale, weight_scale, D, M, N, K, stream);
    }
    return flash_rt::gemm::cutlass_int8_sm8x::run(
        A, B, act_scale, weight_scale, D, M, N, K, stream);
}
