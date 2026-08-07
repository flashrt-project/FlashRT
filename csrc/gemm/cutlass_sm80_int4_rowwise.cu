// ================================================================
// FlashRT — CUTLASS SM8x INT4 (s4 W4A4) rowwise GEMM family for
// Jetson Orin SM87 (QuaRot rotated-GEMM path).
//
// Same EVT structure as the INT8 rowwise kernels (per-row act scale ×
// per-row weight scale), with s4 operands and the m16n8k64 instruction.
// Precision contract: inputs are Hadamard-rotated per GEMM (activation
// side online FHT, weight side offline H·W), which flattens the
// Chameleon massive-activation channels so plain per-row symmetric
// int4 survives (measured worst L0-31 cosine 0.9914 vs 0.9722 for the
// production W8A8).
// Measured speed on Orin (M=1214): QKVO 0.34 ms (2.0x int8), gate/up
// 0.87 ms (1.9x), tile 128x128x128 w64x64x128 s5 Id4 = 120-144 TOPS.
//
// Variants:
//   cutlass_int4_rowwise_fp16out       (O-proj / down if rotated)
//   cutlass_int4_rowwise_fp16out_bias  (Q/K/V with fused per-N bias)
//   cutlass_int4_rowwise_bf16out       (FFN gate -> BF16 for silu_gated)
//   cutlass_int4_silu_gated_bf16out    (FFN up x SiLU(gate) -> BF16)
//
// A: [M, K/2] packed s4 row-major (elem 2i low nibble), 32-elem aligned.
// B: [N, K/2] packed s4 (ColumnMajor K-major), i.e. weight [N, K] rotated
//    + quantized per output row.
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
namespace cutlass_int4_sm8x {

using namespace cute;

using ElementA = cutlass::int4b_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::int4b_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementAccumulator = int32_t;
using ElementCompute = float;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;

using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
constexpr int NumStages = 5;
constexpr int EVTEpilogueStages = 1;

template <typename ElementOutput>
struct Chains {
    using OutputTileThreadMap =
        cutlass::epilogue::threadblock::OutputTileThreadLayout<
            ThreadblockShape, WarpShape, ElementOutput, AlignmentC,
            EVTEpilogueStages>;
    using AccFetch = cutlass::epilogue::threadblock::VisitorAccFetch;
    using ActScaleLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
        OutputTileThreadMap, float, Stride<_1, _0, _0>>;
    using WtScaleLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap, float, Stride<_0, _1, int32_t>>;
    using Mul = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using BiasLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap, cutlass::half_t, Stride<_0, _1, int32_t>>;
    using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
        OutputTileThreadMap, ElementOutput,
        cutlass::FloatRoundStyle::round_to_nearest,
        Stride<int64_t, _1, int64_t>>;

    using EVT_AccMulAct = cutlass::epilogue::threadblock::Sm80EVT<
        Mul, AccFetch, ActScaleLoad>;
    using EVT_MulBoth = cutlass::epilogue::threadblock::Sm80EVT<
        Mul, EVT_AccMulAct, WtScaleLoad>;
    using EVT_NoBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_MulBoth>;
    using EVT_AddBias = cutlass::epilogue::threadblock::Sm80EVT<
        AddBias, EVT_MulBoth, BiasLoad>;
    using EVT_WithBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_AddBias>;
};

// SiLU-gated functor (same as the INT8 silu_gated kernel).
template <class T>
struct GatedSiLUFunctor {
    __device__ T operator()(T up_val, T gate_val) const {
        return impl(up_val, gate_val,
                    typename cutlass::platform::is_floating_point<T>::type{});
    }
private:
    template <class S>
    __device__ S impl(S up, S gate, cutlass::platform::true_type) const {
        float g = float(gate);
        return S(float(up) * g / (1.0f + expf(-g)));
    }
    template <class Arr>
    __device__ Arr impl(Arr const& up, Arr const& gate,
                        cutlass::platform::false_type) const {
        Arr result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Arr::kElements; ++i) {
            float g = float(gate[i]);
            result[i] = typename Arr::Element(float(up[i]) * g / (1.0f + expf(-g)));
        }
        return result;
    }
};

template <typename EVT>
using KernelFor = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    // NOTE: ElementC/alignment used only via the EVT visitors.
    cutlass::half_t, LayoutC, AlignmentC,
    ElementAccumulator, ElementCompute, OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVT,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTEpilogueStages
>::GemmKernel;

using CF16 = Chains<cutlass::half_t>;
using CBF16 = Chains<cutlass::bfloat16_t>;

using GateLoad = cutlass::epilogue::threadblock::VisitorAuxLoad<
    CBF16::OutputTileThreadMap, cutlass::bfloat16_t,
    Stride<int64_t, _1, int64_t>>;
using MulGatedSiLU = cutlass::epilogue::threadblock::VisitorCompute<
    GatedSiLUFunctor, float, float,
    cutlass::FloatRoundStyle::round_to_nearest>;
using EVT_SiluGated = cutlass::epilogue::threadblock::Sm80EVT<
    MulGatedSiLU, CBF16::EVT_MulBoth, GateLoad>;
using EVT_SiluFinal = cutlass::epilogue::threadblock::Sm80EVT<
    CBF16::StoreD, EVT_SiluGated>;

using DevF16NoBias = cutlass::gemm::device::GemmUniversalAdapter<KernelFor<CF16::EVT_NoBias>>;
using DevF16Bias = cutlass::gemm::device::GemmUniversalAdapter<KernelFor<CF16::EVT_WithBias>>;
using DevBF16NoBias = cutlass::gemm::device::GemmUniversalAdapter<KernelFor<CBF16::EVT_NoBias>>;
using DevSilu = cutlass::gemm::device::GemmUniversalAdapter<KernelFor<EVT_SiluFinal>>;

template <typename Device, typename EVTArgs>
static int run_common(EVTArgs const& evt_args,
                      void const* A, void const* B,
                      int M, int N, int K, cudaStream_t stream,
                      const char* what) {
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename Device::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, problem_size, 1, evt_args,
        reinterpret_cast<ElementA const*>(A),
        reinterpret_cast<ElementB const*>(B),
        nullptr, nullptr,
        static_cast<int64_t>(M) * K, static_cast<int64_t>(N) * K, 0, 0,
        K, K, 0, 0);
    Device gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "[int4_rowwise:%s] can_implement failed: %d "
                     "(M=%d N=%d K=%d)\n", what, int(st), M, N, K);
        return int(st) | 0x10000;
    }
    size_t ws_sz = Device::get_workspace_size(args);
    static thread_local void* ws_ptr = nullptr;
    static thread_local size_t ws_cap = 0;
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) {
            ws_ptr = nullptr; ws_cap = 0; return -1;
        }
        ws_cap = ws_sz;
    }
    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) return int(st) | 0x20000;
    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (int(st) | 0x30000);
}

}  // namespace cutlass_int4_sm8x
}  // namespace gemm
}  // namespace flash_rt

using namespace flash_rt::gemm::cutlass_int4_sm8x;

extern "C" int cutlass_int4_rowwise_fp16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream) {
    typename CF16::EVT_NoBias::Arguments evt_args{
        {
            {{}, {reinterpret_cast<float const*>(act_scale), 1.0f, {}}, {}},
            {reinterpret_cast<float const*>(weight_scale), 1.0f,
             {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {reinterpret_cast<cutlass::half_t*>(D),
         {int64_t(N), _1{}, int64_t(M) * N}}
    };
    return run_common<DevF16NoBias>(evt_args, A, B, M, N, K, stream, "f16");
}

extern "C" int cutlass_int4_rowwise_fp16out_bias(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* bias, void* D, int M, int N, int K, cudaStream_t stream) {
    typename CF16::EVT_WithBias::Arguments evt_args{
        {
            {
                {{}, {reinterpret_cast<float const*>(act_scale), 1.0f, {}}, {}},
                {reinterpret_cast<float const*>(weight_scale), 1.0f,
                 {_0{}, _1{}, int32_t(N)}},
                {}
            },
            {reinterpret_cast<cutlass::half_t const*>(bias), cutlass::half_t(0),
             {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {reinterpret_cast<cutlass::half_t*>(D),
         {int64_t(N), _1{}, int64_t(M) * N}}
    };
    return run_common<DevF16Bias>(evt_args, A, B, M, N, K, stream, "f16b");
}

extern "C" int cutlass_int4_rowwise_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void* D, int M, int N, int K, cudaStream_t stream) {
    typename CBF16::EVT_NoBias::Arguments evt_args{
        {
            {{}, {reinterpret_cast<float const*>(act_scale), 1.0f, {}}, {}},
            {reinterpret_cast<float const*>(weight_scale), 1.0f,
             {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {reinterpret_cast<cutlass::bfloat16_t*>(D),
         {int64_t(N), _1{}, int64_t(M) * N}}
    };
    return run_common<DevBF16NoBias>(evt_args, A, B, M, N, K, stream, "bf16");
}

extern "C" int cutlass_int4_silu_gated_bf16out(
    void const* A, void const* B,
    void const* act_scale, void const* weight_scale,
    void const* gate_bf16, void* D, int M, int N, int K,
    cudaStream_t stream) {
    typename EVT_SiluFinal::Arguments evt_args{
        {
            {
                {{}, {reinterpret_cast<float const*>(act_scale), 1.0f, {}}, {}},
                {reinterpret_cast<float const*>(weight_scale), 1.0f,
                 {_0{}, _1{}, int32_t(N)}},
                {}
            },
            {const_cast<cutlass::bfloat16_t*>(
                 reinterpret_cast<cutlass::bfloat16_t const*>(gate_bf16)),
             cutlass::bfloat16_t{},
             {int64_t(N), _1{}, int64_t(M) * N}},
            {}
        },
        {reinterpret_cast<cutlass::bfloat16_t*>(D),
         {int64_t(N), _1{}, int64_t(M) * N}}
    };
    return run_common<DevSilu>(evt_args, A, B, M, N, K, stream, "silu");
}
