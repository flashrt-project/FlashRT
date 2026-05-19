// ================================================================
// FlashRT — CUTLASS SM80 INT8 GEMM + SiLU-Gated epilogue
//
// Fuses the Up-projection GEMM with the gated SiLU activation,
// reading the Gate buffer (from a prior Gate GEMM) via VisitorAuxLoad
// and computing hidden[i,j] = SiLU(gate[i,j]) * up_scaled[i,j]
// directly in the GEMM epilogue — avoiding a separate
// gate_geglu_merged kernel and the associated global-memory round-trip.
//
// Savings vs separate Gate+Up GEMM + gate_geglu_merged:
//   - Eliminates reading/writing the 2×H gate_merged buffer
//   - Eliminates one kernel launch per layer
//   - Per encoder layer: ~36 MB BW saved at 204 GB/s = ~0.18 ms
//   - Over 17 encoder + 180 decoder layers: ~5–11 ms total
// ================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
namespace cutlass_int8_silu_gated {

using namespace cute;

// ── Type aliases (same tile config as cutlass_sm80_int8_rowwise) ──
using ElementA = int8_t;
using LayoutA  = cutlass::layout::RowMajor;
using ElementB = int8_t;
using LayoutB  = cutlass::layout::ColumnMajor;
using ElementOutput = cutlass::bfloat16_t;
using LayoutC  = cutlass::layout::RowMajor;
using ElementAccumulator = int32_t;
using ElementCompute     = float;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 8;

using ArchTag        = cutlass::arch::Sm80;
using OperatorClass  = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64,  64,  64>;
using InstructionShape = cutlass::gemm::GemmShape<16,  8,   32>;
constexpr int NumStages        = 4;
constexpr int EVTEpilogueStages = 1;

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementOutput, AlignmentC, EVTEpilogueStages>;

// ── Standard scale visitors ──
using AccFetch     = cutlass::epilogue::threadblock::VisitorAccFetch;
using ActScaleLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float, Stride<_1, _0, _0>>;
using WtScaleLoad  = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, Stride<_0, _1, int32_t>>;
using MulActScale  = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using MulWtScale   = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

// ── Gate buffer: (M, N) BF16, loaded element-wise by VisitorAuxLoad ──
using GateLoad = cutlass::epilogue::threadblock::VisitorAuxLoad<
    OutputTileThreadMap, cutlass::bfloat16_t, Stride<int64_t, _1, int64_t>>;

// ── Custom binary functor template: out = up_scaled * SiLU(gate)
//    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
//    VisitorCompute<F, ...> requires F to be a class template
//    template<class T> struct F { T operator()(T, T); }.
//    In Sm80EVT, T can be either float (scalar) or
//    cutlass::Array<float, N> (batch). Use tag-dispatch to handle both.
template <class T>
struct GatedSiLUFunctor {
    __device__ T operator()(T up_val, T gate_val) const {
        return impl(up_val, gate_val,
                    typename cutlass::platform::is_floating_point<T>::type{});
    }

private:
    // Scalar path: T = float / double
    template <class S>
    __device__ S impl(S up, S gate,
                      cutlass::platform::true_type) const {
        float g = float(gate);
        return S(float(up) * g / (1.0f + expf(-g)));
    }

    // Array path: T = cutlass::Array<ElementT, N, Align>
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

// ── Multiply scaled up-accumulator by SiLU(gate) ──
using MulGatedSiLU = cutlass::epilogue::threadblock::VisitorCompute<
    GatedSiLUFunctor, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

// ── Output store ──
using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

// ── EVT tree ──
//   AccFetch ─────────────────────────────────────────────────────┐
//   ActScaleLoad ─── MulActScale ─── EVT_AccMulAct               │
//   WtScaleLoad  ─── MulWtScale  ─── EVT_MulBoth (up_scaled) ───┤
//   GateLoad      ── GatedSiLU   ─── EVT_SiluGated               │
//   StoreD ←──────────────────────────────────────────────────────┘
using EVT_AccMulAct = cutlass::epilogue::threadblock::Sm80EVT<
    MulActScale, AccFetch, ActScaleLoad>;
using EVT_MulBoth = cutlass::epilogue::threadblock::Sm80EVT<
    MulWtScale, EVT_AccMulAct, WtScaleLoad>;
using EVT_SiluGated = cutlass::epilogue::threadblock::Sm80EVT<
    MulGatedSiLU, EVT_MulBoth, GateLoad>;
using EVT_Final = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_SiluGated>;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator,
    ElementCompute,
    OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVT_Final,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EVTEpilogueStages
>::GemmKernel;

using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

static int run(
        void const* act_i8,    // (M, K) INT8 RowMajor — pre-quantized activation
        void const* up_w_i8,   // (N, K) INT8 ColMajor — up-projection weights
        void const* act_scale, // (M,)   float32 per-row activation scales
        void const* wt_scale,  // (N,)   float32 per-col weight scales
        void const* gate_buf,  // (M, N) BF16   gate values from prior Gate GEMM
        void*       D,         // (M, N) BF16   output: SiLU(gate) * up
        int M, int N, int K,
        cudaStream_t stream) {

    cutlass::gemm::GemmCoord problem(M, N, K);

    // EVT argument tree: children first, op args last (matches Sm80EVT convention).
    // Mirrors the existing cutlass_sm80_int8_rowwise.cu pattern exactly.
    typename EVT_Final::Arguments evt_args{
        // EVT_SiluGated::Arguments {EVT_MulBoth_args, GateLoad_args, GatedSiLU_op_args}
        {
            // EVT_MulBoth::Arguments {EVT_AccMulAct_args, WtScaleLoad_args, MulWtScale_op}
            {
                // EVT_AccMulAct::Arguments {AccFetch_args, ActScaleLoad_args, MulActScale_op}
                {
                    {},    // AccFetch — no arguments
                    {reinterpret_cast<float const*>(act_scale), 1.0f, {}},  // ActScaleLoad
                    {}     // MulActScale op — empty (simple multiply)
                },
                // WtScaleLoad::Arguments {ptr, fill, stride}
                {reinterpret_cast<float const*>(wt_scale), 1.0f,
                 {_0{}, _1{}, int32_t(N)}},
                {}         // MulWtScale op — empty
            },
            // GateLoad::Arguments {ptr, null_default, layout_stride}
            // CUTLASS VisitorAuxLoad takes non-const ptr internally (read-only semantics).
            {
                const_cast<cutlass::bfloat16_t*>(
                    reinterpret_cast<cutlass::bfloat16_t const*>(gate_buf)),
                cutlass::bfloat16_t{},
                {static_cast<int64_t>(N), _1{},
                 static_cast<int64_t>(M) * static_cast<int64_t>(N)}
            },
            {}             // GatedSiLU op — empty (template functor, stateless)
        },
        // StoreD::Arguments {ptr, layout_stride}
        {
            reinterpret_cast<ElementOutput*>(D),
            {static_cast<int64_t>(N), _1{},
             static_cast<int64_t>(M) * static_cast<int64_t>(N)}
        }
    };

    typename GemmDevice::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem, 1,
        evt_args,
        reinterpret_cast<ElementA const*>(act_i8),
        reinterpret_cast<ElementB const*>(up_w_i8),
        nullptr, nullptr,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(N) * K,
        0, 0, K, K, N, N);

    GemmDevice gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
            "[cutlass_int8_silu_gated] can_implement failed M=%d N=%d K=%d: %d\n",
            M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    static void* ws_ptr = nullptr;
    static size_t ws_cap = 0;
    size_t ws_sz = GemmDevice::get_workspace_size(args);
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) {
            ws_ptr = nullptr; ws_cap = 0; return -1;
        }
        ws_cap = ws_sz;
    }

    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
            "[cutlass_int8_silu_gated] init failed M=%d N=%d K=%d: %d\n",
            M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x20000;
    }

    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace cutlass_int8_silu_gated
}  // namespace gemm
}  // namespace flash_rt


// ── C extern for Python binding ──
extern "C" int cutlass_int8_silu_gated_bf16out(
        void const* act_i8,
        void const* up_w_i8,
        void const* act_scale,
        void const* wt_scale,
        void const* gate_buf,
        void* D,
        int M, int N, int K,
        cudaStream_t stream) {
    return flash_rt::gemm::cutlass_int8_silu_gated::run(
        act_i8, up_w_i8, act_scale, wt_scale, gate_buf, D,
        M, N, K, stream);
}
