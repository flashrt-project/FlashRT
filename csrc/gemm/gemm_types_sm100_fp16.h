// ================================================================
// FlashRT — CUTLASS FP16 GEMM Templates for SM100/SM110
// (Jetson AGX Thor, etc.)
//
// FP16 mirror of gemm_types_sm100.h.  Inputs are FP16, output is FP16,
// accumulate in FP32. Layout follows the same NT convention as the FP8
// templates: A row-major (M, K), B column-major (K, N), D row-major.
// Weight storage is therefore [N, K] row-major in memory — identical to
// PyTorch's nn.Linear weight layout, so no extra transpose is needed
// on the spec side (drop the T() the cuBLAS NN fallback used).
//
// Element alignment 8 (= 128 bits / FP16) — matches TMA load width.
// ================================================================
#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// Reuse cutlass_fp16 alias from the FP8 header (it's defined there too).
// We include neither header into the other; both must compile standalone.
#ifndef FLASHRT_CUTLASS_FP16_TYPES_DEFINED
#define FLASHRT_CUTLASS_FP16_TYPES_DEFINED
using cutlass_fp16_t = cutlass::half_t;
#endif

// GELU tanh-approximation activation matching gate_silu_mul_merged_kernel
// (csrc/kernels/activation.cu): x * sigmoid(1.59576... * x * (1 + 0.04471 x^2))
// Equivalent to x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
// Compute type is FP32 (matches the FP32 accumulator path).
template <typename T>
struct GeluTanhApprox {
  static const bool kIsHeavy = true;
  CUTLASS_HOST_DEVICE
  T operator()(T const& x) const {
    float xf = static_cast<float>(x);
    float k = 1.5957691216057308f;   // 2 * sqrt(2/pi)
    float c = 0.044715f;
    float z = k * xf * (1.0f + c * xf * xf);
    float sig = 1.0f / (1.0f + expf(-z));
    return static_cast<T>(xf * sig);
  }
};

template <typename T, int N>
struct GeluTanhApprox<cutlass::Array<T, N>> {
  static const bool kIsHeavy = true;
  CUTLASS_HOST_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& v) const {
    cutlass::Array<T, N> r;
    GeluTanhApprox<T> op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) r[i] = op(v[i]);
    return r;
  }
};

// ============================================================
//  PlainFp16: 256×128×64, Cluster 2×2×1
//  General FP16→FP16 GEMM (Identity epilogue)
// ============================================================
namespace sm100_fp16_plain {
using Tile = Shape<_256, _128, _64>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16_t, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_fp16_plain

// ============================================================
//  SqFp16: 256×256×128 — deeper K pipeline for square large GEMMs
//  Targets encoder G5 (QKV M=1024 N=2560 K=2048) and G6 (O M=1024 N=2048 K=2048)
// ============================================================
namespace sm100_fp16_sq {
using Tile = Shape<_256, _256, _128>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16_t, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_fp16_sq

// ============================================================
//  T1Fp16: 128×256×128, Cluster 2×1×1, TmaWarpSpecialized2Sm
//  Targets encoder G7 (Gate+Up M=1024 N=32768 K=2048) — wide-N shape
//  Mirrors the FP8 t1 tactic that beat Myelin s128x256.
// ============================================================
namespace sm100_fp16_t1 {
using Tile = Shape<_128, _256, _128>;
using Cluster = Shape<_2, _1, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16_t, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized2Sm, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_fp16_t1

// ============================================================
//  K64Fp16: 256×256×64, Cluster 2×2×1 — small K-tile, deep pipeline
//  R1.2 sweep winner for encoder G6 (O) and G7 (Gate+Up).
//  K-tile=64 admits more pipeline stages, hiding BW latency on
//  square/wide-N shapes (M=1024).
// ============================================================
namespace sm100_fp16_k64 {
using Tile = Shape<_256, _256, _64>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16_t, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_fp16_k64

// ============================================================
//  K64GeluFp16: same tile as k64 but with GELU-tanh epilogue.
//  Used by R3.1 split-G7: gate_buf = GELU(X @ W_gate).
// ============================================================
namespace sm100_fp16_k64_gelu {
using Tile = Shape<_256, _256, _64>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    GeluTanhApprox, cutlass_fp16_t, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_fp16_k64_gelu

// ============================================================
//  K64MulAuxFp16: same tile as k64 with binary `multiplies` epilogue
//  pulling an auxiliary tensor (gate_buf) via TMA.  R3.1 Phase 2:
//  D = acc * Aux, where Aux is GELU(X @ W_gate) loaded element-wise
//  alongside the up-GEMM accumulator.  Replaces the post-GEMM
//  mul_fp16 kernel by folding the multiply into the epilogue.
//
//  Uses Sm90 EVT visitor tree via LinCombDeEltAct (Sm100 callbacks
//  inherit Sm90 implementations for non-block-scale ops).
// ============================================================
namespace sm100_fp16_k64_mul_aux {
using Tile = Shape<_256, _256, _64>;
using Cluster = Shape<_2, _2, _1>;
// LinCombDeEltAct semantics: D = ActivationFn(beta*C + alpha*acc, Aux).
// With beta=0, alpha=1 and ActivationFn=multiplies: D = acc * Aux.
using Fusion = cutlass::epilogue::fusion::LinCombDeEltAct<
    cutlass::layout::RowMajor,
    cutlass::multiplies,
    cutlass_fp16_t,   // ElementOutput
    float,            // ElementCompute
    cutlass_fp16_t,   // ElementAux
    cutlass_fp16_t,   // ElementSource (unused; beta=0)
    float             // ElementScalar
>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_fp16_k64_mul_aux

// ============================================================
//  Sm2x1Fp16: 256×256×128, Cluster 2×1×1, explicit 2SM-Sm100
//  R1.2 sweep winner for encoder G8 (Down M=1024 N=2048 K=16384).
//  K-heavy shapes prefer the 2x1x1 cluster (less cross-SM
//  contention, more SMEM per stage) with the explicit 2SM
//  kernel/epilogue schedule.
// ============================================================
namespace sm100_fp16_2sm21 {
using Tile = Shape<_256, _256, _128>;
using Cluster = Shape<_2, _1, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16_t, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized2Sm, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_fp16_2sm21

// ============================================================
//  WideFp16: 256×128×128 — deeper K for wide-K shapes
//  Targets encoder G8 (Down M=1024 N=2048 K=16384)
// ============================================================
namespace sm100_fp16_wide {
using Tile = Shape<_256, _128, _128>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16_t, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp16_t, cutlass::layout::RowMajor, 8,
    cutlass_fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_fp16_wide
