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
