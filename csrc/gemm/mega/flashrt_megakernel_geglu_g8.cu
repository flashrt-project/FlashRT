// ============================================================================
// FlashRT — encoder split-G7 megakernel (Stage B prototype).
//
// Stage B-v2: visitor-owned SMEM (mirrors Sm90AuxStore pattern).
//   - Phase 1 (gate) fusion = Sm90EVT<Sm100SmemAuxStore, Sm90EVT<Sm90Compute<GELU>,
//                                     Sm90LinearCombination>>
//     captures post-GELU to phase 1 visitor's OWN SharedStorage.
//   - Phase 2 (up) fusion = Sm90EVT<Sm90Compute<multiplies>, Sm90LinearCombination,
//                                   Sm100SmemAuxLoad>
//     loads aux from phase 2 visitor's OWN SharedStorage (NOT phase 1's).
//
// Cross-phase data sharing is NOT YET wired at this stage — phase 1 writes
// to phase1.epilogue.fusion.smem_aux; phase 2 reads from
// phase2.epilogue_2.fusion.smem_aux (different memory regions).  So numerical
// output will be wrong, but the test is: does the kernel RUN without
// illegal address?  That validates the visitor mechanic isolated from
// the cross-phase plumbing.
//
// Once kernel runs: add a kernel-body S2S copy from phase1's smem_aux to
// phase2's smem_aux at phase transition, OR overlay the two via kernel
// TensorStorage union.
// ============================================================================

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/integral_constant.hpp"

#include "sm100_smem_aux_visitor.hpp"
#include "flashrt_megakernel_geglu_g8_kernel.hpp"

#include <cuda_runtime.h>
#include <cstdio>

using namespace cute;
using fp16_t = cutlass::half_t;

namespace {

// Stage E3 best tile: (128, 128, 128) Cluster (2,2,1) with shared SMEM_A.
// Per-CTA (64, 64, 128) — TileK=128 (production sq's K).
// Low Thor regime: 1.06-1.07x faster than production back-to-back.
using Tile    = Shape<_128, _128, _128>;
using Cluster = Shape<_2, _2, _1>;

using FusionGate = flashrt::megakernel::fusion::LinCombEltActSmemAuxStore<
    cutlass::epilogue::thread::GELU_taylor, fp16_t, float, fp16_t>;

using FusionUp = flashrt::megakernel::fusion::LinCombDeEltActSmemAuxLoad<
    cutlass::multiplies, fp16_t, float, fp16_t>;

using CollectiveEpiGate = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionGate>::CollectiveOp;

using CollectiveEpiUp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionUp>::CollectiveOp;

using CollectiveMmaGate = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using CollectiveMmaUp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    fp16_t, cutlass::layout::RowMajor, 8,
    fp16_t, cutlass::layout::ColumnMajor, 8,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::FlashRtMegakernelGeGLUG8FusedGemm<
    Shape<int, int, int, int>,
    CollectiveMmaGate, CollectiveEpiGate,
    CollectiveMmaUp,   CollectiveEpiUp,
    void>;

using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // anonymous namespace

// Forward decls for the existing production kernels we chain in F-2.
// F-3+ will replace these external launches with a single fused kernel.
extern "C" int cutlass_fp16_2sm21(void* A, void* B, void* D,
                                    int M, int N, int K,
                                    float alpha, float beta,
                                    cudaStream_t stream);

// rms_norm_fp16 — declared as a regular (non-extern-C) C++ function in
// csrc/kernels/norm.cu.  Forward-decl with the same C++ linkage here.
void rms_norm_fp16(const __half* x, const __half* weight,
                    __half* out, int seq_len, int dim, float eps,
                    cudaStream_t stream);

extern "C" int cutlass_fp16_k64(void*, void*, void*,
                                  int, int, int, float, float,
                                  cudaStream_t);

// Bundle: rms_norm + QKV (k64) in one C entry.  Same pattern as F-2 bundle.
extern "C" int flashrt_rms_qkv_fp16(
    void* x,
    void* rms_weight,
    void* x_norm_scratch,
    void* qkv_weight,
    void* qkv_out,
    int M, int D, int N_qkv,
    float rms_eps,
    cudaStream_t stream)
{
    rms_norm_fp16(reinterpret_cast<const __half*>(x),
                   reinterpret_cast<const __half*>(rms_weight),
                   reinterpret_cast<__half*>(x_norm_scratch),
                   M, D, rms_eps, stream);
    return cutlass_fp16_k64(x_norm_scratch, qkv_weight, qkv_out,
                            M, N_qkv, D, 1.0f, 0.0f, stream);
}

// ============================================================================
// flashrt_megakernel_geglu_g8_fp16 — MK-2 fused entry.
//
// Computes  x_inout += GeGLU(X @ W_gate, X @ W_up) @ W_down
// in one C-callable.
//
// Stage progression:
//   F-1  : entry exists, signature == MK-1, internally identical (DONE).
//   F-2  : signature extended with W_down + x_inout; internally bundles
//          MK-1 mega + cutlass_fp16_2sm21(beta=1, D=x_inout).  Two kernel
//          launches but bundled API (THIS COMMIT).  Functionally equiv
//          to current Python path "mega + 2sm21_resid_fused".
//   F-3+ : drop the second launch; persistent-CTA fused kernel.
//
// Arguments (M=Se, H=hidden dim, D=embed dim):
//   X              [M, D]   fp16  (RowMajor, MK-1 input)
//   W_gate         [H, D]   fp16  (RowMajor → [N,K] CUTLASS-NT)
//   W_up           [H, D]   fp16
//   W_down         [D, H]   fp16  (RowMajor → [D, H] = G8's [N_d, K_g8])
//   hidden_scratch [M, H]   fp16  (scratch, gmem; F-3+ keeps it in SMEM)
//   x_inout        [M, D]   fp16  (residual in/out, beta=1 fold)
// ============================================================================
extern "C" int flashrt_megakernel_geglu_g8_fp16(
    void* X, void* W_gate, void* W_up,
    void* W_down,
    void* hidden_scratch, void* x_inout,
    int M, int H, int D,
    cudaStream_t stream)
{
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;

    // Phase 1+2: MK-1 mega — produces hidden = GELU(X @ W_gate) * (X @ W_up).
    // Re-uses the same kernel class instance; problem shape (M, H, D).
    auto sA = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideA{}, {M, D, 1});
    auto sB = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideB{}, {H, D, 1});
    auto sD = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideD{}, {M, H, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, H, D, 1},
        {(ElementA*)X, sA, (ElementB*)W_gate, sB},
        {
            { 1.0f, 0.0f, nullptr, nullptr, {}, {}, {} },
            nullptr, {},
            (ElementD*)hidden_scratch, sD  // phase-1 gate gmem (unused values)
        },
        {(ElementA*)X, sA, (ElementB*)W_up, sB},
        {
            { 1.0f, 0.0f, nullptr, nullptr, {}, {}, {} },
            nullptr, {},
            (ElementD*)x_inout, sD  // phase-2 hidden gmem — reuses x_inout
                                    // as throw-away buffer at this stage;
                                    // overwritten by the G8 launch below.
                                    // NOTE: F-3 will keep hidden in SMEM
                                    // and never touch this path.
        }
    };

    // For F-2 we still need hidden in a real gmem buffer for the 2sm21
    // launch.  Override phase-2 epilogue's output ptr to point at the
    // caller's hidden_scratch (same buffer phase-1 uses; phase-2 writes
    // last so wins).  This wastes a 24 MB roundtrip — eliminated in F-3.
    args.epilogue_2.dD = sD;
    args.epilogue_2.ptr_D = (ElementD*)hidden_scratch;

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_geglu_g8] mega phase: cannot implement M=%d H=%d D=%d\n", M, H, D);
        return -1;
    }
    if (gemm.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_geglu_g8] mega phase: init failed\n");
        return -2;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_geglu_g8] mega phase: run failed\n");
        return -3;
    }

    // Phase 3 (F-2 placeholder): launch the production 2sm21 G8 with
    // resid fuse (alpha=1, beta=1, D = x_inout).  F-3+ folds this into
    // the same kernel via persistent CTA + h_chunk loop.
    return cutlass_fp16_2sm21(hidden_scratch, W_down, x_inout,
                              M, D, H, 1.0f, 1.0f, stream);
}

// ============================================================================
// MK-2 F-3: bundled FFN-block entry (rms_norm + mega + G8 + resid)
//
// Collapses four production launches into a single C function.  Pure
// bundling, no kernel-level fusion yet — but F-2 measurements showed
// each removed C-boundary is worth ~0.5 ms in hot regime, so 4→1 may
// recover ~1.5 ms on top of F-2.
//
// Semantics:
//   x_norm = rms_norm(x_resid, rms_weight, eps)
//   hidden = GELU(x_norm @ W_gate) * (x_norm @ W_up)
//   x_resid += hidden @ W_down                     (in-place residual)
//
// Buffers (caller-allocated):
//   x_resid       [M, D] fp16  IN+OUT
//   rms_weight    [D]    fp16  (typically all-ones for noweight rmsnorm)
//   x_norm        [M, D] fp16  scratch
//   W_gate        [H, D] fp16
//   W_up          [H, D] fp16
//   W_down        [D, H] fp16
//   gate_scratch  [M, H] fp16  scratch (phase-1 epilogue throwaway)
//   hidden_scratch[M, H] fp16  scratch (phase-2 epilogue output)
// ============================================================================
extern "C" int flashrt_encoder_ffn_block_fp16(
    void* x_resid,
    void* rms_weight,
    void* x_norm,
    void* W_gate, void* W_up, void* W_down,
    void* gate_scratch, void* hidden_scratch,
    int M, int H, int D,
    float rms_eps,
    cudaStream_t stream)
{
    // 1. rms_norm: x_resid → x_norm
    rms_norm_fp16(reinterpret_cast<const __half*>(x_resid),
                   reinterpret_cast<const __half*>(rms_weight),
                   reinterpret_cast<__half*>(x_norm),
                   M, D, rms_eps, stream);

    // 2. mega GeGLU: x_norm @ W_gate / W_up → hidden_scratch
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;
    auto sA = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideA{}, {M, D, 1});
    auto sB = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideB{}, {H, D, 1});
    auto sD = cutlass::make_cute_packed_stride(typename GemmOp::GemmKernel::StrideD{}, {M, H, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, H, D, 1},
        {(ElementA*)x_norm, sA, (ElementB*)W_gate, sB},
        {
            { 1.0f, 0.0f, nullptr, nullptr, {}, {}, {} },
            nullptr, {},
            (ElementD*)gate_scratch, sD
        },
        {(ElementA*)x_norm, sA, (ElementB*)W_up, sB},
        {
            { 1.0f, 0.0f, nullptr, nullptr, {}, {}, {} },
            nullptr, {},
            (ElementD*)hidden_scratch, sD
        }
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_encoder_ffn_block] mega: cannot implement M=%d H=%d D=%d\n", M, H, D);
        return -1;
    }
    if (gemm.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_encoder_ffn_block] mega: init failed\n");
        return -2;
    }
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_encoder_ffn_block] mega: run failed\n");
        return -3;
    }

    // 3. G8 with resid fuse: x_resid += hidden @ W_down (beta=1, D=x_resid)
    return cutlass_fp16_2sm21(hidden_scratch, W_down, x_resid,
                              M, D, H, 1.0f, 1.0f, stream);
}
