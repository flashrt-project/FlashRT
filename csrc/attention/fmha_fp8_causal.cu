/**
 * fmha_fp8_causal.cu — FP8 (E4M3) causal FMHA for Chameleon-7B
 *
 * FP8 drop-in alternative to libfmha_fp16_causal.so for the Chameleon-7B
 * LLM path. Inputs Q/K/V are FP8 E4M3 (already quantized by
 * the caller via per-tensor static scales); softmax/PV accumulators stay
 * FP32; output O is written back as FP16 (so the rest of the residual
 * stream remains FP16 and the existing ``residual_add_rms_norm_fp8_fp16``
 * fused epilogue is unchanged).
 *
 * Built as a standalone .so loaded via dlopen at runtime. The CUTLASS
 * Sm100FmhaFwdMainloopTmaWarpspecialized has FP8-aware code paths
 * (kPRescale-compensated softmax, FP8 denorm-protection scaling) that
 * activate automatically when ``Element == cutlass::float_e4m3_t``.
 *
 * Exposes:
 *   extern "C" int fmha_fp8_causal(Q, K, V, O,
 *                                  B, SQ, SK, NQ, NKV, HD,
 *                                  scale_q, scale_k, scale_v, inv_scale_o,
 *                                  stream);
 *
 *   - Q, K, V : const void* — FP8 E4M3, [B, S, NH, HD]
 *   - O       : void*       — FP16,    [B, SQ, NQ, HD]
 *   - scale_q, scale_k, scale_v, inv_scale_o : float dequantization /
 *     output-quantization scales (forwarded to the mainloop Arguments).
 *     For our use case where O stays FP16, ``inv_scale_o = 1.0f``.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "device/fmha.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_load_tma_warpspecialized.hpp"
#include "collective/fmha_fusion.hpp"

using namespace cute;

// ── FP8 input, FP16 output (matches the Chameleon residual stream dtype) ──
using Element = cutlass::float_e4m3_t;
using ElementAccQK = float;
using ElementAccPV = float;
using ElementOut = cutlass::half_t;
// FP8 halves smem footprint vs FP16, so the original 256x128x128 tile
// shape from the FP16 paths still fits comfortably.
using TileShape = Shape<_256, _128, _128>;

using StrideQ   = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK   = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV   = StrideK;
using StrideO   = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

// CausalMask<true> + CausalIndividualTileScheduler: same as fmha_fp16_causal
// (Chameleon LLM self-attention is causal).
using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, ElementAccQK, ElementAccPV, TileShape,
    StrideQ, StrideK, StrideV, cutlass::fmha::collective::CausalMask<true>>;
using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
    ElementOut, ElementAccPV, typename Mainloop::TileShapePV, StrideO, StrideLSE>;
using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape, Mainloop, Epilogue,
    cutlass::fmha::kernel::CausalIndividualTileScheduler>;
using FmhaOp = cutlass::fmha::device::FMHA<Kernel>;

// One workspace + LSE buffer per process (lazy-allocated, grows as needed).
static void* g_ws = nullptr; static size_t g_ws_sz = 0;
static float* g_lse = nullptr; static size_t g_lse_sz = 0;

// ═══════════════════════════════════════════════════════════════════
// Causal FP8 FMHA: Q/K/V contiguous [B, S, NH, HD] in FP8, O FP16.
//
// scale_q/k/v: per-tensor dequantize scales for Q/K/V (e.g. amax/448).
// inv_scale_o: per-tensor output quantize scale (1.0 when O stays FP16).
// ═══════════════════════════════════════════════════════════════════
extern "C" int fmha_fp8_causal(
    const void* Q, const void* K, const void* V, void* O,
    int B, int SQ, int SK, int NQ, int NKV, int HD,
    float scale_q, float scale_k, float scale_v, float inv_scale_o,
    cudaStream_t stream)
{
    int H_Q = NQ/NKV, H_K = NKV, H = H_Q*H_K;
    int D = cutlass::round_up(HD, 8);
    auto ps = cute::make_tuple(SQ, SK, D, cute::make_tuple(cute::make_tuple(H_Q, H_K), B));

    // Contiguous layout: same as the FP16 path. The FP8 element size is
    // 1 byte, so the underlying memory layout halves vs FP16 — but the
    // logical strides (in elements, not bytes) stay identical to the
    // FP16 case at the API level.
    StrideQ sQ = make_stride(H*D, _1{}, make_stride(make_stride(D, H_Q*D), H*D*SQ));
    StrideO sO = sQ;
    StrideK sK = make_stride(H_K*D, _1{}, make_stride(make_stride(_0{}, D), H_K*D*SK));
    int SQ_r = ((SQ+127)/128)*128;
    StrideLSE sL = make_stride(_1{}, make_stride(make_stride(SQ_r, SQ_r*H_Q), SQ_r*H));

    size_t lsz = (size_t)B*H*SQ_r*sizeof(float);
    if (lsz > g_lse_sz) { if(g_lse) cudaFree(g_lse); cudaMalloc(&g_lse,lsz); g_lse_sz=lsz; }
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);

    // Build the FMHA Arguments with the FP8 scale fields populated.
    // - scale_softmax = 0 → mainloop defaults to 1/sqrt(D)
    // - scale_q/k/v   = caller-provided dequantize factors (ax/448)
    // - inv_scale_o   = 1.0 (output stays FP16; no output quant)
    typename FmhaOp::Arguments args{ps,
        {{(Element const*)Q, sQ, (Element const*)K, sK, (Element const*)V, sK},
         0.0f, scale_q, scale_k, scale_v, inv_scale_o},
        {(ElementOut*)O, sO, g_lse, sL}, {0, sm}};

    FmhaOp op;
    auto st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        printf("[FMHA fp8 causal] can_implement FAILED (%d) SQ=%d SK=%d NQ=%d HD=%d\n",
               (int)st, SQ, SK, NQ, HD);
        return -1;
    }
    size_t wsz = FmhaOp::get_workspace_size(args);
    if (wsz > g_ws_sz) { if(g_ws) cudaFree(g_ws); cudaMalloc(&g_ws,wsz); g_ws_sz=wsz; }
    if (op.initialize(args, g_ws, stream) != cutlass::Status::kSuccess) return -2;
    return (op.run(stream) == cutlass::Status::kSuccess) ? 0 : -3;
}
