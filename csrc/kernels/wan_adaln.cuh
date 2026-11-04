// Wan2.2 / LingBot-VA AdaLN kernels (arch-neutral: plain bf16/fp8 arithmetic and
// shared-memory reductions, no arch-specific intrinsics).
//
// Shared by csrc/kernels/wan_adaln_fp8.cu; compiled into flash_rt_kernels and
// bound in csrc/bindings.cpp. No separate .so (mirrors the lingbot_* and qwen36
// model kernels in this directory).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace wan_adaln {

// Everything the Wan/LingBot-VA denoise block needs between two FP8 GEMMs:
//
//   y   = AdaLN(LayerNorm(x))            -- fp32 statistics, no affine
//   out = quant_fp8(round_bf16(y))       -- static per-tensor scale
//
// `ss_table` is the block's scale_shift_table ([6, D], fp32) and `tproj` the
// per-token conditioning ([B, S, 6, D], bf16). Only the two modulation rows the
// caller names (`scale_idx`, `shift_idx`) are read, so one kernel serves both
// the self-attention block (rows 0/1) and the FFN block (rows 3/4) of the same
// transformer layer.
//
// Rounding through bf16 between the modulation and the FP8 quantize is what
// keeps this bit-aligned with the eager reference, which casts the modulated
// activation back to bf16 before handing it to the GEMM (same argument as
// lingbot_norm_fp8.cu).
void wan_adaln_ln_fp8(
    uintptr_t x, uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t stream);

// Same, with the residual folded in: residual += x, then AdaLN on the sum.
// `gate_vec` (optional, may be 0) is the per-token gate applied to `x` on the
// way in, i.e. residual += x * gate_vec -- the `hidden + attn_out * gate` step
// of the block.
void wan_adaln_residual_ln_fp8(
    uintptr_t residual, uintptr_t x, uintptr_t gate_vec_fp32,
    uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t stream);

// bf16-output variant of the fused residual form, for sites that feed a
// non-FP8 consumer (the action head).
void wan_adaln_residual_ln_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t gate_vec_fp32,
    uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_bf16,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t stream);


// `residual += x * gate_vec` with the bf16 round the eager block applies
// (`hidden = (hidden.float() + attn_out * gate).type_as(hidden)`). gate_vec is
// the block's per-token gate row [B,S,D] in **fp32** -- the eager block forms it
// from an fp32 scale_shift_table plus the fp32 timestep projection and never
// rounds it, and the FFN output can reach ~1e2, so bf16-rounding the gate costs
// whole percent of the residual. Pass 0 for a plain residual add.
// gate formed inline from (ss_row[D], tp[B*S, D]) -- bit-exact with
// wan_gated_residual_bf16 fed the materialised ss_row + tp sum.
// AdaLN + LayerNorm writing FP8 with 128-wide per-token scales directly,
// replacing this kernel followed by fp8_per_token_block128_quant_bf16. The
// block scale is a max, whose reduction order is free, so this is bit-exact
// with the pair it replaces; out_scale is [B*S, D/128].
// tp_bcast = 1: every token's tproj rows are equal (one timestep for the whole
// sequence); token 0's rows are read for all of them.
void wan_adaln_residual_ln_fp8_block128(
    uintptr_t residual, uintptr_t x, uintptr_t gate_vec,
    uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_fp8, uintptr_t out_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t stream,
    int tp_bcast = 0);

void wan_gated_residual_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t gate_vec_fp32,
    int B, int S, int D, uintptr_t stream);

// Same two kernels with the producer's bias add folded in. The callers ran
// `add_bias_bf16(out, bias, S, D)` on the GEMM output immediately before these,
// and at the three per-block sites the biased buffer has no other reader, so
// the extra launch and its read+write of the whole [S,D] buffer can go.
//
//     add_bias_bf16:  x = bf16(fp32(x) + fp32(bias[c]))
//
// is reproduced exactly -- one fp32 add, one bf16 round, applied to the same two
// operands, before the gate multiply and the residual add run unchanged. Pass 0
// for bias_bf16 to get the previous behaviour.
void wan_gated_residual_sstp_bias_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t bias_bf16, uintptr_t ss_row,
    uintptr_t tp, int B, int S, int D, uintptr_t stream, int tp_bcast = 0);

void wan_gated_residual_bias_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t bias_bf16, uintptr_t gate_vec_fp32,
    int B, int S, int D, uintptr_t stream);

// wan_gated_residual_sstp_bias_bf16 followed by
// wan_adaln_residual_ln_fp8_block128 (x = 0) on the same residual, in one
// launch: residual += bf16(x + xbias) * (gate_ss_row + gate_tp), then the AdaLN
// of the updated residual emitted as FP8 with 128-wide per-token scales.
// Bit-identical to the pair.
void wan_gated_residual_sstp_adaln_fp8_block128(
    uintptr_t residual, uintptr_t x, uintptr_t xbias, uintptr_t gate_ss_row,
    uintptr_t gate_tp, uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_fp8, uintptr_t out_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t stream,
    int tp_bcast = 0);

}  // namespace wan_adaln
}  // namespace flash_rt
