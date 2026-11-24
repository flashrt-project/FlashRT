#pragma once
#include "attention/attention_d72_bf16.h"
#include "attention/attention_simd_bf16.h"
#include "attention/gqa8_causal_prefill_bf16.h"
#include "attention/gqa8_causal_prefill_mma_bf16.h"
#include "attention/gqa8_d256_causal_gated_prefill_mma_bf16.h"
#include "attention/gqa_merge_fp32_bf16.h"
#include "attention/gqa_online_bf16.h"
#include "attention/gqa_split_bf16.h"
#include "conv/patch_pack_bf16.h"
#include "conv/patch_probe_bf16.h"
#include "flint/flint_decode_resident_bf16.h"
#include "flint/flint_moe_rows_bf16.h"
#include "flint/flint_prefill_rows_bf16.h"
#include "gemm/bf16_tensor_ops.h"
#include "gemm/q8_0_rows.h"
#include "gemm/w4a16_affine_qmv.h"
#include "gemm/w4a16_affine_tensor_ops.h"
#include "inkling2/inkling2_decode_resident_bf16.h"
#include "inkling2/inkling2_rows_bf16.h"
#include "inkling2/inkling2_rows_gemm.h"
#include "inkling2/inkling2_mtp_bf16.h"
#include "inkling2/inkling2_verify_resident_bf16.h"
#include "kernels/ada_rms_norm_bf16.h"
#include "kernels/bias_bf16.h"
#include "kernels/gelu_bf16.h"
#include "kernels/gelu_mul_bf16.h"
#include "kernels/gumbel_noise_f32.h"
#include "kernels/layer_norm_bf16.h"
#include "kernels/position_add_bf16.h"
#include "kernels/residual_bf16.h"
#include "kernels/rms_norm_bf16.h"
#include "kernels/rope_bf16.h"
#include "qwen35moe/qwen35moe_decode_resident_bf16.h"
#include "qwen35moe/qwen35moe_prefill_rows_bf16.h"
#include "qwen35moe/qwen35moe_router_rows_bf16.h"
#ifdef FLASHRT_APPLE_INT8_EXPERIMENT
#include "gemm/int8_tensor_ops.h"
#include "quantize/bf16_int8_rowwise.h"
#include "quantize/amax_channel_bf16.h"
#include "quantize/gelu_mul_int8_rowwise.h"
#endif
#include <stdexcept>
#include <string_view>

namespace flashrt::apple::shaders {
struct KernelSource {
    const char* name;  // Dispatch ABI name; old tr_* aliases remain BF16.
    const char* entry;
    std::string (*build_source)();
};

// Explicit dispatch name -> operation/precision source -> Metal entry.
// No name-prefix inference or fallback to a different precision. Each source
// contains only its operation family; this table contains no shader bodies.
inline constexpr KernelSource kernel_registry[] = {
#ifdef FLASHRT_APPLE_INT8_EXPERIMENT
    {"int8_gemm", "gemm_int8_bf16", [] { return gemm_int8_source(); }},
    {"int8_gemm_small", "gemm_int8_bf16", [] { return gemm_int8_source(16,128,4); }},
    {"int8_gemm_bias", "gemm_int8_bf16", [] { return gemm_int8_source(32,64,2,true); }},
    {"int8_gemm_small_bias", "gemm_int8_bf16", [] { return gemm_int8_source(16,128,4,true); }},
    {"int8_gemm_64x64", "gemm_int8_bf16", [] { return gemm_int8_source(64,64,2); }},
    {"int8_gemm_64x64_bias", "gemm_int8_bf16", [] { return gemm_int8_source(64,64,2,true); }},
    {"int8_gemm_64x128", "gemm_int8_bf16", [] { return gemm_int8_source(64,128,4); }},
    {"int8_gemm_64x128_bias", "gemm_int8_bf16", [] { return gemm_int8_source(64,128,4,true); }},
    {"int8_pack", "quantize_bf16_int8_rowwise", [] { return quantize_bf16_int8_rowwise_source(); }},
    {"int8_pack_small", "quantize_bf16_int8_rowwise", [] { return quantize_bf16_int8_rowwise_source(true); }},
    {"int8_pack_smooth", "quantize_bf16_int8_rowwise", [] { return quantize_bf16_int8_rowwise_source(false,true); }},
    {"int8_pack_small_smooth", "quantize_bf16_int8_rowwise", [] { return quantize_bf16_int8_rowwise_source(true,true); }},
    {"int8_observe_channels", "amax_channel_bf16", amax_channel_bf16_source},
    {"int8_gelu_mul_smooth", "gelu_mul_int8_rowwise", [] { return gelu_mul_int8_rowwise_source(); }},
#endif
    {"gemm_bf16", "gemm_bf16", [] { return gemm_bf16_source(); }},
    {"gemm_bias_bf16", "gemm_bias_bf16", [] { return gemm_bf16_source(); }},
    {"gemm_patch_bf16", "gemm_patch_bf16", [] { return gemm_bf16_source(); }},
    {"gemm_strided_bf16", "gemm_strided_bf16", [] { return gemm_bf16_source(); }},
    {"gemm_small_bf16", "gemm_bf16", [] { return gemm_bf16_source(16); }},
    {"gemm_small_bias_bf16", "gemm_bias_bf16", [] { return gemm_bf16_source(16); }},
    {"gemm_small_strided_bf16", "gemm_strided_bf16", [] { return gemm_bf16_source(16); }},
    {"attention_d72_bf16", "attention_d72_bf16", attention_d72_bf16_source},
    {"attention_bf16", "attention_bf16", attention_simd_bf16_source},
    {"tr_gqa", "tr_gqa", [] { return gqa_online_bf16_source(); }},
    {"tr_gqa_masked", "tr_gqa_masked", [] { return gqa_online_bf16_source(true); }},
    {"tr_gqa_split", "tr_gqa_split", [] { return gqa_split_bf16_source(); }},
    {"tr_gqa_split_masked", "tr_gqa_split_masked", [] { return gqa_split_bf16_source(true); }},
    {"tr_gqa_merge", "tr_gqa_merge", gqa_merge_fp32_bf16_source},
    {"tr_patch_pack", "tr_patch_pack", patch_pack_bf16_source},
    {"tr_patch_probe", "tr_patch_probe", patch_probe_bf16_source},
    {"tr_patch_position", "tr_patch_position", position_add_bf16_source},
    {"norm_bf16", "norm_bf16", layer_norm_bf16_source},
    {"tr_rms", "tr_rms", rms_norm_bf16_source},
    {"tr_adarms", "tr_adarms", ada_rms_norm_bf16_source},
    {"bias_bf16", "bias_bf16", bias_bf16_source},
    {"gelu_bf16", "gelu_bf16", gelu_bf16_source},
    {"tr_gelu_mul", "tr_gelu_mul", gelu_mul_bf16_source},
    {"residual_bf16", "residual_bf16", residual_bf16_source},
    {"tr_residual", "tr_residual", residual_bf16_source},
    {"tr_rope", "tr_rope", [] { return rope_bf16_source(); }},
    {"tr_rope_offsets", "tr_rope_offsets", [] { return rope_bf16_source(true); }},
    // Affine int4 (group 64) weights x BF16 activations, TensorOps: plain, residual add, SwiGLU of two
    // matrices, and expert-grouped variants (rows grouped by expert, stacked weights).
    {"gemm_w4a16_affine", "gemm_w4a16_affine", [] { return gemm_w4a16_affine_source(); }},
    {"gemm_w4a16_affine_residual", "gemm_w4a16_affine",
     [] { W4A16Options o; o.epilogue = 1; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_swiglu", "gemm_w4a16_affine",
     [] { W4A16Options o; o.epilogue = 2; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_grouped", "gemm_w4a16_affine",
     [] { W4A16Options o; o.grouped = true; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_grouped_swiglu", "gemm_w4a16_affine",
     [] { W4A16Options o; o.grouped = true; o.epilogue = 2; return gemm_w4a16_affine_source(o); }},
    // The same products on the single-chain path for every row count (expert blocks of any size).
    {"gemm_w4a16_affine_direct", "gemm_w4a16_affine",
     [] { W4A16Options o; o.direct = true; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_direct_residual", "gemm_w4a16_affine",
     [] { W4A16Options o; o.direct = true; o.epilogue = 1; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_direct_swiglu", "gemm_w4a16_affine",
     [] { W4A16Options o; o.direct = true; o.epilogue = 2; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_grouped_direct", "gemm_w4a16_affine",
     [] { W4A16Options o; o.grouped = true; o.direct = true; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_grouped_direct_swiglu", "gemm_w4a16_affine",
     [] { W4A16Options o; o.grouped = true; o.direct = true; o.epilogue = 2; return gemm_w4a16_affine_source(o); }},
    // Split-K partitions (M <= 512 rows) keep the two-SIMD-group threadgroup qualified against MLX.
    {"gemm_w4a16_affine_splitk", "gemm_w4a16_affine",
     [] { W4A16Options o; o.splitk = true; o.simdgroups = 2; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_splitk_residual", "gemm_w4a16_affine",
     [] { W4A16Options o; o.splitk = true; o.simdgroups = 2; o.epilogue = 1; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_splitk_swiglu", "gemm_w4a16_affine",
     [] { W4A16Options o; o.splitk = true; o.simdgroups = 2; o.epilogue = 2; return gemm_w4a16_affine_source(o); }},
    // Affine int4 (group 64) matrix x BF16 row vectors (MLX qmv_fast arithmetic): one vector, and the
    // chunk-row variants used where MLX's matmul takes its vector path (plain, residual, SwiGLU, grouped).
    {"qmv_w4a16_affine", "qmv_w4a16_affine", [] { return qmv_w4a16_affine_source(); }},
    {"qmv_w4a16_affine_rows", "qmv_w4a16_affine",
     [] { QmvOptions o; o.rows = true; return qmv_w4a16_affine_source(o); }},
    {"qmv_w4a16_affine_rows_residual", "qmv_w4a16_affine",
     [] { QmvOptions o; o.rows = true; o.epilogue = 1; return qmv_w4a16_affine_source(o); }},
    {"qmv_w4a16_affine_rows_swiglu", "qmv_w4a16_affine",
     [] { QmvOptions o; o.rows = true; o.epilogue = 2; return qmv_w4a16_affine_source(o); }},
    {"qmv_w4a16_affine_wide", "qmv_w4a16_affine",
     [] { QmvOptions o; o.rows = true; o.wide = true; return qmv_w4a16_affine_source(o); }},
    {"qmv_w4a16_affine_wide_residual", "qmv_w4a16_affine",
     [] { QmvOptions o; o.rows = true; o.wide = true; o.epilogue = 1; return qmv_w4a16_affine_source(o); }},
    {"qmv_w4a16_affine_wide_swiglu", "qmv_w4a16_affine",
     [] { QmvOptions o; o.rows = true; o.wide = true; o.epilogue = 2; return qmv_w4a16_affine_source(o); }},
    {"qmv_w4a16_affine_rows_grouped", "qmv_w4a16_affine",
     [] { QmvOptions o; o.rows = true; o.grouped = true; return qmv_w4a16_affine_source(o); }},
    {"qmv_w4a16_affine_rows_grouped_swiglu", "qmv_w4a16_affine",
     [] { QmvOptions o; o.rows = true; o.grouped = true; o.epilogue = 2; return qmv_w4a16_affine_source(o); }},
    // GQA-8 causal / sliding-window prefill attention with sinks (MLX fused-attention arithmetic):
    // SIMD-group MMAs and TensorOps.
    {"gqa8_causal_prefill_bf16", "gqa8_causal_prefill_bf16", [] { return gqa8_causal_prefill_bf16_source(); }},
    {"gqa8_causal_prefill_mma_bf16", "gqa8_causal_prefill_mma_bf16",
     [] { return gqa8_causal_prefill_mma_bf16_source(); }},
    // Counter-based Gumbel noise rows for sampling.
    {"gumbel_noise_f32", "gumbel_noise_f32", gumbel_noise_f32_source},
    // Flint chunked prefill: embedding/norm/Q-K rows, sliding-ring copies, and mixture-of-experts
    // routing, placement and combination.
    {"flint_embed_affine8_rows_bf16", "flint_embed_affine8_rows_bf16", flint_prefill_rows_bf16_source},
    {"flint_rms_rows_bf16", "flint_rms_rows_bf16", flint_prefill_rows_bf16_source},
    {"flint_qk_norm_rope_kv_rows_bf16", "flint_qk_norm_rope_kv_rows_bf16", flint_prefill_rows_bf16_source},
    {"flint_kv_ring_copy_bf16", "flint_kv_ring_copy_bf16", flint_prefill_rows_bf16_source},
    {"flint_router_topk_rows_bf16", "flint_router_topk_rows_bf16", flint_moe_rows_bf16_source},
    {"flint_expert_count", "flint_expert_count", flint_moe_rows_bf16_source},
    {"flint_expert_scan", "flint_expert_scan", flint_moe_rows_bf16_source},
    {"flint_expert_place_bf16", "flint_expert_place_bf16", flint_moe_rows_bf16_source},
    {"flint_expert_combine_rows_bf16", "flint_expert_combine_rows_bf16", flint_moe_rows_bf16_source},
    // Flint resident decode step (int4 affine g64 weights, BF16 activations/KV): greedy, fused
    // top-k/top-p sampling, and the reference attention schedule (with the residual-stream trace) used
    // for bit-exact qualification.
    {"flint_decode_resident_bf16", "flint_decode_resident_bf16",
     [] { return flint_decode_resident_bf16_source(); }},
    {"flint_decode_resident_sampled_bf16", "flint_decode_resident_bf16",
     [] { FlintDecodeOptions o; o.sampled = true; return flint_decode_resident_bf16_source(o); }},
    {"flint_decode_resident_reference_bf16", "flint_decode_resident_bf16",
     [] { FlintDecodeOptions o; o.exact_attention = true; o.trace = true; return flint_decode_resident_bf16_source(o); }},
    // Verification step over 2..4 consecutive positions (the last token plus draft tokens), each row's
    // arithmetic that of the greedy resident step.
    {"flint_decode_verify2_bf16", "flint_decode_verify_bf16",
     [] { FlintDecodeOptions o; o.verify_rows = 2; return flint_decode_resident_bf16_source(o); }},
    {"flint_decode_verify3_bf16", "flint_decode_verify_bf16",
     [] { FlintDecodeOptions o; o.verify_rows = 3; return flint_decode_resident_bf16_source(o); }},
    {"flint_decode_verify4_bf16", "flint_decode_verify_bf16",
     [] { FlintDecodeOptions o; o.verify_rows = 4; return flint_decode_resident_bf16_source(o); }},
    // qwen3_5_moe resident decode step (MLX affine int4 g64 weights, BF16 activations/KV, FP32 DeltaNet
    // state): greedy, fused top-k/top-p sampling, and greedy with the residual-stream trace used for
    // bit-exact qualification (the three share one arithmetic).
    {"qwen35moe_decode_resident_bf16", "qwen35moe_decode_resident_bf16",
     [] { return qwen35moe_decode_resident_bf16_source(); }},
    {"qwen35moe_decode_resident_sampled_bf16", "qwen35moe_decode_resident_bf16",
     [] { Qwen35MoeDecodeOptions o; o.sampled = true; return qwen35moe_decode_resident_bf16_source(o); }},
    {"qwen35moe_decode_resident_reference_bf16", "qwen35moe_decode_resident_bf16",
     [] { Qwen35MoeDecodeOptions o; o.trace = true; return qwen35moe_decode_resident_bf16_source(o); }},
    // qwen3_5_moe chunked-prefill row kernels and the causal gated attention for 256-wide heads.
    {"qwen35moe_embed_affine4_rows_bf16", "qwen35moe_embed_affine4_rows_bf16", qwen35moe_prefill_rows_bf16_source},
    {"qwen35moe_deltanet_inputs_rows_bf16", "qwen35moe_deltanet_inputs_rows_bf16", qwen35moe_prefill_rows_bf16_source},
    {"qwen35moe_deltanet_recurrence_bf16", "qwen35moe_deltanet_recurrence_bf16", qwen35moe_prefill_rows_bf16_source},
    {"qwen35moe_deltanet_ring_bf16", "qwen35moe_deltanet_ring_bf16", qwen35moe_prefill_rows_bf16_source},
    {"qwen35moe_deltanet_gate_rows_bf16", "qwen35moe_deltanet_gate_rows_bf16", qwen35moe_prefill_rows_bf16_source},
    {"qwen35moe_qk_norm_rope_kv_rows_bf16", "qwen35moe_qk_norm_rope_kv_rows_bf16", qwen35moe_prefill_rows_bf16_source},
    {"qwen35moe_router_top8_dense_rows_bf16", "qwen35moe_router_top8_dense_rows_bf16",
     qwen35moe_router_rows_bf16_source},
    {"qwen35moe_expert_count", "qwen35moe_expert_count", qwen35moe_prefill_rows_bf16_source},
    {"qwen35moe_expert_place_bf16", "qwen35moe_expert_place_bf16", qwen35moe_prefill_rows_bf16_source},
    {"qwen35moe_expert_combine_rows_bf16", "qwen35moe_expert_combine_rows_bf16", qwen35moe_prefill_rows_bf16_source},
    {"gqa8_d256_causal_gated_prefill_mma_bf16", "gqa8_d256_causal_gated_prefill_mma_bf16",
     gqa8_d256_causal_gated_prefill_mma_bf16_source},
    // inkling2_loop (Inkling 2 Mini) resident decode step (BF16 attention, NVFP4 / MXFP8 experts): greedy and
    // greedy with the per-visit trace used for bit-exact qualification (one arithmetic).
    {"inkling2_decode_resident_bf16", "inkling2_decode_resident_bf16",
     [] { return inkling2_decode_resident_bf16_source(); }},
    {"inkling2_decode_resident_reference_bf16", "inkling2_decode_resident_bf16",
     [] { Inkling2DecodeOptions o; o.trace = true; return inkling2_decode_resident_bf16_source(o); }},
    {"inkling2_decode_resident_sampled_bf16", "inkling2_decode_resident_bf16",
     [] { Inkling2DecodeOptions o; o.sampled = true; return inkling2_decode_resident_bf16_source(o); }},
    // the same with the attention projections and LM head from Q8_0 copies (opt-in; not the checkpoint's numerics)
    {"inkling2_decode_resident_q8_bf16", "inkling2_decode_resident_bf16",
     [] { Inkling2DecodeOptions o; o.q8 = true; return inkling2_decode_resident_bf16_source(o); }},
    {"inkling2_decode_resident_q8_reference_bf16", "inkling2_decode_resident_bf16",
     [] { Inkling2DecodeOptions o; o.q8 = true; o.trace = true; return inkling2_decode_resident_bf16_source(o); }},
    {"inkling2_decode_resident_q8_sampled_bf16", "inkling2_decode_resident_bf16",
     [] { Inkling2DecodeOptions o; o.q8 = true; o.sampled = true; return inkling2_decode_resident_bf16_source(o); }},
    // inkling2_loop resident verification step over 2..8 consecutive positions of one sequence (staged like the
    // verify row program): exact (each row's arithmetic that of the greedy resident step) and fast (own summation
    // order in the products). Reference: with the trace.
#define FRT_INK_VERIFY(R) \
    {"inkling2_verify_resident_r" #R "_bf16", "inkling2_verify_resident_bf16", \
     [] { Inkling2VerifyOptions o; o.rows = R; return inkling2_verify_resident_bf16_source(o); }}, \
    {"inkling2_verify_resident_r" #R "_reference_bf16", "inkling2_verify_resident_bf16", \
     [] { Inkling2VerifyOptions o; o.rows = R; o.trace = true; return inkling2_verify_resident_bf16_source(o); }}, \
    {"inkling2_verify_fast_r" #R "_bf16", "inkling2_verify_resident_bf16", \
     [] { Inkling2VerifyOptions o; o.rows = R; o.fast = true; return inkling2_verify_resident_bf16_source(o); }}, \
    {"inkling2_verify_fast_r" #R "_reference_bf16", "inkling2_verify_resident_bf16", \
     [] { Inkling2VerifyOptions o; o.rows = R; o.fast = true; o.trace = true; \
          return inkling2_verify_resident_bf16_source(o); }}, \
    {"inkling2_verify_fast_q8_r" #R "_bf16", "inkling2_verify_resident_bf16", \
     [] { Inkling2VerifyOptions o; o.rows = R; o.fast = true; o.q8 = true; return inkling2_verify_resident_bf16_source(o); }}, \
    {"inkling2_verify_fast_q8_r" #R "_reference_bf16", "inkling2_verify_resident_bf16", \
     [] { Inkling2VerifyOptions o; o.rows = R; o.fast = true; o.q8 = true; o.trace = true; \
          return inkling2_verify_resident_bf16_source(o); }},
    FRT_INK_VERIFY(2) FRT_INK_VERIFY(3) FRT_INK_VERIFY(4) FRT_INK_VERIFY(5) FRT_INK_VERIFY(6) FRT_INK_VERIFY(7)
    FRT_INK_VERIFY(8)
#undef FRT_INK_VERIFY
    // inkling2_loop row programs (prefill chunks and batched decode): per-row kernels and TensorOps products.
    // Inkling 2 Mini DSpark speculative decoding (verify staging / commit, drafter)
    {"ink_ring_stage_rows", "ink_ring_stage_rows", inkling2_mtp_bf16_source},
    {"ink_commit_rows", "ink_commit_rows", inkling2_mtp_bf16_source},
    {"ink_mtp_mix_rows", "ink_mtp_mix_rows", inkling2_mtp_bf16_source},
    {"ink_mtp_ctx_write_rows", "ink_mtp_ctx_write_rows", inkling2_mtp_bf16_source},
    {"ink_mtp_embed_rows", "ink_mtp_embed_rows", inkling2_mtp_bf16_source},
    {"ink_mtp_norm_rows", "ink_mtp_norm_rows", inkling2_mtp_bf16_source},
    {"ink_mtp_qk_rows", "ink_mtp_qk_rows", inkling2_mtp_bf16_source},
    {"ink_mtp_attn_rows", "ink_mtp_attn_rows", inkling2_mtp_bf16_source},
    {"ink_mtp_silu_rows", "ink_mtp_silu_rows", inkling2_mtp_bf16_source},
    {"ink_mtp_rnn_in", "ink_mtp_rnn_in", inkling2_mtp_bf16_source},
    {"ink_mtp_gates", "ink_mtp_gates", inkling2_mtp_bf16_source},
    {"ink_mtp_pick", "ink_mtp_pick", inkling2_mtp_bf16_source},
    {"ink_mtp_topk", "ink_mtp_topk", inkling2_mtp_bf16_source},
    // Q8_0 weights: quantization of BF16 matrices, products with up to 8 rows (FP32 out)
    {"q8_0_quantize_bf16", "q8_0_quantize_bf16", q8_0_rows_source},
    {"q8_0_rows_f32", "q8_0_rows_f32", q8_0_rows_source},
    {"ink_mtp_pick_cand", "ink_mtp_pick_cand", inkling2_mtp_bf16_source},
    {"ink_embed_rows", "ink_embed_rows", inkling2_rows_bf16_source},
    {"ink_attn_inputs_rows", "ink_attn_inputs_rows", inkling2_rows_bf16_source},
    {"ink_gate_rows", "ink_gate_rows", inkling2_rows_bf16_source},
    {"ink_heads_rows", "ink_heads_rows", inkling2_rows_bf16_source},
    {"ink_bias_rows", "ink_bias_rows", inkling2_rows_bf16_source},
    {"ink_attn_decode_rows", "ink_attn_decode_rows", inkling2_rows_bf16_source},
    {"ink_post_attn_rows", "ink_post_attn_rows", inkling2_rows_bf16_source},
    {"ink_route_rows", "ink_route_rows", inkling2_rows_bf16_source},
    {"ink_expert_scan", "ink_expert_scan", inkling2_rows_bf16_source},
    {"ink_expert_place_rows", "ink_expert_place_rows", inkling2_rows_bf16_source},
    {"ink_combine_rows", "ink_combine_rows", inkling2_rows_bf16_source},
    {"ink_ring_rows", "ink_ring_rows", inkling2_rows_bf16_source},
    {"ink_final_rows", "ink_final_rows", inkling2_rows_bf16_source},
    {"ink_argmax_rows", "ink_argmax_rows", inkling2_rows_bf16_source},
    {"ink_trace_rows", "ink_trace_rows", inkling2_rows_bf16_source},
    {"ink_nv_arrange", "ink_nv_arrange", inkling2_rows_bf16_source},
    {"ink_experts_mma_gateup", "ink_experts_mma_gateup", inkling2_rows_bf16_source},
    {"ink_experts_mma_down", "ink_experts_mma_down", inkling2_rows_bf16_source},
    {"ink_experts_mma_gateup_tiles", "ink_experts_mma_gateup_tiles", inkling2_rows_bf16_source},
    {"ink_experts_mma_down_tiles", "ink_experts_mma_down_tiles", inkling2_rows_bf16_source},
    {"ink_gemv_rows", "ink_gemv_rows", inkling2_rows_bf16_source},
    {"ink_experts_qmv_rows", "ink_experts_qmv_rows", inkling2_rows_bf16_source},
    {"ink_gemm_bf16_multi_t8x32s1", "ink_gemm_bf16_multi", [] { Inkling2GemmOptions o; o.tile_m = 8; o.tile_n = 32; o.simdgroups = 1; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_f32_multi_t8x32s1", "ink_gemm_f32_multi", [] { Inkling2GemmOptions o; o.tile_m = 8; o.tile_n = 32; o.simdgroups = 1; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_bf16_multi_t16x32s2", "ink_gemm_bf16_multi", [] { Inkling2GemmOptions o; o.tile_m = 16; o.tile_n = 32; o.simdgroups = 2; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_f32_multi_t16x32s2", "ink_gemm_f32_multi", [] { Inkling2GemmOptions o; o.tile_m = 16; o.tile_n = 32; o.simdgroups = 2; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_bf16_multi_t64x64s4", "ink_gemm_bf16_multi", [] { Inkling2GemmOptions o; o.tile_m = 64; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_f32_multi_t64x64s4", "ink_gemm_f32_multi", [] { Inkling2GemmOptions o; o.tile_m = 64; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_bf16_rows_t16x32s2", "ink_gemm_bf16_rows", [] { Inkling2GemmOptions o; o.tile_m = 16; o.tile_n = 32; o.simdgroups = 2; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_bf16_rows_t64x64s4", "ink_gemm_bf16_rows", [] { Inkling2GemmOptions o; o.tile_m = 64; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_bf16_rows_t8x32s1", "ink_gemm_bf16_rows", [] { Inkling2GemmOptions o; o.tile_m = 8; o.tile_n = 32; o.simdgroups = 1; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_f32_rows_t16x32s2", "ink_gemm_f32_rows", [] { Inkling2GemmOptions o; o.tile_m = 16; o.tile_n = 32; o.simdgroups = 2; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_f32_rows_t64x64s4", "ink_gemm_f32_rows", [] { Inkling2GemmOptions o; o.tile_m = 64; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_f32_rows_t8x32s1", "ink_gemm_f32_rows", [] { Inkling2GemmOptions o; o.tile_m = 8; o.tile_n = 32; o.simdgroups = 1; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_logits_rows_t16x32s2", "ink_gemm_logits_rows", [] { Inkling2GemmOptions o; o.tile_m = 16; o.tile_n = 32; o.simdgroups = 2; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_logits_rows_t64x64s4", "ink_gemm_logits_rows", [] { Inkling2GemmOptions o; o.tile_m = 64; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_gemm_logits_rows_t8x32s1", "ink_gemm_logits_rows", [] { Inkling2GemmOptions o; o.tile_m = 8; o.tile_n = 32; o.simdgroups = 1; return inkling2_rows_gemm_source(o); }},
    {"ink_experts_gateup_t16x64s4", "ink_experts_gateup", [] { Inkling2GemmOptions o; o.tile_m = 16; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_experts_down_t8x32s2", "ink_experts_down", [] { Inkling2GemmOptions o; o.tile_m = 8; o.tile_n = 32; o.simdgroups = 2; return inkling2_rows_gemm_source(o); }},
    {"ink_experts_down_t16x64s4", "ink_experts_down", [] { Inkling2GemmOptions o; o.tile_m = 16; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_experts_gateup_t32x64s4", "ink_experts_gateup", [] { Inkling2GemmOptions o; o.tile_m = 32; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_experts_down_t32x64s4", "ink_experts_down", [] { Inkling2GemmOptions o; o.tile_m = 32; o.tile_n = 64; o.simdgroups = 4; return inkling2_rows_gemm_source(o); }},
    {"ink_experts_gateup_t8x32s2k32", "ink_experts_gateup", [] { Inkling2GemmOptions o; o.tile_m = 8; o.tile_n = 32; o.simdgroups = 2; o.kslice = 32; return inkling2_rows_gemm_source(o); }},
};

inline const KernelSource& find_kernel_source(std::string_view name) {
    for (const auto& source : kernel_registry)
        if (name == source.name) return source;
    throw std::invalid_argument("Unknown Apple kernel: " + std::string(name));
}
}  // namespace flashrt::apple::shaders
