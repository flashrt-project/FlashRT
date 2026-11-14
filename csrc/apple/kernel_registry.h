#pragma once
#include "attention/attention_d72_bf16.h"
#include "attention/attention_simd_bf16.h"
#include "attention/gqa8_causal_prefill_bf16.h"
#include "attention/gqa8_causal_prefill_mma_bf16.h"
#include "attention/gqa_merge_fp32_bf16.h"
#include "attention/gqa_online_bf16.h"
#include "attention/gqa_split_bf16.h"
#include "conv/patch_pack_bf16.h"
#include "conv/patch_probe_bf16.h"
#include "gemm/bf16_tensor_ops.h"
#include "gemm/w4a16_affine_qmv.h"
#include "gemm/w4a16_affine_tensor_ops.h"
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
     [] { W4A16Options o; o.grouped = true; o.simdgroups = 4; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_grouped_swiglu", "gemm_w4a16_affine",
     [] { W4A16Options o; o.grouped = true; o.simdgroups = 4; o.epilogue = 2; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_splitk", "gemm_w4a16_affine",
     [] { W4A16Options o; o.splitk = true; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_splitk_residual", "gemm_w4a16_affine",
     [] { W4A16Options o; o.splitk = true; o.epilogue = 1; return gemm_w4a16_affine_source(o); }},
    {"gemm_w4a16_affine_splitk_swiglu", "gemm_w4a16_affine",
     [] { W4A16Options o; o.splitk = true; o.epilogue = 2; return gemm_w4a16_affine_source(o); }},
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
};

inline const KernelSource& find_kernel_source(std::string_view name) {
    for (const auto& source : kernel_registry)
        if (name == source.name) return source;
    throw std::invalid_argument("Unknown Apple kernel: " + std::string(name));
}
}  // namespace flashrt::apple::shaders
