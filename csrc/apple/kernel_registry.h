#pragma once
#include "attention/attention_d72_bf16.h"
#include "attention/attention_simd_bf16.h"
#include "attention/gqa_merge_fp32_bf16.h"
#include "attention/gqa_online_bf16.h"
#include "attention/gqa_split_bf16.h"
#include "conv/patch_pack_bf16.h"
#include "conv/patch_probe_bf16.h"
#include "gemm/bf16_tensor_ops.h"
#include "kernels/ada_rms_norm_bf16.h"
#include "kernels/bias_bf16.h"
#include "kernels/gelu_bf16.h"
#include "kernels/gelu_mul_bf16.h"
#include "kernels/layer_norm_bf16.h"
#include "kernels/position_add_bf16.h"
#include "kernels/residual_bf16.h"
#include "kernels/rms_norm_bf16.h"
#include "kernels/rope_bf16.h"
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
    {"gemm_bf16", "gemm_bf16", [] { return gemm_bf16_source(); }},
    {"gemm_bias_bf16", "gemm_bias_bf16", [] { return gemm_bf16_source(); }},
    {"gemm_patch_bf16", "gemm_patch_bf16", [] { return gemm_bf16_source(); }},
    {"gemm_strided_bf16", "gemm_strided_bf16", [] { return gemm_bf16_source(); }},
    {"gemm_small_bf16", "gemm_bf16", [] { return gemm_bf16_source(16); }},
    {"gemm_small_bias_bf16", "gemm_bias_bf16", [] { return gemm_bf16_source(16); }},
    {"gemm_small_strided_bf16", "gemm_strided_bf16", [] { return gemm_bf16_source(16); }},
    {"attention_d72_bf16", "attention_d72_bf16", attention_d72_bf16_source},
    {"attention_bf16", "attention_bf16", attention_simd_bf16_source},
    {"tr_gqa", "tr_gqa", gqa_online_bf16_source},
    {"tr_gqa_split", "tr_gqa_split", gqa_split_bf16_source},
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
    {"tr_rope", "tr_rope", rope_bf16_source},
};

inline const KernelSource& find_kernel_source(std::string_view name) {
    for (const auto& source : kernel_registry)
        if (name == source.name) return source;
    throw std::invalid_argument("Unknown Apple kernel: " + std::string(name));
}
}  // namespace flashrt::apple::shaders
