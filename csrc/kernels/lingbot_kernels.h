// Auto-extracted forward declarations for LingBot kernels (compiled into flash_rt_kernels).
#pragma once
#include <cstdint>

void lingbot_ada_rms_residual_fp8_bf16(
    uintptr_t residual, uintptr_t x,
    uintptr_t rms_weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int D, float eps, uintptr_t stream)
;
void lingbot_ada_rms_fp8_mpad_bf16(
    uintptr_t x, uintptr_t rms_weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int PADM, int D, float eps, uintptr_t stream)
;
void lingbot_ada_rms_residual_fp8_mpad_bf16(
    uintptr_t residual, uintptr_t x,
    uintptr_t rms_weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int PADM, int D, float eps, uintptr_t stream)
;
void lingbot_ada_rms_residual_fp16_mpad(
    uintptr_t residual, uintptr_t x,
    uintptr_t rms_weight, uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp16, int B, int S, int PADM, int D, float eps, uintptr_t stream)
;
void lingbot_ada_rms_fp8_bf16(
    uintptr_t x, uintptr_t rms_weight,
    uintptr_t gamma, uintptr_t beta,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int D, float eps, uintptr_t stream)
;
void lingbot_rope_inplace_bf16(
    uintptr_t x, uintptr_t cos_table, uintptr_t sin_table,
    int B, int S, int NH, int HD, uintptr_t stream)
;
void lingbot_rope_to_out_bf16(
    uintptr_t x, uintptr_t out, uintptr_t cos_table, uintptr_t sin_table,
    int B, int S, int NH, int HD, uintptr_t stream)
;
void lingbot_qkv_bias_rope_bf16(
    uintptr_t q_raw, uintptr_t k_raw, uintptr_t v_raw,
    uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NHQ, int NHKV, int HD, uintptr_t stream)
;
void lingbot_qkv_bias_rope_merged_fp16out(
    uintptr_t qkv, uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NHQ, int NHKV, int HD, uintptr_t stream)
;
void lingbot_qkv_bias_rope_merged_bf16(
    uintptr_t qkv, uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NHQ, int NHKV, int HD, uintptr_t stream)
;
void lingbot_qkv_bias_rope_fp16out(
    uintptr_t q_raw, uintptr_t k_raw, uintptr_t v_raw,
    uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NHQ, int NHKV, int HD, uintptr_t stream)
;
void lingbot_vit_qkv_bias_rope_bf16(
    uintptr_t qkv, uintptr_t qkv_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NH, int HD, uintptr_t stream)
;
void lingbot_silu_mul_fp8_mpad_bf16(
    uintptr_t gate, uintptr_t up, uintptr_t out_fp8, uintptr_t act_scale,
    int S, int PADM, int I, uintptr_t stream)
;
void lingbot_silu_mul_merged_fp8_mpad_bf16(
    uintptr_t gu, uintptr_t out_fp8, uintptr_t act_scale,
    int S, int PADM, int I, uintptr_t stream)
;
void lingbot_silu_mul_merged_fp8_mpad_fp16in(
    uintptr_t gu, uintptr_t out_fp8, uintptr_t act_scale,
    int S, int PADM, int I, uintptr_t stream)
;
