"""Pi0.5-owned kernel bindings; no changes to shared-module flags or ABI."""
from types import SimpleNamespace
from flash_rt import flash_rt_kernels as _kernels, flash_rt_fp4 as _fp4
try:
    from flash_rt import flash_rt_pi05_thor as _optimized
except ImportError as exc:
    raise ImportError("Rebuild with -DGPU_ARCH=110 -DFLASHRT_ENABLE_PI05_THOR=ON") from exc

kernels = SimpleNamespace(**{name: getattr(_kernels, name) for name in dir(_kernels)})
fp4 = SimpleNamespace(**{name: getattr(_fp4, name) for name in dir(_fp4)})
fp4.attention_qkv_fp16_lt = _optimized.attention_qkv_fp16_lt
fp4.cutlass_fp4_gemm_bias_gelu_fp4out_v = _optimized.cutlass_fp4_gemm_bias_gelu_fp4out_v
fp4.cutlass_fp4_gemm_bias_res_fp16_v = _optimized.cutlass_fp4_gemm_bias_res_fp16_v
fp4.cutlass_fp4_gemm_geglu_il_hw_nod = _optimized.cutlass_fp4_gemm_geglu_il_hw_nod
fp4.cutlass_fp4_gemm_geglu_il_hw_nod_swap = _optimized.cutlass_fp4_gemm_geglu_il_hw_nod_swap
fp4.cutlass_fp4_gemm_variant = _optimized.cutlass_fp4_gemm_variant
fp4.l2_touch_fork = _optimized.l2_touch_fork
fp4.l2_touch_init = _optimized.l2_touch_init
fp4.l2_touch_join = _optimized.l2_touch_join
fp4.pi05_action_in_fp16 = _optimized.pi05_action_in_fp16
fp4.pi05_adarms_action_out_fp16 = _optimized.pi05_adarms_action_out_fp16
fp4.rowops_layer_norm_fp16_v2 = _optimized.pi05_row_layer_norm_fp16
fp4.rowops_layer_norm_fp8_v5 = _optimized.pi05_row_layer_norm_fp8_swizzled
fp4.rowops_layer_norm_mul_fp4_sfa_v5 = _optimized.pi05_row_layer_norm_mul_fp4_sfa_swizzled
fp4.rowops_quantize_fp4_sfa_v5 = _optimized.pi05_row_quantize_fp4_sfa_swizzled
fp4.rowops_rms_fp8_v5 = _optimized.pi05_row_rms_fp8_swizzled
fp4.rowops_rms_mul_fp4_sfa_v5 = _optimized.pi05_row_rms_mul_fp4_sfa_swizzled

fp4.set_pdl = _optimized.set_pdl
kernels.set_pdl = _optimized.set_pdl
kernels.patch_im2col_uint8_pitch = _optimized.pi05_patch_im2col_uint8_pitch
kernels.patch_embed_bias_pos_v2 = _optimized.pi05_patch_embed_bias_pos

fp4.cutlass_fp4_gemm_num_variants = _optimized.cutlass_fp4_gemm_num_variants

# The entire optimized producer/consumer chain uses this module's local PDL flag.
kernels.quantize_fp8_static_fp16 = _optimized.pi05_quantize_fp8_static_fp16
kernels.cutlass_fp8_sq = _optimized.pi05_cutlass_fp8_sq
kernels.qkv_split_rope_kvcache_fp16_vec = _optimized.pi05_qkv_split_rope_kvcache_fp16_vec
fp4.pi05_adarms_fp4_sfa_native_fp16 = _optimized.pi05_adarms_fp4_sfa_native_fp16
fp4.pi05_gate_res_adarms_fp4_sfa_native_fp16 = _optimized.pi05_gate_res_adarms_fp4_sfa_native_fp16
