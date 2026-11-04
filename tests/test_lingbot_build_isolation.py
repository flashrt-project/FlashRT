"""Runtime export boundary for separately built enabled/disabled kernel sets."""
import importlib
import os
import pytest

EXPECTED = {'flash_rt_kernels': ['bias_rms_norm_bf16_q8',
                      'quantize_i8_f16',
                      'gated_residual_sstp_layer_norm_fp8_block128',
                      'layer_norm_fp8_block128',
                      'wan_adaln_ln_fp8',
                      'wan_adaln_residual_ln_fp8',
                      'wan_adaln_residual_ln_bf16',
                      'wan_gated_residual_bf16',
                      'wan_adaln_residual_ln_fp8_block128',
                      'wan_gated_residual_sstp_adaln_fp8_block128',
                      'wan_gated_residual_sstp_bias_bf16',
                      'wan_gated_residual_bias_bf16',
                      'fp8_block128_to_nvfp4_sf128_swizzled_bf16',
                      'fp8_block128_to_nvfp4_sf128_plain_bf16',
                      'nvfp4_pack_tile_major',
                      'fp8_per_token_block128_quant_warp_bf16',
                      'gelu_tanh_bias_to_fp8_block128_warp_bf16',
                      'bf16_norm_tail_ncdhw',
                      'bf16_norm_tail_ndhwc',
                      'bf16_avgdown3d_ndhwc',
                      'bf16_norm_tail_ndhwc_v8',
                      'qkv_split_bias_norm_rope_v_cat2_bf16',
                      'qkv_split_bias_norm_rope_v_cat2_sage',
                      'v_transpose_f8',
                      'kv_window_shift_sage',
                      'v_transpose_f8_tile',
                      'kv_ring_advance',
                      'sage2_attn_self_f8',
                      'flow_cfg_euler_patch_bf16',
                      'xattn_probs_fp8',
                      'flow_euler_window_bf16',
                      'sage2_attn_self_f8_lse',
                      'sage2_tail_merge_fp8',
                      'sage2_attn_self_f8_tile',
                      'sage2_attn_self'],
 'flash_rt_qwen3_vl_kernels': ['fp8_bs_gemm_cutlass_sm89',
                               'fp8_bs_gemm_cutlass_f16blk_sm89',
                               'fp8_bs_gemm_cutlass_f16blk_k64_sm89',
                               'fp8_bs_gemm_cutlass_f16blk_k64_residual_sm89',
                               'fp8_bs_gemm_cutlass_f16blk_k64_gelu_fp8_sm89',
                               'fp8_bs_gemm_cutlass_sm89_s2',
                               'fp8_bs_gemm_cutlass_sm89_s4',
                               'fp4_bs_gemm_blockscaled_sm89_bf16out',
                               'fp4_bs_gemm_blockscaled_sm89_bf16out_tile',
                               'fp4_bs_gemm_splitk_blockscaled_sm89_bf16out',
                               'fp4_bs_gemm_blockscaled_sm89_bf16out_tm',
                               'fp4_bs_gemm_blockscaled_sm89_bf16out_tile_tm',
                               'fp4_bs_gemm_splitk_blockscaled_sm89_bf16out_tm',
                               'fp4_bs_gemm_splitk_blockscaled_sm89_bf16out_tm_sf128',
                               'fp4_bs_gemm_splitk_blockscaled_sm89_bf16out_tm_sf128_res',
                               'fp4_stream_m16_sm89_bf16out',
                               'fp4_stream_m128_sm89_bf16out',
                               'nvfp4_m128_lut_build',
                               'fp4_stream_m128_lut_sm89_bf16out',
                               'nvfp4_pack_frag_m16',
                               'fp4_bs_gemm_blockscaled_sm89_bf16out_s2_sf128',
                               'fp4_bs_gemm_blockscaled_sm89_bf16out_s1_sf128',
                               'conv3d_ndhwc_f16acc_bf16out',
                               'ndhwc_time_cat_bf16_to_f16']}

@pytest.mark.parametrize("module_name", EXPECTED)
def test_lingbot_exports_follow_build_gate(module_name):
    mode=os.environ.get("FLASHRT_TEST_LINGBOT_EXPORTS")
    if mode not in ("ON","OFF"):
        pytest.skip("set FLASHRT_TEST_LINGBOT_EXPORTS for the corresponding built module")
    module=importlib.import_module("flash_rt."+module_name)
    for name in EXPECTED[module_name]:
        assert hasattr(module,name) == (mode=="ON"), (module_name,name,mode)
