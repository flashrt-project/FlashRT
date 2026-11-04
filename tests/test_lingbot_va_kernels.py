"""Import and ownership test for the opt-in LingBot-VA/Wan SM89 module.

The module is built only with ``-DFLASHRT_BUILD_LINGBOT_VA_KERNELS=ON`` on
``GPU_ARCH=89``.  On other builds the import skips.  The second test guards the
module boundary: the LingBot entry points must not leak into the shared
``flash_rt_kernels`` module.
"""

from __future__ import annotations

import pytest

REQUIRED = {
    "bf16_avgdown3d_ndhwc",
    "bf16_norm_tail_ncdhw",
    "bf16_norm_tail_ndhwc",
    "bf16_norm_tail_ndhwc_v8",
    "bias_rms_norm_bf16_q8",
    "flow_cfg_euler_patch_bf16",
    "flow_euler_window_bf16",
    "fp8_block128_to_nvfp4_sf128_plain_bf16",
    "fp8_block128_to_nvfp4_sf128_swizzled_bf16",
    "fp8_per_token_block128_quant_warp_bf16",
    "gated_residual_sstp_layer_norm_fp8_block128",
    "gelu_tanh_bias_to_fp8_block128_warp_bf16",
    "kv_ring_advance",
    "kv_window_shift_sage",
    "layer_norm_fp8_block128",
    "nvfp4_pack_tile_major",
    "qkv_split_bias_norm_rope_v_cat2_bf16",
    "qkv_split_bias_norm_rope_v_cat2_sage",
    "quantize_i8_f16",
    "sage2_attn_self",
    "sage2_attn_self_f8",
    "sage2_attn_self_f8_lse",
    "sage2_attn_self_f8_tile",
    "sage2_tail_merge_fp8",
    "v_transpose_f8",
    "v_transpose_f8_tile",
    "wan_adaln_ln_fp8",
    "wan_adaln_residual_ln_bf16",
    "wan_adaln_residual_ln_fp8",
    "wan_adaln_residual_ln_fp8_block128",
    "wan_gated_residual_bf16",
    "wan_gated_residual_bias_bf16",
    "wan_gated_residual_sstp_adaln_fp8_block128",
    "wan_gated_residual_sstp_bias_bf16",
    "xattn_probs_fp8",
}

lingbot = pytest.importorskip("flash_rt.flash_rt_lingbot_va_kernels")


def test_required_symbols_present():
    names = {name for name in dir(lingbot) if not name.startswith("_")}
    missing = REQUIRED - names
    assert not missing, sorted(missing)


def test_shared_module_has_no_lingbot_entries():
    shared = pytest.importorskip("flash_rt.flash_rt_kernels")
    names = {name for name in dir(shared) if not name.startswith("_")}
    leaked = REQUIRED & names
    assert not leaked, sorted(leaked)
