"""Import/link regression for the existing default SM120 Motus Sage2 surface."""
import importlib
import os
import pytest


def test_default_sm120_motus_module_import_and_symbols():
    if os.environ.get("FLASHRT_TEST_MOTUS_SAGE2") != "1":
        pytest.skip("set FLASHRT_TEST_MOTUS_SAGE2=1 after the default SM120 build")
    kernels = importlib.import_module("flash_rt.flash_rt_kernels")
    for name in ("sage2_qk_int8_sv_f16_bf16_nhd_d128",
                 "sage2_qk_int8_sv_f16_bf16_gqa_nhd_d256",
                 "sage2_qk_int8_sv_f8_bf16_gqa_nhd_d256"):
        assert callable(getattr(kernels, name))
    assert not hasattr(kernels, "sage2_attn_self_f8_tile")
