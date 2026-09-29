"""Dispatch smoke for HyVLA on RTX consumer Blackwell (SM120)."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")


def test_hyvla_rtx_dispatch_resolves():
    from flash_rt.hardware import resolve_pipeline_class

    try:
        cls = resolve_pipeline_class("hyvla", "torch", "rtx_sm120")
    except ModuleNotFoundError as exc:
        if exc.name != "flash_rt.flash_rt_kernels":
            raise
        pytest.skip("flash_rt_kernels was not built")
    assert cls.__module__ == "flash_rt.frontends.torch.hyvla_rtx"
    assert cls.__name__ == "HyVLATorchFrontendRtx"


def test_hyvla_rtx_binds_its_own_pipeline():
    # The RTX target must bind an SM120 pipeline (the repo contract wants one
    # lowered execution plan per hardware), not reuse the Thor/Orin class.
    try:
        from flash_rt.frontends.torch.hyvla_rtx import (
            HyVLATorchFrontendRtx)
        from flash_rt.models.hyvla.pipeline_orin import HyVLAOrinBF16Pipeline
        from flash_rt.models.hyvla.pipeline_rtx import HyVLARTXBF16Pipeline
    except ModuleNotFoundError as exc:
        if exc.name != "flash_rt.flash_rt_kernels":
            raise
        pytest.skip("flash_rt_kernels was not built")
    assert HyVLATorchFrontendRtx._PIPE_CLS is HyVLARTXBF16Pipeline
    assert issubclass(HyVLARTXBF16Pipeline, HyVLAOrinBF16Pipeline)


def test_hyvla_pipeline_map_is_one_to_one():
    from flash_rt.hardware import _PIPELINE_MAP

    entries = {k: v for k, v in _PIPELINE_MAP.items() if k[0] == "hyvla"}
    assert ("hyvla", "torch", "rtx_sm120") in entries
    classes = [v[1] for v in entries.values()]
    assert len(classes) == len(set(classes)), "multiple tuples share a class"
