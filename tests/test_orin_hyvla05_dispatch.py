"""Dispatch smoke for HyVLA on Jetson Orin SM87."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")


def test_hyvla_orin_dispatch_resolves():
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("hyvla", "torch", "rtx_sm87")
    assert cls.__module__ == "flash_rt.frontends.torch.hyvla_orin"
    assert cls.__name__ == "HyVLATorchFrontendOrin"
