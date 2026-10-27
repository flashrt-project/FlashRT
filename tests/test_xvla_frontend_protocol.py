"""The X-VLA frontend must expose the public ``load_model`` surface."""

import inspect

import pytest

pytest.importorskip("flash_rt")

from flash_rt.frontends.torch.xvla_rtx import XVLATorchFrontendRtx


def test_init_accepts_standard_kwargs():
    params = inspect.signature(XVLATorchFrontendRtx.__init__).parameters
    assert "checkpoint_dir" in params
    assert "num_views" in params
    assert "precision" in params


def test_set_prompt_accepts_text():
    params = inspect.signature(XVLATorchFrontendRtx.set_prompt).parameters
    assert "prompt" in params
    assert "images" in params
    assert "state" in params


def test_infer_accepts_observation_dict():
    params = inspect.signature(XVLATorchFrontendRtx.infer).parameters
    assert "obs" in params
