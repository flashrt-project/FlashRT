"""X-VLA routing/import smoke: the public map must resolve to the RTX frontend."""

import pytest

pytest.importorskip("flash_rt")

from flash_rt.frontends.torch.xvla_rtx import XVLATorchFrontendRtx
from flash_rt.hardware import _PIPELINE_MAP, resolve_pipeline_class
from flash_rt.models.xvla.pipeline_rtx import XVLARTXPipeline


def test_pipeline_map_entry_is_one_to_one():
    key = ("xvla", "torch", "rtx_sm120")
    assert key in _PIPELINE_MAP
    module, cls = _PIPELINE_MAP[key]
    assert module == "flash_rt.frontends.torch.xvla_rtx"
    assert cls == "XVLATorchFrontendRtx"
    # Compare by module/name, not object identity: suite-level tests that
    # purge flash_rt from sys.modules would otherwise make this depend on
    # test ordering.
    resolved = resolve_pipeline_class(*key)
    assert resolved.__module__ == XVLATorchFrontendRtx.__module__
    assert resolved.__name__ == XVLATorchFrontendRtx.__name__
    assert XVLARTXPipeline.__name__ == "XVLARTXPipeline"


def test_frontend_exposes_public_surface():
    assert hasattr(XVLATorchFrontendRtx, "set_prompt")
    assert hasattr(XVLATorchFrontendRtx, "infer")
    assert hasattr(XVLATorchFrontendRtx, "predict_actions")
    assert hasattr(XVLATorchFrontendRtx, "capture")


def test_load_model_accepts_xvla_config():
    import flash_rt

    # The checkpoint does not exist. The important part is that routing gets
    # past the config whitelist; the failure must not be "Unknown config".
    with pytest.raises(Exception) as excinfo:
        flash_rt.load_model(
            "/nonexistent-xvla-checkpoint",
            config="xvla",
            framework="torch",
            hardware="rtx_sm120",
        )
    assert "Unknown config" not in str(excinfo.value)
