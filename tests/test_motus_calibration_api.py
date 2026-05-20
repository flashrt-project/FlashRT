import numpy as np
import pytest
import torch

from flash_rt.frontends.torch.motus_rtx import MotusTorchFrontendRtx


def _frontend_shell():
    rt = MotusTorchFrontendRtx.__new__(MotusTorchFrontendRtx)
    rt.device = torch.device("cpu")
    rt.dtype = torch.bfloat16
    return rt


def test_motus_calibration_sample_shapes():
    rt = _frontend_shell()
    first = torch.randn(1, 3, 384, 320)
    state = torch.randn(1, 14)

    assert rt._normalize_calibration_observations(first) == [first]
    assert rt._extract_calibration_sample(first) == (first, None)
    assert rt._extract_calibration_sample((first, state)) == (first, state)
    assert rt._extract_calibration_sample(
        {"first_frame": first, "state": state}) == (first, state)
    assert rt._extract_calibration_sample({"image": first}) == (first, None)

    with pytest.raises(KeyError):
        rt._extract_calibration_sample({"state": state})
    with pytest.raises(TypeError):
        rt._extract_calibration_sample({"first_frame": first, "state": 1})


def test_motus_calibrate_validates_prompt_and_percentile():
    rt = _frontend_shell()
    rt._t5_embeds = None
    rt._vlm_inputs = None
    rt._graph_state = None
    rt._fp8_swap_stats = None
    rt._fp8_calibrated = True

    first = torch.randn(1, 3, 384, 320)
    with pytest.raises(RuntimeError, match="set_prompt"):
        rt.calibrate([{"first_frame": first}])

    rt._t5_embeds = [torch.empty(1)]
    rt._vlm_inputs = [{}]
    with pytest.raises(ValueError):
        rt.calibrate([{"first_frame": first}], percentile=-1)

    rt._graph_state = object()
    with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
        rt.calibrate([{"first_frame": first}])


class _FakeSite:
    def __init__(self, label, value):
        self.label = label
        self.act_scale = torch.tensor([value], dtype=torch.float32)


class _FakeResampleSite:
    def __init__(self, name, value):
        self.name = name
        self.act_scale = torch.tensor([value], dtype=torch.float32)
        self.act_scale_scalar = float(value)


def test_motus_precision_spec_metadata():
    rt = _frontend_shell()
    root = torch.nn.Module()
    child = torch.nn.Module()
    child._fp8_site = _FakeSite(
        "video_model.wan_model.blocks.0.self_attn.q", 0.125)
    child._fp8_resample_site = _FakeResampleSite("decoder.resample.1", 0.25)
    root.add_module("child", child)
    rt.model = root

    spec = rt._snapshot_precision_spec(
        method="percentile", n=4, percentile=99.9)
    assert spec.source == "calibration"
    assert "video_model.wan_model.blocks.0.self_attn.q" in \
        spec.decoder_layer_specs
    entry = spec.decoder_layer_specs[
        "video_model.wan_model.blocks.0.self_attn.q"]
    assert entry.calibration_method == "percentile"
    assert entry.calibration_samples == 4
    assert entry.calibration_percentile == 99.9
    np.testing.assert_allclose(entry.scale, np.array([0.125], dtype=np.float32))
    assert "vae_resample.decoder.resample.1" in spec.activation_specs
