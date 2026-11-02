"""CPU-testable RDNA N1.7 input and public-loader contracts."""

import functools
import inspect
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flash_rt.amd.frontends.torch.groot_n17_rdna import _host_frames, frames_from_observation


def test_history_and_camera_order_uses_checkpoint_order():
    # Deliberately insert wrist first; dictionary insertion order is not the
    # processor's camera order. Distinct values make any permutation visible.
    front = np.zeros((1, 2, 3, 5, 3), dtype=np.uint8)
    wrist = front.copy()
    front[:, 0], front[:, 1], wrist[:, 0], wrist[:, 1] = 11, 21, 12, 22
    frames = frames_from_observation({"video": {"wrist": wrist, "front": front}},
                                     ["front", "wrist"])
    assert frames[:, 0, 0, 0].tolist() == [11, 12, 21, 22]
    for value in (frames, torch.from_numpy(frames), list(frames)):
        assert np.array_equal(_host_frames(value), frames)
    assert np.array_equal(_host_frames(frames[:, :, ::-1]), frames[:, :, ::-1])


@pytest.mark.parametrize("frames", [
    {}, [], 1, np.zeros((2, 3, 4, 3), dtype=np.float32),
    torch.zeros(2, 3, 4, 3), np.zeros((3, 4, 3), dtype=np.uint8),
    np.zeros((2, 0, 4, 3), dtype=np.uint8),
    np.zeros((2, 3, 4, 4), dtype=np.uint8),
])
def test_frames_reject_ambiguous_or_invalid_inputs(frames):
    with pytest.raises(ValueError, match="frames"):
        _host_frames(frames)


@pytest.mark.parametrize("hardware,selected,use_fp8", [
    (hardware, selected, use_fp8)
    for hardware in ("amd_rdna3", "amd_rdna35", "amd_rdna4")
    for selected in (None, False, True)
    for use_fp8 in (None, False, True)
] + [("rtx_sm87", None, False)])
def test_load_model_forwards_rdna_controls_and_preserves_defaults(
        monkeypatch, caplog, hardware, selected, use_fp8):
    import flash_rt
    import flash_rt.amd
    from flash_rt import _extensions
    from flash_rt.hardware import resolve_pipeline_class

    # This test intercepts construction before any computation. A CUDA binary
    # is irrelevant to keyword forwarding and unavailable in ROCm-only CI.
    monkeypatch.setattr(_extensions, "require", lambda *args, **kwargs: None)
    # Also isolate AMD's native-module gate: this contract test must run in a
    # CPU-only wheel without either vendor's compiled extension installed.
    native = types.ModuleType("flash_rt.amd.flash_rt_amd_kernels")
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(flash_rt.amd, "flash_rt_amd_kernels", native, raising=False)
    cls = resolve_pipeline_class("groot_n17", "torch", hardware)
    original = cls.__init__

    @functools.wraps(original)
    def capture(self, *args, **kwargs):
        bound = inspect.signature(original).bind(self, *args, **kwargs)
        bound.apply_defaults()
        self.arguments = bound.arguments

    monkeypatch.setattr(cls, "__init__", capture)
    backbone = None if hardware == "rtx_sm87" else False
    precision = {} if use_fp8 is None else {"use_fp8": use_fp8}
    model = flash_rt.load_model("/not-opened", config="groot_n17", hardware=hardware,
                               fuse_image_embeds=selected,
                               use_backbone_graph=backbone, **precision)
    default = inspect.signature(original).parameters["fuse_image_embeds"].default
    assert model.pipeline.arguments["fuse_image_embeds"] is (default if selected is None else selected)
    assert model.pipeline.arguments["use_backbone_graph"] is (True if backbone is None else backbone)
    assert model.pipeline.arguments["use_fp8"] is False
    assert ("disabling use_fp8" in caplog.text) is (use_fp8 is not False)


@pytest.mark.parametrize("name", ["fuse_image_embeds", "use_backbone_graph"])
def test_loader_rejects_unsupported_explicit_controls_before_backend_import(name):
    import flash_rt
    with pytest.raises(TypeError, match=name):
        flash_rt.load_model("/not-opened", config="groot_n17", **{name: "False"})
    with pytest.raises(ValueError, match="GR00T N1.7"):
        flash_rt.load_model("/not-opened", config="pi05", **{name: False})
    for arch in ("rtx_sm87", "rtx_sm120", "rtx_sm89", "thor", "amd_cdna3", "amd_cdna4"):
        with pytest.raises(ValueError, match="on RDNA"):
            flash_rt.load_model("/not-opened", config="groot_n17", hardware=arch, **{name: True})


@pytest.mark.parametrize("hardware", ["amd_rdna3", "amd_rdna35", "amd_rdna4"])
@pytest.mark.parametrize("precision", ["use_fp4", "use_fp16"])
def test_rdna_groot_refuses_unsupported_precision_before_backend_import(hardware, precision):
    import flash_rt

    options = {"use_fp8": False, precision: True}
    with pytest.raises(ValueError, match="BF16 only"):
        flash_rt.load_model("/not-opened", config="groot_n17", hardware=hardware, **options)
