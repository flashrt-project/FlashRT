from types import SimpleNamespace
import sys

import numpy as np
import pytest

from flash_rt.frontends.torch.dm05_rtx import Dm05TorchFrontendRtx


def test_dispatch_resolves_the_dedicated_frontend():
    from flash_rt.hardware import resolve_pipeline_class
    assert resolve_pipeline_class("dm05", "torch", "rtx_sm120") is Dm05TorchFrontendRtx


@pytest.mark.parametrize("framework,hardware", [("jax", "rtx_sm120"), ("torch", "thor"),
                                                 ("torch", "rtx_sm89"), ("torch", "amd_rdna35")])
def test_unsupported_routes_fail_before_extension_loading(framework, hardware):
    from flash_rt import load_model
    with pytest.raises(ValueError, match="DM05 supports"):
        load_model("checkpoint", config="dm05", framework=framework, hardware=hardware, use_fp4=True)


def test_public_loader_forwards_the_mixed_tier_and_schedule(monkeypatch):
    from flash_rt import load_model, _extensions
    import flash_rt.hardware as hw
    seen = {}
    class Frontend:
        def __init__(self, checkpoint, num_views=2, use_fp4=False, num_steps=10):
            seen.update(checkpoint=checkpoint, num_views=num_views, use_fp4=use_fp4, num_steps=num_steps)
    monkeypatch.setattr(_extensions, "require", lambda *a, **k: None)
    monkeypatch.setattr(hw, "resolve_pipeline_class", lambda *a: Frontend)
    result = load_model("checkpoint", config="dm05", hardware="rtx_sm120", use_fp4=True, num_steps=7)
    assert result.pipeline.__class__ is Frontend
    assert seen == dict(checkpoint="checkpoint", num_views=2, use_fp4=True, num_steps=7)


@pytest.mark.parametrize("options", [{}, {"use_fp4": True, "use_fp8": False},
                                      {"use_fp4": True, "use_fp16": True}])
def test_unsupported_precision_never_silently_selects_a_tier(options):
    from flash_rt import load_model
    with pytest.raises(ValueError, match="mixed tier"):
        load_model("checkpoint", config="dm05", hardware="rtx_sm120", **options)


def test_preprocessing_builds_wrappers_without_changing_runtime_transforms():
    from flash_rt.models.dm05.preprocess import build_transforms, RawImages, RawAwareTokenization
    class PixelTransform:
        pass
    class ChatTokenization:
        tokenizer = object()
        processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 1), image_token="image")
    original = [PixelTransform(), ChatTokenization()]
    before = list(original)
    wrapped, tok = build_transforms(original)
    assert all(a is b for a, b in zip(original, before))
    assert wrapped is not original and isinstance(wrapped[0], RawImages)
    assert isinstance(wrapped[1], RawAwareTokenization) and tok is wrapped[1]


def test_frontend_never_replaces_external_runtime_methods(monkeypatch, tmp_path):
    import torch
    from flash_rt import _extensions
    import flash_rt.frontends.torch.dm05_rtx as module
    import flash_rt.models.dm05.preprocess as prep
    model = SimpleNamespace(inference_action=object())
    runtime = SimpleNamespace(model=model, _predict=object(),
        input_transform=SimpleNamespace(transforms=[object()]), output_action_dim=7, full_action_mask=False)
    old_predict, old_action = runtime._predict, model.inference_action
    class Pipeline:
        A = 7
        action = SimpleNamespace(T=10)
        vision = SimpleNamespace(nv=2)
        def __init__(self, *a, **k):
            self.invalidate = k["invalidate_graphs"]
        def configure(self, config):
            self.invalidate()
        def load_calibration_state(self, state):
            self.invalidate()
    monkeypatch.setitem(sys.modules, "flash_rt.models.dm05.pipeline_rtx",
                        SimpleNamespace(Dm05Config=lambda: object(), Dm05PipelineRtx=Pipeline))
    monkeypatch.setattr(module, "_load_runtime", lambda *a: (object(), runtime, model))
    monkeypatch.setattr(prep, "build_transforms", lambda tr: (list(tr), object()))
    monkeypatch.setattr(_extensions, "require", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original_empty(*a, **{k:v for k,v in kw.items() if k != "pin_memory"}))
    frontend = Dm05TorchFrontendRtx(tmp_path, use_fp4=True)
    frontend.load_calibration_state({})
    assert frontend.calibrated
    assert runtime._predict is old_predict and model.inference_action is old_action
    assert len(runtime.input_transform.transforms) == 1


def test_inference_requires_explicit_calibration():
    frontend = object.__new__(Dm05TorchFrontendRtx)
    frontend.calibrated = False
    frontend._calibrating = False
    with pytest.raises(RuntimeError, match="needs calibration"):
        frontend.infer({})


def test_state_prompt_updates_are_not_replaced_by_cached_state_tokens(monkeypatch):
    import torch
    from flash_rt.models.dm05.preprocess import RawAwareTokenization
    from flash_rt.models.dm05.tokenize_cache import CachedChatTokenization
    inner = SimpleNamespace(add_state=True, tokenizer=object(), processor=SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 1), image_token="image"))
    tok = RawAwareTokenization(inner)
    tok.pixel_transform = SimpleNamespace(transform_pipeline=lambda image: {"image": image})
    monkeypatch.setattr(CachedChatTokenization, "__call__",
                        lambda self, data: {"input_ids": torch.tensor([[int(data["state"][0])]])})
    raw = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    data = {"prompt": "pick", "images": [object()], "state": [1]}
    tok.cache[tok._key(data)] = ([99], [])
    first = tok(dict(data, _native_raw=raw))
    second = tok(dict(data, state=[2], _native_raw=raw))
    assert first["input_ids"].item() == 1
    assert second["input_ids"].item() == 2


def test_graph_replay_uses_fresh_buffers_and_separates_layouts():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for graph wrapper test")
    pytest.importorskip("flash_rt.flash_rt_kernels")
    frontend = object.__new__(Dm05TorchFrontendRtx)
    frontend._graphs = {}
    frontend._graph_stream = torch.cuda.Stream()
    frontend._calibrating = False
    x = torch.ones(8, device="cuda")
    out = torch.empty_like(x)
    prefix = SimpleNamespace(seq=4, segments=[(0, 4, 4, True)])
    frontend._native = SimpleNamespace(prefix=prefix, action=SimpleNamespace(use_mask=False),
        raw_mode=True, host_staged=True, _run=lambda stream: out.copy_(x * 2))
    frontend._execute()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full_like(out, 2))
    x.fill_(3)
    frontend._execute()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full_like(out, 6))
    assert len(frontend._graphs) == 1
    prefix.seq = 6
    prefix.segments = [(0, 6, 6, True)]
    frontend._execute()
    torch.cuda.synchronize()
    assert len(frontend._graphs) == 2
    prefix.seq = 4
    prefix.segments = [(0, 4, 4, True)]
    x.fill_(4)
    frontend._execute()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full_like(out, 8))
    assert len(frontend._graphs) == 2


def test_calibration_iteration_and_graph_invalidation_belong_to_frontend():
    frontend = object.__new__(Dm05TorchFrontendRtx)
    frontend._graphs = {"old": object()}
    frontend.calibrated = False
    seen = []
    frontend._native = SimpleNamespace(begin_calibration=lambda: seen.append("begin"),
                                      finish_calibration=lambda: {"static_sites": 2})
    def infer(observation):
        assert frontend._calibrating
        seen.append(observation)
    frontend.infer = infer
    result = frontend.calibrate(["first", "second", "third"], percentile=100.0, max_samples=2)
    assert seen == ["begin", "first", "second"]
    assert result == {"static_sites": 2, "observations": 2}
    assert frontend.calibrated and not frontend._calibrating
    assert not frontend._graphs


def test_failed_calibration_cannot_reuse_a_previously_valid_graph():
    frontend = object.__new__(Dm05TorchFrontendRtx)
    frontend._graphs = {"old": object()}
    frontend.calibrated = True
    frontend._native = SimpleNamespace(begin_calibration=lambda: None)
    def infer(observation):
        raise ValueError("bad observation")
    frontend.infer = infer
    with pytest.raises(ValueError, match="bad observation"):
        frontend.calibrate([{}], percentile=100.0)
    assert not frontend.calibrated and not frontend._calibrating
    assert not frontend._graphs
