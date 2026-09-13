"""The Ascend GR00T N1.7 frontend: its routing, and every argument it refuses.

The frontend is the adapter that makes the ported compute reachable through
``load_model``. What it does per frame needs the part; what it *refuses* does
not, and those refusals are the whole reason it is a class and not a page of
example code — a missing prompt key or a mixed-up frame argument has to be named
where the caller made it, not three calls later as a shape error.

``__init__``'s checks all run before the checkpoint is opened, so they need
nothing. Everything after that is exercised with the three bound pieces faked,
which is legitimate here because none of them is what these tests are about.
"""
import types

import pytest
import torch

from flash_rt.hardware import _PIPELINE_MAP, resolve_pipeline_class

MODULE = "flash_rt.npu.frontends.torch.groot_n17"
TAG = "oxe_droid_relative_eef_relative_joint"


def _frontend_class():
    return resolve_pipeline_class("groot_n17", "torch", "npu")


# ── routing ───────────────────────────────────────────────────────────

def test_the_model_is_routed_for_this_backend():
    """Without this entry `load_model(..., hardware="npu")` cannot reach any of
    the compute, which is what kept the port out of the public API."""
    assert _PIPELINE_MAP[("groot_n17", "torch", "npu")] == (
        MODULE, "GrootN17TorchFrontendNpu")
    assert _frontend_class().__name__ == "GrootN17TorchFrontendNpu"


def test_the_contract_is_the_one_this_models_other_backends_have():
    """Same four calls as Thor and CDNA4, so a caller written against one works
    against this: the observation bundle is the caller's on every backend."""
    import inspect

    cls = _frontend_class()
    prompt = inspect.signature(cls.set_prompt).parameters
    assert prompt["aux"].kind is inspect.Parameter.KEYWORD_ONLY
    assert prompt["prompt"].default is None
    infer = inspect.signature(cls.infer).parameters
    assert "state_normalized" in infer
    assert infer["aux"].kind is inspect.Parameter.KEYWORD_ONLY
    assert infer["initial_noise"].kind is inspect.Parameter.KEYWORD_ONLY
    for name in ("normalize_state", "denormalize_action", "set_hf_processor"):
        assert callable(getattr(cls, name))


def test_load_model_forwards_what_this_frontend_declares():
    """`load_model` feature-detects on the signature, so the names have to be the
    ones it looks for or the caller's arguments are silently dropped."""
    import inspect

    parameters = inspect.signature(_frontend_class()).parameters
    for name in ("num_views", "embodiment_tag", "action_horizon", "use_int8"):
        assert name in parameters, f"load_model forwards {name} when declared"


# ── what __init__ refuses before opening a checkpoint ──────────────────

def test_there_is_no_int8_tier_to_select_and_asking_says_so():
    """A request for a tier that is not served must not be quietly downgraded:
    an INT8 request that ran BF16 would be measured as a regression."""
    with pytest.raises(NotImplementedError, match="no INT8 tier"):
        _frontend_class()("somewhere", use_int8=True)


def test_an_embodiment_the_checkpoint_does_not_have_is_refused():
    with pytest.raises(ValueError, match="not in the checkpoint's table"):
        _frontend_class()("somewhere", embodiment_tag="not_a_robot")


def test_a_view_count_the_embodiment_was_not_trained_with_is_refused():
    with pytest.raises(ValueError, match="trained with 2 camera"):
        _frontend_class()("somewhere", embodiment_tag=TAG, num_views=1)


# ── the rest, with the bound pieces faked ─────────────────────────────

class _Frame:
    def __init__(self, *a, **k):
        self.filled = None
        self.captured = False
        self.actions = torch.zeros(1, 40, 132, dtype=torch.bfloat16)

    def capture(self):
        self.captured = True

    def fill(self, patches, state, noise):
        self.filled = (patches, state, noise)

    def replay(self):
        return self.actions


@pytest.fixture
def frontend(monkeypatch):
    import importlib

    module = importlib.import_module(MODULE)
    monkeypatch.setattr(module, "load_frame", lambda path: object())
    monkeypatch.setattr(module.bb, "BoundBackbone", lambda w, **k: object())
    monkeypatch.setattr(module.pl, "BoundChain", lambda w, i, **k: object())
    monkeypatch.setattr(module, "CapturedFrame", _Frame)
    return module.GrootN17TorchFrontendNpu("somewhere", embodiment_tag=TAG,
                                           device="cpu")


def _aux(**extra):
    from flash_rt.npu.frontends.torch.groot_n17 import PROMPT_KEYS

    aux = {key: torch.zeros(1) for key in PROMPT_KEYS}
    aux["views"] = 2
    aux.update(extra)
    return aux


def test_a_prompt_bundle_missing_a_key_names_the_key(frontend):
    aux = _aux()
    del aux["llm_cos"]
    del aux["patch_positions"]
    with pytest.raises(ValueError, match="llm_cos"):
        frontend.set_prompt(aux=aux)


def test_a_prompt_bundle_describing_other_cameras_is_refused(frontend):
    with pytest.raises(ValueError, match="camera view"):
        frontend.set_prompt(aux=_aux(views=1))


def test_set_prompt_captures_the_graph(frontend):
    frontend.set_prompt(aux=_aux(), prompt="pick up the cube")
    assert frontend._frame.captured
    assert frontend._prompt == "pick up the cube"


def test_infer_before_set_prompt_is_refused(frontend):
    with pytest.raises(RuntimeError, match="call set_prompt"):
        frontend.infer(torch.zeros(1, 1, 132))


def test_reusing_the_previous_frames_features_is_refused_not_guessed(frontend):
    """This tier captures the whole frame. A features-only replay is a second
    graph that has not been measured, so omitting the observation says that
    rather than silently returning the previous frame's answer."""
    frontend.set_prompt(aux=_aux())
    with pytest.raises(NotImplementedError, match="every call needs the"):
        frontend.infer(torch.zeros(1, 1, 132))


@pytest.mark.parametrize("frame_aux", [
    {},
    {"frames": torch.zeros(2, 8, 8, 3), "patches": torch.zeros(4, 1536)},
])
def test_a_frame_needs_exactly_one_of_frames_or_patches(frontend, frame_aux):
    frontend.set_prompt(aux=_aux())
    with pytest.raises(ValueError, match="exactly one of"):
        frontend.infer(torch.zeros(1, 1, 132), aux=frame_aux)


def test_patches_go_straight_through_and_the_noise_is_drawn(frontend):
    frontend.set_prompt(aux=_aux())
    patches = torch.zeros(4, 1536, dtype=torch.bfloat16)
    out = frontend.infer(torch.zeros(1, 1, 132), aux={"patches": patches})
    filled_patches, _, noise = frontend._frame.filled
    assert filled_patches is patches
    assert tuple(noise.shape) == (1, 40, 132) and noise.dtype is torch.bfloat16
    assert out.dtype is torch.float32


def test_a_supplied_noise_draw_is_used_and_its_shape_checked(frontend):
    frontend.set_prompt(aux=_aux())
    drawn = torch.full((1, 40, 132), 0.5)
    frontend.infer(torch.zeros(1, 1, 132), aux={"patches": torch.zeros(4, 1536)},
                   initial_noise=drawn)
    _, _, noise = frontend._frame.filled
    assert torch.allclose(noise.float(), drawn, atol=1e-2)
    with pytest.raises(ValueError, match="must end in"):
        frontend.infer(torch.zeros(1, 1, 132),
                       aux={"patches": torch.zeros(4, 1536)},
                       initial_noise=torch.zeros(1, 8, 132))


def test_host_frames_are_refused_where_the_transform_takes_their_address(frontend):
    frontend.set_prompt(aux=_aux())
    with pytest.raises(ValueError, match="Ascend device"):
        frontend.infer(torch.zeros(1, 1, 132),
                       aux={"frames": torch.zeros(2, 180, 320, 3,
                                                  dtype=torch.uint8)})


def test_the_state_and_action_halves_need_the_processor_first(frontend):
    with pytest.raises(RuntimeError, match="set_hf_processor"):
        frontend.normalize_state({})
    with pytest.raises(RuntimeError, match="set_hf_processor"):
        frontend.denormalize_action(torch.zeros(1, 40, 132), state_dict={})


def test_denormalising_a_relative_action_without_the_state_is_refused(frontend):
    frontend._decoder = object()
    with pytest.raises(ValueError, match="reference frame"):
        frontend.denormalize_action(torch.zeros(1, 40, 132))


def test_the_processor_builds_both_halves(frontend, monkeypatch):
    import importlib

    module = importlib.import_module(MODULE)
    monkeypatch.setattr(module, "StateEncoder",
                        lambda *a, **k: types.SimpleNamespace(tag="encoder"))
    monkeypatch.setattr(module, "ActionDecoder",
                        lambda *a, **k: types.SimpleNamespace(tag="decoder"))
    frontend.set_hf_processor(object())
    assert frontend._encoder.tag == "encoder"
    assert frontend._decoder.tag == "decoder"


def test_the_reported_tier_is_bf16_and_names_its_native_units(frontend):
    spec = frontend.precision_spec()
    assert spec["tier"] == "bf16" and spec["activations"] == "bf16"
    assert len(spec["native_kernels"]) == 3


def test_latency_stats_before_any_call_say_so(frontend):
    assert frontend.get_latency_stats() == {"calls": 0}
    frontend.set_prompt(aux=_aux())
    frontend.infer(torch.zeros(1, 1, 132), aux={"patches": torch.zeros(4, 1536)})
    stats = frontend.get_latency_stats()
    assert stats["calls"] == 1 and stats["median_ms"] >= 0.0
