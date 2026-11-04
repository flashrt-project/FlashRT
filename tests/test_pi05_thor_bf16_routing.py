"""The opt-in BF16 route must not load a CUDA module on other paths."""

import sys
from types import ModuleType

import pytest

from flash_rt.api import load_model


def test_thor_bf16_routes_only_when_explicit(monkeypatch):
    module = ModuleType("flash_rt.frontends.torch.pi05_thor_bf16")
    checkpoint = object()
    frontend = object()

    def create(value):
        assert value is checkpoint
        return frontend

    module.Pi05TorchFrontendThorBf16 = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = load_model(checkpoint, hardware="thor", precision="bf16", num_views=1, use_fp8=False)
    assert model.pipeline is frontend


@pytest.mark.parametrize(
    "options, message",
    [
        ({"use_fp8": True}, "requires use_fp8=False"),
        ({"use_fp16": True}, "requires use_fp8=False"),
        ({"num_views": 2}, "requires one view"),
        ({"action_horizon": 16}, "requires one view"),
        ({"state_prompt_mode": "fixed"}, "requires one view"),
        ({"hardware": "rtx_sm120"}, "is unavailable"),
        ({"framework": "jax"}, "is unavailable"),
    ],
)
def test_thor_bf16_rejects_incompatible_options(options, message):
    defaults = {"hardware": "thor", "precision": "bf16", "num_views": 1, "use_fp8": False}
    with pytest.raises(ValueError, match=message):
        load_model("unused", **(defaults | options))
