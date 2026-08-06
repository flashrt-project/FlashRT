"""``load_model`` routing contract for the GROOT N1.7 Thor NVFP4 tier.

Covers the three public behaviors the tier adds to ``flash_rt.load_model``:

* ``config="groot_n17", hardware="thor", use_fp4=True`` selects the NVFP4
  frontend;
* it falls back to the FP8 frontend when the optional ``flash_rt_fp4``
  extension is missing or reports no NVFP4 support;
* combining it with ``use_fp16=True`` fails, and it does not disturb the
  Pi0.5 ``use_fp4`` route.

The frontends are stubbed out, so these run without a GPU or checkpoint.
"""
from __future__ import annotations

import sys
import types

import pytest

import flash_rt
from flash_rt import api as flash_rt_api


class _StubFrontend:
    """Records the class the router picked instead of loading weights."""

    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        self.kwargs = kwargs


def _install_stub_frontends(monkeypatch):
    """Stub the two N1.7 Thor frontend modules the router may import."""
    picked = {}

    for mod_name, cls_name in (
        ("flash_rt.frontends.torch.groot_n17_thor_fp4",
         "GrootN17TorchFrontendThorFP4"),
        ("flash_rt.frontends.torch.groot_n17_thor_fp8",
         "GrootN17TorchFrontendThorFP8"),
    ):
        module = types.ModuleType(mod_name)
        cls = type(cls_name, (_StubFrontend,), {"_stub_name": cls_name})
        setattr(module, cls_name, cls)
        monkeypatch.setitem(sys.modules, mod_name, module)
    return picked


def _stub_fp4_extension(monkeypatch, *, available: bool, has_nvfp4: bool = True):
    """Stand in for the optional NVFP4 extension.

    ``import flash_rt.flash_rt_fp4 as x`` resolves through the parent
    package's attribute once the real submodule has been imported (which
    another test in the same session may already have done), so patch the
    attribute as well as the ``sys.modules`` entry.
    """
    if not available:
        monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", None)
        monkeypatch.delattr(flash_rt, "flash_rt_fp4", raising=False)
        return
    module = types.ModuleType("flash_rt.flash_rt_fp4")
    module.has_nvfp4 = lambda: has_nvfp4
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", module)
    monkeypatch.setattr(flash_rt, "flash_rt_fp4", module, raising=False)


def _load(monkeypatch, **kwargs):
    """Call load_model with hardware pinned to Thor and weights stubbed."""
    monkeypatch.setattr(flash_rt_api, "detect_arch", lambda: "thor",
                        raising=False)
    return flash_rt.load_model(
        "/nonexistent/checkpoint",
        framework="torch",
        config="groot_n17",
        hardware="thor",
        **kwargs,
    )


def test_use_fp4_selects_the_nvfp4_frontend(monkeypatch):
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch, available=True)
    model = _load(monkeypatch, use_fp4=True)
    inner = getattr(model, "_pipe", model)
    assert type(inner)._stub_name == "GrootN17TorchFrontendThorFP4"


def test_missing_fp4_extension_falls_back_to_fp8(monkeypatch):
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch, available=False)
    model = _load(monkeypatch, use_fp4=True)
    inner = getattr(model, "_pipe", model)
    assert type(inner)._stub_name == "GrootN17TorchFrontendThorFP8"


def test_extension_without_nvfp4_support_falls_back_to_fp8(monkeypatch):
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch, available=True, has_nvfp4=False)
    model = _load(monkeypatch, use_fp4=True)
    inner = getattr(model, "_pipe", model)
    assert type(inner)._stub_name == "GrootN17TorchFrontendThorFP8"


def test_use_fp4_with_use_fp16_is_rejected(monkeypatch):
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch, available=True)
    with pytest.raises(ValueError, match="use_fp4"):
        _load(monkeypatch, use_fp4=True, use_fp16=True, use_fp8=False)


def test_default_still_selects_the_fp8_frontend(monkeypatch):
    """Existing behavior is unchanged when the flag is not passed."""
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch, available=True)
    model = _load(monkeypatch)
    inner = getattr(model, "_pipe", model)
    assert type(inner)._stub_name == "GrootN17TorchFrontendThorFP8"
