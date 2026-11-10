"""SM80 (A100 / A800) routing — dispatch table, detection, and the front door.

What this file protects:

  * ``detect_arch()`` maps compute capability 8.0 to ``"rtx_sm80"`` and
    leaves the Orin (8.7) and Ada (8.9) branches where they were;
  * the two SM80 registrations: Pi0.5 on the RTX frontend's BF16 pipeline,
    and the isomorphic Pi0.5 frontend (rollouts on the trainer's
    arithmetic);
  * the allowlist: datacenter Ampere has no FP8 tensor cores, so a config
    without an SM80-validated BF16 pipeline must be refused at resolution
    rather than fail at its first kernel launch;
  * ``load_model(config="pi05_isomorphic")`` redirects to direct
    construction on any machine.

No GPU, checkpoint or kernel build is needed.
"""

from __future__ import annotations

import pytest


def _patch_cuda(monkeypatch, capability):
    torch = pytest.importorskip("torch")
    try:
        import torch_npu  # noqa: F401
        pytest.skip("torch_npu present: the Ascend probe routes first")
    except Exception:
        pass
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)


@pytest.mark.parametrize("capability, arch", [
    ((8, 0), "rtx_sm80"),
    ((8, 7), "rtx_sm87"),
    ((8, 9), "rtx_sm89"),
])
def test_detect_arch_ampere_and_ada(monkeypatch, capability, arch):
    from flash_rt.hardware import detect_arch

    _patch_cuda(monkeypatch, capability)
    assert detect_arch() == arch


def test_detect_arch_still_refuses_sm86(monkeypatch):
    """Consumer Ampere (8.6) has no validated pipeline; 8.0 must not have opened the door for it."""
    from flash_rt.hardware import detect_arch

    _patch_cuda(monkeypatch, (8, 6))
    with pytest.raises(RuntimeError, match="unsupported GPU SM 8.6"):
        detect_arch()


def test_sm80_is_a_documented_arch_string():
    from flash_rt.hardware import detect_arch

    assert "rtx_sm80" in (detect_arch.__doc__ or "")


def test_pipeline_map_has_sm80_entries():
    from flash_rt.hardware import _PIPELINE_MAP

    assert _PIPELINE_MAP[("pi05", "torch", "rtx_sm80")] == (
        "flash_rt.frontends.torch.pi05_rtx", "Pi05TorchFrontendRtx")
    assert _PIPELINE_MAP[("pi05_isomorphic", "torch", "rtx_sm80")] == (
        "flash_rt.frontends.torch.pi05_isomorphic", "Pi05TorchFrontendIsomorphic")


@pytest.mark.parametrize("config", ["groot", "pi0", "groot_n17", "qwen3_vl"])
def test_sm80_refuses_configs_without_a_validated_pipeline(config):
    from flash_rt.hardware import resolve_pipeline_class

    with pytest.raises(RuntimeError, match="SM80"):
        resolve_pipeline_class(config, "torch", "rtx_sm80")


def test_isomorphic_frontend_imports_without_the_kernel_module():
    """The frontend runs on FlashRT-Train; importing it must not need FlashRT's own kernel build."""
    pytest.importorskip("torch")
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("pi05_isomorphic", "torch", "rtx_sm80")
    assert cls.__name__ == "Pi05TorchFrontendIsomorphic"


def test_load_model_redirects_the_isomorphic_config():
    pytest.importorskip("torch")
    from flash_rt.api import load_model

    with pytest.raises(NotImplementedError, match="Pi05TorchFrontendIsomorphic"):
        load_model("/nonexistent", config="pi05_isomorphic")


def test_isomorphic_frontend_requires_three_views(tmp_path):
    pytest.importorskip("torch")
    from flash_rt.frontends.torch.pi05_isomorphic import Pi05TorchFrontendIsomorphic

    with pytest.raises(ValueError, match="3 camera views"):
        Pi05TorchFrontendIsomorphic(tmp_path, num_views=2)
