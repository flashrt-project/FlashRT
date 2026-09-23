"""Routing and fail-closed contracts for the AMD RDNA 3.5 backend."""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace

import pytest


def _patch_rocm(monkeypatch, arch: str) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.15", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(gcnArchName=arch),
    )


def test_rdna35_pipeline_registration_is_lazy():
    from flash_rt.hardware import _PIPELINE_MAP, resolve_pipeline_class

    key = ("pi05", "torch", "amd_rdna35")
    assert _PIPELINE_MAP[key] == (
        "flash_rt.amd.frontends.torch.pi05_rdna35",
        "Pi05TorchFrontendAmdRdna35",
    )
    cls = resolve_pipeline_class(*key)
    assert cls.__name__ == "Pi05TorchFrontendAmdRdna35"


def test_rdna35_v1_has_no_temporal_cache_surface():
    from flash_rt.amd.frontends.torch.pi05_rdna35 import (
        Pi05TorchFrontendAmdRdna35,
    )
    from flash_rt.amd.models.pi05_rdna35.pipeline import Pi05PipelineRdna35

    frontend_parameters = inspect.signature(
        Pi05TorchFrontendAmdRdna35.__init__).parameters
    pipeline_parameters = inspect.signature(
        Pi05PipelineRdna35.forward_with_inputs).parameters
    assert "cache_frames" not in frontend_parameters
    assert "reuse_encoder_cache" not in pipeline_parameters
    assert not hasattr(Pi05TorchFrontendAmdRdna35, "reset_temporal_cache")
    assert not hasattr(Pi05PipelineRdna35, "record_decoder_graph")


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1151:sramecc-:xnack-"])
def test_detect_arch_maps_gfx1151(monkeypatch, arch):
    from flash_rt.hardware import detect_arch

    _patch_rocm(monkeypatch, arch)
    assert detect_arch() == "amd_rdna35"


def test_rdna35_does_not_alias_cdna4(monkeypatch):
    from flash_rt.hardware import detect_arch

    _patch_rocm(monkeypatch, "gfx1151")
    assert detect_arch() != "amd_cdna4"


def test_rdna35_frontend_import_does_not_require_native_extension(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"flash_rt.amd.flash_rt_amd_kernels", "flash_rt_kernels"}:
            raise AssertionError(f"RDNA 3.5 frontend imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    from flash_rt.hardware import resolve_pipeline_class

    cls = resolve_pipeline_class("pi05", "torch", "amd_rdna35")
    assert cls.__name__ == "Pi05TorchFrontendAmdRdna35"


def test_rdna35_reference_fallback_does_not_import_triton(monkeypatch):
    import builtins

    for name in (
        "FLASHRT_RDNA35_HIP_ROPE",
        "FLASHRT_RDNA35_HIP_DECODER",
        "FLASHRT_RDNA35_HIP_FFN_GATE_UP",
        "FLASHRT_RDNA35_HIP_ENCODER_FFN",
        "FLASHRT_RDNA35_HIP_LARGE_OPS",
        "FLASHRT_RDNA35_HIP_GQA",
        "FLASHRT_RDNA35_HIP_ENCODER_ATTN",
    ):
        monkeypatch.setenv(name, "0")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if (
            name == "triton"
            or name.startswith("triton.")
            or (name.startswith("flash_rt.") and ".triton" in name)
        ):
            raise AssertionError(f"RDNA 3.5 imported removed Triton code: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    import flash_rt.amd.hardware.rdna35 as rdna35

    assert rdna35.__all__ == [
        "Rdna35AttentionBackend",
        "Rdna35GemmBackend",
    ]
    attention = rdna35.Rdna35AttentionBackend(SimpleNamespace())
    assert not attention.hip_gqa
    assert not attention.hip_encoder_gqa


def test_load_model_rdna35_defaults_to_bf16_before_checkpoint(caplog):
    import torch
    import flash_rt

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        pytest.skip("requires a visible gfx1151 ROCm device")
    props = torch.cuda.get_device_properties(0)
    if getattr(props, "gcnArchName", "").split(":", 1)[0] != "gfx1151":
        pytest.skip("requires a visible gfx1151 ROCm device")

    with caplog.at_level(logging.WARNING, logger="flash_rt.api"):
        with pytest.raises(FileNotFoundError, match="safetensors"):
            flash_rt.load_model(
                "/nonexistent/flashrt-rdna35-checkpoint",
                framework="torch",
                config="pi05",
                hardware="amd_rdna35",
            )
    assert any("supports BF16 only" in record.getMessage()
               for record in caplog.records)


def test_checkpoint_profile_helpers(tmp_path):
    torch = pytest.importorskip("torch")
    from flash_rt.amd.frontends.torch.pi05_rdna35 import (
        Pi05TorchFrontendAmdRdna35,
        _build_time_embeddings,
    )

    (tmp_path / "config.json").write_text('{"action_horizon": 15}')
    assert Pi05TorchFrontendAmdRdna35._checkpoint_action_horizon(tmp_path) == 15
    stats = {"actions": {"q01": [-1.0, -1.0, 0.0, 0.0],
                         "q99": [1.0, 1.0, 0.0, 0.0]}}
    assert Pi05TorchFrontendAmdRdna35._resolve_action_dim(tmp_path, 2, stats) == 2

    schedule = _build_time_embeddings(5)
    assert schedule.shape == (5, 1024)
    assert schedule.dtype == torch.bfloat16
    assert not schedule[0].equal(schedule[1])


def test_rdna35_profile_matches_cdna_bf16_input_range():
    from flash_rt.amd.frontends.torch.pi05_rdna35 import (
        Pi05TorchFrontendAmdRdna35,
    )
    from flash_rt.amd.models.pi05_rdna35.pipeline import Pi05PipelineRdna35

    frontend = inspect.signature(Pi05TorchFrontendAmdRdna35.__init__).parameters
    pipeline = inspect.signature(Pi05PipelineRdna35.__init__).parameters
    assert "max_prompt_len" in frontend
    assert "prompt_capacity" not in frontend
    assert "max_prompt_len" in pipeline

    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            from flash_rt.amd.frontends.torch.pi05_rdna35 import (
                _build_time_embeddings,
            )
            _build_time_embeddings(invalid)


@pytest.mark.parametrize("dim", [1, 3, 4])
def test_constant_action_channels_are_preserved(tmp_path, dim):
    from flash_rt.amd.frontends.torch.pi05_rdna35 import Pi05TorchFrontendAmdRdna35
    stats = {"actions": {"q01": [-1., -1., .5, .5], "q99": [1., 1., .5, .5]}}
    assert Pi05TorchFrontendAmdRdna35._resolve_action_dim(tmp_path, dim, stats) == dim


@pytest.mark.parametrize("dim", [None, 0, -1, True, 1.5, 33])
def test_output_schema_must_be_explicit_and_valid(tmp_path, dim):
    from flash_rt.amd.frontends.torch.pi05_rdna35 import Pi05TorchFrontendAmdRdna35
    stats = {"actions": {"q01": [0.] * 32, "q99": [0.] * 32}}
    with pytest.raises(ValueError, match="Set action_dim"):
        Pi05TorchFrontendAmdRdna35._resolve_action_dim(tmp_path, dim, stats)


def test_checkpoint_robot_schema_and_override(tmp_path):
    from flash_rt.amd.frontends.torch.pi05_rdna35 import Pi05TorchFrontendAmdRdna35
    stats = {"actions": {"q01": [0.] * 32, "q99": [0.] * 32}}
    (tmp_path / "config.json").write_text('{"action_dim": 32, "output_action_dim": 7}')
    resolve = Pi05TorchFrontendAmdRdna35._resolve_action_dim
    assert resolve(tmp_path, None, stats) == 7
    assert resolve(tmp_path, 8, stats) == 8


def test_import_without_gpu_libraries():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", """
import torch
import ctypes
original = ctypes.CDLL
def guarded(name, *args, **kwargs):
    if any(part in str(name) for part in ('amdhip', 'libcuda', 'libcudart')):
        raise AssertionError('GPU runtime initialized at import: ' + str(name))
    return original(name, *args, **kwargs)
ctypes.CDLL = guarded
from flash_rt.hardware import resolve_pipeline_class
resolve_pipeline_class('pi05', 'torch', 'amd_rdna35')
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_api_forwards_robot_action_schema(monkeypatch):
    import sys
    import types
    import flash_rt
    import flash_rt.hardware as hardware
    captured = {}
    class Frontend:
        def __init__(self, checkpoint, num_views=2, hardware=None, use_fp8=False, action_dim=None):
            captured['action_dim'] = action_dim
    monkeypatch.setitem(sys.modules, 'flash_rt.amd.flash_rt_amd_kernels', types.ModuleType('flash_rt.amd.flash_rt_amd_kernels'))
    monkeypatch.setattr(hardware, 'resolve_pipeline_class', lambda *a, **kw: Frontend)
    flash_rt.load_model('unused', config='pi05', framework='torch', hardware='amd_rdna35', action_dim=3, use_fp8=False)
    assert captured['action_dim'] == 3


def test_public_actions_keep_constant_last_channel(monkeypatch, tmp_path):
    import numpy as np
    import torch
    from flash_rt.amd.frontends.torch.pi05_rdna35 import Pi05TorchFrontendAmdRdna35
    frontend = object.__new__(Pi05TorchFrontendAmdRdna35)
    frontend.norm_stats = {'actions': {'q01': [-1., -1., .5], 'q99': [1., 1., .5]}}
    frontend.action_dim = frontend._resolve_action_dim(tmp_path, 3, frontend.norm_stats)
    frontend._prompt_len = 1
    frontend.dtype = torch.bfloat16
    frontend._noise_buf = torch.zeros(2, 32, dtype=torch.bfloat16)
    frontend._image_buf = torch.empty(0)
    frontend.latency_records = []
    frontend._fill_images = lambda obs: None
    frontend.forward_with_fixed_noise = lambda *a: torch.zeros(1, 2, 32, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    result = frontend.infer({}, noise=np.zeros((2, 32)))
    assert result['actions'].shape == (2, 3)
    np.testing.assert_allclose(result['actions'][:, 2], .5, atol=1e-6)
