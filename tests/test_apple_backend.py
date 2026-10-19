"""Apple routing, precision gates, prompt placement, and attention parity."""

import platform
import sys
from types import SimpleNamespace

import pytest
import torch

from flash_rt.hardware.mps.attention import MpsAttentionBackend
from flash_rt.hardware import detect_arch, resolve_pipeline_class


@pytest.mark.parametrize("machine,available,expected", [
    ("arm64", True, "apple_mps"), ("arm64", False, None), ("x86_64", True, None),
])
def test_detect_apple_requires_arm64_and_usable_mps(monkeypatch, machine, available, expected):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)
    if expected:
        assert detect_arch() == expected
    else:
        with pytest.raises(RuntimeError, match="accelerator"):
            detect_arch()


def test_cuda_keeps_priority_over_mps(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert detect_arch() == "rtx_sm120"


def test_apple_registry_is_pi05_torch_only():
    assert resolve_pipeline_class("pi05", "torch", "apple_mps").__name__ == "Pi05TorchFrontendMps"
    for config, framework in (("pi0", "torch"), ("groot", "torch"), ("pi05", "jax")):
        with pytest.raises(RuntimeError, match="no pipeline"):
            resolve_pipeline_class(config, framework, "apple_mps")


@pytest.mark.parametrize("kwargs", [
    {"precision": "fp8"}, {"precision": "int8"}, {"use_fp4": True},
    {"use_fp16": True}, {"use_fa4": True}, {"use_fp4_decoder": True},
    {"cache_frames": 2}, {"state_prompt_mode": "fixed"},
])
def test_unsupported_options_fail_before_checkpoint_or_extensions(kwargs):
    from flash_rt import load_model
    with pytest.raises(ValueError, match="Apple MPS"):
        load_model("/missing/checkpoint", hardware="apple_mps", **kwargs)


@pytest.mark.parametrize("precision", ["auto", "bf16"])
def test_public_api_skips_native_extensions(monkeypatch, precision):
    from flash_rt import _extensions, load_model
    from flash_rt.frontends.torch import pi05_mps as pi05

    def unexpected(*args, **kwargs):
        pytest.fail("Apple backend tried to load a CUDA/HIP extension")

    monkeypatch.setattr(_extensions, "require", unexpected)
    monkeypatch.setattr(pi05, "Pi05TorchFrontendMps", lambda checkpoint, **kwargs:
                        SimpleNamespace(checkpoint=checkpoint, kwargs=kwargs))
    result = load_model("checkpoint", hardware="apple_mps", precision=precision,
                        num_steps=3, action_dim=7)
    assert result.pipeline.kwargs == {"num_views": 2, "action_dim": 7, "num_steps": 3}


def test_prompt_embedding_can_stay_on_cpu(monkeypatch):
    from flash_rt.frontends.torch.pi05_checkpoint import _embed_prompt
    from flash_rt.utils import paligemma_tokenizer

    monkeypatch.setitem(sys.modules, "openpi.models.tokenizer", None)
    tokenizer = SimpleNamespace(bos_id=lambda: 1, Encode=lambda *args, **kwargs: [2, 3])
    monkeypatch.setattr(paligemma_tokenizer, "load_paligemma_sentencepiece", lambda: tokenizer)
    weight = torch.arange(110 * 8, dtype=torch.float32).reshape(110, 8).bfloat16()
    embeds, length = _embed_prompt("pick", weight, device="cpu")
    assert length == 4 and embeds.device.type == "cpu"
    torch.testing.assert_close(embeds, weight[[1, 2, 3, 108]] * (8 ** 0.5))


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("padded", [False, True])
def test_grouped_attention_matches_explicit_math_and_masks_padding(device, padded):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    generator = torch.Generator().manual_seed(14)
    q, k, v = (torch.randn(shape, generator=generator) for shape in ((2, 4, 8), (8, 2, 8), (8, 2, 8)))
    prefix, capacity = 3, 6
    if not padded:
        k, v = k[:prefix], v[:prefix]
    keep = torch.ones(k.shape[0], dtype=torch.bool)
    if padded:
        keep[prefix:capacity] = False
        k[prefix:capacity] = 10000
        v[prefix:capacity] = -10000
    keys = k[keep].repeat_interleave(2, 1).transpose(0, 1)
    values = v[keep].repeat_interleave(2, 1).transpose(0, 1)
    scores = q.transpose(0, 1) @ keys.transpose(-1, -2) / (q.shape[-1] ** .5)
    expected = (scores.softmax(-1) @ values).transpose(0, 1).reshape(2, -1)
    out = torch.empty(2, 32, device=device)
    result = MpsAttentionBackend().gqa(
        q.to(device), k.to(device), v.to(device),
        valid_prefix=prefix, prefix_capacity=capacity, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("site", ["vision", "prefix", "decode"])
def test_mps_bf16_attention_at_pi05_head_dimensions(site):
    from flash_rt.hardware.torch_reference import TorchAttentionBackend
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    generator = torch.Generator().manual_seed(27)
    rand = lambda *shape: torch.randn(shape, generator=generator).bfloat16()
    target, reference = MpsAttentionBackend(), TorchAttentionBackend()
    if site == "vision":
        qkv = rand(2, 256, 3 * 1152)
        expected = reference.vision(qkv)
        actual = target.vision(qkv.to("mps"))
    else:
        length = 517 if site == "prefix" else 10
        kv_length = 517 if site == "prefix" else 527
        q, k, v = rand(length, 8, 256), rand(kv_length, 1, 256), rand(kv_length, 1, 256)
        kwargs = dict(valid_prefix=517, prefix_capacity=712)
        expected = reference.gqa(q, k, v, **kwargs)
        actual = target.gqa(q.to("mps"), k.to("mps"), v.to("mps"), **kwargs)
    torch.testing.assert_close(actual.cpu(), expected, rtol=.03, atol=.003)
