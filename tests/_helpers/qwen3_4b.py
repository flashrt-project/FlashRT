"""Shared pieces of the Qwen3-4B BF16 prefill tests."""
from __future__ import annotations

import pytest
import torch


def frontend_cls():
    """The frontend registered for this device; skips when there is none."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    from flash_rt.hardware import detect_arch, resolve_pipeline_class
    arch = detect_arch()
    try:
        cls = resolve_pipeline_class("qwen3_4b", "torch", arch)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if torch.cuda.get_device_capability() != cls.TARGET.compute_capability:
        pytest.skip(f"{cls.__name__} is validated on SM{cls.TARGET.compute_capability}, "
                    f"not this device's SM{torch.cuda.get_device_capability()}")
    return cls


def cosine(a, b, chunk=1 << 24):
    """Cosine of two equal-shape tensors, accumulated in fp64 chunk by chunk
    (full-vocabulary logits do not fit twice in fp64 on a 24 GB card)."""
    a, b = a.reshape(-1), b.reshape(-1)
    dot = na = nb = 0.0
    for i in range(0, a.numel(), chunk):
        x, y = a[i:i + chunk].double(), b[i:i + chunk].double()
        dot += float((x * y).sum())
        na += float((x * x).sum())
        nb += float((y * y).sum())
    return dot / (na * nb) ** 0.5
