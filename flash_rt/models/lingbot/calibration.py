"""LingBot-VLA — static FP8 activation calibration.

Replaces the per-call dynamic max-reduce in ``linear_fp8`` with a
pre-computed scale read from a JSON table. Two effects:

1. **Determinism** — the per-call ``quantize_fp8_device`` writes a
   device-side ``act_scale`` whose value can drift slightly between
   runs depending on cuBLAS state at warm-up time. The Thor SM110
   FP8 GEMM heuristic is sensitive to that drift, occasionally
   picking a tactic that produces NaN. Static scale removes the
   freedom.
2. **Latency** — skipping the per-call absmax reduce saves ~5-10 μs
   × thousands of GEMMs per inference.

Usage::

    from flash_rt.models.lingbot import calibration as calib

    # Offline: record activation max(abs) per site.
    with calib.calibration_recorder() as stats:
        sample_actions(...)
    calib.save_calibration(stats, "lingbot_thor_static.json")

    # Production: load scales (one-time at startup), enable, run.
    scales = calib.load_calibration("lingbot_thor_static.json",
                                    device=torch.device("cuda"))
    calib.set_static_scales(scales)
    actions = sample_actions(...)
    calib.set_static_scales(None)      # clear when done

The state is held in module-level singletons (``_CALIB_STATS`` and
``_STATIC_SCALES``); ``linear_fp8`` consults them via ``is_calibrating()``,
``record_max_abs()``, and ``get_static_scale()``. When a ``site_id`` is
``None`` (legacy callers / sites not yet wired) both paths are no-ops
and ``linear_fp8`` falls back to dynamic quantize.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

# FP8 E4M3 max representable magnitude. scale = max_abs / FP8_E4M3_MAX.
FP8_E4M3_MAX = 448.0

# Floor to avoid div-by-zero when a site never sees activation (e.g.,
# masked-out cond) — pick a tiny scale so the descale path is still finite.
DEFAULT_SCALE_FLOOR = 1e-7

# Module-level singletons. None means feature off.
_CALIB_STATS: "dict[str, float] | None" = None
_STATIC_SCALES: "dict[str, torch.Tensor] | None" = None
_LOCK = threading.Lock()


def is_calibrating() -> bool:
    """linear_fp8 calls this once per invocation to decide whether to record."""
    return _CALIB_STATS is not None


def record_max_abs(site_id: "str | None", x: torch.Tensor) -> None:
    """Update running ``max(abs(x))`` for ``site_id`` while calibrating.

    Synchronizes (``.item()``) — this path is only taken during the
    offline calibration run, never during production inference.
    """
    if _CALIB_STATS is None or site_id is None:
        return
    v = float(x.detach().abs().max().item())
    prev = _CALIB_STATS.get(site_id, 0.0)
    if v > prev:
        _CALIB_STATS[site_id] = v


@contextmanager
def calibration_recorder() -> "Iterator[dict[str, float]]":
    """Enable per-site activation max(abs) recording for the duration of
    the with-block. Yields the stats dict (mutated live; also returned
    after the block for save_calibration)."""
    global _CALIB_STATS
    with _LOCK:
        if _CALIB_STATS is not None:
            raise RuntimeError("calibration_recorder already active")
        stats: "dict[str, float]" = {}
        _CALIB_STATS = stats
    try:
        yield stats
    finally:
        with _LOCK:
            _CALIB_STATS = None


def save_calibration(stats: "dict[str, float]", path) -> None:
    """Dump max(abs) per site to JSON. The scale conversion is deferred
    to load time so the JSON stays human-inspectable."""
    payload = {
        "version": 1,
        "format": "max_abs_per_site",
        "fp8_max": FP8_E4M3_MAX,
        "stats": {k: float(v) for k, v in sorted(stats.items())},
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_calibration(path, device: torch.device) -> "dict[str, torch.Tensor]":
    """Read JSON, return dict of ``site_id`` → 1-element fp32 device
    tensor holding ``max_abs / 448``. Tensors are persistent (caller
    holds them for the process lifetime, typically via ``set_static_scales``).
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != 1:
        raise ValueError(f"unsupported calibration version: {payload.get('version')}")
    if payload.get("format") != "max_abs_per_site":
        raise ValueError(
            f"unsupported calibration format: {payload.get('format')}")
    fp8_max = float(payload.get("fp8_max", FP8_E4M3_MAX))
    out: "dict[str, torch.Tensor]" = {}
    for sid, max_abs in payload["stats"].items():
        max_abs = max(float(max_abs), DEFAULT_SCALE_FLOOR)
        scale = max_abs / fp8_max
        out[sid] = torch.tensor([scale], dtype=torch.float32, device=device)
    return out


def set_static_scales(scales: "dict[str, torch.Tensor] | None") -> None:
    """Install (or clear) the static-scale registry consulted by
    linear_fp8. Pass ``None`` to revert to dynamic per-call quantize.
    """
    global _STATIC_SCALES
    with _LOCK:
        _STATIC_SCALES = scales


def get_static_scale(site_id: "str | None") -> "torch.Tensor | None":
    """Look up the static scale for ``site_id``. Returns ``None`` when
    no calibration is loaded OR when ``site_id`` is unknown — in either
    case ``linear_fp8`` reverts to the dynamic quantize path."""
    if _STATIC_SCALES is None or site_id is None:
        return None
    return _STATIC_SCALES.get(site_id)


def has_static_scales() -> bool:
    """Quick check used by tests."""
    return _STATIC_SCALES is not None


__all__ = [
    "FP8_E4M3_MAX",
    "calibration_recorder",
    "get_static_scale",
    "has_static_scales",
    "is_calibrating",
    "load_calibration",
    "record_max_abs",
    "save_calibration",
    "set_static_scales",
]
