"""Ascend NPU pi05 verification helpers.

The NPU fusion work (L1–L3) changes op sequences and reduction orders,
so the correctness and performance gates below are the contract every
change must satisfy. They mirror the repo's Pi0.5 model-test protocol
(``tests/test_amd_pi05_model.py``): canonical inputs + pinned noise +
fp32 eager CPU reference, cosine gated, capture determinism asserted.

Two reference spaces are compared separately:

- ``raw`` (10, 32): the model's normalized action output — the most
  sensitive space, gate cos >= 0.9999 for fusion changes;
- robot (10, 7): unnormalized real actions behind norm_stats.

Same-process capture replay must be bit-identical; eager-vs-captured of
the *same* op sequence must also be bit-identical (capture must not
change numerics).
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import numpy as np

# Canonical recipe (shared with the reference generator and the model test).
CANONICAL_PROMPT = "pick up the cup"
CANONICAL_STATE = np.zeros(8, dtype=np.float32)
CHUNK = 10
ACTION_DIM_RAW = 32


def canonical_images(seed: int = 0, num_views: int = 2) -> dict:
    rng = np.random.RandomState(seed)
    keys = ("image", "wrist_image", "wrist_image_right")
    out = {
        keys[i]: rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for i in range(num_views)
    }
    return out


def pinned_noise(seed: int = 1) -> np.ndarray:
    return np.random.RandomState(seed).randn(CHUNK, ACTION_DIM_RAW).astype(
        np.float32)


def parity_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Official FlashRT judge on numpy arrays.

    Delegates to ``flash_rt.structures.gates.parity_metrics`` (the repo's
    single source of accuracy math) so the NPU gates reuse the same
    cosine / max_abs / p99_abs the structures layer ships, instead of a
    private third copy of the formula. Returns ``{"cosine", "max_abs",
    "p99_abs"}``; a fusion change must clear cosine >= 0.9999 with the
    error metrics available for localising a regression.
    """
    import torch
    from flash_rt.structures.gates import parity_metrics as _judge
    got = torch.as_tensor(np.asarray(a, dtype=np.float64))
    want = torch.as_tensor(np.asarray(b, dtype=np.float64))
    return _judge(got, want)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return parity_metrics(a, b)["cosine"]


def reference_path() -> Optional[pathlib.Path]:
    for var in ("FLASH_RT_NPU_PI05_REF", "PI05_NPU_REF"):
        v = os.environ.get(var)
        if v:
            p = pathlib.Path(v)
            if p.is_file():
                return p
    return None


def load_reference(p: pathlib.Path) -> dict:
    data = np.load(p)
    return {
        "raw": data["raw"].astype(np.float32),       # (10,32) fp32 normalized
        "tokens": data["tokens"],
        "noise_seed": int(data["noise_seed"]),
        "img_seed": int(data["img_seed"]),
    }


def make_reference_actions(fe, checkpoint_dir: pathlib.Path) -> dict:
    """fp32 CPU eager reference (normalized raw actions) for the canonical
    inputs — the numeric floor every BF16 captured change is gated against.
    Slow (a full CPU fp32 pass); run once and store with ``save_reference``.
    """
    fe.set_prompt(CANONICAL_PROMPT)
    obs = canonical_images()
    noise = pinned_noise()
    raw = fe.reference_actions(obs, noise)           # fp32 CPU, normalized
    return {"raw": np.asarray(raw, dtype=np.float32),
            "tokens": np.asarray(getattr(fe, "_current_tokens", []),
                                 dtype=np.int64),
            "noise_seed": 1, "img_seed": 0}


def save_reference(p: pathlib.Path, payload: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, raw=payload["raw"], tokens=payload["tokens"],
             noise_seed=payload["noise_seed"], img_seed=payload["img_seed"])
    print(f"reference saved: {p} ({payload['raw'].shape})")
