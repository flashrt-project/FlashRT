"""Pi0.5 NPU backend end-to-end gates (skip unless NPU + checkpoint exist).

Exercises the real public door: ``flash_rt.load_model(config="pi05",
hardware="npu")`` → ``set_prompt`` (graph capture) → ``infer``. Gates:

- routing returns the NPU frontend;
- actions finite, shape (chunk, 7);
- pinned-noise replays bit-identical (captured-graph determinism);
- **fp32 reference gate**: normalized raw actions (10,32) cosine >= 0.9999
  against the golden file produced by ``scripts/npu/make_pi05_reference.py``
  (same canonical inputs + pinned noise) — the numeric floor every BF16
  fusion change must hold;
- captured BF16 median under a regression tripwire.

Checkpoint resolution follows the repo's shared convention (see
``tests/_helpers/paths.py``): ``FLASH_RT_PI05_CKPT`` > ``PI05_CKPT``,
and the checkpoint tests skip when neither resolves. Reference gate is driven
by ``FLASH_RT_NPU_PI05_REF`` (or ``PI05_NPU_REF``) pointing at a
reference ``.npz``.
"""

import os
import pathlib

import numpy as np
import pytest

from flash_rt.npu import verify


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        import torch
        return bool(torch.npu.is_available())
    except Exception:
        return False


def _resolve_pi05_ckpt() -> pathlib.Path | None:
    """The repo's shared resolver, exactly as the AMD tests use it."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from _helpers.paths import resolve
    shared = resolve("PI05_CKPT", optional=True)
    if shared:
        return pathlib.Path(shared)
    return None


def _checkpoint() -> pathlib.Path | None:
    ck = _resolve_pi05_ckpt()
    if ck is None or not (ck / "model.safetensors").is_file():
        return None
    return ck


_npu = _npu_available()
_ckpt = _checkpoint()
_ref = verify.reference_path()
requires_npu = pytest.mark.skipif(not _npu, reason="no usable Ascend NPU")
requires_ckpt = pytest.mark.skipif(
    _ckpt is None,
    reason="pi05 checkpoint not present (set FLASH_RT_PI05_CKPT)")


@pytest.fixture(scope="module")
def npu_model():
    """load_model on NPU, captured graph built for the canonical prompt."""
    import flash_rt
    model = flash_rt.load_model(str(_ckpt), config="pi05",
                                framework="torch", hardware="npu",
                                num_views=2)
    model.pipeline.set_prompt(verify.CANONICAL_PROMPT)
    return model


@requires_npu
def test_detect_arch_is_npu():
    from flash_rt.hardware import detect_arch
    assert detect_arch() == "npu"


@requires_npu
@requires_ckpt
def test_pi05_actions_shape_finite_and_deterministic(npu_model):
    fe = npu_model.pipeline
    obs = verify.canonical_images()
    noise = verify.pinned_noise()
    a = np.asarray(fe.infer(obs, noise=noise)["actions"])
    b = np.asarray(fe.infer(obs, noise=noise)["actions"])
    assert a.shape == (verify.CHUNK, 7)
    assert np.isfinite(a).all()
    assert np.array_equal(a, b), "pinned-noise replay must be bit-identical"


@requires_npu
@requires_ckpt
def test_pi05_captured_median_latency_tripwire(npu_model):
    fe = npu_model.pipeline
    obs = verify.canonical_images()
    for _ in range(5):
        fe.infer(obs)
    stats = fe.get_latency_stats()
    assert "p50_ms" in stats
    # Optimized captured BF16 full-frame median ~86 ms on 910B4. 350 ms is a
    # regression tripwire (re-enabling eager host dispatch or dropping back
    # to the reference op sequence blows past it), not a perf assertion.
    assert stats["p50_ms"] < 350.0, f"captured median {stats['p50_ms']:.0f} ms"


@requires_npu
@requires_ckpt
def test_pi05_actions_match_fp32_reference(npu_model):
    """Normalized raw actions (10,32) cosine >= 0.9999 vs the golden fp32
    CPU eager reference (canonical recipe, pinned noise). This is the gate
    every BF16 fusion change must hold: a regression lands at ~1e-3 cosine,
    far below 0.9999."""
    if _ref is None:
        pytest.skip("no reference file: set FLASH_RT_NPU_PI05_REF (produce "
                    "with scripts/npu/make_pi05_reference.py)")
    ref = verify.load_reference(_ref)
    fe = npu_model.pipeline
    obs = verify.canonical_images()
    noise = verify.pinned_noise()
    raw = np.asarray(fe.infer(obs, noise=noise)["raw_actions"],
                     dtype=np.float32)
    m = verify.parity_metrics(raw, ref["raw"])
    assert m["cosine"] >= 0.9999, (
        f"raw-actions cos vs fp32 reference = {m['cosine']:.6f} < 0.9999 "
        f"(max_abs={m['max_abs']:.3e} p99_abs={m['p99_abs']:.3e})")
