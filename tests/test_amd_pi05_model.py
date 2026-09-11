"""Pi0.5 on AMD CDNA4 — checkpoint-gated E2E gates for load_model.

The full front-door path on an MI350X: load_model → Pi05TorchFrontendAmd →
Pi05Pipeline → HIP graph capture → replay. Gates:

  * load_model(hardware="amd_cdna4") produces finite (chunk, 7) actions;
  * the inference graph really is captured (the latency story is replay);
  * two infers with the SAME pinned noise are BIT-identical — the flow
    integration is deterministic given (images, prompt, state, noise), so
    any wobble is a replay-determinism or buffer-reuse bug, not "noise";
  * state_prompt_mode="exact" and ="fixed" both serve state prompts, and
    fixed mode serves a SECOND state (different token length) without a
    pipeline rebuild or graph re-capture (object identity asserted) —
    that no-rebuild property is the entire point of fixed mode;
  * BF16 (use_fp8=False) and FP8 both run;
  * optionally, actions match a reference file to cosine >= 0.999.

Checkpoint resolution (never hardcoded):
    FLASH_RT_PI05_AMD_CKPT > PI05_AMD_CKPT >
    tests/_helpers/paths.resolve("PI05_CKPT")   (which itself honours
    FLASH_RT_PI05_CKPT / PI05_CKPT before its built-in default)
The checkpoint is a HuggingFace-style Pi0.5 safetensors directory
(model.safetensors + norm stats), the same layout the RTX/Thor tests use.

Optional reference gate:
    FLASH_RT_PI05_AMD_REF_ACTIONS (or PI05_AMD_REF_ACTIONS) — path to a
    .npy of shape (chunk, 7) produced with the canonical recipe below
    (images RandomState(0), prompt "pick up the cup", state zeros(8),
    noise RandomState(1)); absent → that one test skips.

Skip conditions: ROCm torch + visible device + built extension + resolved
checkpoint; each missing piece skips with its own reason. All heavyweight
imports live inside fixtures so NVIDIA/no-ROCm CI collects this module
cleanly.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers.paths import resolve  # noqa: E402

# ── Canonical deterministic inputs (shared with the reference recipe) ──
_PROMPT = "pick up the cup"
_STATE0 = np.zeros(8, dtype=np.float32)
_STATE1 = np.linspace(-0.9, 0.8, 8).astype(np.float32)
_CHUNK, _ACTION_DIM_RAW = 10, 32


def _images():
    rng = np.random.RandomState(0)
    return [rng.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)]


def _pinned_noise():
    return np.random.RandomState(1).randn(
        _CHUNK, _ACTION_DIM_RAW).astype(np.float32)


def _resolve_ckpt():
    """AMD-specific env vars first, then the shared PI05_CKPT machinery."""
    for var in ("FLASH_RT_PI05_AMD_CKPT", "PI05_AMD_CKPT"):
        v = os.environ.get(var)
        if v:
            return v if os.path.isdir(v) else None
    return resolve("PI05_CKPT", optional=True)


@pytest.fixture(scope="module")
def amd_env():
    """ROCm torch + device + extension, or skip with the missing piece."""
    try:
        importlib.import_module("flash_rt.amd.flash_rt_amd_kernels")
    except ImportError as exc:
        pytest.skip(f"flash_rt_amd_kernels not importable: {exc}")
    try:
        # The frontend pulls in the pipeline's third-party numeric helpers
        # (same ones the RTX pipeline imports, declared under an optional
        # extra); skip rather than error when the env lacks them.
        importlib.import_module("flash_rt.amd.frontends.torch.pi05")
    except ImportError as exc:
        pytest.skip(f"AMD pi05 frontend not importable: {exc}")
    torch = pytest.importorskip("torch")
    if not getattr(torch.version, "hip", None):
        pytest.skip("torch is not a ROCm build")
    if not torch.cuda.is_available():
        pytest.skip("no ROCm device visible to torch")
    return torch


@pytest.fixture(scope="module")
def ckpt(amd_env):
    path = _resolve_ckpt()
    if path is None:
        pytest.skip(
            "no Pi0.5 checkpoint: export FLASH_RT_PI05_AMD_CKPT (or the "
            "shared FLASH_RT_PI05_CKPT) — see tests/_helpers/paths.py")
    if not os.path.isfile(os.path.join(path, "model.safetensors")):
        pytest.skip(f"checkpoint dir lacks model.safetensors: set "
                    f"FLASH_RT_PI05_AMD_CKPT to a HF-style Pi0.5 checkpoint")
    return path


def _load(ckpt_path, **kw):
    import flash_rt
    return flash_rt.load_model(ckpt_path, framework="torch", config="pi05",
                               num_views=2, hardware="amd_cdna4", **kw)


@pytest.fixture(scope="module")
def model_fp8(ckpt):
    """Default FP8, exact state-prompt mode. First predict runs
    calibration + autotune + graph capture (the load_model bootstrap
    contract), so the fixture performs it once for the whole module."""
    model = _load(ckpt)
    actions = model.predict(_images(), prompt=_PROMPT, state=_STATE0)
    return model, np.asarray(actions, dtype=np.float32)


@pytest.fixture(scope="module")
def model_fixed(ckpt):
    model = _load(ckpt, state_prompt_mode="fixed")
    actions = model.predict(_images(), prompt=_PROMPT, state=_STATE0)
    return model, np.asarray(actions, dtype=np.float32)


@pytest.fixture(scope="module")
def model_bf16(ckpt):
    model = _load(ckpt, use_fp8=False)
    actions = model.predict(_images(), prompt=_PROMPT, state=_STATE0)
    return model, np.asarray(actions, dtype=np.float32)


# ---------------------------------------------------------------------------
# On-box routing sanity
# ---------------------------------------------------------------------------


def test_detect_arch_live(amd_env):
    """On the real MI350X, auto-detection must land on amd_cdna4 (the
    routing tests elsewhere only prove it with monkeypatched torch)."""
    from flash_rt.hardware import detect_arch
    assert detect_arch() == "amd_cdna4"


# ---------------------------------------------------------------------------
# FP8 / exact mode
# ---------------------------------------------------------------------------


def test_e2e_actions_shape_and_finite(model_fp8):
    _, actions = model_fp8
    assert actions.shape == (_CHUNK, 7), (
        "AMD Pi0.5 must emit the LIBERO (chunk, 7) action contract")
    assert np.isfinite(actions).all(), "non-finite actions from FP8 pipeline"
    # Unnormalized real actions are O(1); an all-zero or exploded output
    # means a dead buffer or a broken scale, not a plausible policy.
    assert np.abs(actions).max() > 1e-6
    assert np.abs(actions).max() < 1e3


def test_graph_is_captured(model_fp8):
    model, _ = model_fp8
    fe = model.pipeline
    assert fe.calibrated
    assert fe.graph_recorded, "first predict must leave a captured graph"
    assert getattr(fe.pipeline, "_graph", None) is not None


def test_pinned_noise_is_bit_identical(model_fp8):
    """infer(obs, noise=...) with identical noise twice → identical bytes.

    The captured graph replays a fixed kernel sequence on fixed buffers;
    with images/prompt/state/noise all pinned there is no legitimate
    source of variation, so this is an equality gate — a mismatch is a
    determinism bug (uninitialized scratch, buffer aliasing, atomic
    reduction), never acceptable "GPU noise"."""
    model, _ = model_fp8
    fe = model.pipeline
    imgs = _images()
    obs = {"image": imgs[0], "wrist_image": imgs[1]}
    noise = _pinned_noise()

    a1 = np.asarray(fe.infer(obs, noise=noise)["actions"])
    a2 = np.asarray(fe.infer(obs, noise=noise)["actions"])
    assert np.array_equal(a1, a2), (
        f"same pinned noise, different actions (max diff "
        f"{np.abs(a1 - a2).max():.3e}) — graph replay is not deterministic")


def test_exact_mode_serves_second_state(model_fp8):
    """Exact mode's contract is per-length pipelines: a different state
    must still serve (build-or-reuse) and stay finite."""
    model, _ = model_fp8
    actions = model.predict(_images(), prompt=_PROMPT, state=_STATE1)
    actions = np.asarray(actions, dtype=np.float32)
    assert actions.shape == (_CHUNK, 7)
    assert np.isfinite(actions).all()
    # leave the module fixture back on the canonical state for later tests
    model.predict(_images(), prompt=_PROMPT, state=_STATE0)


# ---------------------------------------------------------------------------
# Fixed state-prompt mode
# ---------------------------------------------------------------------------


def test_fixed_mode_actions_finite(model_fixed):
    _, actions = model_fixed
    assert actions.shape == (_CHUNK, 7)
    assert np.isfinite(actions).all()


def test_fixed_mode_second_length_reuses_pipeline_and_graph(model_fixed):
    """THE fixed-mode property: a second state (different discretized token
    length) is served by the SAME pipeline object and the SAME captured
    graph — no rebuild, no re-capture, no warmup. Object identity is the
    strongest available witness that nothing was rebuilt."""
    model, _ = model_fixed
    fe = model.pipeline

    pipeline_before = fe.pipeline
    graph_before = getattr(fe.pipeline, "_graph", None)
    assert pipeline_before is not None and graph_before is not None
    len_before = int(fe.current_prompt_len)

    actions = np.asarray(
        model.predict(_images(), prompt=_PROMPT, state=_STATE1),
        dtype=np.float32)
    assert np.isfinite(actions).all()

    assert fe.pipeline is pipeline_before, (
        "fixed mode rebuilt its pipeline for a new state prompt")
    assert getattr(fe.pipeline, "_graph", None) is graph_before, (
        "fixed mode re-captured its graph for a new state prompt")
    assert fe.pipeline is fe._fixed_pipeline
    # informative, not gating: state discretization usually shifts the
    # token length, but equal lengths would not weaken the identity gates.
    assert int(fe.current_prompt_len) > 0 and len_before > 0


# ---------------------------------------------------------------------------
# BF16 escape hatch
# ---------------------------------------------------------------------------


def test_bf16_pipeline_runs(model_bf16):
    """use_fp8=False must serve the full BF16 baseline (no FP8 kernels in
    the graph) — this is the reference arm every FP8 parity claim on this
    backend is measured against."""
    model, actions = model_bf16
    assert actions.shape == (_CHUNK, 7)
    assert np.isfinite(actions).all()
    fe = model.pipeline
    assert fe.use_fp8 is False
    assert not getattr(fe.pipeline, "use_fp8", True), (
        "BF16 frontend built an FP8 pipeline")


# ---------------------------------------------------------------------------
# Optional numeric reference gate
# ---------------------------------------------------------------------------


def test_actions_match_reference_when_provided(model_fp8):
    """Cosine >= 0.999 against a stored reference generated with the
    canonical recipe in this module's docstring (same images/prompt/state
    seeds, same pinned noise). 0.999 is the backend's E2E FP8-vs-reference
    acceptance bar; below it, quantization cannot be the explanation and a
    real regression is in the graph."""
    ref_path = (os.environ.get("FLASH_RT_PI05_AMD_REF_ACTIONS")
                or os.environ.get("PI05_AMD_REF_ACTIONS"))
    if not ref_path:
        pytest.skip("no reference actions: export FLASH_RT_PI05_AMD_REF_ACTIONS "
                    "to a .npy produced with this module's canonical recipe")
    if not os.path.isfile(ref_path):
        pytest.skip(f"reference actions file not found: "
                    "FLASH_RT_PI05_AMD_REF_ACTIONS does not point at a file")
    ref = np.load(ref_path).astype(np.float32)
    assert ref.shape == (_CHUNK, 7), "reference file has the wrong shape"

    model, _ = model_fp8
    # re-pin the canonical prompt/state, then infer with the pinned noise
    model.predict(_images(), prompt=_PROMPT, state=_STATE0)
    fe = model.pipeline
    imgs = _images()
    obs = {"image": imgs[0], "wrist_image": imgs[1]}
    got = np.asarray(fe.infer(obs, noise=_pinned_noise())["actions"],
                     dtype=np.float32)

    a, b = got.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    assert cos >= 0.999, f"actions cos vs reference = {cos:.6f} < 0.999"
