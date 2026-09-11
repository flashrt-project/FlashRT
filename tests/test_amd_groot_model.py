"""GROOT N1.7 on AMD CDNA4 — checkpoint-gated E2E gates for the frontend.

The full path on an MI350X: :class:`GrootN17TorchFrontendAmd` →
``set_prompt`` (FP8 kernel backbone: ViT → DeepStack → truncated LLM → VL
self-attn, plus the baked FP8 alphas) → ``infer`` (bf16 DiT action head
replayed from a captured graph). Gates:

  * the frontend constructs and reports the AMD class (not a silently
    substituted RTX/Thor base);
  * ``set_prompt`` really ran the FP8 backbone — backbone features of the
    documented shape/dtype AND baked per-layer FP8 alphas for every stage
    (features alone would also appear if a bf16 shadow path had run);
  * ``infer`` returns finite normalized actions of shape
    ``(1, action_horizon, 132)``, neither dead nor exploded;
  * two infers with the SAME pinned ``initial_noise`` are BIT-identical —
    the flow integration is deterministic given (backbone, state, noise),
    so any wobble is a replay-determinism or buffer-reuse bug, never
    acceptable "GPU noise";
  * a different pinned noise changes the output (proves the noise input is
    actually consumed and the equality gate above is not vacuous);
  * optionally, denormalized actions match a reference fixture to
    cosine >= 0.999.

Inputs come from environment variables ONLY — nothing is hardcoded:

    FLASH_RT_GROOT_N17_AMD_CKPT   (or bare GROOT_N17_AMD_CKPT)
        GROOT N1.7 checkpoint directory (must contain statistics.json).
    FLASH_RT_GROOT_N17_AMD_AUX    (or bare GROOT_N17_AMD_AUX)
        torch.load-able bundle of HF-derived setup tensors that
        ``set_prompt`` consumes (input_ids / visual_pos_masks /
        position_ids / rope_cos / rope_sin / llm_input_embeds /
        pixel_features / grid_thw). The prompt lives in this bundle's
        embeddings; the prompt STRING below is cosmetic.
    FLASH_RT_GROOT_N17_AMD_REF    (or bare GROOT_N17_AMD_REF)   [optional]
        torch.load-able reference fixture with ``inputs["state"]``
        (per-modality) and ``actions`` (per-modality). Absent → the one
        cosine test skips; everything else still runs. Requires the aux
        bundle to also carry ``initial_noise`` (the fixture's captured
        noise), otherwise that test skips too.
    FLASH_RT_GROOT_N17_AMD_EMBODIMENT                            [optional]
        Embodiment tag; defaults to the relative-EEF DROID tag the AMD
        line validates against.

Skip conditions: ROCm torch + visible device + built extension + resolved
checkpoint + resolved aux; each missing piece skips with its own reason.
Every heavyweight import lives inside a fixture, so NVIDIA/no-ROCm CI
collects this module cleanly.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


_PROMPT = "Put the blue block in the green bowl"
_ACTION_HORIZON = 40
_ACTION_DIM = 132
_DEFAULT_EMBODIMENT = "oxe_droid_relative_eef_relative_joint"
_REF_MODALITIES = ("eef_9d", "gripper_position", "joint_position")


def _env(*names: str):
    """First non-empty value among the FLASH_RT_-namespaced and bare names.

    Mirrors the two-level lookup the rest of the AMD suite uses: the
    namespaced form is documented, the bare form is accepted so existing
    invocations keep working.
    """
    for name in names:
        value = os.environ.get(f"FLASH_RT_{name}") or os.environ.get(name)
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def amd_env():
    """ROCm torch + device + extension + frontend, or skip with the reason."""
    try:
        importlib.import_module("flash_rt.amd.flash_rt_amd_kernels")
    except ImportError as exc:
        pytest.skip(f"flash_rt_amd_kernels not importable: {exc}")
    try:
        # Pulls in the Thor base and its third-party helpers (declared
        # under an optional extra); skip rather than error on a lean env.
        importlib.import_module("flash_rt.amd.frontends.torch.groot_n17")
    except ImportError as exc:
        pytest.skip(f"AMD GROOT N1.7 frontend not importable: {exc}")
    torch = pytest.importorskip("torch")
    if not getattr(torch.version, "hip", None):
        pytest.skip("torch is not a ROCm build")
    if not torch.cuda.is_available():
        pytest.skip("no ROCm device visible to torch")
    return torch


@pytest.fixture(scope="module")
def ckpt(amd_env):
    path = _env("GROOT_N17_AMD_CKPT")
    if not path:
        pytest.skip("no GROOT N1.7 checkpoint: export "
                    "FLASH_RT_GROOT_N17_AMD_CKPT to a checkpoint directory")
    if not os.path.isdir(path):
        pytest.skip("FLASH_RT_GROOT_N17_AMD_CKPT is not a directory")
    if not os.path.isfile(os.path.join(path, "statistics.json")):
        pytest.skip("checkpoint dir lacks statistics.json — point "
                    "FLASH_RT_GROOT_N17_AMD_CKPT at a GROOT N1.7 checkpoint")
    return path


@pytest.fixture(scope="module")
def aux(amd_env):
    """The HF-derived setup bundle ``set_prompt`` consumes."""
    torch = amd_env
    path = _env("GROOT_N17_AMD_AUX")
    if not path:
        pytest.skip("no aux bundle: export FLASH_RT_GROOT_N17_AMD_AUX to a "
                    "torch.load-able set_prompt input bundle")
    if not os.path.isfile(path):
        pytest.skip("FLASH_RT_GROOT_N17_AMD_AUX does not point at a file")
    bundle = torch.load(path, weights_only=False, map_location="cpu")
    missing = [k for k in ("llm_input_embeds", "pixel_features", "grid_thw",
                           "rope_cos", "rope_sin", "visual_pos_masks")
               if k not in bundle]
    if missing:
        pytest.skip(f"aux bundle lacks required keys: {missing}")
    return bundle


@pytest.fixture(scope="module")
def frontend(amd_env, ckpt, aux):
    """One constructed + prompted frontend for the whole module.

    ``set_prompt`` may only be called once per instance (the base refuses a
    second call), and it is the expensive step — calibration shadow, FP8
    alpha bake, cache write — so it is module-scoped by design.
    """
    from flash_rt.amd.frontends.torch.groot_n17 import GrootN17TorchFrontendAmd

    embodiment = (_env("GROOT_N17_AMD_EMBODIMENT") or _DEFAULT_EMBODIMENT)
    fe = GrootN17TorchFrontendAmd(
        ckpt, num_views=2, embodiment_tag=embodiment)
    fe.set_prompt(aux=aux, prompt=_PROMPT)
    return fe


@pytest.fixture(scope="module")
def synthetic_state(frontend):
    """An in-distribution state built from the checkpoint's own statistics.

    Using the per-modality q01/q99 midpoint keeps the basic gates
    independent of the optional reference fixture while staying inside the
    normalizer's calibrated range (zeros would be out-of-range for some
    modalities and would make "finite output" a weaker claim).
    """
    stats = frontend._read_statistics()["state"]
    state_dict = {}
    for mod_key, mod_stats in stats.items():
        q01 = np.asarray(mod_stats["q01"], dtype=np.float32)
        q99 = np.asarray(mod_stats["q99"], dtype=np.float32)
        mid = (0.5 * (q01 + q99)).reshape(1, 1, -1)
        state_dict[f"state.{mod_key}"] = mid
    return state_dict


@pytest.fixture(scope="module")
def state_normed(frontend, synthetic_state):
    return frontend.normalize_state(synthetic_state)


def _noise(torch, seed: int):
    """Deterministic (1, horizon, 132) initial noise, CPU-seeded.

    Generated on CPU with an explicit generator so the same bytes are
    produced on any device/driver — the bit-identity gate must compare two
    runs of the model, not two draws of a device RNG.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, _ACTION_HORIZON, _ACTION_DIM, generator=gen,
                       dtype=torch.float32)


# ---------------------------------------------------------------------------
# Construction + set_prompt (FP8 kernel backbone)
# ---------------------------------------------------------------------------


def test_frontend_is_the_amd_class(frontend):
    """The AMD frontend must be what was constructed — a silent fallback to
    the Thor/RTX base would still run, on the wrong kernels."""
    assert type(frontend).__name__ == "GrootN17TorchFrontendAmd"
    assert type(frontend).__module__ == "flash_rt.amd.frontends.torch.groot_n17"
    # The AMD production tier is FP8 backbone + bf16 DiT.
    assert frontend._DIT_USE_FP8 is False


def test_frontend_bound_the_amd_kernel_module(frontend):
    """The base resolves ``_fvk``/``_gemm`` lazily and would otherwise fall
    back to the CUDA extension; the AMD subclass must have seeded both with
    the ROCm module before any base method ran."""
    assert frontend._fvk.__name__.endswith("flash_rt_amd_kernels")
    assert frontend._gemm is not None
    assert frontend._mlp_gemm is not None


def test_set_prompt_produced_backbone_features(frontend, aux):
    """``set_prompt`` must leave fp16 backbone features of the documented
    ``(1, S, 2048)`` shape, with S taken from the aux bundle — a shape or
    dtype drift here silently mis-feeds every DiT cross-attention block."""
    torch = pytest.importorskip("torch")
    feats = frontend._backbone_features
    seq = int(aux["llm_input_embeds"].shape[1])
    assert frontend.Se == seq
    assert tuple(feats.shape) == (1, seq, 2048)
    assert feats.dtype is torch.float16
    assert torch.isfinite(feats).all().item(), "non-finite backbone features"
    assert feats.abs().max().item() > 0, "backbone features are all zero"


def test_set_prompt_baked_fp8_alphas_for_every_stage(frontend):
    """THE witness that the FP8 kernel backbone ran, not a bf16 shadow: the
    per-layer act-scale device tensors and host alphas for all four stages
    (ViT 24L, DeepStack 3 mergers, LLM 16L, VL self-attn) are baked. Finite
    positive alphas are required — a zero or NaN alpha would scale a whole
    layer's GEMM output to garbage without raising."""
    per_stage = {
        "_vit_alpha_q": 24, "_vit_alpha_o": 24,
        "_vit_alpha_fc1": 24, "_vit_alpha_fc2": 24,
        "_dsm_alpha_fc1": 3, "_dsm_alpha_fc2": 3,
        "_vlsa_alpha_q": 4, "_vlsa_alpha_k": 4, "_vlsa_alpha_v": 4,
        "_vlsa_alpha_o": 4, "_vlsa_alpha_fc1": 4, "_vlsa_alpha_fc2": 4,
    }
    for attr, count in per_stage.items():
        alphas = getattr(frontend, attr, None)
        assert alphas is not None, f"{attr} not baked by set_prompt"
        assert len(alphas) == count, f"{attr} has {len(alphas)} != {count}"
        for i, a in enumerate(alphas):
            a = float(a)
            assert np.isfinite(a) and a > 0.0, f"{attr}[{i}] = {a}"
    # Device-side act scales for the LLM stage (one per layer, per GEMM).
    assert len(frontend._llm_act_qkv_dev) == 16
    assert len(frontend._llm_act_o_dev) == 16
    assert len(frontend._llm_act_gateup_dev) == 16
    assert len(frontend._llm_act_down_dev) == 16


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def first_actions(amd_env, frontend, state_normed):
    """First infer — also the call that captures the DiT graph chain."""
    torch = amd_env
    out = frontend.infer(state_normed, initial_noise=_noise(torch, 0))
    return out.detach().float().cpu().clone()


def test_infer_shape_and_finite(first_actions):
    """The normalized action contract is (1, action_horizon, 132)."""
    assert tuple(first_actions.shape) == (1, _ACTION_HORIZON, _ACTION_DIM)
    arr = first_actions.numpy()
    assert np.isfinite(arr).all(), "non-finite actions from the FP8 pipeline"
    # Normalized actions are O(1): an all-zero output means a dead buffer,
    # an exploded one means a broken scale — neither is a plausible policy.
    assert np.abs(arr).max() > 1e-6
    assert np.abs(arr).max() < 1e3


def test_dit_graph_was_captured(frontend):
    """The latency story on this backend is graph replay, not eager torch;
    the first infer must have left a captured DiT graph chain behind."""
    assert getattr(frontend, "_k_dit_graph", None) is not None, (
        "first infer did not capture the kernelized DiT graph")


def test_pinned_noise_is_bit_identical(amd_env, frontend, state_normed):
    """Same pinned initial_noise twice → identical bytes.

    The captured graph replays a fixed kernel sequence over persistent
    buffers; with backbone, state and noise all pinned there is no
    legitimate source of variation. This is an equality gate — a mismatch
    is a determinism bug (uninitialized scratch, buffer aliasing, atomic
    reduction), not "GPU noise". Both results are cloned off the returned
    tensor first so the comparison can never degenerate into comparing a
    persistent buffer with itself.
    """
    torch = amd_env
    a1 = frontend.infer(state_normed,
                        initial_noise=_noise(torch, 0)).detach().float().cpu().clone()
    a2 = frontend.infer(state_normed,
                        initial_noise=_noise(torch, 0)).detach().float().cpu().clone()
    assert a1.data_ptr() != a2.data_ptr(), "results alias; gate is vacuous"
    d1, d2 = a1.numpy(), a2.numpy()
    assert np.array_equal(d1, d2), (
        f"same pinned noise, different actions (max diff "
        f"{np.abs(d1 - d2).max():.3e}) — graph replay is not deterministic")


def test_different_noise_changes_the_actions(amd_env, frontend, state_normed,
                                             first_actions):
    """Guards the equality gate above from being vacuous: if the noise
    input were dropped on the floor (never copied into the graph's action
    buffer), every infer would match and the bit-identity test would pass
    for the wrong reason."""
    torch = amd_env
    other = frontend.infer(
        state_normed,
        initial_noise=_noise(torch, 12345)).detach().float().cpu().clone()
    assert not np.array_equal(other.numpy(), first_actions.numpy()), (
        "a different initial_noise produced identical actions — the noise "
        "input is not reaching the diffusion chain")


# ---------------------------------------------------------------------------
# Optional numeric reference gate
# ---------------------------------------------------------------------------


def test_actions_match_reference_when_provided(amd_env, frontend, aux):
    """Cosine >= 0.999 on denormalized per-modality actions against a
    reference fixture, driven by the fixture's own state and the aux
    bundle's captured initial_noise. 0.999 is this backend's E2E
    FP8-vs-reference acceptance bar; below it, quantization cannot be the
    explanation and a real regression is in the graph.

    Runs the full production output path (infer → denormalize_action with
    the state_dict, which the relative-EEF embodiments require), so it also
    covers the denormalization the shape/finite gates above skip.
    """
    torch = amd_env
    ref_path = _env("GROOT_N17_AMD_REF")
    if not ref_path:
        pytest.skip("no reference fixture: export FLASH_RT_GROOT_N17_AMD_REF "
                    "to a torch.load-able fixture with inputs/actions")
    # denormalize_action builds the HF processor, whose tokenizer metadata
    # lookup transformers 4.57.x performs even in offline mode. The library
    # deliberately refuses to patch huggingface_hub for that (it is shared
    # process state); an offline caller injects a processor instead, which
    # is what this harness does when the lookup is blocked.
    try:
        frontend._hf_processor()
    except RuntimeError as exc:
        if "HF_HUB_OFFLINE" not in str(exc):
            raise
        pytest.skip("processor load needs a Hub metadata lookup that this "
                    "environment blocks; build one in setup code and call "
                    "set_hf_processor() to run this gate offline")
    if not os.path.isfile(ref_path):
        pytest.skip("FLASH_RT_GROOT_N17_AMD_REF does not point at a file")
    if "initial_noise" not in aux:
        pytest.skip("aux bundle carries no initial_noise; the reference "
                    "comparison needs the fixture's captured noise")

    fixture = torch.load(ref_path, weights_only=False, map_location="cpu")
    if "actions" not in fixture or "inputs" not in fixture:
        pytest.skip("reference fixture lacks 'inputs'/'actions'")
    modalities = [k for k in _REF_MODALITIES if k in fixture["actions"]]
    if not modalities:
        pytest.skip(f"reference fixture has none of {_REF_MODALITIES}")

    state_dict = {f"state.{k}": v
                  for k, v in fixture["inputs"]["state"].items()}
    normed = frontend.normalize_state(state_dict)
    out = frontend.infer(normed, initial_noise=aux["initial_noise"])
    assert torch.isfinite(out).all().item(), "infer produced NaN/Inf"

    denorm = frontend.denormalize_action(out, state_dict=state_dict)
    pred = torch.cat([torch.as_tensor(denorm[k]).flatten().double()
                      for k in modalities])
    ref = torch.cat([torch.as_tensor(fixture["actions"][k]).flatten().double()
                     for k in modalities])
    assert pred.shape == ref.shape, "denormalized actions shape mismatch"
    cos = float(pred @ ref / (pred.norm() * ref.norm() + 1e-30))
    assert cos >= 0.999, f"actions cos vs reference = {cos:.6f} < 0.999"
