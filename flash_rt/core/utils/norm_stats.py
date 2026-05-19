"""Normalization-statistics loader with lerobot fallback.

The Pi0 family of frontends (Pi0, Pi0.5, Pi0-FAST — torch + jax)
consume a single dict shaped:

    {"actions": {"q01": [...], "q99": [...], ...}, "state": {...}}

This is the openpi convention shipped under
``<ckpt>/assets/physical-intelligence/<task>/norm_stats.json``.

The Hugging Face ``lerobot`` policy releases of the same models (e.g.
``lerobot/pi05_libero_finetuned_v044``) ship a different layout that
the loader has to translate at the boundary. Two on-disk forms are
supported:

  1. **Lerobot policy preprocessor / postprocessor safetensors** — the
     official HF model release format. A pair of files lives at
     ``<ckpt>/policy_preprocessor_step_*_normalizer_processor.safetensors``
     and ``<ckpt>/policy_postprocessor_step_*_unnormalizer_processor.safetensors``
     keyed by flat ``<feature>.<stat>`` names (e.g. ``action.q01``,
     ``observation.state.q99``). Action stats live in the
     postprocessor file, observation stats in the preprocessor file.
  2. **Lerobot dataset stats.json** — the dataset-side convention
     (``meta/stats.json`` in a dataset repo) keyed by feature with
     nested ``min/max/mean/std/q01/q99`` blocks. Sometimes shipped
     alongside model weights for convenience.

Both translate to the same openpi-shaped output dict so downstream
``unnormalize_actions`` (and any future consumer) is schema-agnostic.

This module exposes one entry-point, :func:`load_norm_stats`, that
walks a list of file candidates AND optionally scans a checkpoint
directory for the lerobot policy safetensors pair, returning the
first hit. Adding more checkpoint layouts in the future is a
one-file change.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ─── Lerobot ↔ openpi schema map (JSON / dataset stats path) ──────────
# Disambiguator vs openpi: lerobot always uses singular ``action`` and
# the namespaced ``observation.state`` (with a dot). openpi uses plural
# ``actions`` and bare ``state``. So a payload with ``observation.state``
# or singular ``action`` is unambiguously lerobot; anything with plural
# ``actions`` or bare ``state`` (and no namespaced keys) is openpi.
_LEROBOT_ACTION_KEYS = ("action",)
_LEROBOT_STATE_KEYS = ("observation.state",)

# Lerobot policy preprocessor / postprocessor file naming. Both files
# are produced by ``lerobot.policies.normalize`` at training time and
# uploaded alongside ``model.safetensors`` in the HF model release.
# The ``step_<n>`` index varies between releases (we've seen 0 and 2).
_LEROBOT_NORMALIZER_GLOB = "policy_preprocessor_step_*_normalizer_processor.safetensors"
_LEROBOT_UNNORMALIZER_GLOB = (
    "policy_postprocessor_step_*_unnormalizer_processor.safetensors")


# ──────────────────────────────────────────────────────────────────────
# Lerobot dataset stats.json path
# ──────────────────────────────────────────────────────────────────────


def _is_lerobot_stats(d: dict) -> bool:
    """Return True if the dict looks like lerobot ``stats.json`` output."""
    if not isinstance(d, dict):
        return False
    has_action = any(k in d for k in _LEROBOT_ACTION_KEYS)
    has_state = any(k in d for k in _LEROBOT_STATE_KEYS)
    if not (has_action or has_state):
        return False
    sample = next((d[k] for k in _LEROBOT_ACTION_KEYS + _LEROBOT_STATE_KEYS
                   if k in d), None)
    if not isinstance(sample, dict):
        return False
    return ("q01" in sample and "q99" in sample) or (
        "min" in sample and "max" in sample)


def _lerobot_to_openpi(stats: dict) -> dict:
    """Translate a lerobot ``stats.json`` payload into the openpi schema."""
    out: dict = {}

    def _copy(block: dict) -> Optional[dict]:
        if not isinstance(block, dict):
            return None
        canon: dict = {}
        for k in ("q01", "q99", "mean", "std", "min", "max"):
            if k in block:
                canon[k] = block[k]
        if "q01" not in canon and "min" in canon:
            canon["q01"] = canon["min"]
            canon["q99"] = canon["max"]
            logger.warning(
                "norm_stats: lerobot stats only carry min/max (no "
                "q01/q99); using min/max as q01/q99. Expect a 1-5%% "
                "drift in unnormalized actions vs the openpi-trained "
                "model.")
        return canon if canon else None

    for k in _LEROBOT_ACTION_KEYS:
        if k in stats:
            mapped = _copy(stats[k])
            if mapped is not None:
                out["actions"] = mapped
                break
    for k in _LEROBOT_STATE_KEYS:
        if k in stats:
            mapped = _copy(stats[k])
            if mapped is not None:
                out["state"] = mapped
                break
    return out


# ──────────────────────────────────────────────────────────────────────
# Lerobot policy preprocessor / postprocessor safetensors path
# ──────────────────────────────────────────────────────────────────────

# Stat suffixes we know how to read out of the safetensors keys.
_LEROBOT_POLICY_STAT_SUFFIXES = (
    "q01", "q99", "q10", "q50", "q90",
    "mean", "std", "min", "max",
)


def _extract_feature_stats(
    tensors: dict, feature_key: str,
) -> Optional[dict]:
    """Pull every ``<feature_key>.<stat>`` tensor and return as
    {stat: list[float]}. Returns None if no recognised stats present.
    """
    block: dict = {}
    for stat in _LEROBOT_POLICY_STAT_SUFFIXES:
        full_key = f"{feature_key}.{stat}"
        if full_key in tensors:
            block[stat] = tensors[full_key].reshape(-1).tolist()
    if "q01" not in block and "min" in block:
        block["q01"] = block["min"]
        block["q99"] = block["max"]
        logger.warning(
            "norm_stats: lerobot policy stats only carry min/max for "
            "feature %r; using min/max as q01/q99.", feature_key)
    return block if block else None


def _load_lerobot_policy_safetensors(
    preproc_path: pathlib.Path,
    postproc_path: pathlib.Path,
) -> Optional[dict]:
    """Load the lerobot policy normalizer + unnormalizer safetensors
    pair and return openpi-shaped stats.

    Action stats come from the postprocessor file (it ``un``-normalises
    the model's action output back to data space). State / observation
    stats come from the preprocessor file (it normalises the inputs).
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        logger.warning("norm_stats: safetensors unavailable (%s); "
                       "skipping lerobot policy stats", e)
        return None

    out: dict = {}

    # Postprocessor → action stats.
    try:
        with safe_open(postproc_path, framework="np") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        action = _extract_feature_stats(tensors, "action")
        if action is not None:
            out["actions"] = action
    except (OSError, RuntimeError) as e:
        logger.warning("norm_stats: failed to read %s: %s", postproc_path, e)

    # Preprocessor → observation.state stats.
    try:
        with safe_open(preproc_path, framework="np") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        state = _extract_feature_stats(tensors, "observation.state")
        if state is not None:
            out["state"] = state
    except (OSError, RuntimeError) as e:
        logger.warning("norm_stats: failed to read %s: %s", preproc_path, e)

    return out if out else None


def _find_lerobot_policy_stats(
    checkpoint_dir: pathlib.Path,
) -> Optional[dict]:
    """Discover and load the lerobot policy safetensors pair under
    ``checkpoint_dir``. Returns openpi-shaped stats or None.
    """
    if not checkpoint_dir.is_dir():
        return None
    pre = sorted(checkpoint_dir.glob(_LEROBOT_NORMALIZER_GLOB))
    post = sorted(checkpoint_dir.glob(_LEROBOT_UNNORMALIZER_GLOB))
    if not (pre and post):
        return None
    out = _load_lerobot_policy_safetensors(pre[0], post[0])
    if out is not None:
        logger.info(
            "Loaded norm stats from lerobot policy safetensors: %s + %s",
            pre[0].name, post[0].name)
    return out


# ──────────────────────────────────────────────────────────────────────
# JSON candidate walker
# ──────────────────────────────────────────────────────────────────────


def _read_json(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _try_json_candidate(path: pathlib.Path) -> Optional[dict]:
    """Parse a single JSON candidate; return openpi-shaped dict or None."""
    try:
        exists = path.exists()
    except OSError as e:
        logger.warning("norm_stats: cannot access %s: %s", path, e)
        return None
    if not exists:
        return None
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("norm_stats: failed to parse %s: %s", path, e)
        return None
    if isinstance(data, dict) and "norm_stats" in data:
        data = data["norm_stats"]
    if _is_lerobot_stats(data):
        mapped = _lerobot_to_openpi(data)
        if mapped:
            logger.info(
                "Loaded norm stats from %s (lerobot dataset schema -> openpi)",
                path)
            return mapped
        logger.warning(
            "norm_stats: %s looked like lerobot stats but had no "
            "actionable q01/q99 / min/max blocks; trying next candidate.",
            path)
        return None
    if isinstance(data, dict) and ("actions" in data or "state" in data):
        logger.info("Loaded norm stats from %s (openpi schema)", path)
        return data
    logger.warning(
        "norm_stats: %s parsed but had neither openpi nor lerobot "
        "shape; trying next candidate.", path)
    return None


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def load_norm_stats(
    candidates: Iterable[pathlib.Path],
    *,
    checkpoint_dir: Optional[pathlib.Path] = None,
    strict: bool = True,
) -> Optional[dict]:
    """Walk ``candidates`` (plus optional auto-discovery under
    ``checkpoint_dir``) and return the first parseable norm-stats dict
    in openpi shape.

    Args:
        candidates: ordered iterable of file paths to try (JSON files;
            the loader accepts openpi flat / openpi-wrapped / lerobot
            ``stats.json`` schemas).
        checkpoint_dir: optional directory to scan for the lerobot
            policy preprocessor / postprocessor safetensors pair
            (``policy_*processor_step_*_(un)normalizer_processor
            .safetensors``). Tried after ``candidates`` and before
            raising. Pass the model checkpoint root directory.
        strict: if True (default) raise FileNotFoundError when nothing
            is found; if False return ``None``.

    Returns:
        Dict shaped ``{"actions": {"q01", "q99", ...}, "state": {...}}``
        — the openpi schema that ``unnormalize_actions`` consumes.
    """
    tried: list[pathlib.Path] = []
    for p in candidates:
        p = pathlib.Path(p)
        tried.append(p)
        out = _try_json_candidate(p)
        if out is not None:
            return out

    if checkpoint_dir is not None:
        ckpt = pathlib.Path(checkpoint_dir)
        out = _find_lerobot_policy_stats(ckpt)
        if out is not None:
            return out
        tried.append(
            ckpt / "policy_*processor_step_*_(un)normalizer_processor.safetensors")

    if strict:
        raise FileNotFoundError(
            "norm_stats not found in any of:\n  "
            + "\n  ".join(str(p) for p in tried)
            + "\n\nExpected one of:\n"
            "  - openpi:  <ckpt>/assets/physical-intelligence/<task>/norm_stats.json\n"
            "  - openpi:  <ckpt>/norm_stats.json\n"
            "  - lerobot: <ckpt>/meta/stats.json (dataset schema)\n"
            "  - lerobot: <ckpt>/policy_*processor_step_*_(un)normalizer_processor"
            ".safetensors (HF model release)\n"
            "Pass an explicit candidate path, set checkpoint_dir, or "
            "convert your checkpoint.")
    return None


def lerobot_candidates(checkpoint_dir: pathlib.Path) -> list[pathlib.Path]:
    """Standard lerobot ``stats.json`` (dataset-style) locations.

    The HF model release format (policy safetensors) is auto-discovered
    by passing ``checkpoint_dir`` to :func:`load_norm_stats`; this
    helper only covers the dataset-side ``meta/stats.json`` /
    ``stats.json`` layout. Frontends typically pass both.
    """
    return [
        checkpoint_dir / "meta" / "stats.json",
        checkpoint_dir / "stats.json",
    ]


def pi05_candidates(checkpoint_dir: pathlib.Path) -> list[pathlib.Path]:
    """Common norm-stats locations for Pi0.5 checkpoints.

    Supports both the original ``pi05_libero`` openpi layout and the
    ``pi05_droid`` / ``pi05_droid_pytorch`` layout used by DROID
    checkpoints. The latter stores stats under ``assets/droid`` instead
    of ``assets/physical-intelligence/libero``.
    """
    home = pathlib.Path.home()
    return [
        checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
        checkpoint_dir / "assets" / "droid" / "norm_stats.json",
        checkpoint_dir.parent / "pi05_libero" / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
        checkpoint_dir.parent / "pi05_droid" / "assets" / "droid" / "norm_stats.json",
        checkpoint_dir.parent / "pi05_droid_pytorch" / "assets" / "droid" / "norm_stats.json",
        checkpoint_dir / "norm_stats.json",
        home / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero" / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
        home / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_droid" / "assets" / "droid" / "norm_stats.json",
        home / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_droid_pytorch" / "assets" / "droid" / "norm_stats.json",
        *lerobot_candidates(checkpoint_dir),
    ]
