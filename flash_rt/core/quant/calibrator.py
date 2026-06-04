"""FlashRT — FP8 calibration cache.

Calibration scales are checkpoint-specific and sequence-length-specific.
Caching avoids re-running the dynamic calibration pass (~3-4s) on every startup.

Cache key design:
  - Checkpoint identity: SHA256 of model.safetensors (first 64KB for speed)
  - Sequence length Se: different prompt lengths produce different shapes
  - Both are needed because:
    - Same model, different Se → different buffer shapes → potentially different scales
    - Same Se, different finetune → completely different activation distributions

Cache location: ~/.flash_rt/calibration/{ckpt_hash}_{Se}.json
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".flash_rt" / "calibration"


def _checkpoint_hash(checkpoint_path: str, read_bytes: int = 65536) -> str:
    """Fast hash of checkpoint file (first 64KB + file size).

    Hashes first 64KB + file size instead of the entire file (which can be
    multiple GB) for speed. This is sufficient to distinguish different
    finetunes because:
    - Different models have different first-layer weights → different first 64KB
    - Same model re-saved has identical content → same hash
    - File size as extra discriminator catches truncated/corrupted files
    """
    p = Path(checkpoint_path)
    if p.is_dir():
        # Look for a hashable file in the checkpoint directory
        candidates = [
            "model.safetensors", "model.pkl", "checkpoint.pkl",
            # Sharded safetensors (e.g. GROOT): hash the index or first shard
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00001-of-00004.safetensors",
            # Orbax (JAX): hash the manifest
            "params/manifest.ocdbt",
            "params/ocdbt.process_0/manifest.ocdbt",
        ]
        for name in candidates:
            candidate = p / name
            if candidate.exists():
                p = candidate
                break
        else:
            raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")

    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(read_bytes))
    file_size = p.stat().st_size
    h.update(str(file_size).encode())
    return h.hexdigest()[:16]  # 16 hex chars = 64 bits, collision-safe


def _cache_path(ckpt_hash: str, Se: int) -> Path:
    return CACHE_DIR / f"{ckpt_hash}_Se{Se}.json"


def save_calibration(checkpoint_path: str, Se: int,
                     enc_scales: list, enc_alpha: list,
                     ae_scales: list, enc_w_scales: list,
                     metadata: dict | None = None):
    """Save calibration scales to local cache.

    Args:
        checkpoint_path: Path to the checkpoint (for hashing).
        Se: Encoder sequence length (determines buffer shapes).
        enc_scales: Encoder activation scales, len = num_layers * 4.
        enc_alpha: Encoder alpha (= enc_scales * enc_w_scales), len = num_layers * 4.
        ae_scales: Decoder (AE) activation scales, len = num_layers * 4.
        enc_w_scales: Encoder weight scales (stored for alpha recomputation).
    """
    ckpt_hash = _checkpoint_hash(checkpoint_path)
    cache_file = _cache_path(ckpt_hash, Se)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "version": 1,
        "ckpt_hash": ckpt_hash,
        "Se": Se,
        "num_enc_scales": len(enc_scales),
        "num_ae_scales": len(ae_scales),
        "enc_scales": enc_scales,
        "enc_alpha": enc_alpha,
        "ae_scales": ae_scales,
        "enc_w_scales": enc_w_scales,
    }
    if metadata is not None:
        data["metadata"] = metadata

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Calibration saved: {cache_file} "
                f"(enc={len(enc_scales)}, ae={len(ae_scales)} scales)")
    return cache_file


def load_calibration(checkpoint_path: str, Se: int) -> Optional[dict]:
    """Load calibration scales from local cache.

    Returns dict with enc_scales, enc_alpha, ae_scales, enc_w_scales
    or None if cache miss (file not found or version mismatch).
    """
    try:
        ckpt_hash = _checkpoint_hash(checkpoint_path)
    except FileNotFoundError:
        return None

    cache_file = _cache_path(ckpt_hash, Se)

    if not cache_file.exists():
        logger.info(f"Calibration cache miss: {cache_file}")
        return None

    with open(cache_file) as f:
        data = json.load(f)

    # Validate
    if data.get("version") != 1:
        logger.warning(f"Calibration cache version mismatch, re-calibrating")
        return None
    if data.get("ckpt_hash") != ckpt_hash:
        logger.warning(f"Calibration cache hash mismatch, re-calibrating")
        return None
    if data.get("Se") != Se:
        logger.warning(f"Calibration cache Se mismatch ({data.get('Se')} != {Se})")
        return None

    logger.info(f"Calibration loaded from cache: {cache_file}")
    return data


def clear_calibration(checkpoint_path: str = None):
    """Clear calibration cache.

    Args:
        checkpoint_path: If provided, only clear cache for this checkpoint.
                        If None, clear all cached calibrations.
    """
    if not CACHE_DIR.exists():
        return

    if checkpoint_path is None:
        # Clear all
        count = 0
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} calibration cache files")
    else:
        ckpt_hash = _checkpoint_hash(checkpoint_path)
        count = 0
        for f in CACHE_DIR.glob(f"{ckpt_hash}_*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} calibration cache files for {ckpt_hash}")
