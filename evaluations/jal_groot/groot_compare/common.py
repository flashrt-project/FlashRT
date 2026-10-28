"""CPU-only protocol utilities shared by each isolated backend process."""
import hashlib
import json
from pathlib import Path
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_noise(record_id, seed=20261026, shape=(1, 40, 132)):
    """Generate per-record noise, rounding once to BF16 independently of GPU RNG."""
    key = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    value = rng.standard_normal(shape).astype(np.float32)
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).copy()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported manifest format")
    records = manifest["records"]
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate record IDs")
    hashes = {"calibration": set(), "evaluation": set()}
    for record in records:
        split = record["split"]
        if split not in hashes:
            raise ValueError(f"Unexpected split: {split}")
        obs = path.parent / record["observation"]
        if not obs.is_file():
            raise ValueError(f"Missing observation: {obs}")
        hashes[split].add(file_sha256(obs))
    if hashes["calibration"] & hashes["evaluation"]:
        raise ValueError("Calibration and evaluation observations overlap")
    if not all(hashes.values()):
        raise ValueError("Both calibration and evaluation observations are required")
    return manifest
