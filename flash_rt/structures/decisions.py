"""Band decisions: measured once per box, cached, consumed by every tier.

The seat-level micro-race was refuted in both directions by
production-form measurement; what stands is the captured end-to-end
number. This cache is where those numbers live: a band measurement run
records the winner per (device, band), and both assembly tiers read it
— the automatic path to route its formats, the explicit path as its
default band. An author pin (env) always outranks the cache; an empty
cache falls to the precision-order default. The cache file is a
receipt and is transportable: an edge box can ship with its decisions
measured elsewhere on identical hardware.
"""

from __future__ import annotations

import json
import os
import pathlib

import torch


def _cache_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get(
        "FRT_DECISION_CACHE",
        os.path.expanduser("~/.cache/flashrt/band_decisions.json")))


def _key(band: str) -> str:
    dev = (torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "cpu")
    return f"{dev}|{band}"


def lookup(band: str, default: str | None = None) -> str | None:
    try:
        data = json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return default
    entry = data.get(_key(band))
    return entry.get("winner", default) if entry else default


def record(band: str, winner: str, times_ms: dict) -> pathlib.Path:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data[_key(band)] = {"winner": winner, "times_ms": times_ms}
    path.write_text(json.dumps(data, indent=1))
    return path


def import_decisions(source: str) -> dict:
    """Merge a decision file measured elsewhere into the local cache.

    The peak-avoidance path for a memory-tight box: adjudicate on a
    roomy machine of the same device fingerprint, ship the file, and
    the tight box binds warm from its first run — the race (and its
    peak) never happens there. Entries whose device fingerprint does
    not match this box are carried along untouched (the cache is keyed
    by device, so they are inert here and correct if the file travels
    on). Local entries win on conflict: a number measured on this box
    outranks an imported one. Returns a receipt of what was merged.
    """
    imported = json.loads(pathlib.Path(source).read_text())
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        local = json.loads(path.read_text())
    except (OSError, ValueError):
        local = {}
    added = [k for k in imported if k not in local]
    merged = dict(imported)
    merged.update(local)
    path.write_text(json.dumps(merged, indent=2))
    return {"imported": len(added), "kept_local": len(local),
            "entries": sorted(added)}
