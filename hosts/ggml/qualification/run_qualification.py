#!/usr/bin/env python3
"""Qualification gates for the ggml host adapter.

Offline gates (always run):
  A. manifest  — the pipeline binding validates against the live catalog
                 (structure renames, removed embedded regions, or a
                 malformed manifest turn this red).
  B. pins      — every pinned structure version matches the live catalog
                 (an upstream version bump turns this red and names the
                 structure, which is the cue to re-audit the bound window
                 before adopting the bump).

On-device gate (opt-in, needs a running llama-server with the pi0.5 model):
  C. e2e-golden — drives the fixed synthetic-input protocol against the
                 server and compares the raw action chunk to a stored
                 golden. The comparison is exact by default (the adapter
                 is bitwise deterministic across processes); pass --tol
                 to allow a max-abs band instead. Any kernel or window
                 change that shifts numerics turns this red.

Usage:
  python run_qualification.py                    # gates A+B
  python run_qualification.py --e2e --port 8089  # gates A+B+C
  python run_qualification.py --e2e --update-golden   # refresh the golden
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from flash_rt.structures.binding import load_binding          # noqa: E402
from flash_rt.structures.registry import load as load_structure  # noqa: E402

GOLDEN = _HERE / "goldens" / "pi05_thor_action.json"


def gate_manifest(binding_name: str) -> tuple[bool, str]:
    try:
        spec = load_binding(binding_name, require_pipeline_coverage=True)
    except Exception as exc:  # noqa: BLE001 — any validation failure is red
        return False, f"binding failed validation: {exc}"
    return True, (f"{spec.name} -> {spec.structure.name}@"
                  f"{spec.structure.version}, {len(spec.segments)} segments, "
                  f"contract {spec.coverage_contract}")


def gate_pins() -> tuple[bool, str]:
    pinned = yaml.safe_load((_HERE / "pins.yaml").read_text())["pins"]
    drifted = []
    for name, version in pinned.items():
        try:
            live = load_structure(name).version
        except KeyError:
            drifted.append(f"{name}: pinned @{version}, missing from catalog")
            continue
        if int(live) != int(version):
            drifted.append(f"{name}: pinned @{version}, catalog is @{live}")
    if drifted:
        return False, "; ".join(drifted)
    return True, f"{len(pinned)} structure versions match the catalog"


# ---- gate C: end-to-end action golden --------------------------------------

def _server_request(base: str, method: str, path: str, body=None):
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.dumps(body if body is not None else {}).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=120) as f:
        return json.loads(f.read())


def _synthetic_images(directory: pathlib.Path) -> list[str]:
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(1234)
    paths = []
    for name in ("base.png", "wrist.png"):
        p = directory / name
        if not p.exists():
            Image.fromarray(
                rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    return paths


def run_pipeline(port: int, image_dir: pathlib.Path, warmup: int = 2) -> dict:
    """Fixed synthetic-input protocol; the first evaluations after server
    start are cold (cache fill / capture paths) and differ from the steady
    state, so the comparison value is taken after ``warmup`` full passes —
    steady-state output is bitwise stable across runs and processes."""
    base = f"http://127.0.0.1:{port}"
    images = _synthetic_images(image_dir)
    state = ",".join(f"{0.01 * i:.4f}" for i in range(32))
    resp = None
    for _ in range(warmup + 1):
        _server_request(base, "POST", "/foreground/reset")
        for p in images:
            _server_request(base, "POST", "/foreground/image", {"path": p})
        _server_request(base, "PUT", "/foreground/state", {"state": state})
        resp = _server_request(base, "POST", "/foreground/infer",
                               {"text": "pick up the object"})
    return {"action_final_raw": resp.get("action_final_raw"),
            "action_steps": resp.get("action_steps"),
            "action_dim": resp.get("action_dim")}


def gate_e2e(port: int, image_dir: pathlib.Path, tol: float,
             update_golden: bool) -> tuple[bool, str]:
    got = run_pipeline(port, image_dir)
    if got["action_final_raw"] is None:
        return False, "server returned no action_final_raw"
    if update_golden:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got))
        return True, f"golden updated: {GOLDEN}"
    if not GOLDEN.exists():
        return False, f"no golden at {GOLDEN}; run with --update-golden first"
    want = json.loads(GOLDEN.read_text())
    if (got["action_steps"] != want["action_steps"]
            or got["action_dim"] != want["action_dim"]):
        return False, (f"shape drift: {got['action_steps']}x{got['action_dim']}"
                       f" vs golden {want['action_steps']}x{want['action_dim']}")
    flat_got = [x for row in got["action_final_raw"] for x in row]
    flat_want = [x for row in want["action_final_raw"] for x in row]
    diffs = [abs(a - b) for a, b in zip(flat_got, flat_want)]
    worst = max(diffs)
    n_diff = sum(1 for d in diffs if d > tol)
    if n_diff:
        return False, (f"{n_diff}/{len(diffs)} elements beyond tol={tol:g}, "
                       f"max_abs_diff={worst:.3e}")
    return True, (f"{len(diffs)} elements within tol={tol:g} "
                  f"(max_abs_diff={worst:.3e})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binding", default="jetson_pi_edge_pi05")
    ap.add_argument("--e2e", action="store_true",
                    help="also run the on-device action-golden gate")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--image-dir", type=pathlib.Path,
                    default=_HERE / "goldens")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max-abs tolerance for the e2e gate (default exact)")
    ap.add_argument("--update-golden", action="store_true")
    args = ap.parse_args()

    gates = [("manifest", gate_manifest(args.binding)),
             ("pins", gate_pins())]
    if args.e2e:
        args.image_dir.mkdir(parents=True, exist_ok=True)
        gates.append(("e2e-golden",
                      gate_e2e(args.port, args.image_dir, args.tol,
                               args.update_golden)))

    all_ok = True
    for name, (ok, detail) in gates:
        print(f"[{'GREEN' if ok else 'RED':5s}] {name}: {detail}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
