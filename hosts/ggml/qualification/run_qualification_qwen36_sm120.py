#!/usr/bin/env python3
"""Qualification gates for the SM120/Qwen3.6-35B target.

Offline gates (always run):
  A. manifest — the pipeline binding validates against the live catalog.
  B. pins     — pinned structure versions match the live catalog.
  C. header   — the checked-in binding constants header is up to date with
                the binding yaml (regenerate with tools/gen_binding_header.py).

On-device gates (opt-in; each re-establishes its number by running, never by
quoting — see the gates block in pins_qwen36_sm120.yaml for the recorded
baseline):
  D. selftest — duplicated-token bit-exact replay across batch variants
                (FRT_MOEFUSE_SELFTEST) plus, when a reference pack is given,
                the online-repack byte-identity check.
  E. ppl      — 24-chunk -ub 1 perplexity, safe tier; must match the pinned
                value to the printed precision (bit-stable path).
  F. bench    — tg128 r=5; red below (1 - tol) x pinned.

Usage:
  python run_qualification_qwen36_sm120.py                    # A+B+C
  python run_qualification_qwen36_sm120.py --device \\
      --bin <build/bin dir> --model <gguf> --wikitext <txt> \\
      [--regions-ref <pack>] [--tol 0.03]                     # + D+E+F
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from flash_rt.structures.binding import load_binding          # noqa: E402
from flash_rt.structures.registry import load as load_structure  # noqa: E402

BINDING = "llamacpp_qwen36_35b_sm120"
PINS = _HERE / "pins_qwen36_sm120.yaml"


def gate_manifest() -> tuple[bool, str]:
    try:
        spec = load_binding(BINDING, require_pipeline_coverage=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"binding failed validation: {exc}"
    return True, (f"{spec.name} -> {spec.structure.name}@{spec.structure.version}, "
                  f"{len(spec.segments)} segments, contract {spec.coverage_contract}")


def gate_pins() -> tuple[bool, str]:
    pinned = yaml.safe_load(PINS.read_text())["pins"]
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


def gate_header() -> tuple[bool, str]:
    gen = _HERE.parent / "tools" / "gen_binding_header.py"
    header = _HERE.parent / f"fr_binding_{BINDING.split('llamacpp_')[-1]}.h"
    if not header.is_file():
        return False, f"missing {header.name}"
    current = header.read_text()
    proc = subprocess.run([sys.executable, str(gen), BINDING], capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"generator failed: {proc.stderr.strip()}"
    fresh = header.read_text()
    if fresh != current:
        header.write_text(current)   # restore; the red asks for a deliberate regen+review
        return False, "binding header is stale — regenerate with tools/gen_binding_header.py and review"
    return True, f"{header.name} matches the binding"


# ---- on-device gates -------------------------------------------------------

def _run(cmd, env_extra=None, timeout=1800):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)


def gate_selftest(args) -> tuple[bool, str]:
    env = {"FRT_MOEFUSE_SELFTEST": "1"}
    if args.regions_ref:
        env.update({"FRT_REPACK_CHECK": "1", "FRT_REGIONS_PACK_REF": args.regions_ref})
    proc = _run([f"{args.bin}/llama-cli", "-m", args.model, "-fa", "on", "-st",
                 "-n", "8", "-p", "Hello"], env)
    out = proc.stdout + proc.stderr
    ok_self = "PASS" in out and "FAIL" not in out
    msgs = [f"selftest {'PASS' if ok_self else 'FAIL'}"]
    ok = ok_self
    if args.regions_ref:
        n_ok = len(re.findall(r"packed=OK sf=OK", out))
        n_bad = out.count("MISMATCH")
        msgs.append(f"repack byte-identity {n_ok} OK / {n_bad} mismatch")
        ok = ok and n_bad == 0 and n_ok > 0
    return ok, "; ".join(msgs)


def gate_ppl(args, pinned: float) -> tuple[bool, str]:
    proc = _run([f"{args.bin}/llama-perplexity", "-m", args.model, "-f", args.wikitext,
                 "-ub", "1", "-c", "512", "-b", "512", "--chunks", "24", "-fa", "1"])
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", proc.stdout + proc.stderr)
    if not m:
        return False, "no PPL estimate in output"
    got = float(m.group(1))
    ok = abs(got - pinned) < 5e-5
    return ok, f"ppl {got} vs pinned {pinned}"


def gate_bench(args, pinned: float, tol: float) -> tuple[bool, str]:
    env = {}
    if args.head_pack:
        env = {"FRT_HEAD_SWAP": "1", "FRT_HEAD_PACK": args.head_pack}
    proc = _run([f"{args.bin}/llama-bench", "-m", args.model, "-fa", "1",
                 "-p", "0", "-n", "128", "-r", "5"], env)
    m = re.search(r"tg128\s*\|\s*([0-9.]+)", proc.stdout)
    if not m:
        return False, "no tg128 in output"
    got = float(m.group(1))
    ok = got >= pinned * (1.0 - tol)
    tier = "full" if args.head_pack else "safe/default"
    return ok, f"tg128 {got} ({tier}) vs pinned {pinned} (tol {tol:.0%})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true", help="run the on-device gates")
    ap.add_argument("--bin", help="llama.cpp build/bin directory")
    ap.add_argument("--model", help="target GGUF")
    ap.add_argument("--wikitext", help="wikitext test file for the ppl gate")
    ap.add_argument("--regions-ref", help="offline region pack for the byte-identity check")
    ap.add_argument("--head-pack", help="full-tier head pack (bench gate then judges the full tier)")
    ap.add_argument("--tol", type=float, default=0.03)
    args = ap.parse_args()

    gates = [("manifest", gate_manifest()), ("pins", gate_pins()), ("header", gate_header())]
    if args.device:
        if not (args.bin and args.model):
            ap.error("--device needs --bin and --model")
        g = yaml.safe_load(PINS.read_text())["gates"]
        gates.append(("selftest", gate_selftest(args)))
        if args.wikitext:
            gates.append(("ppl", gate_ppl(args, float(g["quality"]["ppl_24ch_ub1_safe_tier"]))))
        key = "tg128_full_tier_bench" if args.head_pack else "tg128_default_bench"
        gates.append(("bench", gate_bench(args, float(g["perf"][key]), args.tol)))

    red = False
    for name, (ok, msg) in gates:
        print(f"[{'GREEN' if ok else 'RED':5}] {name}: {msg}")
        red = red or not ok
    return 1 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
