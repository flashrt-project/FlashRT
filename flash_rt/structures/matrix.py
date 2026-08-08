"""The structure-by-host activation matrix, generated from receipts.

Reuse is a claim about receipts, not intentions: a structure counts as
activated on a host only where a passing, digest-carrying receipt names
it in the executed chain. This module scans a directory of receipt
JSONs and renders the matrix — rows are structure families, columns are
hosts, cells cite the receipt gates — plus the release rule's tally
(a structure earns its catalog seat with two or more host families).

The matcher is deliberately dumb: exact family-name substrings against
the receipt's ``chain`` and ``gate`` text. A receipt that ran a family
without naming it is the receipt's defect to fix, not the matcher's to
guess around.
"""

from __future__ import annotations

import json
import pathlib

__all__ = ["FAMILIES", "generate"]

#: catalog families plus the doors that behave as families in receipts
FAMILIES = (
    "decode_loop",
    "gated_delta_core",
    "moe_experts",
    "linear_proj",
    "decoder_ffn",
    "qkv_pack",
    "vision_ffn",
    "modnorm_qkv_chain",
    "qkv_rope",
    "qk_norm_rope",
    "per_head_gqa",
    "two_way_fa2",
    "cross_attention",
    "fixed_iter",
    "adopt_prequantized",
    "quantize_on_adopt",
    "mtp",
)


def _host_key(host: str) -> str:
    """Collapse a receipt's host string to a short column name."""
    h = host.strip()
    for prefix in ("transformers ", "diffusers "):
        if h.startswith(prefix):
            h = h[len(prefix):]
    return h


def generate(evidence_dir) -> str:
    """Render the matrix markdown from a directory of receipt JSONs."""
    cells: dict[str, dict[str, set[str]]] = {}
    hosts: list[str] = []
    scanned = passing = 0
    for path in sorted(pathlib.Path(evidence_dir).glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        scanned += 1
        if rec.get("verdict") != "PASS" or "plan_digest" not in rec:
            continue
        host = _host_key(str(rec.get("host", "")))
        if not host:
            continue
        passing += 1
        text = " ".join((str(rec.get("chain", "")),
                         str(rec.get("gate", ""))))
        if host not in hosts:
            hosts.append(host)
        for fam in FAMILIES:
            if fam in text:
                cells.setdefault(fam, {}).setdefault(host, set()).add(
                    str(rec.get("gate", path.stem)))
    lines = [
        "# Structure-by-host activation matrix",
        "",
        f"Generated from {passing} passing receipts "
        f"({scanned} scanned). A cell cites the receipt gates that "
        "executed the family on that host; empty means no receipt, "
        "not no opinion.",
        "",
        "| structure | " + " | ".join(hosts) + " | hosts |",
        "|---|" + "---|" * (len(hosts) + 1),
    ]
    for fam in FAMILIES:
        row = cells.get(fam, {})
        if not row:
            continue
        parts = []
        for host in hosts:
            gates = sorted(row.get(host, ()))
            parts.append("<br>".join(gates) if gates else "")
        lines.append(f"| {fam} | " + " | ".join(parts)
                     + f" | {len(row)} |")
    lines += [
        "",
        "## Release rule tally (a family earns its seat with >=2 hosts)",
        "",
    ]
    for fam in FAMILIES:
        n = len(cells.get(fam, {}))
        if n:
            mark = "meets" if n >= 2 else "single-host"
            lines.append(f"- {fam}: {n} host(s) — {mark}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m flash_rt.structures.matrix "
              "<evidence-dir>")
        return 2
    print(generate(args[0]), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
