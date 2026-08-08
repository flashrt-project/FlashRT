"""structures.explain — the coverage table, from the plan's own notes.

Adoption lives or dies on one question: *what did the system actually
do to my model, and why not more?* The plan already knows — discovered
seams, bound swaps, seams kept at host precision with the scheme's
reasons, refusals with theirs, adapter routes. This renders that
knowledge as one table instead of leaving it in ``plan.notes``.
"""

from __future__ import annotations

from typing import Any


def explain(plan: Any) -> str:
    """Render one plan's coverage as human-readable text."""
    lines: list[str] = []
    notes = getattr(plan, "notes", {}) or {}
    scheme = notes.get("scheme", {}) or {}
    name = scheme.get("name") or notes.get("scheme_name") or "?"
    lines.append(f"scheme: {name}"
                 + (" (auto)" if scheme.get("auto") else ""))

    swaps = getattr(plan, "swaps", {}) or {}
    observed = getattr(plan, "observed", {}) or {}
    lines.append(f"bound: {len(swaps)} swapped seam(s), "
                 f"{len(observed)} adapter-routed seam(s)")
    by_kind: dict[str, int] = {}
    for path in swaps:
        seam = None
        for s in getattr(plan, "seams", []) or []:
            if str(getattr(s, "path", "")) == str(path):
                seam = s
                break
        kind = getattr(seam, "structure", None) or "seam"
        by_kind[kind] = by_kind.get(kind, 0) + 1
    for kind in sorted(by_kind):
        lines.append(f"  {kind}: {by_kind[kind]}")
    adapter = notes.get("gated_delta_adapter")
    if adapter:
        lines.append(f"  gated-delta adapter: {adapter}")

    routed = scheme.get("formats", {}) or {}
    if routed:
        lines.append(f"routed to non-default formats: {len(routed)}")
        for path, fmt in list(sorted(routed.items()))[:8]:
            lines.append(f"  {path} -> {fmt}")
        if len(routed) > 8:
            lines.append(f"  ... {len(routed) - 8} more")

    kept = scheme.get("keep_host", {}) or {}
    if kept:
        lines.append(f"kept at host precision: {len(kept)}")
        for path, why in list(sorted(kept.items()))[:8]:
            lines.append(f"  {path}: {why or 'scheme decision'}")
        if len(kept) > 8:
            lines.append(f"  ... {len(kept) - 8} more")

    refused = notes.get("refused", []) or []
    if refused:
        from collections import Counter

        lines.append(f"refused: {len(refused)}")
        reasons = Counter(str(r[1] if isinstance(r, (tuple, list))
                              else r)[:88] for r in refused)
        for why, cnt in reasons.most_common(8):
            lines.append(f"  x{cnt}: {why}")
    if not refused and not kept:
        lines.append("refused: 0")
    return "\n".join(lines)
