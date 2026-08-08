"""Form adjudication: candidates declared, facts qualify, measurement seats.

No form is ever hard-coded to a shape or a device. A seat family
declares its candidates; qualification predicates read device-neutral
facts; the qualified candidates are timed on the calibrated shapes and
the measurement seats the winner. Precision precedes speed twice over:
a candidate enters only above the parity floor (0.99 — the house
tradeoff-and-calibrate line), and within the win margin the
higher-precision form takes the seat, so noise never flips one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

#: the house parity floor: at or above this, a form is inside the
#: tradeoff-and-calibration band and speed may decide
PARITY_FLOOR = 0.99


@dataclass
class FormCandidate:
    name: str
    build: Callable[[], Any]
    run: Callable[[Any], Any]
    qualify: Callable[[], bool] = lambda: True
    #: tie-break inside the win margin: lower = higher precision
    precision_rank: int = 0


def adjudicate(seat: str, candidates: Sequence[FormCandidate], *,
               margin: float = 0.03, iters: int = 30,
               notes: dict | None = None):
    """Measure the qualified candidates; return (winner_name, built).

    Every candidate's outcome — unqualified, refused (with reason), or
    its measured milliseconds — lands in ``notes["adjudication"]``.
    A candidate that cannot build or run loses by default. Returns
    ``(None, None)`` when nothing survives.
    """
    entries, built = [], {}
    for cand in candidates:
        try:
            if not cand.qualify():
                entries.append({"name": cand.name,
                                "outcome": "unqualified"})
                continue
            built[cand.name] = cand.build()
        except Exception as exc:            # noqa: BLE001 — loses
            entries.append({"name": cand.name,
                            "outcome": f"refused: {str(exc)[:80]}"})
    timed = []
    for cand in candidates:
        if cand.name not in built:
            continue
        b = built[cand.name]
        with torch.no_grad():
            for _ in range(5):
                cand.run(b)
            torch.cuda.synchronize()
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                cand.run(b)
            end.record()
            torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        timed.append((ms, cand.precision_rank, cand.name))
        entries.append({"name": cand.name, "ms": round(ms, 4)})
    winner = None
    if timed:
        timed.sort()
        best = timed[0][0]
        close = [t for t in timed if t[0] <= best * (1.0 + margin)]
        close.sort(key=lambda t: (t[1], t[0]))
        winner = close[0][2]
    if notes is not None:
        notes.setdefault("adjudication", []).append(
            {"seat": seat, "winner": winner, "margin": margin,
             "entries": entries})
    return winner, (built.get(winner) if winner else None)
