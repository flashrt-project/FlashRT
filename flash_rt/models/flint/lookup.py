"""N-gram (prompt lookup) draft tokens and greedy decoding with verification steps.

A draft is the run of tokens that followed the most recent earlier occurrence of the sequence's last
``n`` tokens (``n`` from ``max_ngram`` down to ``min_ngram``). The runtime's verification step computes the
last produced token and the draft tokens as consecutive positions, each with the arithmetic of the greedy
decode step at that position, and keeps the drafts that equal the greedy choice of the position before them.
The produced tokens are therefore exactly those of plain greedy decoding; only the number of passes over the
weights changes.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence, Tuple


class NgramDrafter:
    """Draft tokens from earlier occurrences of the sequence's suffix (extended token by token)."""

    def __init__(self, max_ngram: int = 6, min_ngram: int = 4):
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError("need 1 <= min_ngram <= max_ngram")
        self.max_ngram, self.min_ngram = max_ngram, min_ngram
        self.seq: List[int] = []
        # n-gram -> end index of its most recent occurrence that ends before the sequence's last token
        self._last: Dict[Tuple[int, ...], int] = {}

    def extend(self, tokens: Sequence[int]) -> None:
        seq, last = self.seq, self._last
        for t in tokens:
            end = len(seq) - 1   # the previous last token: its n-grams become earlier occurrences
            for n in range(1, min(self.max_ngram, end + 1) + 1):
                last[tuple(seq[end - n + 1:end + 1])] = end
            seq.append(int(t))

    def draft(self, k: int) -> List[int]:
        """Up to ``k`` tokens that followed the most recent earlier occurrence of the longest matching suffix."""
        seq, L = self.seq, len(self.seq)
        if k < 1:
            return []
        for n in range(min(self.max_ngram, L), self.min_ngram - 1, -1):
            end = self._last.get(tuple(seq[L - n:]))
            if end is not None:
                return seq[end + 1:end + 1 + k]
        return []


def lookup_decode(rt, context: Sequence[int], token: int, position: int, count: int, rows: int,
                  max_ngram: int = 6, min_ngram: int = 4, max_idle: int = 8,
                  eos: Sequence[int] = ()) -> Iterator[List[int]]:
    """Greedy continuation after ``token`` (the last produced token, at ``position``; ``context`` = every token
    before it) for ``count`` tokens, in blocks. ``rt`` is a ``FlintRuntime`` built with ``verify_rows >= rows``.

    Policy: when the drafter has a draft, one verification step of 1 + k rows; k starts at 1, grows by one (up to
    ``rows - 1``) after a step that accepted every draft and returns to 1 after a rejection. Without a draft,
    resident decode steps: 1, then twice as many after every further step without a draft (up to ``max_idle``),
    back to 1 once a draft is found. Stops after a block containing an ``eos`` token."""
    if not 2 <= rows <= rt.verify_rows:
        raise ValueError(f"rows must be in [2, {rt.verify_rows}] (the runtime's verify_rows)")
    if max_idle < 1:
        raise ValueError("max_idle must be positive")
    drafter = NgramDrafter(max_ngram, min_ngram)
    drafter.extend(context)
    drafter.extend([token])
    stop, eos4 = set(eos), list(eos)[:4]
    made, k, idle = 0, 1, 1
    while made < count:
        remaining = count - made
        draft = drafter.draft(min(k, remaining - 1))
        if draft:
            block = rt.verify([token] + draft, position)
            k = min(rows - 1, k + 1) if len(block) == len(draft) + 1 else 1
            idle = 1
        else:
            block = rt.decode(token, position, min(idle, remaining), eos=eos4)
            idle = min(max_idle, idle * 2)
        made += len(block)
        position += len(block)
        token = block[-1]
        drafter.extend(block)
        yield block
        if stop and any(t in stop for t in block):
            return
