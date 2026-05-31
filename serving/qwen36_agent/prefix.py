"""Token-prefix helpers for contiguous FlashRT hot-state cache.

The first Qwen3.6 agent-serving backend is hot-state-first and contiguous: it
keeps one hot GPU KV/state region and reuses it when the next request extends
the same exact token prefix.  A client-provided session id is only a native
affinity hint.  More elaborate cache policies can implement the same policy
interface later without changing the exec contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PrefixMatch:
    """Exact token-prefix match result."""

    matched: int
    cached_len: int
    incoming_len: int

    @property
    def append_only(self) -> bool:
        return self.matched == self.cached_len <= self.incoming_len

    @property
    def exact(self) -> bool:
        return self.cached_len == self.incoming_len == self.matched

    @property
    def divergent(self) -> bool:
        return self.matched < min(self.cached_len, self.incoming_len)


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> PrefixMatch:
    """Return the exact common-prefix length for two token sequences."""

    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return PrefixMatch(matched=i, cached_len=len(a), incoming_len=len(b))


def token_digest(tokens: Iterable[int], *, salt: str = "") -> str:
    """Stable token digest for logs/cache metadata, not for security."""

    h = blake2b(digest_size=16)
    if salt:
        h.update(salt.encode("utf-8"))
        h.update(b"\0")
    for tok in tokens:
        h.update(int(tok).to_bytes(8, "little", signed=True))
    return h.hexdigest()
