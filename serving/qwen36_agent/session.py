"""Session and contiguous-prefix cache policy for Qwen3.6 agent serving."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .prefix import PrefixMatch, longest_common_prefix, token_digest


@dataclass(frozen=True)
class PrefixPlan:
    """How a request should reuse or rebuild session state."""

    session_id: str
    cached_tokens: int
    new_prefill_tokens: int
    incoming_tokens: int
    matched_tokens: int
    action: str

    @property
    def cache_hit(self) -> bool:
        return self.cached_tokens > 0 and self.action in {
            "exact",
            "append",
            "truncate",
            "restore",
        }


@dataclass
class SessionRecord:
    """Serving-layer session metadata.

    GPU KV/linear-attention state remains owned by the Qwen frontend.  This
    record tracks the token journal and cache policy metadata that decide when
    the hot frontend state can be reused.
    """

    session_id: str
    token_ids: List[int] = field(default_factory=list)
    cached_len: int = 0
    cache_salt: str = ""
    protected: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_digest: str = ""

    def plan(self, incoming: Sequence[int]) -> PrefixPlan:
        match = longest_common_prefix(self.token_ids[: self.cached_len], incoming)
        if match.exact:
            action = "exact"
            cached = match.matched
        elif match.append_only:
            action = "append"
            cached = match.matched
        elif match.matched == len(incoming):
            action = "truncate"
            cached = match.matched
        else:
            action = "rebuild"
            cached = 0
        return PrefixPlan(
            session_id=self.session_id,
            cached_tokens=cached,
            new_prefill_tokens=max(0, len(incoming) - cached),
            incoming_tokens=len(incoming),
            matched_tokens=match.matched,
            action=action,
        )

    def commit(self, incoming: Sequence[int], cached_len: Optional[int] = None) -> None:
        self.token_ids = [int(t) for t in incoming]
        self.cached_len = len(self.token_ids) if cached_len is None else int(cached_len)
        self.cached_len = max(0, min(self.cached_len, len(self.token_ids)))
        self.updated_at = time.time()
        self.last_digest = token_digest(self.token_ids, salt=self.cache_salt)


class SessionRegistry:
    """Small LRU registry for contiguous GPU-session policy.

    v1 is latency-first: one hot GPU frontend can serve one active session at a
    time.  The registry is deliberately generic enough for later paged,
    offloaded, or distributed cache backends.
    """

    def __init__(self, *, max_sessions: int = 8):
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.max_sessions = int(max_sessions)
        self._sessions: "OrderedDict[str, SessionRecord]" = OrderedDict()
        self.hot_session_id: Optional[str] = None

    def create(self, *, session_id: Optional[str] = None,
               cache_salt: str = "", protected: bool = False) -> SessionRecord:
        sid = session_id or f"frt-{uuid.uuid4().hex[:24]}"
        if sid in self._sessions:
            raise ValueError(f"session already exists: {sid}")
        rec = SessionRecord(
            session_id=sid,
            cache_salt=cache_salt,
            protected=protected,
        )
        self._sessions[sid] = rec
        self._evict_if_needed()
        return rec

    def get(self, session_id: str) -> Optional[SessionRecord]:
        rec = self._sessions.get(session_id)
        if rec is not None:
            self._sessions.move_to_end(session_id)
        return rec

    def hot_record(self) -> Optional[SessionRecord]:
        if not self.hot_session_id:
            return None
        return self.get(self.hot_session_id)

    def get_or_create(self, session_id: Optional[str],
                      *, cache_salt: str = "") -> SessionRecord:
        if session_id:
            rec = self.get(session_id)
            if rec is not None:
                return rec
            return self.create(session_id=session_id, cache_salt=cache_salt)
        return self.create(cache_salt=cache_salt)

    def delete(self, session_id: str) -> bool:
        existed = self._sessions.pop(session_id, None) is not None
        if self.hot_session_id == session_id:
            self.hot_session_id = None
        return existed

    def plan_request(self, session_id: Optional[str],
                     incoming: Sequence[int], *,
                     cache_salt: str = "") -> tuple[SessionRecord, PrefixPlan]:
        rec = self.get_or_create(session_id, cache_salt=cache_salt)
        plan = rec.plan(incoming)
        return rec, plan

    def mark_hot(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        self.hot_session_id = session_id
        self._sessions.move_to_end(session_id)

    def snapshot(self) -> Dict[str, object]:
        return {
            "max_sessions": self.max_sessions,
            "hot_session_id": self.hot_session_id,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "cached_len": s.cached_len,
                    "tokens": len(s.token_ids),
                    "protected": s.protected,
                    "last_digest": s.last_digest,
                }
                for s in self._sessions.values()
            ],
        }

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > self.max_sessions:
            victim_id = None
            for sid, rec in self._sessions.items():
                if not rec.protected and sid != self.hot_session_id:
                    victim_id = sid
                    break
            if victim_id is None:
                break
            self._sessions.pop(victim_id, None)


@dataclass
class CapsuleEntry:
    """A pinned execution-state capsule plus the metadata to match and bound it.

    ``capsule`` is the opaque frontend capsule object (from
    ``snapshot_capsule``); the store never inspects it beyond ``nbytes``.
    ``aligned_len`` is the chunk-aligned prefix length the capsule was snapshotted
    at, so a restore appends from there.
    """

    key: str
    aligned_len: int
    nbytes: int
    capsule: object


class CapsuleStore:
    """Serving-layer LRU of pinned shared-prefix capsules, bounded by a byte
    budget.

    Keyed by the token digest of the chunk-aligned shared prefix, so a capsule
    pinned by one session can be restored by any later request (or session) whose
    prompt starts with the same prefix. The budget bounds GPU footprint: pinning
    evicts least-recently-used capsules to fit, and a single capsule larger than
    the whole budget is rejected (the caller then serves a plain cold prefill —
    never an OOM). ``budget_bytes <= 0`` disables pinning entirely, keeping the
    default serving path byte-identical.
    """

    def __init__(self, *, budget_bytes: int = 0):
        self.budget_bytes = max(0, int(budget_bytes))
        self._entries: "OrderedDict[str, CapsuleEntry]" = OrderedDict()
        self._bytes = 0

    @property
    def enabled(self) -> bool:
        return self.budget_bytes > 0

    def get(self, key: str) -> Optional[CapsuleEntry]:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def pin(self, entry: CapsuleEntry) -> bool:
        """Store ``entry``, evicting LRU capsules to stay within the byte budget.

        Returns False (and stores nothing) when pinning is disabled or the single
        capsule alone exceeds the budget; the caller has already served a cold
        prefill, so a rejected pin just means no future restore.
        """
        if not self.enabled or entry.nbytes <= 0 or entry.nbytes > self.budget_bytes:
            return False
        existing = self._entries.pop(entry.key, None)
        if existing is not None:
            self._bytes -= existing.nbytes
        while self._bytes + entry.nbytes > self.budget_bytes and self._entries:
            _, victim = self._entries.popitem(last=False)
            self._bytes -= victim.nbytes
        self._entries[entry.key] = entry
        self._bytes += entry.nbytes
        return True

    def footprint(self) -> int:
        return self._bytes

    def snapshot(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "count": len(self._entries),
            "bytes": self._bytes,
            "budget_bytes": self.budget_bytes,
        }
