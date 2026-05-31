"""Automatic prefix selection for OpenAI-compatible agent clients.

OpenAI-compatible clients normally resend the full message history and do not
carry a FlashRT session id.  This module keeps the session id as an optional
native hint, but lets the serving layer first try the current hot execution
state by token/content prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .session import PrefixPlan, SessionRecord, SessionRegistry


MessageAppendPredicate = Callable[[SessionRecord], bool]


@dataclass(frozen=True)
class AutoPrefixSelection:
    session: SessionRecord
    plan: PrefixPlan
    source: str


class AutoPrefixCacheManager:
    """Session-less prefix planner over the existing hot execution state.

    This is intentionally not a vLLM/SGLang block cache.  It only decides
    whether an incoming request can attach to the single hot FlashRT frontend
    state, or should fall back to a normal per-session/cold plan.  Capsule
    restore/pinning remains a separate execution policy in the service.
    """

    def __init__(self, sessions: SessionRegistry):
        self.sessions = sessions

    def select(
            self, session_id: Optional[str], incoming: Sequence[int], *,
            cache_salt: str = "",
            can_message_append: Optional[MessageAppendPredicate] = None
    ) -> AutoPrefixSelection:
        if session_id:
            session, plan = self.sessions.plan_request(
                session_id, incoming, cache_salt=cache_salt)
            return AutoPrefixSelection(session, plan, "explicit_session")

        hot = self.sessions.hot_record()
        if hot is not None and hot.cache_salt == cache_salt:
            plan = hot.plan(incoming)
            if plan.action in ("exact", "append", "truncate"):
                return AutoPrefixSelection(hot, plan, "hot_token_prefix")
            if can_message_append is not None and can_message_append(hot):
                return AutoPrefixSelection(hot, plan, "hot_message_prefix")

        session, plan = self.sessions.plan_request(
            None, incoming, cache_salt=cache_salt)
        return AutoPrefixSelection(session, plan, "new_session")
