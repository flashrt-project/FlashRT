"""Session-aware agent service independent of the HTTP framework."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("qwen36_agent")

from .auto_prefix import AutoPrefixCacheManager
from .engine import AgentEngine, GenerationStats
from .openai_stream import (
    done_chunk,
    event_chunk,
    role_chunk,
    sse_data,
)
from .prefix import token_digest
from .session import CapsuleEntry, CapsuleStore, PrefixPlan, SessionRegistry
from .tool_stream import StreamEvent, ToolCallStreamParser


@dataclass
class AgentRequest:
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 2048
    stream: bool = False
    session_id: Optional[str] = None
    cache_salt: str = ""
    enable_thinking: bool = False
    K: int = 4
    # Pin the shared-prefix capsule: int = number of leading prompt tokens to pin
    # as a reusable capsule; True = pin the whole current prompt's aligned head;
    # None/0 = no pinning. Restore on a later request whose prompt starts with the
    # same chunk-aligned prefix. Effective only when the service has a capsule
    # budget and the prompt takes the long route.
    pin_prefix: Optional[int] = None


@dataclass
class AgentResult:
    completion_id: str
    session_id: str
    text: str
    tool_calls: List[Dict[str, Any]]
    finish_reason: str
    events: List[StreamEvent]
    usage: Dict[str, Any]
    stats: GenerationStats
    prefix_plan: PrefixPlan


class AgentService:
    """Policy layer over a Qwen3.6 split prefill/decode engine."""

    def __init__(self, engine: AgentEngine, *,
                 sessions: Optional[SessionRegistry] = None,
                 capsule_budget_bytes: int = 0,
                 default_k: int = 4,
                 default_max_tokens: int = 2048,
                 max_output_tokens: int = 8192,
                 default_session_id: Optional[str] = None):
        if default_k < 1:
            raise ValueError("default_k must be >= 1")
        if default_max_tokens < 1:
            raise ValueError("default_max_tokens must be >= 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if default_max_tokens > max_output_tokens:
            raise ValueError(
                "default_max_tokens must be <= max_output_tokens")
        self.engine = engine
        self.sessions = sessions or SessionRegistry()
        self.auto_prefix = AutoPrefixCacheManager(self.sessions)
        self.default_k = int(default_k)
        self.default_max_tokens = int(default_max_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.default_session_id = default_session_id or None
        # Pinned shared-prefix capsules (off-by-default: budget 0 keeps the
        # serving path byte-identical). A pinned capsule lets a fresh turn/session
        # restore a clean committed boundary instead of cold-prefilling the shared
        # prefix — the reuse path that survives EOS, unlike contiguous append.
        self.capsules = CapsuleStore(budget_bytes=capsule_budget_bytes)
        # The backend is a single hot GPU frontend with mutable KV / linear /
        # session state. Serialize whole requests so concurrent HTTP calls cannot
        # interleave prefill/decode and corrupt that state. A non-streaming call
        # holds the lock for the whole turn; a streaming call holds it for the
        # life of the generator (released when it is exhausted or closed).
        self._lock = threading.Lock()
        self._active_stream_committed = False

    def request_from_openai(self, req: Dict[str, Any]) -> AgentRequest:
        agent_req = request_from_openai(
            req,
            default_k=self.default_k,
            default_max_tokens=self.default_max_tokens,
            max_output_tokens=self.max_output_tokens,
        )
        if not agent_req.session_id and self.default_session_id:
            agent_req.session_id = self.default_session_id
        return agent_req

    def complete(self, req: AgentRequest) -> AgentResult:
        with self._lock:
            completed = False
            try:
                result = self._complete(req)
                completed = True
                return result
            finally:
                # If generation raised partway, the frontend KV may have advanced
                # while the journal did not — no session is safely hot. Rebuild.
                if not completed:
                    self.sessions.hot_session_id = None

    def stream_openai(self, req: AgentRequest, *,
                      model: str) -> Iterable[str]:
        with self._lock:
            completed = False
            self._active_stream_committed = False
            try:
                yield from self._stream_openai(req, model=model)
                completed = True
            finally:
                # Client disconnect closes this generator (GeneratorExit) before
                # the final commit / mark, and an exception aborts it too. Either
                # way the frontend state advanced past the journal, so clear the
                # hot session and force the next turn to rebuild.
                if not completed and not self._active_stream_committed:
                    self.sessions.hot_session_id = None
                self._active_stream_committed = False

    def _effective_plan(
            self, session, plan: PrefixPlan) -> tuple[int, PrefixPlan]:
        # v1 contiguous policy: only the currently hot session can reuse append
        # or exact GPU state. Non-hot matches and truncation keep their token
        # journal but rebuild until a checkpoint/rollback backend lands.
        effective_cached = plan.cached_tokens
        needs_rebuild = (
            plan.cached_tokens
            and (self.sessions.hot_session_id != session.session_id
                 or plan.action == "truncate")
        )
        if needs_rebuild:
            effective_cached = 0
            plan = PrefixPlan(
                session_id=session.session_id,
                cached_tokens=0,
                new_prefill_tokens=plan.incoming_tokens,
                incoming_tokens=plan.incoming_tokens,
                matched_tokens=plan.matched_tokens,
                action="activate_rebuild",
            )
        return effective_cached, plan

    def _capsule_prefill(
            self, req: AgentRequest, prompt_tokens: List[int], session, *,
            max_tokens: int, K: int
    ) -> Optional[PrefixPlan]:
        """Restore-or-pin a shared-prefix capsule when ``pin_prefix`` is requested
        and viable, performing the prefill on the engine and returning its
        PrefixPlan. Returns None only when the request did not ask for capsule
        pinning. If the request does ask for pinning, fail fast on unsupported
        configs instead of silently falling back to a different prefill route.

        A pinned capsule is keyed by the digest of its chunk-aligned prefix tokens,
        so a later turn or a different session whose prompt starts with the same
        prefix restores a clean committed boundary instead of cold-prefilling it.
        Unlike contiguous append, this survives an EOS-terminated previous turn.
        On a budget-rejected pin the request is already served cold; only a future
        restore is lost (never an OOM, never a false hit — the restore key is an
        exact aligned-prefix digest match).
        """
        pin = req.pin_prefix
        if not pin:
            return None
        if not self.capsules.enabled:
            raise ValueError(
                "flashrt_pin_prefix requires --capsule-budget-mb > 0")
        supports = getattr(self.engine, "supports_capsule", None)
        if not callable(supports) or not supports():
            raise ValueError(
                "flashrt_pin_prefix requires a capsule-capable Qwen3.6 engine")
        prompt_len = len(prompt_tokens)
        pin_len = prompt_len if pin is True else min(int(pin), prompt_len)
        if pin_len <= 0:
            return None
        aligned = self.engine.capsule_aligned_len(pin_len, max_tokens)
        if aligned <= 0 or aligned > prompt_len:
            raise ValueError(
                "flashrt_pin_prefix requires the long FP8-KV route and a "
                "chunk-aligned prefix; start the server with a long-context "
                "max_seq, --route-min-seq 0, and "
                "FLASHRT_QWEN36_LONG_KV_CACHE=fp8")
        key = token_digest(prompt_tokens[:aligned], salt=req.cache_salt)
        entry = self.capsules.get(key)
        if entry is not None:
            self.engine.prefill_from_capsule(
                entry.capsule, prompt_tokens,
                max_tokens=max_tokens, K=K)
            return PrefixPlan(
                session_id=session.session_id,
                cached_tokens=aligned,
                new_prefill_tokens=max(0, prompt_len - aligned),
                incoming_tokens=prompt_len,
                matched_tokens=aligned,
                action="restore",
            )
        cap = self.engine.prefill_and_pin(
            prompt_tokens, aligned_len=aligned,
            max_tokens=max_tokens, K=K)
        nbytes = int(cap.get("nbytes", 0)) if isinstance(cap, dict) else 0
        pinned = self.capsules.pin(CapsuleEntry(
            key=key, aligned_len=aligned, nbytes=nbytes, capsule=cap))
        return PrefixPlan(
            session_id=session.session_id,
            cached_tokens=0,
            new_prefill_tokens=prompt_len,
            incoming_tokens=prompt_len,
            matched_tokens=0,
            action="pin" if pinned else "rebuild",
        )

    def _message_append_prompt_tokens(
            self, session, req: AgentRequest, plan: PrefixPlan
    ) -> tuple[Optional[List[int]], Optional[PrefixPlan]]:
        if self.sessions.hot_session_id != session.session_id:
            return None, None
        previous = getattr(session, "visible_messages", None)
        if not previous or not hasattr(
                self.engine, "append_suffix_tokens_for_messages"):
            return None, None
        previous_for_suffix = previous
        incoming_prefix = req.messages[:len(previous)]
        if incoming_prefix != previous:
            if not self._messages_equivalent_prefix(previous, incoming_prefix):
                return None, None
            previous_for_suffix = incoming_prefix
        suffix = self.engine.append_suffix_tokens_for_messages(
            previous_for_suffix,
            req.messages,
            tools=req.tools,
            enable_thinking=req.enable_thinking,
        )
        if not suffix:
            return None, None
        cached = len(session.token_ids)
        return [*session.token_ids, *suffix], PrefixPlan(
            session_id=session.session_id,
            cached_tokens=cached,
            new_prefill_tokens=len(suffix),
            incoming_tokens=plan.incoming_tokens,
            matched_tokens=plan.matched_tokens,
            action="message_append",
        )

    @classmethod
    def _messages_equivalent_prefix(
            cls, previous: List[Dict[str, Any]],
            incoming: List[Dict[str, Any]]) -> bool:
        if len(incoming) != len(previous):
            return False
        return all(cls._messages_equivalent(a, b)
                   for a, b in zip(previous, incoming))

    @classmethod
    def _messages_equivalent(cls, a: Dict[str, Any],
                             b: Dict[str, Any]) -> bool:
        if a.get("role") != b.get("role"):
            return False
        if (a.get("content") or "") != (b.get("content") or ""):
            return False
        return cls._tool_calls_equivalent(
            a.get("tool_calls"), b.get("tool_calls"))

    @staticmethod
    def _tool_calls_equivalent(a: Any, b: Any) -> bool:
        if not a and not b:
            return True
        if not isinstance(a, list) or not isinstance(b, list):
            return False
        if len(a) != len(b):
            return False

        def norm_call(tc: Any) -> Any:
            if not isinstance(tc, dict):
                return tc
            fn = tc.get("function")
            if not isinstance(fn, dict):
                return fn
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except Exception:
                    pass
            return {
                "type": tc.get("type", "function"),
                "name": fn.get("name"),
                "arguments": args,
            }

        return [norm_call(x) for x in a] == [norm_call(x) for x in b]

    @staticmethod
    def _effective_k(req: AgentRequest) -> int:
        k = int(req.K)
        if k < 1:
            raise ValueError("flashrt_K must be >= 1")
        return k

    def _log_reuse_miss(self, session, plan: PrefixPlan,
                        incoming_messages: List[Dict[str, Any]]) -> None:
        if plan.action in ("append", "exact", "restore", "message_append"):
            return
        previous = getattr(session, "visible_messages", None) or []

        def msg_shape(msg: Dict[str, Any]) -> str:
            role = str(msg.get("role", "?"))
            has_tools = bool(msg.get("tool_calls"))
            content = msg.get("content")
            if content is None:
                ckind = "none"
            elif isinstance(content, str):
                ckind = f"str:{len(content)}"
            elif isinstance(content, list):
                ckind = f"list:{len(content)}"
            else:
                ckind = type(content).__name__
            return f"{role}/{ckind}/tools={int(has_tools)}"

        log.info(
            "reuse_miss sid=%s action=%s matched=%d cached_len=%d "
            "prev_tokens=%d incoming_tokens=%d hot=%s prev_msgs=%d "
            "incoming_msgs=%d prev_tail=%s incoming_tail=%s",
            session.session_id, plan.action, plan.matched_tokens,
            session.cached_len, len(session.token_ids),
            plan.incoming_tokens, self.sessions.hot_session_id,
            len(previous), len(incoming_messages),
            [msg_shape(m) for m in previous[-4:]],
            [msg_shape(m) for m in incoming_messages[-4:]],
        )

    @staticmethod
    def _copy_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(m) for m in messages]

    @staticmethod
    def _assistant_message(text: str,
                           tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": text if text or not tool_calls else None,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _mark_reusable(self, session, state_lookahead: bool) -> None:
        """Mark the session hot (reusable for append) only if the GPU state ends
        exactly at the committed transcript. If a stop token left committed
        lookahead, the frontend KV leads the journal, so no session is safely
        appendable until the next cold prefill resets it: clear the hot session
        and force a rebuild."""
        if state_lookahead:
            self.sessions.hot_session_id = None
        else:
            self.sessions.mark_hot(session.session_id)

    def _select_prefix_session(
            self, req: AgentRequest, prompt_tokens: List[int]):
        def can_message_append(session) -> bool:
            previous = getattr(session, "visible_messages", None)
            if not previous:
                return False
            incoming_prefix = req.messages[:len(previous)]
            return (
                incoming_prefix == previous
                or self._messages_equivalent_prefix(previous, incoming_prefix)
            )

        selected = self.auto_prefix.select(
            req.session_id,
            prompt_tokens,
            cache_salt=req.cache_salt,
            can_message_append=can_message_append,
        )
        return selected.session, selected.plan

    def _effective_max_tokens(self, req: AgentRequest,
                              prompt_len: int) -> int:
        max_tokens = int(req.max_tokens)
        max_seq = int(getattr(self.engine, "max_seq", 0) or 0)
        if max_seq <= 0:
            return max_tokens
        remaining = max_seq - int(prompt_len)
        if remaining < 1:
            raise ValueError(
                f"prompt length {int(prompt_len)} leaves no room under "
                f"max_seq {max_seq}; reduce context or start with a larger "
                "server --max-seq")
        if max_tokens > remaining:
            log.warning(
                "clipping max_tokens from %d to %d for prompt=%d max_seq=%d",
                max_tokens, remaining, int(prompt_len), max_seq)
            return remaining
        return max_tokens

    def validate_request_bounds(self, req: AgentRequest) -> None:
        """Fail hard context-limit errors before a StreamingResponse starts.

        Soft output-budget overflow is handled by clipping in the request path;
        a prompt that already fills the context must be rejected before Starlette
        begins SSE streaming, otherwise the client sees a broken stream and the
        server logs an ASGI traceback.
        """
        tokenizer = getattr(
            self.engine, "tokenize_chat_for_validation", None)
        if not callable(tokenizer):
            tokenizer = self.engine.tokenize_chat
        prompt_tokens = tokenizer(
            req.messages,
            tools=req.tools,
            enable_thinking=req.enable_thinking,
        )
        self._effective_max_tokens(req, len(prompt_tokens))

    @staticmethod
    def _fmt_metric_line(
            kind: str, *, session_id: str, action: str,
            prompt_tokens: int, cached_tokens: int, new_prefill_tokens: int,
            completion_tokens: int, prefill_ms: float,
            first_delta_ms: float, decode_ms: float,
            decode_tok_per_s: float, finish: str, tool_calls: int,
            state_lookahead: bool, hot_after: Optional[str], K: int,
            stream_wall_ms: Optional[float] = None,
            stream_wall_tok_per_s: Optional[float] = None) -> str:
        """Stable one-line serving metrics optimized for terminal scanning."""
        parts = [
            f"{kind:<8}",
            f"sid={session_id}",
            f"act={action:<14}",
            (
                f"tok p={prompt_tokens:>6} cache={cached_tokens:>6} "
                f"new={new_prefill_tokens:>5} out={completion_tokens:>4}"
            ),
            (
                f"ms prefill={prefill_ms:>7.1f} "
                f"ttft={first_delta_ms:>7.1f} decode={decode_ms:>7.1f}"
            ),
            f"speed decode={decode_tok_per_s:>6.1f} tok/s",
        ]
        if stream_wall_ms is not None and stream_wall_tok_per_s is not None:
            parts.append(
                f"stream={stream_wall_ms:>7.1f}ms/"
                f"{stream_wall_tok_per_s:>6.1f} tok/s")
        parts.extend([
            f"finish={finish}",
            f"tools={tool_calls}",
            f"lookahead={int(state_lookahead)}",
            f"hot={hot_after}",
            f"K={K}",
        ])
        return " | ".join(parts)

    def _complete(self, req: AgentRequest) -> AgentResult:
        if req.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        prompt_tokens = self.engine.tokenize_chat(
            req.messages,
            tools=req.tools,
            enable_thinking=req.enable_thinking,
        )
        max_tokens = self._effective_max_tokens(req, len(prompt_tokens))
        session, plan = self._select_prefix_session(req, prompt_tokens)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        engine_prompt_tokens = prompt_tokens
        decode_k = self._effective_k(req)
        t0 = time.perf_counter()
        cap_plan = self._capsule_prefill(
            req, prompt_tokens, session, max_tokens=max_tokens, K=decode_k)
        if cap_plan is not None:
            plan = cap_plan
        else:
            msg_prompt, msg_plan = self._message_append_prompt_tokens(
                session, req, plan)
            if msg_prompt is not None and msg_plan is not None:
                engine_prompt_tokens = msg_prompt
                plan = msg_plan
                effective_cached = plan.cached_tokens
            else:
                effective_cached, plan = self._effective_plan(session, plan)
                self._log_reuse_miss(session, plan, req.messages)
            self.engine.prefill(
                engine_prompt_tokens,
                cached_tokens=effective_cached,
                max_tokens=max_tokens,
                K=decode_k,
            )
        t_prefill = time.perf_counter()

        parser = ToolCallStreamParser()
        events: List[StreamEvent] = []
        generated_ids: List[int] = []
        first_delta_ms = 0.0
        decode_started = time.perf_counter()
        state_lookahead = False
        saw_tool_call = False
        for chunk in self.engine.generate_stream(
                max_tokens=max_tokens, K=decode_k):
            generated_ids.extend(int(t) for t in chunk.token_ids)
            if getattr(chunk, "state_lookahead", 0):
                state_lookahead = True
            evs = parser.feed(chunk.text)
            if evs and first_delta_ms <= 0.0:
                first_delta_ms = (time.perf_counter() - t0) * 1000.0
            events.extend(evs)
            if any(ev.kind == "tool_call" for ev in evs):
                saw_tool_call = True
                break
        tail = parser.finish()
        if tail and first_delta_ms <= 0.0:
            first_delta_ms = (time.perf_counter() - t0) * 1000.0
        events.extend(tail)
        if any(ev.kind == "tool_call" for ev in tail):
            saw_tool_call = True
        t_done = time.perf_counter()

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for ev in events:
            if ev.kind == "tool_call":
                tool_calls.append(ev.payload)
            else:
                text_parts.append(str(ev.payload))
        text = "".join(text_parts)
        finish_reason = "tool_calls" if tool_calls else "stop"
        session.commit([*engine_prompt_tokens, *generated_ids])
        visible_messages = self._copy_messages(req.messages)
        visible_messages.append(self._assistant_message(text, tool_calls))
        session.visible_messages = visible_messages
        self._mark_reusable(session, state_lookahead)

        completion_tokens = len(generated_ids)
        decode_ms = max(0.0, (t_done - decode_started) * 1000.0)
        decode_tok_per_s = (
            completion_tokens * 1000.0 / decode_ms if decode_ms > 0 else 0.0
        )
        stats = GenerationStats(
            prompt_tokens=len(prompt_tokens),
            cached_tokens=plan.cached_tokens,
            new_prefill_tokens=plan.new_prefill_tokens,
            prefill_ms=(t_prefill - t0) * 1000.0,
            first_delta_ms=first_delta_ms,
            decode_ms=decode_ms,
            decode_tok_per_s=decode_tok_per_s,
        )
        usage = {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt_tokens) + completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": int(plan.cached_tokens),
            },
        }
        log.info(self._fmt_metric_line(
            "complete",
            session_id=session.session_id,
            action=plan.action,
            prompt_tokens=len(prompt_tokens),
            cached_tokens=stats.cached_tokens,
            new_prefill_tokens=stats.new_prefill_tokens,
            completion_tokens=completion_tokens,
            prefill_ms=stats.prefill_ms,
            first_delta_ms=stats.first_delta_ms,
            decode_ms=stats.decode_ms,
            decode_tok_per_s=stats.decode_tok_per_s,
            finish=finish_reason,
            tool_calls=len(tool_calls),
            state_lookahead=state_lookahead,
            hot_after=self.sessions.hot_session_id,
            K=decode_k,
        ))
        return AgentResult(
            completion_id=completion_id,
            session_id=session.session_id,
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            events=events,
            usage=usage,
            stats=stats,
            prefix_plan=plan,
        )

    def _stream_openai(self, req: AgentRequest, *,
                       model: str) -> Iterable[str]:
        """Yield OpenAI-compatible SSE chunks as decode commits tokens."""
        if req.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        prompt_tokens = self.engine.tokenize_chat(
            req.messages,
            tools=req.tools,
            enable_thinking=req.enable_thinking,
        )
        max_tokens = self._effective_max_tokens(req, len(prompt_tokens))
        session, plan = self._select_prefix_session(req, prompt_tokens)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        engine_prompt_tokens = prompt_tokens
        decode_k = self._effective_k(req)
        t0 = time.perf_counter()
        cap_plan = self._capsule_prefill(
            req, prompt_tokens, session, max_tokens=max_tokens, K=decode_k)
        if cap_plan is not None:
            plan = cap_plan
        else:
            msg_prompt, msg_plan = self._message_append_prompt_tokens(
                session, req, plan)
            if msg_prompt is not None and msg_plan is not None:
                engine_prompt_tokens = msg_prompt
                plan = msg_plan
                effective_cached = msg_plan.cached_tokens
            else:
                effective_cached, plan = self._effective_plan(session, plan)
                self._log_reuse_miss(session, plan, req.messages)
            self.engine.prefill(
                engine_prompt_tokens,
                cached_tokens=effective_cached,
                max_tokens=max_tokens,
                K=decode_k,
            )
        t_prefill = time.perf_counter()

        parser = ToolCallStreamParser()
        generated_ids: List[int] = []
        visible_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        deferred_tool_events: List[StreamEvent] = []
        seen_tool_call = False
        state_lookahead = False
        yield sse_data(role_chunk(completion_id, model))
        first_delta_ms = 0.0
        stream_started = time.perf_counter()
        backend_decode_ms = 0.0
        saw_tool_call = False
        chunks = iter(self.engine.generate_stream(max_tokens=max_tokens,
                                                  K=decode_k))
        while True:
            next_t0 = time.perf_counter()
            try:
                chunk = next(chunks)
            except StopIteration:
                backend_decode_ms += (
                    time.perf_counter() - next_t0) * 1000.0
                break
            backend_decode_ms += (time.perf_counter() - next_t0) * 1000.0
            generated_ids.extend(int(t) for t in chunk.token_ids)
            if getattr(chunk, "state_lookahead", 0):
                state_lookahead = True
            for ev in parser.feed(chunk.text):
                if first_delta_ms <= 0.0:
                    first_delta_ms = (time.perf_counter() - t0) * 1000.0
                if ev.kind == "tool_call":
                    seen_tool_call = True
                    saw_tool_call = True
                    tool_calls.append(ev.payload)
                    deferred_tool_events.append(ev)
                else:
                    visible_parts.append(str(ev.payload))
                    yield sse_data(event_chunk(completion_id, model, ev))
            if saw_tool_call:
                break
        for ev in parser.finish():
            if first_delta_ms <= 0.0:
                first_delta_ms = (time.perf_counter() - t0) * 1000.0
            if ev.kind == "tool_call":
                seen_tool_call = True
                saw_tool_call = True
                tool_calls.append(ev.payload)
                deferred_tool_events.append(ev)
            else:
                visible_parts.append(str(ev.payload))
                yield sse_data(event_chunk(completion_id, model, ev))
        t_done = time.perf_counter()

        session.commit([*engine_prompt_tokens, *generated_ids])
        visible_messages = self._copy_messages(req.messages)
        visible_messages.append(self._assistant_message(
            "".join(visible_parts), tool_calls))
        session.visible_messages = visible_messages
        self._mark_reusable(session, state_lookahead)
        self._active_stream_committed = True
        usage = {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(prompt_tokens) + len(generated_ids),
            "prompt_tokens_details": {
                "cached_tokens": int(plan.cached_tokens),
            },
        }
        completion_tokens = len(generated_ids)
        decode_ms = max(0.0, backend_decode_ms)
        decode_tok_per_s = (
            completion_tokens * 1000.0 / decode_ms if decode_ms > 0 else 0.0
        )
        stream_wall_ms = max(0.0, (t_done - stream_started) * 1000.0)
        stream_wall_tok_per_s = (
            completion_tokens * 1000.0 / stream_wall_ms
            if stream_wall_ms > 0 else 0.0
        )
        log.info(self._fmt_metric_line(
            "stream",
            session_id=session.session_id,
            action=plan.action,
            prompt_tokens=len(prompt_tokens),
            cached_tokens=plan.cached_tokens,
            new_prefill_tokens=plan.new_prefill_tokens,
            completion_tokens=completion_tokens,
            prefill_ms=(t_prefill - t0) * 1000.0,
            first_delta_ms=first_delta_ms,
            decode_ms=decode_ms,
            decode_tok_per_s=decode_tok_per_s,
            stream_wall_ms=stream_wall_ms,
            stream_wall_tok_per_s=stream_wall_tok_per_s,
            finish="tool_calls" if seen_tool_call else "stop",
            tool_calls=len(tool_calls),
            state_lookahead=state_lookahead,
            hot_after=self.sessions.hot_session_id,
            K=decode_k,
        ))
        for ev in deferred_tool_events:
            yield sse_data(event_chunk(completion_id, model, ev))
        yield sse_data(done_chunk(
            completion_id,
            model,
            finish_reason="tool_calls" if seen_tool_call else "stop",
            usage=usage,
        ))
        yield "data: [DONE]\n\n"


def validate_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages is required (non-empty list)")
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("each message must be an object")
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unsupported role: {role!r}")
        content = msg.get("content")
        if content is None and role == "assistant":
            continue
        if not isinstance(content, str):
            raise ValueError("message.content must be a string")
    return messages


def validate_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("each tool must be an object")
    return tools


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
    raise ValueError("expected boolean")


def parse_int(value: Any, *, name: str, default: int) -> int:
    """Coerce an OpenAI request field to int, raising ValueError (which the HTTP
    layer maps to 400) rather than TypeError (which would surface as a 500) on a
    null / non-numeric value."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got a boolean")
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}")


def parse_pin_prefix(value: Any) -> Optional[int]:
    """Parse ``flashrt_pin_prefix``: a positive int (pin that many leading prompt
    tokens), ``true`` (pin the whole current prompt's aligned head), or
    absent/false/0/null (no pinning)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return True if value else None
    if isinstance(value, int):
        return value if value > 0 else None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"flashrt_pin_prefix must be an integer or boolean, got {value!r}")
    return n if n > 0 else None


def request_from_openai(req: Dict[str, Any], *, default_k: int = 4,
                        default_max_tokens: int = 2048,
                        max_output_tokens: Optional[int] = 8192
                        ) -> AgentRequest:
    if default_max_tokens < 1:
        raise ValueError("default_max_tokens must be >= 1")
    if max_output_tokens is not None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if default_max_tokens > max_output_tokens:
            raise ValueError(
                "default_max_tokens must be <= max_output_tokens")
    messages = validate_messages(req.get("messages"))
    tools = validate_tools(req.get("tools"))
    # Fall back to max_completion_tokens only when max_tokens is absent *or*
    # explicitly null: dict.get("max_tokens", fallback) returns None (not the
    # fallback) when the key is present with a null value, which would drop a
    # caller's max_completion_tokens.
    raw_max_tokens = req.get("max_tokens")
    if raw_max_tokens is None:
        raw_max_tokens = req.get("max_completion_tokens")
    max_tokens = parse_int(
        raw_max_tokens, name="max_tokens", default=default_max_tokens)
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if max_output_tokens is not None and max_tokens > int(max_output_tokens):
        raise ValueError(
            f"max_tokens must be <= {int(max_output_tokens)}")
    K = parse_int(req.get("flashrt_K"), name="flashrt_K", default=default_k)
    if K < 1:
        raise ValueError("flashrt_K must be >= 1")
    return AgentRequest(
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        stream=parse_bool(req.get("stream"), default=False),
        session_id=req.get("flashrt_session_id") or req.get("session_id"),
        cache_salt=str(
            req.get("prompt_cache_key")
            or req.get("cache_salt")
            or req.get("flashrt_cache_salt", "")
        ),
        enable_thinking=parse_bool(req.get("enable_thinking"), default=False),
        K=K,
        pin_prefix=parse_pin_prefix(req.get("flashrt_pin_prefix")),
    )


def result_to_openai(result: AgentResult, *, model: str) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": result.text if result.text or not result.tool_calls else None,
    }
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    return {
        "id": result.completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": result.finish_reason,
        }],
        "usage": result.usage,
        "flashrt": {
            "session_id": result.session_id,
            "cached_tokens": result.stats.cached_tokens,
            "new_prefill_tokens": result.stats.new_prefill_tokens,
            "prefill_ms": result.stats.prefill_ms,
            "first_delta_ms": result.stats.first_delta_ms,
            "decode_ms": result.stats.decode_ms,
            "decode_tok_per_s": result.stats.decode_tok_per_s,
            "prefix_action": result.prefix_plan.action,
        },
    }
