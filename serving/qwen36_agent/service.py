"""Session-aware agent service independent of the HTTP framework."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("qwen36_agent")

from .engine import AgentEngine, GenerationStats
from .openai_stream import (
    done_chunk,
    event_chunk,
    role_chunk,
    sse_data,
)
from .session import PrefixPlan, SessionRegistry
from .tool_stream import StreamEvent, ToolCallStreamParser


@dataclass
class AgentRequest:
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 256
    stream: bool = False
    session_id: Optional[str] = None
    cache_salt: str = ""
    enable_thinking: bool = False
    K: int = 6


@dataclass
class AgentResult:
    completion_id: str
    session_id: str
    text: str
    tool_calls: List[Dict[str, Any]]
    finish_reason: str
    events: List[StreamEvent]
    usage: Dict[str, int]
    stats: GenerationStats
    prefix_plan: PrefixPlan


class AgentService:
    """Policy layer over a Qwen3.6 split prefill/decode engine."""

    def __init__(self, engine: AgentEngine, *,
                 sessions: Optional[SessionRegistry] = None):
        self.engine = engine
        self.sessions = sessions or SessionRegistry()
        # The backend is a single hot GPU frontend with mutable KV / linear /
        # session state. Serialize whole requests so concurrent HTTP calls cannot
        # interleave prefill/decode and corrupt that state. A non-streaming call
        # holds the lock for the whole turn; a streaming call holds it for the
        # life of the generator (released when it is exhausted or closed).
        self._lock = threading.Lock()

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
            try:
                yield from self._stream_openai(req, model=model)
                completed = True
            finally:
                # Client disconnect closes this generator (GeneratorExit) before
                # the final commit / mark, and an exception aborts it too. Either
                # way the frontend state advanced past the journal, so clear the
                # hot session and force the next turn to rebuild.
                if not completed:
                    self.sessions.hot_session_id = None

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

    def _message_append_prompt_tokens(
            self, session, req: AgentRequest, plan: PrefixPlan
    ) -> tuple[Optional[List[int]], Optional[PrefixPlan]]:
        if self.sessions.hot_session_id != session.session_id:
            return None, None
        previous = getattr(session, "visible_messages", None)
        if not previous or not hasattr(
                self.engine, "append_suffix_tokens_for_messages"):
            return None, None
        suffix = self.engine.append_suffix_tokens_for_messages(
            previous,
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

    @staticmethod
    def _copy_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(m) for m in messages]

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

    def _complete(self, req: AgentRequest) -> AgentResult:
        if req.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        prompt_tokens = self.engine.tokenize_chat(
            req.messages,
            tools=req.tools,
            enable_thinking=req.enable_thinking,
        )
        session, plan = self.sessions.plan_request(
            req.session_id,
            prompt_tokens,
            cache_salt=req.cache_salt,
        )

        engine_prompt_tokens = prompt_tokens
        msg_prompt, msg_plan = self._message_append_prompt_tokens(
            session, req, plan)
        if msg_prompt is not None and msg_plan is not None:
            engine_prompt_tokens = msg_prompt
            plan = msg_plan
            effective_cached = plan.cached_tokens
        else:
            effective_cached, plan = self._effective_plan(session, plan)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        t0 = time.perf_counter()
        self.engine.prefill(
            engine_prompt_tokens,
            cached_tokens=effective_cached,
            max_tokens=req.max_tokens,
            K=req.K,
        )
        t_prefill = time.perf_counter()

        parser = ToolCallStreamParser()
        events: List[StreamEvent] = []
        generated_ids: List[int] = []
        first_delta_ms = 0.0
        decode_started = time.perf_counter()
        state_lookahead = False
        for chunk in self.engine.generate_stream(max_tokens=req.max_tokens, K=req.K):
            generated_ids.extend(int(t) for t in chunk.token_ids)
            if getattr(chunk, "state_lookahead", 0):
                state_lookahead = True
            evs = parser.feed(chunk.text)
            if evs and first_delta_ms <= 0.0:
                first_delta_ms = (time.perf_counter() - t0) * 1000.0
            events.extend(evs)
        tail = parser.finish()
        if tail and first_delta_ms <= 0.0:
            first_delta_ms = (time.perf_counter() - t0) * 1000.0
        events.extend(tail)
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
        visible_messages.append({
            "role": "assistant",
            "content": text,
        })
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
        }
        log.info(
            "complete sid=%s action=%s prompt=%d cached=%d new_prefill=%d "
            "completion=%d prefill_ms=%.1f first_delta_ms=%.1f decode_ms=%.1f "
            "decode_tok/s=%.1f",
            session.session_id, plan.action, len(prompt_tokens),
            stats.cached_tokens, stats.new_prefill_tokens, completion_tokens,
            stats.prefill_ms, stats.first_delta_ms, stats.decode_ms,
            stats.decode_tok_per_s,
        )
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
        session, plan = self.sessions.plan_request(
            req.session_id,
            prompt_tokens,
            cache_salt=req.cache_salt,
        )
        engine_prompt_tokens = prompt_tokens
        msg_prompt, msg_plan = self._message_append_prompt_tokens(
            session, req, plan)
        if msg_prompt is not None and msg_plan is not None:
            engine_prompt_tokens = msg_prompt
            effective_cached = msg_plan.cached_tokens
        else:
            effective_cached, _ = self._effective_plan(session, plan)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.engine.prefill(
            engine_prompt_tokens,
            cached_tokens=effective_cached,
            max_tokens=req.max_tokens,
            K=req.K,
        )

        parser = ToolCallStreamParser()
        generated_ids: List[int] = []
        visible_parts: List[str] = []
        seen_tool_call = False
        state_lookahead = False
        yield sse_data(role_chunk(completion_id, model))
        for chunk in self.engine.generate_stream(max_tokens=req.max_tokens, K=req.K):
            generated_ids.extend(int(t) for t in chunk.token_ids)
            if getattr(chunk, "state_lookahead", 0):
                state_lookahead = True
            for ev in parser.feed(chunk.text):
                if ev.kind == "tool_call":
                    seen_tool_call = True
                else:
                    visible_parts.append(str(ev.payload))
                yield sse_data(event_chunk(completion_id, model, ev))
        for ev in parser.finish():
            if ev.kind == "tool_call":
                seen_tool_call = True
            else:
                visible_parts.append(str(ev.payload))
            yield sse_data(event_chunk(completion_id, model, ev))

        session.commit([*engine_prompt_tokens, *generated_ids])
        visible_messages = self._copy_messages(req.messages)
        visible_messages.append({
            "role": "assistant",
            "content": "".join(visible_parts),
        })
        session.visible_messages = visible_messages
        self._mark_reusable(session, state_lookahead)
        usage = {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(prompt_tokens) + len(generated_ids),
        }
        log.info(
            "stream sid=%s action=%s prompt=%d completion=%d",
            session.session_id, plan.action, len(prompt_tokens),
            len(generated_ids),
        )
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


def request_from_openai(req: Dict[str, Any], *, default_k: int = 6) -> AgentRequest:
    messages = validate_messages(req.get("messages"))
    tools = validate_tools(req.get("tools"))
    # Fall back to max_completion_tokens only when max_tokens is absent *or*
    # explicitly null: dict.get("max_tokens", fallback) returns None (not the
    # fallback) when the key is present with a null value, which would drop a
    # caller's max_completion_tokens.
    raw_max_tokens = req.get("max_tokens")
    if raw_max_tokens is None:
        raw_max_tokens = req.get("max_completion_tokens")
    max_tokens = parse_int(raw_max_tokens, name="max_tokens", default=256)
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    K = parse_int(req.get("flashrt_K"), name="flashrt_K", default=default_k)
    return AgentRequest(
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        stream=parse_bool(req.get("stream"), default=False),
        session_id=req.get("flashrt_session_id") or req.get("session_id"),
        cache_salt=str(req.get("flashrt_cache_salt", "")),
        enable_thinking=parse_bool(req.get("enable_thinking"), default=False),
        K=K,
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
