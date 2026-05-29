"""Session-aware agent service independent of the HTTP framework."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .engine import AgentEngine, DecodeChunk, GenerationStats
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

    def complete(self, req: AgentRequest) -> AgentResult:
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

        # v1 contiguous policy: only the currently hot session can reuse GPU
        # state.  Non-hot append/exact matches keep their token journal but must
        # rebuild until a checkpoint/offload backend lands.
        effective_cached = plan.cached_tokens
        if self.sessions.hot_session_id not in (None, session.session_id):
            effective_cached = 0
            plan = PrefixPlan(
                session_id=session.session_id,
                cached_tokens=0,
                new_prefill_tokens=len(prompt_tokens),
                incoming_tokens=len(prompt_tokens),
                matched_tokens=plan.matched_tokens,
                action="activate_rebuild",
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        t0 = time.perf_counter()
        self.engine.prefill(prompt_tokens, cached_tokens=effective_cached)
        t_prefill = time.perf_counter()

        parser = ToolCallStreamParser()
        events: List[StreamEvent] = []
        generated_ids: List[int] = []
        first_delta_ms = 0.0
        decode_started = time.perf_counter()
        for chunk in self.engine.generate_stream(max_tokens=req.max_tokens, K=req.K):
            generated_ids.extend(int(t) for t in chunk.token_ids)
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
        session.commit([*prompt_tokens, *generated_ids])
        self.sessions.mark_hot(session.session_id)

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


def request_from_openai(req: Dict[str, Any], *, default_k: int = 6) -> AgentRequest:
    messages = validate_messages(req.get("messages"))
    tools = validate_tools(req.get("tools"))
    max_tokens = int(req.get("max_tokens", 256))
    return AgentRequest(
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        stream=parse_bool(req.get("stream"), default=False),
        session_id=req.get("flashrt_session_id") or req.get("session_id"),
        cache_salt=str(req.get("flashrt_cache_salt", "")),
        enable_thinking=parse_bool(req.get("enable_thinking"), default=False),
        K=int(req.get("flashrt_K", default_k)),
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
