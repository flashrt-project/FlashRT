"""OpenAI-compatible SSE helpers for agent serving."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable

from .tool_stream import StreamEvent


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def sse_data(obj: Any) -> str:
    return f"data: {json_dumps(obj)}\n\n"


def role_chunk(completion_id: str, model: str) -> Dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }


def event_chunk(completion_id: str, model: str,
                event: StreamEvent) -> Dict[str, Any]:
    delta: Dict[str, Any]
    if event.kind == "tool_call":
        delta = {"tool_calls": [event.payload]}
    else:
        delta = {"content": str(event.payload)}
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None,
        }],
    }


def done_chunk(completion_id: str, model: str, *,
               finish_reason: str = "stop",
               usage: Dict[str, int] | None = None) -> Dict[str, Any]:
    out = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        out["usage"] = usage
    return out


def sse_from_events(completion_id: str, model: str,
                    events: Iterable[StreamEvent], *,
                    finish_reason: str = "stop",
                    usage: Dict[str, int] | None = None) -> Iterable[str]:
    yield sse_data(role_chunk(completion_id, model))
    for ev in events:
        yield sse_data(event_chunk(completion_id, model, ev))
    yield sse_data(done_chunk(
        completion_id, model, finish_reason=finish_reason, usage=usage))
    yield "data: [DONE]\n\n"
