"""FastAPI shell for Qwen3.6 agent serving.

The HTTP layer is intentionally thin: all cache and streaming policy lives in
``service.py`` and all compute goes through an ``AgentEngine`` implementation.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from .openai_stream import sse_from_events
from .service import AgentService, request_from_openai, result_to_openai

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


def build_app(service: AgentService):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="FlashRT Qwen3.6 Agent Serving")

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": service.engine.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "flash-rt",
            }],
        }

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": service.engine.model_name,
            "max_seq": service.engine.max_seq,
            "sessions": service.sessions.snapshot(),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(raw: Dict[str, Any]):
        try:
            req = request_from_openai(raw)
            result = service.complete(req)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        if req.stream:
            return StreamingResponse(
                sse_from_events(
                    result.completion_id,
                    service.engine.model_name,
                    result.events,
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                ),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )
        return result_to_openai(result, model=service.engine.model_name)

    @app.post("/v1/sessions")
    async def create_session(raw: Dict[str, Any] | None = None):
        raw = raw or {}
        rec = service.sessions.create(
            session_id=raw.get("session_id"),
            cache_salt=str(raw.get("cache_salt", "")),
            protected=bool(raw.get("protected", False)),
        )
        return {"session_id": rec.session_id}

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        return {"deleted": service.sessions.delete(session_id)}

    return app
