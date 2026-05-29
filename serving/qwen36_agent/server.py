"""FastAPI shell for Qwen3.6 agent serving.

The HTTP layer is intentionally thin: all cache and streaming policy lives in
``service.py`` and all compute goes through an ``AgentEngine`` implementation.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from .qwen36_engine import Qwen36FrontendAgentEngine
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
            if req.stream:
                return StreamingResponse(
                    service.stream_openai(req, model=service.engine.model_name),
                    media_type="text/event-stream",
                    headers=SSE_HEADERS,
                )
            result = service.complete(req)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(501, str(exc)) from exc

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


def create_app_from_checkpoint(*, checkpoint: str,
                               model_name: str = "qwen36-27b",
                               device: str = "cuda",
                               max_seq: int = 262208):
    engine = Qwen36FrontendAgentEngine.from_checkpoint(
        checkpoint,
        device=device,
        max_seq=max_seq,
        model_name=model_name,
    )
    return build_app(AgentService(engine))


def main(argv: list[str] | None = None) -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="FlashRT Qwen3.6 agent-serving OpenAI API")
    parser.add_argument("--checkpoint", required=True,
                        help="Qwen3.6 NVFP4 checkpoint directory")
    parser.add_argument("--model-name", default="qwen36-27b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq", type=int, default=262208)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    app = create_app_from_checkpoint(
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
        max_seq=args.max_seq,
    )
    uvicorn.run(app, host=args.host, port=args.port,
                log_level=args.log_level)


if __name__ == "__main__":
    main()
