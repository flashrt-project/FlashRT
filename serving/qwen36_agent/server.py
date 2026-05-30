"""FastAPI shell for Qwen3.6 agent serving.

The HTTP layer is intentionally thin: all cache and streaming policy lives in
``service.py`` and all compute goes through an ``AgentEngine`` implementation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from .qwen36_engine import Qwen36FrontendAgentEngine
from .service import AgentService, request_from_openai, result_to_openai

log = logging.getLogger("qwen36_agent")

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
        # spec_enabled / fe are Qwen36-specific, not part of the AgentEngine
        # protocol; guard them so a minimal/fake engine (tests, dev) reports
        # health instead of 500.
        engine = service.engine
        fe = getattr(engine, "fe", None)
        return {
            "status": "ok",
            "model": engine.model_name,
            "max_seq": engine.max_seq,
            "speculative": bool(getattr(engine, "spec_enabled", False)),
            "decode_fastgemm": bool(getattr(fe, "_decode_fastgemm", False)),
            "verify_warpsplit": bool(getattr(fe, "_verify_warpsplit", False)),
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


def _auto_graph_cache_max(max_seq: int) -> int:
    """Default per-cache CUDA-graph LRU bound, scaled to the VRAM headroom that
    ``max_seq`` leaves. Decode graphs are keyed by exact (cur_pos, K, ...), so a
    request traverses ~one-per-position graphs; a cap below that working set
    evicts warmed graphs and forces re-capture on the next request (a repeated
    cold start). A bigger cap lets warmed graphs survive across requests/lengths
    — but each graph holds pooled buffers, so it competes with the KV cache.
    Small max_seq leaves plenty of VRAM (use a large cap, kill the eviction
    thrash); a 256K-capable cache leaves almost none (stay conservative)."""
    max_seq = int(max_seq)
    if max_seq <= 32768:
        return 1024
    if max_seq <= 131072:
        return 256
    return 128


def create_app_from_checkpoint(*, checkpoint: str,
                               model_name: str = "qwen36-27b",
                               device: str = "cuda",
                               max_seq: int = 262208,
                               route_min_seq: int | None = 0,
                               graph_cache_max: int | None = None,
                               warmup_shapes=None,
                               warmup_k: int = 6,
                               warmup_committed_max_prompt: int = 1024,
                               warm_long_prefill_graphs: bool = False):
    if graph_cache_max is None:
        graph_cache_max = _auto_graph_cache_max(max_seq)
    engine = Qwen36FrontendAgentEngine.from_checkpoint(
        checkpoint,
        device=device,
        max_seq=max_seq,
        model_name=model_name,
        route_min_seq=route_min_seq,
        graph_cache_max=graph_cache_max,
    )
    if not engine.spec_enabled:
        log.warning(
            "MTP head not loaded — speculative decode is DISABLED; decode "
            "runs the slower non-spec path. Set FLASHRT_QWEN36_MTP_CKPT_DIR to "
            "a paired MTP checkpoint to enable it (/health reports "
            "\"speculative\": false).")
    else:
        log.info("MTP head loaded; speculative decode enabled (default K=%d)",
                 warmup_k)
    if warmup_shapes:
        log.info("startup warmup: %d shape(s), K=%d", len(warmup_shapes),
                 warmup_k)
        warmed = engine.warmup_committed_stream(
            warmup_shapes,
            K=warmup_k,
            committed_max_prompt=warmup_committed_max_prompt,
            long_decode_graphs=True,
            long_prefill_graphs=warm_long_prefill_graphs,
        )
        for item in warmed:
            log.info("startup warmup result: %s", item)
    return build_app(AgentService(engine))


def _parse_warmup_shapes(spec_csv: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    if not spec_csv.strip():
        return shapes
    for spec in spec_csv.split(","):
        spec = spec.strip()
        if not spec:
            continue
        try:
            prompt_len, max_tokens = spec.split(":")
            shapes.append((int(prompt_len), int(max_tokens)))
        except ValueError as exc:
            raise ValueError(
                f"invalid warmup shape {spec!r}; expected prompt:max_tokens"
            ) from exc
    return shapes


def _dedupe_shapes(shapes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen = set()
    for shape in shapes:
        if shape not in seen:
            out.append(shape)
            seen.add(shape)
    return out


def _warmup_preset_shapes(preset: str, max_seq: int) -> list[tuple[int, int]]:
    preset = (preset or "agent").strip().lower()
    if preset in ("none", "off", "false", "0"):
        return []
    if preset not in ("agent", "short", "long", "all"):
        raise ValueError(
            f"invalid warmup preset {preset!r}; expected agent, short, "
            "long, all, or none")

    short = [(16, 128), (32, 128), (64, 128), (128, 128), (512, 128)]
    long = [
        (2048, 128),
        (8192, 128),
        (32768, 64),
        (131072, 64),
        (204800, 64),
        (262144, 16),
    ]
    if preset == "short":
        candidates = short
    elif preset == "long":
        candidates = long
    elif preset == "all":
        candidates = short + [
            (1024, 128), (4096, 128), (16384, 128), (65536, 64)
        ] + long
    else:
        candidates = short + long
    return [(p, n) for p, n in candidates if p + n <= int(max_seq)]


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
    parser.add_argument(
        "--route-min-seq", type=int, default=0,
        help=(
            "Minimum prompt length routed to the chunked long-context path. "
            "The agent host defaults to 0 so short real prompts avoid "
            "per-position short-route graph capture."))
    parser.add_argument(
        "--graph-cache-max", type=int, default=None,
        help="Per-cache CUDA graph LRU bound for Qwen3.6 frontend graphs. "
             "Default auto-scales with --max-seq (1024 at <=32K, 256 at "
             "<=128K, 128 at 256K) so small-context deployments keep warmed "
             "graphs across requests instead of evicting and re-capturing.")
    parser.add_argument(
        "--warmup-preset", default="agent",
        help="Startup warmup preset: agent, short, long, all, or none.")
    parser.add_argument(
        "--warmup", default="",
        help='Additional comma-separated "prompt_len:max_tokens" shapes.')
    parser.add_argument(
        "--warmup-K", type=int, default=6,
        help="Speculative decode K used for startup warmup.")
    parser.add_argument(
        "--warmup-committed-max-prompt", type=int, default=1024,
        help=(
            "Run real committed-stream warmup up to this prompt length; "
            "larger long-context shapes use graph-only warmup."))
    parser.add_argument(
        "--warm-long-prefill-graphs", action="store_true",
        help="Also capture long-context prefill chunk graphs at startup.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--access-log", dest="access_log", action="store_true", default=False,
        help="Enable uvicorn per-request access logging. Off by default: the "
             "per-request access line adds wall-time jitter and the serving "
             "layer already logs one structured metric line per completion.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    warmup_shapes = _dedupe_shapes(
        _warmup_preset_shapes(args.warmup_preset, args.max_seq)
        + _parse_warmup_shapes(args.warmup)
    )
    app = create_app_from_checkpoint(
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
        max_seq=args.max_seq,
        route_min_seq=args.route_min_seq,
        graph_cache_max=args.graph_cache_max,
        warmup_shapes=warmup_shapes,
        warmup_k=args.warmup_K,
        warmup_committed_max_prompt=args.warmup_committed_max_prompt,
        warm_long_prefill_graphs=args.warm_long_prefill_graphs,
    )
    uvicorn.run(app, host=args.host, port=args.port,
                log_level=args.log_level, access_log=args.access_log)


if __name__ == "__main__":
    main()
