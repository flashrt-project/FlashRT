"""FastAPI shell for Qwen3.6 agent serving.

The HTTP layer is intentionally thin: all cache and streaming policy lives in
``service.py`` and all compute goes through an ``AgentEngine`` implementation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

from .qwen36_engine import Qwen36FrontendAgentEngine
from .service import AgentService, result_to_openai

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
            "capsules": service.capsules.snapshot(),
            "sessions": service.sessions.snapshot(),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(raw: Dict[str, Any]):
        try:
            req = service.request_from_openai(raw)
            if req.stream:
                service.validate_request_bounds(req)
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
    """Default per-cache CUDA-graph LRU bound for opt-in graph replay.

    The agent host defaults long decode to direct kernels so arbitrary growing
    sessions do not pay exact-position graph capture in the request path. This
    cap still matters when a caller explicitly opts back into CUDA Graph replay
    for fixed-shape demos or benchmarks.
    """
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
                               warm_long_prefill_graphs: bool = False,
                               capsule_budget_bytes: int = 0,
                               default_max_tokens: int = 2048,
                               max_output_tokens: int = 8192,
                               default_session_id: str | None = None):
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
    log.info(
        "agent decode graph mode: verify_graph=%s mtp_chain_graph=%s",
        os.environ.get("FLASHRT_QWEN36_TQ_VERIFY_GRAPH", "<unset>"),
        os.environ.get("FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH", "<unset>"),
    )
    if capsule_budget_bytes > 0:
        fe = getattr(engine, "fe", None)
        if not bool(getattr(fe, "_long_ctx_mode", False)):
            raise ValueError(
                "--capsule-budget-mb requires the long FP8-KV route; use a "
                "long-context --max-seq, --route-min-seq 0, and "
                "FLASHRT_QWEN36_LONG_KV_CACHE=fp8")
    if warmup_shapes:
        log.info("startup warmup: %d shape(s), K=%d", len(warmup_shapes),
                 warmup_k)
        for idx, (prompt_len, max_tokens) in enumerate(warmup_shapes, start=1):
            route_hint = (
                "graph-only"
                if int(prompt_len) > int(warmup_committed_max_prompt)
                else "committed-stream"
            )
            log.info(
                "startup warmup queued %d/%d: prompt_len=%d max_tokens=%d "
                "mode=%s",
                idx, len(warmup_shapes), int(prompt_len), int(max_tokens),
                route_hint,
            )

        def _log_warmup_result(index: int, total: int, item: Dict[str, Any]):
            log.info("startup warmup done %d/%d: %s", index, total, item)

        warmed = engine.warmup_committed_stream(
            warmup_shapes,
            K=warmup_k,
            committed_max_prompt=warmup_committed_max_prompt,
            long_decode_graphs=True,
            long_prefill_graphs=warm_long_prefill_graphs,
            on_result=_log_warmup_result,
        )
        total_ms = sum(float(item.get("wall_ms", 0.0)) for item in warmed)
        log.info("startup warmup complete: %d shape(s), total_wall_ms=%.1f",
                 len(warmed), total_ms)
    if capsule_budget_bytes > 0:
        log.info("capsule pinning enabled, budget %.0f MB",
                 capsule_budget_bytes / (1 << 20))
    return build_app(AgentService(
        engine,
        capsule_budget_bytes=capsule_budget_bytes,
        default_max_tokens=default_max_tokens,
        max_output_tokens=max_output_tokens,
        default_session_id=default_session_id,
    ))


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
    preset = (preset or "none").strip().lower()
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
        help="Per-cache CUDA graph LRU bound for opt-in Qwen3.6 frontend "
             "graphs. The agent host defaults long decode to direct kernels; "
             "this only matters when exact graph replay is explicitly enabled.")
    parser.add_argument(
        "--warmup-preset", default="none",
        help="Startup warmup preset: none, agent, short, long, or all. The "
             "production agent default is none because long decode uses direct "
             "kernels by default; use agent/all for fixed-shape graph demos.")
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
    parser.add_argument(
        "--capsule-budget-mb", type=int, default=0,
        help="GPU byte budget (MB) for pinned shared-prefix capsules. 0 (default) "
             "disables pinning. When >0, a request with flashrt_pin_prefix pins "
             "its chunk-aligned shared prefix so later turns/sessions restore a "
             "clean committed boundary instead of cold-prefilling it (survives "
             "EOS, unlike contiguous append). Needs VRAM headroom beyond the "
             "model + KV; capsules are LRU-evicted to fit and an over-budget pin "
             "is rejected, not OOM.")
    parser.add_argument(
        "--default-max-tokens", type=int, default=2048,
        help="Generated-token budget used when an OpenAI request omits both "
             "max_tokens and max_completion_tokens.")
    parser.add_argument(
        "--max-output-tokens", type=int, default=8192,
        help="Hard server-side generated-token cap. Requests above this cap "
             "return HTTP 400 instead of being silently truncated.")
    parser.add_argument(
        "--default-session-id", default=None,
        help="Legacy fallback session id for older single-client local demos. "
             "Normal OpenAI-compatible clients should rely on automatic "
             "hot-prefix reuse instead; multi-client servers should leave this "
             "unset.")
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
        capsule_budget_bytes=int(args.capsule_budget_mb) * (1 << 20),
        default_max_tokens=args.default_max_tokens,
        max_output_tokens=args.max_output_tokens,
        default_session_id=args.default_session_id,
    )
    uvicorn.run(app, host=args.host, port=args.port,
                log_level=args.log_level, access_log=args.access_log)


if __name__ == "__main__":
    main()
