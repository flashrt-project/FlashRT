#!/usr/bin/env python3
"""FlashRT — DM05-LIBERO serving agent.

FastAPI server exposing POST /predictions with a JSON action chunk.

    pip install fastapi uvicorn

    export DM05_CHECKPOINT=/path/to/DM05-libero
    export DM05_CALIBRATION=/path/to/calibration_state.pt
    python -m serving.dm05_agent.server --port 8000

    curl -s localhost:8000/predictions -H 'content-type: application/json' -d '{
      "observation": {"images": {"1": "<base64>", "2": "<base64>"},
                      "state": [...], "prompt": "pick up the cup"}}'

The observation shape matches the LIBERO harness, so a client can post what it
already has. Actions come back as a plain JSON array -- (chunk, action_dim) floats --
because that is what a robot client can use directly.

The server calls flash_rt.load_model(config="dm05") and model.predict().
The frontend owns model loading, preprocessing and graph capture. Historical
measurements are in docs/PERFORMANCE_DM05.md; this integration still needs its
full-action quality gate and a fresh end-to-end measurement.
"""
from __future__ import annotations

import argparse, base64, io, logging, os, sys, time, uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from pydantic import BaseModel

log = logging.getLogger("dm05_agent")

_runtime = None


class Observation(BaseModel):
    images: dict[str, str]                # view name -> base64-encoded image
    state: list[float]
    prompt: str = ""
    robot_type: Optional[str] = None


class PredictRequest(BaseModel):
    observation: Observation


class PredictResponse(BaseModel):
    request_id: str
    actions: list[list[float]]
    latency_ms: float


def _load(checkpoint, calibration, diffusion_steps=10, views=("Head", "Left wrist")):
    global _runtime
    if _runtime is not None:
        return _runtime
    from flash_rt import load_model
    model = load_model(checkpoint, config="dm05", framework="torch", hardware="rtx_sm120",
                       use_fp4=True, num_views=len(views), num_steps=diffusion_steps)
    model.pipeline.load_calibration_state(calibration)
    _runtime = model, views
    return _runtime


def _do_predict(inp: dict) -> dict:
    import torch
    from PIL import Image

    model, views = _runtime
    o = inp["observation"]

    # Ordered by the configured view list where the client supplied those names,
    # otherwise by the key order it sent -- the pipeline binds views positionally.
    names = [v for v in views if v in o["images"]] or sorted(o["images"], key=int)
    images = [Image.open(io.BytesIO(base64.b64decode(o["images"][k]))).convert("RGB")
              for k in names]

    data = {"images": images, "state": o["state"], "robot_type": o.get("robot_type")}
    t0 = time.perf_counter()
    with torch.no_grad():
        action = np.asarray(model.predict(data, prompt=o.get("prompt", "")))
    latency = (time.perf_counter() - t0) * 1000.0

    return {"request_id": uuid.uuid4().hex,
            "actions": np.asarray(action, dtype=np.float32).reshape(
                -1, action.shape[-1]).tolist(),
            "latency_ms": latency}


def _create_app():
    from fastapi import FastAPI
    from fastapi.params import Body as FastAPIBody

    app = FastAPI(title="DM05-LIBERO FlashRT", version="1.0.0")

    @app.get("/health")
    async def health():
        return {"status": "ok" if _runtime is not None else "loading"}

    @app.post("/predictions", response_model=PredictResponse)
    async def predictions(body: PredictRequest = FastAPIBody()):
        return _do_predict(body.model_dump())

    return app


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="DM05-LIBERO FlashRT serving agent")
    p.add_argument("--checkpoint", default=os.environ.get("DM05_CHECKPOINT"),
                   help="DM05-LIBERO checkpoint directory (or $DM05_CHECKPOINT)")
    p.add_argument("--calibration", default=os.environ.get("DM05_CALIBRATION"),
                   help="calibration state path (or $DM05_CALIBRATION)")
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if not args.checkpoint or not args.calibration:
        raise SystemExit("--checkpoint and --calibration (or their env vars) are required")

    import uvicorn

    # load before serving so /predictions is never answered by a cold runtime
    _load(args.checkpoint, args.calibration, args.diffusion_steps)
    uvicorn.run(_create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
