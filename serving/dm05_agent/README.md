# DM05-LIBERO FlashRT serving agent

FastAPI wrapper around the DM05-LIBERO frontend. Loads the model with
`flash_rt.load_model(config="dm05", use_fp4=True)` and serves `model.predict()`
once per request. No external runtime methods are replaced.

## Install

```sh
pip install fastapi uvicorn
```

## Run

```sh
export DM05_CHECKPOINT=/path/to/DM05-libero
export DM05_CALIBRATION=/path/to/calibration_state.pt
python -m serving.dm05_agent.server --port 8000
```

The runtime is loaded before the socket opens, so `/health` reporting `ok` means the
first request will not pay for model load. Static FP8 and NVFP4 scales are loaded from
the calibration state, never recomputed per request.

## API

`GET /health` → `{"status": "ok" | "loading"}`

`POST /predictions`

```json
{
  "observation": {
    "images": {"Head": "<base64 image>", "Left wrist": "<base64 image>"},
    "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "prompt": "pick up the black bowl",
    "robot_type": "franka"
  }
}
```

→

```json
{"request_id": "...", "actions": [[...7 floats...], ...], "latency_ms": 71.0}
```

`actions` is the full action chunk, one row per step. Views are bound positionally in
the order they were configured; a client that sends the configured view names has them
ordered for it, otherwise key order is used.

## Notes

- `latency_ms` is measured around the predict call and **includes** host-side
  preprocessing, because that is what the caller waits for. The historical numbers in
  `docs/PERFORMANCE_DM05.md` precede this frontend integration and must be rechecked.
- Requests are handled one at a time: the pipeline is a captured CUDA graph over a
  fixed prompt layout, so concurrent calls would contend for the same buffers. Put a
  worker pool in front for more than one in flight, one process per GPU.
- Seeding follows `DM05_INFERENCE_SEED` as elsewhere in the pipeline.
