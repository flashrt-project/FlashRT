"""serving/higgs_audio_agent — streaming TTS server (policy layer).

OpenAI-compatible ``POST /v1/audio/speech`` over the FlashRT Higgs Audio v3
frontend. This is the **policy layer** above the execution contract: it owns the
HTTP surface, audio container framing, and request serialisation. It must not
add session/KV/graph verbs — those live in the frontend (which owns all GPU
state) and the exec contract (Buffer/Graph/Plan). The model is driven only
through the frontend's committed ``generate_stream``.

Single stream: TTS requests are serialised behind one lock (the frontend holds
one decode graph + KV buffers). Concurrency/batching would be added here as
policy, never in the contract.

Run:
    export HIGGS_CHECKPOINT=/path/to/higgs-audio-v3-tts-4b
    pip install fastapi uvicorn
    python -m serving.higgs_audio_agent.server --checkpoint "$HIGGS_CHECKPOINT" \
        --host 127.0.0.1 --port 8000
"""
import argparse
import os
import struct
import threading

SAMPLE_RATE = 24_000


def _pcm16(wav) -> bytes:
    import numpy as np
    x = np.clip(wav.numpy() if hasattr(wav, "numpy") else wav, -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def _wav_header(data_bytes: int) -> bytes:
    # Minimal 44-byte PCM WAV header (mono, 16-bit, 24 kHz).
    byte_rate = SAMPLE_RATE * 2
    return (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, byte_rate, 2, 16)
            + b"data" + struct.pack("<I", data_bytes))


def build_app(frontend, model_name: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="FlashRT Higgs Audio v3 TTS")
    lock = threading.Lock()

    class SpeechRequest(BaseModel):
        model: str | None = None
        input: str
        voice: str | None = None
        instructions: str | None = None       # shared voice/style preamble
        response_format: str = "pcm"          # pcm | wav
        stream: bool = True

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name,
                "sample_rate": SAMPLE_RATE, "max_new_frames": frontend.max_new_frames}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        if not req.input.strip():
            raise HTTPException(status_code=400, detail="empty input")
        fmt = req.response_format.lower()
        if fmt not in ("pcm", "wav"):
            raise HTTPException(status_code=400, detail="response_format must be pcm|wav")

        def gen():
            with lock:                         # one decode stream at a time
                if fmt == "wav":
                    # Unknown total length up front; stream with a max-size
                    # header (clients that read incrementally accept this).
                    yield _wav_header(0xFFFFFFFF - 44)
                # ``instructions`` is a shared voice/style preamble: when many
                # requests carry the same one, the frontend reuses its prefix KV
                # (only the new input is prefilled). Policy lives here; the
                # frontend owns the reuse mechanism.
                for chunk in frontend.generate_stream(
                        req.input, system=req.instructions):
                    yield _pcm16(chunk)

        media = "audio/wav" if fmt == "wav" else "audio/pcm"
        return StreamingResponse(gen(), media_type=media)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.environ.get("HIGGS_CHECKPOINT"))
    ap.add_argument("--model-name", default="higgs-audio-v3-tts-4b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true", help="BF16 backbone instead of FP8")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--warmup", default="Warm up the decode graph and codec.")
    args = ap.parse_args()
    if not args.checkpoint:
        ap.error("set --checkpoint or HIGGS_CHECKPOINT")

    import uvicorn

    from flash_rt.frontends.torch.higgs_audio_v3_rtx import (
        HiggsAudioV3TorchFrontendRtx,
    )
    fe = HiggsAudioV3TorchFrontendRtx(
        args.checkpoint, device=args.device, max_seq=args.max_seq,
        fp8=not args.bf16)
    if args.warmup:                            # capture graph + load codec once
        for _ in fe.generate_stream(args.warmup):
            pass
    app = build_app(fe, args.model_name)
    print(f"FlashRT Higgs TTS ready: http://{args.host}:{args.port}  "
          f"backbone={'BF16' if args.bf16 else 'FP8'}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
