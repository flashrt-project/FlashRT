# serving/higgs_audio_agent

Streaming text-to-speech server for Higgs Audio v3 TTS-4B on FlashRT.

This directory is the **policy layer** above the FlashRT execution contract. It
owns the OpenAI-compatible HTTP surface, audio container framing (PCM / WAV),
and request serialisation. It must not add session / KV / graph verbs to
`exec/` — the contract stays Buffer / Graph / Plan / Event / ShapeKey, and all
GPU state (the FP8 backbone, the position-agnostic decode graph, the KV cache,
the codec) is owned by the frontend. The model is driven only through the
frontend's committed `generate_stream`.

## What runs where

```
serving/higgs_audio_agent/server.py        policy: HTTP, audio framing, serialisation
  └─ frontend.generate_stream(text)        frontend: prefill -> committed decode_stream
       ├─ decode_stream()                    one position-agnostic decode graph / frame
       └─ codec (ctx + holdback windows)     streamed chunks == one-shot audio (cos 1.0)
exec/                                       UNTOUCHED (no serving verb added)
```

Single stream: TTS requests serialise behind one lock (one decode graph + KV
buffers). Concurrency/batching would be added here as policy, never in the
contract.

## Quickstart

```bash
export HIGGS_CHECKPOINT=/path/to/higgs-audio-v3-tts-4b
pip install fastapi uvicorn
python -m serving.higgs_audio_agent.server \
    --checkpoint "$HIGGS_CHECKPOINT" --host 127.0.0.1 --port 8000
# startup loads the model, warms the decode graph + codec, then serves.
```

Synthesize (OpenAI `audio/speech`-compatible):

```bash
# WAV
curl -s http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"higgs-audio-v3-tts-4b","input":"Hello from FlashRT.","response_format":"wav"}' \
  -o hello.wav

# raw 24 kHz mono PCM16, streamed as it is generated
curl -N http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Streaming low-latency speech.","response_format":"pcm"}' > out.pcm
```

`GET /health` and `GET /v1/models` report status and the served model.

## Measured (RTX 5090, single stream, FP8 backbone)

- Full pipeline RTF ≈ 0.12 (≈ 8× real time) across short/medium/long.
- Time-to-first-audio ≈ 0.14 s (first committed chunk), vs an upstream
  server's 0.36–0.63 s first-audio.
- Peak VRAM ≈ 6.3 GB.

See `docs/higgs_audio_v3.md` for the frontend, the decode kernels, and
faithfulness validation.

## Notes

- `response_format`: `pcm` (raw 24 kHz mono int16, lowest latency) or `wav`.
- The frontend's greedy decode is deterministic but, like any reimplementation
  of greedy discrete-code TTS, produces a different valid realisation than other
  engines (see the doc); faithfulness is established by teacher-forced cosine 1.0
  and codec cosine 0.99993, not by matching another engine's samples.
- `pip install fastapi uvicorn` is a server-only dependency; the frontend and
  kernels do not require it.
