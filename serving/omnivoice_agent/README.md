# serving/omnivoice_agent

TTS inference server for OmniVoice on FlashRT with BF16 + FP4 hybrid
acceleration.

This directory is the policy layer above the FlashRT execution contract.
It owns the FastAPI HTTP surface, WAV framing, and request serialisation.
The model is driven through `flash_rt.models.omnivoice.inject()` which
patches OmniVoice for FlashRT acceleration.

## What runs where

```
serving/omnivoice_agent/server.py          policy: HTTP, audio framing, serialisation
  └─ flash_rt.models.omnivoice.inject()    acceleration: BF16+FP4 hybrid MaskGIT loop
       ├─ FlashRTLlmBF16                    BF16 CFG forward (step 1)
       └─ FlashRTLlm                        FP4 noCFG forward (steps 2-32)
```

Single stream: TTS requests are serialised (one model instance). The engine
uses CUDA Graph for FP4 forward and fused FlashRT kernels throughout.

## Quickstart

```bash
export OMNIVOICE_CHECKPOINT=/path/to/OmniVoice
pip install fastapi uvicorn soundfile
python -m serving.omnivoice_agent.server \
    --checkpoint "$OMNIVOICE_CHECKPOINT" --host 127.0.0.1 --port 8000
```

Clone mode (with reference audio):

```bash
curl -s http://127.0.0.1:8000/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "text": "Hello from FlashRT.",
      "mode": "clone",
      "reference_audio": "/path/to/ref.wav",
      "num_steps": 32,
      "guidance_scale": 2.0,
      "seed": 42
    }
  }' | python3 -c "import sys,json,base64; d=json.load(sys.stdin)['output']
s=d.split(',')[1]; open('out.wav','wb').write(base64.b64decode(s))"
```

Design mode (text-described voice):

```bash
curl -s http://127.0.0.1:8000/predictions \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "text": "Welcome to FlashRT.",
      "mode": "design",
      "instruct": "A calm male voice with clear pronunciation.",
      "num_steps": 32,
      "guidance_scale": 2.0
    }
  }'
```

`GET /health` reports server status.

## Measured (RTX 5060 Ti, SM120, ns=32, gs=2.0)

- Latency: 100 ms | Speedup: 5.0x | RTF: 0.032
- VRAM: 2.0 GB (70% of PyTorch)
- Mel-cosine: 0.9961 vs PyTorch BF16 reference

See `docs/PERFORMANCE_OMNIVOICE.md` for full performance specifications.

## Notes

- `FLASHRT_ENABLE_OMNIVOICE=ON` must be set at cmake build time to compile
  the `flash_rt_omnivoice` module (`omnivoice_cfg_logsoftmax_bf16` and
  `omnivoice_qk_norm_rope_bf16`). Without it, OmniVoice initialization fails
  fast with a clear `RuntimeError`.
- The output is 24 kHz mono PCM16, returned as `data:audio/wav;base64`.
- `pip install fastapi uvicorn` is a server-only dependency; the engine and
  kernels do not require it.
