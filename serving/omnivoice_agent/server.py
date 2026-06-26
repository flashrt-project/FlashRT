"""serving/omnivoice_agent — TTS inference server for OmniVoice.

FastAPI server exposing POST /predictions with base64 WAV output.
Uses flash_rt.models.omnivoice for BF16+FP4 hybrid acceleration.

Run:
    export OMNIVOICE_CHECKPOINT=/path/to/OmniVoice
    pip install fastapi uvicorn soundfile
    python -m serving.omnivoice_agent.server --checkpoint "$OMNIVOICE_CHECKPOINT" \
        --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse, base64, io, os, sys, time, uuid, tempfile, logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("omnivoice_agent")

_model = None
_vcp_cache = {}


class PredictInput(BaseModel):
    text: str
    language: str = "auto"
    mode: str = "clone"
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    instruct: Optional[str] = None
    speed: float = 1.0
    duration: Optional[float] = None
    num_steps: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    preprocess_prompt: bool = True
    seed: int = 42
    position_temperature: float = 5.0
    postprocess_output: bool = True


class PredictRequest(BaseModel):
    input: PredictInput


def _load(ckpt, cfg_ratio=0.05, bookend=False):
    global _model
    import torch
    from omnivoice import OmniVoice
    from flash_rt.models.omnivoice import inject

    _model = OmniVoice.from_pretrained(str(ckpt), dtype=torch.bfloat16).to("cuda:0")
    _model.eval()
    inject(_model, cfg_ratio=cfg_ratio, bookend=bookend)
    log.info("Engine loaded: cfg_ratio=%.2f bookend=%s", cfg_ratio, bookend)


def _get_vcp(ref_audio_path, ref_text, denoise=True, preprocess=True):
    global _vcp_cache
    key = f"{ref_audio_path}:{ref_text}:{denoise}:{preprocess}"
    if key not in _vcp_cache:
        _vcp_cache[key] = _model.create_voice_clone_prompt(
            ref_audio=ref_audio_path, ref_text=ref_text or None)
    return _vcp_cache[key]


def _do_predict(inp: dict) -> dict:
    import torch

    text = inp["text"]
    mode = inp.get("mode", "clone")
    ref_audio = inp.get("reference_audio")
    ref_text = inp.get("reference_text")
    instruct = inp.get("instruct")
    speed = inp.get("speed", 1.0)
    duration = inp.get("duration")
    num_steps = inp.get("num_steps", 32)
    guidance_scale = inp.get("guidance_scale", 2.0)
    denoise = inp.get("denoise", True)
    preprocess = inp.get("preprocess_prompt", True)
    seed = inp.get("seed", 42)
    position_temperature = inp.get("position_temperature", 5.0)
    postprocess = inp.get("postprocess_output", True)

    if mode == "clone" and not ref_audio:
        return {"id": "", "status": "failed", "error": "reference_audio required for clone mode"}
    if mode == "design" and not instruct:
        return {"id": "", "status": "failed", "error": "instruct required for design mode"}

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)

    vcp = None
    if mode == "clone":
        ref_path = ref_audio
        if ref_audio.startswith("http"):
            import urllib.request
            suffix = ".wav" if ref_audio.endswith(".wav") else ".mp3"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            urllib.request.urlretrieve(ref_audio, tmp.name)
            ref_path = tmp.name
        vcp = _get_vcp(ref_path, ref_text, denoise, preprocess)

    from omnivoice import OmniVoiceGenerationConfig
    gen_kwargs = dict(num_step=num_steps, guidance_scale=guidance_scale,
                      denoise=denoise, preprocess_prompt=preprocess,
                      postprocess_output=postprocess, position_temperature=position_temperature)
    if duration:
        gen_kwargs["duration"] = duration

    t0 = time.perf_counter()
    if mode == "design":
        audio = np.array(_model.generate(text=text, instruct=instruct,
                         generation_config=OmniVoiceGenerationConfig(**gen_kwargs))[0]).squeeze()
    elif mode == "auto":
        audio = np.array(_model.generate(text=text,
                         generation_config=OmniVoiceGenerationConfig(**gen_kwargs))[0]).squeeze()
    else:
        audio = np.array(_model.generate(text=text, generation_config=OmniVoiceGenerationConfig(**gen_kwargs),
                         voice_clone_prompt=vcp)[0]).squeeze()
    predict_time = time.perf_counter() - t0

    if speed != 1.0:
        import librosa
        audio = librosa.effects.time_stretch(audio, rate=speed)

    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), 24000, format="WAV", subtype="PCM_16")
    audio_b64 = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    log_lines = []
    log_lines.append("[Input] text: " + text)
    log_lines.append("[Input] language: " + str(inp.get("language", "auto")))
    log_lines.append("[Input] mode: " + mode)
    if ref_audio:
        log_lines.append("[Input] reference_audio: " + ref_audio)
    if ref_text:
        log_lines.append("[Input] reference_text: " + ref_text)
    log_lines.append("[Input] num_steps: " + str(num_steps))
    log_lines.append("[Input] guidance_scale: " + str(guidance_scale))
    log_lines.append("[Input] seed: " + str(seed))
    log_lines.append("[Metrics] duration: %.2fs" % (len(audio) / 24000))
    log_lines.append("[Metrics] predict: %.0fms" % (predict_time * 1000))
    log_lines.append("[Metrics] RTF: %.4f" % (predict_time / (len(audio) / 24000)))

    return {
        "id": "pred_" + uuid.uuid4().hex[:16],
        "status": "succeeded",
        "output": audio_b64,
        "logs": "\n".join(log_lines),
        "metrics": {"predict_time": round(predict_time, 3)}
    }


def _create_app():
    from fastapi import FastAPI
    from fastapi.params import Body as FastAPIBody

    app = FastAPI(title="OmniVoice FlashRT", version="1.0.0")

    @app.get("/health")
    async def health():
        return {"status": "healthy" if _model is not None else "loading"}

    @app.post("/predictions")
    async def predictions(body: PredictRequest = FastAPIBody()):
        try:
            result = _do_predict(body.input.model_dump())
            return result
        except Exception as e:
            log.exception("Prediction failed")
            return {
                "id": "pred_" + uuid.uuid4().hex[:16],
                "status": "failed",
                "output": None,
                "logs": str(e),
                "metrics": {"predict_time": 0}
            }

    return app


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(description="OmniVoice FlashRT serve")
    parser.add_argument("--checkpoint", required=True, help="Path to OmniVoice checkpoint")
    parser.add_argument("--cfg-ratio", type=float, default=0.05)
    parser.add_argument("--bookend", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    _load(args.checkpoint, cfg_ratio=args.cfg_ratio, bookend=args.bookend)

    app = _create_app()
    import uvicorn
    log.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
