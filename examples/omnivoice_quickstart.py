#!/usr/bin/env python3
"""FlashRT — OmniVoice TTS quickstart.

Usage:
    export OMNIVOICE_CHECKPOINT=/path/to/OmniVoice
    python examples/omnivoice_quickstart.py \
        --text "Hello, this is a test." \
        --reference-audio /path/to/ref.wav \
        --output output.wav

    # Voice design (no reference audio needed):
    python examples/omnivoice_quickstart.py \
        --text "Welcome to FlashRT." \
        --mode design \
        --instruct "A cheerful female voice with clear pronunciation"
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

_model = None
_vcp_cache = {}


def _load(ckpt, cfg_ratio=0.05, bookend=False):
    global _model
    import torch
    from omnivoice import OmniVoice
    from flash_rt.models.omnivoice import inject, free_encoder

    _model = OmniVoice.from_pretrained(str(ckpt), dtype=torch.bfloat16).to("cuda:0")
    _model.eval()
    inject(_model, cfg_ratio=cfg_ratio, bookend=bookend)
    print(f"[FlashRT] Engine loaded: cfg_ratio={cfg_ratio} bookend={bookend}", flush=True)
    return _model


def _get_vcp(ref_audio, ref_text, denoise=True, preprocess=True):
    global _vcp_cache
    key = f"{ref_audio}:{ref_text}:{denoise}:{preprocess}"
    if key not in _vcp_cache:
        _vcp_cache[key] = _model.create_voice_clone_prompt(
            ref_audio=ref_audio, ref_text=ref_text or None)
    return _vcp_cache[key]


def _generate(text, vcp=None, **kwargs):
    import torch
    from omnivoice import OmniVoiceGenerationConfig

    seed = kwargs.pop("seed", 42)
    if seed is not None and seed >= 0:
        torch.manual_seed(seed)

    g = OmniVoiceGenerationConfig(**kwargs)
    t0 = time.perf_counter()
    audio = np.array(_model.generate(
        text=text, generation_config=g, voice_clone_prompt=vcp)[0]).squeeze()
    elapsed = time.perf_counter() - t0
    dur = len(audio) / 24000
    print(f"[FlashRT] {dur:.2f}s audio in {elapsed*1000:.0f}ms "
          f"(RTF={elapsed/dur:.4f}, speedup={dur/elapsed:.1f}x)", flush=True)
    return audio


def _build_parser():
    p = argparse.ArgumentParser(description="OmniVoice FlashRT TTS")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--language", default="auto")
    p.add_argument("--mode", default="clone", choices=["clone", "design", "auto"])
    p.add_argument("--reference-audio", default="", help="Reference audio path for clone mode")
    p.add_argument("--reference-text", default="", help="Reference transcript (optional)")
    p.add_argument("--instruct", default="", help="Voice description for design mode")
    p.add_argument("--duration", type=float, default=None, help="Fixed duration in seconds")
    p.add_argument("--num-steps", type=int, default=32, help="MaskGIT steps (1-32)")
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42, help="Random seed, -1 for random")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--cfg-ratio", type=float, default=0.05,
                   help="Fraction of steps using BF16 CFG (0.0-1.0)")
    p.add_argument("--bookend", action="store_true",
                   help="Use BF16 CFG on the final step")
    p.add_argument("--output", default="output.wav", help="Output WAV path")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    ckpt = os.environ.get("OMNIVOICE_CHECKPOINT", "").strip()
    if not ckpt:
        print("ERROR: OMNIVOICE_CHECKPOINT environment variable not set.",
              file=sys.stderr)
        print("  export OMNIVOICE_CHECKPOINT=/path/to/OmniVoice", file=sys.stderr)
        return 1

    args = _build_parser().parse_args(argv)

    _load(ckpt, cfg_ratio=args.cfg_ratio, bookend=args.bookend)

    vcp = None
    if args.mode == "clone":
        if not args.reference_audio:
            print("ERROR: --reference-audio required for clone mode", file=sys.stderr)
            return 1
        vcp = _get_vcp(args.reference_audio, args.reference_text,
                       denoise=True, preprocess=True)
        from flash_rt.models.omnivoice import free_encoder
        free_encoder(_model)

    gen_kwargs = dict(
        num_step=args.num_steps,
        guidance_scale=args.guidance_scale,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
        position_temperature=5.0,
    )
    if args.duration:
        gen_kwargs["duration"] = args.duration

    audio = _generate(args.text, vcp=vcp, seed=args.seed, **gen_kwargs)

    if args.speed != 1.0:
        import librosa
        audio = librosa.effects.time_stretch(audio, rate=args.speed)

    sf.write(args.output, audio.astype(np.float32), 24000)
    print(f"[FlashRT] Saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
