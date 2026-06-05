#!/usr/bin/env python3
"""Higgs Audio v3 TTS-4B FlashRT quickstart — text -> 24 kHz waveform.

Single-stream zero-shot TTS on RTX SM120 (5090): an FP8 W8A8 Qwen3-4B backbone
drives a fused 8-codebook head under a delay pattern, decoded autoregressively
and synthesised by the bundled neural codec — all in one process.

Set HIGGS_CHECKPOINT to the checkpoint directory (the local snapshot of
``bosonai/higgs-audio-v3-tts-4b``: config.json + model.safetensors +
tokenizer.json), or pass it as --checkpoint.

Examples:
    export HIGGS_CHECKPOINT=/path/to/higgs-audio-v3-tts-4b
    python examples/higgs_audio_v3_quickstart.py \
        --text "The quick brown fox jumps over the lazy dog." --out hello.wav

    # BF16 backbone instead of FP8:
    python examples/higgs_audio_v3_quickstart.py --text "..." --bf16

    # Per-frame decode latency (excludes one-time calibration / codec load):
    python examples/higgs_audio_v3_quickstart.py --text "..." --benchmark 3
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
import wave

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _save_wav(path: str, wav, sample_rate: int = 24_000) -> None:
    import numpy as np

    x = (np.clip(wav.numpy() if hasattr(wav, "numpy") else wav, -1.0, 1.0)
         * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(x.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.environ.get("HIGGS_CHECKPOINT"),
                    help="checkpoint dir (or set HIGGS_CHECKPOINT)")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.",
                    help="text to synthesise")
    ap.add_argument("--out", default="higgs_quickstart.wav", help="output wav path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true", help="use the BF16 backbone")
    ap.add_argument("--benchmark", type=int, default=0,
                    help="re-generate N times and report per-frame decode latency")
    args = ap.parse_args()

    if not args.checkpoint:
        ap.error("set --checkpoint or the HIGGS_CHECKPOINT environment variable")

    from flash_rt.frontends.torch.higgs_audio_v3_rtx import (
        HiggsAudioV3TorchFrontendRtx,
    )

    fe = HiggsAudioV3TorchFrontendRtx(
        args.checkpoint, device=args.device, max_seq=args.max_seq,
        fp8=not args.bf16)
    backbone = "BF16" if args.bf16 else "FP8 W8A8"

    t0 = time.perf_counter()
    wav = fe.generate(args.text)                  # 1st call: lazy calibrate + codec
    dt = time.perf_counter() - t0
    _save_wav(args.out, wav)
    dur = len(wav) / 24_000
    frames = fe.latency_records[-1] if fe.latency_records else 0.0
    print(f"[{backbone}] '{args.text}'")
    print(f"  -> {args.out}  ({dur:.2f}s audio, {dt:.2f}s wall incl 1st-call setup)")

    for i in range(args.benchmark):
        fe.predict(args.text)                     # warm: no calibration/codec load
        ms = fe.latency_records[-1]
        n = max(1, int(round(dur * 25)))          # 25 Hz acoustic frames
        print(f"  bench {i + 1}: AR decode {ms:.0f} ms ({ms / n:.2f} ms/frame)")
    print("done.")


if __name__ == "__main__":
    main()
