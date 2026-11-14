#!/usr/bin/env python3
r"""Run Higgs Audio v3 TTS on W7900D, Radeon 8060S / Halo or R9700.

R9700 defaults to native FP8 W8A8; W7900D and Halo default to BF16.
--bf16 selects BF16 explicitly. Hardware is detected automatically, or checked against:
  W7900D (gfx1100): --hardware amd_rdna3
  Halo   (gfx1151): --hardware amd_rdna35
  R9700  (gfx1201): --hardware amd_rdna4

Examples:
    export HIGGS_CHECKPOINT=/path/to/higgs-tts-3-4b
    python examples/amd/higgs_audio_v3_quickstart.py --out auto.wav
    python examples/amd/higgs_audio_v3_quickstart.py --bf16 --out bf16.wav
    python examples/amd/higgs_audio_v3_quickstart.py \
        --temperature 1.0 \
        --seed 1234 --text "Hello from FlashRT."
"""

import argparse
import os
from pathlib import Path
import sys
import wave

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=os.environ.get("HIGGS_CHECKPOINT"),
                        help="checkpoint directory (or set HIGGS_CHECKPOINT)")
    parser.add_argument("--hardware", default="auto",
                        choices=("auto", "amd_rdna3", "amd_rdna35", "amd_rdna4"))
    parser.add_argument("--device", default="cuda:0", help="ROCm device, using PyTorch's cuda:N naming")
    parser.add_argument("--bf16", action="store_true", help="select BF16 instead of hardware-adaptive precision")
    parser.add_argument("--text", default="Hello from FlashRT.")
    parser.add_argument("--out", default="higgs_rdna.wav", help="output mono 24 kHz PCM WAV")
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 keeps greedy; positive values enable native temperature sampling")
    parser.add_argument("--seed", type=int, default=None, help="optional uint64 sampling seed")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("set --checkpoint or HIGGS_CHECKPOINT")

    import numpy as np
    import torch

    if not torch.version.hip:
        parser.error("this AMD RDNA example requires PyTorch with ROCm")
    from flash_rt.models.higgs_audio_v3 import create_frontend

    frontend = create_frontend(
        args.checkpoint, hardware=args.hardware, device=args.device,
        max_seq=args.max_seq, fp8=False if args.bf16 else None)
    audio = frontend.generate(args.text, temperature=args.temperature, seed=args.seed)
    samples = audio.numpy()
    if not np.isfinite(samples).all():
        raise RuntimeError("cannot write nonfinite audio")
    # Codec floats can exceed PCM full scale. One gain for the complete file
    # preserves its waveform and relative dynamics instead of clipping peaks.
    peak = float(np.max(np.abs(samples), initial=0.0))
    gain = 1.0 / max(1.0, peak)
    pcm = (samples * gain * 32767.0).astype("<i2")
    with wave.open(args.out, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm.tobytes())
    spec = frontend.precision_spec
    print(f"[{frontend.hardware}] prefill={spec['prefill']}, decode={spec['decode']}")
    print(f"{args.out}: {len(audio) / 24_000:.2f}s audio, finish={frontend.last_finish_reason}")
    if gain < 1.0:
        print(f"Applied WAV gain {gain:.6f} to preserve the codec's {peak:.6f} peak without clipping.")


if __name__ == "__main__":
    main()
