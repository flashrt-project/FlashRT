#!/usr/bin/env python3
"""Wan2.2 TI2V-5B official-pipeline quickstart."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flash_rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to Wan2.2-TI2V-5B checkpoint directory")
    parser.add_argument("--prompt", default=(
        "A cinematic shot of a blue sphere rolling across a wooden table"))
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--mode", choices=("t2v", "i2v"), default="t2v")
    parser.add_argument("--image", default=None,
                        help="Input image for i2v mode")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="wan22_out.mp4")
    args = parser.parse_args()

    image = None
    if args.mode == "i2v":
        if args.image is None:
            raise SystemExit("--image is required for --mode i2v")
        from PIL import Image
        image = Image.open(args.image).convert("RGB")

    t0 = time.perf_counter()
    model = flash_rt.load_model(
        args.checkpoint,
        framework="torch",
        config="wan22_ti2v_5b",
        hardware="rtx_sm120",
    )
    print(f"[wan22] load_model wall={time.perf_counter() - t0:.2f}s")

    model.set_prompt(args.prompt, negative_prompt=args.negative_prompt)
    result = model.infer(
        mode=args.mode,
        image=image,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        shift=args.shift,
        guide_scale=args.guide_scale,
        seed=args.seed,
        save_path=args.out,
        return_metadata=True,
    )
    meta = result["metadata"]
    print(f"[wan22] infer={meta['infer_seconds']:.2f}s "
          f"peak={meta['peak_allocated_gib']:.2f} GiB")
    print(f"[wan22] saved {pathlib.Path(args.out)}")


if __name__ == "__main__":
    main()
