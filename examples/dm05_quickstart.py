#!/usr/bin/env python3
"""Run DM05 through FlashRT's public VLA API."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("DM05_CHECKPOINT"), required=False)
    parser.add_argument("--calibration", default=os.environ.get("DM05_CALIBRATION"))
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.checkpoint or not args.calibration:
        parser.error("provide --checkpoint and --calibration or their DM05 environment variables")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    from PIL import Image
    from flash_rt import load_model
    record = json.loads(next(line for line in args.observation.read_text().splitlines() if line.strip()))
    obs = record["observation"]
    images = [Image.open(io.BytesIO(base64.b64decode(obs["images"][key]))).convert("RGB")
              for key in sorted(obs["images"], key=int)]
    model = load_model(args.checkpoint, config="dm05", framework="torch", hardware="rtx_sm120",
                       use_fp4=True, num_views=len(images), num_steps=args.diffusion_steps)
    model.pipeline.load_calibration_state(args.calibration)
    observation = {"images": images, "state": obs["state"], "robot_type": obs.get("robot_type")}
    # First call includes capture; report steady-state wall time separately.
    model.predict(observation, prompt=obs["prompt"])
    times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        actions = model.predict(observation)
        times.append((time.perf_counter() - start) * 1000)
    print(f"actions: {actions.shape}; wall p50: {np.median(times):.2f} ms; p95: {np.percentile(times,95):.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
