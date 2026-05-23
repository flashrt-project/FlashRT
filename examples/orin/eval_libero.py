#!/usr/bin/env python3
"""
FlashRT Orin — LIBERO action similarity benchmark.

Offline comparison of action chunks across cache configurations using
pre-exported LIBERO observation frames. Downloads the evaluation dataset
automatically on first run (42 MB).

Default --frames=30 completes in ~30 seconds per config for a quick
smoke test. Use --frames=300 for the full evaluation (~3 min per config,
stable latency/cosine metrics for PR validation).

Usage:
    python examples/orin/eval_libero.py \
        --checkpoint /path/to/pi05_libero_finetuned_v044

    python examples/orin/eval_libero.py \
        --checkpoint /path/to/pi05_libero_finetuned_v044 \
        --configs bf16_baseline,bf16_cache2,bf16_adaptive_cache3
"""

import argparse
import gc
import logging
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx  # noqa: E402

logging.basicConfig(level=logging.WARNING)

NPZ_URL = ("https://github.com/strayberry/mm-edge-infer-accel/releases/"
           "download/v0.1.0/libero_episodes0_1_2_100frames_each.npz")
CACHE_DIR = Path.home() / ".flash_rt" / "bench_data"
BASELINE = "bf16_baseline"

CONFIGS = {
    BASELINE:               dict(cache=1),
    "bf16_cache2":          dict(cache=2),
    "bf16_cache3":          dict(cache=3),
    "bf16_adaptive_cache3": dict(cache=3, refresh="pixel_delta"),
}


def parse_args():
    p = argparse.ArgumentParser(description="FlashRT Orin LIBERO action similarity benchmark")
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--npz", default="",
                   help="Path to NPZ (auto-downloads if empty)")
    p.add_argument("--frames", type=int, default=30,
                   help="Frames to evaluate (0 = all, 300 for full eval)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threshold", type=float, default=6.5,
                   help="Pixel-MAE refresh threshold for adaptive cache")
    p.add_argument("--configs",
                   default="bf16_baseline,bf16_cache2,bf16_cache3,bf16_adaptive_cache3",
                   help="Comma-separated config names")
    return p.parse_args()


def resize_images(images, size=224):
    if images.shape[1:3] == (size, size):
        return images
    resized = []
    for img in images:
        resized.append(np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR)))
    return np.stack(resized).astype(np.uint8)


def load_npz(npz_path, max_frames):
    data = np.load(npz_path, allow_pickle=True)
    n = max_frames if max_frames > 0 else len(data["images"])
    return {
        "images": resize_images(np.asarray(data["images"][:n])),
        "wrist_images": resize_images(np.asarray(data["wrist_images"][:n])),
    }


def download_npz(target):
    print(f"Downloading evaluation dataset...", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(NPZ_URL, target)
    print(f"Saved to {target}", flush=True)
    return str(target)


def summarize(values):
    vals = sorted(float(x) for x in values)
    if not vals:
        return {}
    return {
        "mean": float(np.mean(vals)),
        "p50": statistics.median(vals),
        "p95": vals[int(len(vals) * 0.95)],
        "min": vals[0],
        "max": vals[-1],
    }


def cosine(a, b):
    x = a.reshape(-1).astype(np.float32)
    y = b.reshape(-1).astype(np.float32)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denom == 0 else float(np.dot(x, y) / denom)


def pixel_delta_mae(left, right):
    return float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))


def main():
    args = parse_args()
    requested = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in requested if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}")
    if BASELINE not in requested:
        requested.insert(0, BASELINE)

    # Resolve NPZ
    npz_path = args.npz or str(CACHE_DIR / Path(NPZ_URL).name)
    if not os.path.isfile(npz_path):
        npz_path = download_npz(Path(npz_path))

    data = load_npz(npz_path, args.frames)
    images = data["images"]
    wrist_images = data["wrist_images"]
    n = len(images)

    print("=" * 55)
    print("FlashRT Orin — LIBERO action similarity benchmark")
    print(f"  Frames:  {n}")
    print(f"  Seed:    {args.seed}")
    print(f"  Configs: {', '.join(requested)}")
    print("=" * 55)

    all_actions = {}
    all_lat = {}
    prompt = "pick up the object"

    for name in requested:
        cfg = CONFIGS[name]
        use_adaptive = cfg.get("refresh") == "pixel_delta"
        print(f"\n  --- {name}: cache={cfg['cache']}" +
              (" adaptive" if use_adaptive else ""))

        pipe = Pi05TorchFrontendRtx(
            args.checkpoint, num_views=2, num_steps=10,
            vision_pool_factor=1, vision_num_layers=27,
            cache_frames=cfg["cache"],
        )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        pipe.set_prompt(prompt)
        pipe.calibrate_with_real_data([{
            "image": images[0], "wrist_image": wrist_images[0],
        }])

        for i in range(min(args.warmup, n)):
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed_all(args.seed + i)
            pipe.infer({"image": images[i], "wrist_image": wrist_images[i]})

        pipe.set_prompt(prompt)
        actions = [None] * n
        lat_ms = np.empty((n,), dtype=np.float32)
        forced = np.zeros((n,), dtype=np.bool_)
        last_full_image = last_full_wrist = None

        for i in range(n):
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed_all(args.seed + i)
            obs = {"image": images[i], "wrist_image": wrist_images[i]}

            force = False
            if use_adaptive and last_full_image is not None:
                delta = max(
                    pixel_delta_mae(images[i], last_full_image),
                    pixel_delta_mae(wrist_images[i], last_full_wrist),
                )
                force = delta >= args.threshold

            t0 = time.perf_counter()
            out = pipe.infer(obs, force_full=force)
            lat_ms[i] = (time.perf_counter() - t0) * 1000.0
            actions[i] = np.asarray(out["actions"], dtype=np.float32)

            if out.get("used_full_pipeline", True):
                last_full_image = images[i]
                last_full_wrist = wrist_images[i]
            forced[i] = out.get("cache_forced_full", False)

        all_actions[name] = np.stack(actions)
        all_lat[name] = lat_ms
        s = summarize(lat_ms)
        forced_count = int(np.sum(forced))
        extra = f"  forced_full={forced_count}/{n}" if use_adaptive else ""
        print(f"    p50={s['p50']:.1f}ms  p95={s['p95']:.1f}ms  "
              f"mean={s['mean']:.1f}ms{extra}")

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Similarity vs baseline
    base = all_actions[BASELINE]
    print("\n  --- Similarity vs baseline ---")
    for name in requested:
        if name == BASELINE:
            continue
        cos_vals = [cosine(base[i], all_actions[name][i]) for i in range(n)]
        cs = summarize(cos_vals)
        ls = summarize(all_lat[name])
        print(f"  {name:20s}  "
              f"cos_mean={cs['mean']:.6f}  cos_min={cs['min']:.6f}  "
              f"lat_p50={ls['p50']:.1f}ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
