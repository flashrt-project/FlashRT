#!/usr/bin/env python3
"""GR00T N1.5 policy-call latency on RTX SM89: FlashRT vs the upstream policy.

Every row serves the same observation: the camera frames given with
--frames (an .npz with one uint8 [H, W, 3] array per data-config video key,
keyed by that key) or, without it, deterministic synthetic frames at the
checkpoint's resolution, a state inside the checkpoint's q01..q99 range and
--prompt. Rows: ``infer`` (upstream transforms + model + unnormalization,
wall clock), ``predict_normalized`` (input staging + graph replay, wall
clock) and the graph replay alone (CUDA events); the upstream row is
``Gr00tPolicy.get_action`` (needs flash_attn). Median of --runs calls after
--warmup, with min / max.

    GROOT_N15_ROOT=<Isaac-GR00T> python benchmarks/groot_n15_latency.py \\
        --checkpoint <gr00t-n1.5 checkpoint> --data-config panda_omron
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch


def _wall(fn, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def _events(fn, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        xs.append(start.elapsed_time(end))
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def _observation(io, checkpoint: str, frames: str | None) -> dict:
    meta = json.load(open(os.path.join(checkpoint, "experiment_cfg", "metadata.json")))
    meta = meta[io.embodiment_tag]
    rng = np.random.default_rng(0)
    obs = {}
    given = np.load(frames) if frames else None
    for key in io.video_keys:
        if given is not None:
            obs[key] = given[key]
            continue
        h, w = meta["modalities"]["video"][key.split(".", 1)[1]]["resolution"]
        low = rng.integers(0, 256, size=(8, 8, 3)).astype(np.float32)
        img = np.kron(low, np.ones((h // 8 + 1, w // 8 + 1, 1)))[:h, :w]
        obs[key] = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    for key in io.state_keys:
        st = meta["statistics"]["state"][key.split(".", 1)[1]]
        lo, hi = np.asarray(st["q01"]), np.asarray(st["q99"])
        obs[key] = lo + (hi - lo) * rng.random(lo.shape)
    return obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", default=os.environ.get("GROOT_N15_DATA_CONFIG"))
    ap.add_argument("--groot-root", default=os.environ.get("GROOT_N15_ROOT"))
    ap.add_argument("--frames", default=None, help=".npz of camera frames by video key")
    ap.add_argument("--prompt", default="open the microwave door")
    ap.add_argument("--precision", default="fp8,bf16", help="comma-separated tiers")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--no-upstream", action="store_true", help="skip Gr00tPolicy")
    ap.add_argument("--json", default=None, help="also write the results here")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    from flash_rt.frontends.torch._groot_n15_io import GrootN15IO
    from flash_rt.hardware import resolve_pipeline_class
    frontend = resolve_pipeline_class("groot_n15", "torch", "rtx_sm89")
    io = GrootN15IO(args.checkpoint, args.data_config, "new_embodiment", args.groot_root)
    obs = _observation(io, args.checkpoint, args.frames)
    results = {}
    for tier in args.precision.split(","):
        fe = frontend(args.checkpoint, num_views=len(io.video_keys),
                      data_config=args.data_config, groot_root=args.groot_root,
                      use_fp8=tier == "fp8")
        fe.set_prompt(args.prompt)
        normalized = fe.io.prepare(fe._observation(obs))
        S = int(torch.as_tensor(normalized["eagle_input_ids"]).shape[1])
        fe.predict_normalized(normalized)
        graph = fe._graphs[S]
        results[tier] = {
            "tokens": S,
            "infer": _wall(lambda: fe.infer(obs), args.warmup, args.runs),
            "predict_normalized": _wall(lambda: fe.predict_normalized(normalized),
                                        args.warmup, args.runs),
            "graph_replay": _events(graph.replay, args.warmup, args.runs),
        }
        del fe, graph
        torch.cuda.empty_cache()
    if not args.no_upstream:
        if args.groot_root:
            sys.path.insert(0, args.groot_root)
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy
        cfg = DATA_CONFIG_MAP[args.data_config]
        policy = Gr00tPolicy(model_path=args.checkpoint, embodiment_tag="new_embodiment",
                             modality_config=cfg.modality_config(),
                             modality_transform=cfg.transform(), denoising_steps=4)
        lang = cfg.modality_config()["language"].modality_keys[0]
        up = {k: np.asarray(v)[None] for k, v in obs.items()}
        up[lang] = [args.prompt]
        results["upstream"] = {"get_action": _wall(lambda: policy.get_action(up), args.warmup,
                                                   args.runs)}
    print(f"GPU {torch.cuda.get_device_name()}, torch {torch.__version__}, "
          f"median [min, max] ms over {args.runs} calls")
    for tier, r in results.items():
        for row, t in r.items():
            if row == "tokens":
                continue
            print(f"{tier:9s} {row:20s} {t['median']:8.2f} [{t['min']:.2f}, {t['max']:.2f}]")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
