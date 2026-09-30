"""GROOT N1.7 AMD tensor-path latency benchmark.

Measures the production per-frame path from processor-derived image/text
tensors through the captured FP8 backbone and four-step BF16 action graph.
The aux fixture can also carry ``inputs.state`` from the official DROID
processor; when present, that real state is used instead of the checkpoint
statistics midpoint.

Example::

    python benchmarks/groot_n17_amd_latency.py \
      --ckpt /workspace/checkpoints/GR00T-N1.7-3B \
      --aux /workspace/data/groot_n17_aux_2v_traj1_step0_single.pt \
      --warmup 50 --iters 100 --json-out /workspace/groot-mi300x.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(values),
        "p50_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values),
    }


def _midpoint_state(frontend) -> dict[str, np.ndarray]:
    state = {}
    for key, stats in frontend._read_statistics()["state"].items():
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        state[f"state.{key}"] = (0.5 * (q01 + q99)).reshape(1, 1, -1)
    return state


def _real_or_midpoint_state(frontend, aux: dict) -> tuple[dict, str]:
    if "inputs" in aux and "state" in aux["inputs"]:
        return ({f"state.{key}": value
                 for key, value in aux["inputs"]["state"].items()},
                "official_fixture")
    return _midpoint_state(frontend), "checkpoint_statistics_midpoint"


def _time_calls(fn, count: int) -> list[float]:
    samples = []
    for _ in range(count):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--embodiment",
                    default="oxe_droid_relative_eef_relative_joint")
    ap.add_argument("--hardware", default=None,
                    help="optional explicit AMD target (default: detect device)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--normalized-out", default=None,
                    help="save the first normalized action tensor")
    ap.add_argument("--profile-e2e-only", action="store_true",
                    help="exit after the E2E samples so they are the trace tail")
    args = ap.parse_args()

    from flash_rt.amd.frontends.torch.groot_n17 import (
        GrootN17TorchFrontendAmd,
    )

    aux = torch.load(args.aux, weights_only=False, map_location="cpu")
    if isinstance(aux, list):
        if len(aux) != 1:
            raise ValueError("--aux must contain one observation for latency")
        aux = aux[0]

    frontend = GrootN17TorchFrontendAmd(
        args.ckpt,
        num_views=args.views,
        embodiment_tag=args.embodiment,
        hardware=args.hardware,
    )
    frontend.set_prompt(aux=aux, prompt="benchmark")
    state, state_source = _real_or_midpoint_state(frontend, aux)
    state_normed = frontend.normalize_state(state)
    noise = aux["initial_noise"].to("cuda").bfloat16().contiguous()

    def backbone_once():
        frontend._backbone_features = frontend.run_backbone_graph(aux)
        return frontend._backbone_features

    def action_once():
        return frontend.infer(state_normed, initial_noise=noise)

    def e2e_once():
        backbone_once()
        return action_once()

    # The mixin's aux-aware infer entry captures the backbone graph; the base
    # infer captures the action graph. Subsequent component calls can then
    # replay either graph independently.
    out0 = frontend.infer(
        state_normed, aux=aux, initial_noise=noise,
    ).detach().float().cpu().clone()
    if args.normalized_out:
        torch.save(out0, args.normalized_out)
    for _ in range(args.warmup):
        e2e_once()
    torch.cuda.synchronize()

    e2e = _time_calls(e2e_once, args.iters)
    if args.profile_e2e_only:
        s = _summary(e2e)
        print(f"e2e: min {s['min_ms']:.2f} ms  p50 {s['p50_ms']:.2f} ms  "
              f"p95 {s['p95_ms']:.2f} ms  max {s['max_ms']:.2f} ms")
        return
    backbone = _time_calls(backbone_once, args.iters)
    action = _time_calls(action_once, args.iters)
    out1 = e2e_once().detach().float().cpu().clone()
    deterministic = bool(torch.equal(out0, out1))

    result = {
        "model": "GR00T-N1.7-3B",
        "hardware": frontend.hardware,
        "device": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "views": args.views,
        "sequence_tokens": int(aux["llm_input_embeds"].shape[1]),
        "vision_tokens": int(aux["pixel_features"].shape[0]),
        "state_source": state_source,
        "warmup": args.warmup,
        "iterations": args.iters,
        "boundary": "processor tensors -> normalized action tensor",
        "e2e": _summary(e2e),
        "backbone": _summary(backbone),
        "action": _summary(action),
        "replay_bit_identical": deterministic,
    }

    for name in ("backbone", "action", "e2e"):
        s = result[name]
        print(f"{name:>8}: min {s['min_ms']:8.2f} ms  "
              f"p50 {s['p50_ms']:8.2f} ms  p95 {s['p95_ms']:8.2f} ms  "
              f"max {s['max_ms']:8.2f} ms")
    print(f"state: {state_source}; replay bit-identical: {deterministic}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
