"""GROOT N1.7 Thor end-to-end latency benchmark.

Measures the full per-frame image->action path (vision backbone graph
replay + action graph replay) with wall-clock timing, reporting the
median over ``--iters`` iterations after ``--warmup`` warmup frames,
plus the per-component split and an FP8-tier action-cosine check when
both tiers are run.

Requires an N1.7 checkpoint snapshot and an aux fixture captured from
the official preprocessing path (see ``tests/_helpers/groot_n17``).

Usage:
    python benchmarks/groot_n17_thor_latency.py \
        --ckpt <path-to-GR00T-N1.7-snapshot> \
        --aux <path-to-aux-fixture.pt> \
        --tier fp4 --views 2 \
        --embodiment oxe_droid_relative_eef_relative_joint
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch


def _cos(a, b):
    a = torch.as_tensor(a).float().flatten().double().cpu()
    b = torch.as_tensor(b).float().flatten().double().cpu()
    return float(a @ b / (a.norm() * b.norm()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="N1.7 checkpoint snapshot directory")
    ap.add_argument("--aux", required=True,
                    help="aux fixture .pt (single dict or list of dicts)")
    ap.add_argument("--tier", choices=["fp8", "fp4"], default="fp4")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--embodiment",
                    default="oxe_droid_relative_eef_relative_joint")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--calib-aux", default=None,
                    help="optional aux-list .pt for multi-sample calibration")
    ap.add_argument("--ref-out", default=None,
                    help="save the first inference output to this .pt")
    ap.add_argument("--ref-in", default=None,
                    help="compare the first inference output against this .pt")
    ap.add_argument("--min-cosine", type=float, default=None,
                    help="fail (exit 1) when the action cosine against "
                         "--ref-in falls below this threshold")
    args = ap.parse_args()

    if args.tier == "fp4":
        from flash_rt.frontends.torch.groot_n17_thor_fp4 import (
            GrootN17TorchFrontendThorFP4 as Frontend)
    else:
        from flash_rt.frontends.torch.groot_n17_thor_fp8 import (
            GrootN17TorchFrontendThorFP8 as Frontend)

    aux = torch.load(args.aux, weights_only=False, map_location="cpu")
    if isinstance(aux, list):
        aux = aux[0]

    fe = Frontend(args.ckpt, num_views=args.views,
                  embodiment_tag=args.embodiment)
    fe.set_prompt(aux=aux, prompt="benchmark")
    if args.calib_aux:
        # Multi-sample act-scale refinement (requires set_prompt first).
        fe.calibrate(torch.load(args.calib_aux, weights_only=False,
                                map_location="cpu"))

    state_normed = torch.zeros(1, 1, 132, dtype=torch.float32)
    noise = aux["initial_noise"].to("cuda").bfloat16().contiguous()

    def e2e_once():
        fe._backbone_features = fe.run_backbone_graph(aux)
        return fe.infer(state_normed, initial_noise=noise.clone())

    out0 = e2e_once()
    if args.ref_out:
        torch.save(out0.detach().float().cpu(), args.ref_out)
    cosine = None
    if args.ref_in:
        ref = torch.load(args.ref_in, weights_only=False, map_location="cpu")
        cosine = _cos(out0, ref)
        print(f"action cosine vs reference: {cosine:.6f}")
    elif args.min_cosine is not None:
        raise SystemExit("--min-cosine requires --ref-in")

    for _ in range(args.warmup):
        e2e_once()
    torch.cuda.synchronize()

    e2e, bb, act = [], [], []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fe._backbone_features = fe.run_backbone_graph(aux)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fe.infer(state_normed, initial_noise=noise.clone())
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        bb.append((t1 - t0) * 1e3)
        act.append((t2 - t1) * 1e3)
        e2e.append((t2 - t0) * 1e3)

    det = e2e_once()
    m = statistics.median
    print(f"[groot_n17-thor-{args.tier}] backbone={m(bb):.2f} ms  "
          f"action={m(act):.2f} ms  e2e={m(e2e):.2f} ms  "
          f"({1000.0 / m(e2e):.1f} Hz)  "
          f"[median of {args.iters} after {args.warmup} warmup]  "
          f"replay-determinism={_cos(out0, det):.6f}")

    if args.min_cosine is not None and cosine < args.min_cosine:
        raise SystemExit(
            f"action cosine {cosine:.6f} is below --min-cosine "
            f"{args.min_cosine:.6f}")


if __name__ == "__main__":
    main()
