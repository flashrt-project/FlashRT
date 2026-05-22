#!/usr/bin/env python
"""LingBot-VLA (Thor sm_110) latency + accuracy benchmark.

Runs the LingBot-VLA flow-matching action expert end-to-end through the
low-level ``graph_runner.sample_actions_graph`` path (LingBot is not registered
in ``load_model`` — see ``docs/lingbot_usage.md``) and reports, per denoise
step count:

- whether the FA4 fast path is active (vs the fmha fallback),
- P50 CUDA-graph-replay latency,
- cosine similarity of the action chunk vs an optional reference ``.pt``.

A/B the attention path with ``FLASHRT_THOR_FA4=0`` (force fmha) vs ``=1`` (FA4).

    CUTE_DSL_ARCH=sm_101a python benchmarks/lingbot_thor_latency.py \
        --checkpoint /path/to/lingbot-vla-4b \
        --calibration /path/to/lingbot_thor_static.json \
        --inputs /path/to/baseline_artifacts_10/inputs \
        --reference /path/to/baseline_artifacts_10/outputs/actions.pt \
        --steps 50 25 10

Missing checkpoint/inputs -> the benchmark prints why and exits 0 (clean skip),
so it is safe to invoke in environments without local fixtures.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_INPUT_KEYS = ["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"]


def _skip(msg: str) -> "None":
    print(f"[lingbot-bench] SKIP: {msg}")
    sys.exit(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="LingBot-VLA Thor latency + accuracy")
    ap.add_argument("--checkpoint", help="lingbot-vla-4b dir (model.safetensors)")
    ap.add_argument("--calibration", help="FP8 static-scale calibration JSON")
    ap.add_argument("--inputs", help="dir with %s .pt" % "/".join(_INPUT_KEYS))
    ap.add_argument("--reference", default=None,
                    help="optional reference action chunk .pt for cosine")
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 25, 10])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    for name, val in (("--checkpoint", args.checkpoint),
                      ("--calibration", args.calibration),
                      ("--inputs", args.inputs)):
        if not val:
            _skip(f"{name} not provided")
        if not Path(val).exists():
            _skip(f"{name} path does not exist: {val}")

    import torch
    from flash_rt.executors.torch_weights import SafetensorsSource
    from flash_rt.executors.weight_loader import WeightLoader
    from flash_rt.frontends.torch._lingbot_thor_spec import build_spec
    from flash_rt.models.lingbot.buffer_binder import bind_target_to_device
    from flash_rt.models.lingbot.kernel_ops import clear_fp8_weight_cache
    from flash_rt.models.lingbot.graph_runner import sample_actions_graph
    from flash_rt.models.lingbot import calibration as calib
    from flash_rt.hardware.thor import fa4_backend

    if not torch.cuda.is_available():
        _skip("CUDA device not available")

    print(f"[lingbot-bench] FA4 status: {fa4_backend.status()}"
          f"  (FLASHRT_THOR_FA4={os.environ.get('FLASHRT_THOR_FA4', '1')})")

    ref = None
    if args.reference and Path(args.reference).exists():
        ref = torch.load(args.reference).float().reshape(-1).cuda()

    clear_fp8_weight_cache()
    src = SafetensorsSource(str(Path(args.checkpoint) / "model.safetensors"),
                            device="cpu", strip_prefix="")
    target = SimpleNamespace()
    WeightLoader(src, target=target, spec=build_spec()).run()
    bind_target_to_device(target, dtype=torch.bfloat16, device="cuda")
    calib.set_static_scales(calib.load_calibration(
        args.calibration, device=torch.device("cuda")))
    inp = {k: torch.load(Path(args.inputs) / f"{k}.pt").cuda() for k in _INPUT_KEYS}

    print(f"[lingbot-bench] {'steps':>5} | {'P50 (ms)':>9} | {'finite':>6} | cosine vs reference")
    for ns in args.steps:
        replay, out, g = sample_actions_graph(
            target=target, num_steps=ns, warmup_iters=3, **inp)
        replay(); torch.cuda.synchronize()
        finite = bool(torch.isfinite(out).all().item())
        cos = "n/a"
        if ref is not None and ref.numel() == out.numel():
            o = out.float().reshape(-1).cuda()
            cos = "%.6f" % float(torch.nn.functional.cosine_similarity(
                o, ref, dim=0).item())
        ts = []
        for _ in range(args.iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            replay(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        p50 = sorted(ts)[len(ts) // 2] * 1000
        print(f"[lingbot-bench] {ns:>5} | {p50:>9.1f} | {str(finite):>6} | {cos}")
        del replay, out, g
        torch.cuda.empty_cache(); clear_fp8_weight_cache()


if __name__ == "__main__":
    main()
