"""The explicit pipeline at full speed: whole-graph CUDA capture.

``groot_n17.py`` builds the assembly and times it under ``torch.compile``.
That form still runs Python between compiled regions — every guarded
seam's admission check, every host glue call — and on a graph with two
hundred seams that Python is a real cost. The deployed form removes it:
compile once, then capture the whole hot path as a single CUDA graph and
replay it. Capture records kernels, not Python, so the guards and the
glue are paid once at capture time and never again.

Capture needs fixed shapes. The Qwen3-VL backbone computes positions,
rope tables and token routing from the request at every call; for a
fixed observation shape those are constants, so this file precomputes
them once and pins the handful of host functions that would otherwise
recompute them (and, in two places, synchronise — which capture
forbids). All constants depend on shapes and token placement, never on
image or state values; the parity check at the end is against the
untouched eager host.

Usage: same arguments as ``groot_n17.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import types
from pathlib import Path

import torch

from flash_rt.structures import capture as capture_stage

from groot_n17 import Assembly, build, clone_tree, load_policy


def pin_action_noise():
    """Pin the flow-matching noise to one tensor across every call.

    Seeding is not enough once a graph is captured: replays advance the
    device RNG, so the fiftieth replay denoises a different draw than
    the reference did and the outputs are not comparable. Returning one
    saved tensor for the noise shape bakes it into the captured graph
    as a constant input — every replay, and the eager reference, then
    integrate from the same starting noise. Returns an undo callable.
    """
    original = torch.randn
    box = {}

    def fixed(*size, **kwargs):
        try:
            shape = tuple(kwargs.get("size", ())) or (
                tuple(size[0])
                if len(size) == 1 and not isinstance(size[0], int)
                else tuple(size))
        except TypeError:
            shape = None
        if shape is not None and box.get("shape") == shape:
            return box["value"]
        value = original(*size, **kwargs)
        if value.is_cuda and "value" not in box:
            box["shape"] = tuple(value.shape)
            box["value"] = value.detach().clone()
        return value

    torch.randn = fixed
    return lambda: setattr(torch, "randn", original)






def replay_ms(stages, *, iters=50, rounds=9):
    timings = {name: [] for name in stages}
    names = list(stages)
    for r in range(rounds):
        for name in names[r % len(names):] + names[:r % len(names)]:
            stage = stages[name]
            for _ in range(5):
                stage.replay(sync=False)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            with torch.cuda.stream(stage.stream):
                start.record()
            for _ in range(iters):
                stage.replay(sync=False)
            with torch.cuda.stream(stage.stream):
                end.record()
            torch.cuda.synchronize()
            timings[name].append(start.elapsed_time(end) / iters)
    return {name: statistics.median(v) for name, v in timings.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("explicit", "auto"),
                        default="explicit")
    parser.add_argument("--baseline", choices=("interleaved", "sequential"),
                        default="interleaved",
                        help="sequential: measure the stock graph first, "
                             "release it, then bind with immediate "
                             "consumption — the tight-memory bind mode; "
                             "peaks are reported per phase")
    parser.add_argument("--scheme", default=None,
                        help="named precision scheme for the auto arm "
                             "(e.g. nvfp4_balance); default negotiates "
                             "FP8 chains")
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-assets", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    policy = load_policy(args.host, args.checkpoint, args.backbone_assets)
    model = policy.model
    fixture = torch.load(args.fixture, map_location="cpu",
                         weights_only=False)["inputs"]

    from flash_rt.structures import swap
    from flash_rt.structures.impls import unavailable_report
    from flash_rt.structures.impls.cadence_static.cross_attention import (
        wire_refresh_to_producer)

    captured = {}
    original_get_action = model.get_action

    def spy(inputs, options=None):
        captured["inputs"] = clone_tree(inputs)
        return original_get_action(inputs, options)

    model.get_action = spy
    with torch.inference_mode():
        policy.get_action(fixture)
    model.get_action = original_get_action

    backbone_inputs, action_inputs = model.prepare_input(
        dict(captured["inputs"]))
    backbone_inputs = clone_tree(backbone_inputs)
    action_inputs = clone_tree(action_inputs)

    unpin = pin_action_noise()

    def hot():
        out = model.backbone(backbone_inputs)
        return model.action_head.get_action(
            out, action_inputs)["action_pred"]

    try:
        with torch.inference_mode():
            reference = hot().detach().float().cpu()

        # the host at full speed: the registered family lowering pins
        # the shape glue, the capture door does the rest
        peaks = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stock = capture_stage(
            torch.compile(hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=model, warmup=8, gate_cos=0, min_speedup=0)

        sequential = args.baseline == "sequential"
        stock_ms = None
        if sequential:
            # the baseline is measured and released before anything
            # binds: under locked clocks a sequential A-then-B holds,
            # and the bind window never carries the stock graph's alias
            # of the original weights
            stock_ms = replay_ms({"stock_graph": stock})["stock_graph"]
            stock.restore_host()
            del stock
            torch._dynamo.reset()
            if torch.cuda.is_available():
                peaks["baseline_phase_gb"] = round(
                    torch.cuda.max_memory_allocated() / 2**30, 3)
                torch.cuda.reset_peak_memory_stats()

        def run_once():
            with torch.inference_mode():
                hot()

        wstore = None
        if sequential:
            from flash_rt.structures.storage import WeightStore
            wstore = WeightStore(checkpoint=str(args.checkpoint))

        if args.arm == "explicit":
            asm, extras = build(model, run_once)
            handle = swap.attach(model, asm.swaps, consume=False,
                                 observe=extras["observed"],
                                 revert=extras["revert"],
                                 on_guard_fail="raise")
            statics = extras["cadence_statics"]
            seats = {"seats_bound": dict(asm.families),
                     "swaps": len(asm.swaps),
                     "refused": len(asm.refused)}
        else:
            from flash_rt import structures
            from flash_rt.structures.impls.cadence_static.cross_attention \
                import (bind_cross_attention_kv, capture_cross_attention_kv,
                        discover_cross_attention_kv)
            sites = discover_cross_attention_kv(model)
            caps = capture_cross_attention_kv(sites, run_once)
            scheme_kw = ({"scheme": args.scheme}
                         if args.scheme else {})
            if wstore is not None:
                scheme_kw["stream_store"] = wstore
            plan = structures.auto_swaps(model, run_once, verbose=True,
                                         **scheme_kw)
            statics = []
            if sites:
                cadence_swaps, statics = bind_cross_attention_kv(
                    sites, caps, replacements=plan.swaps)
                plan.swaps.update(cadence_swaps)
            handle = swap.attach(model, plan.swaps, consume=False,
                                 observe=plan.observed,
                                 revert=plan.revert)
            seats = {"swaps": len(plan.swaps),
                     "observed": len(plan.observed),
                     "refused": len(plan.notes.get("refused", [])),
                     "format_race": plan.notes.get("format_race"),
                     "regions": plan.notes.get("regions"),
                     "regions_bound": plan.notes.get("regions_bound"),
                     "regions_refused": plan.notes.get(
                         "regions_refused")}
        weights_receipt = None
        if sequential:
            weights_receipt = handle.consume(wstore)
        if statics:
            # the refresh rides the producer's forward: the captured
            # graph writes the banks from this call's own features
            wire_refresh_to_producer(model, statics, run_once)

        torch._dynamo.reset()
        treated = capture_stage(
            torch.compile(hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=model, warmup=8, gate_cos=0, min_speedup=0)

        if os.environ.get("FRT_KPROFILE"):
            # kernel census of the bound form: eager names every chain
            # kernel; the replay table shows what capture actually runs
            from torch.profiler import ProfilerActivity, profile

            def _t(row):
                return getattr(row, "self_device_time_total",
                               getattr(row, "self_cuda_time_total", 0.0))

            def census(label, fn):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                rows = prof.key_averages()
                print(f"=== KPROFILE {label} ===", flush=True)
                print(rows.table(sort_by="self_cuda_time_total",
                                 row_limit=100), flush=True)
                print(f"KTOTAL {label} {sum(_t(r) for r in rows) / 3:.1f}"
                      "us per call", flush=True)

            def eager_hot():
                with torch.inference_mode():
                    hot()

            census("eager", eager_hot)
            census("replay", treated.replay)
            return 0

        if sequential:
            if torch.cuda.is_available():
                peaks["bind_phase_gb"] = round(
                    torch.cuda.max_memory_allocated() / 2**30, 3)
                torch.cuda.reset_peak_memory_stats()
            medians = {"stock_graph": stock_ms,
                       "structures_graph": replay_ms(
                           {"structures_graph": treated})
                       ["structures_graph"]}
        else:
            medians = replay_ms({"stock_graph": stock,
                                 "structures_graph": treated})
        treated.replay()
        parity = float(torch.nn.functional.cosine_similarity(
            treated.output.detach().float().cpu().flatten(),
            reference.flatten(), dim=0))
        report = {
            "arm": args.arm,
            "scheme": args.scheme,
            "baseline": args.baseline,
            "peaks": peaks or None,
            "device": torch.cuda.get_device_name(),
            **seats,
            "kernel_unavailable": unavailable_report(),
            "graph_lowering": treated.certification.get("graph_lowering"),
            "parity_cosine": parity,
            "stock_graph_ms": medians["stock_graph"],
            "structures_graph_ms": medians["structures_graph"],
            "speedup_vs_stock_graph": (medians["stock_graph"]
                                       / medians["structures_graph"]),
            "ledger": handle.summary(),
        }
        if weights_receipt is None:
            from flash_rt.structures.storage import WeightStore
            weights_receipt = handle.consume(
                WeightStore(checkpoint=str(args.checkpoint)))
        report["weights"] = weights_receipt
        print(json.dumps(report, indent=2, default=str))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2,
                                              default=str))
        handle.detach()
        treated.restore_host()
        if not sequential:
            stock.restore_host()
    finally:
        unpin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
