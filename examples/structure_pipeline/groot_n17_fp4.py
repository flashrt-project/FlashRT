"""The FP4-specialist explicit assembly.

The explicit tier permits hand-written dataflow where the mechanism
has no family yet. This runner takes the measured FP4 band and adds
the one big lever the census names: FA4 attention for the backbone.
No pin is touched — the capture lowering's fixed vision attention and
the per-head rope route both resolve their kernel through the host's
attention-interface registry, so registering an FA4 interface and
switching the config routes every backbone attention call through it.

Usage: same arguments as full_graph.py (explicit arm implied).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flash_rt.structures import capture as capture_stage

from full_graph import pin_action_noise, replay_ms
from groot_n17 import build, clone_tree, load_policy


def install_fa4_interface(model) -> str | None:
    """Register FA4 as a host attention interface and switch to it."""
    import importlib

    from flash_rt.structures.impls import KernelUnavailable, hub_kernel

    try:
        fa4 = hub_kernel("kernels-community/flash-attn4", ">=0")
    except KernelUnavailable as missing:
        return f"fa4 unavailable: {missing}"

    def fa4_interface(module, query, key, value, attention_mask=None,
                      scaling=None, dropout=0.0, is_causal=False,
                      **kwargs):
        del module, attention_mask, dropout, kwargs
        out = fa4.flash_attn_func(
            query.transpose(1, 2), key.transpose(1, 2),
            value.transpose(1, 2), softmax_scale=scaling,
            causal=bool(is_causal))
        if isinstance(out, tuple):
            out = out[0]
        return out, None

    base = model.backbone.model.model
    modeling = importlib.import_module(type(base.visual).__module__)
    registry = modeling.ALL_ATTENTION_FUNCTIONS
    try:
        registry["flashrt_fa4"] = fa4_interface
    except TypeError:
        registry.register("flashrt_fa4", fa4_interface)
    for cfg in (base.visual.config, base.language_model.config):
        cfg._attn_implementation = "flashrt_fa4"
    qwen = model.backbone.model
    qwen.config._attn_implementation = "flashrt_fa4"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
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
    from flash_rt.structures.impls.cadence_static.cross_attention import (
        wire_refresh_to_producer)

    captured = {}
    original = model.get_action

    def spy(inputs, options=None):
        captured["inputs"] = clone_tree(inputs)
        return original(inputs, options)

    model.get_action = spy
    with torch.inference_mode():
        policy.get_action(fixture)
    model.get_action = original
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

        stock = capture_stage(
            torch.compile(hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=model, warmup=8, gate_cos=0, min_speedup=0)

        fa4_note = install_fa4_interface(model)

        def run_once():
            with torch.inference_mode():
                hot()

        asm, extras = build(model, run_once)
        handle = swap.attach(model, asm.swaps, consume=False,
                             observe=extras["observed"],
                             revert=extras["revert"],
                             on_guard_fail="raise")
        if extras["cadence_statics"]:
            wire_refresh_to_producer(model, extras["cadence_statics"],
                                     run_once)

        torch._dynamo.reset()
        treated = capture_stage(
            torch.compile(hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=model, warmup=8, gate_cos=0, min_speedup=0)

        medians = replay_ms({"stock_graph": stock,
                             "structures_graph": treated})
        treated.replay()
        parity = float(torch.nn.functional.cosine_similarity(
            treated.output.detach().float().cpu().flatten(),
            reference.flatten(), dim=0))
        report = {
            "arm": "fp4_specialist",
            "fa4_interface": fa4_note or "installed",
            "device": torch.cuda.get_device_name(),
            "seats_bound": dict(asm.families),
            "swaps": len(asm.swaps),
            "parity_cosine": parity,
            "stock_graph_ms": medians["stock_graph"],
            "structures_graph_ms": medians["structures_graph"],
            "ledger": handle.summary(),
        }
        print(json.dumps(report, indent=2, default=str))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2,
                                              default=str))
        handle.detach()
        treated.restore_host()
        stock.restore_host()
    finally:
        unpin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
