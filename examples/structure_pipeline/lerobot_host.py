"""GR00T N1.7 on the LeRobot host: baseline and the three-line auto attach.

Same checkpoint and the same prepared input tensors as the official-host
measurements, so host code is the only variable. Timed boundary matches
the official-host example: backbone forward + action-head get_action,
fixed noise.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

from transformers.feature_extraction_utils import BatchFeature


def median_ms(fn, *, iters=5, rounds=9):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) / iters)
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "auto"),
                        default="baseline")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.lerobot_src))
    from lerobot.policies.groot.groot_n1_7 import GR00TN17

    model = GR00TN17.from_pretrained(args.checkpoint).to(
        device="cuda", dtype=torch.bfloat16).eval()

    payload = torch.load(args.inputs, map_location="cpu",
                         weights_only=False)
    vl_input = BatchFeature(data={
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in payload["backbone_inputs"].items()})
    action_input = BatchFeature(data={
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in payload["action_inputs"].items()})

    def hot():
        torch.manual_seed(0)
        features = model.backbone(vl_input)
        return model.action_head.get_action(
            features, action_input)["action_pred"]

    with torch.inference_mode():
        reference = hot().detach().float().cpu()
        repeat = hot().detach().float().cpu()
    exact = bool(torch.equal(reference, repeat))

    report = {
        "host": "lerobot GR00TN17",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "fixed_noise_repeat_exact": exact,
        "action_pred_norm": float(reference.norm()),
    }

    if args.arm == "auto":
        from flash_rt import structures
        from flash_rt.structures import swap
        from flash_rt.structures.impls import unavailable_report

        def run_once():
            with torch.inference_mode():
                hot()

        plan = structures.auto_swaps(model, run_once, verbose=True)
        handle = swap.attach(model, plan.swaps, observe=plan.observed,
                             revert=plan.revert)
        with torch.inference_mode():
            treated = hot().detach().float().cpu()
        report.update({
            "swaps": len(plan.swaps),
            "observed": len(plan.observed),
            "refused": len(plan.notes.get("refused", [])),
            "attention_core_variants": dict(list(
                plan.notes.get("attention_core_variants", {}).items())[:1]),
            "kernel_unavailable": unavailable_report(),
            "parity_cosine": float(torch.nn.functional.cosine_similarity(
                treated.flatten(), reference.flatten(), dim=0)),
        })

    with torch.inference_mode():
        report["eager_ms"] = median_ms(lambda: hot())

    if args.compile:
        torch._dynamo.reset()
        compiled = torch.compile(hot, mode="max-autotune-no-cudagraphs",
                                 fullgraph=False)
        with torch.inference_mode():
            for _ in range(8):
                compiled()
                torch.cuda.synchronize()
            report["compiled_ms"] = median_ms(lambda: compiled())

    if args.arm == "auto":
        report["ledger"] = handle.summary()

    print(json.dumps(report, indent=2, default=str))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
