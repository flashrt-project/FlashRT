"""Compare official PyTorch eager and FlashRT Graph on the same RDNA device.

Run each backend in a separate process with the same independent official
fixture. Model-call timing starts with GPU-ready tensors; workflow timing
includes CPU preparation, transfers, fresh model execution and action decoding.
Loading, compilation, capture, warmup and camera/file acquisition are excluded.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/_helpers/groot_n17"))
from capture_rdna_reference import fixed_noise, official_policy, prepare_official, sha256, to_device


def check_actions(predicted, decoded, entry):
    actual = predicted[..., :17].detach().cpu().double().flatten()
    expected = entry["action_pred"][..., :17].double().flatten()
    error = actual - expected
    quality = {"cosine": float(torch.nn.functional.cosine_similarity(actual, expected, dim=0)),
               "rmse": float(error.square().mean().sqrt()), "max_abs": float(error.abs().max())}
    physical = {key: float(np.abs(np.asarray(value) - entry["actions"][key]).max())
                for key, value in decoded.items()}
    xyz = float(np.abs(np.asarray(decoded["eef_9d"])[..., :3]
                       - entry["actions"]["eef_9d"][..., :3]).max())
    quality.update(physical_max=physical, xyz_max_m=xyz)
    if not (quality["cosine"] >= .999 and quality["rmse"] < .015
            and quality["max_abs"] < .125 and max(physical.values()) < .05 and xyz < .02):
        raise RuntimeError(f"official-reference gates failed: {quality}")
    return quality


def timed(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall, spans = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - started) * 1000)
        spans.append(begin.elapsed_time(end))
    return {"wall_p50_ms": statistics.median(wall),
            "wall_p95_ms": float(np.percentile(wall, 95)),
            "wall_min_ms": min(wall), "wall_max_ms": max(wall),
            "wall_samples_ms": wall, "device_span_samples_ms": spans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("official", "flashrt"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--reference", required=True, help="trusted same-device official aux fixture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.warmup < 1 or args.iters < 1 or args.threads < 1:
        parser.error("warmup, iters and threads must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("requires ROCm PyTorch and a visible RDNA GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] not in (
            "gfx1100", "gfx1151", "gfx1201"):
        parser.error("requires gfx1100, gfx1151 or gfx1201")
    torch.set_num_threads(args.threads)
    entry = torch.load(args.reference, map_location="cpu", weights_only=False)[0]
    obs, noise = entry["inputs"], entry["initial_noise"]
    report = {"backend": args.backend, "torch": torch.__version__, "hip": torch.version.hip,
              "device": str(torch.cuda.get_device_properties(0)),
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "warmup": args.warmup, "iterations": args.iters, "threads": args.threads,
              "reference_sha256": sha256(args.reference), "script_sha256": sha256(__file__)}
    with ExitStack() as stack, torch.inference_mode():
        if args.backend == "official":
            policy, audit = stack.enter_context(official_policy(args.checkpoint, args.processor))
            report["official"] = audit
            ready = to_device(prepare_official(policy, obs))
            with fixed_noise(noise):
                predicted = policy.model.get_action(**ready)["action_pred"].float()
            decoded = policy.processor.decode_action(
                predicted.cpu().numpy(), policy.embodiment_tag, obs["state"])
            calls = {"model_call": lambda: policy.model.get_action(**ready),
                     "workflow": lambda: policy.get_action(obs)}
        else:
            import flash_rt
            from flash_rt.amd.frontends.torch.groot_n17_rdna import frames_from_observation
            from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

            source_root = Path(flash_rt.__file__).resolve().parent
            report["flashrt_source_sha256"] = {
                str(path.relative_to(source_root)): sha256(path)
                for path in (source_root / "api.py",
                             source_root / "amd/frontends/torch/groot_n17_rdna.py",
                             source_root / "amd/models/groot_n17/pipeline_rdna.py",
                             *sorted((source_root / "amd/hardware/rdna").glob("*.py")),
                             *sorted((source_root / "amd/hardware/rdna35").glob("*.py")))}
            model = flash_rt.load_model(args.checkpoint, config="groot_n17", use_fp8=False,
                                       fuse_image_embeds=True)
            frontend = model.pipeline
            processor = Gr00tN1d7Processor.from_pretrained(args.checkpoint, model_name=args.processor)
            frontend.set_hf_processor(processor)
            state, aux = frontend.prepare_observation(obs)
            model.set_prompt(aux=aux)
            predicted = model.infer(state, aux=aux, initial_noise=noise)
            decoded = frontend.denormalize_action(predicted, state_dict=obs["state"])
            ready_state, ready_aux = to_device(state), to_device(aux)
            cameras = processor.modality_configs[frontend.embodiment_tag]["video"].modality_keys

            def workflow():
                state, aux = frontend.prepare_observation(obs)
                result = model.infer(state, aux=aux)
                return frontend.denormalize_action(result, state_dict=obs["state"])

            def frames_workflow():
                frames = frames_from_observation(obs, cameras)
                state = frontend.normalize_state(obs["state"])
                result = model.infer(state, frames=frames)
                return frontend.denormalize_action(result, state_dict=obs["state"])

            calls = {"model_call": lambda: model.infer(ready_state, aux=ready_aux),
                     "workflow": workflow, "frames_workflow": frames_workflow}
        report["quality"] = check_actions(predicted, decoded, entry)
        report["measurements"] = {name: timed(fn, args.warmup, args.iters)
                                  for name, fn in calls.items()}
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value["wall_p50_ms"] for key, value in report["measurements"].items()}))


if __name__ == "__main__":
    main()
