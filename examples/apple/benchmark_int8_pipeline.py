"""Experimental dynamic row/channel INT8 vs BF16 through the complete native Pi0.5 plan.

Reuses the model's semantic traversal. Activation scales are recomputed on
device every replay. Optional SmoothQuant calibrates channel transforms on
disjoint observations. No persistent calibration cache is used; the JSON is
a diagnostic report. Timings exclude setup and preprocessing.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from examples.apple.benchmark_native_pipeline import metrics, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parents[1] / "thor/assets/pi05_libero.npz",
    )
    ap.add_argument("--sample-offset", type=int, default=4)
    ap.add_argument("--validation-samples", type=int, default=4)
    ap.add_argument("--smoothquant", action="store_true")
    ap.add_argument("--smooth-alpha", type=float, default=0.5)
    ap.add_argument("--calibration-samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--prompt-length", type=int)
    ap.add_argument(
        "--variants",
        nargs="+",
        choices=("decode", "prefix-decode", "all", "all-original", "all-tiled"),
        default=["decode"],
    )
    ap.add_argument(
        "--paired",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="Alternate two qualified variants (or bf16); release setup weights before timing",
    )
    args = ap.parse_args()
    if args.paired and (
        len(set(args.paired)) != 2
        or any(v != "bf16" and v not in args.variants for v in args.paired)
    ):
        ap.error("paired names must be distinct and selected in --variants (or bf16)")
    if (
        min(
            args.validation_samples,
            args.steps,
            args.warmup,
            args.iters,
        )
        < 1
    ):
        ap.error("counts must be positive")
    if args.sample_offset < 0:
        ap.error("sample-offset must be nonnegative")
    if not 0 <= args.smooth_alpha <= 1:
        ap.error("smooth-alpha must be in [0,1]")
    if args.smoothquant and not 0 < args.calibration_samples <= args.sample_offset:
        ap.error("SmoothQuant needs nonempty calibration indices before sample-offset")
    if any(
        os.environ.get(k, "0") not in ("", "0")
        for k in (
            "MTL_DEBUG_LAYER",
            "MTL_SHADER_VALIDATION",
            "MTL_CAPTURE_ENABLED",
            "PYTORCH_MPS_TRACE_SIGNPOSTS",
        )
    ):
        raise RuntimeError("Disable GPU validation and tracing for timing")
    os.environ["FLASH_RT_PALIGEMMA_TOKENIZER"] = str(args.tokenizer.resolve())
    import torch

    from flash_rt.core.quant.calibrator import _checkpoint_hash
    from flash_rt.frontends.torch.pi05_mps import Pi05TorchFrontendMps
    from flash_rt.models.pi05.native_plan_mps import NativePi05Plan

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    displays = json.loads(
        subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True
        )
    )
    report = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_hash": _checkpoint_hash(str(args.checkpoint)),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "target": "apple-metal-int8-row-channel-w8a8-bf16-output",
        "hardware": {
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "gpus": [d["sppci_model"] for d in displays["SPDisplaysDataType"]],
        },
        "torch_version": torch.__version__,
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "steps": args.steps,
        "scope": "CPU prepared BF16 inputs -> patch/vision/projector/prefix/all denoising steps -> CPU raw actions; excludes preprocessing, tokenization, output unnormalization and setup",
        "gate": {"cosine": 0.999, "nrmse": 0.03},
        "blocks": [],
        "parity": {},
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    current = None
    try:
        with torch.inference_mode():
            f = Pi05TorchFrontendMps(
                args.checkpoint,
                num_steps=args.steps,
                action_dim=7,
                num_views=2,
            )
            source = f.pipeline
            print("Checkpoint loaded", flush=True)
            inputs = []
            with np.load(args.input, allow_pickle=False) as data:
                count = args.sample_offset + args.validation_samples
                if count > int(data["n"]):
                    raise ValueError("Not enough validation observations")
                for i in range(0 if args.smoothquant else args.sample_offset, count):
                    state = data[f"state_{i}"]
                    lo, hi = (
                        np.asarray(f.norm_stats["state"][k])[: state.size]
                        for k in ("q01", "q99")
                    )
                    f.set_prompt(
                        "pick up the black bowl and place it on the plate",
                        state=2 * (state - lo) / (hi - lo + 1e-6) - 1,
                    )
                    length = args.prompt_length or f._prompt_len
                    if not 0 < length <= f._prompt_len:
                        raise ValueError("Invalid requested prompt length")
                    images = torch.from_numpy(
                        np.stack((data[f"img_{i}"], data[f"wrist_{i}"])).astype(
                            np.float32
                        )
                        / 127.5
                        - 1
                    ).bfloat16()
                    noise = torch.from_numpy(
                        np.random.default_rng(20261102 + i)
                        .standard_normal((f.chunk_size, 32))
                        .astype(np.float32)
                    ).bfloat16()
                    inputs.append((images, f._prompt_cpu[:length].clone(), noise))
            lengths = [x[1].shape[0] for x in inputs]
            heldout = inputs[args.sample_offset :] if args.smoothquant else inputs
            length = heldout[0][1].shape[0]
            report.update(
                prompt_lengths=lengths,
                timed_prompt_len=length,
                num_views=2,
                horizon=f.chunk_size,
                activation_scales="dynamic per token; weights per output channel",
                validation_indices=list(range(args.sample_offset, count)),
            )

            def build(options=None, probes=False, prompt_len=length):
                return NativePi05Plan(
                    source,
                    prompt_len,
                    library=args.library,
                    int8_options=options,
                    capture_probes=probes,
                    dynamic_prompt=False,
                )

            smooth_amax = None
            if args.smoothquant:
                smooth_amax = {}
                for sample in inputs[: args.calibration_samples]:
                    if current is None or current.prompt_len != sample[1].shape[0]:
                        if current is not None:
                            current.close()
                        current = build(
                            {"stages": ("prefix",), "observe": True},
                            prompt_len=sample[1].shape[0],
                        )
                    current.forward(*sample)
                    for key, value in current.plan.activation_maxima().items():
                        smooth_amax[key] = np.maximum(
                            smooth_amax.get(key, value), value
                        )
                current.close()
                current = None
                report["smoothquant"] = {
                    "alpha": args.smooth_alpha,
                    "stages": ["prefix"],
                    "calibration_indices": list(range(args.calibration_samples)),
                    "method": "per-input-channel maximum across BF16 tokens and observations",
                    "scales": "S=amax(X)^alpha / amax(W,input-channel)^(1-alpha); FP32; floor 1e-5",
                    "cache_reused": False,
                    "sites": {
                        key: {
                            "channels": int(v.size),
                            "sha256": hashlib.sha256(v.tobytes()).hexdigest(),
                        }
                        for key, v in smooth_amax.items()
                    },
                }
                print(
                    "SmoothQuant observed", len(smooth_amax), "prefix sites", flush=True
                )
                save()

            references = []
            for sample in heldout:
                if current is None or current.prompt_len != sample[1].shape[0]:
                    if current is not None:
                        current.close()
                    current = build(probes=True, prompt_len=sample[1].shape[0])
                references.append(current.forward(*sample, capture_probes=True))
            current.close()
            current = None
            stage_sets = {
                "decode": ("decode",),
                "prefix-decode": ("prefix", "decode"),
                "all": ("vision", "prefix", "decode"),
                "all-original": ("vision", "prefix", "decode"),
                "all-tiled": ("vision", "prefix", "decode"),
            }

            def variant_options(variant):
                options = {"stages": stage_sets[variant]}
                options.update(
                    tune_tiles=variant != "all-original",
                    fuse_gelu=variant not in ("all-original", "all-tiled"),
                )
                if args.smoothquant:
                    options.update(
                        smooth_amax=smooth_amax, smooth_alpha=args.smooth_alpha
                    )
                return options

            # Qualification plans contain probe copies; timing plans never do.
            qualified_outputs = {}
            heldout_outputs = {}
            for variant in args.variants:
                options = variant_options(variant)
                checks = []
                heldout_outputs[variant] = []
                for i, sample in enumerate(heldout):
                    if current is None or current.prompt_len != sample[1].shape[0]:
                        if current is not None:
                            current.close()
                        current = build(
                            options, probes=True, prompt_len=sample[1].shape[0]
                        )
                    output, probes = current.forward(*sample, capture_probes=True)
                    heldout_outputs[variant].append(output)
                    if i == 0:
                        qualified_outputs[variant] = output
                    ref, expected = references[i]
                    direct, _ = current.forward(*sample, execution="direct")
                    repeat, _ = current.forward(*sample)
                    check = {
                        "index": i + args.sample_offset,
                        "prompt_len": sample[1].shape[0],
                        "robot": metrics(output[:, :7], ref[:, :7]),
                        "full": metrics(output, ref),
                        "probes": {
                            k: metrics(v, expected[k]) for k, v in probes.items()
                        },
                        "direct_icb_equal": bool(np.array_equal(output, direct)),
                        "repeat_equal": bool(np.array_equal(output, repeat)),
                    }
                    check["passed"] = (
                        check["robot"]["finite"]
                        and check["robot"]["cosine"] >= 0.999
                        and check["robot"]["nrmse"] <= 0.03
                        and check["direct_icb_equal"]
                        and check["repeat_equal"]
                        and all(x["finite"] for x in check["probes"].values())
                    )
                    checks.append(check)
                    print(
                        "Parity",
                        variant,
                        check["index"],
                        check["robot"],
                        "passed",
                        check["passed"],
                        flush=True,
                    )
                report["parity"][variant] = checks
                report.setdefault("coverage", {})[variant] = {
                    "linears": current.plan.linears,
                    "counts": dict(
                        Counter(
                            f"{x['stage']}:{x['precision']}"
                            for x in current.plan.linears
                        )
                    ),
                    "smoothquant_calls": sum(
                        x.get("smoothquant", False) for x in current.plan.linears
                    ),
                    "fused_gelu_calls": sum(
                        x.get("fused_gelu", False) for x in current.plan.linears
                    ),
                    "weight_bytes": current.plan.weight_bytes,
                }
                current.close()
                current = None
                save()
            if "all-original" in heldout_outputs:
                report["optimization_equal"] = {
                    v: all(
                        np.array_equal(a, b)
                        for a, b in zip(
                            heldout_outputs["all-original"], heldout_outputs[v]
                        )
                    )
                    for v in ("all-tiled", "all")
                    if v in heldout_outputs
                }
                if not all(report["optimization_equal"].values()):
                    raise AssertionError("Optimized INT8 changed original INT8 actions")
            if not all(
                c["passed"] for checks in report["parity"].values() for c in checks
            ):
                raise AssertionError(
                    "INT8 qualification failed; refusing speed measurement"
                )
            if args.paired:
                import gc

                paired_plans = {}
                try:
                    for variant in args.paired:
                        paired_plans[variant] = build(
                            None if variant == "bf16" else variant_options(variant)
                        )
                        if paired_plans[variant].plan.copy_count:
                            raise AssertionError(
                                "Timed plan contains diagnostic copies"
                            )
                    source = None
                    f = None
                    gc.collect()
                    torch.mps.empty_cache()
                    sample = heldout[0]
                    records = {v: [] for v in args.paired}
                    for _ in range(args.warmup):
                        for plan in paired_plans.values():
                            plan.forward(*sample)
                    for i in range(args.iters * 2):
                        order = args.paired if i % 2 == 0 else args.paired[::-1]
                        for variant in order:
                            plan = paired_plans[variant]
                            start = time.perf_counter()
                            output, _ = plan.forward(*sample)
                            records[variant].append(
                                {
                                    "wall_ms": (time.perf_counter() - start) * 1000,
                                    "gpu_ms": plan.last_timing["gpu_us"] / 1000,
                                }
                            )
                            expected = (
                                references[0][0]
                                if variant == "bf16"
                                else qualified_outputs[variant]
                            )
                            if not np.array_equal(output, expected):
                                raise AssertionError(
                                    "Timed replay changed qualified output"
                                )
                    report["timing_method"] = (
                        "two resident plans; setup source released; alternating AB/BA pairs"
                    )
                    report["paired_plans"] = {
                        v: {
                            "dispatch_count": p.plan.dispatch_count,
                            "weight_bytes": p.plan.weight_bytes,
                        }
                        for v, p in paired_plans.items()
                    }
                    report["records"] = records
                    report["summary"] = {
                        v: {
                            metric: summary([r[metric] for r in rows])
                            for metric in ("wall_ms", "gpu_ms")
                        }
                        for v, rows in records.items()
                    }
                    baseline, candidate = args.paired
                    report["paired_speedup"] = summary(
                        [
                            a["wall_ms"] / b["wall_ms"]
                            for a, b in zip(records[baseline], records[candidate])
                        ]
                    )
                    report["status"] = "measured"
                    save()
                    print(
                        "Paired p50",
                        {k: v["wall_ms"]["p50"] for k, v in report["summary"].items()},
                        flush=True,
                    )
                finally:
                    for plan in paired_plans.values():
                        plan.close()
                return
            # Forward/reverse blocks bracket INT8 variants with warm BF16 runs.
            # One resident plan at a time avoids changing memory pressure by
            # keeping every full checkpoint representation alive concurrently.
            order = ["bf16", *args.variants, *reversed(args.variants), "bf16"]
            for variant in order:
                options = None if variant == "bf16" else variant_options(variant)
                current = build(options)
                if current.plan.copy_count:
                    raise AssertionError(
                        "Timed plan must not contain diagnostic copies"
                    )
                sample = heldout[0]
                for _ in range(args.warmup):
                    current.forward(*sample)
                records = []
                for i in range(args.iters):
                    start = time.perf_counter()
                    output, _ = current.forward(*sample)
                    records.append(
                        {
                            "wall_ms": (time.perf_counter() - start) * 1000,
                            "native": dict(current.last_timing),
                        }
                    )
                ref = references[0][0]
                parity = metrics(output[:, :7], ref[:, :7])
                if variant == "bf16" and not np.array_equal(output, ref):
                    raise AssertionError(
                        "Probe-free BF16 replay changed reference output"
                    )
                if variant != "bf16" and not np.array_equal(
                    output, qualified_outputs[variant]
                ):
                    raise AssertionError(
                        "Probe-free INT8 replay changed qualified output"
                    )
                block = {
                    "variant": variant,
                    "parity": parity,
                    "dispatches": current.plan.dispatch_count,
                    "copies": current.plan.copy_count,
                    "weight_bytes": current.plan.weight_bytes,
                    "records": records,
                    "wall_ms": summary([r["wall_ms"] for r in records]),
                    "gpu_ms": summary([r["native"]["gpu_us"] / 1000 for r in records]),
                }
                report["blocks"].append(block)
                print(
                    "Timing",
                    variant,
                    "wall p50",
                    block["wall_ms"]["p50"],
                    "GPU p50",
                    block["gpu_ms"]["p50"],
                    flush=True,
                )
                current.close()
                current = None
                save()
            report["summary"] = {}
            for variant in ["bf16", *args.variants]:
                records = [
                    r
                    for block in report["blocks"]
                    if block["variant"] == variant
                    for r in block["records"]
                ]
                report["summary"][variant] = {
                    "wall_ms": summary([r["wall_ms"] for r in records]),
                    "gpu_ms": summary([r["native"]["gpu_us"] / 1000 for r in records]),
                }
            baseline = report["summary"]["bf16"]["wall_ms"]["p50"]
            for variant in args.variants:
                report["summary"][variant]["speedup"] = (
                    baseline / report["summary"][variant]["wall_ms"]["p50"]
                )
                report["summary"][variant]["accuracy_passed"] = all(
                    c["passed"] for c in report["parity"][variant]
                )
            report["status"] = "measured"
            save()
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        save()
        raise
    finally:
        if current is not None:
            current.close()


if __name__ == "__main__":
    main()
