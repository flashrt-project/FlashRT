"""Qualify fixed-shape full native replay against the shared MPS traversal.

Prepared inputs; excludes tokenizer, plan construction and output normalization.
Trace separately with --trace-only --no-parity. No per-operation MPS profiler.
"""

from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import numpy as np


def metrics(actual, expected):
    a, b = (
        np.asarray(actual, np.float64).ravel(),
        np.asarray(expected, np.float64).ravel(),
    )
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(finite=False, max_abs=None, nrmse=None, cosine=None)
    delta = a - b
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dict(
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
        max_abs=float(np.max(np.abs(delta))),
        nrmse=float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-12)),
        cosine=float(a @ b / norm) if norm else float(np.array_equal(a, b)),
    )


def summary(values):
    return dict(
        p50=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        samples=list(values),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parents[1] / "thor/assets/pi05_libero.npz",
    )
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument(
        "--prompt-length",
        type=int,
        help="Fixed profile using the first N real prompt embeddings",
    )
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=("torch-mps", "full-icb", "full-icb-mps"),
        default=["torch-mps", "full-icb"],
    )
    ap.add_argument("--schedule", choices=("paired", "serial"), default="paired")
    ap.add_argument(
        "--no-residency",
        action="store_true",
        help="Diagnostic: disable persistent plan residency",
    )
    ap.add_argument(
        "--profile-kernels",
        action="store_true",
        help="After timing, run isolated dispatch diagnostics; not model latency",
    )
    ap.add_argument(
        "--no-split-gqa",
        action="store_true",
        help="Use the unsplit GQA kernel for A/B timing",
    )
    ap.add_argument(
        "--compare-gqa",
        action="store_true",
        help="Warm ABBA comparison of unsplit/split GQA in full ICB replay",
    )
    ap.add_argument(
        "--split-encoder-ffn",
        action="store_true",
        help="Diagnostic: two encoder gate/up GEMMs instead of one merged GEMM",
    )
    ap.add_argument(
        "--compare-encoder-ffn",
        action="store_true",
        help="Warm ABBA comparison of split/merged encoder gate/up in full ICB replay",
    )
    args = ap.parse_args()
    if min(args.steps, args.samples, args.warmup, args.iters) < 1:
        ap.error("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="started",
        started_at=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),
        schedule=args.schedule,
        native_residency=not args.no_residency,
        split_decode=not args.no_split_gqa,
        merged_encoder_ffn=not args.split_encoder_ffn,
        steps=args.steps,
        trace_only=args.trace_only,
        parity=[],
        timing_scope="Prepared inputs. full-icb: CPU inputs -> shared native buffers -> patch/blocks/prefix/decode -> CPU action; other modes: MPS tensor inputs/outputs",
        gates=dict(nrmse=0.03, cosine=0.995),
    )

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    os.environ["FLASH_RT_PALIGEMMA_TOKENIZER"] = str(args.tokenizer.resolve())
    import torch
    from flash_rt.frontends.torch.pi05_mps import Pi05TorchFrontendMps

    if not torch.backends.mps.is_available():
        raise RuntimeError("Host MPS required")
    if not args.trace_only and any(
        os.environ.get(k, "0") not in ("", "0")
        for k in (
            "MTL_DEBUG_LAYER",
            "MTL_SHADER_VALIDATION",
            "PYTORCH_MPS_TRACE_SIGNPOSTS",
            "MTL_CAPTURE_ENABLED",
        )
    ):
        raise RuntimeError("Disable validation/profiling for timing")
    f = Pi05TorchFrontendMps(args.checkpoint, num_steps=args.steps, action_dim=7)
    p = f.pipeline
    print("Checkpoint loaded", flush=True)
    noise = torch.from_numpy(
        np.random.default_rng(20261027)
        .standard_normal((f.chunk_size, 32))
        .astype(np.float32)
    ).to("mps", dtype=torch.bfloat16)
    with np.load(args.input, allow_pickle=False) as data:
        observations = [
            dict(
                images=np.stack((data[f"img_{i}"], data[f"wrist_{i}"])),
                state=data[f"state_{i}"],
            )
            for i in range(min(args.samples, int(data["n"])))
        ]

    host_images = None
    active_mode = None
    host_noise = noise.cpu()

    def prepare(obs):
        nonlocal host_images
        stats = f.norm_stats["state"]
        state = obs["state"]
        lo, hi = (np.asarray(stats[k])[: state.size] for k in ("q01", "q99"))
        f.set_prompt(
            "pick up the black bowl and place it on the plate",
            state=2 * (state - lo) / (hi - lo + 1e-6) - 1,
        )
        if args.prompt_length:
            if not 0 < args.prompt_length <= f._prompt_len:
                raise ValueError("Requested prompt length exceeds fixture")
            f._prompt_len = args.prompt_length
        host_images = torch.from_numpy(
            obs["images"].astype(np.float32) / 127.5 - 1
        ).bfloat16()
        return host_images.to("mps")

    def cpu(x):
        return x.float().cpu().numpy().copy()

    def run(images, probes=False):
        if p._native_pipeline is not None and active_mode != "full-icb-mps":
            return torch.from_numpy(
                p.forward_native(
                    host_images,
                    f._prompt_cpu[: f._prompt_len],
                    host_noise,
                    capture_probes=probes,
                )
            )
        return f.forward_with_fixed_noise(images, noise, capture_probes=probes)[0]

    try:
        with torch.inference_mode():
            images = prepare(observations[0])
            report.update(
                prompt_len=f._prompt_len,
                prefix_m=512 + f._prompt_len,
                decode_m=f.chunk_size,
            )
            if not args.no_parity:
                refs = []
                for obs in observations:
                    images = prepare(obs)
                    ref = cpu(run(images, True))
                    probes = {k: cpu(v) for k, v in p.probes.items()}
                    refs.append((ref, probes))
                report["diagnostic_build_seconds"] = 0.0
                for index, obs in enumerate(observations):
                    images = prepare(obs)
                    if (
                        p._native_pipeline is None
                        or p._native_pipeline.prompt_len != f._prompt_len
                    ):
                        p.disable_native_pipeline()
                        start = time.perf_counter()
                        p.enable_native_pipeline(
                            f._prompt_len,
                            capture_probes=True,
                            residency=not args.no_residency,
                            split_decode=not args.no_split_gqa,
                            merged_encoder_ffn=not args.split_encoder_ffn,
                        )
                        report["diagnostic_build_seconds"] += (
                            time.perf_counter() - start
                        )
                        save()
                        print(
                            "Diagnostic plan built",
                            p._native_pipeline.plan.dispatch_count,
                            "M",
                            512 + f._prompt_len,
                            flush=True,
                        )
                    actual = cpu(run(images, True))
                    ref, expected = refs[index]
                    if set(expected) - set(p.probes):
                        raise AssertionError(
                            f"Missing native probes: {set(expected) - set(p.probes)}"
                        )
                    checks = {
                        k: metrics(cpu(p.probes[k]), v)
                        for k, v in expected.items()
                        if k in p.probes
                    }
                    robot = metrics(actual[:, :7], ref[:, :7])
                    repeat = cpu(run(images))
                    result = dict(
                        index=index,
                        prefix_m=512 + f._prompt_len,
                        snapshot_copy_count=p._native_pipeline.plan.copy_count,
                        diagnostic_dispatch_count=p._native_pipeline.plan.dispatch_count,
                        robot=robot,
                        probes=checks,
                        repeat_equal=bool(np.array_equal(actual, repeat)),
                    )
                    # Same persistent plan and inputs must agree in direct and ICB execution.
                    direct, _ = p._native_pipeline.forward(
                        host_images,
                        f._prompt_cpu[: f._prompt_len],
                        host_noise,
                        execution="direct",
                    )
                    result["direct_icb_equal"] = bool(np.array_equal(actual, direct))
                    compatibility = cpu(f.forward_with_fixed_noise(images, noise)[0])
                    result["mps_compatibility_equal"] = bool(
                        np.array_equal(actual, compatibility)
                    )
                    frontend = f.infer(obs, noise=host_noise, debug=True)
                    result["frontend_equal"] = bool(
                        np.array_equal(actual, frontend["raw_actions"])
                    )
                    report["parity"].append(result)
                    save()
                    print("Parity", index, robot, flush=True)
                    if not (
                        robot["finite"]
                        and robot["nrmse"] <= 0.03
                        and robot["cosine"] >= 0.995
                        and result["repeat_equal"]
                        and result["direct_icb_equal"]
                        and result["mps_compatibility_equal"]
                        and result["frontend_equal"]
                        and all(
                            v["finite"] and v["nrmse"] <= 0.03 for v in checks.values()
                        )
                    ):
                        raise AssertionError("Full native checkpoint parity failed")
                p.disable_native_pipeline()
                p.probes.clear()
            images = prepare(observations[0])
            start = time.perf_counter()
            p.enable_native_pipeline(
                f._prompt_len,
                residency=not args.no_residency,
                split_decode=not args.no_split_gqa,
                merged_encoder_ffn=not args.split_encoder_ffn,
            )
            native = p._native_pipeline
            if native.plan.copy_count:
                raise AssertionError("Production native plan contains buffer copies")
            report.update(
                build_seconds=time.perf_counter() - start,
                dispatch_count=native.plan.dispatch_count,
                native_weight_bytes=native.plan.weight_bytes,
                copy_count=native.plan.copy_count,
            )

            @contextmanager
            def select(mode):
                nonlocal active_mode
                active_mode = mode
                p._native_pipeline = native if mode.startswith("full-icb") else None
                try:
                    yield
                finally:
                    p._native_pipeline = native

            modes = args.modes
            times = {m: [] for m in modes}
            native_times = []
            report["records"] = []
            for mode in modes:
                with select(mode):
                    for _ in range(args.warmup):
                        run(images)
                    torch.mps.synchronize()
            order = (
                [(i, m) for m in modes for i in range(args.iters)]
                if args.schedule == "serial"
                else [
                    (i, m)
                    for i in range(args.iters)
                    for m in (modes if i % 2 == 0 else list(reversed(modes)))
                ]
            )
            profile = (
                p.profile(signposts=True, mps_ops=False)
                if args.trace_only
                else nullcontext()
            )
            with profile, (args.output / "samples.jsonl").open("w") as log:
                for i, mode in order:
                    with select(mode):
                        torch.mps.synchronize()
                        utc = datetime.now(timezone.utc).isoformat()
                        with p._range("inference." + mode):
                            start = time.perf_counter()
                            run(images)
                            torch.mps.synchronize()
                            elapsed = (time.perf_counter() - start) * 1000
                        times[mode].append(elapsed)
                        record = dict(
                            iteration=i + 1, mode=mode, started_at=utc, wall_ms=elapsed
                        )
                        if mode.startswith("full-icb"):
                            record["native"] = dict(native.last_timing)
                            native_times.append(record["native"])
                        report["records"].append(record)
                        log.write(json.dumps(record) + "\n")
                        log.flush()
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            report["timings_ms"] = {m: summary(v) for m, v in times.items()}
            report["native_us"] = {
                k: summary([t[k] for t in native_times])
                for k in (native_times[0] if native_times else [])
                if k.endswith("_us")
            }
            if args.profile_kernels:
                if args.trace_only:
                    raise ValueError(
                        "Collect kernel diagnostics separately from Instruments"
                    )
                with select("full-icb"):
                    expected = cpu(run(images))
                    runs = []
                    for _ in range(3):
                        actual, timings = native.profile(
                            host_images, f._prompt_cpu[:f._prompt_len], host_noise
                        )
                        np.testing.assert_array_equal(actual, expected)
                        runs.append(timings)
                    operations = []
                    groups = {}
                    for index, meta in enumerate(native.plan.dispatches):
                        us = float(np.median([r[index]["gpu_us"] for r in runs]))
                        operations.append(dict(index=index, **meta, gpu_us=us))
                        key = (meta["stage"], meta["kernel"], tuple(meta["params"]))
                        item = groups.setdefault(
                            key, dict(**meta, count=0, total_gpu_us=0)
                        )
                        item["count"] += 1
                        item["total_gpu_us"] += us
                    report["kernel_profile"] = dict(
                        scope="Isolated synchronous dispatches in model order; includes submission GPU overhead. Do not sum as model latency.",
                        output_equal=True,
                        input_reset_each_run=True,
                        operations=operations,
                        groups=sorted(
                            groups.values(), key=lambda x: -x["total_gpu_us"]
                        ),
                    )
            comparisons = []
            if args.compare_gqa:
                comparisons.append(("gqa", "split", "split_decode"))
            if args.compare_encoder_ffn:
                comparisons.append(("encoder_ffn", "merged", "merged_encoder_ffn"))
            for comparison, field, option in comparisons:
                if args.trace_only:
                    raise ValueError("Compare kernel variants without Instruments")
                blocks, reference = [], None
                for enabled in (False, True, True, False):
                    p.disable_native_pipeline()
                    options = dict(
                        split_decode=not args.no_split_gqa,
                        merged_encoder_ffn=not args.split_encoder_ffn,
                    )
                    options[option] = enabled
                    p.enable_native_pipeline(
                        f._prompt_len,
                        residency=not args.no_residency,
                        **options,
                    )
                    native = p._native_pipeline
                    measured = []
                    with select("full-icb"):
                        for _ in range(max(10, args.warmup)):
                            actual = cpu(run(images))
                        if reference is None:
                            reference = actual
                        parity = metrics(actual, reference)
                        if not (parity["finite"] and parity["nrmse"] <= 0.03
                                and parity["cosine"] >= 0.995):
                            raise AssertionError(f"{comparison} A/B full action parity failed")
                        for index in range(args.iters):
                            utc = datetime.now(timezone.utc).isoformat()
                            start = time.perf_counter()
                            run(images)
                            measured.append(
                                dict(
                                    iteration=index + 1,
                                    started_at=utc,
                                    wall_ms=(time.perf_counter() - start) * 1000,
                                    native=dict(native.last_timing),
                                )
                            )
                    block = dict(
                        parity=parity,
                        dispatch_count=native.plan.dispatch_count,
                        native_weight_bytes=native.plan.weight_bytes,
                        records=measured,
                        timing_ms=summary([x["wall_ms"] for x in measured]),
                    )
                    block[field] = enabled
                    blocks.append(block)
                    report[f"{comparison}_abba"] = blocks
                    save()
                    print(comparison, "A/B block", enabled, block["timing_ms"]["p50"], flush=True)
            report["status"] = (
                "trace_emitted"
                if args.trace_only
                else "passed"
                if not args.no_parity
                else "timed_without_parity"
            )
            save()
            print(json.dumps(report["timings_ms"], indent=2), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        save()
        raise
    finally:
        p.disable_native_pipeline()


if __name__ == "__main__":
    main()
