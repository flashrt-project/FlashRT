"""Warm, single-process, sequential Higgs streaming latency on RDNA.

Time starts before public generate_stream() and ends after iterator exhaustion
and device completion. It includes text preparation, prefill, decode, codec and
CPU audio delivery; model loading, warmup, statistics and JSON I/O are excluded.
TTFA is the first CPU packet arrival. RTF is wall time / delivered audio duration.
Packet slack assumes uninterrupted playback starting at the first packet.
The caller must establish an idle GPU and sufficient CPU/memory resources.
"""
import argparse
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def playback(packets, sample_rate=24000):
    """Each packet's deadline uses the actual lengths of all preceding packets."""
    elapsed_audio_ms = 0.0
    for packet in packets:
        packet["playback_slack_ms"] = packets[0]["arrival_ms"] + elapsed_audio_ms - packet["arrival_ms"]
        elapsed_audio_ms += packet["samples"] * 1000 / sample_rate
    later = [packet["playback_slack_ms"] for packet in packets[1:]]
    extra = max(0.0, -min(later, default=0.0))
    return {"min_slack_ms": min(later) if later else None,
            "late_packets": sum(value < 0 for value in later),
            "extra_buffer_ms": extra,
            "continuous_start_ms": packets[0]["arrival_ms"] + extra}


def measure(frontend, text, options):
    import torch

    torch.cuda.synchronize(frontend.device)
    packets, waves = [], []
    start = time.perf_counter()
    for audio in frontend.generate_stream(text, **options):
        arrived = (time.perf_counter() - start) * 1000
        packets.append({"arrival_ms": arrived, "samples": audio.numel()})
        waves.append(audio)
    torch.cuda.synchronize(frontend.device)
    wall_ms = (time.perf_counter() - start) * 1000
    if not waves or any(w.device.type != "cpu" or w.dtype != torch.float32 or w.ndim != 1
                        or w.numel() == 0 or not torch.isfinite(w).all() for w in waves):
        raise RuntimeError("generate_stream must return nonempty, finite CPU float32 audio packets")
    seconds = sum(packet["samples"] for packet in packets) / 24000
    return {"ttfa_ms": packets[0]["arrival_ms"], "wall_ms": wall_ms,
            "audio_seconds": seconds, "rtf": wall_ms / (seconds * 1000),
            "finish_reason": frontend.last_finish_reason, "packets": packets,
            **playback(packets)}


def measure_stages(frontend, text, iterations):
    """Full-prompt host prefill and a fixed first-audio-position decode.

    These isolated stages exclude the codec and are not end-to-end latency.
    Reuse a genuine first delayed-code embedding, reloading the graph input on
    every iteration; otherwise the in-place residual buffer changes the input.
    """
    import torch

    frontend.set_prompt(text)
    frontend.prefill()
    ids = list(frontend._prompt_ids)
    frontend._sample_logits(frontend._gen_logits, 0, 0.0, 0, 0)
    embedding = frontend._embed_buf.clone()
    frontend._decode_logits(embedding, len(ids))
    torch.cuda.synchronize(frontend.device)
    prefill_ms, decode_ms = [], []
    for _ in range(iterations):
        start = time.perf_counter()
        frontend.set_prompt(text)
        frontend.prefill()
        torch.cuda.synchronize(frontend.device)
        prefill_ms.append((time.perf_counter() - start) * 1000)
    for _ in range(iterations):
        start = time.perf_counter()
        frontend._decode_logits(embedding, len(ids))
        torch.cuda.synchronize(frontend.device)
        decode_ms.append((time.perf_counter() - start) * 1000)
    def stats(values):
        return {"p50": percentile(values, .5), "p95": percentile(values, .95), "max": max(values)}
    return {
        "prompt_ids": ids, "prompt_tokens": len(ids), "decode_position": len(ids),
        "iterations": iterations, "prefill_ms": stats(prefill_ms),
        "prefill_ms_per_token": stats([value / len(ids) for value in prefill_ms]),
        "decode_ms_per_frame": stats(decode_ms),
        "prefill_samples_ms": prefill_ms, "decode_samples_ms": decode_ms,
        "prefill_boundary": "set_prompt + full prefill + device completion; no prefix reuse",
        "decode_boundary": "fixed first-audio embedding + position update + graph replay + device completion; no sampling/codec",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp8"), default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-iters", type=int, default=0,
                        help="Also measure isolated prefill/decode stages (at least 20 iterations)")
    # Unspecified options are omitted from the call, preserving API defaults.
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--first-chunk", type=int)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--holdback", type=int)
    args = parser.parse_args()
    if args.warmup < 1 or args.iters < 1:
        parser.error("warmup and iters must be positive")
    if args.stage_iters and args.stage_iters < 20:
        parser.error("stage-iters must be zero or at least 20")
    import torch
    from flash_rt.models.higgs_audio_v3 import create_frontend

    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("requires ROCm PyTorch and a visible RDNA GPU")
    frontend = create_frontend(args.checkpoint, fp8=None if args.precision == "auto" else args.precision == "fp8")
    if not frontend._use_graph or not frontend._batched_prefill:
        raise ValueError("warm latency requires production graph decode and batched prefill")
    names = ("temperature", "seed", "first_chunk", "chunk", "ctx", "holdback")
    options = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    defaults = inspect.signature(frontend.generate_stream).parameters
    report = {
        "status": "running", "configuration": {**vars(args), "output": str(args.output)},
        "stream_options": {name: options.get(name, defaults[name].default) for name in names},
        "precision_spec": frontend.precision_spec,
        "effective_precision": "fp8" if frontend.fp8 else "bf16",
        "boundary": "text to CPU audio; iterator exhaustion and device completion; no HTTP/player/backpressure",
        "warmup": "same text and stream options; warmup requests excluded; one process and sequential requests",
        "sample_rate": 24000,
        "environment": {
            "device": str(torch.cuda.get_device_properties(frontend.device)),
            "python": platform.python_version(), "torch": torch.__version__,
            "hip_build": torch.version.hip,
            "packages": {name: importlib.metadata.version(name) for name in ("triton", "transformers")},
            "overrides": {name: value for name, value in os.environ.items() if name.startswith("FLASHRT_HIGGS_")},
        },
        "warmup_finish_reasons": [], "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for _ in range(args.warmup):
            sample = measure(frontend, args.text, options)
            report["warmup_finish_reasons"].append(sample["finish_reason"])
            if sample["finish_reason"] != "eoc":
                raise RuntimeError(f"incomplete warmup request: {sample['finish_reason']}")
        assert frontend._graph is not None
        torch.cuda.reset_peak_memory_stats(frontend.device)
        for _ in range(args.iters):
            report["samples"].append(measure(frontend, args.text, options))
        samples = report["samples"]
        margins = [sample["min_slack_ms"] for sample in samples if sample["min_slack_ms"] is not None]
        report["summary"] = {
            name: {"p50": percentile([s[name] for s in samples], .5),
                   "p95": percentile([s[name] for s in samples], .95),
                   "max": max(s[name] for s in samples)}
            for name in ("ttfa_ms", "wall_ms", "rtf")
        }
        report["summary"].update(
            min_slack_ms=min(margins) if margins else None,
            late_packets=sum(s["late_packets"] for s in samples),
            incomplete_requests=sum(s["finish_reason"] != "eoc" for s in samples))
        report["memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(frontend.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(frontend.device),
            "boundary": "warm streaming requests; excludes initial loading/packing peak",
        }
        if args.stage_iters:
            report["stages"] = measure_stages(frontend, args.text, args.stage_iters)
        report["status"] = "failed" if report["summary"]["incomplete_requests"] else "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
