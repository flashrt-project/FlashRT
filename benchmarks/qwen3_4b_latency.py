#!/usr/bin/env python3
"""Qwen3-4B BF16 prefill latency: FlashRT vs HuggingFace eager.

Both sides load the same checkpoint and run batch-1 prefill to full logits
(embed + 36 decoder layers + final RMSNorm + lm_head over every position).
Timing is CUDA events around one call, after warm-up; the table reports the
median, with min / max alongside. FlashRT runs the frontend of the device's
target (CUDA graphs on unless --no-graphs).

    python benchmarks/qwen3_4b_latency.py --checkpoint <Qwen3-4B dir> \\
        --seq 8,128,512,1024,2048
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def _time(fn, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        xs.append(start.elapsed_time(end))
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="HF Qwen/Qwen3-4B directory")
    ap.add_argument("--seq", default="8,128,512,1024,2048", help="comma-separated prompt lengths")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--no-hf", action="store_true", help="skip the HuggingFace baseline")
    ap.add_argument("--no-graphs", action="store_true", help="FlashRT without CUDA graphs")
    ap.add_argument("--autotune-gemms", action="store_true",
                    help="FlashRT with autotune_gemms=True (probe-checked GEMM selection)")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to idle between prompt lengths")
    ap.add_argument("--json", default=None, help="also write the results here")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    import transformers
    from flash_rt.hardware import detect_arch, resolve_pipeline_class

    arch = detect_arch()
    frontend = resolve_pipeline_class("qwen3_4b", "torch", arch)
    seqs = [int(s) for s in args.seq.split(",")]
    device = torch.device("cuda", torch.cuda.current_device())
    fe = frontend(args.checkpoint, device=device, max_seq=max(seqs),
                  cuda_graphs=not args.no_graphs, autotune_gemms=args.autotune_gemms)
    hf = None
    if not args.no_hf:
        from transformers import Qwen3ForCausalLM
        try:
            hf = Qwen3ForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16)
        except TypeError:   # transformers < 4.56
            hf = Qwen3ForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
        hf = hf.to(device).eval()

    print(f"GPU          : {torch.cuda.get_device_name(device)} ({arch})")
    print(f"frontend     : {frontend.__name__} (attention {frontend.TARGET.attention}, "
          f"CUDA graphs {'off' if args.no_graphs else 'on'}, "
          f"GEMMs {'autotuned' if args.autotune_gemms else 'strict'})")
    print(f"torch        : {torch.__version__}  transformers {transformers.__version__}")
    print(f"warmup/runs  : {args.warmup}/{args.runs}, median [min, max] ms")
    gen = torch.Generator(device=device).manual_seed(0)
    rows = []
    for S in seqs:
        ids = torch.randint(0, fe.dims.vocab_size, (S,), device=device, generator=gen)
        row = {"seq": S, "flashrt": _time(lambda: fe.forward_prefill(ids), args.warmup, args.runs)}
        if hf is not None:
            with torch.no_grad():
                row["hf"] = _time(lambda: hf(input_ids=ids[None]), args.warmup, args.runs)
        rows.append(row)
        f = row["flashrt"]
        line = f"S={S:5d}  FlashRT {f['median']:8.2f} [{f['min']:.2f}, {f['max']:.2f}]"
        if "hf" in row:
            h = row["hf"]
            line += (f"  HF {h['median']:8.2f} [{h['min']:.2f}, {h['max']:.2f}]"
                     f"  speedup {h['median'] / f['median']:.2f}x")
        print(line, flush=True)
        fe.clear_graphs()
        if args.pause:
            time.sleep(args.pause)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(device), "arch": arch,
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "frontend": frontend.__name__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
