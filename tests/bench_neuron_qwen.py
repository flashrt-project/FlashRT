"""Qwen Trainium correctness gate followed by batch-one latency measurement."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compiled", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--max-prefill-tokens", type=int)
    p.add_argument("--layer-kernels", action="store_true")
    p.add_argument("--speculative", action="store_true")
    p.add_argument("--megakernel", action="store_true")
    args = p.parse_args()
    from flash_rt.neuron.qwen36 import Qwen36NeuronFrontend

    started = time.monotonic()
    model = Qwen36NeuronFrontend(
        args.checkpoint,
        args.compiled,
        tp_degree=args.tp,
        seq_len=args.seq_len,
        layer_kernels=args.layer_kernels,
        speculative=args.speculative,
        megakernel=args.megakernel,
        max_prefill_tokens=args.max_prefill_tokens,
    )
    load_seconds = time.monotonic() - started
    reference = json.loads(args.reference.read_text())
    comparisons = []
    # Reversing order also checks that recurrent/KV state is reset per request.
    for row in reference["rows"] + list(reversed(reference["rows"])):
        result = model.generate(
            row["prompt"], max_new_tokens=reference["max_new_tokens"]
        )
        comparisons.append(
            {
                "prompt": row["prompt"],
                "token_ids": result["token_ids"],
                "exact_match": result["token_ids"] == row["token_ids"],
            }
        )
    passed = all(row["exact_match"] for row in comparisons)
    result = {
        "model": "Qwen3.6-35B-A3B",
        "identity": model.identity,
        "load_compile_seconds": load_seconds,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "correctness": {"passed": passed, "comparisons": comparisons},
        "status": "qualified" if passed else "numerical_failure",
    }
    if passed:
        result["timing"] = model.benchmark(
            reference["rows"][0]["prompt"], max_new_tokens=64, num_runs=args.runs
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError("Neuron output differs from independent reference")


if __name__ == "__main__":
    main()
