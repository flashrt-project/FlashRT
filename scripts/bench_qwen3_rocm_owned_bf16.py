#!/usr/bin/env python3
"""Benchmark the Qwen3 ROCm owned BF16 frontend."""

from __future__ import annotations
import os

import argparse
import statistics
import time

import torch

from flash_rt.frontends.torch.qwen3_rocm_owned import Qwen3RocmOwnedBF16Frontend


def _ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _stats(vals: list[float]) -> str:
    return (
        f"mean={statistics.mean(vals):.2f} ms "
        f"p50={statistics.median(vals):.2f} ms "
        f"min={min(vals):.2f} ms max={max(vals):.2f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--prompt", default="FlashRT ROCm Qwen3 owned BF16 benchmark")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--use-fp8-layers", action="store_true")
    parser.add_argument("--use-fp8-lm-head", action="store_true")
    parser.add_argument("--attn-backend", default="flash_attn")
    args = parser.parse_args()

    frontend = Qwen3RocmOwnedBF16Frontend(
        args.checkpoint,
        max_seq=max(2048, args.prompt_len + args.decode_steps + 8),
        max_q_seq=max(32, args.prompt_len),
        preferred_attn_backend=args.attn_backend,
        use_fp8_layers=bool(args.use_fp8_layers),
        use_fp8_lm_head=bool(args.use_fp8_lm_head),
    )
    enc = frontend.tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    if enc.numel() < args.prompt_len:
        repeats = (args.prompt_len + enc.numel() - 1) // enc.numel()
        enc = enc.repeat(repeats)
    prompt_ids = enc[: args.prompt_len].to("cuda")

    scale_summary = None
    lm_scale = None
    if args.use_fp8_layers:
        scale_summary = frontend.calibrate_fp8_layers(
            prompt_ids, pos_start=0
        )
    if args.use_fp8_lm_head:
        lm_scale = frontend.calibrate_fp8_lm_head(prompt_ids, pos_start=0)

    def run_once(measure_decode: bool) -> tuple[float, list[float]]:
        decode_ms: list[float] = []
        prefill_ms = _ms(
            lambda: frontend.forward_ids(
                prompt_ids, pos_start=0, causal=prompt_ids.numel() > 1
            )
        )
        logits = frontend.logits[: prompt_ids.numel()]
        token = int(logits[-1].float().argmax().item())
        for step in range(args.decode_steps):
            pos = int(prompt_ids.numel()) + step
            token_tensor = torch.tensor([token], device="cuda", dtype=prompt_ids.dtype)
            elapsed = _ms(
                lambda tt=token_tensor, p=pos: frontend.forward_ids(
                    tt, pos_start=p, causal=False
                )
            )
            logits = frontend.logits[:1]
            token = int(logits[-1].float().argmax().item())
            if measure_decode:
                decode_ms.append(elapsed)
        return prefill_ms, decode_ms

    for _ in range(args.warmup):
        run_once(False)

    prefill_vals: list[float] = []
    decode_vals: list[float] = []
    for _ in range(args.repeat):
        prefill_ms, decode_ms = run_once(True)
        prefill_vals.append(prefill_ms)
        decode_vals.extend(decode_ms)

    steady = decode_vals[1:] if len(decode_vals) > 1 else decode_vals
    print("CONFIG", {
        "prompt_len": int(prompt_ids.numel()),
        "decode_steps": args.decode_steps,
        "repeat": args.repeat,
        "attn_backend": frontend.config_summary["attn_backend"],
        "use_fp8_layers": bool(args.use_fp8_layers),
        "use_fp8_lm_head": bool(args.use_fp8_lm_head),
        "fp8_layer_scales": scale_summary,
        "fp8_lm_scale": lm_scale,
    })
    print("PREFILL", _stats(prefill_vals))
    print("DECODE_RAW", _stats(decode_vals))
    if steady:
        tok_s = 1000.0 / statistics.mean(steady)
        print("DECODE_STEADY", _stats(steady), f"tok_s={tok_s:.2f}")
    print("OK qwen3_rocm_owned_bf16_bench")


if __name__ == "__main__":
    main()
