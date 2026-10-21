#!/usr/bin/env python3
"""Qwen3-4B latency: FlashRT vs HuggingFace (vision / llm / e2e).

Qwen3-4B is text-only — ``vision`` is always N/A (reported as 0).

Definitions
-----------
vision : N/A (no ViT). Always 0.0 ms.
llm    : embed + decoder layers + final RMSNorm (no lm_head).
e2e    : llm + lm_head  (= full CausalLM logits).

Uses random weights (same state_dict on both sides). CUDA event timing.
"""
from __future__ import annotations

import argparse
import statistics
from typing import Any

import torch


def _stats(xs: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
        "p50": statistics.median(xs),
    }


def _cuda_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _make_models(device: str, seq: int, *, all_fvk: bool = False):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from flash_rt.frontends.torch._qwen3_4b_rtx_bf16_weights import (
        qwen3_4b_config_dict,
    )
    from flash_rt.frontends.torch.qwen3_4b_rtx import Qwen3_4BTorchFrontendRtx

    cfg = Qwen3Config(**{
        k: v for k, v in qwen3_4b_config_dict().items()
        if k not in ("architectures",)
    })
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    hf = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    sd = {k: v.detach().contiguous() for k, v in hf.state_dict().items()}
    frt_kw = dict(
        checkpoint_path=None,
        device=device,
        max_seq=max(seq * 2, 64),
        max_q_seq=max(seq, 32),
        state_dict=sd,
        config=cfg.to_dict(),
    )
    if all_fvk:
        frt_kw.update(
            use_hf_norm=False,
            use_hf_qkn_rope=False,
            use_sep_gateup=False,
            use_sep_qkv=False,
            use_torch_qkv=False,
            # use_sdpa left as None → auto SDPA on Thor when FA2 missing
        )
    frt = Qwen3_4BTorchFrontendRtx(**frt_kw)
    return hf, frt, cfg


@torch.no_grad()
def _bench_hf(hf, input_ids: torch.Tensor, warmup: int, runs: int) -> dict[str, Any]:
    # Warmup
    for _ in range(warmup):
        _ = hf(input_ids=input_ids).logits
        _ = hf.model(input_ids=input_ids)
    torch.cuda.synchronize()

    e2e, llm = [], []
    for _ in range(runs):
        e2e.append(_cuda_ms(lambda: hf(input_ids=input_ids)))
        llm.append(_cuda_ms(lambda: hf.model(input_ids=input_ids)))

    return {
        "vision_ms": _stats([0.0] * runs),  # N/A
        "llm_ms": _stats(llm),
        "e2e_ms": _stats(e2e),
        "vision_na": True,
    }


@torch.no_grad()
def _bench_flashrt(frt, input_ids: torch.Tensor, warmup: int, runs: int) -> dict[str, Any]:
    ids = input_ids[0]

    for _ in range(warmup):
        frt.forward_prefill(ids)
        frt.forward_prefill_llm_only(ids)
    torch.cuda.synchronize()

    e2e, llm = [], []
    for _ in range(runs):
        e2e.append(_cuda_ms(lambda: frt.forward_prefill(ids)))
        llm.append(_cuda_ms(lambda: frt.forward_prefill_llm_only(ids)))

    return {
        "vision_ms": _stats([0.0] * runs),  # N/A
        "llm_ms": _stats(llm),
        "e2e_ms": _stats(e2e),
        "vision_na": True,
    }


def _fmt(s: dict[str, float]) -> str:
    return f"{s['mean']:8.3f} ± {s['std']:.3f}  (min {s['min']:.3f} / p50 {s['p50']:.3f} / max {s['max']:.3f})"


def _print_table(name: str, r: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    vis = "N/A (text-only Qwen3-4B)" if r.get("vision_na") else _fmt(r["vision_ms"])
    print(f"  vision : {vis}")
    print(f"  llm    : {_fmt(r['llm_ms'])} ms")
    print(f"  e2e    : {_fmt(r['e2e_ms'])} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=128, help="prefill sequence length")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--all-fvk", action="store_true",
        help="FlashRT all-fvk path (disable HF-parity precision ops)")
    ap.add_argument(
        "--compare-paths", action="store_true",
        help="Bench HF + FlashRT default + FlashRT all-fvk")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"seq       : {args.seq}")
    print(f"warmup/runs: {args.warmup}/{args.runs}")
    print("weights   : random (shared HF state_dict)")
    print("modules   : vision=N/A | llm=embed+layers+final_norm | e2e=llm+lm_head")

    paths = [("default", False)]
    if args.compare_paths:
        paths = [("default", False), ("all_fvk", True)]
    elif args.all_fvk:
        paths = [("all_fvk", True)]

    # Shared HF model + input once.
    hf, frt0, cfg = _make_models(args.device, args.seq, all_fvk=paths[0][1])
    torch.manual_seed(123)
    input_ids = torch.randint(
        0, cfg.vocab_size, (1, args.seq), device=args.device, dtype=torch.long)

    hf_r = _bench_hf(hf, input_ids, args.warmup, args.runs)
    _print_table("HuggingFace Qwen3ForCausalLM (BF16)", hf_r)

    results = {}
    for name, all_fvk in paths:
        if name == paths[0][0]:
            frt = frt0
        else:
            _, frt, _ = _make_models(args.device, args.seq, all_fvk=all_fvk)
        frt_r = _bench_flashrt(frt, input_ids, args.warmup, args.runs)
        label = (
            "FlashRT all-fvk"
            if all_fvk else
            "FlashRT default (HF-parity precision ops)"
        )
        _print_table(f"{label}", frt_r)
        results[name] = frt_r

    print("\n=== Speedup (HF_ms / FlashRT_ms) ===")
    for name, frt_r in results.items():
        print(f"  -- {name} --")
        for key, label in (("llm_ms", "llm"), ("e2e_ms", "e2e")):
            hf_m = hf_r[key]["mean"]
            fr_m = frt_r[key]["mean"]
            sp = hf_m / fr_m if fr_m > 0 else float("inf")
            print(f"  {label:6s}: {sp:6.2f}x   ({hf_m:.3f} ms → {fr_m:.3f} ms)")
    print("  vision: N/A")


if __name__ == "__main__":
    main()
