#!/usr/bin/env python3
"""Qwen3-4B adaptation sweep on Thor (random weights).

  S1  torch golden     — HF vs FlashRT full-seq logits cosine
  S2  candidate flags  — leave-one-out / all-fvk timing + cosine gates
  S3  assemble verdict — export kernel_flags JSON (accepted hot path)

No checkpoint required: shared random ``Qwen3ForCausalLM.from_config``
state_dict (same tensor mapping on both sides — never seed-faked twins).

Thor note: FA2 is unavailable on sm_110; attention uses torch SDPA.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


COSINE_FLOOR = 0.999
LAYER_SPEEDUP_FLOOR = 1.02  # require a measurable E2E gain to accept


def _cuda_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _make_hf(device: str):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from flash_rt.frontends.torch._qwen3_4b_rtx_bf16_weights import (
        qwen3_4b_config_dict,
    )

    cfg = Qwen3Config(**{
        k: v for k, v in qwen3_4b_config_dict().items()
        if k not in ("architectures",)
    })
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    hf = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    sd = {k: v.detach().contiguous() for k, v in hf.state_dict().items()}
    return hf, cfg, sd


def _make_frt(device: str, seq: int, sd, cfg, **flags):
    from flash_rt.frontends.torch.qwen3_4b_rtx import Qwen3_4BTorchFrontendRtx

    return Qwen3_4BTorchFrontendRtx(
        checkpoint_path=None,
        device=device,
        max_seq=max(seq * 2, 64),
        max_q_seq=max(seq, 32),
        state_dict=sd,
        config=cfg.to_dict(),
        **flags,
    )


def _cosine(hf, frt, input_ids: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        hf_out = hf(input_ids=input_ids).logits[0].float()
        frt_out = frt.forward_prefill(input_ids[0]).float()
    cos = F.cosine_similarity(frt_out.flatten(), hf_out.flatten(), dim=0).item()
    cos_last = F.cosine_similarity(frt_out[-1], hf_out[-1], dim=0).item()
    return {"cosine_full": cos, "cosine_last": cos_last}


def _bench_e2e(frt, input_ids: torch.Tensor, warmup: int, runs: int) -> float:
    ids = input_ids[0]
    for _ in range(warmup):
        frt.forward_prefill(ids)
    torch.cuda.synchronize()
    xs = [_cuda_ms(lambda: frt.forward_prefill(ids)) for _ in range(runs)]
    return statistics.mean(xs)


def _bench_hf_e2e(hf, input_ids: torch.Tensor, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        hf(input_ids=input_ids)
    torch.cuda.synchronize()
    xs = [_cuda_ms(lambda: hf(input_ids=input_ids)) for _ in range(runs)]
    return statistics.mean(xs)


# Candidate configs. Default = HF-parity ops.
# Each entry is a full flag set (not a leave-one-out patch to merge).
CANDIDATES: dict[str, dict[str, Any]] = {
    "default_hf_parity": {
        # use_sdpa auto on Thor
        "use_hf_norm": False,
        "use_hf_qkn_rope": True,
        "use_sep_gateup": False,
        "use_sep_qkv": True,
        "use_torch_qkv": True,
    },
    "all_fvk": {
        "use_hf_norm": False,
        "use_hf_qkn_rope": False,
        "use_sep_gateup": False,
        "use_sep_qkv": False,
        "use_torch_qkv": False,
    },
    "fvk_qkn_rope": {
        "use_hf_norm": False,
        "use_hf_qkn_rope": False,  # candidate: fused qkn+rope
        "use_sep_gateup": False,
        "use_sep_qkv": True,
        "use_torch_qkv": True,
    },
    "fused_qkv": {
        "use_hf_norm": False,
        "use_hf_qkn_rope": True,
        "use_sep_gateup": False,
        "use_sep_qkv": False,  # candidate: fused qkv
        "use_torch_qkv": False,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256,
                    help="specialization sequence length")
    ap.add_argument("--cosine-seq", type=int, default=8,
                    help="short seq for quick S1 gate (use 1024 for long)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--out", default="artifacts/qwen3_4b_thor_kernel_flags.json",
        help="kernel_flags JSON export path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = args.device
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"seq (bench)  : {args.seq}")
    print(f"seq (cosine) : {args.cosine_seq}")
    print("weights      : random shared state_dict")
    print("workflow     : S1 golden → S2 candidates → kernel_flags")

    t0 = time.time()
    hf, cfg, sd = _make_hf(device)

    # ── S1: torch golden (short + optional long already covered by pytest) ──
    torch.manual_seed(123)
    ids_cos = torch.randint(
        0, cfg.vocab_size, (1, args.cosine_seq), device=device, dtype=torch.long)
    frt_default = _make_frt(device, args.cosine_seq, sd, cfg, **CANDIDATES["default_hf_parity"])
    s1 = _cosine(hf, frt_default, ids_cos)
    s1_ok = s1["cosine_full"] >= COSINE_FLOOR
    print(f"\n=== S1 torch golden (FlashRT default vs HF) ===")
    print(f"  cosine_full={s1['cosine_full']:.6f}  last={s1['cosine_last']:.6f}  "
          f"{'PASS' if s1_ok else 'FAIL'} (floor {COSINE_FLOOR})")
    if not s1_ok:
        raise SystemExit("S1 failed — abort adaptation")

    # Resolve pipeline map (Thor registration).
    from flash_rt.hardware import detect_arch, resolve_pipeline_class
    arch = detect_arch()
    cls = resolve_pipeline_class("qwen3_4b", "torch", arch)
    print(f"  pipeline    : qwen3_4b/torch/{arch} → {cls.__name__}")
    print(f"  use_sdpa    : {frt_default.use_sdpa}")

    # ── S2: candidate block→model timing + cosine ──
    torch.manual_seed(123)
    ids_b = torch.randint(
        0, cfg.vocab_size, (1, args.seq), device=device, dtype=torch.long)
    hf_ms = _bench_hf_e2e(hf, ids_b, args.warmup, args.runs)
    print(f"\n=== S2 candidates (E2E prefill @ S={args.seq}) ===")
    print(f"  HF baseline : {hf_ms:.3f} ms")

    results: dict[str, Any] = {}
    base_ms = None
    for name, flags in CANDIDATES.items():
        frt = _make_frt(device, args.seq, sd, cfg, **flags)
        # Cosine at bench seq (gate)
        cos = _cosine(hf, frt, ids_b)
        ms = _bench_e2e(frt, ids_b, args.warmup, args.runs)
        if name == "default_hf_parity":
            base_ms = ms
        vs_hf = hf_ms / ms if ms > 0 else float("inf")
        vs_base = (base_ms / ms) if (base_ms and ms > 0) else 1.0
        cos_ok = cos["cosine_full"] >= COSINE_FLOOR
        # Accept only if numerically OK AND faster than HF by LAYER_SPEEDUP_FLOOR
        # relative to the HF-parity default when deciding fused replacements.
        if name == "default_hf_parity":
            verdict = "baseline"
        elif not cos_ok:
            verdict = "rejected"  # numerical gate
        elif vs_base >= LAYER_SPEEDUP_FLOOR:
            verdict = "accepted"  # faster than HF-parity default + cos OK
        else:
            verdict = "rejected"  # no meaningful gain vs default
        results[name] = {
            "flags": flags | {"use_sdpa": frt.use_sdpa},
            "e2e_ms": ms,
            "speedup_vs_hf": vs_hf,
            "speedup_vs_default": vs_base,
            "cosine_full": cos["cosine_full"],
            "cosine_last": cos["cosine_last"],
            "cosine_ok": cos_ok,
            "verdict": verdict,
        }
        print(
            f"  {name:20s}  {ms:8.3f} ms  "
            f"vsHF={vs_hf:5.2f}x  vsDef={vs_base:5.2f}x  "
            f"cos={cos['cosine_full']:.6f}  → {verdict}"
        )

    # ── S3: export kernel_flags (product of adaptation) ──
    # Pick the fastest cosine-OK full config (one flag set, not a mash).
    eligible = [
        (name, r) for name, r in results.items()
        if r["cosine_ok"] and name != "default_hf_parity"
    ]
    if eligible:
        best_name, best = min(eligible, key=lambda kv: kv[1]["e2e_ms"])
        if best["speedup_vs_default"] >= LAYER_SPEEDUP_FLOOR:
            selected = best_name
        else:
            selected = "default_hf_parity"
            best = results["default_hf_parity"]
    else:
        selected = "default_hf_parity"
        best = results["default_hf_parity"]
    kernel_flags = dict(best["flags"])
    provenance = {
        "selected": selected,
        "reason": (
            f"fastest cosine>={COSINE_FLOOR} config @ S={args.seq}: "
            f"{best['e2e_ms']:.3f} ms, vsHF={best['speedup_vs_hf']:.2f}x, "
            f"cos={best['cosine_full']:.6f}"
        ),
        "rejected": [
            name for name, r in results.items() if r["verdict"] == "rejected"
        ],
    }

    out = {
        "model_identity": "Qwen/Qwen3-4B",
        "platform": "thor",
        "arch": arch,
        "specialization": {"seq": args.seq, "dtype": "bf16"},
        "weights": "random_shared_state_dict",
        "cosine_floor": COSINE_FLOOR,
        "s1": s1,
        "hf_e2e_ms": hf_ms,
        "candidates": results,
        "kernel_flags": kernel_flags,
        "provenance": provenance,
        "elapsed_sec": time.time() - t0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n=== S3 kernel_flags ===")
    print(json.dumps(kernel_flags, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
