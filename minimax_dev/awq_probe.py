#!/usr/bin/env python3
"""Cheap AWQ-on-NVFP4 viability probe (before any 232GB re-quant).

Memory warns AWQ may not help on top of NVFP4's per-16-block SF (motus VAE
phase9: "AWQ/skip can't break it"). NVFP4 already adapts scale per 16-element
K-block, so per-channel AWQ rescaling may be redundant. This probe quantifies,
on real M3 weights with REAL per-channel activation magnitudes, whether an AWQ
per-input-channel scale measurably lowers the W4A16 quant error.

For each sampled projection:
  - plain NVFP4 fake-quant -> cos vs original  (baseline)
  - AWQ: s_j = act_mag_j^a / w_mag_j^(1-a), W'=W*s, then quant W', dequant,
    divide back by s -> effective W4A16 weight -> cos vs original
    sweep a in {0, .25, .5, .75, 1.0}, report best
  - also report "act-aware proxy" using ||W[:,j]|| when no act stats

A meaningful win (best-AWQ cos - plain cos >= ~0.001 per tensor, which the ~20x
E2E amplification turns into ~0.02 E2E) justifies the re-quant. Otherwise skip
AWQ and reach quality via resident-FP8 instead.

Activation magnitudes: from an instrumented reference pass if --act-stats is
given (per-projection per-channel mean|x|); else falls back to weight-column
proxy (self-calibration), which lower-bounds AWQ's real benefit.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_nvfp4_layerwise_check import nvfp4_fake_quant  # noqa: E402
from raw_st_reader import RawShardReader  # noqa: E402

SRC = "/models/MiniMax-M3"
PFX = "language_model.model."
DEV = "cuda:0"


def cos(a, b):
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(),
                                     dim=0))


def awq_quant(w_bf16, act_mag, alpha):
    """W' = W * s (per input channel j), quant, dequant, /s. act_mag, col
    proxy combined as AWQ. Returns effective dequantized weight."""
    N, K = w_bf16.shape
    w = w_bf16.float()
    w_mag = w.abs().mean(0).clamp(min=1e-8)        # [K]
    a = act_mag.clamp(min=1e-8)                     # [K]
    s = (a ** alpha) / (w_mag ** (1.0 - alpha))     # [K]
    s = (s / s.mean()).clamp(min=1e-2, max=1e2)     # normalize scale
    wq = nvfp4_fake_quant((w * s[None, :]).to(torch.bfloat16)).float()
    return wq / s[None, :]


def probe(name, w, act_mag):
    plain = nvfp4_fake_quant(w).float()
    c_plain = cos(plain, w)
    best_a, best_c = None, -1
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        wq = awq_quant(w, act_mag, a)
        c = cos(wq, w)
        if c > best_c:
            best_c, best_a = c, a
    gain = best_c - c_plain
    flag = "WIN" if gain >= 1e-3 else "noop"
    print(f"  {name:26s} N={w.shape[0]:5d} K={w.shape[1]:5d} "
          f"plain {c_plain:.5f}  awq* {best_c:.5f} (a={best_a}) "
          f"d{gain:+.5f} {flag}", flush=True)
    return gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-stats", default="",
                    help="pt file: {proj_key: per-channel mean|x| [K]}")
    args = ap.parse_args()
    rd = RawShardReader(SRC, DEV)
    act = (torch.load(args.act_stats, map_location=DEV, weights_only=False)
           if args.act_stats else {})

    gains = []
    for li in [5, 30, 59]:
        p = f"{PFX}layers.{li}."
        cases = [
            (f"L{li}.q_proj", "self_attn.q_proj.weight"),
            (f"L{li}.o_proj", "self_attn.o_proj.weight"),
            (f"L{li}.sh_gate", "block_sparse_moe.shared_experts.gate_proj.weight"),
            (f"L{li}.e0.w1", "block_sparse_moe.experts.0.w1.weight"),
            (f"L{li}.e0.w2", "block_sparse_moe.experts.0.w2.weight"),
        ]
        for nm, src in cases:
            w = rd.get(p + src, DEV).to(torch.bfloat16)
            am = act.get(nm, w.float().abs().mean(0))  # proxy if no stats
            gains.append(probe(nm, w, am.to(DEV)))
        rd.drop_all()

    g = torch.tensor(gains)
    print(f"\nAWQ gain over plain NVFP4: mean {g.mean():+.5f} "
          f"max {g.max():+.5f}  ({int((g >= 1e-3).sum())}/{len(gains)} WIN)")
    print("verdict:", "pursue AWQ re-quant" if g.mean() >= 1e-3
          else "AWQ noop on NVFP4 per-block — reach quality via resident-FP8")


if __name__ == "__main__":
    main()
