#!/usr/bin/env python3
"""Per-row (two-level) NVFP4 scaling probe — budget-neutral quality lever.

Current quant: one per-TENSOR fp32 global scale + per-16-block UE4M3 SF.
Per-row variant: each output row (N dim) gets its OWN fp32 global scale
(N floats, ~24KB/proj, negligible) + the same per-16 UE4M3 SF. Rows whose
magnitude differs from the tensor max then quantize with less relative error.

This is free in the W4A16 dequant path (per-row alpha folds into the
dequant), and never touches the FP4 storage size or the streaming budget.
Probe: per-tensor-global vs per-row-global NVFP4 fake-quant, cos vs original.
A meaningful per-tensor->per-row gain justifies extending the quantizer +
dequant kernel to per-row alpha.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_st_reader import RawShardReader  # noqa: E402

SRC = "/models/MiniMax-M3"
PFX = "language_model.model."
DEV = "cuda:0"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)


def nvfp4_q(w, global_scale):
    """w [N,K] fp32, global_scale scalar or [N,1]. Per-16 UE4M3 SF +
    e2m1 nearest. Returns dequantized fp32."""
    N, K = w.shape
    wf = w.view(N, K // 16, 16)
    gs = global_scale  # scalar or [N,1] broadcast over K-blocks
    if torch.is_tensor(gs) and gs.ndim == 2:
        gs = gs.view(N, 1, 1)
    sf = (wf.abs().amax(-1, keepdim=True) * gs / 6.0)
    sf = sf.to(torch.float8_e4m3fn).float().clamp(min=2.0 ** -9)
    q = wf * gs / sf
    mids = (E2M1[1:] + E2M1[:-1]) / 2
    idx = torch.bucketize(q.abs(), mids)
    deq = q.sign() * E2M1[idx] * sf / gs
    return deq.view(N, K)


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def probe(name, w):
    wf = w.float()
    g_tensor = (448.0 * 6.0) / wf.abs().max().clamp(min=1e-9)
    c_tensor = cos(nvfp4_q(wf, g_tensor), wf)
    g_row = (448.0 * 6.0) / wf.abs().amax(1, keepdim=True).clamp(min=1e-9)
    c_row = cos(nvfp4_q(wf, g_row), wf)
    gain = c_row - c_tensor
    flag = "WIN" if gain >= 1e-3 else "noop"
    print(f"  {name:24s} N={w.shape[0]:5d} K={w.shape[1]:5d} "
          f"per-tensor {c_tensor:.5f}  per-row {c_row:.5f} d{gain:+.5f} {flag}",
          flush=True)
    return gain


def main():
    rd = RawShardReader(SRC, DEV)
    gains = []
    for li in [5, 30, 59]:
        p = f"{PFX}layers.{li}."
        for nm, src in [
            ("q_proj", "self_attn.q_proj.weight"),
            ("o_proj", "self_attn.o_proj.weight"),
            ("e0.w1", "block_sparse_moe.experts.0.w1.weight"),
            ("e0.w2", "block_sparse_moe.experts.0.w2.weight"),
            ("e0.w3", "block_sparse_moe.experts.0.w3.weight"),
        ]:
            w = rd.get(p + src, DEV).to(torch.bfloat16)
            gains.append(probe(f"L{li}.{nm}", w))
        rd.drop_all()
    g = torch.tensor(gains)
    print(f"\nper-row gain over per-tensor: mean {g.mean():+.5f} "
          f"max {g.max():+.5f}  ({int((g >= 1e-3).sum())}/{len(gains)} WIN)")
    print("verdict:", "implement per-row alpha" if g.mean() >= 1e-3
          else "per-row noop — per-16 SF already captures it")


if __name__ == "__main__":
    main()
