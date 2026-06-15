#!/usr/bin/env python3
"""V13 — additive over V12. Extends the int8-sage attention boundary from layer 16
down to layer 11 (sage on gen layers [11:36] instead of [16:36]).

WHY (measured, not assumed): at the TRUE gen-attn shape (Q[6300,32,128] x KV[6410,8,
128], the 480p/5-frame vision tokens dominate) bf16 FA2 runs at 3.20ms/call = 1.01x
the bf16 roofline — it is already optimal, so a "faster bf16 attention kernel" has zero
headroom (an earlier premise was computed against a stale 633-token shape).
The only attention lever is reduced precision. int8 sage is 1.60x faster than FA2.
Per-layer error compounds through 36 layers (single-layer cos 0.9998 is misleading), so
sage can only be pushed so far before the action rel_l2 gate (3%) breaks:
  sage_lo 16 -> rel_l2 2.355% (V12)   13 -> 2.574%   11 -> 2.458%   10 -> 2.960%   9 -> 3.163%
Non-contiguous "cheapest-layer" sets are strongly super-additive (10 cheap layers = 3.36%),
so the contiguous boundary is near-optimal. lo=11 dominates lo=12/13 (more layers AND lower
rel_l2, from per-layer error cancellation) with a 2-layer margin before the lo=9 cliff.

Result: E2E denoise 1835 -> 1783 ms (-52 ms, -2.8%), action rel_l2 2.458% (gate 3%, was 2.355%).

Additive: subclasses V12, only changes the COSMOS_SAGE_LO default. V12 stays the 2.355%
baseline. COSMOS_SAGE_LO overrides the boundary explicitly.
"""
import os
from .cosmos_v12 import CosmosV12


class CosmosV13(CosmosV12):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        # V12 default was 16; measured optimal under the 3% gate is 11.
        self.sage_lo = int(os.environ.get("COSMOS_SAGE_LO", "11"))
