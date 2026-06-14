#!/usr/bin/env python3
"""V6 — additive over V5. Fuse qk-norm + partial-rope into one kernel.

V5 per layer: rms_norm(Q) + rms_norm(K) + qwen36_partial_rope_qk (3 launches,
3 HBM round-trips of Q/K). V6 replaces them with a single fused kernel
(fused_qk_norm_rope.py), one launch + one round-trip each for Q/K.

Near-exact (rel ~1e-5, 1 bf16-ULP from reduction order; see fused validation).
Applies to BOTH the gen graph and the und precompute via the overridden
_qk_norm_rope hook. V2-V5 untouched.
"""
import sys
from .cosmos_v5 import CosmosV5
from .cosmos_v2 import H, KV, D, EPS
from ..kernels import qk_norm_rope


class CosmosV6(CosmosV5):
    def _qk_norm_rope(self, lo, hi, n, li, suf):
        qk_norm_rope(self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(),
                     self.Wn[(li, suf, "self_attn.q_norm")].data_ptr(),
                     self.Wn[(li, suf, "self_attn.k_norm")].data_ptr(),
                     self.cos[lo:hi].data_ptr(), self.sin[lo:hi].data_ptr(),
                     n, H, KV, D, EPS, self._s())
