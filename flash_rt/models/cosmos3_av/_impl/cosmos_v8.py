#!/usr/bin/env python3
"""V8 — additive over V7. GQA-native d128 SageAttention (no 8->32 KV expand).

V7 used the non-GQA nhd_d128 sage and had to expand K/V 8->32 heads (extra copy +
4x K int8-quant + 4x KV bandwidth). V8 uses the GQA-native d128 launcher
(sage_gqa_d128.py, reuses FlashRT's compiled f16 attn kernel template with
num_kv_groups) so K/V stay 8 heads.

Same int8-QK approximation as V7 (so same layer-range gating; default [16,36)).
Additive: subclasses V7, overrides _gen_attn; V2-V7 untouched.
"""
import os, sys, torch
import flash_rt.flash_rt_kernels as fvk
from .cosmos_v7 import CosmosV7
from .cosmos_v6 import CosmosV6
from .cosmos_v2 import H, KV, D, NU, NJ, NG, DEV, BF
from ..kernels import sage_gqa_d128


class CosmosV8(CosmosV7):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        # GQA sage buffers (K/V stay KV=8 heads — no expand)
        self.gq8 = torch.empty(1, NG, H, D, dtype=torch.int8, device=DEV)
        self.gk8 = torch.empty(1, NJ, KV, D, dtype=torch.int8, device=DEV)
        self.gqs = torch.empty(1, H, (NG + 31) // 32, dtype=torch.float32, device=DEV)
        self.gks = torch.empty(1, KV, (NJ + 63) // 64, dtype=torch.float32, device=DEV)
        self.gv16 = torch.empty(1, NJ, KV, D, dtype=torch.float16, device=DEV)
        torch.cuda.synchronize()

    def _gen_attn(self, li):
        if not self.use_sage or not (self.sage_lo <= li < self.sage_hi):
            return CosmosV6._gen_attn(self, li)          # bf16 FA2 (skip V7 nhd)
        s = self._s()
        fvk.quant_per_warp_int8_bf16_d128(self.Qb[NU:NJ].data_ptr(), self.gq8.data_ptr(), self.gqs.data_ptr(), 1, NG, H, s)
        fvk.quant_per_block_int8_bf16_d128(self.Kb[0:NJ].data_ptr(), self.gk8.data_ptr(), self.gks.data_ptr(), 1, NJ, KV, s)
        self.gv16.view(NJ, KV, D).copy_(self.Vb[0:NJ])    # bf16 -> f16
        sage_gqa_d128(self.gq8.data_ptr(), self.gk8.data_ptr(), self.gv16.data_ptr(),
                      self.attn[NU:NJ].data_ptr(), self.gqs.data_ptr(), self.gks.data_ptr(),
                      1, NG, NJ, H, KV, D ** -0.5, s)
