#!/usr/bin/env python3
"""V5 — additive over V4. Static text/und-tower KV cache.

Empirical fact (verified on the official AV ref dump): the und/causal tower
(text, NU tokens) is BIT-EXACT static across all denoise steps (text only
attends to text, causally; it never attends to the noisy action tokens). Its
ONLY coupling to the velocity output is the per-layer text K/V that the gen
tower reads in its full attention (Kb[0:NU], Vb[0:NU]).

So we compute the und tower exactly ONCE (precompute_und), snapshot post-rope
text K and raw text V for every layer, and DROP the entire und tower from the
per-step CUDA graph. The per-step graph runs the gen tower only and stamps the
cached text K/V into Kb[0:NU]/Vb[0:NU] before each layer's gen attention.

This is exact (text K/V identical every step) and removes ~36 layers of und
q/k/v/o/gate/up/down GEMMs + norms + rope + causal FA2 from each step.

Additive-only: subclasses CosmosV4; V2/V3/V4 untouched.
"""
import os, sys, time, torch, torch.nn.functional as F
from safetensors import safe_open
import flash_rt.flash_rt_kernels as fvk
from .cosmos_v4 import CosmosV4
from .cosmos_v2 import (te_table, sf_bytes, NL, H, KV, D, FF, HID, NU, NG, NJ, AV,
                       AL, VL, AC, EPS, BF, DEV, R)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler


class CosmosV5(CosmosV4):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        # per-layer static text K/V cache (post-rope K, raw V)
        self.cK = torch.zeros(NL, NU, KV, D, device=DEV, dtype=BF)
        self.cV = torch.zeros(NL, NU, KV, D, device=DEV, dtype=BF)
        self._und_ready = False
        torch.cuda.synchronize()

    def _ffn_updown(self, lo, hi, n, li, suf, pk, sf):  # gate+up+silu+down; V10 fuses
        self._gemm_pre(pk, sf, (li, suf, "gate"), self.g[lo:hi], n, FF, HID)
        self._gemm_pre(pk, sf, (li, suf, "up"), self.u[lo:hi], n, FF, HID)
        if self.fuse_down:
            apk, asf = self.pa[suf]
            self._silu_q(self.g[lo:hi], self.u[lo:hi], apk, asf, n)
            self._gemm_pre(apk, asf, (li, suf, "down"), self.dn[lo:hi], n, HID, FF)
        else:
            self._silu(self.g[lo:hi], self.u[lo:hi], self.act[lo:hi], n * FF)
            self._proj(self.act[lo:hi], (li, suf, "down"), self.dn[lo:hi], HID)

    def _gen_attn(self, li):  # gen full attention (bf16 FA2 default; V7 overrides sage)
        self._fa(self.Qb[NU:NJ], self.Kb[0:NJ], self.Vb[0:NJ], self.attn[NU:NJ].view(NG, H, D), NG, NJ, False)

    def _qk_norm_rope(self, lo, hi, n, li, suf):  # 3-kernel default; V6 overrides fused
        self._rms(self.Qb[lo:hi], self.Wn[(li, suf, "self_attn.q_norm")], self.Qb[lo:hi], n * H, D)
        self._rms(self.Kb[lo:hi], self.Wn[(li, suf, "self_attn.k_norm")], self.Kb[lo:hi], n * KV, D)
        fvk.qwen36_partial_rope_qk_bf16(self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(),
            self.cos[lo:hi].data_ptr(), self.sin[lo:hi].data_ptr(),
            self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(), n, H, KV, D, D, self._s())

    # ---- one-time exact und-tower pass: fills cK/cV for all layers ----
    def precompute_und(self):
        s = self._s()
        suf, lo, hi, n = "", 0, NU, NU
        self.Hb[0:NU].copy_(self.text)
        self._rms_q(self.Hb[0:NU], self.Wn[(0, "", "input_layernorm")], *self.pn[""], NU)
        for li in range(NL):
            last = (li == NL - 1)
            pk, sf = self.pn[suf]
            self._gemm_pre(pk, sf, (li, suf, "q_proj"), self.Qb[lo:hi].view(n, H * D), n, HID, HID)
            self._gemm_pre(pk, sf, (li, suf, "k_proj"), self.Kb[lo:hi].view(n, KV * D), n, KV * D, HID)
            self._gemm_pre(pk, sf, (li, suf, "v_proj"), self.Vb[lo:hi].view(n, KV * D), n, KV * D, HID)
            self._qk_norm_rope(lo, hi, n, li, suf)
            # snapshot text K/V for this layer (post q/k-norm + rope K; raw V)
            self.cK[li].copy_(self.Kb[lo:hi]); self.cV[li].copy_(self.Vb[lo:hi])
            self._fa(self.Qb[0:NU], self.Kb[0:NU], self.Vb[0:NU], self.attn[0:NU].view(NU, H, D), NU, NU, True)
            self._proj(self.attn[lo:hi], (li, suf, "o_proj"), self.ob[lo:hi], HID)
            self._radd_rms_q(self.Hb[lo:hi], self.ob[lo:hi], self.Wn[(li, suf, "post_attention_layernorm")], pk, sf, n)
            self._ffn_updown(lo, hi, n, li, suf, pk, sf)
            if not last:
                self._radd_rms_q(self.Hb[lo:hi], self.dn[lo:hi], self.Wn[(li + 1, suf, "input_layernorm")], pk, sf, n)
        # stamp the static text K/V once into the joint buffer (gen attn reads [0:NJ])
        torch.cuda.synchronize()
        self._und_ready = True

    # ---- per-step graphed forward: GEN tower only, text K/V from cache ----
    def forward(self):
        s = self._s()
        self.gemm.bf16_nn(self.xfull.data_ptr(), self.Wa.data_ptr(), self.aenc.data_ptr(), AL, HID, 64, s)
        fvk.add_bf16_out(self.aenc.data_ptr(), self.ba_modA.data_ptr(), self.aenc.data_ptr(), AL * HID, s)
        fvk.add_bf16_out(self.aenc[AC:AL].data_ptr(), self.te.data_ptr(), self.aenc[AC:AL].data_ptr(), VL * HID, s)
        self.Hb[NU:NU + AV].copy_(self.vision); self.Hb[NU + AV:NJ].copy_(self.aenc)
        self._rms_q(self.Hb[NU:NJ], self.Wn[(0, "_moe_gen", "input_layernorm")], *self.pn["_moe_gen"], NG)
        suf, lo, hi, n = "_moe_gen", NU, NJ, NG
        for li in range(NL):
            last = (li == NL - 1)
            pk, sf = self.pn[suf]
            self._gemm_pre(pk, sf, (li, suf, "q_proj"), self.Qb[lo:hi].view(n, H * D), n, HID, HID)
            self._gemm_pre(pk, sf, (li, suf, "k_proj"), self.Kb[lo:hi].view(n, KV * D), n, KV * D, HID)
            self._gemm_pre(pk, sf, (li, suf, "v_proj"), self.Vb[lo:hi].view(n, KV * D), n, KV * D, HID)
            self._qk_norm_rope(lo, hi, n, li, suf)
            # stamp cached static text K/V into the joint buffer head
            self.Kb[0:NU].copy_(self.cK[li]); self.Vb[0:NU].copy_(self.cV[li])
            self._gen_attn(li)
            self._proj(self.attn[lo:hi], (li, suf, "o_proj"), self.ob[lo:hi], HID)
            self._radd_rms_q(self.Hb[lo:hi], self.ob[lo:hi], self.Wn[(li, suf, "post_attention_layernorm")], pk, sf, n)
            self._ffn_updown(lo, hi, n, li, suf, pk, sf)
            if not last:
                self._radd_rms_q(self.Hb[lo:hi], self.dn[lo:hi], self.Wn[(li + 1, suf, "input_layernorm")], pk, sf, n)
            else:
                fvk.residual_add(self.Hb[lo:hi].data_ptr(), self.dn[lo:hi].data_ptr(), n * HID, s)
        self._rms(self.Hb[NU + AV + AC:NJ], self.norm_g, self.nrm[NU + AV + AC:NJ], VL, HID)
        self.gemm.bf16_nn(self.nrm[NU + AV + AC:NJ].data_ptr(), self.Wll.data_ptr(), self.vtmp.data_ptr(), VL, 64, HID, s)
        fvk.add_bf16_out(self.vtmp.data_ptr(), self.bllV.data_ptr(), self.vel.data_ptr(), VL * 64, s)
        return self.vel

    def capture(self):
        if not self._und_ready:
            self.precompute_und()
        super().capture()
