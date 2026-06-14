#!/usr/bin/env python3
"""V4 — additive over V3. Fuse the FFN down-input path:

  V3:  silu_mul_qwen36_bf16(g,u)->act(bf16);  _proj(act,down) [quantize act->fp4; gemm]
  V4:  silu_mul_to_nvfp4_swizzled_bf16(g,u)->fp4 act directly;  down gemm reads fp4

This removes the bf16 `act[M,FF]` materialization + separate act-quant launch for
the down projection. Numerically bit-identical to V3 (micro-validated: cos 1.00000,
see bench_silu_fuse.py). Only active when `down` is NVFP4 (not in bf16_projs);
the bf16-down path falls back to the V3 behaviour.

Additive-only: subclasses CosmosV3, adds per-tower FP4 down-input buffers, and
overrides forward(). V2/V3 untouched.
"""
import os, sys, time, torch, torch.nn.functional as F
from safetensors import safe_open
import flash_rt.flash_rt_kernels as fvk
from .cosmos_v3 import CosmosV3
from .cosmos_v2 import (te_table, sf_bytes, NL, H, KV, D, FF, HID, NU, NG, NJ, AV,
                       AL, VL, AC, EPS, BF, DEV, R)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler


class CosmosV4(CosmosV3):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        # per-tower FP4 down-input buffers (K=FF): silu fused output feeds down gemm
        z8 = lambda r, c: (torch.empty(r, c // 2, dtype=torch.uint8, device=DEV),
                           torch.zeros(sf_bytes(r, c), dtype=torch.uint8, device=DEV))
        self.pa = {"": z8(NU, FF), "_moe_gen": z8(NG, FF)}
        self.fuse_down = "down" not in self.bf16_projs   # only fuse when down is NVFP4
        torch.cuda.synchronize()

    def _silu_q(self, g, u, pk, sf, n):  # silu(g)*u -> nvfp4 (fused, one launch)
        fvk.silu_mul_to_nvfp4_swizzled_bf16(g.data_ptr(), u.data_ptr(), pk.data_ptr(), sf.data_ptr(), n, FF, self._s())

    def forward(self):
        s = self._s()
        self.gemm.bf16_nn(self.xfull.data_ptr(), self.Wa.data_ptr(), self.aenc.data_ptr(), AL, HID, 64, s)
        fvk.add_bf16_out(self.aenc.data_ptr(), self.ba_modA.data_ptr(), self.aenc.data_ptr(), AL * HID, s)
        fvk.add_bf16_out(self.aenc[AC:AL].data_ptr(), self.te.data_ptr(), self.aenc[AC:AL].data_ptr(), VL * HID, s)
        self.Hb[0:NU].copy_(self.text); self.Hb[NU:NU + AV].copy_(self.vision); self.Hb[NU + AV:NJ].copy_(self.aenc)
        self._rms_q(self.Hb[0:NU], self.Wn[(0, "", "input_layernorm")], *self.pn[""], NU)
        self._rms_q(self.Hb[NU:NJ], self.Wn[(0, "_moe_gen", "input_layernorm")], *self.pn["_moe_gen"], NG)
        for li in range(NL):
            last = (li == NL - 1)
            for (suf, lo, hi, n) in (("", 0, NU, NU), ("_moe_gen", NU, NJ, NG)):
                pk, sf = self.pn[suf]
                self._gemm_pre(pk, sf, (li, suf, "q_proj"), self.Qb[lo:hi].view(n, H * D), n, HID, HID)
                self._gemm_pre(pk, sf, (li, suf, "k_proj"), self.Kb[lo:hi].view(n, KV * D), n, KV * D, HID)
                self._gemm_pre(pk, sf, (li, suf, "v_proj"), self.Vb[lo:hi].view(n, KV * D), n, KV * D, HID)
                self._rms(self.Qb[lo:hi], self.Wn[(li, suf, "self_attn.q_norm")], self.Qb[lo:hi], n * H, D)
                self._rms(self.Kb[lo:hi], self.Wn[(li, suf, "self_attn.k_norm")], self.Kb[lo:hi], n * KV, D)
                fvk.qwen36_partial_rope_qk_bf16(self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(),
                    self.cos[lo:hi].data_ptr(), self.sin[lo:hi].data_ptr(),
                    self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(), n, H, KV, D, D, s)
            self._fa(self.Qb[0:NU], self.Kb[0:NU], self.Vb[0:NU], self.attn[0:NU].view(NU, H, D), NU, NU, True)
            self._fa(self.Qb[NU:NJ], self.Kb[0:NJ], self.Vb[0:NJ], self.attn[NU:NJ].view(NG, H, D), NG, NJ, False)
            for (suf, lo, hi, n) in (("", 0, NU, NU), ("_moe_gen", NU, NJ, NG)):
                pk, sf = self.pn[suf]
                self._proj(self.attn[lo:hi], (li, suf, "o_proj"), self.ob[lo:hi], HID)
                self._radd_rms_q(self.Hb[lo:hi], self.ob[lo:hi], self.Wn[(li, suf, "post_attention_layernorm")], pk, sf, n)
                self._gemm_pre(pk, sf, (li, suf, "gate"), self.g[lo:hi], n, FF, HID)
                self._gemm_pre(pk, sf, (li, suf, "up"), self.u[lo:hi], n, FF, HID)
                if self.fuse_down:                                   # fused silu->fp4->down
                    apk, asf = self.pa[suf]
                    self._silu_q(self.g[lo:hi], self.u[lo:hi], apk, asf, n)
                    self._gemm_pre(apk, asf, (li, suf, "down"), self.dn[lo:hi], n, HID, FF)
                else:                                                # V3 fallback (bf16 down)
                    self._silu(self.g[lo:hi], self.u[lo:hi], self.act[lo:hi], n * FF)
                    self._proj(self.act[lo:hi], (li, suf, "down"), self.dn[lo:hi], HID)
                if not last:
                    self._radd_rms_q(self.Hb[lo:hi], self.dn[lo:hi], self.Wn[(li + 1, suf, "input_layernorm")], pk, sf, n)
                else:
                    fvk.residual_add(self.Hb[lo:hi].data_ptr(), self.dn[lo:hi].data_ptr(), n * HID, s)
        self._rms(self.Hb[NU + AV + AC:NJ], self.norm_g, self.nrm[NU + AV + AC:NJ], VL, HID)
        self.gemm.bf16_nn(self.nrm[NU + AV + AC:NJ].data_ptr(), self.Wll.data_ptr(), self.vtmp.data_ptr(), VL, 64, HID, s)
        fvk.add_bf16_out(self.vtmp.data_ptr(), self.bllV.data_ptr(), self.vel.data_ptr(), VL * 64, s)
        return self.vel
