#!/usr/bin/env python3
"""V9 — additive over V8. fp8 V (PV) in GQA sage instead of f16 V (~1.6x PV).

V needs SageAttention tpp layout [B,D,KV,padK] + per-channel v_scale, quantized
per layer (v_tpp_bf16_quant_fp8_d128). More approximate than f16 V (single-layer
rel_l2 2.69->3.49%) — same layer-range gating. Additive: subclasses V8.
COSMOS_SAGE_PV=f16 falls back to V8 (f16 V).
"""
import os, sys, torch
import flash_rt.flash_rt_kernels as fvk
from .cosmos_v8 import CosmosV8
from .cosmos_v6 import CosmosV6
from .cosmos_v2 import H, KV, D, NU, NJ, NG, DEV, BF
from ..kernels import sage_gqa_f8_d128

PADK = ((NJ + 63) // 64) * 64


class CosmosV9(CosmosV8):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        self.pv = os.environ.get("COSMOS_SAGE_PV", "f8")
        # f8 V only on layers [pv_f8_lo, sage_hi); f16 V on [sage_lo, pv_f8_lo)
        self.pv_f8_lo = int(os.environ.get("COSMOS_PV_F8_LO", str(self.sage_lo)))
        self.v_tpp = torch.zeros(1, D, KV, PADK, dtype=BF, device=DEV)
        self.vfp8 = torch.empty(1, D, KV, PADK, dtype=torch.float8_e4m3fn, device=DEV)
        self.vsc = torch.empty(1, KV, D, dtype=torch.float32, device=DEV)
        torch.cuda.synchronize()

    def _gen_attn(self, li):
        if self.pv != "f8" or not self.use_sage or not (self.pv_f8_lo <= li < self.sage_hi):
            return super()._gen_attn(li)                  # V8 (f16 V) or bf16
        s = self._s()
        fvk.quant_per_warp_int8_bf16_d128(self.Qb[NU:NJ].data_ptr(), self.gq8.data_ptr(), self.gqs.data_ptr(), 1, NG, H, s)
        fvk.quant_per_block_int8_bf16_d128(self.Kb[0:NJ].data_ptr(), self.gk8.data_ptr(), self.gks.data_ptr(), 1, NJ, KV, s)
        # V -> tpp [1,D,KV,padK]: v_tpp[0,d,h,k] = Vb[k,h,d]
        self.v_tpp[0, :, :, :NJ].copy_(self.Vb[0:NJ].permute(2, 1, 0))
        fvk.v_tpp_bf16_quant_fp8_d128(self.v_tpp.data_ptr(), self.vfp8.data_ptr(), self.vsc.data_ptr(), 1, NJ, KV, s)
        sage_gqa_f8_d128(self.gq8.data_ptr(), self.gk8.data_ptr(), self.vfp8.data_ptr(),
                         self.attn[NU:NJ].data_ptr(), self.gqs.data_ptr(), self.gks.data_ptr(), self.vsc.data_ptr(),
                         1, NG, NJ, H, KV, D ** -0.5, s)
