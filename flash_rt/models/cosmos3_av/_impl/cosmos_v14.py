#!/usr/bin/env python3
"""V14 — CONFIRMED-DEAD precision probe: fp8-PV sage on the late layers. Default OFF.

Hypothesis: V13's 25 late layers [11:36] use int8-QK + f16-PV sage (~1.86ms/call);
running PV in fp8 (~1.16ms/call standalone, -0.7ms) could save a lot. RESULT: dead.
E2E action rel_l2 = 17.5% (gate 3%, V13 f16-PV = 2.458%). Root cause = precision
budget: int8-QK on 25 layers already leaves only 0.54% margin, and f8-PV adds ~+0.8%
single-layer quant error that compounds super-additively over 25 layers (same wall the
int8-boundary sweep hit). Even a correct GQA-f8 kernel can't fit at lo=11. (Note: the
GQA-native f8 launcher sage_gqa_f8_d128.py is ALSO buggy — standalone cos 0.24 vs 0.94
for f16, V-tpp GQA strides wrong — so this probe uses the bound nhd f8 kernel + KV->32
expand.) The pipeline is at the attention precision frontier; f8-PV is not shippable.

Default COSMOS_PV_FP8=0 -> identical to V13. Kept in-tree to record the negative.
"""
import os, sys, torch
import flash_rt.flash_rt_kernels as fvk
from .cosmos_v13 import CosmosV13
from .cosmos_v6 import CosmosV6
from .cosmos_v2 import H, KV, D, NU, NJ, NG, DEV, BF

PADNJ = ((NJ + 63) // 64) * 64
_G = H // KV


class CosmosV14(CosmosV13):
    """nhd f8-PV path: GQA-native f8 launcher (sage_gqa_f8_d128) is broken (cos 0.24
    standalone, V-tpp GQA strides wrong), so use the BOUND nhd f8 kernel with KV->32
    expand. This is purely the precision probe — if E2E holds under 3%, the speed path
    is worth fixing; the expand overhead here partly offsets the f8 PV win."""
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        self.pv_fp8 = os.environ.get("COSMOS_PV_FP8", "0") == "1"  # dead: default OFF
        self.fq8 = torch.empty(1, NG, H, D, dtype=torch.int8, device=DEV)
        self.fke = torch.empty(NJ, H, D, device=DEV, dtype=BF)       # K expanded 8->32
        self.fk8 = torch.empty(1, NJ, H, D, dtype=torch.int8, device=DEV)
        self.fqs = torch.empty(1, H, (NG + 31) // 32, dtype=torch.float32, device=DEV)
        self.fks = torch.empty(1, H, (NJ + 63) // 64, dtype=torch.float32, device=DEV)
        self.vtpp = torch.zeros(1, D, H, PADNJ, dtype=BF, device=DEV)  # V expanded 8->32
        self.vf8 = torch.empty(1, D, H, PADNJ, dtype=torch.float8_e4m3fn, device=DEV)
        self.vsc = torch.empty(1, H, D, dtype=torch.float32, device=DEV)
        torch.cuda.synchronize()

    def _gen_attn(self, li):
        if not self.use_sage or not (self.sage_lo <= li < self.sage_hi):
            return CosmosV6._gen_attn(self, li)        # bf16 FA2 (early layers)
        if not self.pv_fp8:
            return CosmosV13._gen_attn(self, li)        # f16-PV sage fallback
        s = self._s()
        # GQA expand 8->32 for K (int8) and V (tpp fp8)
        self.fke.view(NJ, KV, _G, D).copy_(self.Kb[0:NJ].view(NJ, KV, 1, D).expand(NJ, KV, _G, D))
        self.vtpp[0, :, :, :NJ].view(D, KV, _G, NJ).copy_(
            self.Vb[0:NJ].permute(2, 1, 0).unsqueeze(2).expand(D, KV, _G, NJ))
        fvk.quant_per_warp_int8_bf16_d128(self.Qb[NU:NJ].data_ptr(), self.fq8.data_ptr(), self.fqs.data_ptr(), 1, NG, H, s)
        fvk.quant_per_block_int8_bf16_d128(self.fke.data_ptr(), self.fk8.data_ptr(), self.fks.data_ptr(), 1, NJ, H, s)
        fvk.v_tpp_bf16_quant_fp8_d128(self.vtpp.data_ptr(), self.vf8.data_ptr(), self.vsc.data_ptr(), 1, NJ, H, s)
        fvk.sage2_qk_int8_sv_f8_bf16_nhd_d128(self.fq8.data_ptr(), self.fk8.data_ptr(), self.vf8.data_ptr(),
            self.attn[NU:NJ].data_ptr(), self.fqs.data_ptr(), self.fks.data_ptr(), self.vsc.data_ptr(),
            1, NG, NJ, H, D ** -0.5, s)
