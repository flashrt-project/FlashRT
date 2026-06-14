#!/usr/bin/env python3
"""V10 — additive over V8. FFN-mega: fuse the gen up-GEMM + silu_mul into one
cutlass kernel (fp4_silu_aux: up GEMM with epilogue D=silu(gate_aux)*up -> bf16),
eliminating the standalone silu_mul->fp4 launch + the up (u) HBM materialization.

D is bf16 (the fp4-out fused variant hits a cutlass SF-store FragmentSize wall);
a cheap quantize(act->fp4) feeds the down GEMM. Net removes silu read of g,u.

NVFP4 SF tiling needs M % 128 == 0, so the gen FFN runs at M=NGP=6400 (gen NG=6300
padded; tail 100 rows garbage, ignored). pn["_moe_gen"] reallocated to 6400 rows so
radd_rms writes [0:6300] and the M=6400 GEMM reads it with no copy. Only gen tower
fused; und precompute falls back to V5 path.

Additive: subclasses V8; overrides _ffn_updown for the gen tower.
"""
import os, sys, torch
import flash_rt.flash_rt_kernels as fvk
from .cosmos_v8 import CosmosV8
from .cosmos_v2 import sf_bytes, FF, HID, NG, DEV, BF
from ..kernels import fp4_silu_aux

NGP = ((NG + 127) // 128) * 128   # 6400


class CosmosV10(CosmosV8):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        self.ffn_mega = os.environ.get("COSMOS_FFN_MEGA", "1") == "1" and self.fuse_down
        # reallocate gen norm-output buffer to NGP rows (radd_rms writes 6300, GEMM reads 6400)
        self.pn["_moe_gen"] = (torch.empty(NGP, HID // 2, dtype=torch.uint8, device=DEV),
                               torch.zeros(sf_bytes(NGP, HID), dtype=torch.uint8, device=DEV))
        self.g_pad = torch.empty(NGP, FF, device=DEV, dtype=BF)
        self.act_pad = torch.empty(NGP, FF, device=DEV, dtype=BF)
        self.actfp4 = (torch.empty(NG, FF // 2, dtype=torch.uint8, device=DEV),
                       torch.zeros(sf_bytes(NG, FF), dtype=torch.uint8, device=DEV))
        torch.cuda.synchronize()

    def _ffn_updown(self, lo, hi, n, li, suf, pk, sf):
        if not self.ffn_mega or n != NG:
            return super()._ffn_updown(lo, hi, n, li, suf, pk, sf)
        s = self._s()
        Wg, Wgsf = self.Wp[(li, suf, "gate")], self.Wsf[(li, suf, "gate")]
        Wu, Wusf = self.Wp[(li, suf, "up")], self.Wsf[(li, suf, "up")]
        # gate GEMM (M=NGP) -> g_pad bf16
        fvk.fp4_w4a16_gemm_sm120_bf16out(pk.data_ptr(), Wg.data_ptr(), self.g_pad.data_ptr(),
            NGP, FF, HID, sf.data_ptr(), Wgsf.data_ptr(), 1.0, s)
        # up GEMM with fused silu(gate)*up -> act_pad bf16 (M=NGP)
        fp4_silu_aux(pk.data_ptr(), sf.data_ptr(), Wu.data_ptr(), Wusf.data_ptr(),
            self.g_pad.data_ptr(), self.act_pad.data_ptr(), 0, NGP, FF, HID, s)
        # quantize act[0:NG] -> fp4 for down
        apk, asf = self.actfp4
        fvk.quantize_bf16_to_nvfp4_swizzled(self.act_pad.data_ptr(), apk.data_ptr(), asf.data_ptr(), NG, FF, s)
        # down GEMM (standard, M=NG) -> dn
        fvk.fp4_w4a16_gemm_sm120_bf16out(apk.data_ptr(), self.Wp[(li, suf, "down")].data_ptr(),
            self.dn[lo:hi].data_ptr(), NG, HID, FF, asf.data_ptr(), self.Wsf[(li, suf, "down")].data_ptr(), 1.0, s)
