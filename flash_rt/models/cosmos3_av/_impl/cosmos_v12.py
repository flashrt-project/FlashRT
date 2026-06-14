#!/usr/bin/env python3
"""V12 — all-cutlass fp4-DIRECT FFN-mega (no re-quant, no bf16 act materialize).

Over V10: replaces the up-leg's [silu_aux bf16-out + separate quantize] with ONE
cutlass kernel (fp4_silu_aux fp4-out) that does up-GEMM + silu(gate)*up + per-block
NVFP4 quant -> fp4 + swizzled UE4M3 SF, fed straight to the down GEMM. Uses the
cutlass 89% mainloop (KernelTmaWarpSpecializedPingpong) + Sm120BlockScaleFactorRowStore
epilogue. norm_constant=_NORM is the NVFP4 global scale that lifts the tiny
silu(gate)*up blocks off the UE4M3 SF floor (else cos craters on the many ~0 blocks).

Additive: subclasses V10, overrides _ffn_updown. gate/up/down weights swizzled
(inherited). Env COSMOS_FP4_DIRECT=0 falls back to V10; COSMOS_FP4_NORM sets the scale.
"""
import os
import torch
from .cosmos_v10 import CosmosV10, NGP
from .cosmos_v2 import sf_bytes, FF, HID, NG, DEV
import flash_rt.flash_rt_kernels as fvk
from ..kernels import fp4_silu_aux

_NORM = float(os.environ.get("COSMOS_FP4_NORM", "16"))   # best precision (2.355%)


class CosmosV12(CosmosV10):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        # fp4-direct act output (NGP rows; swizzle tiles match NG since both -> 5x128)
        self.actfp4 = (torch.empty(NGP, FF // 2, dtype=torch.uint8, device=DEV),
                       torch.zeros(sf_bytes(NGP, FF), dtype=torch.uint8, device=DEV))
        self.fp4_direct = os.environ.get("COSMOS_FP4_DIRECT", "1") == "1"
        torch.cuda.synchronize()

    def _ffn_updown(self, lo, hi, n, li, suf, pk, sf):
        if not self.fp4_direct or n != NG:
            return super()._ffn_updown(lo, hi, n, li, suf, pk, sf)
        s = self._s()
        Wg, Wgsf = self.Wp[(li, suf, "gate")], self.Wsf[(li, suf, "gate")]
        Wu, Wusf = self.Wp[(li, suf, "up")], self.Wsf[(li, suf, "up")]
        # gate GEMM (M=NGP) -> g_pad bf16
        fvk.fp4_w4a16_gemm_sm120_bf16out(pk.data_ptr(), Wg.data_ptr(), self.g_pad.data_ptr(),
            NGP, FF, HID, sf.data_ptr(), Wgsf.data_ptr(), 1.0, s)
        # up GEMM + silu(gate)*up + NVFP4 quant -> fp4 + swizzled SF (DIRECT, M=NGP)
        apk, asf = self.actfp4
        fp4_silu_aux(pk.data_ptr(), sf.data_ptr(), Wu.data_ptr(), Wusf.data_ptr(),
            self.g_pad.data_ptr(), apk.data_ptr(), asf.data_ptr(), NGP, FF, HID, _NORM, s)
        # down GEMM (M=NG) reads fp4 act directly; divide out the global scale
        # (norm_constant scales the stored act by _NORM, so descale by 1/_NORM here)
        fvk.fp4_w4a16_gemm_sm120_bf16out(apk.data_ptr(), self.Wp[(li, suf, "down")].data_ptr(),
            self.dn[lo:hi].data_ptr(), NG, HID, FF, asf.data_ptr(),
            self.Wsf[(li, suf, "down")].data_ptr(), 1.0 / _NORM, s)
