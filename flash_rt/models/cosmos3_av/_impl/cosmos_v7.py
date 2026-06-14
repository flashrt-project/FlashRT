#!/usr/bin/env python3
"""V7 — additive over V6. SageAttention2 (QK int8 + V f16) for gen attention.

Replaces the bf16 FA2 gen attention with sage2_qk_int8_sv_f16_bf16_nhd_d128.
d128 sage is non-GQA, so KV (8 heads) is expanded to 32 via a graph-safe strided
copy. APPROXIMATE (int8 QK) — eval-only until E2E action rel-L2 is validated.

Buffers preallocated; expand uses expand()+copy_ (no alloc → CUDA-graph safe).
V2-V6 untouched. Env COSMOS_SAGE=0 falls back to bf16 FA2 (V6 behaviour).
"""
import os, sys, torch
import flash_rt.flash_rt_kernels as fvk
from .cosmos_v6 import CosmosV6
from .cosmos_v2 import H, KV, D, HID, NU, NJ, NG, DEV, BF


class CosmosV7(CosmosV6):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        self.use_sage = os.environ.get("COSMOS_SAGE", "1") == "1"
        # apply sage only on layers [SAGE_LO, SAGE_HI). DEFAULT = [16,36): the
        # validated in-budget config (rel_l2 2.34%, ~baseline; early layers are
        # attention-sensitive and stay bf16). COSMOS_SAGE_LO=0 for full sage.
        self.sage_lo = int(os.environ.get("COSMOS_SAGE_LO", "16"))
        self.sage_hi = int(os.environ.get("COSMOS_SAGE_HI", "36"))
        G = H // KV
        self.Ke = torch.empty(NJ, H, D, device=DEV, dtype=BF)            # GQA-expanded K
        self.Ve16 = torch.empty(1, NJ, H, D, device=DEV, dtype=torch.float16)  # expanded V f16
        self.q8 = torch.empty(1, NG, H, D, dtype=torch.int8, device=DEV)
        self.k8 = torch.empty(1, NJ, H, D, dtype=torch.int8, device=DEV)
        self.qs = torch.empty(1, H, (NG + 31) // 32, dtype=torch.float32, device=DEV)
        self.ks = torch.empty(1, H, (NJ + 63) // 64, dtype=torch.float32, device=DEV)
        self._G = G
        torch.cuda.synchronize()

    def _gen_attn(self, li):
        if not self.use_sage or not (self.sage_lo <= li < self.sage_hi):
            return super()._gen_attn(li)
        s = self._s(); G = self._G
        # GQA expand 8->32 (graph-safe strided copy, dtype-cast for V)
        self.Ke.view(NJ, KV, G, D).copy_(self.Kb[0:NJ].view(NJ, KV, 1, D).expand(NJ, KV, G, D))
        self.Ve16.view(NJ, KV, G, D).copy_(self.Vb[0:NJ].view(NJ, KV, 1, D).expand(NJ, KV, G, D))
        # int8 quant Q (gen), K (joint)
        fvk.quant_per_warp_int8_bf16_d128(self.Qb[NU:NJ].data_ptr(), self.q8.data_ptr(), self.qs.data_ptr(), 1, NG, H, s)
        fvk.quant_per_block_int8_bf16_d128(self.Ke.data_ptr(), self.k8.data_ptr(), self.ks.data_ptr(), 1, NJ, H, s)
        # write attention output directly into attn[NU:NJ] (contiguous [NG,H*D]
        # == [1,NG,H,D]); no sout buffer / copy.
        fvk.sage2_qk_int8_sv_f16_bf16_nhd_d128(
            self.q8.data_ptr(), self.k8.data_ptr(), self.Ve16.data_ptr(),
            self.attn[NU:NJ].data_ptr(), self.qs.data_ptr(), self.ks.data_ptr(),
            1, NG, NJ, H, D ** -0.5, s)
