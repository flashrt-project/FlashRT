#!/usr/bin/env python3
"""Cosmos3-Nano text2video denoise MoT.

Reuses the two-tower MoT kernel path from cosmos_v2 (q/k/v/o/gate/up/down
projections, qk-norm, qwen36 partial rope, und-causal + gen-full attention) for
the video path: the gen tower is the all-noisy vision sequence and the head is
llm2vae (-> [N_vis, patch_latent_dim]). The text (und) tower is identical across
denoise steps, so it is computed once and its per-layer K/V cached (V5 pattern);
each step runs only the gen (vision) tower against the cached text K/V.

Weights are the flat Cosmos3 format (vae2llm/llm2vae present). NVFP4 (MSE) weights
by default; COSMOS_VIDEO_BF16_PROJS keeps named projections in bf16 (the FFN is the
FP4 precision-sensitive site). Conditioning (text/vision encode) is upstream and
consumed from the reference dump, as in the AV path.
"""
import os

import torch

from safetensors import safe_open

import flash_rt.flash_rt_kernels as fvk
from .cosmos_v2 import CosmosV2, sf_bytes, H, KV, D, FF, HID, NL, BF, DEV

W = os.environ["COSMOS_W"]
PATCH = 192  # patch_latent_dim = latent_channels(48) * patch(2) * patch(2)
ALL_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down")


def patchify(x, C=48, p=2):
    """[1,C,T,H,W] -> [T*h*w, p*p*C]  (h=H/p, w=W/p)."""
    x = x[0]
    _, T, Hh, Ww = x.shape
    h, w = Hh // p, Ww // p
    x = x.reshape(C, T, h, p, w, p)
    return torch.einsum("cthpwq->thwpqc", x).reshape(T * h * w, p * p * C)


def unpatchify(v, C, T, h, w, p=2):
    """[T*h*w, p*p*C] -> [1,C,T,h*p,w*p]."""
    v = v.reshape(T, h, w, p, p, C)
    return torch.einsum("thwpqc->cthpwq", v).reshape(1, C, T, h * p, w * p)


class CosmosVideo(CosmosV2):
    def __init__(self, nu, ng, quant="fp4"):
        self.quant = quant
        if quant == "bf16":
            self.bf16_projs = set(ALL_PROJS)
        else:
            keep = os.environ.get("COSMOS_VIDEO_BF16_PROJS", "")
            self.bf16_projs = {p for p in keep.split(",") if p}
        self.NU, self.NG, self.NJ = nu, ng, nu + ng
        wf = safe_open(W, "pt", device=DEV)
        T = lambda k: wf.get_tensor(k).to(BF).t().contiguous()
        N = lambda k: wf.get_tensor(k).to(BF)

        def qz(w_nk):
            n_, k_ = w_nk.shape
            p = torch.empty(n_, k_ // 2, dtype=torch.uint8, device=DEV)
            sf = torch.zeros(sf_bytes(n_, k_), dtype=torch.uint8, device=DEV)
            fvk.quantize_bf16_to_nvfp4_swizzled_mse(
                w_nk.contiguous().data_ptr(), p.data_ptr(), sf.data_ptr(), n_, k_, 0)
            return p, sf

        self.Wt_bf16 = {}
        self.Wp = {}
        self.Wsf = {}
        self.Wn = {}
        for li in range(NL):
            P = f"language_model.model.layers.{li}."
            for suf in ("", "_moe_gen"):
                for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    wk = P + f"self_attn.{nm}{suf}.weight"
                    if nm in self.bf16_projs:
                        self.Wt_bf16[(li, suf, nm)] = T(wk)
                    else:
                        self.Wp[(li, suf, nm)], self.Wsf[(li, suf, nm)] = qz(N(wk))
                for nm, mk in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                    wk = P + f"mlp{suf}.{mk}.weight"
                    if nm in self.bf16_projs:
                        self.Wt_bf16[(li, suf, nm)] = T(wk)
                    else:
                        self.Wp[(li, suf, nm)], self.Wsf[(li, suf, nm)] = qz(N(wk))
                for nm in ("input_layernorm", "post_attention_layernorm",
                           "self_attn.q_norm", "self_attn.k_norm"):
                    self.Wn[(li, suf, nm)] = N(P + f"{nm}{suf}.weight")
        self.norm_g = N("language_model.model.norm_moe_gen.weight")
        self.Wll_vae = T("llm2vae.weight")          # [HID, PATCH]
        self.bll_vae = N("llm2vae.bias")            # [PATCH]
        self.Wvae2llm = T("vae2llm.weight")         # [PATCH, HID]
        self.bvae2llm = N("vae2llm.bias")           # [HID]
        self.gemm = fvk.GemmRunner()
        self.NSM = torch.cuda.get_device_properties(0).multi_processor_count
        z = lambda *s: torch.zeros(*s, device=DEV, dtype=BF)
        NJ = self.NJ
        self.Hb = z(NJ, HID); self.nrm = z(NJ, HID); self.nrm2 = z(NJ, HID)
        self.Qb = z(NJ, H, D); self.Kb = z(NJ, KV, D); self.Vb = z(NJ, KV, D)
        self.attn = z(NJ, H * D); self.ob = z(NJ, HID)
        self.g = z(NJ, FF); self.u = z(NJ, FF); self.act = z(NJ, FF); self.dn = z(NJ, HID)
        self.cos = z(NJ, D); self.sin = z(NJ, D)
        self.vtmp = z(self.NG, PATCH); self.vel = z(self.NG, PATCH)
        self.bll_vaeB = self.bll_vae.unsqueeze(0).expand(self.NG, PATCH).contiguous()
        self.lse = torch.empty(1, H, NJ, dtype=torch.float32, device=DEV)
        self.aq = {}
        for (M, K) in [(self.NU, HID), (self.NG, HID), (self.NU, FF), (self.NG, FF)]:
            self.aq[(M, K)] = (
                torch.empty(M, K // 2, dtype=torch.uint8, device=DEV),
                torch.zeros(sf_bytes(M, K), dtype=torch.uint8, device=DEV))
        self.cK = torch.zeros(NL, self.NU, KV, D, device=DEV, dtype=BF)
        self.cV = torch.zeros(NL, self.NU, KV, D, device=DEV, dtype=BF)
        self._und_ready = False
        self.gr = None
        self.calib = False
        torch.cuda.synchronize()

    def set_rope(self, cos_und, cos_gen, sin_und, sin_gen):
        self.cos[0:self.NU].copy_(cos_und); self.cos[self.NU:self.NJ].copy_(cos_gen)
        self.sin[0:self.NU].copy_(sin_und); self.sin[self.NU:self.NJ].copy_(sin_gen)

    def set_gen(self, gen_in):
        self.Hb[self.NU:self.NJ].copy_(gen_in)

    def embed_gen(self, vae2llm_in, timestep_emb):
        """gen hidden = vae2llm(patchified noisy latent) + timestep embedding."""
        import torch.nn.functional as F
        h = F.linear(vae2llm_in.to(BF), self.Wvae2llm.t(), self.bvae2llm)
        self.set_gen(h + timestep_emb.to(BF))

    def _qkv_rope(self, suf, lo, hi, n, li):
        s = self._s()
        self._proj(self.nrm[lo:hi], (li, suf, "q_proj"), self.Qb[lo:hi].view(n, H * D), HID)
        self._proj(self.nrm[lo:hi], (li, suf, "k_proj"), self.Kb[lo:hi].view(n, KV * D), KV * D)
        self._proj(self.nrm[lo:hi], (li, suf, "v_proj"), self.Vb[lo:hi].view(n, KV * D), KV * D)
        self._rms(self.Qb[lo:hi], self.Wn[(li, suf, "self_attn.q_norm")], self.Qb[lo:hi], n * H, D)
        self._rms(self.Kb[lo:hi], self.Wn[(li, suf, "self_attn.k_norm")], self.Kb[lo:hi], n * KV, D)
        fvk.qwen36_partial_rope_qk_bf16(
            self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(),
            self.cos[lo:hi].data_ptr(), self.sin[lo:hi].data_ptr(),
            self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(), n, H, KV, D, D, s)

    def _o_ffn(self, suf, lo, hi, n, li, last):
        s = self._s()
        self._proj(self.attn[lo:hi], (li, suf, "o_proj"), self.ob[lo:hi], HID)
        self._radd_rms(self.Hb[lo:hi], self.ob[lo:hi], self.Wn[(li, suf, "post_attention_layernorm")], self.nrm2[lo:hi], n)
        self._proj(self.nrm2[lo:hi], (li, suf, "gate"), self.g[lo:hi], FF)
        self._proj(self.nrm2[lo:hi], (li, suf, "up"), self.u[lo:hi], FF)
        self._silu(self.g[lo:hi], self.u[lo:hi], self.act[lo:hi], n * FF)
        self._proj(self.act[lo:hi], (li, suf, "down"), self.dn[lo:hi], HID)
        if not last:
            self._radd_rms(self.Hb[lo:hi], self.dn[lo:hi], self.Wn[(li + 1, suf, "input_layernorm")], self.nrm[lo:hi], n)
        else:
            fvk.residual_add(self.Hb[lo:hi].data_ptr(), self.dn[lo:hi].data_ptr(), n * HID, s)

    def precompute_und(self, und_in):
        NU = self.NU
        self.Hb[0:NU].copy_(und_in)
        self._rms(self.Hb[0:NU], self.Wn[(0, "", "input_layernorm")], self.nrm[0:NU], NU, HID)
        for li in range(NL):
            self._qkv_rope("", 0, NU, NU, li)
            self.cK[li].copy_(self.Kb[0:NU]); self.cV[li].copy_(self.Vb[0:NU])
            self._fa(self.Qb[0:NU], self.Kb[0:NU], self.Vb[0:NU], self.attn[0:NU].view(NU, H, D), NU, NU, True)
            self._o_ffn("", 0, NU, NU, li, li == NL - 1)
        torch.cuda.synchronize()
        self._und_ready = True

    def forward(self):
        s = self._s()
        NU, NG, NJ = self.NU, self.NG, self.NJ
        suf = "_moe_gen"
        self._rms(self.Hb[NU:NJ], self.Wn[(0, suf, "input_layernorm")], self.nrm[NU:NJ], NG, HID)
        for li in range(NL):
            self._qkv_rope(suf, NU, NJ, NG, li)
            self.Kb[0:NU].copy_(self.cK[li]); self.Vb[0:NU].copy_(self.cV[li])
            self._fa(self.Qb[NU:NJ], self.Kb[0:NJ], self.Vb[0:NJ], self.attn[NU:NJ].view(NG, H, D), NG, NJ, False)
            self._o_ffn(suf, NU, NJ, NG, li, li == NL - 1)
        self._rms(self.Hb[NU:NJ], self.norm_g, self.nrm[NU:NJ], NG, HID)
        self.gemm.bf16_nn(self.nrm[NU:NJ].data_ptr(), self.Wll_vae.data_ptr(), self.vtmp.data_ptr(), NG, PATCH, HID, s)
        fvk.add_bf16_out(self.vtmp.data_ptr(), self.bll_vaeB.data_ptr(), self.vel.data_ptr(), NG * PATCH, s)
        return self.vel
