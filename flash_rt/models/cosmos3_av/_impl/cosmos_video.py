#!/usr/bin/env python3
"""Cosmos3-Nano text2video denoise MoT (RTX SM120).

Reuses the two-tower MoT kernel path from cosmos_v2 (q/k/v/o/gate/up/down
projections, qk-norm, qwen36 partial rope, und-causal + gen-full attention) for
the video path: the gen tower is the all-noisy vision sequence and the head is
llm2vae (-> [N_vis, patch_latent_dim]). The text (und) tower is identical across
denoise steps, so it is computed once and its per-layer K/V cached (V5); each step
runs only the gen (vision) tower against the cached text K/V.

Quantization:
  - quant="fp8"  : w8a8 FP8 E4M3 GEMMs (fp8_gemm_descale_bf16out). Near-lossless
                   for the vision latent and the production default.
  - quant="bf16" : reference-accuracy path.
  - quant="fp4"  : NVFP4 (MSE) weights + V12 fp4-direct FFN + optional AWQ; faster
                   but lossy on the vision latent — retained for experiments.
COSMOS_VIDEO_BF16_PROJS / COSMOS_VIDEO_BF16_LAYERS keep named projections / layers
in bf16. int8-sage attention (COSMOS_VIDEO_SAGE) is available but off by default
(int8-QK degrades the vision latent). qk-norm+rope is fused (V6).

Conditioning (text / VAE encode) is upstream and consumed from the reference dump;
this module is the denoise policy. Paths are env/arg-driven (COSMOS_W).
"""
import os

import torch

from safetensors import safe_open

import flash_rt.flash_rt_kernels as fvk
from ..kernels import fp4_silu_aux, sage_gqa_d128, qk_norm_rope
from .cosmos_v2 import CosmosV2, sf_bytes, H, KV, D, FF, HID, NL, EPS, BF, DEV

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


def _parse_set(spec):
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part) if part.isdigit() else part)
    return out


class CosmosVideo(CosmosV2):
    def __init__(self, nu, ng, quant="fp8"):
        self.quant = quant
        if quant == "bf16":
            self.bf16_projs = set(ALL_PROJS)
        else:
            self.bf16_projs = {p for p in os.environ.get("COSMOS_VIDEO_BF16_PROJS", "").split(",") if p}
        self.bf16_layers = _parse_set(os.environ.get("COSMOS_VIDEO_BF16_LAYERS", ""))
        self._bf16_keys = set()
        self.NU, self.NG, self.NJ = nu, ng, nu + ng
        wf = safe_open(W, "pt", device=DEV)
        T = lambda k: wf.get_tensor(k).to(BF).t().contiguous()
        N = lambda k: wf.get_tensor(k).to(BF)

        def qz(w_nk):
            n_, k_ = w_nk.shape
            p = torch.empty(n_, k_ // 2, dtype=torch.uint8, device=DEV)
            sf = torch.zeros(sf_bytes(n_, k_), dtype=torch.uint8, device=DEV)
            fvk.quantize_bf16_to_nvfp4_swizzled_mse(w_nk.contiguous().data_ptr(), p.data_ptr(), sf.data_ptr(), n_, k_, 0)
            return p, sf

        def qf8(w_nk):   # per-tensor FP8 E4M3; fp8_gemm_descale wants B as [K,N]
            w = w_nk.t().contiguous()
            s = max(w.float().abs().max().item() / 448.0, 1e-12)
            f8 = (w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            return f8, torch.tensor([s], dtype=torch.float32, device=DEV)

        self.Wt_bf16 = {}; self.Wp = {}; self.Wsf = {}; self.Wf8 = {}; self.Wds = {}; self.Wn = {}
        for li in range(NL):
            P = f"language_model.model.layers.{li}."
            for suf in ("", "_moe_gen"):
                bf16_layer = li in self.bf16_layers

                def store(nm, wk):
                    if nm in self.bf16_projs or bf16_layer:
                        self.Wt_bf16[(li, suf, nm)] = T(wk)
                        self._bf16_keys.add((li, suf, nm))
                    elif quant == "fp8":
                        self.Wf8[(li, suf, nm)], self.Wds[(li, suf, nm)] = qf8(N(wk))
                    else:
                        self.Wp[(li, suf, nm)], self.Wsf[(li, suf, nm)] = qz(N(wk))

                for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    store(nm, P + f"self_attn.{nm}{suf}.weight")
                for nm, mk in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                    store(nm, P + f"mlp{suf}.{mk}.weight")
                for nm in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
                    self.Wn[(li, suf, nm)] = N(P + f"{nm}{suf}.weight")
        self.norm_g = N("language_model.model.norm_moe_gen.weight")
        self.Wll_vae = T("llm2vae.weight")
        self.bll_vae = N("llm2vae.bias")
        self.Wvae2llm = T("vae2llm.weight")
        self.bvae2llm = N("vae2llm.bias")
        self.gemm = fvk.GemmRunner()
        self.NSM = torch.cuda.get_device_properties(0).multi_processor_count
        z = lambda *s: torch.zeros(*s, device=DEV, dtype=BF)
        NJ, NG = self.NJ, self.NG
        self.Hb = z(NJ, HID); self.nrm = z(NJ, HID); self.nrm2 = z(NJ, HID)
        self.Qb = z(NJ, H, D); self.Kb = z(NJ, KV, D); self.Vb = z(NJ, KV, D)
        self.attn = z(NJ, H * D); self.ob = z(NJ, HID)
        self.g = z(NJ, FF); self.u = z(NJ, FF); self.act = z(NJ, FF); self.dn = z(NJ, HID)
        self.cos = z(NJ, D); self.sin = z(NJ, D)
        self.vtmp = z(NG, PATCH); self.vel = z(NG, PATCH)
        self.bll_vaeB = self.bll_vae.unsqueeze(0).expand(NG, PATCH).contiguous()
        self.lse = torch.empty(1, H, NJ, dtype=torch.float32, device=DEV)
        self.aq = {}; self.af8 = {}
        for (M, K) in [(self.NU, HID), (NG, HID), (self.NU, FF), (NG, FF)]:
            self.aq[(M, K)] = (torch.empty(M, K // 2, dtype=torch.uint8, device=DEV),
                               torch.zeros(sf_bytes(M, K), dtype=torch.uint8, device=DEV))
            self.af8[(M, K)] = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=DEV)
        self.asc = torch.empty(1, dtype=torch.float32, device=DEV)
        # V5 static text K/V cache (text tower is identical across denoise steps)
        self.cK = torch.zeros(NL, self.NU, KV, D, device=DEV, dtype=BF)
        self.cV = torch.zeros(NL, self.NU, KV, D, device=DEV, dtype=BF)
        self._und_ready = False
        # int8-sage gen attention (off by default; int8-QK degrades the vision latent)
        self.use_sage = os.environ.get("COSMOS_VIDEO_SAGE", "0") == "1"
        self.sage_lo = int(os.environ.get("COSMOS_VIDEO_SAGE_LO", "0"))
        self.gq8 = torch.empty(1, NG, H, D, dtype=torch.int8, device=DEV)
        self.gk8 = torch.empty(1, NJ, KV, D, dtype=torch.int8, device=DEV)
        self.gqs = torch.empty(1, H, (NG + 31) // 32, dtype=torch.float32, device=DEV)
        self.gks = torch.empty(1, KV, (NJ + 63) // 64, dtype=torch.float32, device=DEV)
        self.gv16 = torch.empty(1, NJ, KV, D, dtype=torch.float16, device=DEV)
        # V12 fp4-direct FFN for the gen tower (norm_constant lifts silu(gate)*up off
        # the UE4M3 SF floor; M padded to NGP for NVFP4 SF tiling)
        self._NORM = float(os.environ.get("COSMOS_FP4_NORM", "16"))
        self._fp4_ffn = (quant == "fp4" and not ({"gate", "up", "down"} & self.bf16_projs))
        self.NGP = ((NG + 127) // 128) * 128
        self.png = (torch.zeros(self.NGP, HID // 2, dtype=torch.uint8, device=DEV),
                    torch.zeros(sf_bytes(self.NGP, HID), dtype=torch.uint8, device=DEV))
        self.g_pad = torch.zeros(self.NGP, FF, device=DEV, dtype=BF)
        self.actfp4 = (torch.zeros(self.NGP, FF // 2, dtype=torch.uint8, device=DEV),
                       torch.zeros(sf_bytes(self.NGP, FF), dtype=torch.uint8, device=DEV))
        self.gr = None
        self.calib = False
        self.cal_q = {}
        self.cal_g = {}
        torch.cuda.synchronize()

    def apply_awq(self, cal_q, cal_g, alpha):
        """Fold per-input-channel scale s into the gen norm (/=s) and q/k/v/gate/up
        weights (*=s), re-quantize. Math-equivalent; both sides quantize better."""
        wf = safe_open(W, "pt", device=DEV)

        def scale(amax):
            return (amax / amax.mean()).clamp(1e-2, 1e2).pow(alpha).to(BF)

        def qz(w_nk):
            n_, k_ = w_nk.shape
            p = torch.empty(n_, k_ // 2, dtype=torch.uint8, device=DEV)
            sf = torch.zeros(sf_bytes(n_, k_), dtype=torch.uint8, device=DEV)
            fvk.quantize_bf16_to_nvfp4_swizzled_mse(w_nk.contiguous().data_ptr(), p.data_ptr(), sf.data_ptr(), n_, k_, 0)
            return p, sf

        suf = "_moe_gen"
        for li in range(NL):
            P = f"language_model.model.layers.{li}."
            for norm_nm, names, key in (
                ("input_layernorm", ("q_proj", "k_proj", "v_proj"), "self_attn"),
                ("post_attention_layernorm", ("gate", "up"), "mlp")):
                cal = cal_q if key == "self_attn" else cal_g
                s = scale(cal[li])
                self.Wn[(li, suf, norm_nm)] = self.Wn[(li, suf, norm_nm)] * (1.0 / s)
                for nm in names:
                    if (li, suf, nm) in self._bf16_keys:
                        self.Wt_bf16[(li, suf, nm)] = self.Wt_bf16[(li, suf, nm)] * s[:, None]
                        continue
                    mk = nm if key == "self_attn" else {"gate": "gate_proj", "up": "up_proj"}[nm]
                    wk = P + (f"self_attn.{nm}{suf}.weight" if key == "self_attn" else f"mlp{suf}.{mk}.weight")
                    w = wf.get_tensor(wk).to(BF) * s[None, :]
                    self.Wp[(li, suf, nm)], self.Wsf[(li, suf, nm)] = qz(w)
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

    def _proj(self, A, key, out, Nn):
        if key in self._bf16_keys:
            M, K = A.shape
            self.gemm.bf16_nn(A.data_ptr(), self.Wt_bf16[key].data_ptr(), out.data_ptr(), M, Nn, K, self._s())
        elif self.quant == "fp8":
            M, K = A.shape; s = self._s()
            af8 = self.af8[(M, K)]
            fvk.quantize_fp8_device(A.data_ptr(), af8.data_ptr(), self.asc.data_ptr(), M * K, s)
            fvk.fp8_gemm_descale_bf16out(af8.data_ptr(), self.Wf8[key].data_ptr(), out.data_ptr(),
                                         M, Nn, K, self.asc.data_ptr(), self.Wds[key].data_ptr(), s)
        else:
            self._nvfp4(A, key, out, Nn)

    def _qkv_rope(self, suf, lo, hi, n, li):
        s = self._s()
        if self.calib and suf == "_moe_gen":
            self._rec(self.cal_q, li, self.nrm[lo:hi])
        self._proj(self.nrm[lo:hi], (li, suf, "q_proj"), self.Qb[lo:hi].view(n, H * D), HID)
        self._proj(self.nrm[lo:hi], (li, suf, "k_proj"), self.Kb[lo:hi].view(n, KV * D), KV * D)
        self._proj(self.nrm[lo:hi], (li, suf, "v_proj"), self.Vb[lo:hi].view(n, KV * D), KV * D)
        qk_norm_rope(self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(),
                     self.Wn[(li, suf, "self_attn.q_norm")].data_ptr(),
                     self.Wn[(li, suf, "self_attn.k_norm")].data_ptr(),
                     self.cos[lo:hi].data_ptr(), self.sin[lo:hi].data_ptr(), n, H, KV, D, EPS, s)

    def _gen_attn(self, li):
        NU, NG, NJ = self.NU, self.NG, self.NJ
        if not self.use_sage or li < self.sage_lo:
            self._fa(self.Qb[NU:NJ], self.Kb[0:NJ], self.Vb[0:NJ], self.attn[NU:NJ].view(NG, H, D), NG, NJ, False)
            return
        s = self._s()
        fvk.quant_per_warp_int8_bf16_d128(self.Qb[NU:NJ].data_ptr(), self.gq8.data_ptr(), self.gqs.data_ptr(), 1, NG, H, s)
        fvk.quant_per_block_int8_bf16_d128(self.Kb[0:NJ].data_ptr(), self.gk8.data_ptr(), self.gks.data_ptr(), 1, NJ, KV, s)
        self.gv16.view(NJ, KV, D).copy_(self.Vb[0:NJ])
        sage_gqa_d128(self.gq8.data_ptr(), self.gk8.data_ptr(), self.gv16.data_ptr(),
                      self.attn[NU:NJ].data_ptr(), self.gqs.data_ptr(), self.gks.data_ptr(),
                      1, NG, NJ, H, KV, D ** -0.5, s)

    def _ffn_fp4_direct(self, suf, lo, hi, n, li):
        s = self._s()
        pk, sf = self.png
        fvk.quantize_bf16_to_nvfp4_swizzled(self.nrm2[lo:hi].data_ptr(), pk.data_ptr(), sf.data_ptr(), n, HID, s)
        Wg, Wgsf = self.Wp[(li, suf, "gate")], self.Wsf[(li, suf, "gate")]
        Wu, Wusf = self.Wp[(li, suf, "up")], self.Wsf[(li, suf, "up")]
        fvk.fp4_w4a16_gemm_sm120_bf16out(pk.data_ptr(), Wg.data_ptr(), self.g_pad.data_ptr(), self.NGP, FF, HID, sf.data_ptr(), Wgsf.data_ptr(), 1.0, s)
        apk, asf = self.actfp4
        fp4_silu_aux(pk.data_ptr(), sf.data_ptr(), Wu.data_ptr(), Wusf.data_ptr(), self.g_pad.data_ptr(), apk.data_ptr(), asf.data_ptr(), self.NGP, FF, HID, self._NORM, s)
        fvk.fp4_w4a16_gemm_sm120_bf16out(apk.data_ptr(), self.Wp[(li, suf, "down")].data_ptr(), self.dn[lo:hi].data_ptr(), n, HID, FF, asf.data_ptr(), self.Wsf[(li, suf, "down")].data_ptr(), 1.0 / self._NORM, s)

    def _o_ffn(self, suf, lo, hi, n, li, last):
        s = self._s()
        self._proj(self.attn[lo:hi], (li, suf, "o_proj"), self.ob[lo:hi], HID)
        self._radd_rms(self.Hb[lo:hi], self.ob[lo:hi], self.Wn[(li, suf, "post_attention_layernorm")], self.nrm2[lo:hi], n)
        if self.calib and suf == "_moe_gen":
            self._rec(self.cal_g, li, self.nrm2[lo:hi])
        ffn_fp4 = self._fp4_ffn and not any((li, suf, nm) in self._bf16_keys for nm in ("gate", "up", "down"))
        if suf == "_moe_gen" and ffn_fp4:
            self._ffn_fp4_direct(suf, lo, hi, n, li)
        else:
            self._proj(self.nrm2[lo:hi], (li, suf, "gate"), self.g[lo:hi], FF)
            self._proj(self.nrm2[lo:hi], (li, suf, "up"), self.u[lo:hi], FF)
            self._silu(self.g[lo:hi], self.u[lo:hi], self.act[lo:hi], n * FF)
            self._proj(self.act[lo:hi], (li, suf, "down"), self.dn[lo:hi], HID)
        if not last:
            self._radd_rms(self.Hb[lo:hi], self.dn[lo:hi], self.Wn[(li + 1, suf, "input_layernorm")], self.nrm[lo:hi], n)
        else:
            fvk.residual_add(self.Hb[lo:hi].data_ptr(), self.dn[lo:hi].data_ptr(), n * HID, s)

    def precompute_und(self, und_in):
        """One-time exact text tower; snapshot post-rope text K + raw V per layer."""
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
        """Per-step gen (vision) tower only; text K/V from the static cache."""
        s = self._s()
        NU, NG, NJ = self.NU, self.NG, self.NJ
        suf = "_moe_gen"
        self._rms(self.Hb[NU:NJ], self.Wn[(0, suf, "input_layernorm")], self.nrm[NU:NJ], NG, HID)
        for li in range(NL):
            self._qkv_rope(suf, NU, NJ, NG, li)
            self.Kb[0:NU].copy_(self.cK[li]); self.Vb[0:NU].copy_(self.cV[li])
            self._gen_attn(li)
            self._o_ffn(suf, NU, NJ, NG, li, li == NL - 1)
        self._rms(self.Hb[NU:NJ], self.norm_g, self.nrm[NU:NJ], NG, HID)
        self.gemm.bf16_nn(self.nrm[NU:NJ].data_ptr(), self.Wll_vae.data_ptr(), self.vtmp.data_ptr(), NG, PATCH, HID, s)
        fvk.add_bf16_out(self.vtmp.data_ptr(), self.bll_vaeB.data_ptr(), self.vel.data_ptr(), NG * PATCH, s)
        return self.vel
