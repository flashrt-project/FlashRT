#!/usr/bin/env python3
"""Cosmos3-Nano text2video denoise compute path (RTX SM120).

Per docs/adding_new_model.md §0 rule 1, this is the model's own
``models/cosmos3_video/pipeline_<hw>.py``. It is fully self-contained: the
two-tower MoT kernel-call sequence, the per-step CUDA-graph capture, and the
model-local fused qk-norm-rope kernel all live in this package — no dependency
on any other model.

The gen tower carries the all-noisy vision sequence and the head is llm2vae
(-> [N_vis, patch_latent_dim]). The text (und) tower is identical across denoise
steps, so it is computed once and its per-layer K/V cached; each step runs only
the gen (vision) tower against the cached text K/V. qk-norm+rope is fused.

Precision (selected by the frontend, not the environment):
  - quant="fp8"  : w8a8 FP8 E4M3 GEMMs (fp8_gemm_descale_bf16out). Near-lossless
                   for the vision latent and the production default.
  - quant="bf16" : reference-accuracy path.
bf16_projs / bf16_layers keep named projections / layers in bf16. This module
reads no environment variables.
"""
import torch
import torch.nn.functional as F
from safetensors import safe_open

import flash_rt.flash_rt_kernels as fvk
from flash_rt import flash_rt_fa2 as fa2
from .kernels import qk_norm_rope

# Cosmos3-Nano two-tower MoT dimensions (fixed by the architecture).
DEV = "cuda"
EPS = 1e-6
H, KV, D, FF, HID, NL = 32, 8, 128, 12288, 4096, 36
BF = torch.bfloat16
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


class CosmosVideo:
    def __init__(self, weights, nu, ng, quant="fp8", *, bf16_projs=(),
                 bf16_layers=()):
        self._wpath = weights
        self.quant = quant
        if quant == "bf16":
            self.bf16_projs = set(ALL_PROJS)
        else:
            self.bf16_projs = {p for p in bf16_projs if p}
        self.bf16_layers = set(bf16_layers)
        self._bf16_keys = set()
        self.NU, self.NG, self.NJ = nu, ng, nu + ng
        wf = safe_open(weights, "pt", device=DEV)
        T = lambda k: wf.get_tensor(k).to(BF).t().contiguous()
        N = lambda k: wf.get_tensor(k).to(BF)

        def qf8(w_nk):   # per-tensor FP8 E4M3; fp8_gemm_descale wants B as [K,N]
            w = w_nk.t().contiguous()
            s = max(w.float().abs().max().item() / 448.0, 1e-12)
            f8 = (w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            return f8, torch.tensor([s], dtype=torch.float32, device=DEV)

        self.Wt_bf16 = {}; self.Wf8 = {}; self.Wds = {}; self.Wn = {}
        for li in range(NL):
            P = f"language_model.model.layers.{li}."
            for suf in ("", "_moe_gen"):
                bf16_layer = li in self.bf16_layers

                def store(nm, wk):
                    if nm in self.bf16_projs or bf16_layer:
                        self.Wt_bf16[(li, suf, nm)] = T(wk)
                        self._bf16_keys.add((li, suf, nm))
                    else:
                        self.Wf8[(li, suf, nm)], self.Wds[(li, suf, nm)] = qf8(N(wk))

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
        # FP8 activation scratch (per distinct (M,K)) + shared device scale
        self.af8 = {}
        for (M, K) in [(self.NU, HID), (NG, HID), (self.NU, FF), (NG, FF)]:
            self.af8[(M, K)] = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=DEV)
        self.asc = torch.empty(1, dtype=torch.float32, device=DEV)
        # static text K/V cache (text tower is identical across denoise steps)
        self.cK = torch.zeros(NL, self.NU, KV, D, device=DEV, dtype=BF)
        self.cV = torch.zeros(NL, self.NU, KV, D, device=DEV, dtype=BF)
        self._und_ready = False
        self.gr = None
        torch.cuda.synchronize()

    # ---- shared two-tower MoT kernel helpers (self-contained) ----
    def _s(self):
        return torch.cuda.current_stream().cuda_stream

    def _rms(self, x, w, out, rows, dim):
        fvk.rms_norm(x.data_ptr(), w.data_ptr(), out.data_ptr(), rows, dim, EPS, self._s())

    def _radd_rms(self, h, x, w, out, rows):
        fvk.residual_add_rms_norm(h.data_ptr(), x.data_ptr(), w.data_ptr(), out.data_ptr(), rows, HID, EPS, self._s())

    def _silu(self, g, u, out, n):
        fvk.silu_mul_qwen36_bf16(g.data_ptr(), u.data_ptr(), out.data_ptr(), n, self._s())

    def _fa(self, q, k, v, o, nq, nk, causal):
        s = self._s()
        fwd = fa2.fwd_bf16_causal if causal else fa2.fwd_bf16
        qs = (nq * H * D, H * D, D); ks = (nk * KV * D, KV * D, D)   # (batch, seq, head) strides
        fwd(Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=o.data_ptr(), softmax_lse=self.lse.data_ptr(),
            softmax_lse_accum=0, o_accum=0, batch=1, seqlen_q=nq, seqlen_k=nk, num_heads_q=H, num_heads_kv=KV, head_dim=D,
            q_strides=qs, k_strides=ks, v_strides=ks, o_strides=qs,
            softmax_scale=D ** -0.5, num_sms=self.NSM, stream=s)

    def capture(self):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): self.forward()
        torch.cuda.current_stream().wait_stream(st)
        self.gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.gr): self.forward()

    def replay(self):
        self.gr.replay(); return self.vel

    def set_rope(self, cos_und, cos_gen, sin_und, sin_gen):
        self.cos[0:self.NU].copy_(cos_und); self.cos[self.NU:self.NJ].copy_(cos_gen)
        self.sin[0:self.NU].copy_(sin_und); self.sin[self.NU:self.NJ].copy_(sin_gen)

    def set_gen(self, gen_in):
        self.Hb[self.NU:self.NJ].copy_(gen_in)

    def embed_gen(self, vae2llm_in, timestep_emb):
        """gen hidden = vae2llm(patchified noisy latent) + timestep embedding."""
        h = F.linear(vae2llm_in.to(BF), self.Wvae2llm.t(), self.bvae2llm)
        self.set_gen(h + timestep_emb.to(BF))

    def _proj(self, A, key, out, Nn):
        if key in self._bf16_keys:
            M, K = A.shape
            self.gemm.bf16_nn(A.data_ptr(), self.Wt_bf16[key].data_ptr(), out.data_ptr(), M, Nn, K, self._s())
        else:
            M, K = A.shape; s = self._s()
            af8 = self.af8[(M, K)]
            fvk.quantize_fp8_device(A.data_ptr(), af8.data_ptr(), self.asc.data_ptr(), M * K, s)
            fvk.fp8_gemm_descale_bf16out(af8.data_ptr(), self.Wf8[key].data_ptr(), out.data_ptr(),
                                         M, Nn, K, self.asc.data_ptr(), self.Wds[key].data_ptr(), s)

    def _qkv_rope(self, suf, lo, hi, n, li):
        s = self._s()
        self._proj(self.nrm[lo:hi], (li, suf, "q_proj"), self.Qb[lo:hi].view(n, H * D), HID)
        self._proj(self.nrm[lo:hi], (li, suf, "k_proj"), self.Kb[lo:hi].view(n, KV * D), KV * D)
        self._proj(self.nrm[lo:hi], (li, suf, "v_proj"), self.Vb[lo:hi].view(n, KV * D), KV * D)
        qk_norm_rope(self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(),
                     self.Wn[(li, suf, "self_attn.q_norm")].data_ptr(),
                     self.Wn[(li, suf, "self_attn.k_norm")].data_ptr(),
                     self.cos[lo:hi].data_ptr(), self.sin[lo:hi].data_ptr(), n, H, KV, D, EPS, s)

    def _gen_attn(self, li):
        NU, NG, NJ = self.NU, self.NG, self.NJ
        self._fa(self.Qb[NU:NJ], self.Kb[0:NJ], self.Vb[0:NJ], self.attn[NU:NJ].view(NG, H, D), NG, NJ, False)

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
