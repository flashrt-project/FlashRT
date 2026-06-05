"""FP8 (W8A8) decode engine for the Higgs Audio v3 TTS-4B backbone.

Per-tensor FP8 e4m3 weights (amax/448) with static per-tensor activation scales
calibrated from a short BF16 free-run; the math path is fully kernelised:

  rms_norm_fp8 (norm+quant) -> dedicated M=1 FP8 GEMV (warp-per-row, with fused
  residual epilogue on o_proj / down_proj) -> quantize_fp8_static for the
  attention-out / SwiGLU-out points -> silu_mul -> fused q/k-norm+RoPE -> FA2.

Teacher-forced logits reproduce the eager BF16 backbone (cos 1.0); free-run
greedy diverges from any other implementation by bf16 near-tie noise (an
intrinsic property of greedy discrete-code AR, not a quantisation error).

Used by :class:`HiggsAudioV3TorchFrontendRtx` when ``fp8=True``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

BF16, F8 = torch.bfloat16, torch.float8_e4m3fn

# Per-shape M=1 GEMV kernel selection (warp count tuned per shape; o/down use
# the fused-residual epilogue so the residual add folds into the GEMV).
_GEMV = {
    "qkv": "ht_gemv_fp8_m1_w8", "o": "ht_gemv_fp8_m1_resadd_w8",
    "gu": "ht_gemv_fp8_m1_w4", "dn": "ht_gemv_fp8_m1_resadd_w4",
    "head": "ht_gemv_fp8_m1_w4",
}


def _qw(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """bf16 weight -> (fp8 e4m3 [N,K], per-tensor scale = amax/448)."""
    sc = (w.float().abs().amax() / 448).item()
    return (w.float() / sc).clamp(-448, 448).to(F8).contiguous(), sc


class HiggsAudioV3Fp8Decoder:
    """Holds FP8 weights + calibrated scales + scratch; runs the FP8 decode."""

    def __init__(self, frontend: Any) -> None:
        self.fe = frontend
        self.dev = frontend.device
        c = frontend._cfg
        self.H, self.NQ, self.NKV, self.HD = (
            c["hidden"], c["num_q_heads"], c["num_kv_heads"], c["head_dim"])
        self.INTER, self.EPS = c["intermediate"], c["rms_norm_eps"]
        self.NL = c["num_layers"]
        self.NC, self.CV = c["num_codebooks"], c["codebook_vocab"]
        self.NQK = self.NQ * self.HD
        self.KV = self.NKV * self.HD
        self._quantize_weights()
        self._calibrated = False

    # ── one-time setup ──

    def _quantize_weights(self) -> None:
        w = self.fe._weights
        self.WL = []
        for L in w["layers"]:
            qkvq, qkvs = _qw(torch.cat([L["q"], L["k"], L["v"]], 0))
            guq, gus = _qw(torch.cat([L["gate"], L["up"]], 0))
            oq, os_ = _qw(L["o"])
            dnq, dns = _qw(L["down"])
            self.WL.append(dict(
                qkvq=qkvq, qkvs=qkvs, oq=oq, os=os_, guq=guq, gus=gus,
                dnq=dnq, dns=dns, in_norm=L["in_norm"], post_norm=L["post_norm"],
                qn=L["qn"], kn=L["kn"]))
        self.CBq, self.CBs = _qw(w["codebook"])

    def _alloc(self) -> None:
        d = self.dev
        self.fp8a = torch.empty(1, self.H, device=d, dtype=F8)
        self.fp8o = torch.empty(1, self.NQK, device=d, dtype=F8)
        self.fp8d = torch.empty(1, self.INTER, device=d, dtype=F8)
        self.Dq = torch.empty(1, self.NQK + 2 * self.KV, device=d, dtype=BF16)
        self.Dg = torch.empty(1, 2 * self.INTER, device=d, dtype=BF16)
        self.Dh = torch.empty(1, self.NC * self.CV, device=d, dtype=BF16)
        self.act = torch.empty(1, self.INTER, device=d, dtype=BF16)
        self.h = torch.empty(1, self.H, device=d, dtype=BF16)
        self.hn = torch.empty(1, self.H, device=d, dtype=BF16)

    # ── calibration: amax of the 4 quant points/layer + final, from a short
    #    BF16 free-run of a built-in sentence (activation ranges are stable). ──

    @torch.no_grad()
    def calibrate(self, prompt_ids: list[int], n_frames: int = 24) -> None:
        fe = self.fe
        c = fe._cfg
        H, NQ, NKV, HD, INTER = self.H, self.NQ, self.NKV, self.HD, self.INTER
        NQK, KV = self.NQK, self.KV
        be = fe._attn
        be.reset_cache()
        am = torch.zeros(self.NL, 4, device=self.dev)
        amf = torch.zeros(1, device=self.dev)
        te = fe._weights["text_embed"]
        layers = fe._weights["layers"]
        import math

        def rms(x, w):
            o = torch.empty(1, H, device=self.dev, dtype=BF16)
            from flash_rt import flash_rt_kernels as fvk
            fvk.rms_norm(x.contiguous().data_ptr(), w.data_ptr(), o.data_ptr(),
                         1, H, self.EPS, torch.cuda.current_stream().cuda_stream)
            return o

        def step(h, t):
            for L in range(self.NL):
                w = layers[L]
                xn = rms(h, w["in_norm"])
                am[L, 0] = torch.maximum(am[L, 0], xn.float().abs().amax())
                qkv = xn.float() @ torch.cat([w["q"], w["k"], w["v"]], 0).float().t()
                q = qkv[:, :NQK].view(NQ, HD)
                k = qkv[:, NQK:NQK + KV].view(NKV, HD)
                v = qkv[:, NQK + KV:].view(NKV, HD)
                be.K_cache[L, t:t + 1].copy_(k.to(BF16).view(1, NKV, HD))
                be.V_cache[L, t:t + 1].copy_(v.to(BF16).view(1, NKV, HD))
                Kc = be.K_cache[L, :t + 1].float()
                Vc = be.V_cache[L, :t + 1].float()
                krep = Kc.repeat_interleave(NQ // NKV, 1).view(t + 1, NQ, HD)
                vrep = Vc.repeat_interleave(NQ // NKV, 1).view(t + 1, NQ, HD)
                sc = (q.float().unsqueeze(0) * krep).sum(-1) / math.sqrt(HD)
                ao = (torch.softmax(sc, 0).unsqueeze(-1) * vrep).sum(0).reshape(1, NQK)
                am[L, 1] = torch.maximum(am[L, 1], ao.abs().amax())
                h = h.float() + ao @ w["o"].float().t()
                xn2 = rms(h.to(BF16), w["post_norm"])
                am[L, 2] = torch.maximum(am[L, 2], xn2.float().abs().amax())
                gu = xn2.float() @ torch.cat([w["gate"], w["up"]], 0).float().t()
                act = F.silu(gu[:, :INTER]) * gu[:, INTER:]
                am[L, 3] = torch.maximum(am[L, 3], act.abs().amax())
                h = (h + act @ w["down"].float().t()).to(BF16)
            amf[0] = torch.maximum(amf[0], rms(h, fe._weights["final_norm"]).float().abs().amax())
            return h

        for t, tok in enumerate(prompt_ids):
            h = step(F.embedding(torch.tensor([tok], device=self.dev), te), t)
        P = len(prompt_ids)
        for j in range(n_frames):
            logits = (rms(h, fe._weights["final_norm"]).float()
                      @ fe._weights["codebook"].float().t()).view(self.NC, self.CV)
            codes = logits.argmax(-1)
            emb = F.embedding(codes.long() + fe._cb_offsets,
                              fe._weights["codebook"]).sum(0, keepdim=True)
            h = step(emb, P + j)

        self.asc = (am / 448)
        self.asc_f = (amf / 448).item()
        self.DS = [[torch.tensor([max(self.asc[L, i].item(), 1e-9)], device=self.dev,
                    dtype=torch.float32) for i in range(4)] for L in range(self.NL)]
        self.DSF = torch.tensor([max(self.asc_f, 1e-9)], device=self.dev, dtype=torch.float32)
        self.ALP = [[self.asc[L, i].item() for i in range(4)] for L in range(self.NL)]
        self._alloc()
        self._calibrated = True

    # ── FP8 decode step (eager, fully kernelised) ──

    @torch.no_grad()
    def step(self, t: int) -> torch.Tensor:
        from flash_rt import flash_rt_kernels as fvk
        fe = self.fe
        be = fe._attn
        H, NQ, NKV, HD, INTER = self.H, self.NQ, self.NKV, self.HD, self.INTER
        NQK, KV, EPS = self.NQK, self.KV, self.EPS
        gv = {k: getattr(fvk, v) for k, v in _GEMV.items()}
        s = torch.cuda.current_stream().cuda_stream
        h, fp8a, fp8o, fp8d = self.h, self.fp8a, self.fp8o, self.fp8d
        for L in range(self.NL):
            w = self.WL[L]
            fvk.rms_norm_fp8(h.data_ptr(), w["in_norm"].data_ptr(), fp8a.data_ptr(),
                             1, H, EPS, self.DS[L][0].data_ptr(), s)
            gv["qkv"](fp8a.data_ptr(), w["qkvq"].data_ptr(), self.Dq.data_ptr(),
                      1, NQK + 2 * KV, H, self.ALP[L][0] * w["qkvs"], s)
            q = self.Dq[:, :NQK].view(NQ, HD).contiguous()
            k = self.Dq[:, NQK:NQK + KV].view(NKV, HD).contiguous()
            v = self.Dq[:, NQK + KV:].view(NKV, HD).contiguous()
            cos_t, sin_t = fe._rope_cos[t], fe._rope_sin[t]
            fvk.qwen3_q_norm_rope_qstage_bf16(
                q_pre=q.data_ptr(), q_norm_w=w["qn"].data_ptr(),
                cos=cos_t.data_ptr(), sin=sin_t.data_ptr(),
                q_buf_dst=be.Q_buf[:, :1].data_ptr(), n_q_heads=NQ, eps=EPS, stream=s)
            fvk.qwen3_k_norm_rope_kvwrite_bf16(
                k_pre=k.data_ptr(), v_pre=v.data_ptr(), k_norm_w=w["kn"].data_ptr(),
                cos=cos_t.data_ptr(), sin=sin_t.data_ptr(),
                k_cache_dst=be.K_cache[L, t:t + 1].data_ptr(),
                v_cache_dst=be.V_cache[L, t:t + 1].data_ptr(),
                n_kv_heads=NKV, eps=EPS, stream=s)
            kv = t + 1
            qb, kc = be.Q_buf[:, :1], be.K_cache[L:L + 1, :kv]
            vc, ob = be.V_cache[L:L + 1, :kv], be.O_buf[:, :1]
            be._fa2_fwd(
                Q=qb.data_ptr(), K=kc.data_ptr(), V=vc.data_ptr(), O=ob.data_ptr(),
                softmax_lse=be.lse_buf.data_ptr(), softmax_lse_accum=0, o_accum=0,
                batch=1, seqlen_q=1, seqlen_k=kv, num_heads_q=NQ, num_heads_kv=NKV,
                head_dim=HD, q_strides=(qb.stride(0), qb.stride(1), qb.stride(2)),
                k_strides=(kc.stride(0), kc.stride(1), kc.stride(2)),
                v_strides=(vc.stride(0), vc.stride(1), vc.stride(2)),
                o_strides=(ob.stride(0), ob.stride(1), ob.stride(2)),
                softmax_scale=HD ** -0.5, num_sms=be._num_sms, stream=s)
            ao = be.O_buf[:, :1].reshape(1, NQK).contiguous()
            fvk.quantize_fp8_static(ao.data_ptr(), fp8o.data_ptr(), self.DS[L][1].data_ptr(), NQK, s)
            gv["o"](fp8o.data_ptr(), w["oq"].data_ptr(), h.data_ptr(),
                    1, H, NQK, self.ALP[L][1] * w["os"], s)
            fvk.rms_norm_fp8(h.data_ptr(), w["post_norm"].data_ptr(), fp8a.data_ptr(),
                             1, H, EPS, self.DS[L][2].data_ptr(), s)
            gv["gu"](fp8a.data_ptr(), w["guq"].data_ptr(), self.Dg.data_ptr(),
                     1, 2 * INTER, H, self.ALP[L][2] * w["gus"], s)
            fvk.silu_mul_qwen36_bf16(self.Dg[:, :INTER].contiguous().data_ptr(),
                                     self.Dg[:, INTER:].contiguous().data_ptr(),
                                     self.act.data_ptr(), INTER, s)
            fvk.quantize_fp8_static(self.act.data_ptr(), fp8d.data_ptr(), self.DS[L][3].data_ptr(), INTER, s)
            gv["dn"](fp8d.data_ptr(), w["dnq"].data_ptr(), h.data_ptr(),
                     1, H, INTER, self.ALP[L][3] * w["dns"], s)
        fvk.rms_norm(h.data_ptr(), fe._weights["final_norm"].data_ptr(),
                     self.hn.data_ptr(), 1, H, EPS, s)
        fvk.quantize_fp8_static(self.hn.data_ptr(), fp8a.data_ptr(), self.DSF.data_ptr(), H, s)
        gv["head"](fp8a.data_ptr(), self.CBq.data_ptr(), self.Dh.data_ptr(),
                   1, self.NC * self.CV, H, self.asc_f * self.CBs, s)
        return self.Dh.view(1, self.NC, self.CV)

    def set_input(self, embed_row: torch.Tensor) -> None:
        self.h.copy_(embed_row)
