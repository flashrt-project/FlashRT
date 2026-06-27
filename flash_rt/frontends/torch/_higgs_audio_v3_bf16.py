"""BF16 (W16A16) decode engine for the Higgs Audio v3 TTS-4B backbone.

The all-BF16 sibling of :class:`HiggsAudioV3Fp8Decoder` for hosts that cannot run
FP8 — same fully-kernelised, zero-torch math path and the same position-agnostic
CUDA graph + batched prefill + prefix-reuse surface, at BF16 weight bandwidth (no
quantisation). Extreme fusion: the residual add folds into the *following*
RMSNorm via the single ``residual_add_rms_norm`` kernel (in-place ``h += x`` in
fp32, then ``out = rmsnorm(h)·w``) at both the attention-out and FFN-out
boundaries, including the cross-layer boundary (down-proj residual folds into the
next layer's input norm). Per layer: 4 GEMVs (qkv, o, gate/up, down) + 2 fused
residual-norms + fused q/k-norm+RoPE+KV-write + FA2 + silu_mul; nothing else.

GEMV kernel is the dedicated M=1 ``bf16_matvec_qwen36_bf16`` (warp-per-output-row,
no MMA BLOCK_M padding tax), selected over the tiled ``bf16_matmul`` by measured
full-decode latency. Prefill uses the tiled ``bf16_matmul_bf16`` at M=P.

Used by :class:`HiggsAudioV3TorchFrontendRtx` when ``fp8=False``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16


class HiggsAudioV3Bf16Decoder:
    """Holds concatenated BF16 weights + scratch; runs the BF16 decode."""

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
        self._prepare_weights()
        self._alloc()

    # ── one-time setup ──

    def _prepare_weights(self) -> None:
        """Concatenate q/k/v and gate/up into single GEMV weights (one launch,
        better N-tiling), keep o/down/norms, and free the separate projections."""
        w = self.fe._weights
        self.WL = []
        for L in w["layers"]:
            qkv = torch.cat([L["q"], L["k"], L["v"]], 0).contiguous()
            gu = torch.cat([L["gate"], L["up"]], 0).contiguous()
            self.WL.append(dict(
                qkv=qkv, gu=gu, o=L["o"], down=L["down"],
                in_norm=L["in_norm"], post_norm=L["post_norm"],
                qn=L["qn"], kn=L["kn"]))
        for L in w["layers"]:           # reclaim the separate projections
            for k in ("q", "k", "v", "gate", "up"):
                L.pop(k, None)
        self.CB = w["codebook"]
        self.final_norm = w["final_norm"]
        torch.cuda.empty_cache()

    # Dedicated M=1 BF16 GEMV — 86% HBM BW (2.1x the generic bf16_matvec/matmul
    # at 40%; the generic kernels stage x in smem, 2x the bytes of FP8, capping
    # blocks/SM on K=9728). Warp count w4 measured fastest on the decode shapes.
    _GEMV = "ht_gemv_bf16_m1_w4"

    def _alloc(self) -> None:
        d = self.dev
        self.Dq = torch.empty(1, self.NQK + 2 * self.KV, device=d, dtype=BF16)
        self.Dg = torch.empty(1, 2 * self.INTER, device=d, dtype=BF16)
        self.Dh = torch.empty(1, self.NC * self.CV, device=d, dtype=BF16)
        self.act = torch.empty(1, self.INTER, device=d, dtype=BF16)
        self.h = torch.empty(1, self.H, device=d, dtype=BF16)
        self.xn = torch.empty(1, self.H, device=d, dtype=BF16)
        self.xn2 = torch.empty(1, self.H, device=d, dtype=BF16)
        self.otmp = torch.empty(1, self.H, device=d, dtype=BF16)
        self.dtmp = torch.empty(1, self.H, device=d, dtype=BF16)
        self.hn = torch.empty(1, self.H, device=d, dtype=BF16)

    # ── BF16 decode step (eager, fully kernelised, extreme fusion) ──

    @torch.no_grad()
    def step(self, t: int) -> torch.Tensor:
        from flash_rt import flash_rt_kernels as fvk
        fe, be = self.fe, self.fe._attn
        H, NQ, NKV, HD, INTER = self.H, self.NQ, self.NKV, self.HD, self.INTER
        NQK, KV, EPS = self.NQK, self.KV, self.EPS
        gv = getattr(fvk, self._GEMV)

        def mv(xp, wp, op, N, K, st):
            gv(xp, wp, op, 1, N, K, 1.0, st)
        s = torch.cuda.current_stream().cuda_stream
        h, xn, xn2 = self.h, self.xn, self.xn2
        otmp, dtmp = self.otmp, self.dtmp
        fvk.rms_norm(h.data_ptr(), self.WL[0]["in_norm"].data_ptr(),
                     xn.data_ptr(), 1, H, EPS, s)
        for L in range(self.NL):
            w = self.WL[L]
            mv(xn.data_ptr(), w["qkv"].data_ptr(), self.Dq.data_ptr(),
               NQK + 2 * KV, H, s)
            q = self.Dq[:, :NQK].view(NQ, HD)
            k = self.Dq[:, NQK:NQK + KV].view(NKV, HD)
            v = self.Dq[:, NQK + KV:].view(NKV, HD)
            cos_t, sin_t = fe._rope_cos[t], fe._rope_sin[t]
            fvk.qwen3_q_norm_rope_qstage_bf16(
                q_pre=q.data_ptr(), q_norm_w=w["qn"].data_ptr(),
                cos=cos_t.data_ptr(), sin=sin_t.data_ptr(),
                q_buf_dst=be.Q_buf[:, :1].data_ptr(), n_q_heads=NQ, eps=EPS,
                stream=s)
            fvk.qwen3_k_norm_rope_kvwrite_bf16(
                k_pre=k.data_ptr(), v_pre=v.data_ptr(),
                k_norm_w=w["kn"].data_ptr(),
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
            ao = be.O_buf[:, :1].reshape(1, NQK)
            mv(ao.data_ptr(), w["o"].data_ptr(), otmp.data_ptr(), H, NQK, s)
            # fused: h += o  (fp32 accum, in-place), xn2 = rmsnorm(h)·post_norm
            fvk.residual_add_rms_norm(h.data_ptr(), otmp.data_ptr(),
                                      w["post_norm"].data_ptr(), xn2.data_ptr(),
                                      1, H, EPS, s)
            mv(xn2.data_ptr(), w["gu"].data_ptr(), self.Dg.data_ptr(),
               2 * INTER, H, s)
            fvk.silu_mul_qwen36_bf16(self.Dg[:, :INTER].contiguous().data_ptr(),
                                     self.Dg[:, INTER:].contiguous().data_ptr(),
                                     self.act.data_ptr(), INTER, s)
            mv(self.act.data_ptr(), w["down"].data_ptr(), dtmp.data_ptr(),
               H, INTER, s)
            if L < self.NL - 1:
                # fused: h += down, xn = rmsnorm(h)·next-layer in_norm
                fvk.residual_add_rms_norm(
                    h.data_ptr(), dtmp.data_ptr(),
                    self.WL[L + 1]["in_norm"].data_ptr(), xn.data_ptr(),
                    1, H, EPS, s)
            else:
                fvk.residual_add(h.data_ptr(), dtmp.data_ptr(), H, s)
                fvk.rms_norm(h.data_ptr(), self.final_norm.data_ptr(),
                             self.hn.data_ptr(), 1, H, EPS, s)
        mv(self.hn.data_ptr(), self.CB.data_ptr(), self.Dh.data_ptr(),
           self.NC * self.CV, H, s)
        return self.Dh.view(1, self.NC, self.CV)

    def set_input(self, embed_row: torch.Tensor) -> None:
        self.h.copy_(embed_row)

    # ── batched prompt prefill (single M=S forward, optional cached prefix) ──
    # Mirrors the FP8 path: one M=S pass over ids[start_pos:] (tiled BF16 GEMM),
    # per-position fused q/k-norm+RoPE+KV-write, one causal FA2, residual+norm
    # fused via residual_add_rms_norm. ``start_pos`` reuses a resident prefix's
    # KV (serving prefix caching) — bit-identical to a full prefill.

    @torch.no_grad()
    def prefill_batched(self, ids: list[int], start_pos: int = 0) -> torch.Tensor:
        from flash_rt import flash_rt_kernels as fvk
        fe, be = self.fe, self.fe._attn
        dev = self.dev
        P = len(ids)
        S = P - start_pos
        H, NQ, NKV, HD, INTER = self.H, self.NQ, self.NKV, self.HD, self.INTER
        NQK, KV, EPS = self.NQK, self.KV, self.EPS
        s = torch.cuda.current_stream().cuda_stream
        # One-time eager prefill GEMM: cuBLAS (torch.matmul) reads each weight
        # ONCE across the P prompt rows (5-14x the warp-per-row bf16_matmul,
        # which re-reads W per row -> 11%/5% effective BW at P=13/33). This is
        # the one-time setup path, not the per-frame decode graph (which stays
        # the hand-written bf16 GEMV); the norms/qk-norm+RoPE/silu/attention all
        # stay kernelised. mv is the M=1 head GEMV.
        mv = fvk.bf16_matvec_qwen36_bf16
        rc, rs = fe._rope_cos, fe._rope_sin

        ids_t = torch.tensor(ids[start_pos:], device=dev)
        h = F.embedding(ids_t, fe._weights["text_embed"]).contiguous()  # [S,H]
        xn = torch.empty(S, H, device=dev, dtype=BF16)
        xn2 = torch.empty(S, H, device=dev, dtype=BF16)
        Dq = torch.empty(S, NQK + 2 * KV, device=dev, dtype=BF16)
        Dg = torch.empty(S, 2 * INTER, device=dev, dtype=BF16)
        act = torch.empty(S, INTER, device=dev, dtype=BF16)
        tmp = torch.empty(S, H, device=dev, dtype=BF16)

        fvk.rms_norm(h.data_ptr(), self.WL[0]["in_norm"].data_ptr(),
                     xn.data_ptr(), S, H, EPS, s)
        for L in range(self.NL):
            w = self.WL[L]
            torch.matmul(xn, w["qkv"].t(), out=Dq)
            for j in range(S):
                pos = start_pos + j
                qj = Dq[j, :NQK].contiguous()
                kj = Dq[j, NQK:NQK + KV].contiguous()
                vj = Dq[j, NQK + KV:].contiguous()
                fvk.qwen3_q_norm_rope_qstage_bf16(
                    q_pre=qj.data_ptr(), q_norm_w=w["qn"].data_ptr(),
                    cos=rc[pos].data_ptr(), sin=rs[pos].data_ptr(),
                    q_buf_dst=be.Q_buf[:, j].data_ptr(), n_q_heads=NQ, eps=EPS,
                    stream=s)
                fvk.qwen3_k_norm_rope_kvwrite_bf16(
                    k_pre=kj.data_ptr(), v_pre=vj.data_ptr(),
                    k_norm_w=w["kn"].data_ptr(),
                    cos=rc[pos].data_ptr(), sin=rs[pos].data_ptr(),
                    k_cache_dst=be.K_cache[L, pos].data_ptr(),
                    v_cache_dst=be.V_cache[L, pos].data_ptr(),
                    n_kv_heads=NKV, eps=EPS, stream=s)
            be.O_buf[:, :S].zero_()
            be.run("full", L, S, kv_seq=P, causal=True, stream=s,
                   softmax_scale=HD ** -0.5)
            ao = be.O_buf[:, :S].reshape(S, NQK).contiguous()
            torch.matmul(ao, w["o"].t(), out=tmp)
            fvk.residual_add_rms_norm(h.data_ptr(), tmp.data_ptr(),
                                      w["post_norm"].data_ptr(), xn2.data_ptr(),
                                      S, H, EPS, s)
            torch.matmul(xn2, w["gu"].t(), out=Dg)
            g = Dg[:, :INTER].contiguous()
            u = Dg[:, INTER:].contiguous()
            fvk.silu_mul_qwen36_bf16(g.data_ptr(), u.data_ptr(), act.data_ptr(),
                                     S * INTER, s)
            torch.matmul(act, w["down"].t(), out=tmp)
            if L < self.NL - 1:
                fvk.residual_add_rms_norm(
                    h.data_ptr(), tmp.data_ptr(),
                    self.WL[L + 1]["in_norm"].data_ptr(), xn.data_ptr(),
                    S, H, EPS, s)
            else:
                fvk.residual_add(h.data_ptr(), tmp.data_ptr(), S * H, s)

        hlast = h[S - 1:S].contiguous()
        fvk.rms_norm(hlast.data_ptr(), self.final_norm.data_ptr(),
                     self.hn.data_ptr(), 1, H, EPS, s)
        mv(self.hn.data_ptr(), self.CB.data_ptr(), self.Dh.data_ptr(),
           self.NC * self.CV, H, s)
        return self.Dh.view(1, self.NC, self.CV)

    # ── position-agnostic single decode graph ──

    def _alloc_graph(self) -> None:
        d, HALF = self.dev, self.HD // 2
        self.rope_cos_buf = torch.empty(HALF, device=d, dtype=BF16)
        self.rope_sin_buf = torch.empty(HALF, device=d, dtype=BF16)
        self.cur_pos_dev = torch.zeros(1, device=d, dtype=torch.int32)
        self.seqused_dev = torch.zeros(1, device=d, dtype=torch.int32)
        self._graph = None
        self._gs = torch.cuda.Stream(device=d)

    @torch.no_grad()
    def _step_graphable(self):
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt import flash_rt_fa2 as fa2
        fe, be = self.fe, self.fe._attn
        H, NQ, NKV, HD, INTER = self.H, self.NQ, self.NKV, self.HD, self.INTER
        NQK, KV, EPS = self.NQK, self.KV, self.EPS
        MAXS = be._max_seq
        gv = getattr(fvk, self._GEMV)

        def mv(xp, wp, op, N, K, st):
            gv(xp, wp, op, 1, N, K, 1.0, st)
        s = torch.cuda.current_stream().cuda_stream
        rc, rs = self.rope_cos_buf, self.rope_sin_buf
        cp, su = self.cur_pos_dev, self.seqused_dev
        h, xn, xn2 = self.h, self.xn, self.xn2
        otmp, dtmp = self.otmp, self.dtmp
        qb, ob = be.Q_buf[:, :1], be.O_buf[:, :1]
        qst = (qb.stride(0), qb.stride(1), qb.stride(2))
        ost = (ob.stride(0), ob.stride(1), ob.stride(2))
        fvk.rms_norm(h.data_ptr(), self.WL[0]["in_norm"].data_ptr(),
                     xn.data_ptr(), 1, H, EPS, s)
        for L in range(self.NL):
            w = self.WL[L]
            mv(xn.data_ptr(), w["qkv"].data_ptr(), self.Dq.data_ptr(),
               NQK + 2 * KV, H, s)
            q = self.Dq[:, :NQK].view(NQ, HD)
            k = self.Dq[:, NQK:NQK + KV].view(NKV, HD)
            v = self.Dq[:, NQK + KV:].view(NKV, HD)
            fvk.qwen3_q_norm_rope_qstage_bf16(
                q_pre=q.data_ptr(), q_norm_w=w["qn"].data_ptr(),
                cos=rc.data_ptr(), sin=rs.data_ptr(),
                q_buf_dst=qb.data_ptr(), n_q_heads=NQ, eps=EPS, stream=s)
            fvk.qwen3_k_norm_rope_kvwrite_devpos_bf16(
                k.data_ptr(), v.data_ptr(), w["kn"].data_ptr(),
                rc.data_ptr(), rs.data_ptr(),
                be.K_cache[L, 0].data_ptr(), be.V_cache[L, 0].data_ptr(),
                cp.data_ptr(), NKV * HD, NKV, EPS, s)
            kf, vf = be.K_cache[L:L + 1, :MAXS], be.V_cache[L:L + 1, :MAXS]
            fa2.fwd_bf16_seqused(
                Q=qb.data_ptr(), K=kf.data_ptr(), V=vf.data_ptr(), O=ob.data_ptr(),
                softmax_lse=be.lse_buf.data_ptr(), seqused_k=su.data_ptr(),
                batch=1, seqlen_q=1, seqlen_k=MAXS, num_heads_q=NQ,
                num_heads_kv=NKV, head_dim=HD, q_strides=qst,
                k_strides=(kf.stride(0), kf.stride(1), kf.stride(2)),
                v_strides=(vf.stride(0), vf.stride(1), vf.stride(2)),
                o_strides=ost, softmax_scale=HD ** -0.5, num_sms=0, stream=s)
            ao = be.O_buf[:, :1].reshape(1, NQK)
            mv(ao.data_ptr(), w["o"].data_ptr(), otmp.data_ptr(), H, NQK, s)
            fvk.residual_add_rms_norm(h.data_ptr(), otmp.data_ptr(),
                                      w["post_norm"].data_ptr(), xn2.data_ptr(),
                                      1, H, EPS, s)
            mv(xn2.data_ptr(), w["gu"].data_ptr(), self.Dg.data_ptr(),
               2 * INTER, H, s)
            fvk.silu_mul_qwen36_bf16(self.Dg[:, :INTER].contiguous().data_ptr(),
                                     self.Dg[:, INTER:].contiguous().data_ptr(),
                                     self.act.data_ptr(), INTER, s)
            mv(self.act.data_ptr(), w["down"].data_ptr(), dtmp.data_ptr(),
               H, INTER, s)
            if L < self.NL - 1:
                fvk.residual_add_rms_norm(
                    h.data_ptr(), dtmp.data_ptr(),
                    self.WL[L + 1]["in_norm"].data_ptr(), xn.data_ptr(),
                    1, H, EPS, s)
            else:
                fvk.residual_add(h.data_ptr(), dtmp.data_ptr(), H, s)
                fvk.rms_norm(h.data_ptr(), self.final_norm.data_ptr(),
                             self.hn.data_ptr(), 1, H, EPS, s)
        mv(self.hn.data_ptr(), self.CB.data_ptr(), self.Dh.data_ptr(),
           self.NC * self.CV, H, s)
        return self.Dh.view(1, self.NC, self.CV)

    @torch.no_grad()
    def _set_pos(self, t: int) -> None:
        self.rope_cos_buf.copy_(self.fe._rope_cos[t])
        self.rope_sin_buf.copy_(self.fe._rope_sin[t])
        self.cur_pos_dev.fill_(t)
        self.seqused_dev.fill_(t + 1)

    @torch.no_grad()
    def capture_graph(self, embed_row: torch.Tensor, warm_pos: int) -> None:
        if getattr(self, "_graph", None) is not None:
            return
        if not hasattr(self, "rope_cos_buf"):
            self._alloc_graph()
        gs = self._gs
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            for _ in range(3):
                self.h.copy_(embed_row)
                self._set_pos(warm_pos)
                self._step_graphable()
            self.h.copy_(embed_row)
            self._set_pos(warm_pos)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=gs):
                self._step_graphable()
        torch.cuda.current_stream().wait_stream(gs)
        self._graph = g

    @torch.no_grad()
    def decode_graph(self, embed_row: torch.Tensor, t: int) -> torch.Tensor:
        gs = self._gs
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            self.h.copy_(embed_row)
            self._set_pos(t)
            self._graph.replay()
        torch.cuda.current_stream().wait_stream(gs)
        return self.Dh.view(1, self.NC, self.CV)
