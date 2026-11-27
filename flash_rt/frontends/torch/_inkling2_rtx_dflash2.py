"""DFlash 2 drafter of Inkling 2 Mini (incoai/Inkling-2-Mini-DFlash2) on RTX SM120.

The drafter (SGLang's ``DFlash2DraftModel`` with the Inkling integration) keeps its own context KV: each committed
target position's four features (the hidden states entering visits 5, 23 and 41 and the final normalized state)
go through ``fc`` and ``hidden_norm`` into per-layer K / V (``k_norm``, RoPE). A block of the anchor and 7 mask
tokens (the target's embedding and ``embed_norm``) runs 5 bidirectional layers (32 query / 8 KV heads over a
2047-position window and the block, two-tap dynamic grouped convolutions around attention and MLP, SwiGLU); the
target's LM head gives each mask row its 16 largest logits over the real vocabulary, and the candidate selector
(predecessor / successor codebooks of rank 256, scores softcapped after the correction) walks them greedily.

Programs, recorded into the target runtime and captured as CUDA graphs like the DSpark drafter's:
``df_ctx<R>`` (context of R committed rows), ``df_first`` (drafts after a prompt), ``df_round<W>`` (verification of
W rows with the tokens logged, the context of the accepted rows, the next drafts). The target verifies every draft,
so the drafts decide how many tokens a step accepts, never which: the drafter's products read E4M3 copies of its
weights (FP32 scale per row) and the mask rows' logits an NVFP4 copy of the LM head. ``use_kernels = False``: the
context programs recorded afterwards run PyTorch operations over the BF16 weights (the reference of the context
kernels, run eagerly).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.frontends.torch._inkling2_rtx_runtime import (
    BF16, F32, I32, D, Inkling2ConfigError, Inkling2RtxRuntime, P, _require_module, check_tensors, e4m3_rows,
    gemm_call, nvfp4_rows, safetensors_header)

BLOCK = 8
# drafter geometry the kernels are written for (config.json, dflash_config)
DF_GEOMETRY = {"hidden_size": D, "num_hidden_layers": 5, "num_attention_heads": 32, "num_key_value_heads": 8,
               "head_dim": 128, "intermediate_size": 6912, "sliding_window": 2048}
DF_CONFIG = {"block_size": BLOCK, "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256,
             "selector_top_k": 16, "selector_softcap_mode": "after_correction"}
FEATURE_VISITS = (4, 22, 40)     # target_layer_ids 5, 23, 41: the hidden states entering these visits


def check_dflash2(drafter_dir, target: dict, visits: int) -> dict:
    """The drafter's ``config.json`` and ``model.safetensors`` (header only) checked against the kernels and the
    target's text config; returns the drafter config."""
    d = Path(drafter_dir)
    path = d / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (the DFlash 2 drafter)")
    cfg = json.loads(path.read_text())
    for key, value in DF_GEOMETRY.items():
        if cfg.get(key) != value:
            raise Inkling2ConfigError(f"DFlash 2 {key}={cfg.get(key)!r}; the drafter kernels are built for {value!r}")
    dc = cfg.get("dflash_config")
    if not dc:
        raise Inkling2ConfigError(f"{path} has no dflash_config (not a DFlash 2 drafter)")
    for key, value in DF_CONFIG.items():
        if dc.get(key) != value:
            raise Inkling2ConfigError(f"DFlash 2 dflash_config {key}={dc.get(key)!r}; expected {value!r}")
    targets = list(dc.get("target_layer_ids") or ())
    if targets != [v + 1 for v in FEATURE_VISITS] + [visits]:
        raise Inkling2ConfigError(f"DFlash 2 target_layer_ids {targets}: expected "
                                  f"{[v + 1 for v in FEATURE_VISITS] + [visits]}")
    values = {"rms_norm_eps": cfg.get("rms_norm_eps"),
              "rope_parameters.rope_theta": (cfg.get("rope_parameters") or {}).get("rope_theta"),
              "dflash_config.final_logit_softcapping": dc.get("final_logit_softcapping")}
    for key, v in values.items():
        if not isinstance(v, (int, float)) or v <= 0:
            raise Inkling2ConfigError(f"DFlash 2 {key}={v!r}: a positive number expected")
    V = int(target["vocab_size"])
    if not 0 <= int(dc.get("mask_token_id", -1)) < V:
        raise Inkling2ConfigError(f"DFlash 2 mask_token_id {dc.get('mask_token_id')!r} outside the vocabulary")
    model = d / "model.safetensors"
    if not model.is_file():
        raise FileNotFoundError(f"{model} not found (the DFlash 2 drafter)")
    check_tensors(safetensors_header(model), expected_dflash2_tensors(V), "DFlash 2 drafter weights")
    return cfg


def expected_dflash2_tensors(vocab_size: int) -> dict:
    """The drafter tensors the kernels read: (safetensors dtype, shape)."""
    bf, g = "BF16", DF_GEOMETRY
    H, Q, KV, I = D, g["num_attention_heads"] * g["head_dim"], g["num_key_value_heads"] * g["head_dim"], \
        g["intermediate_size"]
    conv = 2 * DF_CONFIG["conv_kernel_size"] * (D // DF_CONFIG["conv_group_size"])
    R = DF_CONFIG["selector_rank"]
    t = {"norm.weight": (bf, (H,)), "fc.weight": (bf, (H, 4 * H)), "hidden_norm.weight": (bf, (H,)),
         "candidate_selector.hidden_projection.weight": (bf, (R, H)),
         "candidate_selector.predecessor_codebook": (bf, (vocab_size, R)),
         "candidate_selector.successor_codebook": (bf, (vocab_size, R))}
    for l in range(g["num_hidden_layers"]):
        p, a = f"layers.{l}.", f"layers.{l}.self_attn."
        t.update({p + "input_layernorm.weight": (bf, (H,)), p + "post_attention_layernorm.weight": (bf, (H,)),
                  a + "q_proj.weight": (bf, (Q, H)), a + "k_proj.weight": (bf, (KV, H)),
                  a + "v_proj.weight": (bf, (KV, H)), a + "o_proj.weight": (bf, (H, Q)),
                  a + "q_norm.weight": (bf, (g["head_dim"],)), a + "k_norm.weight": (bf, (g["head_dim"],)),
                  p + "mlp.gate_proj.weight": (bf, (I, H)), p + "mlp.up_proj.weight": (bf, (I, H)),
                  p + "mlp.down_proj.weight": (bf, (H, I))})
        for c in ("attention_conv", "mlp_conv"):
            t.update({p + f"{c}.base_kernel": (bf, (2, DF_CONFIG["conv_kernel_size"], H)),
                      p + f"{c}.kernel_projection.weight": (bf, (conv, H))})
    return t


class Inkling2RtxDFlash2:
    """DFlash 2 speculative decoding over an :class:`Inkling2RtxRuntime` (greedy, one sequence).

    ``adaptive``: each step verifies the number of rows (2 to 8) with the most expected tokens per unit of time (see
    :meth:`adaptive_steps`); else every step verifies ``width`` rows.
    """

    def __init__(self, rt: Inkling2RtxRuntime, drafter_dir: str, width: int = 8, adaptive: bool = True):
        _require_module()
        if getattr(rt, "drafter", None) is not None:
            raise RuntimeError("the runtime already has a drafter")
        if not 2 <= width <= min(BLOCK, rt.exact_rows):
            raise ValueError(f"width must be in 2..{min(BLOCK, rt.exact_rows)}")
        d = Path(drafter_dir)
        cfg = check_dflash2(d, rt.c, len(rt.visits))
        dc = cfg["dflash_config"]
        self.cfg = cfg
        self.L = int(cfg["num_hidden_layers"])
        self.nh, self.nkv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        self.hd = int(cfg["head_dim"])
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg["rope_parameters"]["rope_theta"])
        self.window = int(cfg["sliding_window"]) - 1                  # keys left of a query (SGLang window_left)
        self.mask_id = int(dc["mask_token_id"])
        self.topk = int(dc["selector_top_k"])
        self.taps, self.gs = int(dc["conv_kernel_size"]), int(dc["conv_group_size"])
        self.G = D // self.gs
        self.cap = float(dc["final_logit_softcapping"])
        self.feature_visits = FEATURE_VISITS
        self.rt, self.width, self.adaptive = rt, int(width), bool(adaptive)
        dev = rt.dev
        from safetensors import safe_open
        with safe_open(str(d / "model.safetensors"), "pt", device=str(dev)) as st:
            t = {k: st.get_tensor(k) for k in expected_dflash2_tensors(int(rt.c["vocab_size"]))}
        g = lambda l, n: t[f"layers.{l}.{n}"]
        # the products' E4M3 copies (FP32 scale per row); BF16 kept for the norms, the convolution bases and the
        # K / V rows and fc (prompt chunks write the context from BF16 GEMMs)
        self.layers, self.w8 = [], {"fc": e4m3_rows(t["fc.weight"])}
        kv0 = self.nh * self.hd
        for l in range(self.L):
            qkv = torch.cat([g(l, "self_attn.q_proj.weight"), g(l, "self_attn.k_proj.weight"),
                             g(l, "self_attn.v_proj.weight")], 0)
            c8, s8 = e4m3_rows(qkv)
            self.w8[(l, "qkv")], self.w8[(l, "kv")] = (c8, s8), (c8[kv0:], s8[kv0:])    # (the K / V rows: views)
            self.w8[(l, "o")] = e4m3_rows(g(l, "self_attn.o_proj.weight"))
            self.w8[(l, "gu")] = e4m3_rows(torch.cat([g(l, "mlp.gate_proj.weight"), g(l, "mlp.up_proj.weight")], 0))
            self.w8[(l, "down")] = e4m3_rows(g(l, "mlp.down_proj.weight"))
            self.w8[(l, "ac_proj")] = e4m3_rows(g(l, "attention_conv.kernel_projection.weight"))
            self.w8[(l, "mc_proj")] = e4m3_rows(g(l, "mlp_conv.kernel_projection.weight"))
            self.layers.append(dict(
                in_norm=g(l, "input_layernorm.weight"), post_norm=g(l, "post_attention_layernorm.weight"),
                q_norm=g(l, "self_attn.q_norm.weight"), k_norm=g(l, "self_attn.k_norm.weight"),
                ac_base=g(l, "attention_conv.base_kernel"), mc_base=g(l, "mlp_conv.base_kernel"),
                kv=qkv[kv0:].contiguous()))
            del qkv
        self.norm, self.fc, self.hidden_norm = t["norm.weight"], t["fc.weight"], t["hidden_norm.weight"]
        self.sel_proj = t["candidate_selector.hidden_projection.weight"]
        self.pred_cb = t["candidate_selector.predecessor_codebook"]
        self.succ_cb = t["candidate_selector.successor_codebook"]
        del t
        self.vocab = rt.vocab
        self.embed, self.embed_norm = rt.w["embed.weight"], rt.w["embed_norm.weight"]
        self.target_eps = float(rt.c["rms_norm_eps"])
        kvs = self.nkv * self.hd
        self.kcap = rt.max_seq + 2 * BLOCK
        z = lambda *s, dt=BF16: torch.zeros(*s, device=dev, dtype=dt)
        self.kv = z(self.L, 2, self.nkv, self.kcap, self.hd)
        self.K, self.V = self.kv[:, 0], self.kv[:, 1]
        self.inter = int(cfg["intermediate_size"])
        # kernel buffers (one block of 8 rows; context of up to rt.RC rows)
        RC = rt.RC
        ncoef = 2 * self.taps * self.G
        self.dh, self.x, self.u = z(BLOCK, D, dt=F32), z(BLOCK, D), z(BLOCK, D)
        self.coef, self.mcoef = z(BLOCK, ncoef), z(BLOCK, ncoef)
        self.qkv_b = z(BLOCK, (self.nh + 2 * self.nkv) * self.hd)
        self.q_out, self.att = z(BLOCK, self.nh * self.hd), z(BLOCK, self.nh * self.hd)
        self.o = z(BLOCK, D)
        self.mlp = z(BLOCK, D, dt=F32)                 # the down projection (FP32)
        self.gu_b, self.act = z(BLOCK, 2 * self.inter), z(BLOCK, self.inter)
        self.hn = z(BLOCK, D)                          # the final norm of the block rows
        self.att_part = z(8 * 8 * 32 * 132, dt=F32)    # per KV head: 8 partials x 32 query rows x (O, max, sum)
        self.logits = z(BLOCK - 1, self.vocab, dt=F32)
        self.vals = z(BLOCK - 1, self.topk, dt=F32)
        self.prob = z(BLOCK - 1, dt=F32)
        self.top_part = z((BLOCK - 1) * 64 * (2 * self.topk + 2), dt=F32)     # df_topk partials
        self.stat = z(BLOCK, dt=F32)                     # [accepted, prob of drafts 1..7] after a round
        self.host = torch.zeros(BLOCK, dtype=F32, pin_memory=True)
        self.cand = z(BLOCK - 1, self.topk, dt=torch.int64)
        self.hp = z(BLOCK - 1, int(dc["selector_rank"]))
        self.feats = z(RC, 4 * D)
        self.ctxh, self.ctxn = z(RC, D, dt=F32), z(RC, D)
        self.kvc = z(self.L, RC, 2 * kvs, dt=F32)
        # weight-only NVFP4 copy of the target's LM head (E2M1, E4M3 per 16, FP32 per row) for the mask rows' logits
        self.head4, self.head4_gs, self.head4_rs = nvfp4_rows(rt.w["unembed.weight"][: self.vocab])
        m = self.embed[self.mask_id].float()
        m = m * torch.rsqrt(m.pow(2).mean() + self.target_eps)
        self.mask_row = (m * self.embed_norm.float()).to(BF16).contiguous()
        self.use_kernels = True                       # False: context programs recorded afterwards on PyTorch
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=dev, dtype=F32) / self.hd))
        self.brow = torch.arange(BLOCK, device=dev, dtype=torch.int64)
        self.log = z(rt.max_seq + 1, dt=I32)
        self.log_count = z(1, dt=I32)
        self._kvs = kvs
        # the widths the adaptive steps choose from (verification programs of at most exact_rows rows)
        self.widths_all = tuple(W for W in self.WIDTHS if W <= min(BLOCK, rt.exact_rows))
        rt.programs["df_first"] = self._draft_program() + [("call", self._report)]
        self.round = self.add_width(self.width)
        rt.drafter = self

    # ------------------------------------------------------------------------------------------- helpers
    def _rms(self, x, w, eps=None):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + (self.eps if eps is None else eps))
        return (y * w.float()).to(BF16)

    def _rope(self, x, pos):
        """x [rows, heads, hd] at positions pos [rows] (rotate halves; cos / sin rounded to BF16)."""
        fr = pos.to(F32)[:, None] * self.inv_freq[None, :]
        c, s = fr.cos().to(BF16).float()[:, None, :], fr.sin().to(BF16).float()[:, None, :]
        h = self.hd // 2
        x1, x2 = x[..., :h].float(), x[..., h:].float()
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(BF16)

    # ------------------------------------------------------------------------------------------- context
    def _context_rows(self, R, first_pos):
        """K / V of the drafter context for target rows 0..R-1 at positions first_pos() + j."""
        rt = self.rt
        def run():
            feats = torch.cat([rt.hid[v][:R] for v in self.feature_visits] + [rt.feat_final[:R]], -1)
            ctx = self._rms(F.linear(feats, self.fc), self.hidden_norm)
            pos = first_pos() + self.brow[:R] if R <= BLOCK else first_pos() + torch.arange(R, device=rt.dev)
            for l, L in enumerate(self.layers):
                kv = F.linear(ctx, L["kv"])
                k = self._rms(kv[:, : self._kvs].view(R, self.nkv, self.hd), L["k_norm"])
                k = self._rope(k, pos)
                v = kv[:, self._kvs:].view(R, self.nkv, self.hd)
                self.K[l].index_copy_(1, pos, k.transpose(0, 1))
                self.V[l].index_copy_(1, pos, v.transpose(0, 1))
        return run

    def context_program(self, R, after_verify=False):
        """``df_ctx<R>``: rows written by a prompt chunk / decode step (positions pos - R ..), or after a verification
        the accepted rows (positions pos - accepted - 1 ..; the rows past them land at positions not yet committed,
        overwritten when those are)."""
        name = f"df_ctx{R}" + ("v" if after_verify else "")
        rt = self.rt
        if name not in rt.programs:
            if not self.use_kernels:
                fp = (lambda: (rt.pos - rt.accepted - 1).long()) if after_verify else (lambda: (rt.pos - R).long())
                rt.programs[name] = [("call", self._context_rows(R, fp))]
            else:
                rt.programs[name] = self._context_calls(R, after_verify)
        return name

    def _context_calls(self, R, after_verify):
        ik, rt = _require_module(), self.rt
        feats, ctxh, ctxn = self.feats, self.ctxh, self.ctxn
        kvc = self.kvc.view(-1)[: self.L * R * 2 * self._kvs].view(self.L, R, 2 * self._kvs)   # [layers][R][2048]
        src = [rt.hid[v] for v in self.feature_visits] + [rt.feat_final]

        def gather():
            torch.cat([t[:R] for t in src], -1, out=feats[:R])
        calls = [("call", gather)]
        wkv = [L["kv"] for L in self.layers]
        if R <= BLOCK:          # decode steps / verifications: the E4M3 copies; prompt chunks: BF16 GEMMs
            c8, s8 = self.w8["fc"]
            calls.append(("gemv_fp8", [(P(c8), P(feats), P(ctxh), D, 4 * D, 0, 1, ik.GEMV_PLAIN, P(s8), R)], []))
        else:
            calls.append(gemm_call(feats[:R], self.fc, ctxh[:R], rt.gemm_ws))
        calls.append(("norm", [(P(ctxh), 1, 0, P(self.hidden_norm), P(ctxn), R, D)]))
        if R <= BLOCK:
            calls.append(("gemv_fp8", [(P(self.w8[(l, "kv")][0]), P(ctxn), P(kvc[l]), 2 * self._kvs, D, 0, 1,
                                        ik.GEMV_PLAIN, P(self.w8[(l, "kv")][1]), R) for l in range(self.L)], []))
        else:
            calls += [gemm_call(ctxn[:R], w, kvc[l][:R], rt.gemm_ws) for l, w in enumerate(wkv)]
        cnt = dict(count=P(rt.accepted), count_add=1) if after_verify else dict(count_add=R)
        calls.append((ik.df_ctx_write, dict(rows=R, layers=self.L, pos=P(rt.pos), kvc=P(kvc), kv=P(self.kv),
                                            cap=self.kcap, theta=self.theta, eps=self.eps,
                                            k_norm_l=[P(L["k_norm"]) for L in self.layers], **cnt)))
        return calls

    def _draft_program(self):
        ik, rt = _require_module(), self.rt

        def prod(l, name, x, y, n, k, yf=0):
            c8, s8 = self.w8[(l, name)]
            return ("gemv_fp8", [(P(c8), P(x), P(y), n, k, 0, yf, ik.GEMV_PLAIN, P(s8), BLOCK)], [])
        base = lambda t, i: P(t) + i * self.taps * D * 2
        com = dict(pos=P(rt.pos), kv=P(self.kv), cap=self.kcap, theta=self.theta, eps=self.eps)
        calls = [(ik.df_embed, dict(com, anchor=P(rt.vtokens), embed=P(self.embed), embed_norm=P(self.embed_norm),
                                    mask_row=P(self.mask_row), dh=P(self.dh), x=P(self.x),
                                    norm_w=P(self.layers[0]["in_norm"]), eps2=self.target_eps))]
        ncoef = 2 * self.taps * self.G
        for l, L in enumerate(self.layers):
            calls.append(prod(l, "ac_proj", self.x, self.coef, ncoef, D))
            calls.append((ik.df_conv, dict(com, src=P(self.x), dst=P(self.u), coef=P(self.coef), coef_part=0,
                                           base=base(L["ac_base"], 0))))
            calls.append(prod(l, "qkv", self.u, self.qkv_b, self.qkv_b.shape[1], D))
            calls.append((ik.df_qk, dict(com, layer=l, qkv=P(self.qkv_b), q_norm=P(L["q_norm"]),
                                         k_norm=P(L["k_norm"]), q_out=P(self.q_out))))
            calls.append((ik.df_attention, dict(com, layer=l, window=self.window, q_out=P(self.q_out),
                                                attn_out=P(self.att), dh=P(self.att_part))))
            calls.append(prod(l, "o", self.att, self.o, D, self.nh * self.hd))
            calls.append((ik.df_conv_add_norm, dict(com, src=P(self.o), coef=P(self.coef), coef_part=1,
                                                    base=base(L["ac_base"], 1), dh=P(self.dh), x=P(self.x),
                                                    norm_w=P(L["post_norm"]))))
            calls.append(prod(l, "mc_proj", self.x, self.mcoef, ncoef, D))
            calls.append((ik.df_conv, dict(com, src=P(self.x), dst=P(self.u), coef=P(self.mcoef), coef_part=0,
                                           base=base(L["mc_base"], 0))))
            calls.append(prod(l, "gu", self.u, self.gu_b, 2 * self.inter, D))
            calls.append((ik.df_silu, dict(gu=P(self.gu_b), act=P(self.act), inter=self.inter)))
            calls.append(prod(l, "down", self.act, self.mlp, D, self.inter, yf=1))
            # the next layer's input norm, after the last layer the final norm (the head's and the selector's rows)
            nxt = self.layers[l + 1]["in_norm"] if l + 1 < self.L else self.norm
            calls.append((ik.df_conv_add_norm, dict(com, src_f32=P(self.mlp), coef=P(self.mcoef), coef_part=1,
                                                    base=base(L["mc_base"], 1), dh=P(self.dh), norm_w=P(nxt),
                                                    x=P(self.x if l + 1 < self.L else self.hn))))
        # rows 1..7 (the mask rows): logits on the NVFP4 head, the 16 largest and the largest probability, selector
        hn = P(self.hn) + D * 2
        calls.append(("gemv_fp4w", [(P(self.head4), hn, P(self.logits), self.vocab, D, 0, 1, ik.GEMV_PLAIN,
                                     P(self.head4_rs), BLOCK - 1, 0, P(self.head4_gs))], []))
        calls.append((ik.df_topk, dict(rows=BLOCK - 1, vocab=self.vocab, logits=P(self.logits),
                                       top_part=P(self.top_part), top_vals=P(self.vals), top_ids=P(self.cand),
                                       top_prob=P(self.prob))))
        calls.append(("gemv_mma", [(P(self.sel_proj), hn, P(self.hp), self.hp.shape[1], D, 0, 0, ik.GEMV_PLAIN, 0,
                                    BLOCK - 1)], []))
        calls.append((ik.df_select, dict(anchor=P(rt.vtokens), cand=P(self.cand), vals=P(self.vals), hp=P(self.hp),
                                         pred_cb=P(self.pred_cb), succ_cb=P(self.succ_cb), softcap=self.cap,
                                         drafts=P(rt.vtokens) + 4)))
        return calls

    # ------------------------------------------------------------------------------------------- drafts
    def _report(self):
        """The round's accepted drafts and the next drafts' confidences into pinned host memory (in the graph)."""
        torch.cat([self.rt.accepted.to(F32), self.prob], out=self.stat)
        self.host.copy_(self.stat, non_blocking=True)

    # ------------------------------------------------------------------------------------------- programs
    def add_width(self, W: int) -> str:
        ik, rt = _require_module(), self.rt
        name = f"df_round{W}"
        if name in rt.programs:
            return name
        verify = rt.programs[rt.program("verify", W)]
        logged = [(ik.accept, dict(c[1], log=P(self.log), log_count=P(self.log_count))) if c[0] is ik.accept else c
                  for c in verify]
        rt.programs[name] = logged + rt.programs[self.context_program(W, after_verify=True)] + rt.programs["df_first"]
        return name

    def set_width(self, W: int):
        if not 2 <= W <= min(BLOCK, self.rt.exact_rows):
            raise ValueError(f"width must be in 2..{min(BLOCK, self.rt.exact_rows)}")
        self.round, self.width = self.add_width(W), int(W)

    # ------------------------------------------------------------------------------------------- generation
    def prefill(self, prompt, exact: bool = False) -> int:
        rt = self.rt
        first = rt.prefill(prompt, exact=exact, after=lambda kind, rows: rt.run(self.context_program(rows)))
        rt.vtokens[0].copy_(rt.token.view(-1)[0])
        rt.run("df_first")
        return first

    # adaptive rows: each step verifies the W in widths_all maximizing expected tokens - rate x cost[W]; a draft's
    # agreement rate by its confidence bin (ADAPT_EDGES), learned online from the verified steps with a prior: the
    # agreement of this drafter's drafts with greedy targets by confidence (ADAPT_P -> ADAPT_A, the calibration the
    # Apple path uses, _inkling2_mtp.ADAPT_P_DF / ADAPT_A_DF). COST / COST_FP8: CUDA-graph round times (ms) on an RTX
    # 5090 at 512 context (verification, context, next drafts), scaled online by the measured ones.
    WIDTHS = (2, 3, 4, 5, 6, 7, 8)
    COST = {2: 5.87, 3: 6.81, 4: 7.40, 5: 7.99, 6: 8.58, 7: 8.99, 8: 9.41}
    COST_FP8 = {2: 4.98, 3: 5.76, 4: 6.45, 5: 7.04, 6: 7.54, 7: 8.04, 8: 8.33}
    ADAPT_EDGES = (0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.01)
    ADAPT_P = (0.0, 0.336, 0.464, 0.646, 0.752, 0.855, 0.929, 0.967, 0.985, 0.993, 1.0)
    ADAPT_A = (0.0, 0.06, 0.34, 0.37, 0.56, 0.56, 0.63, 0.71, 0.76, 0.82, 0.983)
    ADAPT_PRIOR, ADAPT_DECAY = 10.0, 0.99

    def adaptive_steps(self, first: int, max_new_tokens: int, check_every: int = 4) -> Iterator[List[int]]:
        """Steps of adaptive widths: after each round the host reads the accepted count and the next drafts'
        confidences (pinned memory, written by the round's graph) and picks the next round's width."""
        rt = self.rt
        widths = self.widths_all
        rounds = {W: self.add_width(W) for W in widths}
        self.log[0] = first
        self.log_count.fill_(1)
        edges = np.asarray(self.ADAPT_EDGES)
        mids = np.minimum((edges[:-1] + edges[1:]) / 2, 1.0)
        reach = np.full(len(mids), self.ADAPT_PRIOR)
        agreed = reach * np.interp(mids, self.ADAPT_P, self.ADAPT_A)
        table = self.COST_FP8 if rt.fp8 else self.COST
        cost = np.array([table[W] for W in widths])
        rate, scale = 0.25, 1.0
        stream = torch.cuda.current_stream()
        stream.synchronize()
        probs = self.host[1:].numpy().copy()
        self.rounds = done = tokens = 0
        self.widths = []
        while done < max_new_tokens:
            bins = np.clip(np.searchsorted(edges, probs, "right") - 1, 0, len(mids) - 1)
            agree = np.cumprod(agreed[bins] / reach[bins])
            expect = np.concatenate([[1.0], 1.0 + np.cumsum(agree)])         # expected tokens of V rows: [V - 1]
            gain = np.array([expect[W - 1] for W in widths]) - rate * scale * cost
            V = widths[int(np.argmax(gain))]
            t0 = time.perf_counter()
            rt.run(rounds[V])
            stream.synchronize()
            ms = 1e3 * (time.perf_counter() - t0)
            n = int(self.host[0])
            probs = self.host[1:].numpy().copy()
            reach *= self.ADAPT_DECAY
            agreed *= self.ADAPT_DECAY
            k = np.arange(min(n, V - 2) + 1)
            np.add.at(reach, bins[k], 1.0)
            np.add.at(agreed, bins[k], (k < n).astype(np.float64))
            rate = 0.9 * rate + 0.1 * ((n + 1) / max(1e-3, ms))
            scale = 0.9 * scale + 0.1 * (ms / table[V])
            self.rounds += 1
            self.widths.append(V)
            tokens += n + 1
            if self.rounds % check_every == 0 or tokens >= max_new_tokens:
                c = min(int(self.log_count.item()), max_new_tokens)
                if c > done:
                    yield self.log[done:c].tolist()
                    done = c

    def steps(self, first: int, max_new_tokens: int, check_every: int = 4) -> Iterator[List[int]]:
        if self.adaptive:
            yield from self.adaptive_steps(first, max_new_tokens, check_every)
            return
        rt = self.rt
        self.log[0] = first
        self.log_count.fill_(1)
        self.rounds = done = 0
        name = self.round
        while done < max_new_tokens:
            for _ in range(check_every):
                rt.run(name)
            self.rounds += check_every
            n = min(int(self.log_count.item()), max_new_tokens)
            if n > done:
                yield self.log[done:n].tolist()
                done = n
