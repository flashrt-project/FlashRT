"""DSpark drafter of Inkling 2 Mini (``mtp.safetensors``) on the ``flash_rt_inkling2`` kernels.

The drafter is a three-layer decoder over a block of eight positions (the anchor and seven mask embeddings) that
attends to its own context KV (one row per committed target position, written from four target features), followed
by a Markov RNN head that walks the block: each draft's logits are the target LM head on the drafter's normalized
row plus a rank-256 correction folded into the vocabulary (``unembed @ hidden_out``). Programs, recorded into the
target runtime and captured as CUDA graphs:

* ``ctx1`` / ``ctx_chunk`` / ``ctx_exact``: the context of a decode step's row / of a prompt chunk's rows;
* ``first_draft<W>``: the block and the walk (drafts 1 .. W - 1 into ``vtokens``);
* ``round<W>``: verification of the anchor and W - 1 drafts (tokens logged on the device), the context of the
  accepted rows and the next drafts: one graph per speculative step, no host round trip.

The target verifies every draft, so the output is the target's greedy continuation whatever the drafts are; the
draft-only vocabulary products (base logits, the folded correction) run on E4M3 copies with per-row scales.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import torch

from flash_rt.frontends.torch._inkling2_rtx_runtime import (
    BF16, F32, I32, D, HD, HKV, Inkling2ConfigError, Inkling2RtxRuntime, P, _require_module, check_tensors,
    gemm_call, safetensors_header)

MTP_GEOMETRY = {
    "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24, "num_key_value_heads": 8,
    "head_dim": 128, "intermediate_size": 9216, "local_layer_ids": [0, 2], "sliding_window_size": 1024,
    "block_size": 8, "markov_rank": 256, "markov_head_type": "rnn_hidden", "q_bias": False, "o_bias": False,
}
HQ, RANK, BLOCK, LAYERS, INTER, WINDOW = 24, 256, 8, 3, 9216, 1024


def check_mtp_config(raw: dict, visits: int) -> dict:
    """The checkpoint's ``mtp_config``, checked against the drafter kernels; ``target_layer_ids`` are the target
    features (hidden states after visits t - 1; the last one, ``visits``, is the final normalized row)."""
    m = raw.get("mtp_config")
    if not m:
        raise Inkling2ConfigError("config.json has no mtp_config (no DSpark drafter)")
    for key, value in MTP_GEOMETRY.items():
        if m.get(key) != value:
            raise Inkling2ConfigError(f"mtp_config {key}={m.get(key)!r}; the drafter kernels are built for {value!r}")
    targets = list(m.get("target_layer_ids") or ())
    if len(targets) != 4 or targets[-1] != visits or not all(0 < t < visits for t in targets[:-1]):
        raise Inkling2ConfigError(f"mtp_config target_layer_ids {targets}: expected three visits and {visits}")
    return m


def expected_mtp_tensors(vocab_size: int) -> dict:
    """The drafter tensors the kernels read (``mtp.safetensors``): (safetensors dtype, shape)."""
    bf = "BF16"
    t = {"fuse_weight": (bf, (4, 4)), "mask_embed": (bf, (D,)), "norm.weight": (bf, (D,)),
         "markov_head.dflare_norm.weight": (bf, (D,)), "markov_head.hidden_out.weight": (bf, (D, RANK)),
         "markov_head.hidden_out.bias": (bf, (D,)), "markov_head.markov_w1.weight": (bf, (vocab_size, RANK)),
         "markov_head.rnn_proj.weight": (bf, (3 * RANK, 2 * RANK + 2 * D)), "markov_head.rnn_proj.bias": (bf, (3 * RANK,))}
    for l in range(LAYERS):
        p, a = f"layers.{l}.", f"layers.{l}.self_attn."
        t.update({p + n: (bf, (D,)) for n in ("dflare_norm.weight", "input_layernorm.weight",
                                              "post_attention_layernorm.weight")})
        t.update({p + "mlp.gate_proj.weight": (bf, (INTER, D)), p + "mlp.up_proj.weight": (bf, (INTER, D)),
                  p + "mlp.down_proj.weight": (bf, (D, INTER)), a + "q_proj.weight": (bf, (HQ * HD, D)),
                  a + "o_proj.weight": (bf, (D, HQ * HD))})
        t.update({a + f"{n}.weight": (bf, (HKV * HD, D)) for n in ("k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx")})
        t.update({a + f"{n}.weight": (bf, (HD,)) for n in ("q_norm", "k_norm", "k_norm_ctx")})
    return t


class Inkling2RtxDrafter:
    """DSpark speculative decoding over an :class:`Inkling2RtxRuntime` (greedy, one sequence).

    ``width``: verified rows per speculative step (the anchor and width - 1 drafts, 2..8; 4 is fastest on the
    512 / 512 benchmark). ``fp8``: E4M3 copies for the draft-only vocabulary products (half the bytes)."""

    def __init__(self, rt: Inkling2RtxRuntime, width: int = 4, fp8: bool = True):
        ik = _require_module()
        if getattr(rt, "drafter", None) is not None:
            # its programs and graphs point at the first drafter's buffers
            raise RuntimeError("the runtime already has a drafter")
        if not 2 <= width <= min(BLOCK, rt.exact_rows):
            raise ValueError(f"width must be in 2..{min(BLOCK, rt.exact_rows)}")
        m = check_mtp_config(rt.raw, len(rt.visits))
        self.feature_visits = tuple(t - 1 for t in m["target_layer_ids"][:-1])
        self.theta = float(m["rope_theta"])
        self.rt, self.fp8, self.width = rt, fp8, int(width)
        dev = rt.dev
        z = lambda *s, dt=BF16: torch.zeros(*s, device=dev, dtype=dt)
        path = Path(rt.dir) / "mtp.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found (the DSpark drafter)")
        expected = expected_mtp_tensors(rt.c["vocab_size"])
        check_tensors(safetensors_header(path), expected, "drafter weights")
        from safetensors import safe_open
        with safe_open(str(path), "pt", device=str(dev)) as st:
            w = {k: st.get_tensor(k) for k in expected}
        for l in range(LAYERS):              # down projections split along k (two FP32 partial sums)
            dn = w.pop(f"layers.{l}.mlp.down_proj.weight")
            w[f"layers.{l}.down_a"] = dn[:, :INTER // 2].contiguous()
            w[f"layers.{l}.down_b"] = dn[:, INTER // 2:].contiguous()
            del dn
        self.w = w
        self.alpha = torch.softmax(w["fuse_weight"].float(), -1).to(BF16).float().reshape(-1).tolist()
        head = rt.w["unembed.weight"]
        V = head.shape[0]
        # the folded correction W_lm @ hidden_out (and its bias) in FP32 sums of exact BF16 products, then BF16
        hw_t = w["markov_head.hidden_out.weight"].t().contiguous()            # [RANK][D]
        hb = w["markov_head.hidden_out.bias"].reshape(1, D)
        self.fold_w, self.fold_b = z(V, RANK), z(V)
        acc_w, acc_b = z(16384, RANK, dt=F32), z(16384, 1, dt=F32)
        for s in range(0, V, 16384):
            n = min(16384, V - s)
            rt._run([gemm_call(head[s:s + n], hw_t, acc_w[:n], rt.gemm_ws),
                     gemm_call(head[s:s + n], hb, acc_b[:n], rt.gemm_ws)])
            self.fold_w[s:s + n] = acc_w[:n].to(BF16)
            self.fold_b[s:s + n] = acc_b[:n, 0].to(BF16)
        del acc_w, acc_b
        if fp8:
            self.head8, self.head_scale = self._e4m3_rows(head)
            self.fold8, self.fold_scale = self._e4m3_rows(self.fold_w)
        self.vocab = rt.vocab
        self.cap = rt.max_seq + 2 * BLOCK
        self.kv = z(LAYERS, 2, HKV, self.cap, HD)
        self.ctx_x = z(LAYERS, BLOCK, D)
        self.kvc = z(LAYERS, 2, BLOCK, HKV * HD)
        self.ctx_x_big = z(LAYERS, rt.prefill_rows, D)          # the context of a tensor-core prompt chunk
        self.kvc_big = z(LAYERS, 2, rt.prefill_rows, HKV * HD)
        self.frnn = z(D)
        self.dh = z(BLOCK, D, dt=F32)
        self.x = z(BLOCK, D)
        self.q, self.k, self.v = z(BLOCK, HQ * HD), z(BLOCK, HKV * HD), z(BLOCK, HKV * HD)
        self.qr, self.att, self.o = z(BLOCK, HQ * HD), z(BLOCK, HQ * HD), z(BLOCK, D)
        self.g, self.u, self.act = z(BLOCK, INTER), z(BLOCK, INTER), z(BLOCK, INTER)
        self.mlp_a, self.mlp_b = z(BLOCK, D, dt=F32), z(BLOCK, D, dt=F32)
        self.hn = z(BLOCK, D)
        self.logits = z(BLOCK - 1, self.vocab, dt=F32)
        self.lm_best = z(rt.lm_blocks * BLOCK, dt=F32)
        self.lm_index = z(rt.lm_blocks * BLOCK, dt=I32)
        self.lm_counter = z(1, dt=I32)
        self.lm_token = z(BLOCK, dt=I32)
        self.rnn_in, self.proj = z(2 * RANK + 2 * D), z(3 * RANK, dt=F32)
        self.state, self.latent = z(RANK), z(RANK)
        self.pick_best = z(ik.MTP_PICK_BLOCKS, dt=F32)
        self.pick_index = z(ik.MTP_PICK_BLOCKS, dt=I32)
        self.pick_counter = z(1, dt=I32)
        self.log = z(rt.max_seq + 1, dt=I32)               # tokens of the speculative steps (one per position)
        self.log_count = z(1, dt=I32)
        rt.programs["ctx1"] = self._context(None, 1)
        rt.programs["ctx_chunk"] = self._context(rt.accepted, 1, rt.prefill_rows)
        rt.programs["ctx_exact"] = self._context(rt.accepted, 1)
        self.round = self.add_width(self.width)
        rt.drafter = self

    @staticmethod
    def _e4m3_rows(w, chunk=16384):
        """E4M3 codes [rows][cols] (uint8) and FP32 scales [rows] = amax / 448 per row."""
        codes = torch.empty(w.shape, device=w.device, dtype=torch.uint8)
        scale = torch.empty(w.shape[0], device=w.device, dtype=F32)
        with torch.no_grad():
            for s in range(0, w.shape[0], chunk):
                x = w[s:s + chunk].float()
                sc = (x.abs().amax(1) / 448.0).clamp_min(1e-30)
                codes[s:s + chunk] = (x / sc[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
                scale[s:s + chunk] = sc
        return codes, scale

    # ------------------------------------------------------------------------------------------- programs
    def _context(self, count, count_add, rows=BLOCK):
        """The drafter context of the last committed rows (``count`` + ``count_add`` of at most ``rows``: the
        block's eight, or a prompt chunk's rows with the projections on cuBLAS)."""
        ik, rt, w = _require_module(), self.rt, self.w
        x = self.ctx_x_big if rows > BLOCK else self.ctx_x
        kvc = self.kvc_big if rows > BLOCK else self.kvc
        cnt = {"count": P(count)} if count is not None else {}
        calls = [(ik.mtp_context_mix, dict(feat=[P(rt.hid[v]) for v in self.feature_visits] + [P(rt.feat_final)],
                                           count_add=count_add, alpha=self.alpha, ctx_rows=rows,
                                           norm=[P(w[f"layers.{l}.dflare_norm.weight"]) for l in range(LAYERS)] +
                                                [P(w["markov_head.dflare_norm.weight"])],
                                           x=P(x), frnn=P(self.frnn), **cnt))]
        tasks = []
        for l in range(LAYERS):
            a = f"layers.{l}.self_attn."
            for i, n in enumerate(("k", "v")):
                if rows > BLOCK:
                    calls.append(gemm_call(x[l], w[a + f"{n}_proj_ctx.weight"], kvc[l, i], rt.gemm_ws))
                else:
                    tasks.append((P(w[a + f"{n}_proj_ctx.weight"]), P(x[l]), P(kvc[l, i]), HKV * HD, D, 0, 0,
                                  ik.GEMV_PLAIN, 0, BLOCK))
        if tasks:
            calls.append(("gemv", tasks, []))
        calls.append((ik.mtp_context_write, dict(count_add=count_add, kvc=P(kvc), pos=P(rt.pos), kv=P(self.kv),
                                                 cap=self.cap, theta=self.theta, ctx_rows=rows,
                                                 k_norm=[P(w[f"layers.{l}.self_attn.k_norm_ctx.weight"])
                                                         for l in range(LAYERS)], **cnt)))
        return calls

    def _block_and_walk(self, steps):
        """The block and ``steps`` Markov steps (drafts 1 .. steps into ``vtokens``)."""
        ik, rt, w = _require_module(), self.rt, self.w
        anchor, drafts = P(rt.vtokens), P(rt.vtokens) + 4
        calls = [(ik.mtp_block_embed, dict(anchor=anchor, embed=P(rt.w["embed.weight"]),
                                           embed_norm=P(rt.w["embed_norm.weight"]), mask_embed=P(w["mask_embed"]),
                                           dh=P(self.dh)))]

        def g(W, x, y, n, k, yf=0, xs=0):
            return (P(W), x if isinstance(x, int) else P(x), P(y), n, k, 0, yf, ik.GEMV_PLAIN, 0, BLOCK, xs)

        for l in range(LAYERS):
            p, a = f"layers.{l}.", f"layers.{l}.self_attn."
            norm = dict(dh=P(self.dh), norm_w=P(w[p + "input_layernorm.weight"]), x=P(self.x))
            if l:
                norm["add_part"] = [P(self.mlp_a), P(self.mlp_b)]
            calls.append((ik.mtp_norm, norm))
            calls.append(("gemv", [g(w[a + "q_proj.weight"], self.x, self.q, HQ * HD, D),
                                   g(w[a + "k_proj.weight"], self.x, self.k, HKV * HD, D),
                                   g(w[a + "v_proj.weight"], self.x, self.v, HKV * HD, D)], []))
            calls.append((ik.mtp_qk, dict(pos=P(rt.pos), kv=P(self.kv), cap=self.cap, theta=self.theta, layer=l,
                                          q_in=P(self.q), k_in=P(self.k), v_in=P(self.v),
                                          q_norm=P(w[a + "q_norm.weight"]), kb_norm=P(w[a + "k_norm.weight"]),
                                          q_out=P(self.qr))))
            calls.append((ik.mtp_attention, dict(pos=P(rt.pos), kv=P(self.kv), cap=self.cap, layer=l,
                                                 window=WINDOW if l in MTP_GEOMETRY["local_layer_ids"] else 0,
                                                 q_out=P(self.qr), attn_out=P(self.att))))
            calls.append(("gemv", [g(w[a + "o_proj.weight"], self.att, self.o, D, HQ * HD)], []))
            calls.append((ik.mtp_norm, dict(dh=P(self.dh), add=P(self.o),
                                            norm_w=P(w[p + "post_attention_layernorm.weight"]), x=P(self.x))))
            calls.append(("gemv", [g(w[p + "mlp.gate_proj.weight"], self.x, self.g, INTER, D),
                                   g(w[p + "mlp.up_proj.weight"], self.x, self.u, INTER, D)], []))
            calls.append((ik.mtp_silu, dict(gate=P(self.g), up=P(self.u), act=P(self.act))))
            calls.append(("gemv", [g(w[p + "down_a"], self.act, self.mlp_a, D, INTER // 2, yf=1, xs=INTER),
                                   g(w[p + "down_b"], P(self.act) + INTER, self.mlp_b, D, INTER // 2, yf=1,
                                     xs=INTER)], []))
        # the last MLP into dh; base logits and the normalized rows 1..steps from the LM head (no softcap)
        calls.append((ik.mtp_norm, dict(dh=P(self.dh), add_part=[P(self.mlp_a), P(self.mlp_b)],
                                        norm_w=P(w["norm.weight"]), x=P(self.hn))))
        fp8_head = dict(weight_fp8=P(self.head8), weight_scale=P(self.head_scale)) if self.fp8 else {}
        calls.append((ik.lm_head, dict(rows=steps, in_stride=D, **{"in": P(self.dh) + D * 4},
                                       norm_w=P(w["norm.weight"]), weight=P(rt.w["unembed.weight"]), vocab=self.vocab,
                                       softcap=0.0, blocks=rt.lm_blocks, logits=P(self.logits),
                                       block_best=P(self.lm_best), block_index=P(self.lm_index),
                                       counter=P(self.lm_counter), token=P(self.lm_token), **fp8_head)))
        calls.append(("zero", self.state))
        fp8_fold = dict(fold_w8=P(self.fold8), fold_scale=P(self.fold_scale)) if self.fp8 else {}
        for step in range(steps):
            calls.append((ik.mtp_rnn_in, dict(step=step, anchor=anchor, drafts=drafts,
                                              markov_w1=P(w["markov_head.markov_w1.weight"]), state=P(self.state),
                                              hn=P(self.hn), frnn=P(self.frnn), rnn_in=P(self.rnn_in))))
            calls.append(("gemv", [(P(w["markov_head.rnn_proj.weight"]), P(self.rnn_in), P(self.proj), 3 * RANK,
                                    2 * RANK + 2 * D, 0, 1, ik.GEMV_PLAIN, 0, 1)], []))
            calls.append((ik.mtp_gates, dict(proj=P(self.proj), rnn_bias=P(w["markov_head.rnn_proj.bias"]),
                                             state=P(self.state), latent=P(self.latent))))
            calls.append((ik.mtp_pick, dict(step=step, latent=P(self.latent), logits=P(self.logits),
                                            vocab_stride=self.vocab, vocab=self.vocab, fold_w=P(self.fold_w),
                                            fold_b=P(self.fold_b), pick_best=P(self.pick_best),
                                            pick_index=P(self.pick_index), pick_counter=P(self.pick_counter),
                                            drafts=drafts, **fp8_fold)))
        return calls

    def add_width(self, W: int) -> str:
        """Programs ``first_draft<W>`` and ``round<W>`` (verification of the anchor and W - 1 drafts with the
        tokens logged, the context of the accepted rows, the next W - 1 drafts); returns the round's name."""
        ik, rt = _require_module(), self.rt
        if f"round{W}" in rt.programs:
            return f"round{W}"
        verify = rt.programs[rt.program("verify", W)]
        logged = [(ik.accept, dict(c[1], log=P(self.log), log_count=P(self.log_count))) if c[0] is ik.accept else c
                  for c in verify]
        rt.programs[f"round{W}"] = logged + self._context(rt.accepted, 1) + self._block_and_walk(W - 1)
        rt.programs[f"first_draft{W}"] = self._block_and_walk(W - 1)
        return f"round{W}"

    def set_width(self, W: int):
        """Verify W rows per speculative step from the next ``prefill`` on."""
        if not 2 <= W <= min(BLOCK, self.rt.exact_rows):
            raise ValueError(f"width must be in 2..{min(BLOCK, self.rt.exact_rows)}")
        self.round, self.width = self.add_width(W), int(W)

    # ------------------------------------------------------------------------------------------- generation
    def prefill(self, prompt, exact: bool = False) -> int:
        """The prompt (``rt.prefill``) with the drafter context of every position, then the first drafts; returns
        the first token (also the anchor in ``vtokens[0]``)."""
        rt = self.rt
        ctx = {"chunk": "ctx_chunk", "exact": "ctx_exact", "step": "ctx1"}
        first = rt.prefill(prompt, exact=exact, after=lambda kind, rows: rt.run(ctx[kind]))
        rt.vtokens[0].copy_(rt.token.view(-1)[0])
        rt.run(f"first_draft{self.width}")
        return first

    def steps(self, first: int, max_new_tokens: int, check_every: int = 4) -> Iterator[List[int]]:
        """Tokens of the speculative steps after ``prefill`` (``first`` included), read back every ``check_every``
        steps, until ``max_new_tokens`` are produced (the last block is cut there; ``rounds`` counts the steps)."""
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
