"""Inkling 2 Mini speculative decoding with the checkpoint's DSpark drafter (``mtp.safetensors``) on Apple Silicon.

The drafter follows SGLang's ``Inkling2ForConditionalGenerationMTP`` (a DFlash block of 8 positions over the
target's features, plus the RNN Markov head of DSpark): ``fuse_weight`` softmax mixes of the target features
(the hidden state entering visits 5, 23, 41 and the final normalized state) give each drafter layer its context
K / V and the Markov head its recurrent input; the Markov head's ``hidden_out`` is folded into the LM head
(``W_lm @ hidden_out``) so a drafted position's bias is one (vocab x 256) product. Greedy drafting and greedy
verification: a step drafts after the anchor, the target scores the anchor and the first W - 1 drafts in one row
program (W by batch size, ``mtp_widths``), and the longest agreeing prefix plus the target's next token are
committed. Verification uses the row programs' numerics, so where two candidates are within the rounding of the
logits the output can differ from the single-row decode step's greedy choice.

Mixed into ``Inkling2Runtime`` (programs must be recorded before the plan is sealed).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from flash_rt.hardware.apple_native.native import READ, READ_WRITE, WRITE
from flash_rt.models.inkling2.checkpoint import SafetensorsFile

D_HEAD = 5                        # LM head descriptor of the final table row
MTP_LAYERS = 3
MTP_BLOCK = 8
MTP_RANK = 256
MTP_INTER = 9216
MTP_THETA = 1.0e6
MTP_WINDOW = 1024
STAGE_ELEMS = 6656 + 2 * 8 * 128


class _MTPMixin:
    # ------------------------------------------------------------------ weights
    def _mtp_upload(self, layout) -> None:
        cfg = self.cfg
        H, V = cfg.hidden_size, cfg.vocab_size
        st = SafetensorsFile(Path(self.ckpt.directory) / "mtp.safetensors")
        w: Dict[str, int] = {}

        def bf16(name, shape):
            a = st.tensor(name)
            if st.dtype(name) != "BF16" or tuple(a.shape) != tuple(shape):
                raise ValueError(f"mtp.safetensors: {name} is {st.dtype(name)} {a.shape}, expected BF16 {shape}")
            return layout.add(a)

        for l in range(MTP_LAYERS):
            p = f"layers.{l}."
            a = p + "self_attn."
            w[f"{l}.dflare"] = bf16(p + "dflare_norm.weight", (H,))
            w[f"{l}.in_norm"] = bf16(p + "input_layernorm.weight", (H,))
            w[f"{l}.post_norm"] = bf16(p + "post_attention_layernorm.weight", (H,))
            w[f"{l}.q"] = bf16(a + "q_proj.weight", (3072, H))
            w[f"{l}.k"] = bf16(a + "k_proj.weight", (1024, H))
            w[f"{l}.v"] = bf16(a + "v_proj.weight", (1024, H))
            w[f"{l}.o"] = bf16(a + "o_proj.weight", (H, 3072))
            w[f"{l}.qn"] = bf16(a + "q_norm.weight", (128,))
            w[f"{l}.kn"] = bf16(a + "k_norm.weight", (128,))
            w[f"{l}.knc"] = bf16(a + "k_norm_ctx.weight", (128,))
            w[f"{l}.kc"] = bf16(a + "k_proj_ctx.weight", (1024, H))
            w[f"{l}.vc"] = bf16(a + "v_proj_ctx.weight", (1024, H))
            w[f"{l}.gate"] = bf16(p + "mlp.gate_proj.weight", (MTP_INTER, H))
            w[f"{l}.up"] = bf16(p + "mlp.up_proj.weight", (MTP_INTER, H))
            w[f"{l}.down"] = bf16(p + "mlp.down_proj.weight", (H, MTP_INTER))
        w["norm"] = bf16("norm.weight", (H,))
        w["mask"] = bf16("mask_embed", (H,))
        w["m.dflare"] = bf16("markov_head.dflare_norm.weight", (H,))
        w["m.w1"] = bf16("markov_head.markov_w1.weight", (V, MTP_RANK))
        w["m.rnn"] = bf16("markov_head.rnn_proj.weight", (3 * MTP_RANK, 2 * MTP_RANK + 2 * H))
        w["m.rnn_b"] = bf16("markov_head.rnn_proj.bias", (3 * MTP_RANK,))
        # fuse alpha: softmax(fuse_weight) in FP32, rounded to BF16 (the reference keeps the parameter dtype)
        fw = st.tensor("fuse_weight").astype(np.uint32) << 16
        fw = fw.view(np.float32).reshape(MTP_LAYERS + 1, 4).astype(np.float32)
        e = np.exp(fw - fw.max(axis=1, keepdims=True))
        alpha = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        self._mtp_alpha = _bf16_round(alpha)
        # fold: W_lm @ hidden_out (FP64 in row chunks, stored BF16 like the reference)
        head = self.ckpt.tensor("unembed.weight", "BF16", (V, H))
        hw = _bf16_to_f32(st.tensor("markov_head.hidden_out.weight")).astype(np.float64)   # [H, 256]
        hb = _bf16_to_f32(st.tensor("markov_head.hidden_out.bias")).astype(np.float64)     # [H]
        fold_w = np.empty((V, MTP_RANK), np.uint16)
        fold_b = np.empty((V,), np.uint16)
        for s in range(0, V, 8192):
            rows = _bf16_to_f32(head[s:s + 8192]).astype(np.float64)
            fold_w[s:s + 8192] = _f32_to_bf16(rows @ hw)
            fold_b[s:s + 8192] = _f32_to_bf16(rows @ hb)
        w["fold_w"] = layout.add(fold_w)
        w["fold_b"] = layout.add(fold_b)
        self._mtp_file = st              # closed once the layout has been copied (its arrays view the file)
        self.mtp_w = w

    # ------------------------------------------------------------------ buffers
    def _mtp_allocate(self) -> None:
        H, V = self.cfg.hidden_size, self.cfg.vocab_size
        S = self.sequences
        RV = MTP_BLOCK * S                       # verify / draft rows
        RC = max(self.row_capacity, RV)          # context rows (a prefill chunk or a verify step)
        self.mtp_cap = self.max_seq + MTP_BLOCK
        self.mtp_seq_kv = MTP_LAYERS * 2 * 8 * self.mtp_cap * 128
        f32, bf = 4, 2
        sizes = {
            "m_kv": self.mtp_seq_kv * S * bf, "m_feat": 4 * RC * H * bf,
            "m_stage": len(self.cfg.visits()) * RV * STAGE_ELEMS * bf,
            "m_commit": RV * 2 * 4, "m_cinfo": RC * 16, "m_dinfo": RV * 16, "m_anchor": S * 4,
            "m_emb": S * H * bf, "m_ctxx": 3 * RC * H * bf, "m_kvc": RC * 3 * 2048 * bf,
            "m_frnn": S * H * bf, "m_h": RV * H * f32, "m_x": RV * H * bf, "m_qkv": RV * 5120 * bf,
            "m_q": RV * 3072 * bf, "m_attn": RV * 3072 * bf, "m_o": RV * H * bf, "m_gu": RV * 2 * MTP_INTER * bf,
            "m_act": RV * MTP_INTER * bf, "m_mlp": RV * H * bf, "m_hn": RV * H * bf, "m_logits": RV * V * f32,
            "m_state": S * MTP_RANK * bf, "m_rnnin": S * (2 * MTP_RANK + 2 * H) * bf, "m_proj": S * 3 * MTP_RANK * f32,
            "m_latent": S * MTP_RANK * bf, "m_bias": S * V * f32, "m_prev": S * 4, "m_drafts": S * 7 * 4,
        }
        for name, nbytes in sizes.items():
            self.buf[name] = self.plan.buffer(nbytes)

    # verified rows per sequence and step (the anchor + W - 1 drafts) by batch size: verification grows with the
    # rows while the accepted length saturates, so larger batches verify fewer drafts
    mtp_widths = {1: 8, 4: 8, 16: 4, 64: 3}

    def mtp_width(self, C: int) -> int:
        for limit in sorted(self.mtp_widths):
            if C <= limit:
                return self.mtp_widths[limit]
        return self.mtp_widths[max(self.mtp_widths)]

    # ------------------------------------------------------------------ programs
    def _mtp_record(self) -> None:
        """Programs per batch size C in self.mtp_batches: verify (8C rows, staged), commit, draft (context of the
        last verify's accepted rows, the block, the Markov walk); per prefill length: features + context."""
        self.mtp_programs: Dict[tuple, int] = {}
        for C in self.mtp_batches:
            W = self.mtp_width(C)
            R = W * C
            self.mtp_programs[("verify", C)] = self._record_rows(R, R, stage=True, features=True)
            self.mtp_programs[("commit", C)] = self._mtp_record_commit(R)
            self.mtp_programs[("draft", C)] = self._mtp_record_draft(C, W)
        for P in self.mtp_prefill:
            self.mtp_programs[("prefill", P)] = self._record_rows(P, 1, features=True)
            self.mtp_programs[("context", P)] = self._mtp_record_context_only(P)

    def _mtp_record_commit(self, R: int) -> int:
        b, plan = self.buf, self.plan
        program = plan.program()
        nv = len(self.cfg.visits())
        kv_off = np.zeros(48, np.uint64)
        ring_off = np.zeros(48, np.uint64)
        glob = np.zeros(48, np.uint32)
        for v, (layer, _) in enumerate(self.cfg.visits()):
            row = self.table[self.cfg.num_layers + 1 + v]
            kv_off[v], ring_off[v] = row[2], row[4]
            glob[v] = 1 if self.cfg.is_global(layer) else 0
        blob = np.array([self.kv_elements, self.ring_elements], np.uint64).tobytes() + \
            np.array([R, nv, 0, 0], np.uint32).tobytes() + kv_off.tobytes() + ring_off.tobytes() + glob.tobytes()
        prm = self._param(blob)
        plan.dispatch("ink_commit_rows", [(b["m_stage"], 0, READ), (b["rings"], 0, READ_WRITE), (b["kv"], 0, READ_WRITE),
                                          (b["r_info"], 0, READ), (b["m_commit"], 0, READ), (b["args"], prm, READ)],
                      (R, nv, 1), 256, label="commit")
        return program

    def _mtp_context(self, R: int) -> None:
        """Context K / V of rows flagged in m_cinfo (features of a prefill chunk or a verify step, R rows)."""
        b, plan, w = self.buf, self.plan, self.mtp_w
        H = self.cfg.hidden_size
        A = b["args"]
        a = self._mtp_alpha.reshape(-1)
        mix = self._param(np.array([w["0.dflare"], w["1.dflare"], w["2.dflare"], w["m.dflare"]], np.uint64).tobytes() +
                          a.astype(np.float32).tobytes() + np.array([R, 0, 0, 0], np.uint32).tobytes())
        plan.dispatch("ink_mtp_mix_rows", [(b["weights"], 0, READ), (b["m_feat"], 0, READ), (b["m_ctxx"], 0, WRITE),
                                           (b["m_frnn"], 0, READ_WRITE), (b["m_cinfo"], 0, READ), (A, mix, READ)],
                      (R, 1, 1), 256, label="mtp_mix")
        mats = []
        for l in range(MTP_LAYERS):
            mats.append((w[f"{l}.kc"], 1024, H, l, H, 0, l * 2048, 3 * 2048))
            mats.append((w[f"{l}.vc"], 1024, H, l, H, 0, l * 2048 + 1024, 3 * 2048))
        self._gemm_multi("ink_gemm_bf16_multi", R, mats, [(b["m_ctxx"], l * R * H * 2) for l in range(MTP_LAYERS)],
                         [(b["m_kvc"], 0)], "mtp_ctx_kv")
        ctx = self._param(np.array([w["0.knc"], w["1.knc"], w["2.knc"], self.mtp_seq_kv], np.uint64).tobytes() +
                          np.array([R, self.mtp_cap], np.uint32).tobytes() + np.array([MTP_THETA], np.float32).tobytes() +
                          np.zeros(1, np.uint32).tobytes())
        plan.dispatch("ink_mtp_ctx_write_rows", [(b["weights"], 0, READ), (b["m_kvc"], 0, READ), (b["m_kv"], 0, READ_WRITE),
                                                 (b["m_cinfo"], 0, READ), (A, ctx, READ)], (R, 1, 1), 256,
                      label="mtp_ctx_write")

    def _mtp_record_context_only(self, R: int) -> int:
        program = self.plan.program()
        self._mtp_context(R)
        return program

    def _mtp_record_draft(self, C: int, W: int) -> int:
        b, plan, w, cfg = self.buf, self.plan, self.mtp_w, self.cfg
        H, V = cfg.hidden_size, cfg.vocab_size
        R = MTP_BLOCK * C
        A = b["args"]
        program = plan.program()
        self._mtp_context(W * C)                  # the last verify step's rows
        # anchors' normalized embeddings (the target's embedding kernel), block rows
        rv0 = self._visit_param(0, C, 0, [], 0)
        plan.dispatch("ink_embed_rows", [(b["weights"], 0, READ), (b["table"], 0, READ), (b["m_anchor"], 0, READ),
                                         (b["m_emb"], 0, WRITE), (A, rv0, READ)], (C, 1, 1), 256, label="mtp_anchor")
        emb = self._param(np.array([w["mask"]], np.uint64).tobytes() + np.array([R, 0], np.uint32).tobytes())
        plan.dispatch("ink_mtp_embed_rows", [(b["weights"], 0, READ), (b["m_emb"], 0, READ), (b["m_h"], 0, WRITE),
                                             (b["m_dinfo"], 0, READ), (A, emb, READ)], (R, 1, 1), 256, label="mtp_embed")

        def norm(weight, add, final=False):
            flags = (1 if add else 0) | (2 if final else 0)
            prm = self._param(np.array([weight], np.uint64).tobytes() + np.array([R, flags], np.uint32).tobytes())
            plan.dispatch("ink_mtp_norm_rows", [(b["weights"], 0, READ), (b["m_h"], 0, READ_WRITE),
                                                (b[add] if add else b["m_o"], 0, READ),
                                                (b["m_hn"] if final else b["m_x"], 0, WRITE), (A, prm, READ),
                                                (b["m_dinfo"], 0, READ)], (R, 1, 1), 256, label="mtp_norm")

        layer_elems = 2 * 8 * self.mtp_cap * 128
        for l in range(MTP_LAYERS):
            norm(w[f"{l}.in_norm"], "m_mlp" if l else None)
            self._gemm_multi("ink_gemm_bf16_multi", R, [(w[f"{l}.q"], 3072, H, 0, H, 0, 0, 5120),
                                                        (w[f"{l}.k"], 1024, H, 0, H, 0, 3072, 5120),
                                                        (w[f"{l}.v"], 1024, H, 0, H, 0, 4096, 5120)],
                             [(b["m_x"], 0)], [(b["m_qkv"], 0)], "mtp_qkv")
            qk = self._param(np.array([w[f"{l}.qn"], w[f"{l}.kn"], self.mtp_seq_kv, l * layer_elems], np.uint64).tobytes() +
                             np.array([R, self.mtp_cap], np.uint32).tobytes() +
                             np.array([MTP_THETA], np.float32).tobytes() + np.zeros(1, np.uint32).tobytes())
            plan.dispatch("ink_mtp_qk_rows", [(b["weights"], 0, READ), (b["m_qkv"], 0, READ), (b["m_q"], 0, WRITE),
                                              (b["m_kv"], 0, READ_WRITE), (b["m_dinfo"], 0, READ), (A, qk, READ)],
                          (R, 1, 1), 256, label="mtp_qk")
            at = self._param(np.array([self.mtp_seq_kv, l * layer_elems], np.uint64).tobytes() +
                             np.array([R, self.mtp_cap, MTP_WINDOW if l in (0, 2) else 0, 0], np.uint32).tobytes())
            plan.dispatch("ink_mtp_attn_rows", [(b["m_q"], 0, READ), (b["m_kv"], 0, READ), (b["m_attn"], 0, WRITE),
                                                (b["m_dinfo"], 0, READ), (A, at, READ)], (8, R, 1), 256, label="mtp_attn")
            self._gemm("ink_gemm_bf16_rows", b["m_attn"], 0, 3072, w[f"{l}.o"], H, 3072, b["m_o"], 0, H, 0, R, True)
            norm(w[f"{l}.post_norm"], "m_o")
            self._gemm_multi("ink_gemm_bf16_multi", R, [(w[f"{l}.gate"], MTP_INTER, H, 0, H, 0, 0, 2 * MTP_INTER),
                                                        (w[f"{l}.up"], MTP_INTER, H, 0, H, 0, MTP_INTER, 2 * MTP_INTER)],
                             [(b["m_x"], 0)], [(b["m_gu"], 0)], "mtp_gate_up")
            nr = self._param(np.array([R], np.uint32).tobytes())
            plan.dispatch("ink_mtp_silu_rows", [(b["m_gu"], 0, READ), (b["m_act"], 0, WRITE), (A, nr, READ)],
                          ((R * MTP_INTER + 255) // 256, 1, 1), 256, label="mtp_silu")
            self._gemm("ink_gemm_bf16_rows", b["m_act"], 0, MTP_INTER, w[f"{l}.down"], H, MTP_INTER, b["m_mlp"], 0, H, 0,
                       R, True)
        norm(w["norm"], "m_mlp", final=True)
        f = self.table[cfg.num_layers]
        nd = W - 1                                 # drafted positions verified per step
        self._gemm("ink_gemm_f32_rows", b["m_hn"], 0, H, int(f[D_HEAD]), V, H, b["m_logits"], 0, V, 0, nd * C, True)
        # Markov walk
        for step in range(nd):
            rnn = self._param(np.array([w["m.w1"]], np.uint64).tobytes() + np.array([C, step], np.uint32).tobytes())
            plan.dispatch("ink_mtp_rnn_in", [(b["weights"], 0, READ), (b["m_hn"], 0, READ), (b["m_frnn"], 0, READ),
                                             (b["m_state"], 0, READ), (b["m_prev"], 0, READ), (b["m_rnnin"], 0, WRITE),
                                             (A, rnn, READ)], (C, 1, 1), 256, label="mtp_rnn_in")
            K = 2 * MTP_RANK + 2 * H
            self._gemm("ink_gemm_f32_rows", b["m_rnnin"], 0, K, w["m.rnn"], 3 * MTP_RANK, K, b["m_proj"], 0,
                       3 * MTP_RANK, 0, C, True)
            gt = self._param(np.array([w["m.rnn_b"]], np.uint64).tobytes() + np.array([C, 0], np.uint32).tobytes())
            plan.dispatch("ink_mtp_gates", [(b["weights"], 0, READ), (b["m_proj"], 0, READ), (b["m_state"], 0, READ_WRITE),
                                            (b["m_latent"], 0, WRITE), (A, gt, READ)], ((C * MTP_RANK + 255) // 256, 1, 1),
                          256, label="mtp_gates")
            self._gemm("ink_gemm_f32_rows", b["m_latent"], 0, MTP_RANK, w["fold_w"], V, MTP_RANK, b["m_bias"], 0, V, 0,
                       C, True)
            pk = self._param(np.array([w["fold_b"]], np.uint64).tobytes() +
                             np.array([C, step, cfg.unpadded_vocab_size, V], np.uint32).tobytes())
            plan.dispatch("ink_mtp_pick", [(b["weights"], 0, READ), (b["m_logits"], 0, READ), (b["m_bias"], 0, READ),
                                           (b["m_prev"], 0, READ_WRITE), (b["m_drafts"], 0, WRITE), (A, pk, READ)],
                          (C, 1, 1), 1024, label="mtp_pick")
        return program

    # ------------------------------------------------------------------ generation
    def mtp_generate(self, prompts: Sequence[Sequence[int]], new_tokens: int, stats: dict | None = None
                     ) -> List[List[int]]:
        """Greedy generation of exactly ``new_tokens`` tokens per prompt (C = len(prompts) sequences, all prompts
        of one length with a recorded prefill program) with DSpark drafting. Returns the generated tokens."""
        C = len(prompts)
        P = len(prompts[0])
        if ("draft", C) not in self.mtp_programs or ("prefill", P) not in self.mtp_programs:
            raise ValueError(f"no MTP programs for {C} sequences / prompt length {P}")
        if any(len(p) != P for p in prompts):
            raise ValueError("prompts must have one length")
        if P + new_tokens + MTP_BLOCK > self.max_seq:
            raise ValueError("prompt + new tokens exceed max_seq")
        plan, b = self.plan, self.buf
        W = self.mtp_width(C)
        R = W * C
        RB = MTP_BLOCK * C
        out: List[List[int]] = [[] for _ in range(C)]
        anchor = np.zeros(C, np.uint32)
        pos = np.full(C, P, np.int64)
        # prefill: target rows with features, then the drafter context of every prompt position
        for s, prompt in enumerate(prompts):
            info = np.array([[s, i, 0, P - 1] for i in range(P)], np.uint32)
            plan.write(b["r_tokens"], 0, np.asarray(prompt, np.uint32))
            plan.write(b["r_info"], 0, info)
            plan.run(self.mtp_programs[("prefill", P)])
            anchor[s] = plan.read(b["r_out"], 0, np.uint32, 1)[0]
            ci = np.zeros((P, 4), np.uint32)
            ci[:, 0], ci[:, 1], ci[:, 2] = s, np.arange(P), 1
            ci[-1, 3] = 1
            plan.write(b["m_cinfo"], 0, ci)
            plan.run(self.mtp_programs[("context", P)])
            out[s].append(int(anchor[s]))
        cinfo = np.zeros((R, 4), np.uint32)            # first draft step: no new context rows
        steps = accepted = active = 0
        gpu = {"draft": 0.0, "verify": 0.0, "commit": 0.0}
        t_start = time.perf_counter()
        while min(len(o) for o in out) < new_tokens:
            # draft
            dinfo = np.zeros((RB, 4), np.uint32)
            for s in range(C):
                dinfo[s * 8:(s + 1) * 8] = [[s, pos[s] + j, pos[s], j] for j in range(8)]
            plan.write(b["m_cinfo"], 0, cinfo)
            plan.write(b["m_dinfo"], 0, dinfo)
            plan.write(b["m_anchor"], 0, anchor)
            plan.write(b["m_prev"], 0, anchor)
            plan.view(b["m_state"], np.uint16, C * MTP_RANK)[:] = 0
            gpu["draft"] += plan.run(self.mtp_programs[("draft", C)])["gpu_us"]
            drafts = plan.read(b["m_drafts"], 0, np.uint32, C * 7).reshape(C, 7)[:, :W - 1]
            # verify
            toks = np.concatenate([anchor[:, None], drafts], axis=1).reshape(-1)
            vinfo = np.zeros((R, 4), np.uint32)
            for s in range(C):
                vinfo[s * W:(s + 1) * W] = [[s, pos[s] + j, pos[s], pos[s] + W - 1] for j in range(W)]
            plan.write(b["r_tokens"], 0, toks.astype(np.uint32))
            plan.write(b["r_info"], 0, vinfo)
            gpu["verify"] += plan.run(self.mtp_programs[("verify", C)])["gpu_us"]
            target = plan.read(b["r_out"], 0, np.uint32, R).reshape(C, W)
            commit = np.zeros((R, 2), np.uint32)
            cinfo = np.zeros((R, 4), np.uint32)
            for s in range(C):
                if len(out[s]) >= new_tokens:
                    continue                    # finished: frozen (rows recomputed, nothing committed)
                n = 0
                while n < W - 1 and drafts[s, n] == target[s, n]:
                    n += 1
                new = list(drafts[s, :n]) + [target[s, n]]
                out[s].extend(int(t) for t in new)
                commit[s * W:s * W + n + 1, 0] = 1
                commit[s * W:(s + 1) * W, 1] = pos[s] + n
                cinfo[s * W:s * W + n + 1] = [[s, pos[s] + j, 1, 1 if j == n else 0] for j in range(n + 1)]
                anchor[s] = target[s, n]
                pos[s] += n + 1
                accepted += n + 1
                active += 1
            plan.write(b["m_commit"], 0, commit)
            gpu["commit"] += plan.run(self.mtp_programs[("commit", C)])["gpu_us"]
            steps += 1
        if stats is not None:
            stats.update(steps=steps, tokens_per_step=accepted / max(1, active), decode_s=time.perf_counter() - t_start,
                         **{k + "_ms_per_step": v / 1000 / max(1, steps) for k, v in gpu.items()})
        return [o[:new_tokens] for o in out]


def _bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_round(a: np.ndarray) -> np.ndarray:
    return _bf16_to_f32(_f32_to_bf16(a))
