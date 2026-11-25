"""Inkling 2 Mini speculative decoding with the checkpoint's DSpark drafter (``mtp.safetensors``) on Apple Silicon.

The drafter follows SGLang's ``Inkling2ForConditionalGenerationMTP`` (a DFlash block of 8 positions over the
target's features, plus the RNN Markov head of DSpark): ``fuse_weight`` softmax mixes of the target features
(the hidden state entering visits 5, 23, 41 and the final normalized state) give each drafter layer its context
K / V and the Markov head its recurrent input; the Markov head's ``hidden_out`` is folded into the LM head
(``W_lm @ hidden_out``) so a drafted position's bias is one (vocab x 256) product. Greedy drafting and greedy
verification: a step drafts after the anchor, the target scores the anchor and the first W - 1 drafts (W by batch
size, ``mtp_widths``), and the longest agreeing prefix plus the target's next token are committed. Verification of
one sequence (``mtp_verify``): "fast" (default), a resident step over the rows with its own summation order (half
code products, position-block attention; within the rounding of the decode step); "resident", a resident step with
every row the decode step's arithmetic (the output is the greedy decode's bit for bit); "rows", the row programs (also
for several sequences). With the fast verification of one sequence a step verifies 2-8 rows, chosen from the walk's
probabilities of the drafts (``mtp_adaptive``); its drafter layers and the LM head of its base logits run from Q8_0
copies. The Markov walk ranks the 256 largest base logits of a drafted position: drafts only decide how many tokens
a step accepts. A step's commit runs in the next step's draft command buffer.

Mixed into ``Inkling2Runtime`` (programs must be recorded before the plan is sealed).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np

from flash_rt.frontends.torch._inkling2_dflash2 import DF_MASK
from flash_rt.hardware.apple_native.native import READ, READ_WRITE, WRITE
from flash_rt.models.inkling2.checkpoint import SafetensorsFile, check_mtp_config

D_HEAD = 5                        # LM head descriptor of the final table row
MTP_LAYERS = 3
MTP_BLOCK = 8
MTP_RANK = 256
MTP_INTER = 9216
MTP_THETA = 1.0e6
MTP_WINDOW = 1024
STAGE_ELEMS = 6656 + 2 * 8 * 128
MTP_CAND = 256                    # Markov walk candidates per drafted position (ink_mtp_topk)
# resident verification step (inkling2_verify_resident_bf16.h): per-row workspace and the step's shared tail
VERIFY_ROW_BYTES = 1177792 + 83072
VERIFY_MAX = 16                   # rows of a resident step at most (the drafts and lookup drafts after them)
VERIFY_WORKSPACE = VERIFY_MAX * VERIFY_ROW_BYTES + 8192
VERIFY_ROWS = (2, 3, 4, 5, 6, 7, 8, 12, 16)


class _MTPMixin:
    # ------------------------------------------------------------------ weights
    def _mtp_upload(self, layout) -> None:
        cfg = self.cfg
        H, V = cfg.hidden_size, cfg.vocab_size
        check_mtp_config(json.loads((Path(self.ckpt.directory) / "config.json").read_text()))
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

    def _mtp_q8_matrices(self):
        """Drafter layer matrices with a Q8_0 copy: (name, byte offset, rows, inputs)."""
        H, w = self.cfg.hidden_size, self.mtp_w
        out = []
        for l in range(MTP_LAYERS):      # q | k | v and gate | up stacked: one product each, outputs in that layout
            out.append((f"mtp{l}.qkv", [(int(w[f"{l}.q"]), 3072), (int(w[f"{l}.k"]), 1024), (int(w[f"{l}.v"]), 1024)],
                        5120, H))
            out.append((f"mtp{l}.o", int(w[f"{l}.o"]), H, 3072))
            out.append((f"mtp{l}.gate_up", [(int(w[f"{l}.gate"]), MTP_INTER), (int(w[f"{l}.up"]), MTP_INTER)],
                        2 * MTP_INTER, H))
            out.append((f"mtp{l}.down", int(w[f"{l}.down"]), H, MTP_INTER))
        return out

    def _q8_prep(self, x, x_off, x_stride, rows, k, label):
        """Inputs of q8_0_rows8: rows x k BF16 -> half times a power of two per row (m_xh, m_xsc)."""
        prm = self._param(np.array([rows, 0, k, x_stride, 0, 0, 0, 0], np.uint32).tobytes())
        self.plan.dispatch("q8_0_prep_rows", [(x, x_off, READ), (self.buf["m_xh"], 0, WRITE),
                                              (self.buf["m_xsc"], 0, WRITE), (self.buf["args"], prm, READ)],
                           (rows, 1, 1), 256, label=label + "_prep")

    def _q8_rows8(self, name, rows, y, y_stride, y_col, label, row0=0, nrows=None):
        """y[r][y_col + n] (BF16) = Q8_0 matrix `name` (its rows row0 .. row0 + nrows) x the prepared inputs,
        rows <= 16."""
        _, n, k, codes, scales = self._q8[name]
        codes, scales = codes + row0 * k, scales + row0 * (k // 32) * 2
        n = n - row0 if nrows is None else nrows
        kc = next(c for c in (1536, 1024, 768, 512, 256) if k % c == 0 and c * max(rows, 8) <= 12288)
        tiles = 2 if (n + 127) // 128 >= self.q8_groups and rows <= 8 else 1      # 64-row tiles per work item
        prm = self._param(np.array([rows, n, k, 0, y_stride, y_col, kc, 1 | (2 if tiles == 1 else 0)],
                                   np.uint32).tobytes())
        self.plan.dispatch("q8_0_rows8" if rows <= 8 else "q8_0_rows16", [(self.buf["m_xh"], 0, READ),
                                                                       (self.buf["m_xsc"], 0, READ),
                                          (self.buf["q8"], codes, READ), (self.buf["q8"], scales, READ),
                                          (y, 0, WRITE), (self.buf["args"], prm, READ)],
                           (min(self.q8_groups, (n + 64 * tiles - 1) // (64 * tiles)), 1, 1), 256, label=label)

    # ------------------------------------------------------------------ buffers
    def _mtp_allocate(self) -> None:
        H, V = self.cfg.hidden_size, self.cfg.vocab_size
        S = self.sequences
        RV = MTP_BLOCK * S                       # draft rows
        RW = max([RV] + [self.mtp_width(c) * c for c in self.mtp_batches])   # verify rows
        RC = max(self.row_capacity, RW)          # context rows (a prefill chunk or a verify step)
        self.mtp_cap = self.max_seq + MTP_BLOCK
        self.mtp_seq_kv = MTP_LAYERS * 2 * 8 * self.mtp_cap * 128
        f32, bf = 4, 2
        sizes = {
            "m_kv": self.mtp_seq_kv * S * bf, "m_feat": 4 * RC * H * bf,
            "m_stage": len(self.cfg.visits()) * RW * STAGE_ELEMS * bf,
            "m_commit": RW * 2 * 4, "m_cinfo": RC * 16, "m_dinfo": RV * 16, "m_anchor": S * 4,
            "m_emb": S * H * bf, "m_ctxx": 3 * RC * H * bf, "m_kvc": RC * 3 * 2048 * bf,
            "m_frnn": S * H * bf, "m_h": RV * H * f32, "m_x": RV * H * bf, "m_qkv": RV * 5120 * bf,
            "m_q": RV * 3072 * bf, "m_attn": RV * 3072 * bf, "m_o": RV * H * bf, "m_gu": RV * 2 * MTP_INTER * bf,
            "m_act": RV * MTP_INTER * bf, "m_mlp": RV * H * bf, "m_hn": RV * H * bf, "m_logits": RV * V * f32,
            "m_state": S * MTP_RANK * bf, "m_rnnin": S * (2 * MTP_RANK + 2 * H) * bf, "m_proj": S * 3 * MTP_RANK * f32,
            "m_latent": S * MTP_RANK * bf, "m_bias": S * V * f32, "m_prev": S * 4, "m_drafts": S * 7 * 4,
            "v_ws": VERIFY_WORKSPACE, "v_params": 16,
            "m_cidx": S * 7 * MTP_CAND * 4, "m_cval": S * 7 * MTP_CAND * f32, "m_dtop": S * 7 * 16 * f32,
            "m_xh": VERIFY_MAX * MTP_INTER * 2, "m_xsc": VERIFY_MAX * f32,
        }
        for name, nbytes in sizes.items():
            self.buf[name] = self.plan.buffer(nbytes)

    # verification program of one sequence: "rows" (the row program's numerics) or "resident" (the resident step:
    # every row the decode step's arithmetic, so the output is the greedy decode's continuation bit for bit)
    mtp_verify = "fast"
    # Markov walk over the MTP_CAND largest base logits of each drafted position (0: the whole vocabulary); drafts
    # only decide how many tokens a step accepts, never which
    mtp_draft_candidates = MTP_CAND

    # verified rows per sequence and step (the anchor + W - 1 drafts) by batch size: verification grows with the
    # rows while the accepted length saturates, so larger batches verify fewer drafts
    mtp_widths = {1: VERIFY_MAX, 4: 8, 16: 4, 64: 3}

    # one sequence, fast verification: rows per step chosen from the drafts' walk probabilities, maximizing expected
    # tokens - rate x verification time (rate: measured tokens per ms; time: a relative cost by rows, scaled by the
    # measured times). Agreement probability of a drafted token by its walk probability (measured on the corpus):
    mtp_adaptive = True
    ADAPT_P = (0.0, 0.308, 0.503, 0.707, 0.852, 0.927, 0.976, 0.999, 1.0)
    ADAPT_A = (0.0, 0.19, 0.37, 0.585, 0.62, 0.744, 0.82, 0.981, 0.99)
    # the agreement rates follow the text: per probability bin, accepted / reached drafted positions of the
    # verified steps, the table above as a prior of ADAPT_PRIOR positions, decayed by ADAPT_DECAY per step
    ADAPT_EDGES = (0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.01)
    ADAPT_PRIOR, ADAPT_DECAY = 10.0, 0.99
    # every ADAPT_EXPLORE-th step verifies all rows (the positions a shorter step leaves out are observed too)
    ADAPT_EXPLORE = 0
    # DFlash 2: a mask row's largest softmax probability of the LM head
    ADAPT_P_DF = (0.0, 0.336, 0.464, 0.646, 0.752, 0.855, 0.929, 0.967, 0.985, 0.993, 1.0)
    ADAPT_A_DF = (0.0, 0.06, 0.34, 0.37, 0.56, 0.56, 0.63, 0.71, 0.76, 0.82, 0.983)
    # fast verification GPU ms by rows (M4 Pro 20-core): BF16, Q8_0 attention / head
    ADAPT_COST = {2: 35.1, 3: 40.4, 4: 44.5, 5: 49.0, 6: 53.5, 7: 58.0, 8: 61.7, 12: 79.3, 16: 92.3}
    ADAPT_COST_Q8 = {2: 29.4, 3: 34.7, 4: 39.5, 5: 45.4, 6: 48.6, 7: 53.6, 8: 58.0, 12: 78.5, 16: 92.1}
    # lookup drafts (one sequence, fast verification): after the drafter's 7, the 8 tokens that followed the latest
    # earlier occurrence of the last LOOKUP_N drafted tokens in the committed text (rows 12 or 16; an occurrence
    # closer than 8 tokens repeats with its period). Their agreement rate by the match length (how many tokens of the
    # committed text + drafts before them equal the text before the occurrence, bins from LOOKUP_EDGES) learned like
    # the drafter's, LOOKUP_A as the prior (measured on the corpus)
    mtp_lookup = True
    LOOKUP_N = 4
    LOOKUP_EDGES = (0, 8, 16, 32)
    LOOKUP_A = (0.6, 0.85, 0.93, 0.99)
    # lookup steps: the committed text's own continuation (after the latest earlier occurrence of its last LOOKUP_N
    # tokens) as all W - 1 drafts, the drafter skipped (only its context runs), when the step's expected gain beats
    # that of the drafter steps in the same bin of the committed match length (SOLO_EDGES). Priors: the agreement per
    # position SOLO_A (SOLO_PRIOR positions), the drafter steps' tokens SOLO_DRAFT (ADAPT_PRIOR steps of the 8-row
    # cost + 7 ms)
    SOLO_EDGES = (0, 16, 32, 48)
    SOLO_A = (0.86, 0.94, 0.95, 0.993)
    SOLO_DRAFT = (6.9, 7.4, 7.1, 7.1)
    SOLO_PRIOR = 60.0

    def mtp_width(self, C: int) -> int:
        for limit in sorted(self.mtp_widths):
            if C <= limit:
                return self.mtp_widths[limit]
        return self.mtp_widths[max(self.mtp_widths)]

    # ------------------------------------------------------------------ programs
    def _mtp_record(self) -> None:
        """Programs per batch size C in self.mtp_batches: verify (W x C rows, staged), commit, draft (context of
        the last verify's accepted rows, the block, the Markov walk); per prefill chunk: rows with features and the
        drafter context of those rows."""
        self.mtp_programs: Dict[tuple, int] = {}
        for C in self.mtp_batches:
            W = self.mtp_width(C)
            R = W * C
            RR = min(W, MTP_BLOCK) * C          # the row programs verify the drafter's block (no lookup drafts)
            self.mtp_programs[("verify", C)] = self._record_rows(RR, RR, stage=True, features=True)
            if C == 1 and R in VERIFY_ROWS:     # one sequence: the resident steps (exact / fast; Q8_0: fast only)
                for kind in (("fast",) if self.q8 else ("resident", "fast")):
                    # fast: every row count (adaptive rows, lookup drafts); exact: the drafter's block
                    for rows in (VERIFY_ROWS if kind == "fast" else (MTP_BLOCK,)):
                        self.mtp_programs[(f"verify_{kind}", rows)] = self._record_verify(rows, kind, reference=False)
                    if self.trace_rows:
                        self.mtp_programs[(f"verify_{kind}_reference", MTP_BLOCK)] = self._record_verify(
                            MTP_BLOCK, kind, reference=True)
            self.mtp_programs[("commit", C)] = self._mtp_record_commit(R)
            # the last step's commit and the next draft in one command buffer (independent: the draft reads the
            # drafter's own context, not the target caches)
            if self.dflash2:
                self.mtp_programs[("draft", C)] = self._df_record_draft(W)
                self.mtp_programs[("commit_draft", C)] = self._df_record_draft(W, commit=R)
            else:
                self.mtp_programs[("draft", C)] = self._mtp_record_draft(C, W)
                self.mtp_programs[("commit_draft", C)] = self._mtp_record_draft(C, W, commit=R)
            if C == 1 and W > MTP_BLOCK:        # lookup steps: the commit and the context, no draft
                self.mtp_programs[("commit_context", C)] = self._mtp_record_context(C, W, R)
        for P in self.mtp_prefill:     # target rows with features, then the drafter context of the same rows
            context = (lambda P=P: self._df_context(P, q8=False)) if self.dflash2 else (lambda P=P: self._mtp_context(P))
            self.mtp_programs[("prefill", P)] = self._record_rows(P, 1, features=True, then=context)

    def _record_verify(self, R: int, kind: str = "resident", reference: bool = False) -> int:
        b, plan = self.buf, self.plan
        program = plan.program()
        name = f"inkling2_verify_{kind}{'_q8' if self.q8 else ''}_r{R}_{'reference_' if reference else ''}bf16"
        extra = [(b["q8"], 0, READ), (b["q8table"], 0, READ)] if self.q8 else []
        plan.dispatch(name, [
            (b["weights"], 0, READ), (b["table"], 0, READ), (b["kv"], 0, READ_WRITE), (b["rings"], 0, READ),
            (b["v_params"], 0, READ), (b["r_tokens"], 0, READ), (b["m_stage"], 0, WRITE), (b["m_feat"], 0, WRITE),
            (b["r_logits"], 0, WRITE), (b["r_out"], 0, WRITE), (b["v_ws"], 0, READ_WRITE), (b["trace"], 0, WRITE),
            (b["control"], 0, READ_WRITE), (b["diagnostic"], 0, WRITE),
        ] + extra, (self._workers, 1, 1), self._threads, label="verify_resident")
        return program

    def verify_resident(self, tokens: Sequence[int], position: int, reference: bool = False,
                        kind: str = "resident") -> np.ndarray:
        """Resident verification step of sequence 0: rows ``tokens`` (2..8) at positions ``position ..``; caches
        and rings are read, not written (rows staged in m_stage, global K / V written by position); drafter
        features into m_feat, logits into r_logits. Returns the greedy target of every row."""
        R = len(tokens)
        key = (f"verify_{kind}_reference" if reference else f"verify_{kind}", R)
        if key not in getattr(self, "mtp_programs", {}):
            raise ValueError(f"no resident verification program for {R} rows")
        if position + R > self.max_seq:
            raise ValueError("verification rows exceed max_seq")
        if any(not 0 <= int(t) < self.cfg.unpadded_vocab_size for t in tokens):
            raise ValueError("token id outside the checkpoint vocabulary")
        self.plan.write(self.buf["r_tokens"], 0, np.asarray(tokens, np.uint32))
        self._verify_params[:] = [position, 0, 0, 0]
        self._control[:] = 0
        self._diagnostic[:] = 0
        self.last_verify_timing = self.plan.run(self.mtp_programs[key])
        if self._diagnostic.any():
            raise RuntimeError(f"resident verification barrier timed out (diagnostic {self._diagnostic.tolist()})")
        return self.plan.read(self.buf["r_out"], 0, np.uint32, R).copy()

    def _mtp_record_commit(self, R: int, program: int | None = None) -> int:
        b, plan = self.buf, self.plan
        program = plan.program() if program is None else program
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

    def _mtp_record_context(self, C: int, W: int, commit: int) -> int:
        """The last step's commit and the drafter context of its accepted rows, without a draft (a lookup step)."""
        program = self.plan.program()
        self._mtp_record_commit(commit, program)
        if self.dflash2:
            self._df_context(W, q8=True)
        else:
            self._mtp_context(W * C)
        return program

    def _mtp_record_draft(self, C: int, W: int, commit: int = 0) -> int:
        b, plan, w, cfg = self.buf, self.plan, self.mtp_w, self.cfg
        H, V = cfg.hidden_size, cfg.vocab_size
        R = MTP_BLOCK * C
        A = b["args"]
        program = plan.program()
        if commit:
            self._mtp_record_commit(commit, program)
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
        q8 = self.q8_drafter and R <= 8          # drafter layers from their Q8_0 copies (one sequence)
        for l in range(MTP_LAYERS):
            norm(w[f"{l}.in_norm"], "m_mlp" if l else None)
            if q8:
                self._q8_prep(b["m_x"], 0, H, R, H, "mtp_qkv")
                self._q8_rows8(f"mtp{l}.qkv", R, b["m_qkv"], 5120, 0, "mtp_qkv")
            else:
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
            if q8:
                self._q8_prep(b["m_attn"], 0, 3072, R, 3072, "mtp_o")
                self._q8_rows8(f"mtp{l}.o", R, b["m_o"], H, 0, "mtp_o")
            else:
                self._gemm("ink_gemm_bf16_rows", b["m_attn"], 0, 3072, w[f"{l}.o"], H, 3072, b["m_o"], 0, H, 0, R, True)
            norm(w[f"{l}.post_norm"], "m_o")
            if q8:
                self._q8_prep(b["m_x"], 0, H, R, H, "mtp_gate_up")
                self._q8_rows8(f"mtp{l}.gate_up", R, b["m_gu"], 2 * MTP_INTER, 0, "mtp_gate_up")
            else:
                self._gemm_multi("ink_gemm_bf16_multi", R, [(w[f"{l}.gate"], MTP_INTER, H, 0, H, 0, 0, 2 * MTP_INTER),
                                                            (w[f"{l}.up"], MTP_INTER, H, 0, H, 0, MTP_INTER, 2 * MTP_INTER)],
                                 [(b["m_x"], 0)], [(b["m_gu"], 0)], "mtp_gate_up")
            nr = self._param(np.array([R], np.uint32).tobytes())
            plan.dispatch("ink_mtp_silu_rows", [(b["m_gu"], 0, READ), (b["m_act"], 0, WRITE), (A, nr, READ)],
                          ((R * MTP_INTER + 255) // 256, 1, 1), 256, label="mtp_silu")
            if q8:
                self._q8_prep(b["m_act"], 0, MTP_INTER, R, MTP_INTER, "mtp_down")
                self._q8_rows8(f"mtp{l}.down", R, b["m_mlp"], H, 0, "mtp_down")
            else:
                self._gemm("ink_gemm_bf16_rows", b["m_act"], 0, MTP_INTER, w[f"{l}.down"], H, MTP_INTER, b["m_mlp"], 0,
                           H, 0, R, True)
        norm(w["norm"], "m_mlp", final=True)
        f = self.table[cfg.num_layers]
        nd = min(W, MTP_BLOCK) - 1                 # drafted positions of the block
        if "head" in self._q8 and nd * C <= 8:
            self._q8_rows("head", b["m_hn"], 0, H, nd * C, b["m_logits"], 0, V, "mtp_logits_q8")
        else:
            self._gemm("ink_gemm_f32_rows", b["m_hn"], 0, H, int(f[D_HEAD]), V, H, b["m_logits"], 0, V, 0, nd * C, True)
        cand = self.mtp_draft_candidates
        if cand:
            tk = self._param(np.array([w["fold_b"]], np.uint64).tobytes() +
                             np.array([nd * C, cfg.unpadded_vocab_size, V, 0], np.uint32).tobytes())
            plan.dispatch("ink_mtp_topk", [(b["weights"], 0, READ), (b["m_logits"], 0, READ), (b["m_cidx"], 0, WRITE),
                                           (b["m_cval"], 0, WRITE), (A, tk, READ)], (nd * C, 1, 1), 1024,
                          label="mtp_topk")
        # Markov walk
        for step in range(nd):
            rnn = self._param(np.array([w["m.w1"]], np.uint64).tobytes() + np.array([C, step], np.uint32).tobytes())
            plan.dispatch("ink_mtp_rnn_in", [(b["weights"], 0, READ), (b["m_hn"], 0, READ), (b["m_frnn"], 0, READ),
                                             (b["m_state"], 0, READ), (b["m_prev"], 0, READ), (b["m_rnnin"], 0, WRITE),
                                             (A, rnn, READ)], (C, 1, 1), 256, label="mtp_rnn_in")
            K = 2 * MTP_RANK + 2 * H
            if C == 1:      # one row: a SIMD group per output (the TensorOps tiles stream the matrix for 8 rows)
                self._gemv_rows(C, [(w["m.rnn"], 3 * MTP_RANK, K, 0, 0, 3 * MTP_RANK, 2)], [(b["m_rnnin"], 0)],
                                (b["m_proj"], 0), yf=(b["m_proj"], 0), label="mtp_rnn")
            else:
                self._gemm("ink_gemm_f32_rows", b["m_rnnin"], 0, K, w["m.rnn"], 3 * MTP_RANK, K, b["m_proj"], 0,
                           3 * MTP_RANK, 0, C, True)
            gt = self._param(np.array([w["m.rnn_b"]], np.uint64).tobytes() + np.array([C, 0], np.uint32).tobytes())
            plan.dispatch("ink_mtp_gates", [(b["weights"], 0, READ), (b["m_proj"], 0, READ), (b["m_state"], 0, READ_WRITE),
                                            (b["m_latent"], 0, WRITE), (A, gt, READ)], ((C * MTP_RANK + 255) // 256, 1, 1),
                          256, label="mtp_gates")
            if cand:     # the step's candidates only (the largest base logits + fold bias of the position)
                pc = self._param(np.array([w["fold_w"]], np.uint64).tobytes() + np.array([C, step, 0, 0], np.uint32).tobytes())
                plan.dispatch("ink_mtp_pick_cand", [(b["weights"], 0, READ), (b["m_cidx"], 0, READ), (b["m_cval"], 0, READ),
                                                    (b["m_latent"], 0, READ), (b["m_prev"], 0, READ_WRITE),
                                                    (b["m_drafts"], 0, WRITE), (A, pc, READ), (b["m_dtop"], 0, WRITE)],
                              (C, 1, 1), 256,
                              label="mtp_pick_cand")
                continue
            self._gemm("ink_gemm_f32_rows", b["m_latent"], 0, MTP_RANK, w["fold_w"], V, MTP_RANK, b["m_bias"], 0, V, 0,
                       C, True)
            pk = self._param(np.array([w["fold_b"]], np.uint64).tobytes() +
                             np.array([C, step, cfg.unpadded_vocab_size, V], np.uint32).tobytes())
            plan.dispatch("ink_mtp_pick", [(b["weights"], 0, READ), (b["m_logits"], 0, READ), (b["m_bias"], 0, READ),
                                           (b["m_prev"], 0, READ_WRITE), (b["m_drafts"], 0, WRITE), (A, pk, READ)],
                          (C, 1, 1), 1024, label="mtp_pick")
        return program

    # ------------------------------------------------------------------ generation
    def mtp_steps(self, prompts: Sequence[Sequence[int]], limit: int, stats: dict | None = None
                  ) -> Iterator[List[List[int]]]:
        """Greedy DSpark generation for C = len(prompts) sequences (sequence slots 0..C-1): yields per step the new
        tokens of every sequence (the first yield: each prompt's first token). A sequence with ``limit`` tokens
        is frozen (its rows still run, nothing is committed); the caller stops iterating when it is done."""
        C = len(prompts)
        if ("draft", C) not in getattr(self, "mtp_programs", {}):
            raise ValueError(f"no MTP programs for {C} sequences (mtp_batches)")
        if any(not p for p in prompts):
            raise ValueError("prompts must be non-empty")
        # a sequence can pass `limit` by W - 1 tokens and keeps verifying W rows after that
        if any(len(p) + limit + 2 * MTP_BLOCK > self.max_seq for p in prompts):
            raise ValueError("prompt + new tokens + 16 exceed max_seq")
        plan, b = self.plan, self.buf
        W = self.mtp_width(C)
        R = W * C
        RB = MTP_BLOCK * C
        anchor = np.zeros(C, np.uint32)
        pos = np.array([len(p) for p in prompts], np.int64)
        # prefill: target rows with features, then the drafter context of every prompt position
        for s, prompt in enumerate(prompts):
            anchor[s] = self.prefill(prompt, seq=s, features=True)
        count = [1] * C
        yield [[int(t)] for t in anchor]
        cinfo = np.zeros((R, 4), np.uint32)            # first draft step: no new context rows
        lookup = self.mtp_lookup and C == 1 and self.mtp_verify == "fast" and W > MTP_BLOCK
        N = self.LOOKUP_N
        hist = [int(t) for t in prompts[0]] + [int(anchor[0])]
        grams: Dict[tuple, int] = {}                   # n-gram -> index of the token after its latest occurrence
        for i in range(N, len(hist)):
            grams[tuple(hist[i - N:i])] = i
        reach_l = np.full(len(self.LOOKUP_A), self.ADAPT_PRIOR)
        agreed_l = reach_l * np.asarray(self.LOOKUP_A)
        lbin = 0

        def match(text: List[int], i: int) -> int:
            """How many tokens before index i equal the last tokens of text (at most 64)."""
            L, end = 0, len(text)
            while L < 64 and i - 1 - L >= 0 and text[i - 1 - L] == text[end - 1 - L]:
                L += 1
            return L
        steps = accepted = active = rows_total = lookup_steps = solo_steps = 0
        gpu = {"draft": 0.0, "verify": 0.0}
        adaptive = (self.mtp_adaptive and C == 1 and self.mtp_verify == "fast"
                    and all(("verify_fast", r) in self.mtp_programs for r in self.ADAPT_COST))
        rate, scale = 0.075, 1.0            # tokens per ms of a step; measured / relative verification time
        cost = self.ADAPT_COST_Q8 if self.q8 else self.ADAPT_COST
        cal_p, cal_a = (self.ADAPT_P_DF, self.ADAPT_A_DF) if self.dflash2 else (self.ADAPT_P, self.ADAPT_A)
        edges = np.asarray(self.ADAPT_EDGES)
        mids = np.minimum((edges[:-1] + edges[1:]) / 2, 1.0)
        reach = np.full(len(mids), self.ADAPT_PRIOR)
        agreed = reach * np.interp(mids, cal_p, cal_a)
        pending = False                     # a commit written, run with the next draft (or when the loop ends)
        solo_ok = lookup and adaptive and ("commit_context", C) in self.mtp_programs
        s_reach = np.full(len(self.SOLO_A), self.SOLO_PRIOR)
        s_agreed = s_reach * np.asarray(self.SOLO_A)
        d_n = np.full(len(self.SOLO_DRAFT), self.ADAPT_PRIOR)
        d_tok = d_n * np.asarray(self.SOLO_DRAFT)
        d_ms = d_n * (cost[MTP_BLOCK] + 7.0)
        solo_ms = 2.0                       # a lookup step's time beyond its verification (context, host)
        t_start = time.perf_counter()
        try:
            while min(count) < limit:
                t_step = time.perf_counter()
                plan.write(b["m_cinfo"], 0, cinfo)
                solo, sbin = False, -1
                if solo_ok and pending:     # a lookup step instead of a draft?
                    i = grams.get(tuple(hist[-N:]))
                    if i is not None:
                        sbin = int(np.searchsorted(self.SOLO_EDGES, match(hist, i), "right")) - 1
                        rho = s_agreed[sbin] / s_reach[sbin]
                        e = np.concatenate([[1.0], 1.0 + np.cumsum(rho ** np.arange(1, W))])
                        Vs = max(cost, key=lambda r: e[r - 1] - rate * (scale * cost[r] + solo_ms))
                        gain = e[Vs - 1] - rate * (scale * cost[Vs] + solo_ms)
                        solo = bool(gain > (d_tok[sbin] - rate * d_ms[sbin]) / d_n[sbin])
                if solo:
                    ext: List[int] = []
                    for j in range(W - 1):  # a near occurrence repeats with its period
                        ext.append(hist[i + j] if i + j < len(hist) else ext[i + j - len(hist)])
                    gpu["draft"] += plan.run(self.mtp_programs[("commit_context", C)])["gpu_us"]
                    pending = False
                    drafts = np.array([ext], np.uint32)
                    V = Vs
                else:
                    # draft
                    dinfo = np.zeros((RB, 4), np.uint32)
                    for s in range(C):
                        dinfo[s * 8:(s + 1) * 8] = [[s, pos[s] + j, pos[s], j] for j in range(8)]
                    plan.write(b["m_dinfo"], 0, dinfo)
                    plan.write(b["m_anchor"], 0, anchor)
                    plan.write(b["m_prev"], 0, anchor)
                    plan.view(b["m_state"], np.uint16, C * MTP_RANK)[:] = 0
                    if self.dflash2:            # the block: the anchor and 7 mask tokens
                        plan.write(b["df_tok"], 0, np.array([anchor[0]] + [DF_MASK] * 7, np.uint32))
                    gpu["draft"] += plan.run(self.mtp_programs[("commit_draft" if pending else "draft", C)])["gpu_us"]
                    pending = False
                    drafted = plan.read(b["m_drafts"], 0, np.uint32, C * 7).reshape(C, 7)
                    drafts = np.zeros((C, W - 1), np.uint32)
                    drafts[:, :min(7, W - 1)] = drafted[:, :min(7, W - 1)]
                    n_lk = 0
                    if lookup:                  # the tokens after the latest earlier occurrence of the last N drafts
                        i = grams.get(tuple(int(t) for t in drafted[0, 7 - N:7]))
                        if i is not None:
                            text = hist + [int(t) for t in drafted[0]]
                            lbin = int(np.searchsorted(self.LOOKUP_EDGES, match(text, i), "right")) - 1
                            n_lk = W - 8
                            for j in range(n_lk):
                                text.append(text[i + j])        # i + j < len(text): a near occurrence repeats
                            drafts[0, 7:] = text[-n_lk:]
                    if W - 1 > 7 + n_lk:        # rows without a draft: the last draft repeated
                        drafts[:, 7 + n_lk:] = drafts[:, 6 + n_lk:7 + n_lk]
                    V = min(W, MTP_BLOCK)
                    if adaptive:
                        top = plan.read(b["m_dtop"], 0, np.float32, 7 * 16).reshape(7, 16)
                        bins = np.clip(np.searchsorted(edges, np.exp(top[:W - 1, 4] - top[:W - 1, 8]), "right") - 1,
                                       0, len(mids) - 1)
                        agree = np.cumprod((agreed / reach)[bins])
                        expect = np.concatenate([[1.0], 1.0 + np.cumsum(agree)])     # expected tokens of V rows: [V - 1]
                        lk = agree[-1] * np.cumprod(np.full(W - 8, agreed_l[lbin] / reach_l[lbin]))  # after the 7
                        expect = np.concatenate([expect, expect[-1] + np.cumsum(lk)])
                        V = max((r for r in cost if r <= 8 + n_lk), key=lambda r: expect[r - 1] - rate * scale * cost[r])
                        if self.ADAPT_EXPLORE and steps % self.ADAPT_EXPLORE == self.ADAPT_EXPLORE - 1:
                            V = MTP_BLOCK + n_lk
                # verify
                toks = np.concatenate([anchor[:, None], drafts[:, :V - 1]], axis=1).reshape(-1)
                vinfo = np.zeros((R, 4), np.uint32)
                for s in range(C):
                    vinfo[s * W:s * W + V] = [[s, pos[s] + j, pos[s], pos[s] + V - 1] for j in range(V)]
                plan.write(b["r_tokens"], 0, toks.astype(np.uint32))
                plan.write(b["r_info"], 0, vinfo)
                resident = (f"verify_{self.mtp_verify}", V)
                if self.mtp_verify in ("resident", "fast") and C == 1 and resident in self.mtp_programs:
                    self._verify_params[:] = [pos[0], 0, 0, 0]
                    self._control[:] = 0
                    self._diagnostic[:] = 0
                    us = plan.run(self.mtp_programs[resident])["gpu_us"]
                    gpu["verify"] += us
                    if self._diagnostic.any():
                        raise RuntimeError(f"resident verification barrier timed out ({self._diagnostic.tolist()})")
                    if adaptive:
                        scale = 0.9 * scale + 0.1 * (us / 1000 / cost[V])
                else:
                    gpu["verify"] += plan.run(self.mtp_programs[("verify", C)])["gpu_us"]
                target = plan.read(b["r_out"], 0, np.uint32, C * V).reshape(C, V)
                commit = np.zeros((R, 2), np.uint32)
                cinfo = np.zeros((R, 4), np.uint32)
                new_tokens: List[List[int]] = [[] for _ in range(C)]
                for s in range(C):
                    if count[s] >= limit:
                        continue                    # finished: frozen (rows recomputed, nothing committed)
                    n = 0
                    while n < V - 1 and drafts[s, n] == target[s, n]:
                        n += 1
                    new_tokens[s] = [int(t) for t in drafts[s, :n]] + [int(target[s, n])]
                    count[s] += n + 1
                    commit[s * W:s * W + n + 1, 0] = 1
                    commit[s * W:(s + 1) * W, 1] = pos[s] + n
                    cinfo[s * W:s * W + n + 1] = [[s, pos[s] + j, 1, 1 if j == n else 0] for j in range(n + 1)]
                    anchor[s] = target[s, n]
                    pos[s] += n + 1
                    if C == 1:                  # the committed text and its n-grams
                        for t in new_tokens[s]:
                            if len(hist) >= N:
                                grams[tuple(hist[-N:])] = len(hist)
                            hist.append(t)
                    accepted += n + 1
                    active += 1
                    if adaptive and solo:   # lookup positions 0 .. min(n, V - 2) reached, n of them agreed
                        s_reach[sbin] = s_reach[sbin] * self.ADAPT_DECAY + min(n + 1, V - 1)
                        s_agreed[sbin] = s_agreed[sbin] * self.ADAPT_DECAY + n
                    elif adaptive:          # drafted positions 0 .. min(n, V - 2) were reached, n of them agreed
                        reach *= self.ADAPT_DECAY
                        agreed *= self.ADAPT_DECAY
                        k = np.arange(min(n, V - 2, 6) + 1)
                        np.add.at(reach, bins[k], 1.0)
                        np.add.at(agreed, bins[k], (k < n).astype(np.float64))
                        if V > 8 and n >= 7:            # lookup positions 7 .. min(n, V - 2) reached
                            reach_l[lbin] = reach_l[lbin] * self.ADAPT_DECAY + (min(n, V - 2) - 6)
                            agreed_l[lbin] = agreed_l[lbin] * self.ADAPT_DECAY + (n - 7)
                plan.write(b["m_commit"], 0, commit)
                pending = True
                steps += 1
                rows_total += V
                lookup_steps += V > MTP_BLOCK
                solo_steps += solo
                now = time.perf_counter()
                if adaptive:
                    ms, got = 1e3 * (now - t_step), sum(len(t) for t in new_tokens)
                    rate = 0.9 * rate + 0.1 * (got / max(1e-3, ms))
                    if solo:
                        solo_ms = 0.9 * solo_ms + 0.1 * max(0.0, ms - us / 1000)
                    elif sbin >= 0:         # a drafter step that had a lookup step to compare with
                        d_n[sbin] = d_n[sbin] * self.ADAPT_DECAY + 1.0
                        d_tok[sbin] = d_tok[sbin] * self.ADAPT_DECAY + got
                        d_ms[sbin] = d_ms[sbin] * self.ADAPT_DECAY + ms
                if stats is not None:
                    stats.update(steps=steps, tokens_per_step=accepted / max(1, active), rows_per_step=rows_total / steps,
                                 lookup_steps=lookup_steps, lookup_only_steps=solo_steps, decode_s=now - t_start,
                                 **{k + "_ms_per_step": v / 1000 / steps for k, v in gpu.items()})
                yield new_tokens
        finally:
            if pending:
                plan.run(self.mtp_programs[("commit", C)])

    def mtp_generate(self, prompts: Sequence[Sequence[int]], new_tokens: int, stats: dict | None = None
                     ) -> List[List[int]]:
        """Greedy generation of exactly ``new_tokens`` tokens per prompt with DSpark drafting (see mtp_steps)."""
        out: List[List[int]] = [[] for _ in prompts]
        for step in self.mtp_steps(prompts, new_tokens, stats):
            for o, new in zip(out, step):
                o.extend(new)
        return [o[:new_tokens] for o in out]


def _bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_round(a: np.ndarray) -> np.ndarray:
    return _bf16_to_f32(_f32_to_bf16(a))
