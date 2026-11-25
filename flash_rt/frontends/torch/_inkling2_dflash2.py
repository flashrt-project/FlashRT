"""DFlash 2 drafter (incoai/Inkling-2-Mini-DFlash2) for Inkling 2 Mini speculative decoding on Apple Silicon.

Follows SGLang's ``DFlash2DraftModel`` and its greedy draft step with the Inkling integration: the context of a
committed position is fc(the target's 4 features) -> hidden_norm -> per layer K / V (k_norm, RoPE theta 1e4) in the
drafter's own cache; a block of 8 rows (the anchor and 7 mask tokens through the target's embedding and
embed_norm) runs the 5 layers (residual RMSNorm, two-tap dynamic grouped convolutions around attention and MLP,
32 query / 8 KV heads without causal mask over a 2047-position window and the block, SwiGLU), the final norm, and
the LM head's 16 largest logits per mask row over the real vocabulary, and the candidate selector's greedy path
(per position the predecessor's codebook row times the projected hidden state against each candidate's successor
codebook row, added to its logit, softcapped after the correction). ``dflash2_selector = False``: each row's largest
logit instead. Drafts only decide how many tokens a step accepts. One sequence; the drafter's products run from
Q8_0 copies (contexts of prefill chunks from the BF16 weights on TensorOps tiles).

Mixed into ``Inkling2Runtime`` with ``dflash2=<draft directory>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from flash_rt.hardware.apple_native.native import READ, READ_WRITE, WRITE
from flash_rt.models.inkling2.checkpoint import SafetensorsFile

DF_LAYERS = 5
DF_Q, DF_KV, DF_I = 4096, 1024, 6912
DF_CONV = 2 * 2 * 144                  # deltas of a sublayer: [in | out][tap][group of 16 channels]
DF_MASK = 200056
DF_RANK, DF_TOPK = 256, 16


class _DFlash2Mixin:
    dflash2_selector = True
    # ------------------------------------------------------------------ weights
    def _df_upload(self, layout) -> None:
        H = self.cfg.hidden_size
        d = Path(self.dflash2)
        cfg = json.loads((d / "config.json").read_text())
        dc = cfg.get("dflash_config", {})
        expect = dict(hidden_size=H, num_hidden_layers=DF_LAYERS, num_attention_heads=32, num_key_value_heads=8,
                      head_dim=128, intermediate_size=DF_I)
        for k, v in expect.items():
            if cfg.get(k) != v:
                raise ValueError(f"DFlash 2 config: {k} = {cfg.get(k)}, expected {v}")
        dexp = dict(block_size=8, mask_token_id=DF_MASK, conv_kernel_size=2, conv_group_size=16,
                    target_layer_ids=[5, 23, 41, 48])
        for k, v in dexp.items():
            if dc.get(k) != v:
                raise ValueError(f"DFlash 2 dflash_config: {k} = {dc.get(k)}, expected {v}")
        st = SafetensorsFile(d / "model.safetensors")
        w: Dict[str, int] = {}

        def bf16(name, shape):
            a = st.tensor(name)
            if st.dtype(name) != "BF16" or tuple(a.shape) != tuple(shape):
                raise ValueError(f"DFlash 2: {name} is {st.dtype(name)} {a.shape}, expected BF16 {shape}")
            return layout.add(a)

        for l in range(DF_LAYERS):
            p = f"layers.{l}."
            w[f"{l}.in_norm"] = bf16(p + "input_layernorm.weight", (H,))
            w[f"{l}.post_norm"] = bf16(p + "post_attention_layernorm.weight", (H,))
            w[f"{l}.q"] = bf16(p + "self_attn.q_proj.weight", (DF_Q, H))
            w[f"{l}.k"] = bf16(p + "self_attn.k_proj.weight", (DF_KV, H))
            w[f"{l}.v"] = bf16(p + "self_attn.v_proj.weight", (DF_KV, H))
            w[f"{l}.o"] = bf16(p + "self_attn.o_proj.weight", (H, DF_Q))
            w[f"{l}.qn"] = bf16(p + "self_attn.q_norm.weight", (128,))
            w[f"{l}.kn"] = bf16(p + "self_attn.k_norm.weight", (128,))
            w[f"{l}.gate"] = bf16(p + "mlp.gate_proj.weight", (DF_I, H))
            w[f"{l}.up"] = bf16(p + "mlp.up_proj.weight", (DF_I, H))
            w[f"{l}.down"] = bf16(p + "mlp.down_proj.weight", (H, DF_I))
            w[f"{l}.ac_base"] = bf16(p + "attention_conv.base_kernel", (2, 2, H))
            w[f"{l}.ac"] = bf16(p + "attention_conv.kernel_projection.weight", (DF_CONV, H))
            w[f"{l}.mc_base"] = bf16(p + "mlp_conv.base_kernel", (2, 2, H))
            w[f"{l}.mc"] = bf16(p + "mlp_conv.kernel_projection.weight", (DF_CONV, H))
        w["norm"] = bf16("norm.weight", (H,))
        w["fc"] = bf16("fc.weight", (H, 4 * H))
        w["hidden_norm"] = bf16("hidden_norm.weight", (H,))
        V = self.cfg.vocab_size
        w["sel"] = bf16("candidate_selector.hidden_projection.weight", (DF_RANK, H))
        w["pred_cb"] = bf16("candidate_selector.predecessor_codebook", (V, DF_RANK))
        w["succ_cb"] = bf16("candidate_selector.successor_codebook", (V, DF_RANK))
        self.df_softcap = float(dc.get("final_logit_softcapping", 30.0))
        if dc.get("selector_rank", DF_RANK) != DF_RANK or dc.get("selector_top_k", DF_TOPK) != DF_TOPK:
            raise ValueError("DFlash 2 selector: rank 256 and top-k 16 expected")
        self._df_file = st               # closed once the layout has been copied
        self.df_w = w

    def _df_q8_matrices(self):
        """Drafter matrices with a Q8_0 copy (q | k | v and gate | up stacked)."""
        H, w = self.cfg.hidden_size, self.df_w
        out = []
        for l in range(DF_LAYERS):
            out.append((f"df{l}.qkv", [(int(w[f"{l}.q"]), DF_Q), (int(w[f"{l}.k"]), DF_KV), (int(w[f"{l}.v"]), DF_KV)],
                        DF_Q + 2 * DF_KV, H))
            out.append((f"df{l}.o", int(w[f"{l}.o"]), H, DF_Q))
            out.append((f"df{l}.gate_up", [(int(w[f"{l}.gate"]), DF_I), (int(w[f"{l}.up"]), DF_I)], 2 * DF_I, H))
            out.append((f"df{l}.down", int(w[f"{l}.down"]), H, DF_I))
            out.append((f"df{l}.ac", int(w[f"{l}.ac"]), DF_CONV, H))
            out.append((f"df{l}.mc", int(w[f"{l}.mc"]), DF_CONV, H))
        out.append(("df.fc", int(w["fc"]), H, 4 * H))
        out.append(("df.sel", int(w["sel"]), DF_RANK, H))
        return out

    # ------------------------------------------------------------------ buffers
    def _df_allocate(self) -> None:
        H = self.cfg.hidden_size
        RC = max(self.row_capacity, 8)
        self.df_cap = self.max_seq + 8
        self.df_layer_elems = 2 * 8 * self.df_cap * 128
        bf, f32 = 2, 4
        sizes = {
            "df_kv": DF_LAYERS * self.df_layer_elems * bf, "df_tok": 8 * 4,
            "df_x": 8 * H * bf, "df_res": 8 * H * f32, "df_n": 8 * H * bf, "df_c": 8 * DF_CONV * bf,
            "df_u": 8 * H * bf, "df_u2": 8 * H * bf, "df_qkv": 8 * (DF_Q + 2 * DF_KV) * bf, "df_q": 8 * DF_Q * bf,
            "df_att": 8 * DF_Q * bf, "df_o": 8 * H * bf, "df_gu": 8 * 2 * DF_I * bf, "df_act": 8 * DF_I * bf,
            "df_mlp": 8 * H * bf, "df_hl": 8 * H * bf, "df_hf": 8 * H * bf,
            "df_cx": RC * 4 * H * bf, "df_cn": RC * H * bf, "df_ckv": RC * DF_LAYERS * 2 * DF_KV * bf,
            "df_cand": 8 * DF_TOPK * 4, "df_cval": 8 * DF_TOPK * f32, "df_stat": 8 * 2 * f32, "df_hp": 8 * DF_RANK * bf,
        }
        for name, nbytes in sizes.items():
            self.buf[name] = self.plan.buffer(nbytes)

    # ------------------------------------------------------------------ programs
    def _df_norm(self, x, x_stride, rows, w, out, out_stride, label):
        prm = self._param(np.array([w], np.uint64).tobytes() + np.array([rows, x_stride, out_stride, 0], np.uint32).tobytes())
        self.plan.dispatch("df_norm_rows", [(self.buf["weights"], 0, READ), (x, 0, READ), (out, 0, WRITE),
                                            (self.buf["args"], prm, READ)], (rows, 1, 1), 256, label=label)

    def _df_context(self, R: int, q8: bool) -> None:
        """Context K / V of the rows flagged in m_cinfo (features in m_feat, R rows) into the drafter cache."""
        b, plan, w = self.buf, self.plan, self.df_w
        H = self.cfg.hidden_size
        A = b["args"]
        nr = self._param(np.array([R, 0, 0, 0], np.uint32).tobytes())
        plan.dispatch("df_ctx_gather", [(b["m_feat"], 0, READ), (b["df_cx"], 0, WRITE), (A, nr, READ)],
                      ((R * 4 * H + 255) // 256, 1, 1), 256, label="df_ctx_gather")
        if q8:
            self._q8_prep(b["df_cx"], 0, 4 * H, R, 4 * H, "df_fc")
            self._q8_rows8("df.fc", R, b["df_cn"], H, 0, "df_fc")
        else:
            self._gemm("ink_gemm_bf16_rows", b["df_cx"], 0, 4 * H, w["fc"], H, 4 * H, b["df_cn"], 0, H, 0, R, True)
        self._df_norm(b["df_cn"], H, R, w["hidden_norm"], b["df_cn"], H, "df_hidden_norm")
        kvs = DF_LAYERS * 2 * DF_KV
        if q8:
            self._q8_prep(b["df_cn"], 0, H, R, H, "df_ctx_kv")
            for l in range(DF_LAYERS):      # the k | v rows of the stacked q | k | v copy
                self._q8_rows8(f"df{l}.qkv", R, b["df_ckv"], kvs, l * 2 * DF_KV, "df_ctx_kv", row0=DF_Q, nrows=2 * DF_KV)
        else:
            mats = []
            for l in range(DF_LAYERS):
                mats.append((w[f"{l}.k"], DF_KV, H, 0, H, 0, l * 2 * DF_KV, kvs))
                mats.append((w[f"{l}.v"], DF_KV, H, 0, H, 0, l * 2 * DF_KV + DF_KV, kvs))
            for m0 in range(0, len(mats), 8):
                self._gemm_multi("ink_gemm_bf16_multi", R, mats[m0:m0 + 8], [(b["df_cn"], 0)], [(b["df_ckv"], 0)],
                                 "df_ctx_kv")
        ctx = self._param(np.array([w[f"{l}.kn"] for l in range(DF_LAYERS)] + [self.df_layer_elems], np.uint64).tobytes() +
                          np.array([R, self.df_cap, 0, 0], np.uint32).tobytes())
        plan.dispatch("df_ctx_write", [(b["weights"], 0, READ), (b["df_ckv"], 0, READ), (b["df_kv"], 0, READ_WRITE),
                                       (b["m_cinfo"], 0, READ), (A, ctx, READ)], (R, DF_LAYERS, 1), 256,
                      label="df_ctx_write")

    def _df_record_draft(self, W: int, commit: int = 0) -> int:
        """Draft program of one sequence: [the last step's commit], the drafter context of its accepted rows, the
        block of 8 through the 5 layers, the LM head of the 7 mask rows and their largest logits."""
        b, plan, w, cfg = self.buf, self.plan, self.df_w, self.cfg
        H, V = cfg.hidden_size, cfg.vocab_size
        A = b["args"]
        R = 8
        program = plan.program()
        if commit:
            self._mtp_record_commit(commit, program)
        self._df_context(W, q8=True)
        rv0 = self._visit_param(0, R, 0, [], 0)
        plan.dispatch("ink_embed_rows", [(b["weights"], 0, READ), (b["table"], 0, READ), (b["df_tok"], 0, READ),
                                         (b["df_x"], 0, WRITE), (A, rv0, READ)], (R, 1, 1), 256, label="df_embed")

        def add_norm(add, weight, flags, out, label):
            prm = self._param(np.array([weight], np.uint64).tobytes() + np.array([R, flags, 0, 0], np.uint32).tobytes())
            plan.dispatch("df_add_norm", [(b["weights"], 0, READ), (b["df_res"], 0, READ_WRITE), (b[add], 0, READ),
                                          (b[out], 0, WRITE), (A, prm, READ)], (R, 1, 1), 256, label=label)

        def conv(x, base, io, out, label):
            prm = self._param(np.array([base], np.uint64).tobytes() + np.array([R, io, 0, 0], np.uint32).tobytes())
            plan.dispatch("df_conv", [(b["weights"], 0, READ), (b[x], 0, READ), (b["df_c"], 0, READ), (b[out], 0, WRITE),
                                      (A, prm, READ)], ((R * H + 255) // 256, 1, 1), 256, label=label)

        for l in range(DF_LAYERS):
            add_norm("df_x" if l == 0 else "df_hl", w[f"{l}.in_norm"], 2 if l == 0 else 1, "df_n", "df_in_norm")
            self._q8_prep(b["df_n"], 0, H, R, H, "df_ac")
            self._q8_rows8(f"df{l}.ac", R, b["df_c"], DF_CONV, 0, "df_ac")
            conv("df_n", w[f"{l}.ac_base"], 0, "df_u", "df_conv")
            self._q8_prep(b["df_u"], 0, H, R, H, "df_qkv")
            self._q8_rows8(f"df{l}.qkv", R, b["df_qkv"], DF_Q + 2 * DF_KV, 0, "df_qkv")
            qk = self._param(np.array([w[f"{l}.qn"], w[f"{l}.kn"], self.df_layer_elems], np.uint64).tobytes() +
                             np.array([R, self.df_cap, l, 0], np.uint32).tobytes())
            plan.dispatch("df_qk_rows", [(b["weights"], 0, READ), (b["df_qkv"], 0, READ), (b["df_q"], 0, WRITE),
                                         (b["df_kv"], 0, READ_WRITE), (b["m_dinfo"], 0, READ), (A, qk, READ)],
                          (R, 1, 1), 256, label="df_qk")
            at = self._param(np.array([self.df_layer_elems], np.uint64).tobytes() +
                             np.array([R, self.df_cap, l, 0], np.uint32).tobytes())
            plan.dispatch("df_attn_rows", [(b["df_q"], 0, READ), (b["df_kv"], 0, READ), (b["df_att"], 0, WRITE),
                                           (b["m_dinfo"], 0, READ), (A, at, READ)], (8, R, 1), 256, label="df_attn")
            self._q8_prep(b["df_att"], 0, DF_Q, R, DF_Q, "df_o")
            self._q8_rows8(f"df{l}.o", R, b["df_o"], H, 0, "df_o")
            conv("df_o", w[f"{l}.ac_base"], 1, "df_u2", "df_conv")
            add_norm("df_u2", w[f"{l}.post_norm"], 1, "df_n", "df_post_norm")
            self._q8_prep(b["df_n"], 0, H, R, H, "df_mc")
            self._q8_rows8(f"df{l}.mc", R, b["df_c"], DF_CONV, 0, "df_mc")
            conv("df_n", w[f"{l}.mc_base"], 0, "df_u", "df_conv")
            self._q8_prep(b["df_u"], 0, H, R, H, "df_gate_up")
            self._q8_rows8(f"df{l}.gate_up", R, b["df_gu"], 2 * DF_I, 0, "df_gate_up")
            nr = self._param(np.array([R, 0, 0, 0], np.uint32).tobytes())
            plan.dispatch("df_silu_rows", [(b["df_gu"], 0, READ), (b["df_act"], 0, WRITE), (A, nr, READ)],
                          ((R * DF_I + 255) // 256, 1, 1), 256, label="df_silu")
            self._q8_prep(b["df_act"], 0, DF_I, R, DF_I, "df_down")
            self._q8_rows8(f"df{l}.down", R, b["df_mlp"], H, 0, "df_down")
            conv("df_mlp", w[f"{l}.mc_base"], 1, "df_hl", "df_conv")
        add_norm("df_hl", w["norm"], 1, "df_hf", "df_final_norm")
        # the mask rows' LM head (the drafter's Q8_0 copy of the target head)
        E = W - 1
        self._q8_rows("head", b["df_hf"], H * 2, H, E, b["m_logits"], 0, V, "df_logits")
        pk = self._param(np.array([E, cfg.unpadded_vocab_size, V, 0], np.uint32).tobytes())
        if not self.dflash2_selector:      # each row's largest logit
            plan.dispatch("df_pick", [(b["m_logits"], 0, READ), (b["m_drafts"], 0, WRITE), (b["m_dtop"], 0, WRITE),
                                      (A, pk, READ)], (E, 1, 1), 1024, label="df_pick")
            return program
        plan.dispatch("df_top16", [(b["m_logits"], 0, READ), (b["df_cand"], 0, WRITE), (b["df_cval"], 0, WRITE),
                                   (b["df_stat"], 0, WRITE), (A, pk, READ)], (E, 1, 1), 1024, label="df_top16")
        self._q8_prep(b["df_hf"], H * 2, H, E, H, "df_sel")
        self._q8_rows8("df.sel", E, b["df_hp"], DF_RANK, 0, "df_sel")
        sp = self._param(np.array([w["pred_cb"], w["succ_cb"]], np.uint64).tobytes() + np.array([E, 0], np.uint32).tobytes() +
                         np.array([self.df_softcap, 0.0], np.float32).tobytes())
        plan.dispatch("df_select", [(b["weights"], 0, READ), (b["df_cand"], 0, READ), (b["df_cval"], 0, READ),
                                    (b["df_stat"], 0, READ), (b["df_hp"], 0, READ), (b["df_tok"], 0, READ),
                                    (b["m_drafts"], 0, WRITE), (b["m_dtop"], 0, WRITE), (A, sp, READ)], (1, 1, 1), 256,
                      label="df_select")
        return program
