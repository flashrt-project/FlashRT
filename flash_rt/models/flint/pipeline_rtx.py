"""Flint SM120 compute plan over buffers owned by the torch frontend.

This module only composes the model-owned raw-pointer kernel bindings. Weight
IO, allocation, graph capture and request state live in the frontend runtime.
"""
from __future__ import annotations

import math
from typing import Optional


class FlintPipeline:
    """Deterministic launch order; the frontend supplies all buffers and handles."""

    def _gemv(self, m, x, out, epilogue, s):
        self.fl.gemv_w4a16(m.codes.data_ptr(), m.scales.data_ptr(), x.data_ptr(), out.data_ptr(),
                           m.rows, m.k, m.gs, epilogue, s)

    def launch_step(self, attn_splits: Optional[int] = None):
        """Enqueue one decode step for the token at device position ``pos``.

        Reads ``tokens[pos]``; writes the fp32 logits; the argmax kernel stores
        the greedy next token in ``tokens[pos + 1]`` once past the prompt
        (prompt tokens are never overwritten) and advances ``pos``.
        """
        fl, cfg, s = self.fl, self.cfg, self.stream.cuda_stream
        eps = cfg.rms_norm_eps
        splits = attn_splits or self.attn_splits
        scale = 1.0 / math.sqrt(cfg.head_dim)
        # token = tokens[pos]: embed reads through a pointer computed on device
        fl.embed_at_pos_bf16(self.tokens.data_ptr(), self.pos.data_ptr(), self.embed.data_ptr(),
                             self.x.data_ptr(), s)
        for il, L in enumerate(self.layers):
            fl.add_rmsnorm_bf16(self.x.data_ptr(), self.res.data_ptr(), L.attn_norm.data_ptr(), self.xn.data_ptr(),
                                1, eps, 1 if il == 0 else 0, s)
            self._gemv(L.qkv, self.xn, self.qkv_buf, 0, s)
            fl.qk_norm_rope_kv_bf16(self.qkv_buf.data_ptr(), L.q_norm.data_ptr(), L.k_norm.data_ptr(),
                                    self.rope_cos.data_ptr(), self.rope_sin.data_ptr(), self.pos.data_ptr(),
                                    self.q_buf.data_ptr(), L.cache.data_ptr(), self.ring if L.sliding else 0,
                                    1 if L.sliding else 0, eps, s)
            fl.decode_attention_bf16(self.q_buf.data_ptr(), L.cache.data_ptr(), self.pos.data_ptr(),
                                     self.ring if L.sliding else 0, cfg.sliding_window if L.sliding else 0,
                                     L.sinks.data_ptr(), scale, 4 if L.sliding else splits,
                                     self.o_part.data_ptr(), self.ml_part.data_ptr(), self.attn_out.data_ptr(), s)
            self._gemv(L.o, self.attn_out, self.x, 0, s)
            fl.add_rmsnorm_bf16(self.x.data_ptr(), self.res.data_ptr(), L.ffn_norm.data_ptr(), self.xn.data_ptr(),
                                1, eps, 0, s)
            if L.router is None:
                self._gemv(L.gate_up, self.xn, self.act, 2, s)
                self._gemv(L.down, self.act, self.x, 0, s)
            else:
                fl.router_topk_bf16(self.xn.data_ptr(), L.router.data_ptr(), self.router_logits.data_ptr(),
                                    self.router_counter.data_ptr(), self.ids.data_ptr(), self.tw.data_ptr(), s)
                fl.moe_gate_up_bf16(self.xn.data_ptr(), self.ids.data_ptr(), L.exp_gate_up.codes.data_ptr(),
                                    L.exp_gate_up.scales.data_ptr(), L.exp_gate_up.gs_dev.data_ptr(),
                                    L.sh_gate_up.codes.data_ptr(), L.sh_gate_up.scales.data_ptr(), L.sh_gate_up.gs,
                                    self.act.data_ptr(), s)
                fl.moe_down_combine_bf16(self.act.data_ptr(), self.ids.data_ptr(), self.tw.data_ptr(),
                                         L.exp_down.codes.data_ptr(), L.exp_down.scales.data_ptr(),
                                         L.exp_down.gs_dev.data_ptr(), L.sh_down.codes.data_ptr(),
                                         L.sh_down.scales.data_ptr(), L.sh_down.gs, self.x.data_ptr(), s)
        fl.add_rmsnorm_bf16(self.x.data_ptr(), self.res.data_ptr(), self.out_norm.data_ptr(), self.xn.data_ptr(),
                            1, eps, 0, s)
        self._gemv(self.lm_head, self.xn, self.logits, 1, s)
        fl.argmax_next_f32(self.logits.data_ptr(), cfg.vocab_size, self.cand_v.data_ptr(), self.cand_i.data_ptr(),
                           self.argmax_counter.data_ptr(), self.tokens.data_ptr(), self.pos.data_ptr(),
                           self.n_prompt_dev.data_ptr(), s)

    def _gemm(self, A, m, lda, mat, C, ldc, epi, s):
        self.fl.gemm_w4a16(A.data_ptr(), m, lda, mat.codes.data_ptr(), mat.scales.data_ptr(), mat.rows, mat.k,
                           mat.gs, C.data_ptr(), ldc, epi, s)

    def _pf_attn_splits(self, m: int, pos0: int, window: int) -> int:
        """KV splits per (16-token block, kv head) unit: enough CTAs to fill the GPU several times over."""
        units = (m + 15) // 16 * 4
        keys = min(pos0 + m, window + 15) if window else pos0 + m
        tiles = -(-keys // self.pf_attn_pkv)
        want = -(-8 * self.sm_count // units)
        cap = min(self.pf_split_max, self.pf_split_units // units)
        return max(1, min(want, cap, tiles // 32))   # >= ~32 tiles per split: merge traffic stays small

    def launch_prefill_chunk(self, m: int, pos0: int = 0):
        """Enqueue the layers for ``m`` prompt tokens starting at device position ``pos`` (== ``pos0`` on host).

        Leaves the final hidden state of the chunk in (pf_x + pf_res) and advances ``pos`` by m. Rows at
        positions >= ``n_prompt_dev`` (padding up to a graph shape) are not routed to any expert.
        """
        fl, cfg, s = self.fl, self.cfg, self.stream.cuda_stream
        eps, H = cfg.rms_norm_eps, cfg.hidden_size
        scale = 1.0 / math.sqrt(cfg.head_dim)
        T, I = cfg.num_experts_per_tok, cfg.expert_intermediate_size
        splits = (self._pf_attn_splits(m, pos0, 0), self._pf_attn_splits(m, pos0, cfg.sliding_window))
        # Rows per grouped-GEMM tile from the mean routed rows per expert (small chunks -> small tiles).
        mean_rows = -(-m * T // cfg.num_experts)
        tile_rows = 16 if mean_rows <= 12 else 32 if mean_rows <= 24 else 64
        max_tiles = (m * T) // tile_rows + cfg.num_experts
        fl.embed_rows_bf16(self.tokens.data_ptr(), self.pos.data_ptr(), m, self.embed.data_ptr(),
                           self.pf_x.data_ptr(), s)
        normed = False   # pf_xn / pf_res already hold this layer's normed input (fused into the MoE combine)
        for il, L in enumerate(self.layers):
            if not normed:
                fl.add_rmsnorm_bf16(self.pf_x.data_ptr(), self.pf_res.data_ptr(), L.attn_norm.data_ptr(),
                                    self.pf_xn.data_ptr(), m, eps, 1 if il == 0 else 0, s)
            normed = False
            ring = self.ring if L.sliding else 0
            if self.pf_fuse:   # Q/K norm + RoPE + KV write in the QKV GEMM epilogue
                q = L.qkv
                fl.gemm_w4a16_qkrope(self.pf_xn.data_ptr(), m, H, q.codes.data_ptr(), q.scales.data_ptr(), q.rows,
                                     q.k, q.gs, L.q_norm.data_ptr(), L.k_norm.data_ptr(), self.rope_cos.data_ptr(),
                                     self.rope_sin.data_ptr(), self.pos.data_ptr(), self.pf_q.data_ptr(),
                                     L.cache.data_ptr(), ring, 1 if L.sliding else 0, eps, s)
            else:
                self._gemm(self.pf_xn, m, H, L.qkv, self.pf_qkv, cfg.qkv_dim, fl.EPI_BF16, s)
                fl.qk_norm_rope_kv_rows_bf16(self.pf_qkv.data_ptr(), m, L.q_norm.data_ptr(), L.k_norm.data_ptr(),
                                             self.rope_cos.data_ptr(), self.rope_sin.data_ptr(), self.pos.data_ptr(),
                                             self.pf_q.data_ptr(), L.cache.data_ptr(), ring, 1 if L.sliding else 0,
                                             eps, s)
            window = cfg.sliding_window if L.sliding else 0
            fl.prefill_attention_bf16(self.pf_q.data_ptr(), m, L.cache.data_ptr(), self.pos.data_ptr(), ring, window,
                                      L.sinks.data_ptr(), scale, self.pf_att.data_ptr(), splits[1 if L.sliding else 0],
                                      self.pf_opart.data_ptr(), self.pf_ml.data_ptr(), self.pf_attn_ctr.data_ptr(),
                                      self.pf_attn_variant, s)
            self._gemm(self.pf_att, m, cfg.q_dim, L.o, self.pf_x, H, fl.EPI_BF16, s)
            fl.add_rmsnorm_bf16(self.pf_x.data_ptr(), self.pf_res.data_ptr(), L.ffn_norm.data_ptr(),
                                self.pf_xn.data_ptr(), m, eps, 0, s)
            if L.router is None:
                self._gemm(self.pf_xn, m, H, L.gate_up, self.pf_act, cfg.dense_intermediate_size,
                           fl.EPI_SWIGLU_DENSE, s)
                self._gemm(self.pf_act, m, cfg.dense_intermediate_size, L.down, self.pf_x, H, fl.EPI_BF16, s)
            else:
                self._gemm(self.pf_xn, m, H, L.sh_gate_up, self.pf_sh_act, cfg.shared_intermediate_size,
                           fl.EPI_SWIGLU_DENSE, s)
                self._gemm(self.pf_sh_act, m, cfg.shared_intermediate_size, L.sh_down, self.pf_sh, H,
                           fl.EPI_BF16, s)
                fl.router_topk_rows_bf16(self.pf_xn.data_ptr(), m, L.router.data_ptr(), self.pf_ids.data_ptr(),
                                         self.pf_tw.data_ptr(), self.pf_counts.data_ptr(), self.pos.data_ptr(),
                                         self.n_prompt_dev.data_ptr(), s)
                fl.moe_scan_place(self.pf_ids.data_ptr(), m, tile_rows, self.pf_counts.data_ptr(),
                                  self.pf_sorted.data_ptr(), self.pf_inv.data_ptr(), self.pf_tiles.data_ptr(),
                                  self.pf_ntiles.data_ptr(), self.pos.data_ptr(), self.n_prompt_dev.data_ptr(), s)
                g = L.exp_gate_up
                fl.gemm_w4a16_grouped(self.pf_xn.data_ptr(), self.pf_sorted.data_ptr(), H, self.pf_tiles.data_ptr(),
                                      self.pf_ntiles.data_ptr(), max_tiles, g.codes.data_ptr(),
                                      g.scales.data_ptr(), 2 * I, H, g.gs_dev.data_ptr(), self.pf_act_s.data_ptr(),
                                      I, fl.EPI_SWIGLU_ROUTED, tile_rows, mean_rows, s)
                d = L.exp_down
                fl.gemm_w4a16_grouped(self.pf_act_s.data_ptr(), 0, I, self.pf_tiles.data_ptr(),
                                      self.pf_ntiles.data_ptr(), max_tiles, d.codes.data_ptr(),
                                      d.scales.data_ptr(), H, I, d.gs_dev.data_ptr(), self.pf_y_s.data_ptr(), H,
                                      fl.EPI_BF16, tile_rows, mean_rows, s)
                if il + 1 < len(self.layers) and self.pf_fuse:
                    fl.moe_combine_norm_rows_bf16(self.pf_y_s.data_ptr(), self.pf_inv.data_ptr(),
                                                  self.pf_tw.data_ptr(), self.pf_sh.data_ptr(),
                                                  self.pf_res.data_ptr(), self.layers[il + 1].attn_norm.data_ptr(),
                                                  self.pf_xn.data_ptr(), m, eps, s)
                    normed = True
                else:
                    fl.moe_combine_rows_bf16(self.pf_y_s.data_ptr(), self.pf_inv.data_ptr(), self.pf_tw.data_ptr(),
                                             self.pf_sh.data_ptr(), self.pf_x.data_ptr(), m, s)
        fl.advance_pos(self.pos.data_ptr(), m, s)

    def launch_prefill_logits(self, m: int, logits_out=None):
        """Final norm + LM head for a finished chunk.

        logits_out is None: last row only into ``logits`` + greedy next token (tokens[pos], pos unchanged
        semantics: the argmax kernel writes tokens[pos_prev + 1] and sets pos accordingly).
        Otherwise: fp32 logits for all m rows into ``logits_out`` [m, vocab] (evaluation).
        """
        fl, cfg, s = self.fl, self.cfg, self.stream.cuda_stream
        H = cfg.hidden_size
        if logits_out is None:
            off = (m - 1) * H * 2  # bytes
            fl.add_rmsnorm_bf16(self.pf_x.data_ptr() + off, self.pf_res.data_ptr() + off, self.out_norm.data_ptr(),
                                self.xn.data_ptr(), 1, cfg.rms_norm_eps, 0, s)
            self._gemv(self.lm_head, self.xn, self.logits, 1, s)
            fl.advance_pos(self.pos.data_ptr(), -1, s)   # argmax writes tokens[pos + 1] and advances
            fl.argmax_next_f32(self.logits.data_ptr(), cfg.vocab_size, self.cand_v.data_ptr(),
                               self.cand_i.data_ptr(), self.argmax_counter.data_ptr(), self.tokens.data_ptr(),
                               self.pos.data_ptr(), self.n_prompt_dev.data_ptr(), s)
        else:
            fl.add_rmsnorm_bf16(self.pf_x.data_ptr(), self.pf_res.data_ptr(), self.out_norm.data_ptr(),
                                self.pf_xn.data_ptr(), m, cfg.rms_norm_eps, 0, s, 0)
            self._gemm(self.pf_xn, m, H, self.lm_head, logits_out, cfg.vocab_size, fl.EPI_F32, s)

    def launch_step_fused(self, attn_splits: Optional[int] = None, attn_warps: Optional[int] = None):
        """Fused decode step (6 kernels per MoE layer, PDL). Same contract as ``launch_step``."""
        fl, cfg, s = self.fl, self.cfg, self.stream.cuda_stream
        eps = cfg.rms_norm_eps
        splits = attn_splits or self.attn_splits
        full_warps = attn_warps or self.full_attn_warps
        scale = 1.0 / math.sqrt(cfg.head_dim)
        H = cfg.hidden_size
        res = [self.res2[0].data_ptr(), self.res2[1].data_ptr()]
        site = 0   # norm site k reads res[k & 1], block 0 writes res[(k + 1) & 1]
        fl.embed_pdl_bf16(self.tokens.data_ptr(), self.pos.data_ptr(), self.embed.data_ptr(), self.x.data_ptr(), s)
        for il, L in enumerate(self.layers):
            q = L.qkv
            fl.gemv_fused_w4a16(q.codes.data_ptr(), q.scales.data_ptr(), self.x.data_ptr(), res[site & 1],
                           res[(site + 1) & 1], L.attn_norm.data_ptr(), 1 if il == 0 else 0, eps,
                           self.qkv_buf.data_ptr(), q.rows, q.k, q.gs, 0, s)
            site += 1
            ring = self.ring if L.sliding else 0
            pf = (L.sliding or splits <= self.attn_pf_max_splits) and L.pf_n > 0
            fl.decode_attention_fused_bf16(self.qkv_buf.data_ptr(), L.q_norm.data_ptr(), L.k_norm.data_ptr(),
                                           self.rope_cos.data_ptr(), self.rope_sin.data_ptr(), self.pos.data_ptr(),
                                           L.cache.data_ptr(), ring, cfg.sliding_window if L.sliding else 0,
                                           1 if L.sliding else 0, eps, L.sinks.data_ptr(), scale,
                                           (self.swa_attn[0] if L.sliding else splits), self.o_part.data_ptr(),
                                           self.ml_part.data_ptr(), self.attn_counters.data_ptr(),
                                           self.attn_out.data_ptr(), s,
                                           self.swa_attn[1] if L.sliding else full_warps,
                                           L.pf_ranges.data_ptr() if pf else 0, L.pf_n if pf else 0)
            o = L.o
            fl.gemv_fused_w4a16(o.codes.data_ptr(), o.scales.data_ptr(), self.attn_out.data_ptr(), 0, 0, 0, 0, eps,
                           self.x_attn.data_ptr(), o.rows, o.k, o.gs, 0, s)
            if L.router is None:
                g = L.gate_up
                fl.gemv_fused_w4a16(g.codes.data_ptr(), g.scales.data_ptr(), self.x_attn.data_ptr(), res[site & 1],
                               res[(site + 1) & 1], L.ffn_norm.data_ptr(), 0, eps, self.act.data_ptr(), g.rows, g.k,
                               g.gs, 2, s)
                d = L.down
                fl.gemv_fused_w4a16(d.codes.data_ptr(), d.scales.data_ptr(), self.act.data_ptr(), 0, 0, 0, 0, eps,
                               self.x.data_ptr(), d.rows, d.k, d.gs, 0, s)
            else:
                fl.router_norm_bf16(self.x_attn.data_ptr(), res[site & 1], L.ffn_norm.data_ptr(), eps,
                                    res[(site + 1) & 1], self.xn.data_ptr(), L.router.data_ptr(),
                                    self.router_logits.data_ptr(), self.router_counter.data_ptr(),
                                    self.ids.data_ptr(), self.tw.data_ptr(), s)
                fl.moe_gate_up_pdl_bf16(self.xn.data_ptr(), self.ids.data_ptr(), L.exp_gate_up.codes.data_ptr(),
                                     L.exp_gate_up.scales.data_ptr(), L.exp_gate_up.gs_dev.data_ptr(),
                                     L.sh_gate_up.codes.data_ptr(), L.sh_gate_up.scales.data_ptr(), L.sh_gate_up.gs,
                                     self.act.data_ptr(), s)
                fl.moe_down_combine_pdl_bf16(self.act.data_ptr(), self.ids.data_ptr(), self.tw.data_ptr(),
                                          L.exp_down.codes.data_ptr(), L.exp_down.scales.data_ptr(),
                                          L.exp_down.gs_dev.data_ptr(), L.sh_down.codes.data_ptr(),
                                          L.sh_down.scales.data_ptr(), L.sh_down.gs, self.x.data_ptr(), s)
            site += 1
        lm = self.lm_head
        if self.lm_greedy:
            fl.lm_head_greedy_w4a16(lm.codes.data_ptr(), lm.scales.data_ptr(), self.x.data_ptr(), res[site & 1],
                                    res[(site + 1) & 1], self.out_norm.data_ptr(), eps, self.logits.data_ptr(),
                                    lm.rows, lm.gs, self.lm_cand_v.data_ptr(), self.lm_cand_i.data_ptr(),
                                    self.lm_cand_v.numel(), self.argmax_counter.data_ptr(), self.tokens.data_ptr(),
                                    self.pos.data_ptr(), self.n_prompt_dev.data_ptr(), s)
            return
        fl.add_rmsnorm_pdl_bf16(self.x.data_ptr(), res[site & 1], self.out_norm.data_ptr(), eps, res[(site + 1) & 1],
                             self.xn.data_ptr(), s)
        fl.gemv_fused_w4a16(lm.codes.data_ptr(), lm.scales.data_ptr(), self.xn.data_ptr(), 0, 0, 0, 0, eps,
                       self.logits.data_ptr(), lm.rows, lm.k, lm.gs, 1, s)
        fl.argmax_next_f32(self.logits.data_ptr(), cfg.vocab_size, self.cand_v.data_ptr(), self.cand_i.data_ptr(),
                           self.argmax_counter.data_ptr(), self.tokens.data_ptr(), self.pos.data_ptr(),
                           self.n_prompt_dev.data_ptr(), s)



def __getattr__(name: str):
    # Preserve the model-local runtime import used by existing scripts.
    if name == "FlintRuntime":
        from flash_rt.frontends.torch._flint_rtx_runtime import FlintRuntime
        return FlintRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
