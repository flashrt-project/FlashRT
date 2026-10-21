"""Flint runtime on RTX SM120: buffers, weight upload, and the native kernel sequence.

Python here only prepares buffers and launches kernels that take raw device
pointers; the steady-state decode step is a fixed kernel sequence that reads
the position and the token from device memory, so it is captured once into a
CUDA graph and replayed for every step (prompt tokens and generated tokens
alike). PyTorch is used as the allocator and for graph capture only.

Numerics follow the official W4A16 contract: exact NVFP4 dequantization, BF16
activations and residual stream, fp32 accumulation, BF16 KV cache, fp32 logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from flash_rt.models.flint.config import FlintConfig, load_config
from flash_rt.models.flint.gguf import GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_NVFP4, GGUFFile
from flash_rt.models.flint.weights import check_scale_bytes

try:  # the extension is optional at import time (see _require_kernels)
    from flash_rt import flash_rt_flint as _fl
except ImportError:  # pragma: no cover - exercised on machines without the build
    _fl = None


def _require_kernels():
    if _fl is None:
        raise RuntimeError(
            "flash_rt_flint is not built. Configure with -DGPU_ARCH=120 "
            "-DFLASHRT_ENABLE_FLINT=ON and rebuild (see docs/flint_usage.md).")
    return _fl


@dataclass
class _Mat:
    """One NVFP4 matrix in runtime layout."""
    codes: object
    scales: object
    gs: float = 1.0          # per-tensor global scale (host value, baked into launches)
    gs_dev: object = None    # per-expert global scales (device f32)
    rows: int = 0
    k: int = 0


@dataclass
class _Layer:
    attn_norm: object
    ffn_norm: object
    q_norm: object
    k_norm: object
    sinks: object
    qkv: _Mat
    o: _Mat
    sliding: bool
    gate_up: Optional[_Mat] = None       # dense layers
    down: Optional[_Mat] = None
    router: object = None                # MoE layers (bf16 [256, 2048])
    router_t: object = None              # same, K-major [2048, 256] (prefill)
    exp_gate_up: Optional[_Mat] = None
    exp_down: Optional[_Mat] = None
    sh_gate_up: Optional[_Mat] = None
    sh_down: Optional[_Mat] = None
    cache: object = None


class FlintRuntime:
    """Owns every device buffer and launches the Flint kernel sequence."""

    def __init__(self, gguf_path: str, max_seq: int = 32768, sliding_ring: int = 4096,
                 attn_splits: int = 16, prefill_chunk: int = 2048, device: int = 0):
        import torch
        self.torch = torch
        self.fl = _require_kernels()
        if sliding_ring & (sliding_ring - 1):
            raise ValueError("sliding_ring must be a power of two")
        self.gguf = GGUFFile(gguf_path)
        self.cfg: FlintConfig = load_config(self.gguf)
        if sliding_ring < self.cfg.sliding_window:
            raise ValueError(f"sliding_ring {sliding_ring} < window {self.cfg.sliding_window}")
        self.max_seq = int(max_seq)
        self.ring = int(sliding_ring)
        self.attn_splits = int(attn_splits)
        self.prefill_chunk = int(prefill_chunk)
        if self.prefill_chunk + self.cfg.sliding_window > sliding_ring:
            raise ValueError("prefill_chunk + sliding_window must fit in sliding_ring")
        self.dev = torch.device("cuda", device)
        torch.cuda.set_device(self.dev)
        self.stream = torch.cuda.Stream(self.dev)
        self._anchors: List[object] = []
        self._load_weights()
        self._alloc_buffers()
        self.graph = None

    # ------------------------------------------------------------------ loading
    def _empty(self, shape, dtype):
        t = self.torch.empty(shape, dtype=dtype, device=self.dev)
        return t

    def _f32(self, name):
        a = self.gguf.array(name)
        if a.dtype != np.float32:
            raise ValueError(f"{name}: expected F32")
        return self.torch.from_numpy(np.array(a, copy=True)).to(self.dev)

    def _raw_to_dev(self, name):
        raw = self.gguf.raw_bytes(name)
        n = raw.size
        if self._staging is None or self._staging.numel() < n:
            self._staging = self._empty((n,), self.torch.uint8)
        host = self.torch.from_numpy(np.array(raw, copy=False).view(np.uint8))
        self._staging[:n].copy_(host, non_blocking=False)
        return self._staging

    def _scale_values(self, name):
        sname = name[: -len(".weight")] + ".scale"
        return np.array(self.gguf.array(sname), dtype=np.float32).reshape(-1)

    def _nvfp4_into(self, name, codes, scales, group_rows=None, group_stride=None, row_offset=0):
        t = self.gguf.tensor(name)
        if t.ggml_type != GGML_TYPE_NVFP4:
            raise ValueError(f"{name}: expected NVFP4, got {t.type_name}")
        k = t.shape[0]
        rows = t.num_values // k
        check_scale_bytes(self.gguf.nvfp4_blocks(name)[:, :, :4], name)
        src = self._raw_to_dev(name)
        gr = group_rows or rows
        self.fl.repack_nvfp4(src.data_ptr(), rows, k, codes.data_ptr(), scales.data_ptr(), gr,
                             group_stride or gr, row_offset, self.stream.cuda_stream)
        self.stream.synchronize()
        return rows, k

    def _gate_up(self, gate: str, up: str, rows: int, k: int) -> _Mat:
        """Gate and up matrices interleaved by 8 rows (gate a -> 16*(a//8) + a%8, up a -> +8)."""
        codes = self._empty((2 * rows, k // 2), self.torch.uint8)
        scales = self._empty((2 * rows, k // 16), self.torch.uint8)
        sg, su = self._scale_values(gate), self._scale_values(up)
        if sg.size != 1 or float(sg[0]) != float(su[0]):
            raise ValueError(f"{gate}/{up}: fused gate/up need identical global scales (lossless fusion)")
        self._nvfp4_into(gate, codes, scales, 8, 16, 0)
        self._nvfp4_into(up, codes, scales, 8, 16, 8)
        return _Mat(codes, scales, gs=float(sg[0]), rows=2 * rows, k=k)

    def _fused(self, names: List[str], rows_each: List[int], k: int) -> _Mat:
        """Several GGUF matrices stacked row-wise into one runtime matrix with a shared global scale."""
        total = sum(rows_each)
        codes = self._empty((total, k // 2), self.torch.uint8)
        scales = self._empty((total, k // 16), self.torch.uint8)
        gs = None
        off = 0
        for n, r in zip(names, rows_each):
            s = self._scale_values(n)
            if s.size != 1:
                raise ValueError(f"{n}: expected a per-tensor scale")
            if gs is None:
                gs = float(s[0])
            elif float(s[0]) != gs:
                raise ValueError(f"{names}: fused matrices need identical global scales (lossless fusion)")
            rr, kk = self._nvfp4_into(n, codes, scales, r, total, off)
            if (rr, kk) != (r, k):
                raise ValueError(f"{n}: shape ({rr},{kk}) != expected ({r},{k})")
            off += r
        return _Mat(codes, scales, gs=gs, rows=total, k=k)

    def _experts_gate_up(self, gate: str, up: str) -> _Mat:
        cfg = self.cfg
        E, I, H = cfg.num_experts, cfg.expert_intermediate_size, cfg.hidden_size
        codes = self._empty((E * 2 * I, H // 2), self.torch.uint8)
        scales = self._empty((E * 2 * I, H // 16), self.torch.uint8)
        sg, su = self._scale_values(gate), self._scale_values(up)
        if sg.shape != (E,) or not np.array_equal(sg, su):
            raise ValueError(f"{gate}/{up}: per-expert gate and up scales must match (lossless fusion)")
        # gate|up interleaved by 8 rows within each expert (expert stride 2*I rows)
        self._nvfp4_into(gate, codes, scales, 8, 16, 0)
        self._nvfp4_into(up, codes, scales, 8, 16, 8)
        gs_dev = self.torch.from_numpy(sg.copy()).to(self.dev)
        return _Mat(codes, scales, gs_dev=gs_dev, rows=E * 2 * I, k=H)

    def _experts_down(self, name: str) -> _Mat:
        cfg = self.cfg
        E, I, H = cfg.num_experts, cfg.expert_intermediate_size, cfg.hidden_size
        codes = self._empty((E * H, I // 2), self.torch.uint8)
        scales = self._empty((E * H, I // 16), self.torch.uint8)
        s = self._scale_values(name)
        if s.shape != (E,):
            raise ValueError(f"{name}: expected per-expert scales")
        self._nvfp4_into(name, codes, scales)
        return _Mat(codes, scales, gs_dev=self.torch.from_numpy(s.copy()).to(self.dev), rows=E * H, k=I)

    def _single(self, name: str) -> _Mat:
        t = self.gguf.tensor(name)
        k = t.shape[0]
        rows = t.num_values // k
        return self._fused([name], [rows], k)

    def _bf16(self, name):
        t = self.gguf.tensor(name)
        if t.ggml_type != GGML_TYPE_BF16:
            raise ValueError(f"{name}: expected BF16")
        a = self.gguf.array(name)
        return self.torch.from_numpy(np.array(a, copy=True)).view(self.torch.bfloat16).to(self.dev)

    def _load_weights(self):
        cfg = self.cfg
        self._staging = None
        H, Q, KV = cfg.hidden_size, cfg.q_dim, cfg.kv_dim
        self.embed = self._bf16("token_embd.weight")
        self.out_norm = self._f32("output_norm.weight")
        self.lm_head = self._single("output.weight")
        self.layers: List[_Layer] = []
        for il in range(cfg.num_layers):
            p = f"blk.{il}."
            qkv = self._fused([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], [Q, KV, KV], H)
            o = self._single(p + "attn_output.weight")
            L = _Layer(attn_norm=self._f32(p + "attn_norm.weight"), ffn_norm=self._f32(p + "ffn_norm.weight"),
                       q_norm=self._f32(p + "attn_q_norm.weight"), k_norm=self._f32(p + "attn_k_norm.weight"),
                       sinks=self._f32(p + "attn_sinks.weight"), qkv=qkv, o=o, sliding=cfg.sliding_layers[il])
            if not cfg.is_moe(il):
                F = cfg.dense_intermediate_size
                L.gate_up = self._gate_up(p + "ffn_gate.weight", p + "ffn_up.weight", F, H)
                L.down = self._single(p + "ffn_down.weight")
            else:
                S = cfg.shared_intermediate_size
                L.router = self._bf16(p + "ffn_gate_inp.weight")
                L.router_t = L.router.t().contiguous()   # K-major copy for the many-token router
                L.exp_gate_up = self._experts_gate_up(p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight")
                L.exp_down = self._experts_down(p + "ffn_down_exps.weight")
                L.sh_gate_up = self._gate_up(p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight", S, H)
                L.sh_down = self._single(p + "ffn_down_shexp.weight")
            self.layers.append(L)
        self._staging = None
        self.torch.cuda.empty_cache()

    # ------------------------------------------------------------------ buffers
    def _alloc_buffers(self):
        torch, cfg = self.torch, self.cfg
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        H = cfg.hidden_size
        for L in self.layers:
            slots = self.ring if L.sliding else self.max_seq
            L.cache = torch.zeros((slots, 2 * cfg.kv_dim), dtype=bf, device=self.dev)
        inv = 1.0 / cfg.rope_theta ** (np.arange(0, cfg.head_dim // 2, dtype=np.float64) * 2 / cfg.head_dim)
        ang = np.arange(self.max_seq, dtype=np.float64)[:, None] * inv[None, :]
        self.rope_cos = torch.from_numpy(np.cos(ang).astype(np.float32)).to(self.dev)
        self.rope_sin = torch.from_numpy(np.sin(ang).astype(np.float32)).to(self.dev)
        self.x = torch.zeros(H, dtype=bf, device=self.dev)          # branch output feeding the next add
        self.res = torch.zeros(H, dtype=bf, device=self.dev)        # residual stream
        self.xn = torch.zeros(H, dtype=bf, device=self.dev)
        self.qkv_buf = torch.zeros(cfg.qkv_dim, dtype=bf, device=self.dev)
        self.q_buf = torch.zeros(cfg.q_dim, dtype=bf, device=self.dev)
        self.attn_out = torch.zeros(cfg.q_dim, dtype=bf, device=self.dev)
        self.o_part = torch.zeros((1024, cfg.num_heads, cfg.head_dim), dtype=f32, device=self.dev)
        self.ml_part = torch.zeros((1024, cfg.num_heads, 2), dtype=f32, device=self.dev)
        self.act = torch.zeros(max(cfg.dense_intermediate_size,
                                   cfg.num_experts_per_tok * cfg.expert_intermediate_size
                                   + cfg.shared_intermediate_size), dtype=bf, device=self.dev)
        self.router_logits = torch.zeros(cfg.num_experts, dtype=f32, device=self.dev)
        self.router_counter = torch.zeros(1, dtype=i32, device=self.dev)
        self.ids = torch.zeros(cfg.num_experts_per_tok, dtype=i32, device=self.dev)
        self.tw = torch.zeros(cfg.num_experts_per_tok, dtype=f32, device=self.dev)
        self.logits = torch.zeros(cfg.vocab_size, dtype=f32, device=self.dev)
        nb = self.fl.ARGMAX_BLOCKS
        self.cand_v = torch.zeros(nb, dtype=f32, device=self.dev)
        self.cand_i = torch.zeros(nb, dtype=i32, device=self.dev)
        self.argmax_counter = torch.zeros(1, dtype=i32, device=self.dev)
        # Device-side sequence state: the token at position pos is tokens[pos].
        self.tokens = torch.zeros(self.max_seq + 1, dtype=i32, device=self.dev)
        self.pos = torch.zeros(1, dtype=i32, device=self.dev)
        self.n_prompt_dev = torch.zeros(1, dtype=i32, device=self.dev)
        # Prefill (chunk of up to prefill_chunk tokens).
        C, T = self.prefill_chunk, cfg.num_experts_per_tok
        self.pf_x = torch.zeros((C, H), dtype=bf, device=self.dev)
        self.pf_res = torch.zeros((C, H), dtype=bf, device=self.dev)
        self.pf_xn = torch.zeros((C, H), dtype=bf, device=self.dev)
        self.pf_qkv = torch.zeros((C, cfg.qkv_dim), dtype=bf, device=self.dev)
        self.pf_q = torch.zeros((C, cfg.q_dim), dtype=bf, device=self.dev)
        self.pf_att = torch.zeros((C, cfg.q_dim), dtype=bf, device=self.dev)
        self.pf_act = torch.zeros((C, cfg.dense_intermediate_size), dtype=bf, device=self.dev)
        self.pf_ids = torch.zeros((C, T), dtype=i32, device=self.dev)
        self.pf_tw = torch.zeros((C, T), dtype=f32, device=self.dev)
        self.pf_sorted = torch.zeros(C * T, dtype=i32, device=self.dev)
        self.pf_inv = torch.zeros(C * T, dtype=i32, device=self.dev)
        self.pf_max_tiles = (C * T) // 64 + cfg.num_experts
        self.pf_tiles = torch.zeros(3 * self.pf_max_tiles, dtype=i32, device=self.dev)
        self.pf_ntiles = torch.zeros(1, dtype=i32, device=self.dev)
        self.pf_act_s = torch.zeros((C * T, cfg.expert_intermediate_size), dtype=bf, device=self.dev)
        self.pf_y_s = torch.zeros((C * T, H), dtype=bf, device=self.dev)
        self.pf_sh_act = torch.zeros((C, cfg.shared_intermediate_size), dtype=bf, device=self.dev)
        self.pf_sh = torch.zeros((C, H), dtype=bf, device=self.dev)

    # ------------------------------------------------------------------ decode step
    def _gemv(self, m: _Mat, x, out, epilogue, s):
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

    # ------------------------------------------------------------------ prefill
    def _gemm(self, A, m, lda, mat: _Mat, C, ldc, epi, s):
        self.fl.gemm_w4a16(A.data_ptr(), m, lda, mat.codes.data_ptr(), mat.scales.data_ptr(), mat.rows, mat.k,
                           mat.gs, C.data_ptr(), ldc, epi, s)

    def launch_prefill_chunk(self, m: int):
        """Enqueue the layers for ``m`` prompt tokens starting at device position ``pos``.

        Leaves the final hidden state of the chunk in (pf_x + pf_res) and advances ``pos`` by m.
        """
        fl, cfg, s = self.fl, self.cfg, self.stream.cuda_stream
        eps, H = cfg.rms_norm_eps, cfg.hidden_size
        scale = 1.0 / math.sqrt(cfg.head_dim)
        T, I = cfg.num_experts_per_tok, cfg.expert_intermediate_size
        fl.embed_rows_bf16(self.tokens.data_ptr(), self.pos.data_ptr(), m, self.embed.data_ptr(),
                           self.pf_x.data_ptr(), s)
        for il, L in enumerate(self.layers):
            fl.add_rmsnorm_bf16(self.pf_x.data_ptr(), self.pf_res.data_ptr(), L.attn_norm.data_ptr(),
                                self.pf_xn.data_ptr(), m, eps, 1 if il == 0 else 0, s)
            self._gemm(self.pf_xn, m, H, L.qkv, self.pf_qkv, cfg.qkv_dim, fl.EPI_BF16, s)
            ring = self.ring if L.sliding else 0
            fl.qk_norm_rope_kv_rows_bf16(self.pf_qkv.data_ptr(), m, L.q_norm.data_ptr(), L.k_norm.data_ptr(),
                                         self.rope_cos.data_ptr(), self.rope_sin.data_ptr(), self.pos.data_ptr(),
                                         self.pf_q.data_ptr(), L.cache.data_ptr(), ring, 1 if L.sliding else 0,
                                         eps, s)
            fl.prefill_attention_bf16(self.pf_q.data_ptr(), m, L.cache.data_ptr(), self.pos.data_ptr(), ring,
                                      cfg.sliding_window if L.sliding else 0, L.sinks.data_ptr(), scale,
                                      self.pf_att.data_ptr(), s)
            self._gemm(self.pf_att, m, cfg.q_dim, L.o, self.pf_x, H, fl.EPI_BF16, s)
            fl.add_rmsnorm_bf16(self.pf_x.data_ptr(), self.pf_res.data_ptr(), L.ffn_norm.data_ptr(),
                                self.pf_xn.data_ptr(), m, eps, 0, s)
            if L.router is None:
                self._gemm(self.pf_xn, m, H, L.gate_up, self.pf_act, cfg.dense_intermediate_size,
                           fl.EPI_SWIGLU_DENSE, s)
                self._gemm(self.pf_act, m, cfg.dense_intermediate_size, L.down, self.pf_x, H, fl.EPI_BF16, s)
            else:
                fl.router_rows_bf16(self.pf_xn.data_ptr(), m, L.router_t.data_ptr(), self.pf_ids.data_ptr(),
                                    self.pf_tw.data_ptr(), s)
                fl.moe_sort(self.pf_ids.data_ptr(), m, 64, self.pf_sorted.data_ptr(), self.pf_inv.data_ptr(),
                            self.pf_tiles.data_ptr(), self.pf_ntiles.data_ptr(), s)
                g = L.exp_gate_up
                fl.gemm_w4a16_grouped(self.pf_xn.data_ptr(), self.pf_sorted.data_ptr(), H, self.pf_tiles.data_ptr(),
                                      self.pf_ntiles.data_ptr(), self.pf_max_tiles, g.codes.data_ptr(),
                                      g.scales.data_ptr(), 2 * I, H, g.gs_dev.data_ptr(), self.pf_act_s.data_ptr(),
                                      I, fl.EPI_SWIGLU_ROUTED, s)
                d = L.exp_down
                fl.gemm_w4a16_grouped(self.pf_act_s.data_ptr(), 0, I, self.pf_tiles.data_ptr(),
                                      self.pf_ntiles.data_ptr(), self.pf_max_tiles, d.codes.data_ptr(),
                                      d.scales.data_ptr(), H, I, d.gs_dev.data_ptr(), self.pf_y_s.data_ptr(), H,
                                      fl.EPI_BF16, s)
                self._gemm(self.pf_xn, m, H, L.sh_gate_up, self.pf_sh_act, cfg.shared_intermediate_size,
                           fl.EPI_SWIGLU_DENSE, s)
                self._gemm(self.pf_sh_act, m, cfg.shared_intermediate_size, L.sh_down, self.pf_sh, H,
                           fl.EPI_BF16, s)
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

    def prefill(self, prompt_tokens: List[int], logits_cb=None):
        """Run the prompt through the chunked prefill path and pick the first generated token.

        ``logits_cb(chunk_start, m, logits)`` (evaluation only) receives fp32 logits for every prompt row.
        """
        torch = self.torch
        self.reset(prompt_tokens)
        n = len(prompt_tokens)
        C = self.prefill_chunk
        buf = None
        with torch.cuda.stream(self.stream):
            for c0 in range(0, n, C):
                m = min(C, n - c0)
                self.launch_prefill_chunk(m)
                if logits_cb is not None:
                    if buf is None:
                        buf = torch.empty((C, self.cfg.vocab_size), dtype=torch.float32, device=self.dev)
                    self.launch_prefill_logits(m, buf)
                    self.stream.synchronize()
                    logits_cb(c0, m, buf[:m])
            self.launch_prefill_logits(m)
        return n

    # ------------------------------------------------------------------ control
    def reset(self, prompt_tokens: List[int]):
        """Load a prompt into the device token buffer and rewind the position."""
        torch = self.torch
        n = len(prompt_tokens)
        if n == 0 or n > self.max_seq:
            raise ValueError(f"prompt length {n} outside [1, {self.max_seq}]")
        self.tokens.zero_()
        self.tokens[:n].copy_(torch.tensor(prompt_tokens, dtype=torch.int32))
        self.pos.zero_()
        self.n_prompt_dev.fill_(n)
        torch.cuda.synchronize(self.dev)

    def capture(self, attn_splits: Optional[int] = None):
        """Capture one decode step into a CUDA graph (call after load)."""
        torch = self.torch
        saved = (self.pos.clone(), self.tokens.clone())
        with torch.cuda.stream(self.stream):
            self.launch_step(attn_splits=attn_splits)   # warm-up (lazy module/kernel init)
        self.stream.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.stream):
            self.launch_step(attn_splits=attn_splits)
        self.stream.synchronize()
        self.pos.copy_(saved[0]); self.tokens.copy_(saved[1])
        torch.cuda.synchronize(self.dev)
        self.graph = g
        return g

    def replay(self, n: int = 1):
        for _ in range(n):
            self.graph.replay()
