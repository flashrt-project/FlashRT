"""Flint runtime on RTX SM120: buffers, weight upload, and the native kernel sequence.

Python here only prepares buffers and launches kernels that take raw device
pointers; the steady-state decode step is a fixed kernel sequence that reads
the position and the token from device memory, so it is captured once into a
CUDA graph and replayed for every step (prompt tokens and generated tokens
alike). Prefill chunks are captured per launch shape on first use and replayed
as well. PyTorch is used as the allocator and for graph capture only.

Numerics follow the official W4A16 contract: exact NVFP4 dequantization, BF16
activations and residual stream, fp32 accumulation, BF16 KV cache, fp32 logits.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from flash_rt.models.flint.config import FlintConfig, load_config
from flash_rt.models.flint.pipeline_rtx import FlintPipeline
from flash_rt.models.flint.gguf import GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_NVFP4, GGUFFile
from flash_rt.models.flint.weights import check_scale_bytes

try:  # the extension is optional at import time (see _require_kernels)
    from flash_rt import flash_rt_flint as _fl
except ImportError:  # pragma: no cover - exercised on machines without the build
    _fl = None


_REQUIRED_KERNELS = (
    'add_rmsnorm_bf16',
    'add_rmsnorm_pdl_bf16',
    'advance_pos',
    'argmax_next_f32',
    'decode_attention_bf16',
    'decode_attention_fused_bf16',
    'embed_at_pos_bf16',
    'embed_pdl_bf16',
    'embed_rows_bf16',
    'gemm_w4a16',
    'gemm_w4a16_grouped',
    'gemm_w4a16_qkrope',
    'gemv_fused_w4a16',
    'gemv_w4a16',
    'lm_head_greedy_w4a16',
    'moe_combine_norm_rows_bf16',
    'moe_combine_rows_bf16',
    'moe_down_combine_bf16',
    'moe_down_combine_pdl_bf16',
    'moe_gate_up_bf16',
    'moe_gate_up_pdl_bf16',
    'moe_scan_place',
    'prefill_attention_bf16',
    'qk_norm_rope_kv_bf16',
    'qk_norm_rope_kv_rows_bf16',
    'repack_nvfp4',
    'router_norm_bf16',
    'router_topk_bf16',
    'router_topk_rows_bf16',
    'set_gemv_launch_config',
    'set_lm_head_rows',
    'set_moe_launch_config',
    'ARGMAX_BLOCKS',
    'EPI_BF16',
    'EPI_F32',
    'EPI_SWIGLU_DENSE',
    'EPI_SWIGLU_ROUTED',
)


def _require_kernels():
    if _fl is None:
        raise RuntimeError(
            "flash_rt_flint is not built. Configure with -DGPU_ARCH=120 "
            "-DFLASHRT_ENABLE_FLINT=ON and rebuild (see docs/flint_usage.md).")
    missing = [name for name in _REQUIRED_KERNELS if not hasattr(_fl, name)]
    if missing:
        raise RuntimeError("flash_rt_flint is missing required symbols: " + ", ".join(missing) +
                           ". Rebuild with FLASHRT_ENABLE_FLINT=ON.")
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
    exp_gate_up: Optional[_Mat] = None
    exp_down: Optional[_Mat] = None
    sh_gate_up: Optional[_Mat] = None
    sh_down: Optional[_Mat] = None
    cache: object = None
    pf_ranges: object = None
    pf_n: int = 0


class FlintRuntime(FlintPipeline):
    """Owns every device buffer and launches the Flint kernel sequence."""

    def __init__(self, gguf_path: str, max_seq: int = 32768, sliding_ring: int = 16384,
                 attn_splits: int = 16, prefill_chunk: Optional[int] = None, device: int = 0):
        self.torch = torch
        self.fl = _require_kernels()
        if max_seq < 1:
            raise ValueError("max_seq must be positive")
        if not 1 <= attn_splits <= 1024:
            raise ValueError("attn_splits must be in [1, 1024]")
        if prefill_chunk is not None and not 1 <= prefill_chunk <= 8192:
            raise ValueError("prefill_chunk must be in [1, 8192]")
        if sliding_ring < 1 or sliding_ring & (sliding_ring - 1):
            raise ValueError("sliding_ring must be a power of two")
        self.gguf = GGUFFile(gguf_path)
        self.cfg: FlintConfig = load_config(self.gguf)
        if sliding_ring < self.cfg.sliding_window:
            raise ValueError(f"sliding_ring {sliding_ring} < window {self.cfg.sliding_window}")
        if max_seq > self.cfg.context_length:
            raise ValueError(f"max_seq {max_seq} exceeds checkpoint context {self.cfg.context_length}")
        self.max_seq = int(max_seq)
        self.ring = int(sliding_ring)
        # Oldest absolute position still resident in every sliding KV ring. Track physical writes,
        # including padded prefill rows and decode lookahead, rather than the logical token prefix.
        self.swa_kv_start = 0
        self.attn_splits = int(attn_splits)
        self.dev = torch.device("cuda", device)
        capability = torch.cuda.get_device_capability(self.dev)
        if capability != (12, 0):
            raise RuntimeError(f"Flint requires SM120; device has compute capability {capability}")
        torch.cuda.set_device(self.dev)
        if prefill_chunk is None:
            # Buffers hold up to 8192 tokens; full chunks are the largest multiple of lcm(16, 4 * SMs) within
            # that, so a chunk's attention units (16-token blocks x 4 KV heads) fill whole waves.
            sms = torch.cuda.get_device_properties(self.dev).multi_processor_count
            q = math.lcm(16, 4 * sms)
            self.pf_capacity = 8192
            prefill_chunk = 8192 // q * q if q <= 4096 else 8192
        else:
            self.pf_capacity = int(prefill_chunk)
        self.prefill_chunk = int(prefill_chunk)
        if self.pf_capacity + self.cfg.sliding_window > sliding_ring:
            raise ValueError("prefill_chunk + sliding_window must fit in sliding_ring")
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
        with warnings.catch_warnings():  # read-only mmap view: only ever copied from
            warnings.simplefilter("ignore", UserWarning)
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
        # Fused decode step: ping-pong residual, branch outputs, attention counters.
        self.res2 = torch.zeros((2, H), dtype=bf, device=self.dev)
        self.x_attn = torch.zeros(H, dtype=bf, device=self.dev)
        self.attn_counters = torch.zeros(cfg.num_kv_heads, dtype=i32, device=self.dev)
        # Decode attention launch configs (measured split/warp search on SM120, HBM-bound KV):
        # sliding layers (513 keys): 8 splits x 8 warps; global layers by position bucket.
        self.swa_attn = (8, 8)
        self.full_attn_warps = 8
        # (pos upper bound, attention splits, attention warps, (qkv rows, qkv warps, o rows, o warps)):
        # the wider GEMV shapes win below ~48K context and lose at 100K (measured A/B/A).
        g_short, g_long = (4, 8, 2, 8), (8, 4, 4, 4)
        self.attn_buckets = ((512, 1, 4, g_short), (12288, 8, 8, g_short), (49152, 16, 8, g_short),
                             (1 << 30, 16, 8, g_long))
        import os as _os
        # L2 warm-up of each layer's o_proj/router/shared-expert weights from the attention kernel when the
        # attention moves little data (sliding layers; global layers in short-context buckets).
        self.attn_pf_max_splits = 8
        bcfg = _os.environ.get("FLINT_ATTN_BUCKETS")      # "ub:splits:warps;..." (search/bisect hook)
        if bcfg:
            self.attn_buckets = tuple((*[int(v) for v in b.split(":")[:3]], (4, 4, 2, 4)) for b in bcfg.split(";"))
        scfg = _os.environ.get("FLINT_SWA_ATTN")          # "splits:warps"
        if scfg:
            self.swa_attn = tuple(int(v) for v in scfg.split(":"))
        gcfg = _os.environ.get("FLINT_GEMV_CFG")          # "qkv_rows,qkv_warps,o_rows,o_warps" (search hook)
        if gcfg:
            self.fl.set_gemv_launch_config(*[int(v) for v in gcfg.replace("_", ",").split(",")])
        mcfg = _os.environ.get("FLINT_MOE_CFG")           # 6 ints, see set_moe_launch_config (search hook)
        if mcfg:
            self.fl.set_moe_launch_config(*[int(v) for v in mcfg.replace("_", ",").split(",")])
        # Decode tail: final norm + LM head + greedy pick in one kernel (0: three kernels).
        self.lm_greedy = _os.environ.get("FLINT_LM_GREEDY", "1") != "0"
        if _os.environ.get("FLINT_LM_ROWS"):              # rows per warp of that kernel (search hook)
            self.fl.set_lm_head_rows(int(_os.environ["FLINT_LM_ROWS"]))
        for L in self.layers:
            rng = []
            for t in (L.o.codes, L.o.scales):
                rng += [t.data_ptr(), t.numel()]
            if L.router is not None:
                for t in (L.router, L.sh_gate_up.codes, L.sh_gate_up.scales, L.sh_down.codes, L.sh_down.scales):
                    rng += [t.data_ptr(), t.numel() * t.element_size()]
            L.pf_ranges = torch.tensor(rng, dtype=torch.int64, device=self.dev)
            L.pf_n = len(rng) // 2
        # Prefill (chunk of up to prefill_chunk tokens).
        C, T = self.pf_capacity, cfg.num_experts_per_tok
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
        self.pf_tiles = torch.zeros(3 * ((C * T) // 16 + cfg.num_experts), dtype=i32, device=self.dev)
        self.pf_ntiles = torch.zeros(1, dtype=i32, device=self.dev)
        self.pf_counts = torch.zeros(cfg.num_experts, dtype=i32, device=self.dev)   # re-zeroed by moe_scan_place
        self.pf_act_s = torch.zeros((C * T, cfg.expert_intermediate_size), dtype=bf, device=self.dev)
        self.pf_y_s = torch.zeros((C * T, H), dtype=bf, device=self.dev)
        self.pf_sh_act = torch.zeros((C, cfg.shared_intermediate_size), dtype=bf, device=self.dev)
        self.pf_sh = torch.zeros((C, H), dtype=bf, device=self.dev)
        # Prefill attention: KV splits (host-chosen per chunk) with partials merged in-kernel.
        self.sm_count = torch.cuda.get_device_properties(self.dev).multi_processor_count
        self.pf_attn_variant = int(_os.environ.get("FLINT_PF_ATTN_VARIANT", "0"))
        self.pf_attn_pkv = 64 if self.pf_attn_variant in (2, 3) else 32
        self.pf_split_max = int(_os.environ.get("FLINT_PF_SPLIT_MAX", "4"))
        units = (C + 15) // 16 * 4
        self.pf_split_units = 2048            # partial buffers hold units * splits <= this (128 MiB)
        self.pf_opart = torch.empty(self.pf_split_units * 128 * 128, dtype=f32, device=self.dev)
        self.pf_ml = torch.empty(self.pf_split_units * 128 * 2, dtype=f32, device=self.dev)
        self.pf_attn_ctr = torch.zeros(units, dtype=i32, device=self.dev)
        # Prefill chunks replay CUDA graphs captured on first use per launch shape (LRU-bounded).
        self.pf_graph_on = _os.environ.get("FLINT_PF_GRAPH", "1") != "0"
        # Prefill epilogue fusions (MoE combine + next input norm; QKV GEMM + Q/K norm + RoPE + KV write).
        self.pf_fuse = _os.environ.get("FLINT_PF_FUSE", "1") != "0"
        self.pf_graph_max = 64
        self.pf_graphs: "OrderedDict[tuple, object]" = OrderedDict()
        self._pf_pool = None
        # LM-head greedy candidates (one per block); allocated last so earlier buffers keep their placement.
        self.lm_cand_v = torch.zeros(-(-cfg.vocab_size // 16), dtype=f32, device=self.dev)
        self.lm_cand_i = torch.zeros(-(-cfg.vocab_size // 16), dtype=i32, device=self.dev)

    # ------------------------------------------------------------------ prefill


    def _pf_rows(self, m: int, pos0: int) -> int:
        """Rows launched for an m-token chunk: m rounded up to 16 so that graphs are shared across nearby
        lengths. Padded rows only write KV past the chunk, which decode overwrites before any query reads it
        (attention reads positions <= the query's own), take no expert slots, and each valid row's result is
        independent of them."""
        b = -(-m // 16) * 16
        return b if b <= self.pf_capacity and pos0 + b <= self.max_seq else m

    def _pf_graph(self, b: int, pos0: int):
        """CUDA graph of ``launch_prefill_chunk(b)``. Positions come from device memory, so the only host
        inputs are b and the attention split counts (a function of pos0): one graph per such key."""
        torch, cfg = self.torch, self.cfg
        key = (b, self._pf_attn_splits(b, pos0, 0), self._pf_attn_splits(b, pos0, cfg.sliding_window))
        g = self.pf_graphs.pop(key, None)
        if g is None:
            if not self.pf_graphs:   # the graphs share one (empty) memory pool, released with the last of them
                self._pf_pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            g.capture_begin(pool=self._pf_pool, capture_error_mode="thread_local")
            try:
                self.launch_prefill_chunk(b, pos0)
            finally:
                g.capture_end()
            if len(self.pf_graphs) >= self.pf_graph_max:
                self.pf_graphs.popitem(last=False)
        self.pf_graphs[key] = g
        return g

    def run_prefill_chunk(self, m: int, pos0: int):
        """Enqueue one chunk on the current stream (graph replay unless FLINT_PF_GRAPH=0); advances ``pos`` by m."""
        b = self._pf_rows(m, pos0) if self.pf_graph_on else m
        self.swa_kv_start = max(self.swa_kv_start, pos0 + b - self.ring)
        if not self.pf_graph_on:
            self.launch_prefill_chunk(m, pos0)
            return
        self._pf_graph(b, pos0).replay()
        if b != m:
            self.fl.advance_pos(self.pos.data_ptr(), m - b, self.stream.cuda_stream)


    def chunk_plan(self, n: int) -> List[int]:
        """Chunk sizes for an n-token prompt: one chunk if it fits the buffers, else wave-aligned full chunks
        and the remainder (measured: a short last chunk costs less than splitting the tail evenly)."""
        cap, full, out = self.pf_capacity, self.prefill_chunk, []
        while n > cap:
            out.append(full)
            n -= full
        out.append(n)
        return out

    def prefill(self, prompt_tokens: List[int], logits_cb=None, start: int = 0):
        """Run the prompt through the chunked prefill path and pick the first generated token.

        ``start`` > 0 continues a sequence whose KV cache already holds positions [0, start) of these same
        tokens (e.g. the previous request's prompt and output): only positions >= start are processed.
        ``logits_cb(chunk_start, m, logits)`` (evaluation only) receives fp32 logits for every processed row.
        """
        torch = self.torch
        n = len(prompt_tokens)
        if not 0 <= start < n:
            raise ValueError(f"prefill start {start} outside [0, {n})")
        if start and max(0, start - self.cfg.sliding_window + 1) < self.swa_kv_start:
            raise ValueError("sliding KV history was overwritten; prefill the full prompt with start=0")
        if start == 0:
            self.reset(prompt_tokens)
        else:
            self._load_suffix(prompt_tokens, start)
        buf = None
        with torch.cuda.stream(self.stream):
            c0 = start
            for m in self.chunk_plan(n - start):
                self.run_prefill_chunk(m, c0)
                if logits_cb is not None:
                    if buf is None:
                        buf = torch.empty((self.pf_capacity, self.cfg.vocab_size), dtype=torch.float32,
                                          device=self.dev)
                    self.launch_prefill_logits(m, buf)
                    self.stream.synchronize()
                    logits_cb(c0, m, buf[:m])
                c0 += m
            self.launch_prefill_logits(m)
        self.host_pos = n
        return n


    # ------------------------------------------------------------------ control
    def _load_suffix(self, prompt_tokens: List[int], start: int):
        """Write tokens [start, n) after a cached prefix and point the device position at ``start``."""
        torch = self.torch
        n = len(prompt_tokens)
        if n > self.max_seq:
            raise ValueError(f"prompt length {n} exceeds max_seq {self.max_seq}")
        self.tokens[start:].zero_()
        self.tokens[start:n].copy_(torch.tensor(prompt_tokens[start:], dtype=torch.int32))
        self.pos.fill_(start)
        self.n_prompt_dev.fill_(n)
        self.host_pos = start
        torch.cuda.synchronize(self.dev)

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
        self.host_pos = 0
        torch.cuda.synchronize(self.dev)
        self.swa_kv_start = 0

    def capture(self, attn_splits: Optional[int] = None, fused: Optional[bool] = None):
        """Capture the decode step into CUDA graphs (call after load).

        The fused step is captured once per attention bucket (``attn_buckets``); ``replay``
        picks the graph from the host-tracked position. ``attn_splits`` forces one graph.
        """
        import os
        torch = self.torch
        if fused is None:
            fused = os.environ.get("FLINT_DECODE", "fused") != "basic"
        if fused and attn_splits is None:
            configs = list(self.attn_buckets)
        else:
            configs = [(1 << 30, attn_splits or self.attn_splits, self.full_attn_warps, None)]
        saved = (self.pos.clone(), self.tokens.clone())
        self.stream.wait_stream(torch.cuda.current_stream(self.dev))
        self.graphs = []
        gemv_env = os.environ.get("FLINT_GEMV_CFG")
        for ub, sp, w, gcfg in configs:
            if gcfg is not None and not gemv_env:
                self.fl.set_gemv_launch_config(*gcfg)   # captured into this bucket's graph
            if fused:
                step = lambda sp=sp, w=w: self.launch_step_fused(attn_splits=sp, attn_warps=w)  # noqa: E731
            else:
                step = lambda sp=sp: self.launch_step(attn_splits=sp)  # noqa: E731
            with torch.cuda.stream(self.stream):
                step()   # warm-up (lazy module/kernel init, smem attributes)
            self.stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self.stream):
                step()
            self.stream.synchronize()
            self.pos.copy_(saved[0]); self.tokens.copy_(saved[1])
            self.stream.wait_stream(torch.cuda.current_stream(self.dev))
            self.graphs.append((ub, g))
        torch.cuda.synchronize(self.dev)
        self.graph = self.graphs[-1][1]
        self.host_pos = 0
        return self.graphs

    def _graph_for(self, pos: int):
        for ub, g in self.graphs:
            if pos < ub:
                return g
        return self.graphs[-1][1]

    def replay(self, n: int = 1):
        """Replay n decode steps; each step processes the token at host_pos and advances it."""
        if n < 0 or self.host_pos + n > self.max_seq:
            raise ValueError(f"decode range [{self.host_pos}, {self.host_pos + n}) exceeds max_seq {self.max_seq}")
        self.swa_kv_start = max(self.swa_kv_start, self.host_pos + n - self.ring)
        import os
        if os.environ.get("FLINT_EAGER") == "1":     # debugging aid: launch the step without a graph
            with self.torch.cuda.stream(self.stream):
                for _ in range(n):
                    ub, sp, w, gcfg = next(b for b in self.attn_buckets if self.host_pos < b[0])
                    if gcfg is not None:
                        self.fl.set_gemv_launch_config(*gcfg)
                    self.launch_step_fused(attn_splits=sp, attn_warps=w)
                    self.host_pos += 1
            return
        for _ in range(n):
            self._graph_for(self.host_pos).replay()
            self.host_pos += 1
