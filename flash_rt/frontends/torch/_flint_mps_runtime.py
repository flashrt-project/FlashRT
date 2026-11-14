"""Flint runtime on Apple Silicon: weight upload, buffers and the native Metal programs.

Weights stay in the checkpoint's MLX affine form (4-bit codes with BF16 scale/bias per 64 inputs,
8-bit embedding, BF16 router, norms and sinks): the bytes are copied verbatim into one shared Metal
buffer and addressed through a per-layer descriptor table of 64-bit offsets. The KV cache is BF16:
global layers hold ``max_seq`` slots (slot = position), sliding-window layers a 513-slot ring in the
reference engine's layout (after a prefill of ``initial`` positions the ring holds them linearly, and
decode position ``p`` uses slot ``(p - initial) % 513`` once ``initial >= 513``, ``p % 513`` before).
The steady-state decode step is one resident dispatch that runs up to eight tokens; the host writes
the input token and position, runs the program and reads the produced tokens back.

Numerics are those of the export (rec's decode arithmetic): FP32 accumulation, BF16 activations,
residual stream and KV, BF16 logits.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from flash_rt.hardware.apple_native.native import NativePlan
from flash_rt.models.flint.config import FlintConfig, FlintConfigError
from flash_rt.models.flint.mlx_checkpoint import AffineMatrix, MLXCheckpoint, bf16_to_f32
from flash_rt.models.flint.pipeline_mps import (
    ATTENTION, BLOCK, D_BIAS, D_DOWN, D_EMBED, D_FINAL_NORM, D_FLAGS, D_GATE, D_HEAD, D_IN_NORM, D_K, D_K_NORM,
    D_KV, D_KV_CAP, D_O, D_POST_NORM, D_Q, D_Q_NORM, D_ROUTER, D_SH_DOWN, D_SH_GATE, D_SH_UP, D_SINKS, D_UP, D_V,
    DECODE_KERNELS, DESCRIPTOR, GROUPED_TM, SMALL_ROWS, WORKERS, FlintPipelineMps, FlintPrefillMps, check_geometry)

ALIGN = 256
MAX_CHUNK = 2048                # prefill chunk limit (router split-K and expert tiling rules assume <= 2048 rows)
WORKSPACE = 78848               # BF16 elements: q/k/v, attention output, o, dense gate/up/act, expert outputs
PARTIALS = 32 * 1024 * 128 // 2  # FP32: per-head partitions x 128 (or BF16 rounded partitions)
STATISTICS = 32 * 1024 * 2
ROUTING = 1280
CONTROL = 1024


class _Layout:
    """Byte layout of the weight buffer (offsets aligned to 256 bytes)."""

    def __init__(self):
        self.size = 0
        self.items: List[tuple] = []

    def add(self, array: np.ndarray) -> int:
        self.size += (-self.size) % ALIGN
        offset = self.size
        self.items.append((offset, array))
        self.size += array.nbytes
        return offset


PARAM_BYTES = 2 << 20   # prefill parameter blocks (all prefill programs)

_PARAM_LAYOUTS = {
    # kind: (uint32 fields, uint64 fields, float32 fields) in MSL struct order (uints, then ulongs, then floats)
    "gemm": (("rows", "columns", "inner", "x_stride", "y_stride", "experts", "limit", "per_row"),
             ("w", "s", "b", "w2", "s2", "b2"), ()),
    "embed": (("hidden", "pad"), ("w", "s", "b"), ()),
    "rms": (("hidden", "last"), ("w",), ("eps", "pad2")),
    "qmv": (("inner", "columns", "x_stride", "y_stride", "limit", "experts", "per_row", "pad"),
            ("w", "s", "b", "w2", "s2", "b2", "x", "y"), ()),
    "qk": (("q_heads", "kv_heads", "sliding", "cap", "scratch", "pad0", "pad1", "pad2"), ("q_norm", "k_norm", "kv"),
           ("eps", "logbase", "pad3", "pad4")),
    "ring": (("kv_heads", "ring", "scratch_cap", "window"), ("ring_kv", "scratch_kv"), ()),
    "attention": (("kv_heads", "cap", "sliding", "window"), ("kv", "sinks"), ("scale", "pad")),
    "router": ((), ("weight", "bias"), ()),
    "scan": (("tile_rows", "pad"), (), ()),
    "uint": (("value",), (), ()),
}


class FlintRuntime(FlintPipelineMps, FlintPrefillMps):
    """Owns the Metal plan, every buffer, the decode programs and the chunked prefill program (batch 1)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 32768, library: Optional[str] = None,
                 prefill_chunk: int = MAX_CHUNK, prefill_layers: int = 0, layer_programs=(),
                 prefill_attention: Optional[str] = None):
        """``prefill_chunk``: rows per prefill chunk (the reference engine uses 2048 on M4 Pro / M4 Max; a
        different value changes the chunking and hence the rounding). ``prefill_attention``: key of
        ``pipeline_mps.ATTENTION``. ``prefill_layers`` / ``layer_programs``: diagnostics (prefill only the
        first layers; extra single-layer programs over the residual stream in ``h``)."""
        if max_seq < 1:
            raise ValueError("max_seq must be positive")
        if not 1 <= prefill_chunk <= MAX_CHUNK:
            raise ValueError(f"prefill_chunk must be in [1, {MAX_CHUNK}]")
        if prefill_attention is not None and prefill_attention not in ATTENTION:
            raise ValueError(f"prefill_attention must be one of {sorted(ATTENTION)}")
        self.ckpt = MLXCheckpoint(checkpoint_dir)
        self.cfg: FlintConfig = self.ckpt.config
        check_geometry(self.cfg)
        if max_seq > self.cfg.context_length:
            raise ValueError(f"max_seq {max_seq} exceeds the checkpoint context {self.cfg.context_length}")
        self.max_seq = int(max_seq)
        if prefill_attention is not None:
            self.attention = prefill_attention
        self.plan = NativePlan(library)
        self.buf: Dict[str, int] = {}
        started = time.perf_counter()
        self._upload_weights()
        self.ckpt.close()   # weights now live in the Metal buffer; release the shard mappings
        self._allocate_state()
        self.prefill_chunk = int(prefill_chunk)
        self._allocate_prefill(self.prefill_chunk)
        self.vocab_param = self.param("uint", value=self.cfg.vocab_size)
        self.programs = {name: self.record_decode(kernel) for name, kernel in DECODE_KERNELS.items()}
        self.programs["noise"] = self.record_noise()
        self.programs["prefill"] = self.record_prefill(self.prefill_chunk, prefill_layers)
        self.programs["prefill_small"] = self.record_prefill(self.prefill_chunk, prefill_layers, small=True)
        self.programs["prefill_head"] = self.record_prefill_head()
        for i in layer_programs:   # single-layer programs over the residual stream in `h` (diagnostics)
            self.programs[f"prefill_layer{i}"] = self.record_prefill(self.prefill_chunk, i + 1, first=i)
        self.plan.write(self.buf["args"], 0, np.frombuffer(bytes(self._param_blob), dtype=np.uint8))
        self.plan.finish(resident=True)
        self._state = self.plan.view(self.buf["state"], np.uint32, 4)
        self._params = self.plan.view(self.buf["params"], np.uint32, 16)
        self._packet = self.plan.view(self.buf["packet"], np.uint32, 17)
        self._diagnostic = self.plan.view(self.buf["diagnostic"], np.uint32, WORKERS)
        self._control = self.plan.view(self.buf["control"], np.uint32, CONTROL)
        self.load_seconds = time.perf_counter() - started
        self.pos = 0
        self.initial = 0        # sliding-ring layout (prompt length of the last prefill)
        self.swa_kv_start = 0   # first position whose sliding-window KV is still in the rings

    # ------------------------------------------------------------------ weights
    def _matrix(self, layout: _Layout, m: AffineMatrix, rows_per_slice: int) -> List[int]:
        w = layout.add(m.codes)
        s = layout.add(m.scales)
        b = layout.add(m.biases)
        return [w, s, b, (rows_per_slice << 32) | m.k]

    def _affine(self, name: str, rows: int, k: int, bits: int = 4, experts: Optional[int] = None) -> AffineMatrix:
        """Affine matrix ``name`` with the shape the kernels expect: [rows, k] (or [experts, rows, k])."""
        m = self.ckpt.affine(name, bits=bits, k=k)
        shape = (rows,) if experts is None else (experts, rows)
        if m.scales.shape[:-1] != shape:
            raise FlintConfigError(f"{name}: {m.scales.shape[:-1]} output rows, expected {shape}")
        return m

    def _bf16(self, name: str, shape) -> np.ndarray:
        t = self.ckpt.tensor(name)
        if t.dtype != np.uint16 or t.shape != tuple(shape):
            raise FlintConfigError(f"{name}: {t.dtype} {t.shape}, expected BF16 {tuple(shape)}")
        return t

    def _upload_weights(self):
        cfg, ck = self.cfg, self.ckpt
        H, E = cfg.hidden_size, cfg.num_experts
        layout = _Layout()
        table = np.zeros((cfg.num_layers + 1, DESCRIPTOR), dtype=np.uint64)
        zero_bias = layout.add(np.zeros(E, dtype=np.float32))
        names = set(ck.names())
        kv_offset = 0
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            d = table[i]
            sliding = cfg.sliding_layers[i]
            sparse = cfg.is_moe(i)
            d[D_FLAGS] = (0 if sliding else 1) | (2 if sparse else 0)
            d[D_IN_NORM] = layout.add(self._bf16(p + "input_layernorm.weight", (H,)))
            d[D_POST_NORM] = layout.add(self._bf16(p + "post_attention_layernorm.weight", (H,)))
            d[D_Q_NORM] = layout.add(self._bf16(p + "self_attn.q_norm.weight", (cfg.head_dim,)))
            d[D_K_NORM] = layout.add(self._bf16(p + "self_attn.k_norm.weight", (cfg.head_dim,)))
            d[D_SINKS] = layout.add(self._bf16(p + "self_attn.softmax_offset", (cfg.num_heads,)))
            for base, name, rows, k in ((D_Q, "q_proj", cfg.q_dim, H), (D_K, "k_proj", cfg.kv_dim, H),
                                        (D_V, "v_proj", cfg.kv_dim, H), (D_O, "o_proj", H, cfg.q_dim)):
                m = self._affine(p + "self_attn." + name, rows, k)
                d[base:base + 4] = self._matrix(layout, m, rows)
            if sparse:
                I, S = cfg.expert_intermediate_size, cfg.shared_intermediate_size
                d[D_ROUTER] = layout.add(self._bf16(p + "mlp.gate.weight", (E, H)))
                bias_name = p + "mlp.gate.expert_bias"
                if bias_name in names:   # FP32 for the kernels
                    bias = ck.tensor(bias_name)
                    if bias.shape != (E,) or bias.dtype not in (np.uint16, np.float32):
                        raise FlintConfigError(f"{bias_name}: {bias.dtype} {bias.shape}, expected ({E},)")
                    d[D_BIAS] = layout.add(bf16_to_f32(bias) if bias.dtype == np.uint16 else bias)
                else:
                    d[D_BIAS] = zero_bias
                for base, name, rows, k in ((D_GATE, "gate_proj", I, H), (D_UP, "up_proj", I, H),
                                            (D_DOWN, "down_proj", H, I)):
                    m = self._affine(p + "mlp.switch_mlp." + name, rows, k, experts=E)
                    d[base:base + 4] = self._matrix(layout, m, rows)
                for base, name, rows, k in ((D_SH_GATE, "gate_proj", S, H), (D_SH_UP, "up_proj", S, H),
                                            (D_SH_DOWN, "down_proj", H, S)):
                    m = self._affine(p + "mlp.shared_expert." + name, rows, k)
                    d[base:base + 4] = self._matrix(layout, m, rows)
            else:
                F = cfg.dense_intermediate_size
                for base, name, rows, k in ((D_GATE, "gate_proj", F, H), (D_UP, "up_proj", F, H),
                                            (D_DOWN, "down_proj", H, F)):
                    m = self._affine(p + "mlp." + name, rows, k)
                    d[base:base + 4] = self._matrix(layout, m, rows)
            cap = cfg.sliding_window if sliding else self.max_seq
            d[D_KV], d[D_KV_CAP] = kv_offset, cap
            kv_offset += 2 * cfg.num_kv_heads * cap * cfg.head_dim
        f = table[cfg.num_layers]
        f[D_FINAL_NORM] = layout.add(self._bf16("model.norm.weight", (H,)))
        emb = self._affine("model.embed_tokens", cfg.vocab_size, H, bits=8)
        f[D_EMBED:D_EMBED + 4] = self._matrix(layout, emb, cfg.vocab_size)
        head = self._affine("lm_head", cfg.vocab_size, H)
        f[D_HEAD:D_HEAD + 4] = self._matrix(layout, head, cfg.vocab_size)

        self.weight_bytes = layout.size
        self.buf["weights"] = self.plan.buffer(layout.size)
        dst = self.plan.view(self.buf["weights"], np.uint8, layout.size)
        for offset, array in layout.items:
            flat = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
            dst[offset:offset + flat.nbytes] = flat
        self.buf["layers"] = self.plan.buffer(table.nbytes)
        self.plan.write(self.buf["layers"], 0, table)
        self.layer_table = table
        self.kv_elements = kv_offset

    # -------------------------------------------------------------------- state
    def _allocate_state(self):
        v = self.cfg.vocab_size
        sizes = {
            "kv": self.kv_elements * 2,
            "params": 64,
            "noise": BLOCK * v * 4,
            "logits": BLOCK * v * 2,
            "trace": BLOCK * (self.cfg.num_layers + 1) * self.cfg.hidden_size * 2,
            "packet": 17 * 4,
            "workspace": WORKSPACE * 2,
            "partials": PARTIALS * 4,
            "statistics": STATISTICS * 4,
            "routing": ROUTING * 4,
            "control": CONTROL * 4,
            "diagnostic": WORKERS * 4,
        }
        for name, nbytes in sizes.items():
            self.buf[name] = self.plan.buffer(nbytes)

    # ------------------------------------------------------------------ prefill
    def _allocate_prefill(self, C: int):
        cfg = self.cfg
        H, W, K = cfg.hidden_size, cfg.sliding_window, cfg.num_experts_per_tok
        sizes = {
            "tokens": self.max_seq * 4, "state": 16,
            "h": C * H * 2, "n": C * H * 2, "qkv": C * cfg.qkv_dim * 2, "qt": C * cfg.q_dim * 2,
            "attn": C * cfg.q_dim * 2, "act": C * cfg.dense_intermediate_size * 2,
            "scratch": 2 * cfg.num_kv_heads * (W + C) * cfg.head_dim * 2,
            "ids": C * K * 4, "gates": C * K * 4, "counts": cfg.num_experts * 4,
            "starts": (cfg.num_experts + 1) * 4, "tiles": (cfg.num_experts + 1) * 4, "cursors": cfg.num_experts * 4,
            "slots": C * K * 4, "xp": C * K * H * 2, "actp": C * K * cfg.expert_intermediate_size * 2,
            "yp": C * K * H * 2, "acts": C * cfg.shared_intermediate_size * 2, "shared": C * H * 2,
            "args": PARAM_BYTES,   # constant parameter blocks of the prefill programs
        }
        for name, nbytes in sizes.items():
            self.buf[name] = self.plan.buffer(nbytes)
        self.plan.write(self.buf["counts"], 0, np.zeros(cfg.num_experts, dtype=np.uint32))
        self._param_blob = bytearray()
        self.logbase = float(np.log2(cfg.rope_theta))
        self.const_zero = self.param("uint", value=0)
        self.const_one = self.param("uint", value=1)
        self.scan_param = self.param("scan", tile_rows=GROUPED_TM)

    def param(self, kind: str, **fields) -> int:
        """Append one constant parameter block (256-byte aligned) and return its byte offset."""
        uints, ulongs, floats = _PARAM_LAYOUTS[kind]
        if kind in ("gemm", "embed", "qmv") and not np.isscalar(fields.get("w", 0)):
            w = fields.pop("w")
            fields.update(w=int(w[0]), s=int(w[1]), b=int(w[2]))
            if "w2" in fields:
                w2 = fields.pop("w2")
                fields.update(w2=int(w2[0]), s2=int(w2[1]), b2=int(w2[2]))
        blob = np.array([fields.get(k, 0) for k in uints], dtype=np.uint32).tobytes()
        if ulongs:
            blob += b"\0" * ((-len(blob)) % 8)
            blob += np.array([fields.get(k, 0) for k in ulongs], dtype=np.uint64).tobytes()
        blob += np.array([fields.get(k, 0.0) for k in floats], dtype=np.float32).tobytes()
        self._param_blob += b"\0" * ((-len(self._param_blob)) % 256)
        offset = len(self._param_blob)
        self._param_blob += blob
        if len(self._param_blob) > PARAM_BYTES:
            raise RuntimeError("prefill parameter blocks exceed the parameter buffer")
        return offset

    def prefill(self, tokens: Sequence[int], start: int = 0, program: str = "greedy", top_k: int = 0,
                top_p: float = 1.0, temperature: float = 1.0, seed: Optional[int] = None) -> int:
        """Populate the KV cache for prompt positions [start, n) with the chunked prefill program, compute the
        last position's logits (final norm + LM head) and return the first generated token (greedy, or the
        sampled program's top-k/top-p rule with Gumbel noise from ``seed`` at position n - 1; ``seed=None``
        uses the noise row written by ``set_noise``)."""
        n = len(tokens)
        if not 0 <= start < n:
            raise ValueError(f"prefill start {start} outside [0, {n})")
        if n > self.max_seq:
            raise ValueError(f"prompt length {n} exceeds max_seq {self.max_seq}")
        self.plan.write(self.buf["tokens"], 0, np.asarray(tokens, dtype=np.uint32))
        if start == 0:
            self.initial = 0
            self.swa_kv_start = 0
        pos = start
        timings = []
        # Every chunk leaves the sliding rings laid out linearly behind its end; the last chunk's layout is
        # that of an n-token prefill (initial = n), the decode step's slot rule.
        while pos < n:
            rows = min(self.prefill_chunk, n - pos)
            end = pos + rows
            self._state[:4] = (pos, rows, self.initial, end)
            timings.append(self.plan.run(self.programs["prefill_small" if rows < SMALL_ROWS else "prefill"]))
            self.swa_kv_start = max(self.swa_kv_start, end - self.cfg.sliding_window)
            self.initial = end
            pos = end
        timings.append(self.plan.run(self.programs["prefill_head"]))
        self.last_prefill_timings = timings
        self.pos = n
        if program == "sampled" and seed is not None:
            self._params[:] = self._decode_params(0, n - 1, 1, (), top_k, top_p, temperature, seed)
            self.plan.run(self.programs["noise"])
        return self.select(program, top_k=top_k, top_p=top_p, temperature=temperature)

    def select(self, program: str = "greedy", top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0) -> int:
        """Token from the logits in decode slot 0 by the decode programs' rule: greedy argmax (lowest id on
        ties), or top-k by logit (ties: larger id first), nucleus cut on the cumulative probability, then the
        maximum of log-probability / temperature + noise (ties: lowest id)."""
        x = bf16_to_f32(self.logits(0)).astype(np.float64)
        if program != "sampled":
            return int(np.argmax(x))
        k = max(1, min(int(top_k), 32))
        order = np.lexsort((-np.arange(x.size), -x))[:k]
        m = x.max()
        lse = m + np.log(np.exp(x - m).sum())
        lp = x[order] - lse
        before = np.concatenate(([0.0], np.cumsum(np.exp(lp))[:-1]))
        kept = np.ones(k, bool) if top_p >= 1.0 else (1.0 - before) > (1.0 - top_p)
        noise = self.plan.read(self.buf["noise"], 0, np.float32, x.size)
        inv_t = 1.0 / temperature if temperature else 1.0
        score = np.where(kept, lp * inv_t + noise[order], -np.inf)
        best = score.max()
        return int(order[(score == best) & kept].min())

    def prefill_stepwise(self, tokens: Sequence[int], start: int = 0, program: str = "greedy", **sampling) -> int:
        """Reference prefill: every prompt position through the decode program (one per dispatch)."""
        n = len(tokens)
        if start == 0:
            self.initial = 0
            self.swa_kv_start = 0
        for i in range(start, n - 1):
            self.decode(tokens[i], i, 1, program="greedy")
        return self.decode(tokens[n - 1], n - 1, 1, program=program, **sampling)[0]

    def sliding_state(self) -> tuple:
        """Copy of the sliding-window KV rings with their layout, for ``restore_sliding_state``. Global-layer
        KV needs no copy to restart decoding at ``pos``: decode rewrites every global slot it attends to
        beyond that position."""
        rings = [self.plan.read(self.buf["kv"], 2 * int(d[D_KV]), np.uint16, self._ring_elements)
                 for d, sliding in zip(self.layer_table, self.cfg.sliding_layers) if sliding]
        return rings, self.initial, self.swa_kv_start, self.pos

    def restore_sliding_state(self, state: tuple) -> None:
        """Return to the state captured by ``sliding_state`` (e.g. to time several decodes from one prompt)."""
        rings, self.initial, self.swa_kv_start, self.pos = state
        layers = [d for d, sliding in zip(self.layer_table, self.cfg.sliding_layers) if sliding]
        for d, ring in zip(layers, rings):
            self.plan.write(self.buf["kv"], 2 * int(d[D_KV]), ring)

    @property
    def _ring_elements(self) -> int:
        return 2 * self.cfg.num_kv_heads * self.cfg.sliding_window * self.cfg.head_dim

    # ------------------------------------------------------------------- decode
    def _decode_params(self, token, position, steps, eos, top_k, top_p, temperature, seed) -> np.ndarray:
        p = np.zeros(16, dtype=np.uint32)
        p[0], p[1], p[2], p[3] = token, position, steps, len(eos)
        p[4:4 + len(eos)] = list(eos)
        p[8] = top_k
        p[9:11] = np.array([top_p, 1.0 / temperature if temperature else 1.0], dtype=np.float32).view(np.uint32)
        p[11] = self.initial
        if seed is not None:
            p[12], p[13], p[14] = int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 1
        return p

    def decode(self, token: int, position: int, steps: int, program: str = "greedy",
               top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0,
               eos: Sequence[int] = (), seed: Optional[int] = None) -> List[int]:
        """Run up to ``steps`` (<= 8) decode steps from ``token`` at ``position``; returns the produced
        tokens (stops after an ``eos`` token, which is included). The sampled program draws Gumbel noise on
        the GPU from ``seed`` and the positions (``seed=None``: rows written by ``set_noise``)."""
        if not 1 <= steps <= BLOCK:
            raise ValueError(f"steps must be in [1, {BLOCK}]")
        if position + steps > self.max_seq:
            raise ValueError(f"decode range [{position}, {position + steps}) exceeds max_seq {self.max_seq}")
        if len(eos) > 4:
            raise ValueError("at most four stop tokens")
        self._params[:] = self._decode_params(token, position, steps, eos, top_k, top_p, temperature, seed)
        self._control[:] = 0
        self._diagnostic[:] = 0
        timing = self.plan.run(self.programs[program])
        if self._diagnostic.any():
            raise RuntimeError(f"resident decode barrier timed out (diagnostic {self._diagnostic.tolist()}); "
                               "the GPU did not keep all threadgroups resident")
        produced = int(self._packet[16])
        pairs = self._packet[:2 * produced].tolist()
        if produced < 1 or pairs[0] != token or pairs[2::2] != pairs[1:-1:2]:
            raise RuntimeError(f"invalid decode packet {pairs}")
        self.last_timing = timing
        self.pos = position + produced
        self.swa_kv_start = max(self.swa_kv_start, self.pos - self.cfg.sliding_window)
        return pairs[1::2]

    def logits(self, step: int = 0) -> np.ndarray:
        """BF16 logits (raw uint16) of ``step`` in the last decode block."""
        v = self.cfg.vocab_size
        return self.plan.read(self.buf["logits"], step * v * 2, np.uint16, v)

    def trace(self, step: int = 0) -> np.ndarray:
        """Residual stream after the embedding and after every layer of ``step`` in the last block of the
        ``reference`` program (raw BF16 [layers + 1, hidden]); the other programs do not write it."""
        rows = self.cfg.num_layers + 1
        n = rows * self.cfg.hidden_size
        return self.plan.read(self.buf["trace"], step * n * 2, np.uint16, n).reshape(rows, -1)

    def set_noise(self, noise: np.ndarray):
        """Gumbel noise for the sampled program: float32 [steps, vocab]."""
        self.plan.write(self.buf["noise"], 0, np.ascontiguousarray(noise, dtype=np.float32))
