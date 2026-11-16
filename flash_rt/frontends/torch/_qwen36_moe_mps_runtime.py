"""qwen3_5_moe (Qwen3.6-35B-A3B) runtime on Apple Silicon: weight upload, buffers and the native Metal programs.

Weights stay in the checkpoint's MLX affine form (4-bit codes with BF16 scale/bias per 64 inputs; 8-bit for
the router and the shared-expert gate; BF16 norms, conv and DeltaNet gate parameters): the bytes are copied
verbatim into one shared Metal buffer and addressed through a per-layer descriptor table of 64-bit offsets.
Model state: BF16 KV cache of the ten attention layers (slot = position), FP32 recurrent state of the thirty
DeltaNet layers ([32, 128, 128] each) and their last conv input frames (BF16, a four-slot ring, slot = position
mod 4). The steady-state decode step is one resident dispatch that runs up to eight tokens; the host writes the
input token and position, runs the program and reads the produced tokens back.

Numerics are those of mlx-lm on the same checkpoint for one-token steps: FP32 accumulation, BF16 activations,
residual stream and KV, BF16 logits.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from flash_rt.hardware.apple_native.native import NativePlan
from flash_rt.models.flint.mlx_checkpoint import AffineMatrix, bf16_to_f32
from flash_rt.models.qwen36_moe.mlx_checkpoint import (
    Qwen35MoeConfig, Qwen35MoeConfigError, Qwen35MoeMLXCheckpoint)
from flash_rt.models.qwen36_moe.pipeline_mps import (
    BLOCK, D_ALOG, D_AUX, D_CONV, D_DOWN, D_DT, D_EMBED, D_FINAL_NORM, D_FLAGS, D_GATE, D_GDN_NORM, D_HEAD,
    D_IN_NORM, D_K_NORM, D_M0, D_M1, D_M2, D_M3, D_OUT, D_POST_NORM, D_Q_NORM, D_ROUTER, D_SGATE, D_SH_DOWN,
    D_SH_GATE, D_SH_UP, D_STATE, D_UP, DECODE_KERNELS, DESCRIPTOR, PARTIALS, RING_ELEMENTS, STATE_ELEMENTS,
    STATISTICS, WORKERS, WORKSPACE, Qwen36MoePipelineMps, check_geometry)

ALIGN = 256
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


class Qwen36MoeRuntime(Qwen36MoePipelineMps):
    """Owns the Metal plan, every buffer and the decode programs (batch 1)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 32768, library: Optional[str] = None):
        if max_seq < 1:
            raise ValueError("max_seq must be positive")
        self.ckpt = Qwen35MoeMLXCheckpoint(checkpoint_dir)
        self.cfg: Qwen35MoeConfig = self.ckpt.config
        check_geometry(self.cfg)
        if max_seq > self.cfg.context_length:
            raise ValueError(f"max_seq {max_seq} exceeds the checkpoint context {self.cfg.context_length}")
        self.max_seq = int(max_seq)
        self.plan = NativePlan(library)
        self.buf: Dict[str, int] = {}
        started = time.perf_counter()
        self._upload_weights()
        self.ckpt.close()   # weights now live in the Metal buffer; release the shard mappings
        self._allocate_state()
        self.vocab_param = 0
        self.plan.write(self.buf["args"], 0, np.array([self.cfg.vocab_size], dtype=np.uint32))
        self.programs = {name: self.record_decode(kernel) for name, kernel in DECODE_KERNELS.items()}
        self.programs["noise"] = self.record_noise()
        self.plan.finish(resident=True)
        self._params = self.plan.view(self.buf["params"], np.uint32, 16)
        self._packet = self.plan.view(self.buf["packet"], np.uint32, 17)
        self._diagnostic = self.plan.view(self.buf["diagnostic"], np.uint32, WORKERS)
        self._control = self.plan.view(self.buf["control"], np.uint32, CONTROL)
        self.load_seconds = time.perf_counter() - started
        self.pos = 0
        self.reset()

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
            raise Qwen35MoeConfigError(f"{name}: {m.scales.shape[:-1]} output rows, expected {shape}")
        return m

    def _bf16(self, name: str, shape) -> np.ndarray:
        t = self.ckpt.tensor(name)
        if t.dtype != np.uint16 or t.shape != tuple(shape):
            raise Qwen35MoeConfigError(f"{name}: {t.dtype} {t.shape}, expected BF16 {tuple(shape)}")
        return t

    def _upload_weights(self):
        cfg = self.cfg
        H, E, I, S = cfg.hidden_size, cfg.num_experts, cfg.expert_intermediate_size, cfg.shared_intermediate_size
        layout = _Layout()
        table = np.zeros((cfg.num_layers + 1, DESCRIPTOR), dtype=np.uint64)
        kv_offset = state_offset = ring_offset = 0
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            d = table[i]
            full = cfg.is_full_attention(i)
            d[D_FLAGS] = 1 if full else 0
            d[D_IN_NORM] = layout.add(self._bf16(p + "input_layernorm.weight", (H,)))
            d[D_POST_NORM] = layout.add(self._bf16(p + "post_attention_layernorm.weight", (H,)))
            if full:
                a = p + "self_attn."
                d[D_Q_NORM] = layout.add(self._bf16(a + "q_norm.weight", (cfg.head_dim,)))
                d[D_K_NORM] = layout.add(self._bf16(a + "k_norm.weight", (cfg.head_dim,)))
                for base, name, rows, k in ((D_M0, "q_proj", 2 * cfg.q_dim, H), (D_M1, "k_proj", cfg.kv_dim, H),
                                            (D_M2, "v_proj", cfg.kv_dim, H), (D_OUT, "o_proj", H, cfg.q_dim)):
                    d[base:base + 4] = self._matrix(layout, self._affine(a + name, rows, k), rows)
                d[D_STATE], d[D_AUX] = kv_offset, self.max_seq
                kv_offset += 2 * cfg.num_kv_heads * self.max_seq * cfg.head_dim
            else:
                a = p + "linear_attn."
                V = cfg.linear_value_heads
                d[D_GDN_NORM] = layout.add(self._bf16(a + "norm.weight", (cfg.linear_value_dim,)))
                d[D_CONV] = layout.add(self._bf16(a + "conv1d.weight", (cfg.conv_dim, cfg.conv_kernel, 1)))
                d[D_ALOG] = layout.add(self._bf16(a + "A_log", (V,)))
                d[D_DT] = layout.add(self._bf16(a + "dt_bias", (V,)))
                for base, name, rows, k in ((D_M0, "in_proj_qkv", cfg.conv_dim, H), (D_M1, "in_proj_z", cfg.value_dim, H),
                                            (D_M2, "in_proj_b", V, H), (D_M3, "in_proj_a", V, H),
                                            (D_OUT, "out_proj", H, cfg.value_dim)):
                    d[base:base + 4] = self._matrix(layout, self._affine(a + name, rows, k), rows)
                d[D_STATE], d[D_AUX] = state_offset, ring_offset
                state_offset += STATE_ELEMENTS
                ring_offset += RING_ELEMENTS
            m = p + "mlp."
            d[D_ROUTER:D_ROUTER + 4] = self._matrix(layout, self._affine(m + "gate", E, H, bits=8), E)
            d[D_SGATE:D_SGATE + 4] = self._matrix(layout, self._affine(m + "shared_expert_gate", 1, H, bits=8), 1)
            for base, name, rows, k in ((D_GATE, "gate_proj", I, H), (D_UP, "up_proj", I, H), (D_DOWN, "down_proj", H, I)):
                d[base:base + 4] = self._matrix(layout, self._affine(m + "switch_mlp." + name, rows, k, experts=E), rows)
            for base, name, rows, k in ((D_SH_GATE, "gate_proj", S, H), (D_SH_UP, "up_proj", S, H),
                                        (D_SH_DOWN, "down_proj", H, S)):
                d[base:base + 4] = self._matrix(layout, self._affine(m + "shared_expert." + name, rows, k), rows)
        f = table[cfg.num_layers]
        f[D_FINAL_NORM] = layout.add(self._bf16("model.norm.weight", (H,)))
        f[D_EMBED:D_EMBED + 4] = self._matrix(layout, self._affine("model.embed_tokens", cfg.vocab_size, H), cfg.vocab_size)
        f[D_HEAD:D_HEAD + 4] = self._matrix(layout, self._affine("lm_head", cfg.vocab_size, H), cfg.vocab_size)

        self.weight_bytes = layout.size
        self.buf["weights"] = self.plan.buffer(layout.size)
        dst = self.plan.view(self.buf["weights"], np.uint8, layout.size)
        for offset, array in layout.items:
            flat = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
            dst[offset:offset + flat.nbytes] = flat
        self.buf["layers"] = self.plan.buffer(table.nbytes)
        self.plan.write(self.buf["layers"], 0, table)
        self.layer_table = table
        self.kv_elements, self.state_elements, self.ring_elements = kv_offset, state_offset, ring_offset

    # -------------------------------------------------------------------- state
    def _allocate_state(self):
        v = self.cfg.vocab_size
        sizes = {
            "kv": self.kv_elements * 2,
            "states": self.state_elements * 4,
            "rings": self.ring_elements * 2,
            "params": 64,
            "args": 256,
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

    def reset(self) -> None:
        """Empty context: zero DeltaNet states and conv frames (the KV cache needs no clearing)."""
        self.plan.view(self.buf["states"], np.uint8, self.state_elements * 4)[:] = 0
        self.plan.view(self.buf["rings"], np.uint8, self.ring_elements * 2)[:] = 0
        self.pos = 0

    # ------------------------------------------------------------------- decode
    def _decode_params(self, token, position, steps, eos, top_k, top_p, temperature, seed) -> np.ndarray:
        p = np.zeros(16, dtype=np.uint32)
        p[0], p[1], p[2], p[3] = token, position, steps, len(eos)
        p[4:4 + len(eos)] = list(eos)
        p[8] = top_k
        p[9:11] = np.array([top_p, 1.0 / temperature if temperature else 1.0], dtype=np.float32).view(np.uint32)
        p[11] = getattr(self, "debug_layer", 0)
        if seed is not None:
            p[12], p[13], p[14] = int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 1
        return p

    def decode(self, token: int, position: int, steps: int, program: str = "greedy",
               top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0,
               eos: Sequence[int] = (), seed: Optional[int] = None) -> List[int]:
        """Run up to ``steps`` (<= 8) decode steps from ``token`` at ``position``; returns the produced tokens
        (stops after an ``eos`` token, which is included). The recurrent state must be that of the positions
        before ``position``. The sampled program draws Gumbel noise on the GPU from ``seed`` and the positions
        (``seed=None``: rows written by ``set_noise``)."""
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
        return pairs[1::2]

    def prefill_stepwise(self, tokens: Sequence[int], start: int = 0, program: str = "greedy", **sampling) -> int:
        """Reference prefill: every prompt position through the decode program (one per dispatch)."""
        n = len(tokens)
        if start == 0:
            self.reset()
        for i in range(start, n - 1):
            self.decode(tokens[i], i, 1, program="greedy")
        return self.decode(tokens[n - 1], n - 1, 1, program=program, **sampling)[0]

    def logits(self, step: int = 0) -> np.ndarray:
        """BF16 logits (raw uint16) of ``step`` in the last decode block."""
        v = self.cfg.vocab_size
        return self.plan.read(self.buf["logits"], step * v * 2, np.uint16, v)

    def trace(self, step: int = 0) -> np.ndarray:
        """Residual stream after the embedding and after every layer of ``step`` in the last block of the
        ``trace`` program (raw BF16 [layers + 1, hidden]); the other programs do not write it."""
        rows = self.cfg.num_layers + 1
        n = rows * self.cfg.hidden_size
        return self.plan.read(self.buf["trace"], step * n * 2, np.uint16, n).reshape(rows, -1)

    def set_noise(self, noise: np.ndarray):
        """Gumbel noise for the sampled program: float32 [steps, vocab]."""
        self.plan.write(self.buf["noise"], 0, np.ascontiguousarray(noise, dtype=np.float32))

    def close(self):
        self.plan.close()
