"""Inkling 2 Mini (``inkling2_loop``) runtime on Apple Silicon: weight upload, buffers and native Metal programs.

Weights stay in the released precisions: BF16 attention, gates, embedding and LM head; NVFP4 / MXFP8 routed
experts (codes and scales copied byte for byte). Derived data: each NVFP4 expert's FP32 global scale
``amax * (448 * 6 / outer_scale_denominator)`` (as mlx-lm computes it) and the combined message queries of the
looped layers. Everything lives in one shared Metal buffer addressed through a per-layer descriptor table.
Model state per layer visit (48 per token: the looped layers 6..17 run three times with their own caches): BF16
KV (a 1024-slot ring for sliding-window layers, ``max_seq`` slots for the global ones) and the last three inputs
of the four short convolutions (BF16, a four-slot ring, slot = position mod 4).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from flash_rt.hardware.apple_native.native import READ, READ_WRITE, WRITE, NativePlan
from flash_rt.models.inkling2.checkpoint import Inkling2Checkpoint, Inkling2ConfigError, check_geometry

WORKERS = 36
THREADS = 256
BLOCK = 8
DESCRIPTOR = 64
VISIT = 8
ALIGN = 256
WORKSPACE = 1177792
DEBUG_FLOATS = 24 * 4096
CONTROL = 1024
RING_FRAME = 6656
SLIDING_SLOTS = 1024

# descriptor indices (csrc/apple/inkling2/inkling2_decode_resident_bf16.h)
D_FLAGS, D_ATTN_Q, D_MLP_Q, D_IN_NORM, D_K_NORM, D_V_NORM, D_POST_NORM = range(7)
D_WQ, D_WK, D_WV, D_WR, D_WG, D_WS, D_WO = range(7, 14)
D_QCLIP, D_RCLIP, D_REL, D_KCONV, D_VCONV, D_KDYN, D_VDYN, D_ACONV, D_MCONV = range(14, 23)
D_ROUTER, D_RBIAS, D_W13, D_S13, D_GS13, D_W2, D_S2, D_GS2 = range(23, 31)
D_MG = 32
D_EMBED, D_EMBED_NORM, D_FINAL_Q, D_FINAL_NORM, D_HEAD, D_MSG_Q = range(1, 7)
V_LAYER, V_PASS, V_KV, V_CAP, V_RING = range(5)

DECODE_KERNELS = {
    "greedy": "inkling2_decode_resident_bf16",
    "reference": "inkling2_decode_resident_reference_bf16",
}


class _Layout:
    def __init__(self):
        self.size = 0
        self.items: List[tuple] = []

    def add(self, array: np.ndarray) -> int:
        self.size += (-self.size) % ALIGN
        offset = self.size
        self.items.append((offset, np.ascontiguousarray(array)))
        self.size += array.nbytes
        return offset


def global_scale(amax: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """mlx-lm: amax * ((6 * 448) / denominator), FP32 operations."""
    scale = np.float32(6.0 * 448.0) / denominator.astype(np.float32)
    return (amax.astype(np.float32) * scale).astype(np.float32)


class Inkling2Runtime:
    """Owns the Metal plan, the buffers and the decode programs (batch 1)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 8192, library: Optional[str] = None):
        if max_seq < 1:
            raise ValueError("max_seq must be positive")
        self.ckpt = Inkling2Checkpoint(checkpoint_dir)
        self.cfg = self.ckpt.config
        check_geometry(self.cfg)
        if max_seq > self.cfg.context_length:
            raise ValueError(f"max_seq {max_seq} exceeds the checkpoint context {self.cfg.context_length}")
        self.max_seq = int(max_seq)
        self.plan = NativePlan(library)
        self.buf: Dict[str, int] = {}
        started = time.perf_counter()
        self._upload_weights()
        self.ckpt.close()
        self._allocate_state()
        self.programs = {name: self._record_decode(kernel) for name, kernel in DECODE_KERNELS.items()}
        self.plan.finish(resident=True)
        self._params = self.plan.view(self.buf["params"], np.uint32, 16)
        self._packet = self.plan.view(self.buf["packet"], np.uint32, 17)
        self._diagnostic = self.plan.view(self.buf["diagnostic"], np.uint32, WORKERS)
        self._control = self.plan.view(self.buf["control"], np.uint32, CONTROL)
        self.load_seconds = time.perf_counter() - started
        self.rotation = 0
        self.debug_visit = 0xFFFFFFFF
        self.pos = 0
        self.reset()

    # ------------------------------------------------------------------ weights
    def _upload_weights(self):
        cfg, ck = self.cfg, self.ckpt
        H, E, I = cfg.hidden_size, cfg.num_experts, cfg.expert_intermediate_size
        layout = _Layout()
        table = np.zeros((cfg.num_layers + 1 + len(cfg.visits()), DESCRIPTOR), dtype=np.uint64)

        def bf16(name, shape):
            return layout.add(ck.tensor(name, "BF16", shape))

        def f32(name, shape):
            return layout.add(ck.tensor(name, "F32", shape))

        for i in range(cfg.num_layers):
            p = f"layers.{i}."
            d = table[i]
            full, looped = cfg.is_global(i), cfg.is_looped(i)
            fmt = ck.expert_format(i)
            d[D_FLAGS] = (1 if full else 0) | (2 if looped else 0) | (4 if fmt == "mxfp8" else 0)
            if i > 0:
                d[D_ATTN_Q] = f32(f"attn_res.attn_queries.{i - 1}", (3, H) if full else (H,))
            d[D_MLP_Q] = f32(f"attn_res.mlp_queries.{i}", (H,))
            if full:
                d[D_IN_NORM] = bf16(p + "q_input_norm.weight", (H,))
                d[D_K_NORM] = bf16(p + "k_input_norm.weight", (H,))
                d[D_V_NORM] = bf16(p + "v_input_norm.weight", (H,))
            else:
                d[D_IN_NORM] = bf16(p + "attn_norm.weight", (H,))
            d[D_POST_NORM] = bf16(p + "mlp_norm.weight", (H,))
            a = p + "attn."
            q_dim, kv_dim = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
            for idx, name, rows, k in ((D_WQ, "wq_du", q_dim, H), (D_WK, "wk_dv", kv_dim, H),
                                       (D_WV, "wv_dv", kv_dim, H), (D_WR, "wr_du", cfg.num_heads * cfg.d_rel, H),
                                       (D_WG, "gate_attn", q_dim, H), (D_WS, "attn_sink_proj", cfg.num_heads, H),
                                       (D_WO, "wo_ud", H, q_dim)):
                d[idx] = bf16(a + name + ".weight", (rows, k))
            d[D_QCLIP] = bf16(a + "q_l2_clip.scale", (cfg.num_heads, cfg.head_dim))
            d[D_RCLIP] = bf16(a + "rel_l2_clip.scale", (cfg.num_heads, cfg.d_rel))
            d[D_REL] = bf16(a + "rel_logits_proj.proj", (cfg.d_rel, cfg.rel_extent))
            if full:
                d[D_KCONV] = bf16(a + "k_sconv.static_w", (cfg.conv_kernel, kv_dim))
                d[D_VCONV] = bf16(a + "v_sconv.static_w", (cfg.conv_kernel, kv_dim))
                d[D_KDYN] = bf16(a + "k_sconv.weight_proj", (cfg.num_kv_heads * cfg.conv_kernel, H))
                d[D_VDYN] = bf16(a + "v_sconv.weight_proj", (cfg.num_kv_heads * cfg.conv_kernel, H))
            else:
                d[D_KCONV] = bf16(a + "k_sconv.weight", (kv_dim, 1, cfg.conv_kernel))
                d[D_VCONV] = bf16(a + "v_sconv.weight", (kv_dim, 1, cfg.conv_kernel))
            d[D_ACONV] = bf16(p + "attn_sconv.weight", (H, 1, cfg.conv_kernel))
            d[D_MCONV] = bf16(p + "mlp_sconv.weight", (H, 1, cfg.conv_kernel))
            m = p + "mlp."
            d[D_ROUTER] = bf16(m + "gate.weight", (E, H))
            d[D_RBIAS] = f32(m + "gate.bias", (E,))
            x = m + "experts."
            if fmt == "nvfp4":
                d[D_W13] = layout.add(ck.tensor(x + "w13_weight", "U8", (E, 2 * I, H // 2)))
                d[D_S13] = layout.add(ck.tensor(x + "w13_weight_scale", "F8_E4M3", (E, 2 * I, H // 16)))
                d[D_W2] = layout.add(ck.tensor(x + "w2_weight", "U8", (E, H, I // 2)))
                d[D_S2] = layout.add(ck.tensor(x + "w2_weight_scale", "F8_E4M3", (E, H, I // 16)))
                d[D_GS13] = layout.add(global_scale(ck.tensor(x + "w13_weight_amax", "F32", (E,)),
                                                    ck.tensor(x + "w13_weight_outer_scale_denominator", "F32", (1,))))
                d[D_GS2] = layout.add(global_scale(ck.tensor(x + "w2_weight_amax", "F32", (E,)),
                                                   ck.tensor(x + "w2_weight_outer_scale_denominator", "F32", (1,))))
            else:
                d[D_W13] = layout.add(ck.tensor(x + "w13_weight", "F8_E4M3", (E, 2 * I, H)))
                d[D_S13] = layout.add(ck.tensor(x + "w13_weight_scale", "U8", (E, 2 * I, H // 32)))
                d[D_W2] = layout.add(ck.tensor(x + "w2_weight", "F8_E4M3", (E, H, I)))
                d[D_S2] = layout.add(ck.tensor(x + "w2_weight_scale", "U8", (E, H, I // 32)))
            if looped:
                for g, gate in enumerate(("attn", "k", "v", "mlp")):
                    c = f"{p}controllers.{gate}."
                    d[D_MG + 3 * g] = bf16(c + "down_weight", (cfg.gate_rank, 2 * H))
                    d[D_MG + 3 * g + 1] = bf16(c + "forget_weight", (H, cfg.gate_rank))
                    d[D_MG + 3 * g + 2] = bf16(c + "gain_weight", (H, cfg.gate_rank))
        f = table[cfg.num_layers]
        f[D_EMBED] = bf16("embed.weight", (cfg.vocab_size, H))
        f[D_EMBED_NORM] = bf16("embed_norm.weight", (H,))
        f[D_FINAL_Q] = f32("attn_res.output_query", (H,))
        f[D_FINAL_NORM] = bf16("norm.weight", (H,))
        f[D_HEAD] = bf16("unembed.weight", (cfg.vocab_size, H))
        n_loop = cfg.loop_end - cfg.loop_start
        q = ck.tensor("middle_cycle_cross_pass_queries", "F32", (2 * n_loop, H)).reshape(n_loop, 2, H)
        kv = ck.tensor("middle_cycle_cross_pass_kv_queries", "F32", (2 * n_loop, H)).reshape(n_loop, 2, H)
        f[D_MSG_Q] = layout.add(np.concatenate([q[:, :1], kv, q[:, 1:]], axis=1))
        # visits: KV and ring offsets per visit
        kv_offset = ring_offset = 0
        dkv = cfg.num_kv_heads * cfg.head_dim
        for v, (layer, p) in enumerate(cfg.visits()):
            row = table[cfg.num_layers + 1 + v]
            cap = self.max_seq if cfg.is_global(layer) else SLIDING_SLOTS
            row[V_LAYER], row[V_PASS], row[V_KV], row[V_CAP], row[V_RING] = layer, p, kv_offset, cap, ring_offset
            kv_offset += 2 * dkv * cap
            ring_offset += 4 * RING_FRAME
        self.weight_bytes = layout.size
        self.buf["weights"] = self.plan.buffer(layout.size)
        dst = self.plan.view(self.buf["weights"], np.uint8, layout.size)
        for offset, array in layout.items:
            flat = array.reshape(-1).view(np.uint8)
            dst[offset:offset + flat.nbytes] = flat
        self.buf["table"] = self.plan.buffer(table.nbytes)
        self.plan.write(self.buf["table"], 0, table)
        self.table = table
        self.kv_elements, self.ring_elements = kv_offset, ring_offset

    # -------------------------------------------------------------------- state
    def _allocate_state(self):
        v, H = self.cfg.vocab_size, self.cfg.hidden_size
        visits = len(self.cfg.visits())
        sizes = {
            "kv": self.kv_elements * 2, "rings": self.ring_elements * 2, "params": 64,
            "noise": BLOCK * v * 4, "logits": BLOCK * v * 4, "trace": BLOCK * visits * H * 2,
            "packet": 17 * 4, "workspace": WORKSPACE, "debug": DEBUG_FLOATS * 4,
            "control": CONTROL * 4, "diagnostic": WORKERS * 4,
        }
        for name, nbytes in sizes.items():
            self.buf[name] = self.plan.buffer(nbytes)

    def _record_decode(self, kernel: str) -> int:
        b = self.buf
        program = self.plan.program()
        self.plan.dispatch(kernel, [
            (b["weights"], 0, READ), (b["table"], 0, READ), (b["kv"], 0, READ_WRITE), (b["rings"], 0, READ_WRITE),
            (b["params"], 0, READ), (b["noise"], 0, READ), (b["logits"], 0, WRITE), (b["trace"], 0, WRITE),
            (b["packet"], 0, WRITE), (b["workspace"], 0, READ_WRITE), (b["debug"], 0, WRITE),
            (b["control"], 0, READ_WRITE), (b["diagnostic"], 0, WRITE),
        ], (WORKERS, 1, 1), THREADS, label=kernel)
        return program

    def reset(self) -> None:
        """Empty context (the caches need no clearing: a step reads only slots it or its predecessors wrote,
        and the conv windows before position 0 read as zero)."""
        self.pos = 0
        self.rotation = 0

    # ------------------------------------------------------------------- decode
    def decode(self, token: int, position: int, steps: int, program: str = "greedy",
               eos: Sequence[int] = ()) -> List[int]:
        if not 1 <= steps <= BLOCK:
            raise ValueError(f"steps must be in [1, {BLOCK}]")
        if position + steps > self.max_seq:
            raise ValueError(f"decode range [{position}, {position + steps}) exceeds max_seq {self.max_seq}")
        if len(eos) > 4:
            raise ValueError("at most four stop tokens")
        if not 0 <= int(token) < self.cfg.unpadded_vocab_size:
            raise ValueError("token id outside the checkpoint vocabulary")
        p = np.zeros(16, dtype=np.uint32)
        p[0], p[1], p[2], p[3] = token, position, steps, len(eos)
        p[4:4 + len(eos)] = list(eos)
        p[11], p[12], p[13] = self.rotation, 0, self.debug_visit
        self._params[:] = p
        self._control[:] = 0
        self._diagnostic[:] = 0
        timing = self.plan.run(self.programs[program])
        if self._diagnostic.any():
            raise RuntimeError(f"resident decode barrier timed out (diagnostic {self._diagnostic.tolist()})")
        produced = int(self._packet[16])
        pairs = self._packet[:2 * produced].tolist()
        if produced < 1 or pairs[0] != token or pairs[2::2] != pairs[1:-1:2]:
            raise RuntimeError(f"invalid decode packet {pairs}")
        self.last_timing = timing
        self.pos = position + produced
        return pairs[1::2]

    def logits(self, step: int = 0) -> np.ndarray:
        """FP32 softcapped logits of ``step`` in the last block, [unpadded vocabulary]."""
        v = self.cfg.vocab_size
        return self.plan.read(self.buf["logits"], step * v * 4, np.float32, self.cfg.unpadded_vocab_size)

    def trace(self, step: int = 0) -> np.ndarray:
        """Hidden state after every visit (raw BF16 [48, hidden]) from the reference program."""
        n = len(self.cfg.visits()) * self.cfg.hidden_size
        return self.plan.read(self.buf["trace"], step * n * 2, np.uint16, n).reshape(-1, self.cfg.hidden_size)

    def debug(self) -> np.ndarray:
        return self.plan.read(self.buf["debug"], 0, np.float32, DEBUG_FLOATS).reshape(24, 4096)

    def close(self):
        self.plan.close()
