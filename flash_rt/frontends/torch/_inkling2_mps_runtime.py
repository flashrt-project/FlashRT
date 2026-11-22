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
from flash_rt.frontends.torch._inkling2_mtp import MTP_BLOCK, _MTPMixin

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


class Inkling2Runtime(_MTPMixin):
    """Owns the Metal plan, the buffers and the decode programs (batch 1)."""

    def __init__(self, checkpoint_dir: str, max_seq: int = 8192, library: Optional[str] = None,
                 sequences: int = 1, row_capacity: int = 1024, row_shapes=None, trace_rows: bool = False,
                 mtp_batches: Sequence[int] = (), mtp_prefill: Sequence[int] = ()):
        """``sequences``: KV / conv-state slots (batched decode); ``row_capacity``: rows of a prefill chunk or a
        batched decode step. ``mtp_batches`` / ``mtp_prefill``: batch sizes and prompt lengths of DSpark
        speculative decoding programs (mtp.safetensors; see _inkling2_mtp)."""
        if max_seq < 1 or sequences < 1 or row_capacity < 1:
            raise ValueError("max_seq, sequences and row_capacity must be positive")
        self.mtp_batches, self.mtp_prefill = [int(c) for c in mtp_batches], [int(p) for p in mtp_prefill]
        self.mtp = bool(self.mtp_batches)
        if self.mtp:
            row_capacity = max(int(row_capacity), MTP_BLOCK * max(self.mtp_batches), *self.mtp_prefill)
            if max(self.mtp_batches) > sequences:
                raise ValueError("MTP batch sizes must not exceed sequences")
        self.sequences, self.row_capacity = int(sequences), int(row_capacity)
        self.trace_rows = bool(trace_rows)
        self.row_shapes = None if row_shapes is None else [tuple(x) for x in row_shapes]
        if self.row_shapes and any(r > self.row_capacity or not 1 <= lr <= r for r, lr in self.row_shapes):
            raise ValueError("row shapes must satisfy 1 <= logit rows <= rows <= row_capacity")
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
        if self.mtp:
            self._mtp_file.close()
        self._allocate_state()
        self._allocate_rows()
        if self.mtp:
            self._mtp_allocate()
        self.programs = {name: self._record_decode(kernel) for name, kernel in DECODE_KERNELS.items()}
        self.row_programs: Dict[tuple, int] = {}
        for rows, logit_rows in self._row_shapes():
            self.row_programs[(rows, logit_rows)] = self._record_rows(rows, logit_rows)
        if self.mtp:
            self._mtp_record()
        self.plan.write(self.buf["args"], 0, np.frombuffer(bytes(self._param_blob), dtype=np.uint8))
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
        if self.mtp:
            self._mtp_upload(layout)
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
            "kv": self.kv_elements * 2 * self.sequences, "rings": self.ring_elements * 2 * self.sequences,
            "params": 64,
            "noise": BLOCK * v * 4, "logits": BLOCK * v * 4, "trace": BLOCK * visits * H * 2,
            "packet": 17 * 4, "workspace": WORKSPACE, "debug": DEBUG_FLOATS * 4,
            "control": CONTROL * 4, "diagnostic": WORKERS * 4,
        }
        for name, nbytes in sizes.items():
            self.buf[name] = self.plan.buffer(nbytes)

    # ---------------------------------------------------------------- row programs
    def _row_shapes(self):
        """(rows, logit rows) programs recorded at load: prefill chunks (logits of the last row) and batched decode
        steps (logits of every row)."""
        shapes = getattr(self, "row_shapes", None)
        return shapes if shapes is not None else [(self.row_capacity, 1)]

    def _allocate_rows(self):
        C, H = self.row_capacity, self.cfg.hidden_size
        f32, bf = 4, 2
        sizes = {
            "r_tokens": C * 4, "r_info": C * 16, "r_h": C * H * bf, "r_prelude": C * H * bf, "r_bs": 6 * C * H * f32,
            "r_bank": 2 * C * H * f32, "r_x": 3 * C * H * bf, "r_ain": 4 * C * H * f32, "r_s": 4 * C * 2 * H * bf,
            "r_msg": C * 4 * H * f32, "r_mdown": 4 * C * 64 * bf, "r_forget": 4 * C * H * f32, "r_gain": 4 * C * H * f32,
            "r_gated": C * H * f32, "r_proj": C * 8672 * bf, "r_qc": C * 3072 * bf, "r_rclip": C * 384 * bf,
            "r_bias": C * 24 * 1024 * bf, "r_chunk": 2 * 8 * C * 128 * bf, "r_attn": C * 3072 * bf, "r_o": C * H * bf,
            "r_resid": C * H * f32, "r_y": C * H * bf, "r_rlog": C * 128 * f32, "r_ids": C * 8 * 4, "r_rw": C * 8 * f32,
            "r_slot": C * 8 * 4, "r_counts": 128 * 4, "r_starts": 129 * 4, "r_cursors": 128 * 4, "r_tiles": 129 * 4,
            "r_xp": C * 8 * H * bf, "r_act": C * 8 * 1152 * bf, "r_ep": C * 8 * H * bf, "r_moe": C * H * bf,
            "r_xf": C * H * bf, "r_logits": max([lr for _, lr in self._row_shapes()] +
                                                [MTP_BLOCK * c for c in self.mtp_batches]) * self.cfg.vocab_size * f32,
            "r_out": C * 4,
            "args": 4 << 20, "r_trace": (48 * C * H * bf if self.trace_rows else 4),
        }
        for name, nbytes in sizes.items():
            self.buf[name] = self.plan.buffer(nbytes)
        self.plan.view(self.buf["r_counts"], np.uint32, 128)[:] = 0
        self._param_blob = bytearray()

    def _param(self, blob: bytes) -> int:
        self._param_blob += b"\0" * ((-len(self._param_blob)) % 256)
        offset = len(self._param_blob)
        self._param_blob += blob
        return offset

    def _visit_param(self, v: int, rows: int, nblocks: int, slots, bslot: int, pad: int = 0) -> int:
        cfg = self.cfg
        layer, pss = cfg.visits()[v]
        row = self.table[cfg.num_layers + 1 + v]
        d = self.table[layer]
        flags = (1 if cfg.is_global(layer) else 0) | (2 if cfg.is_looped(layer) and pss > 0 else 0) | \
                (4 if int(d[D_FLAGS]) & 4 else 0) | (8 if layer % cfg.layers_per_block == 0 else 0)
        sl = list(slots) + [0] * (4 - len(slots))
        blob = np.array([row[V_KV], self.kv_elements, row[V_RING], self.ring_elements], np.uint64).tobytes()
        blob += np.array([layer, pss, row[V_CAP], rows, nblocks, bslot, *sl[:4], flags, pad], np.uint32).tobytes()
        return self._param(blob)

    # TensorOps tile (rows, columns, SIMD groups) by row count: few rows want narrow column tiles (more
    # threadgroups to stream the weights), many rows one row tile (weights read once).
    gemm_tiles = {4: (8, 32, 1), 16: (16, 32, 2), 64: (16, 32, 2)}
    # Row-count paths. Up to `gemv_rows` rows: multi-row gemv / per-expert qmv, each row's arithmetic the decode
    # step's (results independent of the batch). Up to `hybrid_limit` rows: experts with at most `hybrid_rows`
    # rows take the qmv kernel, the others TensorOps tiles (a tile pays its weights for up to tile_m rows).
    gemv_rows = 4
    hybrid_rows = 3
    hybrid_limit = 1 << 30
    # expert tiles by row count (8 assignments per row over 128 experts: R / 16 rows per expert on average)
    expert_tiles = {4: (8, 64, 4), 64: (8, 32, 2), 256: (16, 64, 4), 1 << 30: (32, 64, 4)}

    @staticmethod
    def _pick(table, rows):
        for limit in sorted(table):
            if rows <= limit:
                return table[limit]
        return (64, 64, 4)

    def _gemm(self, kernel, x, x_off, x_stride, w, n, k, y, y_off, y_stride, y_col, rows, small, rowsp=None):
        tm, tn, sg = self._pick(self.gemm_tiles, rows)
        blob = np.array([rows, n, k, x_stride, y_stride, y_col, 0, 0], np.uint32).tobytes() + \
            np.array([w], np.uint64).tobytes()
        prm = self._param(blob)
        name = f"{kernel}_t{tm}x{tn}s{sg}"
        self.plan.dispatch(name, [(x, x_off, READ), (self.buf["weights"], 0, READ), (y, y_off, READ_WRITE),
                                  (self.buf["args"], prm, READ)],
                           ((n + tn - 1) // tn, (rows + tm - 1) // tm, 1), sg * 32, label=name)

    def _gemm_multi(self, kernel, rows, mats, xs, ys, label):
        """Independent TensorOps products in one dispatch (ink_gemm_*_multi): mats = [(weight offset, n, k, x index,
        x stride, y index, y column, y stride)]; xs = up to three (buffer, offset) inputs, ys = up to two outputs."""
        tm, tn, sg = self._pick(self.gemm_tiles, rows)
        blob = np.array([len(mats), rows, 0, 0], np.uint32).tobytes()
        tiles = 0
        for w, n, k, xsel, xstride, ysel, ycol, ystride in mats:
            blob += np.array([w], np.uint64).tobytes() + \
                np.array([n, k, xsel, xstride, ysel, ycol, ystride, tiles, 0, 0], np.uint32).tobytes()
            tiles += (n + tn - 1) // tn
        blob += b"\0" * (48 * (8 - len(mats)))
        prm = self._param(blob)
        xs = list(xs) + [xs[0]] * (3 - len(xs))
        ys = list(ys) + [ys[0]] * (2 - len(ys))
        name = f"{kernel}_t{tm}x{tn}s{sg}"
        self.plan.dispatch(name, [(xs[0][0], xs[0][1], READ), (xs[1][0], xs[1][1], READ), (xs[2][0], xs[2][1], READ),
                                  (self.buf["weights"], 0, READ), (ys[0][0], ys[0][1], READ_WRITE),
                                  (ys[1][0], ys[1][1], READ_WRITE), (self.buf["args"], prm, READ)],
                           (tiles, (rows + tm - 1) // tm, 1), sg * 32, label=label)

    def _gemv_rows(self, R, mats, xs, y, yf=None, label="gemv_rows"):
        """Multi-row BF16 products (ink_gemv_rows): mats = [(weight offset, rows, k, x index, y column, y stride,
        mode)]; xs = up to three (buffer, offset) inputs."""
        blob = np.array([len(mats), R, 0, 0], np.uint32).tobytes()
        units = 0
        for w, rows, k, xsel, ycol, ystride, mode in mats:
            blob += np.array([w], np.uint64).tobytes() + np.array([rows, k, xsel, ycol, ystride, mode], np.uint32).tobytes()
            units += rows // 4 if mode == 0 else (rows // 32 if mode == 4 else rows)
        blob += b"\0" * (32 * (10 - len(mats)))
        prm = self._param(blob)
        xs = list(xs) + [xs[0]] * (3 - len(xs))
        yf = yf if yf is not None else y
        self.plan.dispatch("ink_gemv_rows", [(self.buf["weights"], 0, READ), (xs[0][0], xs[0][1], READ),
                           (xs[1][0], xs[1][1], READ), (xs[2][0], xs[2][1], READ), (y[0], y[1], READ_WRITE),
                           (yf[0], yf[1], READ_WRITE), (self.buf["args"], prm, READ)],
                           ((units + 7) // 8, 1, 1), 256, label=label)

    def _record_rows(self, R: int, logit_rows: int, stage: bool = False, features: bool = False) -> int:
        """Program for R rows (rows_info and tokens written per run): all 48 visits, then the logits of the last
        ``logit_rows`` rows (softcapped FP32) and their greedy tokens. ``stage``: conv frames and sliding K / V
        staged for ink_commit_rows (speculative verification); ``features``: the drafter features of every row
        (m_feat: the hidden state after visits 4, 22, 40 and the final normalized state)."""
        cfg, b, plan = self.cfg, self.buf, self.plan
        H, I, E = cfg.hidden_size, cfg.expert_intermediate_size, cfg.num_experts
        small = R <= 64
        tm, etn, esg = self._pick(self.expert_tiles, R * 8 // 8)
        gemv = R <= self.gemv_rows           # multi-row gemv products and per-expert qmv
        hybrid = not gemv and R <= self.hybrid_limit and self.hybrid_rows > 0
        program = plan.program()
        A = self.buf["args"]
        rv0 = self._visit_param(0, R, 0, [], 0)
        plan.dispatch("ink_embed_rows", [(b["weights"], 0, READ), (b["table"], 0, READ), (b["r_tokens"], 0, READ),
                                         (b["r_h"], 0, WRITE), (A, rv0, READ)], (R, 1, 1), 256, label="embed")
        nblocks, slots = 0, []
        mode0 = self._param(np.array([0], np.uint32).tobytes())
        mode1 = self._param(np.array([1], np.uint32).tobytes())
        tmp = self._param(np.array([tm, self.hybrid_rows if hybrid else 0], np.uint32).tobytes())
        for v, (layer, pss) in enumerate(cfg.visits()):
            if v >= getattr(self, "max_visits", 48):
                break
            d = self.table[layer]
            full, msgs = cfg.is_global(layer), cfg.is_looped(layer) and pss > 0
            boundary = layer % cfg.layers_per_block == 0
            if layer == cfg.loop_start:
                nblocks, slots = 1, [0]
            bslot = 0 if layer == 0 else (1 if layer == 6 else (2 + pss if layer == 12 else 5))
            rv = self._visit_param(v, R, nblocks, slots, bslot)
            vis = (A, rv, READ)
            W = (b["weights"], 0, READ)
            T = (b["table"], 0, READ)
            plan.dispatch("ink_attn_inputs_rows", [W, T, (b["r_h"], 0, READ_WRITE), (b["r_bs"], 0, READ_WRITE),
                          (b["r_x"], 0, WRITE), (b["r_ain"], 0, WRITE), (b["r_s"], 0, WRITE), (b["r_msg"], 0, READ_WRITE),
                          (b["r_bank"], 0, READ), (b["r_prelude"], 0, READ_WRITE), vis], (R, 1, 1), 256, label="attn_in")
            if msgs:
                if gemv:
                    self._gemv_rows(R, [(int(d[D_MG + 3 * g]), 64, 2 * H, g, g * R * 64, 64, 1) for g in range(3)],
                                    [(b["r_s"], g * R * 2 * H * 2) for g in range(3)], (b["r_mdown"], 0), label="msg_down")
                else:
                    self._gemm_multi("ink_gemm_bf16_multi", R,
                                     [(int(d[D_MG + 3 * g]), 64, 2 * H, g, 2 * H, 0, g * R * 64, 64) for g in range(3)],
                                     [(b["r_s"], g * R * 2 * H * 2) for g in range(3)], [(b["r_mdown"], 0)], "msg_down")
                if not gemv:
                    self._gemm_multi("ink_gemm_f32_multi", R,
                                     [(int(d[D_MG + 3 * g + which]), H, 64, g, 64, which - 1, g * R * H, H)
                                      for which in (1, 2) for g in range(3)],
                                     [(b["r_mdown"], g * R * 64 * 2) for g in range(3)],
                                     [(b["r_forget"], 0), (b["r_gain"], 0)], "msg_forget_gain")
                else:
                    for which, buf in ((1, "r_forget"), (2, "r_gain")):
                        self._gemv_rows(R, [(int(d[D_MG + 3 * g + which]), H, 64, g, g * R * H, H, 4) for g in range(3)],
                                        [(b["r_mdown"], g * R * 64 * 2) for g in range(3)], (b[buf], 0), (b[buf], 0),
                                        label="msg_" + buf[2:])
                plan.dispatch("ink_gate_rows", [W, T, (b["r_ain"], 0, READ), (b["r_msg"], 0, READ),
                              (b["r_forget"], 0, READ), (b["r_gain"], 0, READ), (b["r_gated"], 0, WRITE),
                              (b["r_x"], 0, WRITE), vis, (A, mode0, READ)], (R, 1, 1), 256, label="gate_attn")
            P = b["r_proj"]
            projs = ((0, D_WQ, 3072, 0, 0), (3072, D_WK, 1024, 1, 0), (4096, D_WV, 1024, 2, 0), (5120, D_WR, 384, 0, 0),
                     (5504, D_WG, 3072, 0, 0), (8576, D_WS, 24, 0, 1)) + \
                (((8608, D_KDYN, 32, 1, 1), (8640, D_VDYN, 32, 2, 1)) if full else ())
            if gemv:
                self._gemv_rows(R, [(int(d[idx]), n, H, xj, col, 8672, mode) for col, idx, n, xj, mode in projs],
                                [(b["r_x"], j * R * H * 2) for j in range(3)], (P, 0), label="proj")
            else:
                self._gemm_multi("ink_gemm_bf16_multi", R, [(int(d[idx]), n, H, xj, H, 0, col, 8672)
                                                            for col, idx, n, xj, _ in projs],
                                 [(b["r_x"], j * R * H * 2) for j in range(3)], [(P, 0)], "proj")
            plan.dispatch("ink_heads_rows", [W, T, (P, 0, READ), (b["r_qc"], 0, WRITE), (b["r_rclip"], 0, WRITE),
                          (b["kv"], 0, READ_WRITE), (b["r_chunk"], 0, WRITE), (b["rings"], 0, READ), vis,
                          (b["r_info"], 0, READ)], (R, 1, 1), 256, label="heads")
            plan.dispatch("ink_bias_rows", [W, T, (b["r_rclip"], 0, READ), (b["r_bias"], 0, WRITE), vis],
                          (16, R, 1), 256, label="bias")
            plan.dispatch("ink_attn_decode_rows", [(P, 0, READ), (b["r_qc"], 0, READ), (b["r_bias"], 0, READ),
                          (b["kv"], 0, READ), (b["r_chunk"], 0, READ), (b["r_attn"], 0, WRITE), vis,
                          (b["r_info"], 0, READ)], (8, R, 1), 256, label="attention")
            if gemv:
                self._gemv_rows(R, [(int(d[D_WO]), H, 3072, 0, 0, H, 0)], [(b["r_attn"], 0)], (b["r_o"], 0), label="o_proj")
            else:
                self._gemm("ink_gemm_bf16_rows", b["r_attn"], 0, 3072, int(d[D_WO]), H, 3072, b["r_o"], 0, H, 0, R, small)
            plan.dispatch("ink_post_attn_rows", [W, T, (b["r_o"], 0, READ), (b["r_h"], 0, READ), (b["r_bs"], 0, READ),
                          (b["r_resid"], 0, WRITE), (b["r_y"], 0, WRITE), (b["r_ain"], 0, WRITE), (b["r_s"], 0, WRITE),
                          (b["r_msg"], 0, READ), (b["rings"], 0, READ), vis, (b["r_info"], 0, READ)],
                          (R, 1, 1), 256, label="post_attn")
            if msgs:
                if gemv:
                    self._gemv_rows(R, [(int(d[D_MG + 9]), 64, 2 * H, 0, 3 * R * 64, 64, 1)],
                                    [(b["r_s"], 3 * R * 2 * H * 2)], (b["r_mdown"], 0), label="msg_down")
                else:
                    self._gemm("ink_gemm_bf16_rows", b["r_s"], 3 * R * 2 * H * 2, 2 * H, int(d[D_MG + 9]), 64, 2 * H,
                               b["r_mdown"], 3 * R * 64 * 2, 64, 0, R, small)
                if not gemv:
                    self._gemm_multi("ink_gemm_f32_multi", R,
                                     [(int(d[D_MG + 9 + which]), H, 64, 0, 64, which - 1, 3 * R * H, H) for which in (1, 2)],
                                     [(b["r_mdown"], 3 * R * 64 * 2)], [(b["r_forget"], 0), (b["r_gain"], 0)],
                                     "msg_forget_gain")
                else:
                    for which, buf in ((1, "r_forget"), (2, "r_gain")):
                        self._gemv_rows(R, [(int(d[D_MG + 9 + which]), H, 64, 0, 3 * R * H, H, 4)],
                                        [(b["r_mdown"], 3 * R * 64 * 2)], (b[buf], 0), (b[buf], 0), label="msg_" + buf[2:])
                plan.dispatch("ink_gate_rows", [W, T, (b["r_ain"], 0, READ), (b["r_msg"], 0, READ),
                              (b["r_forget"], 0, READ), (b["r_gain"], 0, READ), (b["r_gated"], 0, WRITE),
                              (b["r_y"], 0, WRITE), vis, (A, mode1, READ)], (R, 1, 1), 256, label="gate_mlp")
            if gemv:
                self._gemv_rows(R, [(int(d[D_ROUTER]), E, H, 0, 0, E, 3)], [(b["r_y"], 0)], (b["r_rlog"], 0),
                                (b["r_rlog"], 0), label="router")
            else:
                self._gemm("ink_gemm_f32_rows", b["r_y"], 0, H, int(d[D_ROUTER]), E, H, b["r_rlog"], 0, E, 0, R, small)
            plan.dispatch("ink_route_rows", [W, T, (b["r_rlog"], 0, READ), (b["r_ids"], 0, WRITE), (b["r_rw"], 0, WRITE),
                          (b["r_counts"], 0, READ_WRITE), vis], ((R + 7) // 8, 1, 1), 256, label="route")
            plan.dispatch("ink_expert_scan", [(b["r_counts"], 0, READ_WRITE), (b["r_starts"], 0, WRITE),
                          (b["r_cursors"], 0, WRITE), (b["r_tiles"], 0, WRITE), (A, tmp, READ)], (1, 1, 1), 32,
                          label="scan")
            plan.dispatch("ink_expert_place_rows", [(b["r_ids"], 0, READ), (b["r_starts"], 0, READ),
                          (b["r_cursors"], 0, READ_WRITE), (b["r_slot"], 0, WRITE), (b["r_y"], 0, READ),
                          (b["r_xp"], 0, WRITE), vis], (R * 8, 1, 1), 256, label="place")
            mx = 1 if int(d[D_FLAGS]) & 4 else 0
            if gemv or hybrid:
                # per-expert qmv: every expert (gemv) or the experts with at most hybrid_rows rows (hybrid)
                limit = self.hybrid_rows if hybrid else 0
                for gu, w, s_, gs, k, src, dst in ((1, D_W13, D_S13, D_GS13, H, "r_xp", "r_act"),
                                                   (0, D_W2, D_S2, D_GS2, I, "r_act", "r_ep")):
                    wq = self._param(np.array([d[w], d[s_], d[gs] if not mx else 0, limit], np.uint64).tobytes())
                    sh = self._param(np.array([gu, mx, 2 * I if gu else H, k], np.uint32).tobytes())
                    plan.dispatch("ink_experts_qmv_rows", [W, (b[src], 0, READ), (b[dst], 0, WRITE), (A, wq, READ),
                                  (b["r_starts"], 0, READ), (A, sh, READ)], ((2 * I if gu else H) // 32, E, 1), 256,
                                  label="experts_gateup" if gu else "experts_down")
            tiles_y = (R * 8 + tm - 1) // tm + E
            for name, w, s_, gs, k, n_out, src, dst in (() if gemv else (
                    ("ink_experts_gateup", D_W13, D_S13, D_GS13, H, I, "r_xp", "r_act"),
                    ("ink_experts_down", D_W2, D_S2, D_GS2, I, H, "r_act", "r_ep"))):
                blob = np.array([R * 8, k, n_out, mx], np.uint32).tobytes() + \
                    np.array([d[w], d[s_], d[gs] if not mx else 0], np.uint64).tobytes()
                prm = self._param(blob)
                kn = f"{name}_t{tm}x{etn}s{esg}"
                plan.dispatch(kn, [(b[src], 0, READ), W, (b[dst], 0, WRITE), (A, prm, READ), (b["r_starts"], 0, READ),
                              (b["r_tiles"], 0, READ)], ((n_out + etn - 1) // etn, tiles_y, 1), esg * 32, label=kn)
            rv2 = self._visit_param(v, R, nblocks, slots, bslot, pad=1)
            for prm, lbl in ((rv, "combine"), (rv2, "mlp_conv")):
                plan.dispatch("ink_combine_rows", [W, T, (b["r_ep"], 0, READ), (b["r_slot"], 0, READ),
                              (b["r_rw"], 0, READ), (b["r_resid"], 0, READ), (b["r_moe"], 0, READ_WRITE),
                              (b["r_h"], 0, WRITE), (b["r_bank"], 0, WRITE), (b["rings"], 0, READ), (A, prm, READ),
                              (b["r_info"], 0, READ)], (R, 1, 1), 256, label=lbl)
            if stage:
                vix = self._param(np.array([v], np.uint32).tobytes())
                plan.dispatch("ink_ring_stage_rows", [(P, 0, READ), (b["r_o"], 0, READ), (b["r_moe"], 0, READ),
                              (b["r_chunk"], 0, READ), (b["m_stage"], 0, WRITE), vis, (A, vix, READ)], (R, 1, 1), 256,
                              label="ring_stage")
            else:
                plan.dispatch("ink_ring_rows", [(P, 0, READ), (b["r_o"], 0, READ), (b["r_moe"], 0, READ),
                              (b["r_chunk"], 0, READ), (b["rings"], 0, WRITE), (b["kv"], 0, READ_WRITE), vis,
                              (b["r_info"], 0, READ)], (R, 1, 1), 256, label="ring")
            if features and v in (4, 22, 40):     # the hidden state entering visits 5, 23, 41
                n = R * H
                cnt = self._param(np.array([n], np.uint32).tobytes())
                plan.dispatch("ink_trace_rows", [(b["r_h"], 0, READ), (b["m_feat"], (4, 22, 40).index(v) * n * 2, WRITE),
                                                 (A, cnt, READ)], ((n + 255) // 256, 1, 1), 256, label="feature")
            if boundary:
                slots = slots + [bslot]
                nblocks += 1
            if self.trace_rows:
                n = R * H
                cnt = self._param(np.array([n], np.uint32).tobytes())
                plan.dispatch("ink_trace_rows", [(b["r_h"], 0, READ), (b["r_trace"], v * n * 2, WRITE), (A, cnt, READ)],
                              ((n + 255) // 256, 1, 1), 256, label="trace")
        rvf = self._visit_param(len(cfg.visits()) - 1, R, 4, [0, 1, 4, 5], 0)
        if features:
            # every row's final state into m_feat[3]; the logit rows read it there
            first = self._param(np.array([0], np.uint32).tobytes())
            xf, xf_off, final_rows = b["m_feat"], 3 * R * H * 2, R
            lx_off = xf_off + (R - logit_rows) * H * 2
        else:
            first = self._param(np.array([R - logit_rows], np.uint32).tobytes())
            xf, xf_off, final_rows, lx_off = b["r_xf"], 0, logit_rows, 0
        plan.dispatch("ink_final_rows", [(b["weights"], 0, READ), (b["table"], 0, READ), (b["r_h"], 0, READ),
                      (b["r_bs"], 0, READ), (xf, xf_off, WRITE), (A, rvf, READ), (A, first, READ)],
                      (final_rows, 1, 1), 256, label="final")
        f = self.table[cfg.num_layers]
        self._gemm("ink_gemm_logits_rows", xf, lx_off, H, int(f[D_HEAD]), cfg.vocab_size, H, b["r_logits"], 0,
                   cfg.vocab_size, 0, logit_rows, logit_rows <= 64)
        nrows = self._param(np.array([logit_rows], np.uint32).tobytes())
        plan.dispatch("ink_argmax_rows", [(b["r_logits"], 0, READ), (b["r_out"], 0, WRITE), (A, nrows, READ)],
                      (logit_rows, 1, 1), 256, label="argmax")
        return program

    def run_rows(self, tokens: Sequence[int], info: np.ndarray, logit_rows: int) -> np.ndarray:
        """Run the row program for len(tokens) rows with rows_info ``info`` [rows, 4] (sequence, position, first,
        last); returns the greedy tokens of the last ``logit_rows`` rows."""
        R = len(tokens)
        key = (R, logit_rows)
        if key not in self.row_programs:
            raise ValueError(f"no row program recorded for {key}; pass row_shapes")
        self.plan.write(self.buf["r_tokens"], 0, np.asarray(tokens, np.uint32))
        self.plan.write(self.buf["r_info"], 0, np.ascontiguousarray(info, np.uint32))
        self.last_rows_timing = self.plan.run(self.row_programs[key])
        return self.plan.read(self.buf["r_out"], 0, np.uint32, logit_rows).copy()

    def rows_logits(self, row: int = 0) -> np.ndarray:
        return self.plan.read(self.buf["r_logits"], row * self.cfg.vocab_size * 4, np.float32,
                              self.cfg.unpadded_vocab_size)

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
