"""Inkling 2 Mini (``inkling2_loop``) runtime on Apple Silicon: weight upload, buffers and native Metal programs.

Weights stay in the released precisions: BF16 attention, gates, embedding and LM head; NVFP4 / MXFP8 routed
experts (codes and scales copied byte for byte). Derived data: each NVFP4 expert's FP32 global scale
``amax * (448 * 6 / outer_scale_denominator)`` (as mlx-lm computes it) and the combined message queries of the
looped layers. Everything lives in one shared Metal buffer addressed through a per-layer descriptor table. Row
programs with SIMD-group MMA experts (see ``expert_mma``) add a second copy of the NVFP4 experts, re-arranged on
the GPU at load (11.5 GB).
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
from flash_rt.frontends.torch._inkling2_dflash2 import _DFlash2Mixin
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
ARGS_BYTES = 16 << 20          # parameter blocks of every recorded program (~0.3 MB per row program)

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
    "sampled": "inkling2_decode_resident_sampled_bf16",
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


def nvfp4_mma_arrangement(codes: np.ndarray, scales: np.ndarray):
    """Reference of ink_nv_arrange for one expert matrix: codes [rows][K / 2] (two E2M1 per byte, low nibble first),
    scales [rows][K / 16] -> (codes [rows / 8][K / 128][32 lanes][16 fragments], scales [rows / 8][K / 128][8][8]).
    Lane l (8x8 fragment place fm, fn) of fragment f = 4q + j holds weight rows fn and fn + 1 (low / high nibble) at
    input 32q + 8 * (fm // 2) + 2j + fm % 2 of its 128-input block."""
    rows, half = codes.shape
    K = 2 * half
    nrb, nkb = rows // 8, K // 128
    lane = np.arange(32)
    fm = ((lane // 4) & 4) + (lane // 2) % 4
    fn = ((lane // 4) & 2) * 2 + (lane % 2) * 2
    f = np.arange(16)
    k = (f[None, :] // 4) * 32 + (fm[:, None] & 6) * 4 + (f[None, :] % 4) * 2 + (fm[:, None] & 1)   # [lane][f]
    blocks = codes.reshape(nrb, 8, nkb, 64).transpose(0, 2, 1, 3)                     # [nrb][nkb][row][byte]
    shift = (4 * (k & 1)).astype(np.uint8)
    lo = (blocks[:, :, fn[:, None], k // 2] >> shift) & 15                            # [nrb][nkb][lane][f]
    hi = (blocks[:, :, (fn + 1)[:, None], k // 2] >> shift) & 15
    return np.ascontiguousarray((lo | (hi << 4)).astype(np.uint8)), np.ascontiguousarray(scales.reshape(nrb, 8, nkb, 8).transpose(0, 2, 1, 3))


def global_scale(amax: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """mlx-lm: amax * ((6 * 448) / denominator), FP32 operations."""
    scale = np.float32(6.0 * 448.0) / denominator.astype(np.float32)
    return (amax.astype(np.float32) * scale).astype(np.float32)


def chunk_plan(n: int, sizes: Sequence[int]) -> List[int]:
    """Prefill chunks for ``n`` positions from the recorded program sizes: the largest that fits, repeatedly."""
    sizes = sorted(set(int(c) for c in sizes), reverse=True)
    if n < 1 or not sizes or sizes[-1] != 1:
        raise ValueError("chunk_plan needs n >= 1 and a one-row size")
    out = []
    while n:
        c = next(r for r in sizes if r <= n)
        out.append(c)
        n -= c
    return out


class Inkling2Runtime(_MTPMixin, _DFlash2Mixin):
    """Owns the Metal plan, the buffers and the decode programs (batch 1)."""
    _workers, _threads = WORKERS, THREADS

    def __init__(self, checkpoint_dir: str, max_seq: int = 8192, library: Optional[str] = None,
                 sequences: int = 1, row_capacity: int = 1024, row_shapes=None, trace_rows: bool = False,
                 mtp_batches: Sequence[int] = (), mtp_prefill: Sequence[int] = (), q8: bool = False,
                 dflash2: Optional[str] = None):
        """``sequences``: KV / conv-state slots (batched decode); ``row_capacity``: rows of a prefill chunk or a
        batched decode step. ``mtp_batches`` / ``mtp_prefill``: batch sizes and prompt lengths of DSpark
        speculative decoding programs (mtp.safetensors; see _inkling2_mtp). ``dflash2``: the directory of the
        DFlash 2 drafter (incoai/Inkling-2-Mini-DFlash2) to draft with instead of DSpark (one sequence:
        ``mtp_batches=[1]``; see _inkling2_dflash2)."""
        if max_seq < 1 or sequences < 1 or row_capacity < 1:
            raise ValueError("max_seq, sequences and row_capacity must be positive")
        self.q8 = bool(q8)
        self.mtp_batches, self.mtp_prefill = [int(c) for c in mtp_batches], [int(p) for p in mtp_prefill]
        self.mtp = bool(self.mtp_batches)
        self.dflash2 = str(dflash2) if dflash2 else None
        if self.dflash2 and self.mtp_batches != [1]:
            raise ValueError("the DFlash 2 drafter serves one sequence (mtp_batches=[1])")
        if self.mtp:
            row_capacity = max(int(row_capacity), MTP_BLOCK * max(self.mtp_batches), *self.mtp_prefill)
            if max(self.mtp_batches) > sequences:
                raise ValueError("MTP batch sizes must not exceed sequences")
        self.sequences, self.row_capacity = int(sequences), int(row_capacity)
        self.trace_rows = bool(trace_rows)
        self.row_shapes = None if row_shapes is None else [tuple(x) for x in row_shapes]
        if self.row_shapes is not None and not self.row_shapes:
            raise ValueError("row_shapes must name at least one (rows, logit rows) program")
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
        if self.dflash2:
            self._df_file.close()
        self._allocate_state()
        self._allocate_rows()
        self._arranged: Optional[Dict[int, tuple]] = None   # allocated by the first row program with MMA experts
        if self.mtp:
            self._mtp_allocate()
        if self.dflash2:
            self._df_allocate()
        self._q8_allocate()
        self.programs = {name: self._record_decode(kernel) for name, kernel in DECODE_KERNELS.items()}
        self.programs["noise"] = self._record_noise(self.plan.program())
        self.row_programs: Dict[tuple, int] = {}
        for rows, logit_rows in self._row_shapes():
            self.row_programs[(rows, logit_rows)] = self._record_rows(rows, logit_rows)
        if self.mtp:
            self._mtp_record()
        if self._arranged is not None:
            self.programs["arrange"] = self._record_arrange()
        if self._q8:
            self.programs["q8_quantize"] = self._record_q8_quantize()
        if len(self._param_blob) > ARGS_BYTES:
            raise ValueError(f"recorded programs need {len(self._param_blob)} parameter bytes (> {ARGS_BYTES}): "
                             "fewer row shapes / MTP batch sizes")
        self.plan.write(self.buf["args"], 0, np.frombuffer(bytes(self._param_blob), dtype=np.uint8))
        self.plan.finish(resident=True)
        if self._arranged is not None:
            self.plan.run(self.programs["arrange"])     # NVFP4 experts re-arranged per MMA fragment, once
        if self._q8:
            self.plan.run(self.programs["q8_quantize"])  # Q8_0 copies of BF16 matrices, once
        self._params = self.plan.view(self.buf["params"], np.uint32, 16)
        self._packet = self.plan.view(self.buf["packet"], np.uint32, 17)
        self._diagnostic = self.plan.view(self.buf["diagnostic"], np.uint32, WORKERS)
        self._control = self.plan.view(self.buf["control"], np.uint32, CONTROL)
        if self.mtp:
            self._verify_params = self.plan.view(self.buf["v_params"], np.uint32, 4)
        self.load_seconds = time.perf_counter() - started
        self.rotation = 0
        self.debug_visit = 0xFFFFFFFF
        self.pos = 0
        self.reset()

    # ------------------------------------------------------------------ Q8_0 copies
    # The drafter's base logits use a Q8_0 copy of the LM head and its layers Q8_0 copies of their matrices (drafts
    # only decide how many tokens a step accepts).
    q8_draft_head = True
    q8_drafter = True

    def _q8_matrices(self):
        """(name, byte offset in the weight buffer, rows, inputs) of the BF16 matrices that get a Q8_0 copy; the byte
        offset may be a list of (offset, rows) parts stacked into one Q8_0 matrix."""
        cfg = self.cfg
        H = cfg.hidden_size
        out = []
        if self.q8:     # per layer q, k, v, r, gate, o (the order of the kernels' q8 table)
            for i in range(cfg.num_layers):
                d = self.table[i]
                for slot, (rows, k) in zip((D_WQ, D_WK, D_WV, D_WR, D_WG, D_WO),
                                           ((3072, H), (1024, H), (1024, H), (384, H), (3072, H), (H, 3072))):
                    out.append((f"{i}.{slot}", int(d[slot]), rows, k))
        if self.q8 or (self.mtp and self.q8_draft_head):
            out.append(("head", int(self.table[cfg.num_layers][D_HEAD]), cfg.vocab_size, H))
        if self.mtp and self.q8_drafter:
            out += self._mtp_q8_matrices()
        if self.dflash2:
            out += self._df_q8_matrices()
        return out

    def _q8_allocate(self):
        self._q8: Dict[str, tuple] = {}
        size = 0
        for name, src, rows, k in self._q8_matrices():
            codes = size
            size += rows * k
            size += (-size) % ALIGN
            scales = size
            size += rows * (k // 32) * 2
            size += (-size) % ALIGN
            self._q8[name] = (src if isinstance(src, list) else [(src, rows)], rows, k, codes, scales)
        if self._q8:
            self.buf["q8"] = self.plan.buffer(size)
        if self.q8:
            qt = np.zeros(self.cfg.num_layers * 12 + 2, np.uint64)
            for i in range(self.cfg.num_layers):
                for j, slot in enumerate((D_WQ, D_WK, D_WV, D_WR, D_WG, D_WO)):
                    qt[i * 12 + 2 * j:i * 12 + 2 * j + 2] = self._q8[f"{i}.{slot}"][3:5]
            qt[-2:] = self._q8["head"][3:5]
            self.buf["q8table"] = self.plan.buffer(qt.nbytes)
            self.plan.write(self.buf["q8table"], 0, qt)

    def _record_q8_quantize(self) -> int:
        program = self.plan.program()
        for name, (parts, rows, k, codes, scales) in self._q8.items():
            r0 = 0
            for src, pr in parts:
                prm = self._param(np.array([src], np.uint64).tobytes() + np.array([pr, k, 0, 0], np.uint32).tobytes())
                n = pr * (k // 32)
                self.plan.dispatch("q8_0_quantize_bf16", [(self.buf["weights"], 0, READ),
                                                          (self.buf["q8"], codes + r0 * k, WRITE),
                                                          (self.buf["q8"], scales + r0 * (k // 32) * 2, WRITE),
                                                          (self.buf["args"], prm, READ)],
                                   ((n + 255) // 256, 1, 1), 256, label=f"q8_quantize_{name}")
                r0 += pr
        return program

    q8_groups = 80                       # threadgroups of the staged Q8_0 product

    def _q8_rows(self, name, x, x_off, x_stride, rows, y, y_off, y_stride, label):
        """y[r] = Q8_0 matrix `name` x x[r] (FP32 out) for rows <= 8."""
        _, n, k, codes, scales = self._q8[name]
        prm = self._param(np.array([rows, n, k, x_stride, y_stride, 0, 0, 0], np.uint32).tobytes())
        staged = rows * k <= 16128           # whole input in threadgroup memory: a fixed grid, the input staged once
        self.plan.dispatch("q8_0_rows_f32_staged" if staged else "q8_0_rows_f32",
                           [(x, x_off, READ), (self.buf["q8"], codes, READ), (self.buf["q8"], scales, READ),
                            (y, y_off, WRITE), (self.buf["args"], prm, READ)],
                           (min(self.q8_groups, (n + 63) // 64) if staged else (n + 63) // 64, 1, 1), 256, label=label)

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
        if self.dflash2:
            self._df_upload(layout)
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
            "control": CONTROL * 4, "diagnostic": WORKERS * 4, "noise_params": 64,
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
            "args": ARGS_BYTES, "r_trace": (48 * C * H * bf if self.trace_rows else 4),
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
    gemm_tiles = {8: (8, 32, 1), 16: (16, 32, 2), 64: (16, 32, 2)}
    # Row-count paths. Up to `gemv_rows` rows: multi-row gemv / per-expert qmv, each row's arithmetic the decode
    # step's (results independent of the batch). Up to `hybrid_limit` rows: experts with at most `hybrid_rows`
    # rows take the qmv kernel, the others TensorOps tiles (a tile pays its weights for up to tile_m rows).
    gemv_rows = 4
    hybrid_rows = 3
    hybrid_limit = 1 << 30
    # NVFP4 experts (layers 0-19) of programs with more than gemv_rows and at most mma_limit rows: SIMD-group MMAs
    # over a load-time re-arranged copy of the expert weights (+11.5 GB; ink_nv_arrange / ink_experts_mma_*)
    # instead of TensorOps tiles, experts with at most mma_rows rows on qmv. Up to mma_tile_rows rows a
    # threadgroup runs all passes of one expert; above, one 32-row pass (scan tiles: large experts spread over
    # many threadgroups). expert_mma = 0: no copy, TensorOps.
    expert_mma = 1
    mma_rows = 2
    mma_limit = 1 << 30
    mma_tile_rows = 64
    # Programs of at most expert_overlap_rows rows run the qmv experts and the MMA / TensorOps experts of a stage
    # concurrently (no barrier between their dispatches; NativePlan.dispatch(barrier=False)): 1% faster up to 32
    # rows, not above. 0 disables.
    expert_overlap_rows = 32
    expert_k32 = 64             # up to this many rows the gate/up tiles dequantize 32-input slices (64 otherwise)
    # expert tiles by row count (8 assignments per row over 128 experts: R / 16 rows per expert on average)
    expert_tiles = {64: (8, 32, 2), 256: (16, 64, 4), 1 << 30: (32, 64, 4)}

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

    def _record_rows(self, R: int, logit_rows: int, stage: bool = False, features: bool = False,
                     then=None) -> int:
        """Program for R rows (rows_info and tokens written per run): all 48 visits, then the logits of the last
        ``logit_rows`` rows (softcapped FP32) and their greedy tokens. ``stage``: conv frames and sliding K / V
        staged for ink_commit_rows (speculative verification); ``features``: the drafter features of every row
        (m_feat: the hidden state after visits 4, 22, 40 and the final normalized state). ``then``: records more
        dispatches at the end of the same program."""
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
        few_mma = self.mma_rows if hybrid else 0     # MMA layers: experts with at most few_mma rows on qmv, no tiles
        tmp_mma = self._param(np.array([32, few_mma], np.uint32).tobytes())
        for v, (layer, pss) in enumerate(cfg.visits()):
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
            mx = 1 if int(d[D_FLAGS]) & 4 else 0
            mma = bool(self.expert_mma) and not gemv and not mx and R <= self.mma_limit
            few = few_mma if mma else (self.hybrid_rows if hybrid else 0)
            mtiles = mma and R > self.mma_tile_rows
            plan.dispatch("ink_expert_scan", [(b["r_counts"], 0, READ_WRITE), (b["r_starts"], 0, WRITE),
                          (b["r_cursors"], 0, WRITE), (b["r_tiles"], 0, WRITE), (A, tmp_mma if mtiles else tmp, READ)],
                          (1, 1, 1), 32, label="scan")
            plan.dispatch("ink_expert_place_rows", [(b["r_ids"], 0, READ), (b["r_starts"], 0, READ),
                          (b["r_cursors"], 0, READ_WRITE), (b["r_slot"], 0, WRITE), (b["r_y"], 0, READ),
                          (b["r_xp"], 0, WRITE), vis], (R * 8, 1, 1), 256, label="place")
            qmv = gemv or few > 0      # per-expert qmv: every expert (gemv) or those with at most `few` rows
            tiles = not (gemv or mma)  # TensorOps tiles for the experts qmv leaves
            if mma and self._arranged is None:
                self._allocate_arranged()
            # gate/up, then down. Within a stage the qmv experts and the others write disjoint rows of the
            # same output, and each down product reads only its own gate/up rows: up to expert_overlap_rows
            # rows the larger experts' dispatch has no barrier, so the two run concurrently.
            concurrent = R <= self.expert_overlap_rows and qmv and (mma or tiles)
            for gu in (1, 0):
                src, dst = ("r_xp", "r_act") if gu else ("r_act", "r_ep")
                if qmv:
                    w, s_, gs, k = (D_W13, D_S13, D_GS13, H) if gu else (D_W2, D_S2, D_GS2, I)
                    wq = self._param(np.array([d[w], d[s_], d[gs] if not mx else 0, few], np.uint64).tobytes())
                    sh = self._param(np.array([gu, mx, 2 * I if gu else H, k], np.uint32).tobytes())
                    plan.dispatch("ink_experts_qmv_rows", [W, (b[src], 0, READ), (b[dst], 0, WRITE), (A, wq, READ),
                                  (b["r_starts"], 0, READ), (A, sh, READ)], ((2 * I if gu else H) // 32, E, 1), 256,
                                  label="experts_gateup" if gu else "experts_down")
                if mma:
                    ag, ags, ad, ads = self._arranged[layer]
                    name, codes, scales, gs, k, wrows, nout = (
                        ("ink_experts_mma_gateup", ag, ags, D_GS13, H, 2 * I, I) if gu else
                        ("ink_experts_mma_down", ad, ads, D_GS2, I, H, H))
                    prm = self._param(np.array([codes, scales, d[gs]], np.uint64).tobytes() +
                                      np.array([k, wrows, few, nout], np.uint32).tobytes())
                    if mtiles:
                        plan.dispatch(name + "_tiles", [W, (b["arranged"], 0, READ), (b[src], 0, READ),
                                      (b[dst], 0, WRITE), (A, prm, READ), (b["r_starts"], 0, READ),
                                      (b["r_tiles"], 0, READ)], (wrows // 64, (R * 8 + 31) // 32 + E, 1), 256,
                                      label=name, barrier=not concurrent)
                    else:
                        plan.dispatch(name, [W, (b["arranged"], 0, READ), (b[src], 0, READ), (b[dst], 0, WRITE),
                                             (A, prm, READ), (b["r_starts"], 0, READ)], (wrows // 64, E, 1), 256,
                                      label=name, barrier=not concurrent)
                if tiles:
                    name, w, s_, gs, k, n_out = (("ink_experts_gateup", D_W13, D_S13, D_GS13, H, I) if gu else
                                                 ("ink_experts_down", D_W2, D_S2, D_GS2, I, H))
                    blob = np.array([R * 8, k, n_out, mx], np.uint32).tobytes() + \
                        np.array([d[w], d[s_], d[gs] if not mx else 0], np.uint64).tobytes()
                    prm = self._param(blob)
                    k32 = gu and R <= self.expert_k32
                    kn = f"{name}_t{tm}x{etn}s{esg}{'k32' if k32 else ''}"
                    tiles_y = (R * 8 + tm - 1) // tm + E
                    plan.dispatch(kn, [(b[src], 0, READ), W, (b[dst], 0, WRITE), (A, prm, READ),
                                  (b["r_starts"], 0, READ), (b["r_tiles"], 0, READ)],
                                  ((n_out + etn - 1) // etn, tiles_y, 1), esg * 32, label=kn, barrier=not concurrent)
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
        if then is not None:
            then()
        return program

    def prefill_chunks(self, features: bool = False) -> List[int]:
        """Row counts of the recorded prefill programs (one logit row; with drafter features when ``features``)."""
        if features:
            return sorted((k[1] for k in getattr(self, "mtp_programs", {}) if k[0] == "prefill"), reverse=True)
        return sorted((r for r, lr in self.row_programs if lr == 1), reverse=True)

    def prefill(self, ids: Sequence[int], seq: int = 0, start: int = 0, features: bool = False) -> int:
        """Prefill positions ``start .. start + len(ids) - 1`` of sequence ``seq`` in chunks of the recorded prefill
        programs (largest first); returns the greedy next token. ``features``: the programs that also write the
        drafter features and the drafter context of their positions."""
        if not ids:
            raise ValueError("ids must be non-empty")
        if not 0 <= seq < self.sequences:
            raise ValueError("sequence slot out of range")
        if start + len(ids) > self.max_seq:
            raise ValueError(f"prefill range exceeds max_seq {self.max_seq}")
        if any(not 0 <= int(t) < self.cfg.unpadded_vocab_size for t in ids):
            raise ValueError("token id outside the checkpoint vocabulary")
        sizes = self.prefill_chunks(features)
        if not sizes or sizes[-1] != 1:
            raise ValueError("prefill needs programs down to one row (row_shapes / mtp_prefill)")
        done, token = 0, 0
        for c in chunk_plan(len(ids), sizes):
            first = start + done
            info = np.array([[seq, first + i, first, first + c - 1] for i in range(c)], np.uint32)
            self.plan.write(self.buf["r_tokens"], 0, np.asarray(ids[done:done + c], np.uint32))
            self.plan.write(self.buf["r_info"], 0, info)
            if features:            # the program ends with the drafter context of these positions
                ci = np.zeros((c, 4), np.uint32)
                ci[:, 0], ci[:, 1], ci[:, 2] = seq, np.arange(first, first + c), 1
                if done + c == len(ids):
                    ci[-1, 3] = 1
                self.plan.write(self.buf["m_cinfo"], 0, ci)
            self.last_rows_timing = self.plan.run(self.mtp_programs[("prefill", c)] if features else
                                                  self.row_programs[(c, 1)])
            token = int(self.plan.read(self.buf["r_out"], 0, np.uint32, 1)[0])
            done += c
        return token

    def run_rows(self, tokens: Sequence[int], info: np.ndarray, logit_rows: int) -> np.ndarray:
        """Run the row program for len(tokens) rows with rows_info ``info`` [rows, 4] (sequence, position, first,
        last); returns the greedy tokens of the last ``logit_rows`` rows."""
        R = len(tokens)
        key = (R, logit_rows)
        if key not in self.row_programs:
            raise ValueError(f"no row program recorded for {key}; pass row_shapes")
        info = np.ascontiguousarray(info, np.uint32)
        if info.shape != (R, 4):
            raise ValueError("info must be [rows, 4]: (sequence, position, first, last)")
        seq, pos, first, last = info.T
        if (seq >= self.sequences).any() or (last >= self.max_seq).any() or (pos < first).any() or (pos > last).any():
            raise ValueError("rows_info out of range: sequence < sequences, first <= position <= last < max_seq")
        # rows of a sequence at positions first..last must be consecutive rows of the dispatch
        prev_pos, prev_seq = np.r_[np.uint32(0xFFFFFFFF), pos[:-1]], np.r_[np.uint32(0xFFFFFFFF), seq[:-1]]
        if ((pos > first) & ((prev_pos != pos - 1) | (prev_seq != seq))).any():
            raise ValueError("rows of one sequence must hold consecutive positions in consecutive rows")
        if any(not 0 <= int(t) < self.cfg.unpadded_vocab_size for t in tokens):
            raise ValueError("token id outside the checkpoint vocabulary")
        self.plan.write(self.buf["r_tokens"], 0, np.asarray(tokens, np.uint32))
        self.plan.write(self.buf["r_info"], 0, info)
        self.last_rows_timing = self.plan.run(self.row_programs[key])
        return self.plan.read(self.buf["r_out"], 0, np.uint32, logit_rows).copy()

    def rows_logits(self, row: int = 0) -> np.ndarray:
        return self.plan.read(self.buf["r_logits"], row * self.cfg.vocab_size * 4, np.float32,
                              self.cfg.unpadded_vocab_size)

    def _allocate_arranged(self) -> None:
        """NVFP4 experts re-arranged for ink_experts_mma_* (codes per lane and fragment, scales per 8-row block)."""
        cfg = self.cfg
        H, I, E = cfg.hidden_size, cfg.expert_intermediate_size, cfg.num_experts
        self._arranged = {}
        size = 0
        for layer in range(cfg.num_layers):
            if int(self.table[layer][D_FLAGS]) & 4:
                continue                                 # MXFP8 layers keep the TensorOps tiles
            offs = []
            for rows, k in ((2 * I, H), (H, I)):
                codes, scales = E * rows * k // 2, E * rows * k // 16
                offs += [size, size + codes]
                size += codes + scales
                size += (-size) % 256
            self._arranged[layer] = tuple(offs)
        self.buf["arranged"] = self.plan.buffer(max(size, 256))

    def _record_arrange(self) -> int:
        cfg, b = self.cfg, self.buf
        H, I, E = cfg.hidden_size, cfg.expert_intermediate_size, cfg.num_experts
        program = self.plan.program()
        for layer, (gc, gs_, dc, ds) in self._arranged.items():
            d = self.table[layer]
            for (rows, k), (w, sc), (ac, asc) in zip(((2 * I, H), (H, I)), ((D_W13, D_S13), (D_W2, D_S2)),
                                                    ((gc, gs_), (dc, ds))):
                prm = self._param(np.array([d[w], d[sc], ac, asc], np.uint64).tobytes() +
                                  np.array([E, rows, k, 0], np.uint32).tobytes())
                n = E * (rows // 8) * (k // 128) * 32
                self.plan.dispatch("ink_nv_arrange", [(b["weights"], 0, READ), (b["arranged"], 0, WRITE),
                                                      (b["args"], prm, READ)], ((n + 255) // 256, 1, 1), 256,
                                   label="arrange")
        return program

    def _record_noise(self, program: int) -> int:
        """Gumbel noise rows [steps][vocab] for the sampled step from (seed, position) (gumbel_noise_f32, whose
        parameter words live in noise_params: position [1], steps [2], seed [12], [13], enable [14])."""
        v = self.cfg.vocab_size
        prm = self._param(np.array([v], np.uint32).tobytes())
        self.plan.dispatch("gumbel_noise_f32", [(self.buf["noise"], 0, WRITE), (self.buf["noise_params"], 0, READ),
                                                (self.buf["args"], prm, READ)], ((BLOCK * v + 255) // 256, 1, 1), 256,
                           label="gumbel_noise_f32")
        return program

    def _record_decode(self, kernel: str) -> int:
        b = self.buf
        program = self.plan.program()
        if kernel == DECODE_KERNELS["sampled"]:
            self._record_noise(program)
        extra = []
        if self.q8:     # Q8_0 attention projections and LM head
            kernel = kernel.replace("inkling2_decode_resident_", "inkling2_decode_resident_q8_")
            extra = [(b["q8"], 0, READ), (b["q8table"], 0, READ)]
        self.plan.dispatch(kernel, [
            (b["weights"], 0, READ), (b["table"], 0, READ), (b["kv"], 0, READ_WRITE), (b["rings"], 0, READ_WRITE),
            (b["params"], 0, READ), (b["noise"], 0, READ), (b["logits"], 0, WRITE), (b["trace"], 0, WRITE),
            (b["packet"], 0, WRITE), (b["workspace"], 0, READ_WRITE), (b["debug"], 0, WRITE),
            (b["control"], 0, READ_WRITE), (b["diagnostic"], 0, WRITE),
        ] + extra, (WORKERS, 1, 1), THREADS, label=kernel)
        return program

    def reset(self) -> None:
        """Empty context (the caches need no clearing: a step reads only slots it or its predecessors wrote,
        and the conv windows before position 0 read as zero)."""
        self.pos = 0
        self.rotation = 0

    # ------------------------------------------------------------------- decode
    def _set_noise(self, position: int, steps: int, seed: int) -> None:
        q = np.zeros(16, np.uint32)
        q[1], q[2], q[12], q[13], q[14] = position, steps, seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 1
        self.plan.write(self.buf["noise_params"], 0, q)

    @staticmethod
    def _check_sampling(top_k: int, top_p: float, temperature: float) -> None:
        if not temperature > 0:
            raise ValueError("sampling needs temperature > 0 (temperature 0 is the greedy program)")
        if not 1 <= int(top_k) <= 32 or not 0.0 < float(top_p) <= 1.0:
            raise ValueError("the sampled program needs 1 <= top_k <= 32 and 0 < top_p <= 1")

    def sample_logits(self, logits: np.ndarray, position: int, seed: int, top_k: int, top_p: float,
                      temperature: float) -> int:
        """The sampled step's selection rule on the host (the first token after a prefill): the top_k largest
        logits (ties: larger id first), cut to the nucleus top_p of the full-vocabulary probabilities (mlx-lm:
        cumulative from the largest), then the maximum of log-probability / temperature + Gumbel noise of
        (seed, position) (ties: lowest id)."""
        self._check_sampling(top_k, top_p, temperature)
        self._set_noise(position, 1, seed)
        self.plan.run(self.programs["noise"])
        noise = self.plan.read(self.buf["noise"], 0, np.float32, self.cfg.unpadded_vocab_size)
        x = np.asarray(logits, np.float32)[:self.cfg.unpadded_vocab_size].astype(np.float64)
        lse = x.max() + np.log(np.exp(x - x.max()).sum())
        ids = np.arange(x.size)
        order = np.lexsort((-ids, -x))[:int(top_k)]          # value descending, then larger id first
        lp = x[order] - lse
        before = np.concatenate([[0.0], np.cumsum(np.exp(lp))[:-1]])
        kept = (float(top_p) >= 1.0) | ((1.0 - before) > (1.0 - float(top_p)))
        score = np.where(kept, lp / float(temperature) + noise[order], -np.inf)
        best = score.max()
        return int(order[score == best].min())

    def decode(self, token: int, position: int, steps: int, program: str = "greedy",
               eos: Sequence[int] = (), top_k: int = 20, top_p: float = 1.0, temperature: float = 1.0,
               seed: int = 0) -> List[int]:
        """``steps`` (1..8) tokens after ``token`` at ``position``. ``program="sampled"``: fused top-k / top-p
        selection with Gumbel noise drawn from ``seed`` and each step's position (reproducible per seed)."""
        if program not in ("greedy", "reference", "sampled"):
            raise ValueError("program must be greedy, reference or sampled")
        if program == "sampled":
            self._check_sampling(top_k, top_p, temperature)
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
        if program == "sampled":
            p[8] = int(top_k)
            p[9:11] = np.array([top_p, 1.0 / temperature], np.float32).view(np.uint32)
            self._set_noise(position, steps, int(seed))
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
