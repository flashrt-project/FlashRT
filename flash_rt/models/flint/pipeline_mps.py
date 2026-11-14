"""Flint launch sequence on Apple Silicon (native Metal plan).

The decode step is one resident dispatch: ``workers`` co-resident threadgroups run up to eight
consecutive tokens (embedding, 49 layers, LM head, token selection) between device-scope barriers.
This mixin only records programs over buffers owned by the runtime (``self.plan``, ``self.buf``); it
imports no tokenizer, server or framework code.
"""

from __future__ import annotations

from typing import Optional

from flash_rt.hardware.apple_native.native import READ, READ_WRITE, WRITE
from flash_rt.models.flint.config import FlintConfig, FlintConfigError

WORKERS = 36            # co-resident threadgroups of 256 threads (M4 Pro / M4 Max GPU profile)
THREADS = 256
BLOCK = 8               # tokens per resident dispatch

# Geometry hard-coded by the csrc/apple/flint kernels (decode step, prefill rows, routing).
GEOMETRY = {
    "num_layers": 49, "num_dense_layers": 2, "hidden_size": 2048, "num_heads": 32, "num_kv_heads": 4,
    "head_dim": 128, "vocab_size": 150144, "dense_intermediate_size": 6144, "num_experts": 256,
    "num_experts_per_tok": 6, "expert_intermediate_size": 512, "shared_intermediate_size": 1024,
    "sliding_window": 513, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
}

# Layer descriptor: DESCRIPTOR uint64 per layer plus one final entry. Byte offsets into the weight buffer
# (an affine matrix takes four: codes, scales, biases, rows << 32 | inputs) and element offsets into the KV
# buffer; the decode kernel reads the same layout.
DESCRIPTOR = 64
D_FLAGS, D_IN_NORM, D_POST_NORM, D_Q_NORM, D_K_NORM, D_SINKS, D_ROUTER, D_BIAS = range(8)
D_Q, D_K, D_V, D_O, D_GATE, D_UP, D_DOWN, D_SH_GATE, D_SH_UP, D_SH_DOWN = range(8, 48, 4)
D_KV, D_KV_CAP = 48, 49
D_FINAL_NORM, D_EMBED, D_HEAD = 1, 8, 12   # final entry


def check_geometry(cfg: FlintConfig) -> None:
    """Refuse a checkpoint whose geometry differs from what the Apple Flint kernels hard-code."""
    for key, value in GEOMETRY.items():
        if getattr(cfg, key) != value:
            raise FlintConfigError(f"{key}={getattr(cfg, key)}; the Apple Flint kernels are built for {value}")


DECODE_KERNELS = {
    "greedy": "flint_decode_resident_bf16",
    "sampled": "flint_decode_resident_sampled_bf16",
    "reference": "flint_decode_resident_reference_bf16",
}


class FlintPipelineMps:
    """Records the decode programs; the runtime supplies ``plan`` and the buffer table ``buf``."""

    def record_noise(self, program: Optional[int] = None) -> int:
        """Gumbel noise rows for the sampled step (into ``noise``), keyed by the decode parameters' seed and
        position; appended to ``program`` when given, else recorded as its own program."""
        b = self.buf
        if program is None:
            program = self.plan.program()
        rows = BLOCK * self.cfg.vocab_size
        self.plan.dispatch("gumbel_noise_f32", [(b["noise"], 0, WRITE), (b["params"], 0, READ),
                                                (b["args"], self.vocab_param, READ)],
                           ((rows + 255) // 256, 1, 1), 256, label="gumbel_noise_f32")
        return program

    def record_decode(self, kernel: str) -> int:
        b = self.buf
        program = self.plan.program()
        if kernel == DECODE_KERNELS["sampled"]:
            self.record_noise(program)
        self.plan.dispatch(kernel, [
            (b["weights"], 0, READ),
            (b["layers"], 0, READ),
            (b["kv"], 0, READ_WRITE),
            (b["params"], 0, READ),
            (b["noise"], 0, READ),
            (b["logits"], 0, WRITE),
            (b["trace"], 0, WRITE),
            (b["packet"], 0, WRITE),
            (b["workspace"], 0, READ_WRITE),
            (b["partials"], 0, READ_WRITE),
            (b["statistics"], 0, READ_WRITE),
            (b["routing"], 0, READ_WRITE),
            (b["control"], 0, READ_WRITE),
            (b["diagnostic"], 0, WRITE),
        ], (WORKERS, 1, 1), THREADS, label=kernel)
        return program


TM = 64   # dense GEMM tile rows (w4a16 TensorOps kernels)
TN = 64   # GEMM tile columns
GROUPED_TM = 64   # expert-grouped GEMM tile rows (expert blocks are tiled by GROUPED_TM rows)
SMALL_ROWS = 513    # chunks below this many rows take the small-chunk program (split-K / vector paths)
SPLITK_ROW_TILES = 8   # split-K products only arise for M <= 512 rows
GROUPED_VECTOR_ASSIGNMENTS = 1024   # expert products below this many assignments take the vector path
# Prefill attention kernels: name -> (registry entry, positions per threadgroup, threads). "mma" (8x8
# SIMD-group MMAs, Q in registers) is the default; "tensor_ops" computes the same arithmetic with TensorOps.
ATTENTION = {"mma": ("gqa8_causal_prefill_mma_bf16", 8, 256), "tensor_ops": ("gqa8_causal_prefill_bf16", 8, 128)}


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def qmv_limit(k: int, n: int) -> int:
    """Rows below which MLX's quantized matmul takes its vector path (get_qmv_batch_limit for MLX
    architecture generation 15 and later with the 's' size class, e.g. M4 Pro / M4 Max: applegpu_g16s)."""
    if k <= 2048 and n <= 2048:
        return 18
    if k <= 4096 and n <= 4096:
        return 12
    return 10


class FlintPrefillMps:
    """Records the chunked-prefill programs: KV population for positions [pos0, pos0 + rows) with
    rows <= capacity. The runtime supplies ``plan``, ``buf``, ``param(kind, **fields) -> offset`` and
    the per-layer descriptors; chunk position and row count come from the ``state`` buffer. The small-chunk
    program (``small=True``, rows < SMALL_ROWS) adds the split-K and row-vector product paths that MLX's
    matmuls take for small M, each kernel returning when its path does not apply."""

    attention = "mma"   # prefill attention kernel (key of ATTENTION)

    def record_prefill(self, capacity: int, layers: int = 0, first: int = 0, small: bool = False) -> int:
        b, cfg, L = self.buf, self.cfg, self.layer_table
        plan = self.plan
        program = plan.program()
        C, H, K = capacity, cfg.hidden_size, cfg.num_experts_per_tok
        state = (b["state"], 0, READ)

        def gemm(epilogue, x, w_off, y, rows, n, k, x_stride, y_stride, y_off=0, w2_off=None, grouped=False):
            limit = qmv_limit(k, n)
            w2 = w_off if w2_off is None else w2_off
            params = self.param("gemm", rows=rows, columns=n, inner=k, x_stride=x_stride, y_stride=y_stride,
                                experts=cfg.num_experts if grouped else 0, per_row=K if grouped else 1,
                                limit=limit if small else 0, w=w_off, w2=w2)
            binds = [(x, 0, READ), (b["weights"], 0, READ), (y, y_off, READ_WRITE), (b["args"], params, READ),
                     (b["starts"], 0, READ), (b["tiles"], 0, READ), state]
            name = "gemm_w4a16_affine" + ("_grouped" if grouped else "") + epilogue
            groups_y = _ceil(rows, GROUPED_TM) + cfg.num_experts if grouped else _ceil(rows, TM)
            plan.dispatch(name, binds, (_ceil(n, TN), groups_y, 1), 128 if grouped else 64, label=name)
            if not small:
                return
            if not grouped:
                split = "gemm_w4a16_affine_splitk" + epilogue
                plan.dispatch(split, binds, (_ceil(n, TN), min(_ceil(rows, TM), SPLITK_ROW_TILES), 1), 64, label=split)
            vec = self.param("qmv", inner=k, columns=n, x_stride=x_stride, y_stride=y_stride, limit=limit,
                             experts=cfg.num_experts if grouped else 0, per_row=K if grouped else 1, w=w_off, w2=w2)
            name = "qmv_w4a16_affine_rows" + ("_grouped" if grouped else "") + epilogue
            vec_rows = min(rows, GROUPED_VECTOR_ASSIGNMENTS - 1) if grouped else min(rows, limit - 1)
            vbinds = [(b["weights"], 0, READ), (x, 0, READ), (y, y_off, READ_WRITE), (b["args"], vec, READ),
                      state, (b["starts"], 0, READ)]
            plan.dispatch(name, vbinds, (vec_rows, n // 8, 1), 64, label=name)
            if not grouped:
                wide = "qmv_w4a16_affine_wide" + epilogue
                plan.dispatch(wide, vbinds, (vec_rows, n // 8, 1), 64, label=wide)

        if first == 0:
            emb = self.param("embed", hidden=H, w=L[cfg.num_layers][D_EMBED:D_EMBED + 3])
            plan.dispatch("flint_embed_affine8_rows_bf16", [
                (b["tokens"], 0, READ), (b["weights"], 0, READ), (b["h"], 0, WRITE), (b["args"], emb, READ), state,
            ], (H // 256, C, 1), 256)
        for i in range(first, layers or cfg.num_layers):
            d = L[i]
            sliding = cfg.sliding_layers[i]
            rms_in = self.param("rms", hidden=H, w=int(d[D_IN_NORM]), eps=cfg.rms_norm_eps)
            plan.dispatch("flint_rms_rows_bf16", [
                (b["h"], 0, READ), (b["weights"], 0, READ), (b["n"], 0, WRITE), (b["args"], rms_in, READ), state,
            ], (C, 1, 1), 256)
            qkv_w = cfg.qkv_dim
            for base, col, n in ((D_Q, 0, cfg.q_dim), (D_K, cfg.q_dim, cfg.kv_dim),
                                 (D_V, cfg.q_dim + cfg.kv_dim, cfg.kv_dim)):
                gemm("", b["n"], d[base:base + 3], b["qkv"], C, n, H, H, qkv_w, y_off=col * 2)
            kv_buf, kv_off, kv_cap = (b["scratch"], 0, cfg.sliding_window + C) if sliding else \
                (b["kv"], int(d[D_KV]), self.max_seq)
            if sliding:
                ring = self.param("ring", kv_heads=cfg.num_kv_heads, ring=cfg.sliding_window, scratch_cap=kv_cap,
                                  window=cfg.sliding_window, ring_kv=int(d[D_KV]), scratch_kv=0)
                plan.dispatch("flint_kv_ring_copy_bf16", [
                    (b["kv"], 0, READ_WRITE), (b["scratch"], 0, READ_WRITE), (b["args"], ring, READ), state,
                    (b["args"], self.const_zero, READ)], (1, max(C, cfg.sliding_window), 2 * cfg.num_kv_heads), 32)
            qk = self.param("qk", q_heads=cfg.num_heads, kv_heads=cfg.num_kv_heads, sliding=int(sliding), cap=kv_cap,
                            scratch=cfg.sliding_window, q_norm=int(d[D_Q_NORM]), k_norm=int(d[D_K_NORM]), kv=kv_off,
                            eps=cfg.rms_norm_eps, logbase=self.logbase)
            plan.dispatch("flint_qk_norm_rope_kv_rows_bf16", [
                (b["qkv"], 0, READ), (b["weights"], 0, READ), (b["qt"], 0, WRITE), (kv_buf, 0, READ_WRITE),
                (b["args"], qk, READ), state], (C, 1, 1), 256)
            att = self.param("attention", kv_heads=cfg.num_kv_heads, cap=kv_cap, sliding=int(sliding),
                             window=cfg.sliding_window, kv=kv_off, sinks=int(d[D_SINKS]), scale=cfg.head_dim ** -0.5)
            kernel, per_group, threads = ATTENTION[self.attention]
            plan.dispatch(kernel, [
                (b["qt"], 0, READ), (kv_buf, 0, READ), (b["weights"], 0, READ), (b["attn"], 0, WRITE),
                (b["args"], att, READ), state], (_ceil(C, per_group), cfg.num_kv_heads, 1), threads)
            if sliding:
                plan.dispatch("flint_kv_ring_copy_bf16", [
                    (b["kv"], 0, READ_WRITE), (b["scratch"], 0, READ_WRITE), (b["args"], ring, READ), state,
                    (b["args"], self.const_one, READ)], (1, max(C, cfg.sliding_window), 2 * cfg.num_kv_heads), 32)
            gemm("_residual", b["attn"], d[D_O:D_O + 3], b["h"], C, H, cfg.q_dim, cfg.q_dim, H)
            rms_post = self.param("rms", hidden=H, w=int(d[D_POST_NORM]), eps=cfg.rms_norm_eps)
            plan.dispatch("flint_rms_rows_bf16", [
                (b["h"], 0, READ), (b["weights"], 0, READ), (b["n"], 0, WRITE), (b["args"], rms_post, READ), state,
            ], (C, 1, 1), 256)
            if not cfg.is_moe(i):
                gemm("_swiglu", b["n"], d[D_GATE:D_GATE + 3], b["act"], C, cfg.dense_intermediate_size, H, H,
                     cfg.dense_intermediate_size, w2_off=d[D_UP:D_UP + 3])
                gemm("_residual", b["act"], d[D_DOWN:D_DOWN + 3], b["h"], C, H, cfg.dense_intermediate_size,
                     cfg.dense_intermediate_size, H)
                continue
            router = self.param("router", weight=int(d[D_ROUTER]), bias=int(d[D_BIAS]))
            plan.dispatch("flint_router_topk_rows_bf16", [
                (b["n"], 0, READ), (b["weights"], 0, READ), (b["ids"], 0, WRITE), (b["gates"], 0, WRITE),
                (b["args"], router, READ), state], (_ceil(C, 16), 1, 1), 128)
            plan.dispatch("flint_expert_count", [
                (b["ids"], 0, READ), (b["counts"], 0, READ_WRITE), state], (_ceil(C * K, 256), 1, 1), 256)
            plan.dispatch("flint_expert_scan", [
                (b["counts"], 0, READ_WRITE), (b["starts"], 0, WRITE), (b["tiles"], 0, WRITE),
                (b["cursors"], 0, WRITE), (b["args"], self.scan_param, READ)], (1, 1, 1), 256)
            plan.dispatch("flint_expert_place_bf16", [
                (b["n"], 0, READ), (b["ids"], 0, READ), (b["cursors"], 0, READ_WRITE), (b["slots"], 0, WRITE),
                (b["xp"], 0, WRITE), state], (C * K, 1, 1), 256)
            I = cfg.expert_intermediate_size
            gemm("_swiglu", b["xp"], d[D_GATE:D_GATE + 3], b["actp"], C * K, I, H, H, I,
                 w2_off=d[D_UP:D_UP + 3], grouped=True)
            gemm("", b["actp"], d[D_DOWN:D_DOWN + 3], b["yp"], C * K, H, I, I, H, grouped=True)
            S = cfg.shared_intermediate_size
            gemm("_swiglu", b["n"], d[D_SH_GATE:D_SH_GATE + 3], b["acts"], C, S, H, H, S,
                 w2_off=d[D_SH_UP:D_SH_UP + 3])
            gemm("", b["acts"], d[D_SH_DOWN:D_SH_DOWN + 3], b["shared"], C, H, S, S, H)
            plan.dispatch("flint_expert_combine_rows_bf16", [
                (b["h"], 0, READ_WRITE), (b["yp"], 0, READ), (b["shared"], 0, READ), (b["slots"], 0, READ),
                (b["gates"], 0, READ), state], (C, 1, 1), 256)
        return program

    def record_prefill_head(self) -> int:
        """Final norm of the chunk's last row and the LM head (one row vector) into decode logits slot 0."""
        b, cfg = self.buf, self.cfg
        final = self.layer_table[cfg.num_layers]
        program = self.plan.program()
        norm = self.param("rms", hidden=cfg.hidden_size, last=1, w=int(final[D_FINAL_NORM]), eps=cfg.rms_norm_eps)
        self.plan.dispatch("flint_rms_rows_bf16", [(b["h"], 0, READ), (b["weights"], 0, READ), (b["n"], 0, WRITE),
                                                 (b["args"], norm, READ), (b["state"], 0, READ)], (1, 1, 1), 256)
        head = self.param("qmv", inner=cfg.hidden_size, columns=cfg.vocab_size, w=final[D_HEAD:D_HEAD + 3], x=0, y=0)
        self.plan.dispatch("qmv_w4a16_affine", [(b["weights"], 0, READ), (b["n"], 0, READ), (b["logits"], 0, WRITE),
                                                (b["args"], head, READ)], (1, cfg.vocab_size // 8, 1), 64)
        return program
