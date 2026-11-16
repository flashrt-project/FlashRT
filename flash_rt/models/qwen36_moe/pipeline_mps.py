"""qwen3_5_moe launch sequence on Apple Silicon (native Metal plan).

The decode step is one resident dispatch: ``WORKERS`` co-resident threadgroups run up to eight consecutive
tokens (embedding, 40 layers, LM head, token selection) between device-scope barriers. This mixin only records
programs over buffers owned by the runtime (``self.plan``, ``self.buf``); it imports no tokenizer, server or
framework code.
"""

from __future__ import annotations

from typing import Optional

from flash_rt.hardware.apple_native.native import READ, READ_WRITE, WRITE
from flash_rt.models.qwen36_moe.mlx_checkpoint import Qwen35MoeConfig, Qwen35MoeConfigError

WORKERS = 36            # co-resident threadgroups of 256 threads (M4 Pro / M4 Max GPU profile)
THREADS = 256
BLOCK = 8               # tokens per resident dispatch

# Geometry hard-coded by the csrc/apple/qwen35moe kernels.
GEOMETRY = {
    "num_layers": 40, "full_attention_interval": 4, "hidden_size": 2048, "vocab_size": 248320,
    "num_heads": 16, "num_kv_heads": 2, "head_dim": 256, "rotary_dims": 64, "rope_theta": 1e7,
    "linear_key_heads": 16, "linear_value_heads": 32, "linear_key_dim": 128, "linear_value_dim": 128,
    "conv_kernel": 4, "num_experts": 256, "num_experts_per_tok": 8, "expert_intermediate_size": 512,
    "shared_intermediate_size": 512, "rms_norm_eps": 1e-6,
}

# Layer descriptor: DESCRIPTOR uint64 per layer plus one final entry. Byte offsets into the weight buffer (an
# affine matrix takes four: codes, scales, biases, rows << 32 | inputs) and element offsets into the state
# buffers; the decode kernel reads the same layout.
DESCRIPTOR = 64
D_FLAGS, D_IN_NORM, D_POST_NORM = 0, 1, 2
D_GDN_NORM, D_CONV, D_ALOG, D_DT = 3, 4, 5, 6          # DeltaNet layers
D_ROUTER_DENSE = 7                                     # router dequantized to BF16 (prefill)
D_Q_NORM, D_K_NORM = 3, 4                              # attention layers
D_M0, D_M1, D_M2, D_M3, D_OUT = 8, 12, 16, 20, 24      # DeltaNet: qkv, z, b, a, out; attention: q, k, v, -, o
D_ROUTER, D_SGATE, D_GATE, D_UP, D_DOWN, D_SH_GATE, D_SH_UP, D_SH_DOWN = range(28, 60, 4)
D_STATE, D_AUX = 60, 61                                # DeltaNet: state, conv ring; attention: KV, capacity
D_FINAL_NORM, D_EMBED, D_HEAD = 1, 8, 12               # final entry

STATE_ELEMENTS = 32 * 128 * 128     # FP32 DeltaNet state per layer
RING_ELEMENTS = 4 * 8192            # BF16 conv input frames per layer (slot = position mod 4)
WORKSPACE = 42240                   # BF16 elements (see the kernel's W_* offsets)
PARTIALS = 16 * 32 * 1024 * 8 // 2  # FP32 (or BF16 rounded) attention partitions
STATISTICS = 16 * 1024 * 2


def check_geometry(cfg: Qwen35MoeConfig) -> None:
    """Refuse a checkpoint whose geometry differs from what the Apple kernels hard-code."""
    for key, value in GEOMETRY.items():
        if getattr(cfg, key) != value:
            raise Qwen35MoeConfigError(
                f"{key}={getattr(cfg, key)}; the Apple qwen3_5_moe kernels are built for {value}")


DECODE_KERNELS = {
    "greedy": "qwen35moe_decode_resident_bf16",
    "sampled": "qwen35moe_decode_resident_sampled_bf16",
    "reference": "qwen35moe_decode_resident_reference_bf16",
}


class Qwen36MoePipelineMps:
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
            (b["states"], 0, READ_WRITE),
            (b["rings"], 0, READ_WRITE),
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


# ---------------------------------------------------------------------------------------------- prefill
# The w4a16 products are the model-independent kernels of the Flint prefill (same tiles and MLX dispatch rules).
from flash_rt.models.flint.pipeline_mps import (  # noqa: E402
    GEMM_THREADS, GROUPED_TM, GROUPED_VECTOR_ASSIGNMENTS, SMALL_ROWS, SPLITK_ROW_TILES, SPLITK_THREADS, TM, TN,
    _ceil, qmv_limit)

STATE_WORDS = 4   # uint32 per state record: position, rows, 0, end


class Qwen36MoePrefillMps:
    """Records the chunked-prefill programs: one program fills the model state (KV cache, DeltaNet states and conv
    rings) for positions [pos0, pos0 + rows) with rows <= capacity. The runtime supplies ``plan``, ``buf``,
    ``param(kind, **fields) -> offset`` and the per-layer descriptors; chunk position and row count come from the
    ``state`` buffer. The small-chunk program (``small=True``, rows < SMALL_ROWS) adds the split-K and row-vector
    product paths that MLX's matmuls take for small M, each kernel returning when its path does not apply."""

    def record_prefill(self, capacity: int, layers: int = 0, small: bool = False) -> int:
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
            plan.dispatch(name, binds, (_ceil(n, TN), groups_y, 1), GEMM_THREADS, label=name)
            # MLX splits K for few output tiles: small chunks, and narrow products (n < 512) at any chunk size
            if not grouped and (small or n < 512):
                split = "gemm_w4a16_affine_splitk" + epilogue
                plan.dispatch(split, binds, (_ceil(n, TN), _ceil(rows, TM) if n < 512 else
                                             min(_ceil(rows, TM), SPLITK_ROW_TILES), 1), SPLITK_THREADS, label=split)
            if not small:
                return
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

        emb = self.param("embed", hidden=H, w=L[cfg.num_layers][D_EMBED:D_EMBED + 3])
        plan.dispatch("qwen35moe_embed_affine4_rows_bf16", [
            (b["tokens"], 0, READ), (b["weights"], 0, READ), (b["h"], 0, WRITE), (b["args"], emb, READ), state,
        ], (H // 256, C, 1), 256)
        I, S = cfg.expert_intermediate_size, cfg.shared_intermediate_size
        for i in range(layers or cfg.num_layers):
            d = L[i]
            rms_in = self.param("rms", hidden=H, w=int(d[D_IN_NORM]), eps=cfg.rms_norm_eps)
            plan.dispatch("flint_rms_rows_bf16", [
                (b["h"], 0, READ), (b["weights"], 0, READ), (b["n"], 0, WRITE), (b["args"], rms_in, READ), state,
            ], (C, 1, 1), 256)
            if not cfg.is_full_attention(i):
                V = cfg.linear_value_heads
                gemm("", b["n"], d[D_M0:D_M0 + 3], b["qkv"], C, cfg.conv_dim, H, H, cfg.conv_dim)
                gemm("", b["n"], d[D_M1:D_M1 + 3], b["z"], C, cfg.value_dim, H, H, cfg.value_dim)
                gemm("", b["n"], d[D_M2:D_M2 + 3], b["ba"], C, V, H, H, 2 * V)
                gemm("", b["n"], d[D_M3:D_M3 + 3], b["ba"], C, V, H, H, 2 * V, y_off=V * 2)
                delta = self.param("delta", conv=int(d[D_CONV]), alog=int(d[D_ALOG]), dt=int(d[D_DT]),
                                   norm=int(d[D_GDN_NORM]), ring=int(d[D_AUX]), state=int(d[D_STATE]))
                plan.dispatch("qwen35moe_deltanet_inputs_rows_bf16", [
                    (b["qkv"], 0, READ), (b["rings"], 0, READ), (b["weights"], 0, READ), (b["ba"], 0, READ),
                    (b["dqk"], 0, WRITE), (b["dv"], 0, WRITE), (b["dgates"], 0, WRITE), (b["args"], delta, READ),
                    state], (C, 1, 1), 256)
                plan.dispatch("qwen35moe_deltanet_recurrence_bf16", [
                    (b["dqk"], 0, READ), (b["dv"], 0, READ), (b["dgates"], 0, READ), (b["states"], 0, READ_WRITE),
                    (b["dy"], 0, WRITE), (b["args"], delta, READ), state], (128, 1, 1), 256)
                plan.dispatch("qwen35moe_deltanet_ring_bf16", [
                    (b["qkv"], 0, READ), (b["rings"], 0, READ_WRITE), (b["args"], delta, READ), state],
                    (cfg.conv_dim // 256, 4, 1), 256)
                plan.dispatch("qwen35moe_deltanet_gate_rows_bf16", [
                    (b["dy"], 0, READ), (b["z"], 0, READ), (b["weights"], 0, READ), (b["attn"], 0, WRITE),
                    (b["args"], delta, READ), state], (C, 1, 1), 256)
                inner = cfg.value_dim
            else:
                width = 2 * cfg.q_dim + 2 * cfg.kv_dim
                for base, col, n in ((D_M0, 0, 2 * cfg.q_dim), (D_M1, 2 * cfg.q_dim, cfg.kv_dim),
                                     (D_M2, 2 * cfg.q_dim + cfg.kv_dim, cfg.kv_dim)):
                    gemm("", b["n"], d[base:base + 3], b["qkv"], C, n, H, H, width, y_off=col * 2)
                qk = self.param("qk", cap=self.max_seq, q_norm=int(d[D_Q_NORM]), k_norm=int(d[D_K_NORM]),
                                kv=int(d[D_STATE]), logbase=self.logbase)
                plan.dispatch("qwen35moe_qk_norm_rope_kv_rows_bf16", [
                    (b["qkv"], 0, READ), (b["weights"], 0, READ), (b["qt"], 0, WRITE), (b["kv"], 0, READ_WRITE),
                    (b["args"], qk, READ), state], (C, 1, 1), 256)
                att = self.param("attention", kv_heads=cfg.num_kv_heads, cap=self.max_seq, gate_stride=width,
                                 kv=int(d[D_STATE]), scale=cfg.head_dim ** -0.5)
                plan.dispatch("gqa8_d256_causal_gated_prefill_mma_bf16", [
                    (b["qt"], 0, READ), (b["kv"], 0, READ), (b["qkv"], 0, READ), (b["attn"], 0, WRITE),
                    (b["args"], att, READ), state], (_ceil(C, 8), cfg.num_kv_heads, 1), 256)
                inner = cfg.q_dim
            gemm("_residual", b["attn"], d[D_OUT:D_OUT + 3], b["h"], C, H, inner, inner, H)
            rms_post = self.param("rms", hidden=H, w=int(d[D_POST_NORM]), eps=cfg.rms_norm_eps)
            plan.dispatch("flint_rms_rows_bf16", [
                (b["h"], 0, READ), (b["weights"], 0, READ), (b["n"], 0, WRITE), (b["args"], rms_post, READ), state,
            ], (C, 1, 1), 256)
            router = self.param("router_dense", dense=int(d[D_ROUTER_DENSE]), gw=int(d[D_SGATE]),
                                gs=int(d[D_SGATE + 1]), gb=int(d[D_SGATE + 2]))
            plan.dispatch("qwen35moe_router_top8_dense_rows_bf16", [
                (b["n"], 0, READ), (b["weights"], 0, READ), (b["ids"], 0, WRITE), (b["gates"], 0, WRITE),
                (b["sgate"], 0, WRITE), (b["args"], router, READ), state], (_ceil(C, 16), 1, 1), 128)
            plan.dispatch("qwen35moe_expert_count", [
                (b["ids"], 0, READ), (b["counts"], 0, READ_WRITE), state], (_ceil(C * K, 256), 1, 1), 256)
            plan.dispatch("flint_expert_scan", [
                (b["counts"], 0, READ_WRITE), (b["starts"], 0, WRITE), (b["tiles"], 0, WRITE),
                (b["cursors"], 0, WRITE), (b["args"], self.scan_param, READ)], (1, 1, 1), 256)
            plan.dispatch("qwen35moe_expert_place_bf16", [
                (b["n"], 0, READ), (b["ids"], 0, READ), (b["cursors"], 0, READ_WRITE), (b["slots"], 0, WRITE),
                (b["xp"], 0, WRITE), state], (C * K, 1, 1), 256)
            gemm("_swiglu", b["xp"], d[D_GATE:D_GATE + 3], b["actp"], C * K, I, H, H, I,
                 w2_off=d[D_UP:D_UP + 3], grouped=True)
            gemm("", b["actp"], d[D_DOWN:D_DOWN + 3], b["yp"], C * K, H, I, I, H, grouped=True)
            gemm("_swiglu", b["n"], d[D_SH_GATE:D_SH_GATE + 3], b["acts"], C, S, H, H, S,
                 w2_off=d[D_SH_UP:D_SH_UP + 3])
            gemm("", b["acts"], d[D_SH_DOWN:D_SH_DOWN + 3], b["shared"], C, H, S, S, H)
            plan.dispatch("qwen35moe_expert_combine_rows_bf16", [
                (b["h"], 0, READ_WRITE), (b["yp"], 0, READ), (b["shared"], 0, READ), (b["slots"], 0, READ),
                (b["gates"], 0, READ), (b["sgate"], 0, READ), state], (C, 1, 1), 256)
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
