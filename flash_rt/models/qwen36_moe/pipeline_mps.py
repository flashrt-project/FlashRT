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
    "trace": "qwen35moe_decode_resident_trace_bf16",
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
