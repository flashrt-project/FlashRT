"""FlashRT -- ROCm Qwen3.6-27B pipeline metadata.

The ROCm implementation mirrors the RTX Qwen3.6 structure but keeps its own
module so AMD-specific FP8/block-scale decisions do not leak into the CUDA
path. The first ROCm stage is a raw-weight, owned-buffer pipeline; forward
kernels land incrementally after the ABI is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen36RocmDims:
    """Static text-path dimensions for Qwen3.6-27B FP8."""

    hidden: int = 5120
    num_layers: int = 64
    vocab_size: int = 248320
    intermediate: int = 17408
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 10_000_000.0
    max_position_embeddings: int = 262144

    # Full attention layers: indices 3, 7, ..., 63.
    full_q_heads: int = 24
    full_kv_heads: int = 4
    full_head_dim: int = 256
    partial_rotary_factor: float = 0.25

    # Linear attention layers: 48 Gated-DeltaNet layers.
    lin_k_heads: int = 16
    lin_v_heads: int = 48
    lin_head_dim: int = 128
    lin_conv_kernel: int = 4

    @property
    def full_q_dim(self) -> int:
        return self.full_q_heads * self.full_head_dim

    @property
    def full_kv_dim(self) -> int:
        return self.full_kv_heads * self.full_head_dim

    @property
    def full_q_proj_dim(self) -> int:
        # Qwen3.6 full-attn q_proj emits Q plus an output gate.
        return 2 * self.full_q_dim

    @property
    def lin_qkv_dim(self) -> int:
        return 10240

    @property
    def lin_z_dim(self) -> int:
        return self.lin_v_heads * self.lin_head_dim


def expected_layer_types() -> list[str]:
    """Return the canonical 3 linear-attn + 1 full-attn repeating pattern."""

    return [
        "full_attention" if (idx + 1) % 4 == 0 else "linear_attention"
        for idx in range(Qwen36RocmDims.num_layers)
    ]
