"""FlashRT — RTX SM120 Qwen3 (plain dense) inference pipeline.

This file defines the static dimension contract for the Qwen3-8B
NVFP4 inference path. The actual compute lives directly in the
frontend (``flash_rt.frontends.torch.qwen3_rtx``) — by analogy with
``flash_rt.models.qwen36.pipeline_rtx`` which keeps the same split.

Architecture summary (Qwen3-8B, all 36 layers full_attention)::

    [input_ids]
        |
        v  embed_tokens (BF16, vocab=151936, hidden=4096)
        |
        v  per layer ×36:
        v    h_in
        v       │
        v       ├─ RMSNorm(input_layernorm.weight)
        v       ├─ q_proj / k_proj / v_proj   (NVFP4 W4A4 GEMM)
        v       ├─ q_norm / k_norm            (per-head RMSNorm, head_dim)
        v       ├─ RoPE (full rotary_dim=128)
        v       ├─ KV cache write
        v       ├─ FA2 causal GQA (32Q / 8KV / HD=128)
        v       ├─ o_proj                     (NVFP4 W4A4 GEMM)
        v       ├─ residual
        v       ├─ RMSNorm(post_attention_layernorm.weight)
        v       ├─ gate_proj / up_proj         (NVFP4 W4A4 GEMM, both 12288)
        v       ├─ silu(gate) * up             (SwiGLU)
        v       ├─ down_proj                  (NVFP4 W4A4 GEMM)
        v       └─ residual
        |
        v  final RMSNorm(model.norm.weight)
        v  lm_head (BF16, NOT quantized — ckpt's ``ignore`` list)
        |
    [logits: (B, S, 151936)]
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Qwen3Dims:
    """Static dim contract for Qwen3-8B NVFP4.

    Constants are read from the ckpt's config.json at load time and
    asserted against these values; this dataclass is the authoritative
    record of what the frontend hard-codes.
    """

    # Top-level
    hidden: int = 4096
    num_layers: int = 36
    vocab_size: int = 151_936
    intermediate: int = 12288

    # Attention
    num_q_heads: int = 32
    num_kv_heads: int = 8           # GQA 4:1
    head_dim: int = 128
    rotary_dim: int = 128           # full RoPE (rotary_dim == head_dim)
    rope_theta: float = 1_000_000.0
    max_pos: int = 40_960

    # Norm
    rms_norm_eps: float = 1e-6


class Qwen3Pipeline:
    """Framework-agnostic Qwen3 pipeline placeholder.

    The actual forward path is hand-written in
    ``flash_rt.frontends.torch.qwen3_rtx`` against the
    flash_rt_kernels / flash_rt_fa2 entry points.
    Holds a reference to the frontend's WeightHandles so external
    tooling (benches, profilers) can inspect dims without instantiating
    the full forward stack.
    """

    DIMS = Qwen3Dims()

    def __init__(self, weights) -> None:
        self.weights = weights

    @property
    def num_layers(self) -> int:
        return int(self.weights.ptrs.get('num_layers', self.DIMS.num_layers))

    @property
    def hidden(self) -> int:
        return int(self.weights.ptrs.get('hidden', self.DIMS.hidden))
