"""FlashRT -- Nex-N2-mini static dims + BF16 reference pipeline.

This module holds the static dimension constants (``Nexn2Dims``) and the
``Nexn2Pipeline`` BF16-eager wrapper around the HF reference model. The
reference pipeline is used **only** when the frontend is constructed with
``kernelized=False`` (the correctness baseline for the golden cosine fixture);
the production path -- NVFP4 kernels, CUDA-graph decode, chunked long-context
prefill -- lives in the frontend forward/decode modules
(``flash_rt.frontends.torch._nexn2_rtx_{forward,decode}``), not here.

Architecture summary (Nex-N2-mini = model_type qwen3_5_moe)::

    [input_ids]
        |
        v  embed_tokens (BF16, vocab=248320, hidden=2048)
        v
    40 decoder layers, alternating linear-attn (3) + full-attn (1):
        layer 0,1,2:   linear_attention   (Gated DeltaNet, conv1d k=4,
                                            16 K-heads / 32 V-heads)
        layer 3:       full_attention     (GQA 16Q/2KV, head_dim=256,
                                            output_gate, partial RoPE 0.25)
        layer 4..39:   same pattern repeats (linear x3, full x1) ...
        |
        v  per layer:  RMSNorm -> attn (linear or full)
        v              + residual -> RMSNorm -> MoE FFN -> residual
        v              MoE: 256 experts, top-8 routed + 1 shared expert
        v
        v  final RMSNorm -> lm_head (BF16, untied)
        v
    [logits: (B, S, 248320)]

The config declares one MTP (multi-token-prediction) layer, but the released
Nex-N2-mini checkpoint ships no MTP tensors, so speculative decode is not wired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Nexn2Dims:
    """Static dimension constants for Nex-N2-mini.

    Source: config.json:text_config (model_type=qwen3_5_moe_text). Fixed
    for the mini (35B-A3B) variant; if another size is added later this
    becomes a per-checkpoint loader instead of a class-level constant.
    """
    hidden: int = 2048
    num_layers: int = 40
    full_attn_period: int = 4          # full at indices 3, 7, ..., 39
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6

    # full-attention sites (10 layers)
    full_q_heads: int = 16
    full_kv_heads: int = 2             # GQA 8:1
    full_head_dim: int = 256
    partial_rotary_factor: float = 0.25   # rotary_dim = 64
    rope_theta: float = 1.0e7
    mrope_section: tuple[int, ...] = (11, 11, 10)

    # linear-attention sites (30 layers, Gated DeltaNet)
    lin_k_heads: int = 16
    lin_v_heads: int = 32             # differs from qwen36 (48)
    lin_head_dim: int = 128
    lin_conv_kernel: int = 4

    # MoE FFN (every layer)
    moe_num_experts: int = 256
    moe_experts_per_tok: int = 8
    moe_intermediate: int = 512
    shared_expert_intermediate: int = 512

    # MTP head
    mtp_layers: int = 1


class Nexn2Pipeline:
    """Framework-agnostic Nex-N2-mini inference pipeline (RTX SM120).

    Phase-1 implementation: hosts an HF reference model and delegates
    ``forward(input_ids)``. The class signature is the Phase-2+ target --
    only the internals will change.

    Future shape (Phase 2+):
        gemm:  fvk.GemmRunner
        fvk:   flash_rt_kernels module
        attn:  AttentionBackend (RtxFlashAttnBackendNexn2)
        bufs:  pre-allocated CudaBuffer dict
        weights: nvfp4-quantized + bf16 device pointers
    """

    DIMS = Nexn2Dims()

    def __init__(self, hf_model: Any) -> None:
        """Wrap an HF model object.

        Args:
            hf_model: Output of the HF auto-loader for the qwen3_5_moe
                checkpoint. In Phase 1 we own the reference; in Phase 2+
                we ingest only the safetensors path and own weight
                loading ourselves.
        """
        self.hf = hf_model
        self.config = hf_model.config
        text_cfg = getattr(self.config, 'text_config', self.config)
        # Sanity-check the dim assumptions against the checkpoint config.
        assert text_cfg.hidden_size == self.DIMS.hidden, (
            f'expected hidden={self.DIMS.hidden}, got {text_cfg.hidden_size}'
        )
        assert text_cfg.num_hidden_layers == self.DIMS.num_layers
        assert text_cfg.head_dim == self.DIMS.full_head_dim
        assert text_cfg.num_experts == self.DIMS.moe_num_experts
        assert (
            text_cfg.layer_types.count('full_attention')
            == self.DIMS.num_layers // self.DIMS.full_attn_period
        )

    def forward(self, input_ids):
        """Single forward pass: token IDs -> logits. Phase-1 thin shim.

        Args:
            input_ids: (B, S) torch.long on cuda.

        Returns:
            logits: (B, S, vocab_size) bf16 on cuda.
        """
        import torch  # local import; pipeline_rtx is import-time-light.
        with torch.no_grad():
            out = self.hf(
                input_ids=input_ids, use_cache=False, return_dict=True,
            )
        return out.logits

    def generate(self, input_ids, *, max_new_tokens: int, do_sample: bool = False):
        """Greedy/sampled autoregressive generate. Phase-1 delegates to HF.

        Phase-3 replaces this with a C++-driven decode loop that captures
        CUDA Graphs and bypasses HF .generate() entirely.
        """
        import torch
        with torch.no_grad():
            return self.hf.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
            )
