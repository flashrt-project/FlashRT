"""FlashRT — RTX SM120 Qwen3-VL-8B inference pipeline (dim contract).

This file defines the static dimension contract for the Qwen3-VL-8B
NVFP4 inference path. As with ``flash_rt.models.qwen3.pipeline_rtx``,
the actual compute lives in the frontend
(``flash_rt.frontends.torch.qwen3_vl_rtx``) — this module only records
the dimensions the frontend hard-codes, so external tooling (benches,
profilers, tests) can read them without instantiating the forward stack.

Architecture summary (Qwen3-VL-8B-Instruct)::

    image pixels (patchified: 3 × Tp=2 × 16 × 16 = 1536 per patch)
        |
        v  visual.patch_embed.proj   (Conv as Linear → 1152)
        v  + visual.pos_embed (learned 2304×1152, grid-interpolated)
        v  per ViT block ×27 (hidden 1152, 16 heads × head_dim 72):
        v     LayerNorm(norm1) → qkv(+bias) → 2D-RoPE → MHA(bidirectional)
        v        → proj(+bias) → residual
        v     LayerNorm(norm2) → fc1(+bias) → GELU(tanh) → fc2(+bias) → residual
        v     (layers 8/16/24 also feed a DeepStack merger)
        v  visual.merger: 2×2 spatial merge (1152·4=4608) → norm → fc1
        v        → GELU → fc2 → out_hidden 4096
        |
        v  scatter image features into the text embedding stream at the
        v  image-placeholder positions (image_token_id 151655)
        v  + DeepStack: add merger[k] features (ViT layers 8/16/24) into
        v        the first 3 LLM hidden states at the image positions
        |
        v  ── language backbone: reuses the dense Qwen3-8B decoder ──
        v  embed_tokens (BF16) → 36 × [Qwen3 full-attn NVFP4 W4A4 layer]
        v       with interleaved-MRoPE (mrope_section [24,20,20]) cos/sin
        v  final RMSNorm → lm_head (BF16)
        |
    [logits: (B, S, 151936)]

The decoder layer body is byte-for-byte the same kernel sequence as
``Qwen3TorchFrontendRtx._layer_forward_full_nvfp4``; the only
language-side delta vs plain Qwen3 is RoPE (interleaved-MRoPE cos/sin
table instead of plain 1D) and ``rope_theta`` (5e6 vs 1e6).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Qwen3VlTextDims:
    """Static dim contract for the Qwen3-VL-8B language backbone.

    Read from ``config.json:text_config`` at load time and asserted
    against these values. Structurally identical to ``Qwen3Dims``
    except ``rope_theta`` (5e6) and the MRoPE sectioning.
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
    rope_theta: float = 5_000_000.0
    max_pos: int = 262_144

    # interleaved-MRoPE: per-axis frequency split over t / h / w.
    # sum(mrope_section) == head_dim // 2 == 64.
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    mrope_interleaved: bool = True

    # Norm
    rms_norm_eps: float = 1e-6


@dataclass
class Qwen3VlVisionDims:
    """Static dim contract for the Qwen3-VL ViT tower.

    Read from ``config.json:vision_config`` at load time.
    """

    depth: int = 27
    hidden: int = 1152
    num_heads: int = 16
    head_dim: int = 72              # 1152 / 16
    intermediate: int = 4304
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2     # 2×2 patch merge before the projector
    out_hidden: int = 4096          # projector output == text hidden
    num_position_embeddings: int = 2304
    # ViT layers whose features are routed through a DeepStack merger and
    # added into the first len(...) LLM hidden states.
    deepstack_visual_indexes: tuple[int, int, int] = (8, 16, 24)
    # hidden_act == gelu_pytorch_tanh ; norm == LayerNorm(weight+bias)
    layer_norm_eps: float = 1e-6


@dataclass
class Qwen3VlDims:
    """Combined Qwen3-VL-8B dimension contract.

    The special token ids drive the multimodal scatter (where image
    features replace placeholder embeddings) and MRoPE position
    construction.
    """

    text: Qwen3VlTextDims = field(default_factory=Qwen3VlTextDims)
    vision: Qwen3VlVisionDims = field(default_factory=Qwen3VlVisionDims)

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    bos_token_id: int = 151643
    eos_token_id: int = 151645


class Qwen3VlPipeline:
    """Framework-agnostic Qwen3-VL pipeline placeholder.

    The actual forward path is hand-written in
    ``flash_rt.frontends.torch.qwen3_vl_rtx`` against the
    flash_rt_kernels / flash_rt_fa2 entry points. This class only holds
    a reference to the frontend's WeightHandles so external tooling can
    inspect dims without instantiating the full forward stack — mirrors
    ``flash_rt.models.qwen3.pipeline_rtx.Qwen3Pipeline``.
    """

    DIMS = Qwen3VlDims()

    def __init__(self, weights) -> None:
        self.weights = weights

    @property
    def num_layers(self) -> int:
        return int(self.weights.ptrs.get('num_layers', self.DIMS.text.num_layers))

    @property
    def hidden(self) -> int:
        return int(self.weights.ptrs.get('hidden', self.DIMS.text.hidden))
