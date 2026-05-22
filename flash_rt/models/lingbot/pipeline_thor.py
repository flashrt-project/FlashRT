"""LingBot-VLA pipeline dimensions for Thor SM110.

Holds ``LingBotPipelineDims`` — the static shape constants (layer counts,
hidden/head dims, FFN widths) shared by the compute functions in
``forward.py`` / ``sample_actions.py`` and by the torch frontend. The
actual forward functions live in ``forward.py`` (prefix encode) and
``sample_actions.py`` / ``graph_runner.py`` (denoise loop + CUDA Graph
capture); they operate on raw device pointers via ``fvk.*`` and the
LingBot kernel wrappers in ``kernel_ops.py``.

Architecture summary: SigLIP-style ViT vision tower, a 36-layer
Qwen2.5-VL backbone (VLM prefix), and a 36-layer flow-matching action
expert that runs a multi-step Euler ODE captured into a CUDA Graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LingBotPipelineDims:
    """Static shape constants for LingBot-VLA on Thor.

    These are the source of truth (yaml is metadata). Verified against
    the baseline weight inventory (1555 tensors, 36L VLM + 36L expert +
    32L ViT) on 2026-05-19.
    """
    # ── Vision Tower (Qwen2.5-VL ViT) ─────────────────────────────
    vit_num_blocks: int = 32
    vit_hidden_dim: int = 1280
    vit_num_heads: int = 16
    vit_head_dim: int = 80                    # 1280 / 16
    vit_patch_size: int = 14
    vit_temporal_patch_size: int = 2
    vit_spatial_merge_size: int = 2
    vit_intermediate: int = 3420
    vit_out_hidden_size: int = 2048
    vit_num_patches_per_img: int = 256        # (224/14)^2
    vit_num_tokens_per_img: int = 64          # after spatial_merge=2

    # ── Prefix LLM (Qwen2.5-VL 3B) ────────────────────────────────
    vlm_num_layers: int = 36
    vlm_hidden_dim: int = 2048
    vlm_num_heads: int = 16
    vlm_num_kv_heads: int = 2                 # GQA n_rep=8
    vlm_head_dim: int = 128
    vlm_intermediate: int = 11008
    vlm_rope_theta: float = 1_000_000.0
    vlm_mrope_section: tuple[int, int, int] = (16, 24, 24)
    vlm_rms_norm_eps: float = 1e-6
    vlm_vocab_size: int = 151936

    # ── Action Expert (Qwen2-768, Mixed-Head) ─────────────────────
    expert_num_layers: int = 36
    expert_hidden_dim: int = 768
    expert_num_heads: int = 16
    expert_num_kv_heads: int = 2
    expert_head_dim: int = 128
    expert_q_proj_out_dim: int = 2048         # 16 * 128 — Mixed-Head asymmetry
    expert_intermediate: int = 2752
    expert_adanorm_cond_dim: int = 768

    # ── Action head / Flow Matching ───────────────────────────────
    state_dim: int = 75
    action_dim: int = 75
    proj_width: int = 768                     # state_proj/action_in_proj out
    action_time_mlp_in: int = 1536            # sin/cos timestep emb (2*768)

    # ── Inference ─────────────────────────────────────────────────
    num_steps: int = 50
    num_cams: int = 3
    img_size: int = 224
    tokenizer_max_length: int = 72

    # ── Sequence length budgets (max, for site allocation) ────────
    # Prefix tokens = num_cams * num_tokens_per_img + tokenizer_max_length
    #               = 3 * 64 + 72 = 264; round up to 384 for headroom.
    max_prefix_seq: int = 384
    # Suffix tokens = 1 state + n_action_steps actions = 51; round to 64.
    max_suffix_seq: int = 64


def make_attention_spec(dims: LingBotPipelineDims):
    """Build an AttentionSpec for LingBot-VLA's three attention sites.

    Sites:
      * ``vit``   — Qwen2.5-VL ViT, 32 blocks, full MHA (16Q/16KV, HD=80),
                    Q seq = 3 cams × 256 patches = 768.
      * ``vlm_prefix`` — Mixed-Head joint attention during *prefix encode*,
                    Q seq = prefix tokens only (img+lang), 36 layers,
                    GQA 16Q/2KV, HD=128. Even though Expert also computes
                    Q/K/V here, embed_prefix only feeds prefix embeddings;
                    the expert path stays empty until denoise.
      * ``vlm_suffix`` — Mixed-Head joint attention during *denoise step*,
                    Q seq = suffix only (state + 50 actions = 51), KV =
                    prefix + suffix concatenated. Same 36 layers,
                    GQA 16Q/2KV, HD=128.

    : spec is declared but the backend is a stub that does not run
    kernels yet.
    """
    from flash_rt.hardware.backend import SiteSpec, AttentionSpec

    return AttentionSpec(sites={
        "vit": SiteSpec(
            num_layers=dims.vit_num_blocks,
            num_q_heads=dims.vit_num_heads,
            num_kv_heads=dims.vit_num_heads,        # MHA
            head_dim=dims.vit_head_dim,
            max_q_seq=dims.num_cams * dims.vit_num_patches_per_img,
            batch_axis=1,
        ),
        "vlm_prefix": SiteSpec(
            num_layers=dims.vlm_num_layers,
            num_q_heads=dims.vlm_num_heads,
            num_kv_heads=dims.vlm_num_kv_heads,     # GQA
            head_dim=dims.vlm_head_dim,
            max_q_seq=dims.max_prefix_seq,
        ),
        "vlm_suffix": SiteSpec(
            num_layers=dims.vlm_num_layers,
            num_q_heads=dims.vlm_num_heads,
            num_kv_heads=dims.vlm_num_kv_heads,
            head_dim=dims.vlm_head_dim,
            max_q_seq=dims.max_suffix_seq,
            max_kv_seq=dims.max_prefix_seq + dims.max_suffix_seq,
        ),
    })


class LingBotPipelineThor:
    """LingBot-VLA pipeline on Thor SM110.

    : constructor stub — accepts dims + an attention backend, validates
    geometry, then any compute method raises NotImplementedError.
    """

    def __init__(self, gemm, fvk, attn_backend, weights, dims: LingBotPipelineDims):
        self._gemm = gemm
        self._fvk = fvk
        self._attn = attn_backend
        self._weights = weights
        self.dims = dims

    def encode_prefix(self, *, stream: int = 0) -> int:
        """Run ViT + VLM 36L (prefix-only attention, fill KV cache).

        Returns: int ptr to prefix KV cache base.
        """
        raise NotImplementedError("encode_prefix — implemented later+")

    def denoise_step(self, *, t_emb_ptr: int, x_t_ptr: int, stream: int = 0) -> int:
        """One Euler ODE step: 36L expert + Mixed-Head attention into KV.

        Returns: int ptr to velocity v_t [B, 50, 75].
        """
        raise NotImplementedError("denoise_step — implemented later+")

    def run_pipeline(self, *, stream: int = 0):
        """Full forward: prefix → 50 × denoise_step → action_out_proj."""
        raise NotImplementedError("run_pipeline — implemented later+")

    def forward(self):
        """Replay captured CUDA graph or fall back to run_pipeline."""
        raise NotImplementedError("forward — CUDA graph wired later+")
