"""FlashRT — Motus AttentionSpec factory (RTX, sm120) — G1 stub.

Motus has THREE distinct attention sites in the denoise loop:

    1. ``mot_joint``  Tri-model self-attention. Each layer concatenates
                      Q/K/V from {video, action, und} experts (each
                      expert projects from its own native hidden dim
                      onto a shared head_dim=128) and runs a single
                      MHA over the unified sequence. NUM_Q_HEADS varies
                      per expert; the joint site uses NUM_Q_HEADS = 24
                      (Wan native) and KV is broadcast / GQA-style.
                      30 layers × 1 site.

    2. ``wan_cross``  Wan video tokens cross-attend to T5 text context
                      (umt5-xxl, 512 padded tokens). Only the video
                      expert participates; action / und do not. KV is
                      computed once per call from T5 ctx and CACHED for
                      the whole 10-step denoise loop. 30 layers × 1 site.

    3. ``vlm_qwen3``  Qwen3-VL internal attention (the frozen VLM that
                      produces Und tokens). At G2 we will likely run
                      the VLM in PyTorch outside the FlashRT graph (one
                      forward per call, then cache und_tokens), so this
                      site is reserved for a future "VLM inside graph"
                      optimization. 28 layers (Qwen3-VL-2B depth).

────────────────────────────────────────────────────────────────────
G1 status
────────────────────────────────────────────────────────────────────
The factory returns a SiteSpec layout that is import-safe but is
NOT YET tuned for actual shapes — `max_q_seq` / `max_kv_seq` are set
to conservative upper bounds to allow buffer allocation in G2 without
re-deriving formulas. G2 will tighten them based on the actual
post-G0 measurements.

Conservative upper bounds chosen:
    video tokens     S_v = T_l * H'/2 * W'/2 = 3 * 12 * 10 = 360
    action tokens    S_a = action_chunk + state(0) + reg(0) = 8
    und tokens       S_u ≤ 256  (Qwen3-VL last-layer; image grid + text
                                  prompt — actual measured at runtime)
    joint Sq_total  ≤ 360 + 8 + 256 = 624
    T5 ctx          = 512  (fixed by Wan text_len)
    Sq_qwen3        ≤ 256  (Qwen3-VL prompt+image)
"""

from __future__ import annotations

from flash_rt.hardware.backend import AttentionSpec


# Conservative upper bounds — tightened in G2 based on real measurements.
_MAX_VIDEO_TOKENS = 360       # 3 * 12 * 10
_MAX_ACTION_TOKENS = 24       # action_chunk(8) + state(1) + registers(R)
                              # — keep slack so finetune-mode ckpt can also load
_MAX_UND_TOKENS = 256
_MAX_T5_CTX = 512


def make_motus_attention_spec(
    *,
    max_und_tokens: int = _MAX_UND_TOKENS,
    max_video_tokens: int = _MAX_VIDEO_TOKENS,
    max_action_tokens: int = _MAX_ACTION_TOKENS,
    max_t5_ctx: int = _MAX_T5_CTX,
) -> AttentionSpec:
    """Build the Motus AttentionSpec.

    Three sites; see module docstring for derivation.

    Args:
        max_und_tokens:    upper bound on Qwen3-VL last-layer tokens
                            (depends on prompt length + image grid).
        max_video_tokens:  upper bound on Wan video tokens after
                            patch_embedding (= T_l * H'/2 * W'/2).
        max_action_tokens: upper bound on action expert sequence length.
        max_t5_ctx:        Wan text_len (fixed = 512).

    Returns:
        AttentionSpec ready to be passed to ``RtxFlashAttnBackend``.
    """
    spec = AttentionSpec()

    # 1) Tri-model joint self-attention.
    # head_dim is 128 across all three experts (24h / 8h / 4h × 128).
    # NUM_Q_HEADS = 24 (the union); per-expert head selection is done
    # inside the pipeline via slicing into the unified Q/K/V buffer.
    sq_joint_max = int(max_video_tokens + max_action_tokens + max_und_tokens)
    spec.add_site(
        "mot_joint",
        num_layers=30,
        num_q_heads=24,
        num_kv_heads=24,           # not GQA at the joint level; experts
                                    # are pre-projected to common HD=128
        head_dim=128,
        max_q_seq=sq_joint_max,
        max_kv_seq=sq_joint_max,
    )

    # 2) Wan cross-attention into T5 text context.
    # Only video tokens query; T5 ctx is the KV side.
    spec.add_site(
        "wan_cross",
        num_layers=30,
        num_q_heads=24,
        num_kv_heads=24,
        head_dim=128,
        max_q_seq=int(max_video_tokens),
        max_kv_seq=int(max_t5_ctx),
    )

    # 3) Qwen3-VL internal self-attention.
    # G2 likely keeps VLM in PyTorch; this site is reserved for a future
    # optimization where VLM forward is also captured into the graph.
    # Qwen3-VL-2B-Instruct: 28 layers, GQA 16Q/2KV, HD=128.
    # NOTE: until G2 we won't actually allocate this site (the frontend
    # can choose which sites to include via spec); kept here as
    # documentation of the path.
    spec.add_site(
        "vlm_qwen3",
        num_layers=28,
        num_q_heads=16,
        num_kv_heads=2,           # GQA
        head_dim=128,
        max_q_seq=int(max_und_tokens),
        max_kv_seq=int(max_und_tokens),
    )

    return spec
