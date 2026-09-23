"""Pi0.5 safetensors conversion and prompt embedding shared by torch targets.

Importing this module does not initialize a GPU runtime. Device work happens
only when embedding a prompt or when callers transfer converted weights.
"""
from __future__ import annotations
import logging
import math
import pathlib
from typing import Union
import torch
import torch.nn.functional as F
from flash_rt.core.utils.pi05_prompt import format_pi05_prompt

logger = logging.getLogger(__name__)
bf16 = torch.bfloat16
VIS_L, ENC_L, DEC_L, DEC_D = 27, 18, 18, 1024
ACTION_DIM, IMG_HW, NUM_STEPS_DEFAULT = 32, 224, 10

def _interleave_qk(w: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Interleave Q/K output dim from HF contiguous to JAX RoPE format."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
         .reshape(num_heads, 2, head_dim // 2, in_dim)
         .permute(0, 2, 1, 3)
         .reshape(out_dim, in_dim)
    )


def convert_pi05_safetensors(safetensors_path: Union[str, pathlib.Path]) -> dict:
    """Convert a HuggingFace Pi0.5 safetensors file to BF16 torch tensor dict.

    Key transformations (verified bit-exact against the openpi PyTorch
    reference forward on LIBERO data):

      - Vision attention: separate Q/K/V → merged, transposed (in, 3*out).
      - Vision patch embedding: ``(C_out, C_in, H, W)`` → ``(H, W, C_in, C_out)``.
      - Encoder RMSNorm fold: multiply Q/K/V/gate/up weights by ``(1 + norm_w)``
        in FP32 to avoid bf16 rounding near -1.0.
      - Encoder Q/K heads: interleave for fused RoPE kernel.
      - Decoder Q/K heads: interleave (no RMS fold — AdaRMSNorm is runtime).
      - Decoder AdaRMSNorm modulation: ``input_layernorm.dense`` →
        ``pre_attn_norm_mod`` (kept separate, BF16).
      - Output projection: frontend pre-scales ``decoder_action_out_proj_w/b``
        by ``-1.0 / num_steps`` (matching the flow-matching residual accumulation).
      - 10-step sinusoidal time embeddings.
    """
    from safetensors import safe_open
    from flash_rt.executors.torch_weights import _autodetect_strip_prefix

    logger.info("Loading Pi0.5 safetensors: %s", safetensors_path)
    f = safe_open(str(safetensors_path), framework="pt")
    # Auto-strip the lerobot HF policy ``model.`` wrap so the openpi
    # bare-key lookups below resolve transparently on either layout.
    _strip = _autodetect_strip_prefix(set(f.keys()))

    def g(key: str) -> torch.Tensor:
        return f.get_tensor((_strip + key) if _strip else key).to(bf16)

    def g_raw(key: str) -> torch.Tensor:
        return f.get_tensor((_strip + key) if _strip else key)

    ckpt: dict = {}

    # ── Vision encoder (27 SigLIP layers) ──
    vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    pe_w = g(f"{vp}.embeddings.patch_embedding.weight")   # (1152, 3, 14, 14)
    # Target layout (14, 14, 3, 1152) flattens contiguously to (588, 1152)
    # row-major as (h, w, c, o) — matches the patch_im2col output order.
    ckpt["vision_patch_embedding_w"] = pe_w.permute(2, 3, 1, 0).contiguous()
    ckpt["vision_patch_embedding_b"] = g(f"{vp}.embeddings.patch_embedding.bias")
    ckpt["vision_position_embedding"] = g(f"{vp}.embeddings.position_embedding.weight")

    qkv_w_list, qkv_b_list = [], []
    o_w_list, o_b_list = [], []
    up_w_list, up_b_list = [], []
    down_w_list, down_b_list = [], []
    ln1_w_list, ln1_b_list = [], []
    ln2_w_list, ln2_b_list = [], []

    for i in range(VIS_L):
        lp = f"{vp}.encoder.layers.{i}"
        q_w = g(f"{lp}.self_attn.q_proj.weight")
        k_w = g(f"{lp}.self_attn.k_proj.weight")
        v_w = g(f"{lp}.self_attn.v_proj.weight")
        qkv_w_list.append(torch.cat([q_w, k_w, v_w], dim=0).t())

        q_b = g(f"{lp}.self_attn.q_proj.bias")
        k_b = g(f"{lp}.self_attn.k_proj.bias")
        v_b = g(f"{lp}.self_attn.v_proj.bias")
        qkv_b_list.append(torch.cat([q_b, k_b, v_b]))

        o_w_list.append(g(f"{lp}.self_attn.out_proj.weight").t())
        o_b_list.append(g(f"{lp}.self_attn.out_proj.bias"))

        up_w_list.append(g(f"{lp}.mlp.fc1.weight").t())
        up_b_list.append(g(f"{lp}.mlp.fc1.bias"))

        down_w_list.append(g(f"{lp}.mlp.fc2.weight").t())
        down_b_list.append(g(f"{lp}.mlp.fc2.bias"))

        ln1_w_list.append(g(f"{lp}.layer_norm1.weight"))
        ln1_b_list.append(g(f"{lp}.layer_norm1.bias"))
        ln2_w_list.append(g(f"{lp}.layer_norm2.weight"))
        ln2_b_list.append(g(f"{lp}.layer_norm2.bias"))

    ckpt["vision_attn_qkv_w"] = torch.stack(qkv_w_list)
    ckpt["vision_attn_qkv_b"] = torch.stack(qkv_b_list)
    ckpt["vision_attn_o_w"] = torch.stack(o_w_list)
    ckpt["vision_attn_o_b"] = torch.stack(o_b_list)
    ckpt["vision_ffn_up_w"] = torch.stack(up_w_list)
    ckpt["vision_ffn_up_b"] = torch.stack(up_b_list)
    ckpt["vision_ffn_down_w"] = torch.stack(down_w_list)
    ckpt["vision_ffn_down_b"] = torch.stack(down_b_list)
    ckpt["vision_pre_attn_norm_w"] = torch.stack(ln1_w_list)
    ckpt["vision_pre_attn_norm_b"] = torch.stack(ln1_b_list)
    ckpt["vision_pre_ffn_norm_w"] = torch.stack(ln2_w_list)
    ckpt["vision_pre_ffn_norm_b"] = torch.stack(ln2_b_list)
    ckpt["vision_final_norm_w"] = g(f"{vp}.post_layernorm.weight")
    ckpt["vision_final_norm_b"] = g(f"{vp}.post_layernorm.bias")

    # ── Multi-modal projector ──
    mp = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    ckpt["encoder_multi_modal_projector_w"] = g(f"{mp}.weight").t()
    ckpt["encoder_multi_modal_projector_b"] = g(f"{mp}.bias")

    # ── Encoder (18 Gemma-2B layers with RMSNorm fold) ──
    ep = "paligemma_with_expert.paligemma.model.language_model.layers"
    enc_qkv_list, enc_o_list = [], []
    enc_gate_list, enc_up_list, enc_down_list = [], [], []

    for i in range(ENC_L):
        # CRITICAL: fuse in FP32 — bf16 rounds values near -1.0 to exactly
        # -1.0, collapsing (1 + scale) to 0 and zeroing entire channels.
        attn_scale = g_raw(f"{ep}.{i}.input_layernorm.weight").float()
        fuse_attn = 1.0 + attn_scale  # (2048,)

        q_w = g_raw(f"{ep}.{i}.self_attn.q_proj.weight").float()
        k_w = g_raw(f"{ep}.{i}.self_attn.k_proj.weight").float()
        v_w = g_raw(f"{ep}.{i}.self_attn.v_proj.weight").float()
        q_w = _interleave_qk(q_w, 8)
        k_w = _interleave_qk(k_w, 1)
        q_w = q_w * fuse_attn.unsqueeze(0)
        k_w = k_w * fuse_attn.unsqueeze(0)
        v_w = v_w * fuse_attn.unsqueeze(0)
        qkv = torch.cat([q_w, k_w, v_w], dim=0).t().to(bf16)
        enc_qkv_list.append(qkv)

        enc_o_list.append(g(f"{ep}.{i}.self_attn.o_proj.weight").t())

        ffn_scale = g_raw(f"{ep}.{i}.post_attention_layernorm.weight").float()
        fuse_ffn = 1.0 + ffn_scale

        gate_w = g_raw(f"{ep}.{i}.mlp.gate_proj.weight").float() * fuse_ffn.unsqueeze(0)
        up_w = g_raw(f"{ep}.{i}.mlp.up_proj.weight").float() * fuse_ffn.unsqueeze(0)
        enc_gate_list.append(gate_w.t().to(bf16))
        enc_up_list.append(up_w.t().to(bf16))

        enc_down_list.append(g(f"{ep}.{i}.mlp.down_proj.weight").t())

    ckpt["encoder_attn_qkv_w"] = torch.stack(enc_qkv_list)
    ckpt["encoder_attn_o_w"] = torch.stack(enc_o_list)
    ckpt["encoder_ffn_gate_w"] = torch.stack(enc_gate_list)
    ckpt["encoder_ffn_up_w"] = torch.stack(enc_up_list)
    ckpt["encoder_ffn_down_w"] = torch.stack(enc_down_list)

    # ── Decoder (18 Gemma-300M layers) ──
    dp = "paligemma_with_expert.gemma_expert.model.layers"
    dec_qkv_list, dec_o_list = [], []
    dec_gate_list, dec_up_list, dec_down_list = [], [], []
    dec_attn_mod_w_list, dec_attn_mod_b_list = [], []
    dec_ffn_mod_w_list, dec_ffn_mod_b_list = [], []

    for i in range(DEC_L):
        dec_attn_mod_w_list.append(g(f"{dp}.{i}.input_layernorm.dense.weight").t())
        dec_attn_mod_b_list.append(g(f"{dp}.{i}.input_layernorm.dense.bias"))

        q_w = g(f"{dp}.{i}.self_attn.q_proj.weight")
        k_w = g(f"{dp}.{i}.self_attn.k_proj.weight")
        v_w = g(f"{dp}.{i}.self_attn.v_proj.weight")
        q_w = _interleave_qk(q_w.float(), 8).to(q_w.dtype)
        k_w = _interleave_qk(k_w.float(), 1).to(k_w.dtype)
        dec_qkv_list.append(torch.cat([q_w, k_w, v_w], dim=0).t())

        dec_o_list.append(g(f"{dp}.{i}.self_attn.o_proj.weight").t())

        dec_ffn_mod_w_list.append(
            g(f"{dp}.{i}.post_attention_layernorm.dense.weight").t())
        dec_ffn_mod_b_list.append(
            g(f"{dp}.{i}.post_attention_layernorm.dense.bias"))

        dec_gate_list.append(g(f"{dp}.{i}.mlp.gate_proj.weight").t())
        dec_up_list.append(g(f"{dp}.{i}.mlp.up_proj.weight").t())
        dec_down_list.append(g(f"{dp}.{i}.mlp.down_proj.weight").t())

    ckpt["decoder_attn_qkv_w"] = torch.stack(dec_qkv_list)
    ckpt["decoder_attn_o_w"] = torch.stack(dec_o_list)
    ckpt["decoder_ffn_gate_w"] = torch.stack(dec_gate_list)
    ckpt["decoder_ffn_up_w"] = torch.stack(dec_up_list)
    ckpt["decoder_ffn_down_w"] = torch.stack(dec_down_list)
    ckpt["decoder_pre_attn_norm_mod_w"] = torch.stack(dec_attn_mod_w_list)
    ckpt["decoder_pre_attn_norm_mod_b"] = torch.stack(dec_attn_mod_b_list)
    ckpt["decoder_pre_ffn_norm_mod_w"] = torch.stack(dec_ffn_mod_w_list)
    ckpt["decoder_pre_ffn_norm_mod_b"] = torch.stack(dec_ffn_mod_b_list)

    ckpt["decoder_final_norm_mod_w"] = g(
        "paligemma_with_expert.gemma_expert.model.norm.dense.weight").t()
    ckpt["decoder_final_norm_mod_b"] = g(
        "paligemma_with_expert.gemma_expert.model.norm.dense.bias")

    # ── Time MLP + sinusoidal embeddings ──
    ckpt["decoder_time_mlp_in_w"] = g("time_mlp_in.weight").t()
    ckpt["decoder_time_mlp_in_b"] = g("time_mlp_in.bias")
    ckpt["decoder_time_mlp_out_w"] = g("time_mlp_out.weight").t()
    ckpt["decoder_time_mlp_out_b"] = g("time_mlp_out.bias")

    num_steps = NUM_STEPS_DEFAULT
    dt = -1.0 / num_steps
    t = torch.tensor(1.0, dtype=torch.float32)
    min_period, max_period = 4e-3, 4.0
    embedding_dim = DEC_D
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    time_emb_list = []
    for _ in range(num_steps):
        sinusoid_input = t.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        time_emb_list.append(
            torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(bf16)
        )
        t = t + dt
    ckpt["decoder_time_embeds"] = torch.cat(time_emb_list, dim=0)  # (10, 1024)

    # ── Action projections (pre-scaled by frontend before pipeline build) ──
    ckpt["decoder_action_in_proj_w"] = g("action_in_proj.weight").t()
    ckpt["decoder_action_in_proj_b"] = g("action_in_proj.bias")
    ckpt["decoder_action_out_proj_w"] = g("action_out_proj.weight").t()
    ckpt["decoder_action_out_proj_b"] = g("action_out_proj.bias")

    # ── Embedding matrix (for prompt tokenisation) ──
    ckpt["embedding_weight"] = g("paligemma_with_expert.paligemma.lm_head.weight")

    logger.info("Converted %d weight groups", len(ckpt))
    return ckpt


def _embed_prompt(prompt_text: str, embedding_weight: torch.Tensor,
                  max_len: int = 48, state=None) -> tuple[torch.Tensor, int]:
    """Tokenise + embed via PaliGemma embedding table (device, bf16)."""
    # PaliGemma tokenizer resolution — see
    # `flash_rt.utils.paligemma_tokenizer` for the search order and
    # the download instructions emitted on failure.
    try:
        # Preferred: openpi's PaligemmaTokenizer (exact same vocab,
        # same prompt prefix logic FlashRT was built against).
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer(max_len=max_len)
        tokens_np, mask_np = tokenizer.tokenize(prompt_text, state=state)
        prompt_len = int(mask_np.sum())
        token_ids = torch.tensor(
            tokens_np[:prompt_len], dtype=torch.long, device="cuda")
    except (ImportError, FileNotFoundError, OSError, RuntimeError):
        # Fallback: locate the SentencePiece model directly via the
        # FlashRT helper (clear error if not found — never silent
        # segfault).
        from flash_rt.utils.paligemma_tokenizer import (
            load_paligemma_sentencepiece,
        )
        sp = load_paligemma_sentencepiece()
        if state is None:
            # 108 is PaliGemma's `\n` token, used by openpi as the
            # prompt-end separator before the action prefix.
            tokens = [sp.bos_id()] + sp.Encode(prompt_text) + [108]
        else:
            tokens = sp.Encode(format_pi05_prompt(prompt_text, state),
                               add_bos=True)
        token_ids = torch.tensor(tokens, dtype=torch.long, device="cuda")
        prompt_len = len(token_ids)

    if embedding_weight.device.type != "cuda":
        embedding_weight = embedding_weight.to(device="cuda")

    embeds = F.embedding(token_ids, embedding_weight)
    embeds = embeds * float(embeds.shape[-1] ** 0.5)
    return embeds, prompt_len


