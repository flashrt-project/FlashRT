"""Shared helpers for Thor frontends — pure utility functions.

Consolidates module-level helpers that previously lived as copy-pasted
code in each of the 7 Thor frontends (pi05_thor / pi0 / pi0fast / groot
× torch/jax). Only **zero-risk numerical** helpers live here; anything
that touches class state or model-specific logic stays in the frontend.

Stage 5 rollout adds functions incrementally:
  5.1 — ``quant_fp8``
  5.2 — ``interleave_qk``
  5.3 — ``embed_prompt_torch`` / ``embed_prompt_numpy``
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F


# Resolved once at import time; all Thor frontends use the same FP8 dtype.
_FP8 = torch.float8_e4m3fn


def interleave_qk(w, num_heads):
    """Q/K weight output-dim layout conversion.

    Converts HF-contiguous head storage to the pair-interleaved layout
    expected by the JAX/csrc RoPE kernels:
        HF:   [h0_d0, h0_d1, ..., h0_d127, h1_d0, ...]
        RoPE: [h0_d0, h0_d64, h0_d1, h0_d65, ...] (per-head pair-interleaved)

    ``w`` is ``[out_dim, in_dim]`` where ``out_dim = num_heads * head_dim``.
    Returns a tensor with the same shape but the out_dim axis rearranged.
    """
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (w.reshape(num_heads, head_dim, in_dim)
             .reshape(num_heads, 2, head_dim // 2, in_dim)
             .permute(0, 2, 1, 3)
             .reshape(out_dim, in_dim))


def quant_fp8(w):
    """Quantize a weight tensor to FP8 E4M3 with per-tensor scale.

    Returns (fp8_tensor, scale_float) where
    ``fp8 = clamp(w / scale, [-448, 448]).to(float8_e4m3fn)``
    and ``scale = max(|w|.max() / 448, 1e-12)``.

    ``w.contiguous()`` is always applied — this is a no-op for weights
    loaded from safetensors (torch-side) but protects JAX-side weights
    that come via ``.T.astype(...)`` from being laid out column-major.
    CUTLASS reads by raw data pointer assuming row-major contiguous
    storage; non-contiguous inputs would silently produce wrong outputs.
    """
    w = w.contiguous()
    a = w.float().abs().max().item()
    s = max(a / 448.0, 1e-12)
    return (w.float() / s).clamp(-448, 448).to(_FP8), s


# ════════════════════════════════════════════════════════════════════
#  Prompt tokenization + embedding
# ════════════════════════════════════════════════════════════════════

def _tokenize_sentencepiece(prompt_text: str):
    """SentencePiece-direct tokenizer path.

    Returns a python list[int] of token ids: [bos] + encode(text) + [108].
    Token 108 is the PaliGemma BOT/SOP marker used by openpi prompts.

    The tokenizer model file is resolved via
    `flash_rt.utils.paligemma_tokenizer.load_paligemma_sentencepiece`,
    which raises a `FileNotFoundError` with a copy-pasteable download
    command if the file is missing — see
    USAGE.md → 'PaliGemma tokenizer setup' for details.
    """
    from flash_rt.utils.paligemma_tokenizer import (
        load_paligemma_sentencepiece,
    )
    sp = load_paligemma_sentencepiece()
    return [sp.bos_id()] + sp.Encode(prompt_text) + [108]


def embed_prompt_torch(prompt_text, embedding_weight, max_len: int = 48):
    """Torch-side tokenize + embed.

    Tries openpi's PaligemmaTokenizer first (matches training exactly);
    falls back to raw sentencepiece (via the FlashRT helper) if openpi
    isn't importable, can't fetch its tokenizer, or raises any other
    initialization error. Returns (embeds, prompt_len) where embeds is
    fp16 CUDA tensor, already multiplied by sqrt(hidden_dim) per Gemma
    convention.
    """
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer(max_len=max_len)
        tokens_np, mask_np = tokenizer.tokenize(prompt_text)
        prompt_len = int(mask_np.sum())
        token_ids = torch.tensor(
            tokens_np[:prompt_len], dtype=torch.long, device='cuda')
    except (ImportError, FileNotFoundError, OSError, RuntimeError):
        tokens = _tokenize_sentencepiece(prompt_text)
        token_ids = torch.tensor(tokens, dtype=torch.long, device='cuda')
        prompt_len = len(token_ids)

    if embedding_weight.device.type != 'cuda':
        embedding_weight = embedding_weight.to(device='cuda')
    embeds = F.embedding(token_ids, embedding_weight)
    embeds = embeds * float(embeds.shape[-1] ** 0.5)
    return embeds, prompt_len


def embed_prompt_numpy(prompt_text, embedding_weight_np, max_len: int = 48):
    """Numpy-side tokenize + embed (used by JAX frontends).

    No torch dependency. Returns (embeds_fp16_np, prompt_len).
    """
    tokens = _tokenize_sentencepiece(prompt_text)
    token_ids = np.array(tokens, dtype=np.int32)
    prompt_len = len(token_ids)
    embeds = embedding_weight_np[token_ids]
    embeds = embeds * float(embeds.shape[-1] ** 0.5)
    return embeds.astype(np.float16), prompt_len
