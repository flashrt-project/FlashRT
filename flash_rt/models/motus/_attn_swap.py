"""G3c — Replace Wan's flash_attention with vendored FA2.

Wan's ``flash_attention`` (bak/wan/modules/attention.py) is the single
attention dispatch point for both:

    1. Tri-modal joint self-attention (concatenated [video|action|und]
       Q/K/V, ~600 tokens at default Motus geometry, called per layer
       per step → 30*10 = 300 calls/inference).

    2. Wan cross-attention into T5 ctx (video Q × T5 K/V, 360 vs 512
       tokens, also 300 calls/inference).

We monkey-patch ``wan.modules.attention.flash_attention`` AND
``wan.modules.model.flash_attention`` (the symbol bound at import time
inside model.py) to route through FlashRT's vendored FA2 entry
``flash_rt.flash_rt_fa2.fwd_bf16``.

────────────────────────────────────────────────────────────────────
Why monkey-patch the global vs use AttentionBackend.run()
────────────────────────────────────────────────────────────────────

The clean FlashRT pattern is ``attn.run("site", layer_idx, q_seq=...)``
(see docs/adding_new_model.md §2.1). That requires pulling Q/K/V out
of the upstream Wan code so the backend can pre-allocate per-site
buffers and dispatch.

For G3c we want a SMALLER surface — preserve the upstream forward
shape, just swap the kernel underneath. The full AttentionBackend
plumbing arrives in G5 alongside CUDA Graph capture, where we need
pre-allocated buffers for graph replay anyway. Until then, in-place
symbol patching is sufficient and keeps cos = 0.998+ trivially since
we're calling the same math.

────────────────────────────────────────────────────────────────────
Layout contract
────────────────────────────────────────────────────────────────────

Wan calls ``flash_attention(q, k, v, k_lens=seq_lens, ...)`` with:
    q / k / v: dense [B, S, num_heads, head_dim] BF16
    k_lens: [B] int (uniform = [S_total]*B for Motus B=1)

Our FA2 entry expects exactly the same shape. For B=1 with uniform
seqlens (always true at Motus default) the conversion is a no-op —
just pass pointers through.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from flash_rt.models.motus._stream import cs

logger = logging.getLogger(__name__)


# Cached softmax_lse scratch buffer keyed by (batch, num_heads, max_seq_q).
# Re-allocated lazily on first call per shape; reused across denoise steps.
_LSE_CACHE: dict[tuple, torch.Tensor] = {}


def _get_lse(batch: int, num_heads_q: int, seqlen_q: int, device, dtype) -> torch.Tensor:
    """Allocate or reuse a softmax_lse fp32 buffer.

    FA2 wants shape (batch, num_heads_q, seqlen_q) fp32. We don't need
    the LSE values themselves — they're just an output sink — but
    the kernel writes to it.
    """
    key = (batch, num_heads_q, seqlen_q, device)
    buf = _LSE_CACHE.get(key)
    if buf is None or buf.numel() < batch * num_heads_q * seqlen_q:
        buf = torch.empty(
            (batch, num_heads_q, seqlen_q),
            dtype=torch.float32, device=device,
        )
        _LSE_CACHE[key] = buf
    return buf


def _make_fa2_flash_attention():
    """Build a replacement ``flash_attention(q, k, v, ...)`` that calls
    vendored FA2.

    Mirrors the public surface of upstream wan.modules.attention.flash_attention
    so any callsite inside Wan / Motus (no-op self-attn, joint self-attn,
    cross-attn) just works.
    """
    import flash_rt.flash_rt_fa2 as _fa2

    # 5090 SM count for splitkv heuristic (cheaper to query once).
    _num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    def flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        q_scale: Optional[float] = None,
        causal: bool = False,
        window_size=(-1, -1),
        deterministic: bool = False,
        dtype=torch.bfloat16,
        version: Optional[int] = None,
        out_dtype: Optional[torch.dtype] = None,
        **_kwargs,
    ) -> torch.Tensor:
        # Wan calls this with dense [B, L, n, d] — no packed conversion.
        # We don't honour window_size or non-uniform q_lens / k_lens at
        # G3c (Motus B=1 uniform); fall back to the original SDPA shim
        # path when these are exercised.
        if (window_size is not None and window_size != (-1, -1)) or causal:
            # Fall back to upstream's path for the (rare) windowed/causal
            # case. Joint MoT and Wan cross-attn both use full (-1,-1).
            return _orig_flash_attention(
                q, k, v, q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p, softmax_scale=softmax_scale,
                q_scale=q_scale, causal=causal, window_size=window_size,
                deterministic=deterministic, dtype=dtype, version=version,
            )

        # Apply q_scale if upstream requested (Wan never sets it; safety
        # only).
        if q_scale is not None:
            q = q * q_scale

        # Coerce to bf16. FA2 fwd_bf16 reinterprets bytes as bf16.
        q_b = q if q.dtype == torch.bfloat16 else q.to(torch.bfloat16)
        k_b = k if k.dtype == torch.bfloat16 else k.to(torch.bfloat16)
        v_b = v if v.dtype == torch.bfloat16 else v.to(torch.bfloat16)

        # FA2 layout: [B, S, H, D].
        if q_b.dim() != 4:
            raise ValueError(
                f"flash_attention(fa2): expected q [B,S,H,D]; got {q_b.shape}")
        B, Sq, Hq, D = q_b.shape
        Sk = k_b.shape[1]
        Hk = k_b.shape[2]

        # Contiguity required by FA2 (strides matter, see docstring).
        if not q_b.is_contiguous():
            q_b = q_b.contiguous()
        if not k_b.is_contiguous():
            k_b = k_b.contiguous()
        if not v_b.is_contiguous():
            v_b = v_b.contiguous()

        if softmax_scale is None:
            softmax_scale = 1.0 / (D ** 0.5)

        out = torch.empty_like(q_b)
        lse = _get_lse(B, Hq, Sq, q_b.device, torch.float32)

        _fa2.fwd_bf16(
            Q=int(q_b.data_ptr()), K=int(k_b.data_ptr()),
            V=int(v_b.data_ptr()), O=int(out.data_ptr()),
            softmax_lse=int(lse.data_ptr()),
            softmax_lse_accum=0, o_accum=0,           # no splitkv at G3c
            batch=B, seqlen_q=Sq, seqlen_k=Sk,
            num_heads_q=Hq, num_heads_kv=Hk, head_dim=D,
            q_strides=(q_b.stride(0), q_b.stride(1), q_b.stride(2)),
            k_strides=(k_b.stride(0), k_b.stride(1), k_b.stride(2)),
            v_strides=(v_b.stride(0), v_b.stride(1), v_b.stride(2)),
            o_strides=(out.stride(0), out.stride(1), out.stride(2)),
            softmax_scale=float(softmax_scale),
            num_sms=int(_num_sms),
            stream=cs(),
        )

        if out_dtype is not None and out.dtype != out_dtype:
            return out.to(out_dtype)
        return out

    return flash_attention


# ──────────────────────────────────────────────────────────────────
# Globals captured at install — used for fallback in fast-path branches
# we do not handle (e.g., upstream SDPA shim for causal/window cases).
# ──────────────────────────────────────────────────────────────────

_orig_flash_attention = None


def install_fa2_attention(model) -> dict:
    """Patch ``wan.modules.attention.flash_attention`` and
    ``wan.modules.model.flash_attention`` (and any other re-imports)
    to route through the vendored FA2 entry.

    Returns a stats dict counting which symbols were patched.

    The model arg is unused at G3c (no per-instance state to swap)
    but the signature mirrors install_fvk_norms / install_fvk_linears
    for consistency.
    """
    global _orig_flash_attention
    import wan.modules.attention as _wan_attn   # noqa: WPS433
    import wan.modules.model as _wan_model      # noqa: WPS433

    _orig_flash_attention = _wan_attn.flash_attention
    new_fn = _make_fa2_flash_attention()

    counts = {"wan.modules.attention": 0, "wan.modules.model": 0}

    if getattr(_wan_attn, "flash_attention", None) is not None:
        _wan_attn.flash_attention = new_fn
        counts["wan.modules.attention"] = 1

    # wan/modules/model.py also did ``from .attention import flash_attention``
    # at import time; that bound the original symbol. Override the
    # module-local reference too.
    if getattr(_wan_model, "flash_attention", None) is not None:
        _wan_model.flash_attention = new_fn
        counts["wan.modules.model"] = 1

    logger.info(
        f"[g3c] FA2 attention patched: "
        f"wan.modules.attention={counts['wan.modules.attention']}, "
        f"wan.modules.model={counts['wan.modules.model']}")
    return counts
