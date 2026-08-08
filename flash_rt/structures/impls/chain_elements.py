"""Shared chain elements: the parts every region candidate assembles.

A region family owns two things — a structural identifier and an
assembly recipe. Everything a recipe *uses* that another recipe could
use too lives here: weight packing, layout equivalences, activation
checks, cache duck-typing, and the attention ladder. Keeping the
elements out of any one family is what keeps a family thin enough to
read as its recipe, and what keeps two hosts' chains assembling the
same certified parts instead of drifting copies.

- :func:`fp8_weight` — per-tensor static FP8 packing with the scale
  returned for alpha folding.
- :func:`interleave_rows` — the rotate-half ↔ adjacent-pair rotation
  equivalence, applied to projection rows at pack time; attention dot
  products are invariant under a shared head-dim permutation.
- :func:`gelu_tanh_like` — a numeric activation check: the host's
  callable against tanh-GELU, never a class name.
- :func:`cache_kv` — per-layer K/V access across cache generations.
- :data:`ATTN_RUNGS` / :func:`attention_rungs` — the attention
  element ladder: the house CuTe FA4 runtime first (D256 2CTA for
  single-KV stacks), the FA2 used-keys entry after it. A rung that
  loads but cannot execute is eliminated by the binder's functional
  probe at the bound shapes, never by a device list.
"""

from __future__ import annotations

from typing import Callable

import torch

from . import KernelUnavailable, hub_kernel

FP8_MAX = 448.0

ATTN_RUNGS = (("fa4_cute", "flashrt/fa4-cute-runtime", ">=1",
               "forward_static"),
              ("fa2_seqused", "flashrt/fa2-seqused-runtime", ">=1",
               "forward_seqused_static"))


def attention_rungs() -> list[tuple[str, object]]:
    rungs = []
    for mode, repo, version, symbol in ATTN_RUNGS:
        try:
            kern = hub_kernel(repo, version)
        except KernelUnavailable:
            continue
        if hasattr(kern, symbol):
            rungs.append((mode, kern))
    return rungs


def fp8_weight(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    w = w.detach().to("cuda", torch.float32)
    scale = float(w.abs().amax()) / FP8_MAX
    if scale <= 0.0:
        scale = 1.0
    packed = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return packed.contiguous(), scale


def interleave_rows(w: torch.Tensor, heads: int,
                    head_dim: int) -> torch.Tensor:
    """Permute projection rows so adjacent-pair rotation carries the
    host's rotate-half convention."""
    half = head_dim // 2
    w = w.reshape(heads, head_dim, w.shape[-1])
    out = torch.empty_like(w)
    out[:, 0::2] = w[:, :half]
    out[:, 1::2] = w[:, half:]
    return out.reshape(heads * head_dim, w.shape[-1])


def cache_kv(cache, idx: int):
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return layers[idx].keys, layers[idx].values
    return cache.key_cache[idx], cache.value_cache[idx]


def gelu_tanh_like(act: Callable) -> bool:
    t = torch.linspace(-4, 4, 65, device="cuda", dtype=torch.bfloat16)
    try:
        got = act(t)
    except Exception:       # noqa: BLE001 — a weird act refuses, not kills
        return False
    ref = torch.nn.functional.gelu(t.float(), approximate="tanh")
    return bool(torch.allclose(got.float(), ref, atol=2e-2))
