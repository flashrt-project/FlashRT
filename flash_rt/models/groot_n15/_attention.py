"""FlashRT -- non-causal attention bindings for the GR00T N1.5 forward.

Four sites, all bf16 and non-causal, all on ``flash_rt_fa2.fwd_bf16``:

=============  =====================  ===========  =========================
site           q x k (per batch)      heads x dim  reference kernel
=============  =====================  ===========  =========================
``vision``     256 x 256, one per     16 x 72      flash-attn (SigLIP FA2)
               camera tile
``vl``         S x S                  32 x 64      torch SDPA (flash)
``dit_cross``  49 x S (cached K/V)    32 x 48      torch SDPA (flash)
``dit_self``   49 x 49                32 x 48      torch SDPA (flash)
=============  =====================  ===========  =========================

Above head_dim 64 the split-KV count follows FA2's wave-efficiency
heuristic, the one both reference kernels use; the split partials are
combined in a fixed order, so the result is deterministic. At head_dim 64
and below the sites run unsplit: ``fwd_bf16`` serves a split request there
with the head_dim-96 split kernel while sizing the partials for 64-wide
rows, so a split would not produce the right result.

A binding is built once per frontend over the frontend's buffers and runs
once at construction, so a build for another architecture fails there
rather than in the middle of a call (an absent head-dim instantiation aborts
in the FA2 wrapper at that point; the frontend checks the module and its
entries before loading anything). Nothing falls back to another backend.
"""
from __future__ import annotations

import math
from typing import Any

import torch


def _max_splits(head_dim: int, max_k: int) -> int:
    """Largest split count FA2's heuristic can pick for keys up to ``max_k``
    (1 at head_dim 64 and below, where the sites run unsplit)."""
    if head_dim <= 64:
        return 1
    block_n = 128 if head_dim <= 128 else 64
    return max(1, min(128, math.ceil(max_k / block_n)))


class Fa2Site:
    """``fwd_bf16`` over fixed q/o buffers and default k/v buffers.

    Rows are ``[batch * seq, heads * head_dim]`` with heads contiguous inside
    a row, which is how the projections write them.
    """

    def __init__(self, name: str, *, q: int, k: int, v: int, o: int,
                 batch: int, heads: int, head_dim: int, max_q: int,
                 max_k: int, device: torch.device) -> None:
        try:
            from flash_rt import flash_rt_fa2
        except ImportError as exc:
            raise RuntimeError(
                "GR00T N1.5 needs flash_rt_fa2 (bf16 FA2 forward), which is "
                f"not importable: {exc}") from exc
        if not hasattr(flash_rt_fa2, "fwd_bf16"):
            raise RuntimeError("flash_rt_fa2 has no fwd_bf16; rebuild with "
                               "FA2_DTYPES including bf16")
        self.name = name
        self._fwd = flash_rt_fa2.fwd_bf16
        self._q, self._k, self._v, self._o = q, k, v, o
        self._batch, self._heads, self._hd = batch, heads, head_dim
        self._scale = head_dim ** -0.5
        self._num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        d_rounded = (head_dim + 31) // 32 * 32
        rows = batch * heads * max_q
        splits = _max_splits(head_dim, max_k)
        f32 = dict(device=device, dtype=torch.float32)
        self._lse = torch.empty(rows, **f32)
        if splits > 1:
            self._lse_accum = torch.empty(splits * rows, **f32)
            self._o_accum = torch.empty(splits * rows * d_rounded, **f32)
            self._accum = (self._lse_accum.data_ptr(), self._o_accum.data_ptr())
        else:
            self._accum = (0, 0)   # no partials: fwd_bf16 runs one split
        # Construction-time check that the instantiation exists and runs here.
        self.run(1, 1, torch.cuda.current_stream(device).cuda_stream)
        torch.cuda.synchronize(device)

    def run(self, sq: int, sk: int, stream: int) -> None:
        self.run_kv(self._k, self._v, sq, sk, stream)

    def run_kv(self, k: int, v: int, sq: int, sk: int, stream: int) -> None:
        width = self._heads * self._hd
        self._fwd(
            Q=self._q, K=k, V=v, O=self._o,
            softmax_lse=self._lse.data_ptr(),
            softmax_lse_accum=self._accum[0],
            o_accum=self._accum[1],
            batch=self._batch, seqlen_q=sq, seqlen_k=sk,
            num_heads_q=self._heads, num_heads_kv=self._heads, head_dim=self._hd,
            q_strides=(sq * width, width, self._hd),
            k_strides=(sk * width, width, self._hd),
            v_strides=(sk * width, width, self._hd),
            o_strides=(sq * width, width, self._hd),
            softmax_scale=self._scale, num_sms=self._num_sms, stream=stream)


def make_fa2_site(site: str, **kw: Any) -> Fa2Site:
    """Target hook: build the FA2 binding of one attention site."""
    return Fa2Site(site, **kw)
