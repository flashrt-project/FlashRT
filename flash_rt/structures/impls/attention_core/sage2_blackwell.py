"""attention_core — the sage2 (Blackwell INT8-QK) dense form.

The consumer-Blackwell quantized twin of the dense attention family: the
same stateless seam — complete Q/K/V every call, host SDPA layout — executed
by the ``flashrt/sageattention2-blackwell`` kernel: per-warp (or per-thread)
INT8 quantization of Q and K, FP8 per-channel (or FP16) V, one fused
attention, BF16 out. Where the bf16 forms preserve numerics exactly, this
form trades a bounded quantization error for roughly half the attention
time on long unmasked sites; the parity gates downstream judge that trade
on real captures like any other impl's.

Two precision variants, both from the same artifact:

- ``pv_fp8``  — INT8 QK / FP8 V. The speed point of the family.
- ``pv_fp16`` — INT8 QK / FP16 V. Recovers most of the quantization error
  at ~60% more attention time; the option for hosts whose gates reject the
  FP8 point.

Qualification, decided from the artifact and the captures, refusal legible:

- head_dim must be advertised by the artifact (128 today); other dims
  return no binding so the host keeps its own attention,
- masked sites are not claimed: a mask has no form this kernel accepts, and
  the packed-KV plan of the FA2 form does not transfer (the quantizers
  consume dense NHD), so any allowed-ranges request refuses here,
- the workspace (INT8/FP8 staging, scales, output) is caller-owned and
  allocated once per shape at bind — call sequences are pointer-stable and
  the artifact declares itself CUDA-graph safe.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/sageattention2-blackwell",
    "version": ">=1",
}

_VARIANTS = ("pv_fp8", "pv_fp16")


class _Staging(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    workspace: object


#: One staging set and workspace per distinct (shape, dtype, device,
#: variant, granularity). A transformer reaches this seam once per block
#: and per attention site, all with the same shapes, and the scratch is
#: large: at 24576 tokens over 32 heads a single set is about 800MB, so
#: forty-eight blocks owning their own would not fit on any consumer part.
#: Sharing is safe because the scratch holds nothing between calls -- it
#: is filled at the top of every forward and read before returning -- and
#: because these calls are sequential on one stream by construction: the
#: host runs its blocks in order. The pool is keyed, never emptied, and
#: pointer-stable, which is also what a captured graph needs.
_STAGING: dict[tuple, _Staging] = {}


def _staging_for(q_shape, kv_shape, dtype, device, variant, granularity):
    key = (tuple(q_shape), tuple(kv_shape), dtype,
           str(torch.device(device)), variant, granularity)
    staging = _STAGING.get(key)
    if staging is None:
        art = _artifact()
        q = torch.empty(*q_shape, dtype=dtype, device=device)
        k = torch.empty(*kv_shape, dtype=dtype, device=device)
        v = torch.empty_like(k)
        out = torch.empty_like(q)
        staging = _Staging(q, k, v, out, art.allocate_workspace(
            q, k, v, fp8v=(variant == "pv_fp8"),
            qk_quant_granularity=granularity))
        _STAGING[key] = staging
    return staging


def _artifact():
    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def supported_head_dims() -> tuple[int, ...]:
    """Executable envelope, read from the artifact — never duplicated here."""
    caps = _artifact().capabilities()
    dims = tuple(sorted(int(d) for d in caps["head_dims"]))
    if not dims:
        raise ValueError(
            "attention_core sage2: artifact advertised no head dims")
    return dims


class DenseAttentionSage2(GuardedSeam, torch.nn.Module):
    """sage2 replacement for an ordinary dense unmasked SDPA call.

    Inputs and outputs use the host SDPA layout ``[B, H, S, D]``; the kernel
    consumes NHD ``[B, S, H, D]``. Quantization runs per call inside the
    artifact against the caller-owned workspace, so repeated calls launch an
    identical sequence on identical pointers.
    """

    def __init__(self, q_shape, kv_shape, dtype: torch.dtype, device,
                 variant: str = "pv_fp8",
                 qk_quant_granularity: str = "per_warp"):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(
                f"attention_core sage2: unknown variant {variant!r} "
                f"(expected one of {_VARIANTS})")
        b, heads, seq_q, head_dim = q_shape
        kb, kv_heads, seq_kv, kv_dim = kv_shape
        if kb != b or kv_dim != head_dim:
            raise ValueError(
                "attention_core sage2: Q and KV batch/head dims differ")
        if kv_heads != heads:
            raise ValueError(
                "attention_core sage2: GQA sites are not claimed by this "
                "form yet; query and KV head counts must match")
        if head_dim not in supported_head_dims():
            raise ValueError(
                f"attention_core sage2: head_dim {head_dim} outside the "
                f"artifact envelope {supported_head_dims()}")
        if dtype != torch.bfloat16:
            raise ValueError(
                "attention_core sage2: the artifact consumes bf16 inputs")
        self.q_shape = tuple(q_shape)
        self.kv_shape = tuple(kv_shape)
        self.variant = variant
        self.granularity = qk_quant_granularity

        art = _artifact()
        self._fn = (art.sage2_prefill_fp8v_bf16_d128 if variant == "pv_fp8"
                    else art.sage2_prefill_f16_bf16_d128)
        staging = _staging_for(
            (b, seq_q, heads, head_dim), (b, seq_kv, kv_heads, head_dim),
            dtype, device, variant, qk_quant_granularity)
        # Held as plain attributes, not buffers: this scratch belongs to
        # the shared pool, and registering it would make every bound seam
        # claim the same storage in its own state_dict.
        self._q_nhd, self._k_nhd = staging.q, staging.k
        self._v_nhd, self._out_nhd = staging.v, staging.out
        self._workspace = staging.workspace
        self._frt_arm(
            dtypes=(dtype,), device=torch.device(device),
            k=int(head_dim), rows=int(b * heads * seq_q))

    def forward(self, query, key, value, *, scale=None):
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        # BHSD -> NHD staging copies (fused away once the host adopts the
        # NHD projection layout; kept explicit and pointer-stable here).
        self._q_nhd.copy_(query.transpose(1, 2))
        self._k_nhd.copy_(key.transpose(1, 2))
        self._v_nhd.copy_(value.transpose(1, 2))
        self._fn(
            self._q_nhd, self._k_nhd, self._v_nhd,
            softmax_scale=scale, out=self._out_nhd,
            workspace=self._workspace,
            qk_quant_granularity=self.granularity)
        return self._out_nhd.transpose(1, 2)


def bind_dense_attention(captures, *, variant: str = "pv_fp8",
                         qk_quant_granularity: str = "per_warp"):
    """Bind one stateless dense sage2 core from repeated host captures.

    ``captures`` is the family's own convention -- a sequence of per-call
    dicts holding ``q``, ``key``, ``value`` and ``mask`` in host layout --
    so this form qualifies a site the same way its BF16 siblings do:
    the shape, dtype and mask must not move across the calibration call,
    an unsupported shape returns ``None`` for the caller to keep its own
    path, and a device the package does not serve raises so the family
    binder can record it and move to the next rung.
    """
    if not captures:
        raise ValueError("attention_core sage2: no captures")
    first = captures[0]
    query, key, value = first["q"], first["key"], first["value"]
    if first.get("mask") is not None:
        # No packed-KV plan transfers here: the quantizers consume dense
        # NHD, so a masked site is not this form's, and saying so is what
        # keeps the host's own attention on it.
        return None
    if query.shape[-1] not in supported_head_dims():
        return None
    if tuple(key.shape) != tuple(value.shape):
        return None
    if key.shape[0] != query.shape[0] or key.shape[1] != query.shape[1]:
        # A grouped-query site is a shape this form does not claim, which
        # is an answer the caller acts on by keeping its own attention --
        # not an error. The constructor still raises on it, because
        # reaching it with such a shape would be this module's bug.
        return None
    expected = (tuple(query.shape), tuple(key.shape), tuple(value.shape),
                query.dtype, key.dtype, value.dtype)
    for capture in captures[1:]:
        got = (tuple(capture["q"].shape), tuple(capture["key"].shape),
               tuple(capture["value"].shape), capture["q"].dtype,
               capture["key"].dtype, capture["value"].dtype)
        if got != expected:
            raise ValueError(
                "attention_core sage2: shape or dtype moved within one "
                f"calibration call: {expected} -> {got}")
        if capture.get("mask") is not None:
            raise ValueError(
                "attention_core sage2: a mask appeared within one "
                "calibration call")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("attention_core sage2: Q/K/V dtypes differ")
    if query.dtype != torch.bfloat16:
        return None
    return DenseAttentionSage2(
        query.shape, key.shape, query.dtype, query.device,
        variant=variant, qk_quant_granularity=qk_quant_granularity)
